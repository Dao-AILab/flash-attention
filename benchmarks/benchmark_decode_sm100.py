"""Single-token decode benchmark: flash_attn_decode_func (SM100 swap-AB) vs flash_attn_func.

Variants of flash_attn_decode_func (all at their default split count):
    decode         reduction_mode="kernel", use_pdl=True   (default)
    decode_nopdl   reduction_mode="kernel", use_pdl=False
    atomic         reduction_mode="atomic", use_pdl=True   (includes zeroing `out`)
    atomic_nopdl   reduction_mode="atomic", use_pdl=False
    auto           reduction_mode="auto"    (PDL on for kernel mode, off for atomic)

Each call is timed inside a CUDA graph (at least --iters calls per replay, median of --reps
replays). The graph traverses whole input-pool cycles so its KV working set reaches --l2-mb;
this is streaming-L2 rotation, not an explicit cache flush before each call. Achieved bandwidth
counts the bytes of K, V, Q and O once.

    python benchmarks/benchmark_decode_sm100.py --shapes blog
    python benchmarks/benchmark_decode_sm100.py --shapes models --sweep --json out.jsonl
"""

import argparse
import json
import math

import torch

from flash_attn.cute.flash_fwd_decode_sm100 import (
    flash_attn_decode_func,
    get_decode_config,
)
from flash_attn.cute.interface import flash_attn_func

SHAPES = {
    # Colfax "S/P ping-pong for FA4 decode" (Fig. 9): GQA 16:1, 16:2, plus MHA 16:16
    "blog": [
        (1, hq, hkv, s, d)
        for d in (64, 128)
        for (hq, hkv) in ((16, 1), (16, 2), (16, 16))
        for s in (1024, 4096, 16384, 32768, 65536, 131072)
    ],
    # Llama-3-8B (32/8), Llama-3-70B (64/8) style decode at several batch sizes
    "models": [
        (b, hq, 8, s, 128)
        for hq in (32, 64)
        for b in (1, 8, 32, 64)
        for s in (1024, 8192, 32768)
    ],
}


def graph_time_us(fns, iters, reps):
    # Allocating a large pool is insufficient if capture only visits its first `iters` entries.
    # Whole cycles also prevent a short reuse distance across replay boundaries.
    iters = math.ceil(iters / len(fns)) * len(fns)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for f in fns:
                f()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters):
            fns[i % len(fns)]()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    return sorted(times)[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", choices=sorted(SHAPES), default="blog")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--sweep", action="store_true", help="also time other split counts"
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument(
        "--rounds", type=int, default=3, help="interleaved rounds, min is kept"
    )
    parser.add_argument("--l2-mb", type=int, default=512)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    shapes = SHAPES[args.shapes][args.start : args.end]
    print(
        f"{'b':>3} {'hq':>4} {'hkv':>4} {'seqlen':>7} {'d':>4} | {'kernel':>7} {'no-pdl':>7} "
        f"{'atomic':>7} {'no-pdl':>7} {'auto':>7} | {'FA4 us':>8} | {'kernel':>6} {'atomic':>6} "
        f"{'auto':>6}  (speedup vs FA4)"
    )
    for batch, hq, hkv, seqlen, d in shapes:
        kv_bytes = 2 * batch * seqlen * hkv * d * 2
        total_bytes = kv_bytes + 2 * batch * hq * d * 2
        copies = max(1, math.ceil(args.l2_mb * 2**20 / kv_bytes))
        inputs = [
            (
                torch.randn(batch, 1, hq, d, device="cuda", dtype=dtype),
                torch.randn(batch, seqlen, hkv, d, device="cuda", dtype=dtype),
                torch.randn(batch, seqlen, hkv, d, device="cuda", dtype=dtype),
            )
            for _ in range(copies)
        ]
        default_splits = get_decode_config(batch, hq, hkv, seqlen, d)[0]
        variants = {
            "decode": dict(reduction_mode="kernel", use_pdl=True),
            "decode_nopdl": dict(reduction_mode="kernel", use_pdl=False),
            "atomic": dict(reduction_mode="atomic", use_pdl=True),
            "atomic_nopdl": dict(reduction_mode="atomic", use_pdl=False),
            "auto": dict(reduction_mode="auto"),
        }
        cands = {
            (name, 0): [
                lambda t=t, kw=kw: flash_attn_decode_func(*t, **kw) for t in inputs
            ]
            for name, kw in variants.items()
        }
        cands[("fa4", 0)] = [
            lambda t=t: flash_attn_func(*t, num_splits=0) for t in inputs
        ]
        if args.sweep:
            num_ctas = batch * hkv * max(1, (hq // hkv + 31) // 32)
            for target in (32, 128, 256):
                ns = min(max(1, round(target / num_ctas)), math.ceil(seqlen / 256))
                if ns != default_splits:
                    cands[("decode", ns)] = [
                        lambda t=t, ns=ns: flash_attn_decode_func(*t, num_splits=ns)
                        for t in inputs
                    ]
            for ns in (1, 4, 8, 16, 32, 64):
                if ns <= math.ceil(seqlen / 128):
                    cands[("fa4", ns)] = [
                        lambda t=t, ns=ns: flash_attn_func(*t, num_splits=ns)
                        for t in inputs
                    ]
        # Independent fp32 reference, grouped without materializing repeated K/V heads.
        q, k, v = inputs[0]
        qg = q[:, 0].float().reshape(batch, hkv, hq // hkv, d)
        scores = torch.einsum("bhgd,bshd->bhgs", qg, k.float()) / math.sqrt(d)
        ref = torch.einsum("bhgs,bshd->bhgd", scores.softmax(-1), v.float())
        ref = ref.reshape_as(q)
        errors = {}
        for key, fns in cands.items():
            result = fns[0]()
            if isinstance(result, tuple):
                result = result[0]
            torch.testing.assert_close(result.float(), ref, atol=2e-2, rtol=2e-2)
            errors[f"{key[0]}@{key[1]}"] = (result.float() - ref).abs().max().item()
        del q, k, v, qg, scores, ref, result
        times = {key: [] for key in cands}
        keys = list(cands)
        for round_idx in range(args.rounds):
            # Rotate the order so one variant is not always the first or last measurement.
            order = keys[round_idx:] + keys[:round_idx]
            for key in order:
                times[key].append(graph_time_us(cands[key], args.iters, args.reps))
        t = {f"{name}@{ns}": round(min(v), 2) for (name, ns), v in times.items()}
        fa4 = t["fa4@0"]
        v = [t[f"{name}@0"] for name in variants]
        print(
            f"{batch:>3} {hq:>4} {hkv:>4} {seqlen:>7} {d:>4} | "
            + " ".join(f"{x:>7.2f}" for x in v)
            + f" | {fa4:>8.2f} | {fa4 / v[0]:>5.2f}x {fa4 / v[2]:>5.2f}x {fa4 / v[4]:>5.2f}x"
        )
        if args.json:
            with open(args.json, "a") as f:
                row = dict(b=batch, hq=hq, hkv=hkv, s=seqlen, d=d, bytes=total_bytes)
                row.update(
                    default_splits=default_splits,
                    times=t,
                    round_times={f"{name}@{ns}": v for (name, ns), v in times.items()},
                    max_abs_errors=errors,
                    input_copies=copies,
                    graph_iters=math.ceil(args.iters / copies) * copies,
                    kv_working_set_bytes=copies * kv_bytes,
                    reps=args.reps,
                    rounds=args.rounds,
                )
                f.write(json.dumps(row) + "\n")
        del inputs, cands
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
