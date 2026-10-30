"""Compare standard FA4 and SwapAB decode with num_splits=0 under CUDA graphs.

A shape fails if FLASH_ATTENTION_SM100_BACKEND=swapab falls back to the standard kernel.

Each graph traverses whole cycles of the same seeded input pool (>=512 MiB KV,
>=2 copies). Latency is the minimum of interleaved round medians, in us/call.
Bandwidth counts useful Q/K/V/O bytes once, excluding workspace traffic.
Run single-threaded: the backend selector is process-global.

    python benchmarks/benchmark_decode_sm100.py --shapes all --json results.jsonl
    python benchmarks/benchmark_decode_sm100.py --shape 1 32 4 8192 256

--shape B HQ HKV S D is repeatable and replaces the preset group. JSONL output
is overwritten at startup and contains one row per completed shape, with samples.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

SHAPES = {
    # Colfax "S/P ping-pong for FA4 decode" (Fig. 9): GQA 16:1, 16:2, MHA 16:16
    "blog": [
        (1, hq, hkv, s, d)
        for d in (64, 128)
        for (hq, hkv) in ((16, 1), (16, 2), (16, 16))
        for s in (1024, 4096, 16384, 32768, 65536, 131072)
    ],
    # Llama-3-8B (32/8), Llama-3-70B (64/8) style decode
    "models": [
        (b, hq, 8, s, 128)
        for hq in (32, 64)
        for b in (1, 8, 32, 64)
        for s in (1024, 8192, 32768)
    ],
}
SHAPES["all"] = SHAPES["blog"] + SHAPES["models"]
SELECTOR = "FLASH_ATTENTION_SM100_BACKEND"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", choices=sorted(SHAPES), default="all")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        action="append",
        metavar=("B", "HQ", "HKV", "S", "D"),
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--l2-mb", type=int, default=512, help="KV pool MiB (minimum 512)"
    )
    parser.add_argument("--start", type=int, default=0, help="inclusive shape index")
    parser.add_argument("--end", type=int, help="exclusive shape index")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="overwrite JSONL output")
    args = parser.parse_args(argv)
    from flash_attn.cute.config import SWAPAB

    if min(args.iters, args.reps, args.rounds) < 1:
        parser.error("--iters, --reps, and --rounds must be positive")
    if args.l2_mb < 512:
        parser.error("--l2-mb must be at least 512")
    args.selected_shapes = args.shape or SHAPES[args.shapes]
    for batch, hq, hkv, seqlen, dim in args.selected_shapes:
        if (
            min(batch, hq, hkv, seqlen) < 1
            or hq % hkv
            or hq // hkv > SWAPAB.max_qhead_per_kvhead
            or not SWAPAB.supports_head_dims(dim, dim)
        ):
            parser.error(
                "shapes require positive sizes, a GQA ratio SwapAB admits "
                f"(<= {SWAPAB.max_qhead_per_kvhead}), and a head dim it supports"
            )
    end = len(args.selected_shapes) if args.end is None else args.end
    if not 0 <= args.start < end <= len(args.selected_shapes):
        parser.error("require 0 <= start < end <= shape count")
    args.selected_shapes = args.selected_shapes[args.start : end]
    return args


@contextmanager
def backend_selector(mode):
    previous = os.environ.get(SELECTOR)
    os.environ[SELECTOR] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SELECTOR, None)
        else:
            os.environ[SELECTOR] = previous


def call_with_selected_config(interface, call):
    """Run call() and return its result with the FwdConfig the forward selected."""
    selected = []
    select = interface.select_fwd_config

    def record(inputs):
        selected.append(select(inputs))
        return selected[-1]

    interface.select_fwd_config = record
    try:
        result = call()
    finally:
        interface.select_fwd_config = select
    (config,) = selected
    return result, config


def output_tensor(result):
    # Tensor[0] would silently drop the batch axis.
    return result[0] if isinstance(result, tuple) else result


def pool_plan(shape, requested_iters, l2_mb):
    batch, hq, hkv, seqlen, dim = shape
    kv_bytes = 2 * batch * seqlen * hkv * dim * 2
    q_bytes = batch * hq * dim * 2
    copies = max(2, math.ceil(l2_mb * 2**20 / kv_bytes))
    actual_calls = math.ceil(requested_iters / copies) * copies
    return {
        "input_copies": copies,
        "requested_calls": requested_iters,
        "actual_calls": actual_calls,
        "kv_working_set_bytes": copies * kv_bytes,
        "input_pool_bytes": copies * (kv_bytes + q_bytes),
        "useful_bytes_per_call": kv_bytes + 2 * q_bytes,
    }


def reference(torch, tensors):
    # Process batches separately to bound FP32 K/V intermediates.
    q, k, v = tensors
    batch, _, hq, dim = q.shape
    hkv = k.shape[2]
    result = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(batch):
        qg = q[b, 0].float().reshape(hkv, hq // hkv, dim)
        probabilities = (
            torch.einsum("hgd,shd->hgs", qg, k[b].float()) / math.sqrt(dim)
        ).softmax(-1)
        out = torch.einsum("hgs,shd->hgd", probabilities, v[b].float())
        result[b, 0].copy_(out.reshape(hq, dim))
    return result


def check_output(torch, result, ref):
    result = output_tensor(result)
    torch.testing.assert_close(result.float(), ref, atol=2e-2, rtol=2e-2)
    return (result.float() - ref).abs().max().item()


def capture_backend(torch, call, inputs, actual_calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for tensors in inputs:
                call(tensors)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(actual_calls):
            final_output = output_tensor(call(inputs[i % len(inputs)]))
    return graph, final_output


def measure_round(torch, graph, actual_calls, reps):
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / actual_calls)
    return samples


def benchmark_shape(torch, shape, shape_index, args, interface):
    batch, hq, hkv, seqlen, dim = shape
    plan = pool_plan(shape, args.iters, args.l2_mb)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + shape_index)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    inputs = [
        tuple(
            torch.randn(size, dtype=dtype, device="cuda", generator=generator)
            for size in (
                (batch, 1, hq, dim),
                (batch, seqlen, hkv, dim),
                (batch, seqlen, hkv, dim),
            )
        )
        for _ in range(plan["input_copies"])
    ]

    def call(tensors, mode):
        with backend_selector(mode):
            return interface.flash_attn_func(*tensors, num_splits=0, causal=False)

    eager_ref = reference(torch, inputs[0])
    graphs, details = {}, {}
    for mode in ("standard", "swapab"):
        result, config = call_with_selected_config(interface, lambda: call(inputs[0], mode))
        if config.use_swapab != (mode == "swapab"):
            raise RuntimeError(f"{SELECTOR}={mode} selected {config}")
        eager_error = check_output(torch, result, eager_ref)
        graphs[mode] = capture_backend(
            torch,
            lambda tensors, mode=mode: call(tensors, mode),
            inputs,
            plan["actual_calls"],
        )
        details[mode] = {"eager_max_abs_error": eager_error, "round_samples_us": []}
        if mode == "swapab":
            details[mode]["heuristic_num_splits"] = config.num_splits
    del eager_ref

    # Whole pool cycles end on the last input, shared by both captured graphs.
    ref = reference(torch, inputs[-1])
    for mode, (graph, output) in graphs.items():
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        details[mode]["graph_max_abs_error"] = check_output(torch, output, ref)
    for round_index in range(args.rounds):
        order = (
            ("standard", "swapab") if round_index % 2 == 0 else ("swapab", "standard")
        )
        for mode in order:
            details[mode]["round_samples_us"].append(
                measure_round(torch, graphs[mode][0], plan["actual_calls"], args.reps)
            )
    for mode, (_, output) in graphs.items():
        details[mode]["post_timing_max_abs_error"] = check_output(torch, output, ref)
        medians = [
            statistics.median(samples) for samples in details[mode]["round_samples_us"]
        ]
        details[mode]["round_medians_us"] = medians
        details[mode]["latency_us"] = min(medians)
        details[mode]["useful_bandwidth_GB_s"] = (
            plan["useful_bytes_per_call"] / min(medians) / 1000
        )
    return dict(
        shape_index=shape_index,
        b=batch,
        hq=hq,
        hkv=hkv,
        s=seqlen,
        d=dim,
        dtype=args.dtype,
        seed=args.seed + shape_index,
        requested_num_splits=0,
        aggregation="min_of_round_medians",
        **plan,
        backends=details,
    )


def run_header(torch, root, argv):
    """Provenance for the JSONL: the SwapAB split policy in config.py is fitted per device."""
    from dataclasses import asdict
    from importlib.metadata import version

    from flash_attn.cute.config import SWAPAB

    def git(*cmd):
        result = subprocess.run(
            ["git", *cmd], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() or None

    revision = git("rev-parse", "HEAD")
    return {
        "record_type": "run",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "git_revision": revision,
        "git_dirty": None
        if revision is None
        else bool(git("status", "--porcelain", "--untracked-files=no")),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_cutlass_dsl": version("nvidia-cutlass-dsl"),
        "swapab_policy": asdict(SWAPAB),
    }


def main(argv=None):
    args = parse_args(argv)
    import torch

    from flash_attn.cute import interface

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("requires a real SM100 GPU")

    root = Path(__file__).resolve().parents[1]
    interface_path = Path(interface.__file__).resolve()
    if interface_path != root / "flash_attn/cute/interface.py":
        raise RuntimeError(
            f"public API loaded from unexpected source: {interface_path}"
        )
    print(
        f"{len(args.selected_shapes)} shapes; latency us/call = minimum of {args.rounds} round medians; num_splits=0"
    )
    with (
        torch.no_grad(),
        args.json.open("w", encoding="utf-8") if args.json else nullcontext() as output,
    ):
        if output is not None:
            output.write(json.dumps(run_header(torch, root, argv)) + "\n")
        for index, shape in enumerate(args.selected_shapes, args.start):
            row = benchmark_shape(torch, shape, index, args, interface)
            if output is not None:
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
            timings = " ".join(
                f"{mode}={data['latency_us']:.3f}"
                for mode, data in row["backends"].items()
            )
            print(f"{index:02d} {tuple(shape)}: {timings}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
