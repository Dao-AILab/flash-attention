#!/usr/bin/env python3
"""Benchmark native SM120 FA4 forward kernels with explicit tile overrides."""

import argparse
import time

import torch

from flash_attn.cute.bench_utils import bandwidth_fwd_bytes, flops
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.testing import attention_ref


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

VARLEN_MODES = ("full", "staggered")


def parse_int_k(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


def parse_tile_mn(value: str) -> tuple[int, int] | None:
    value = value.strip().lower()
    if value == "auto":
        return None
    if "x" in value:
        m_str, n_str = value.split("x", 1)
    else:
        m_str, n_str = value.split(",", 1)
    return int(m_str), int(n_str)


def parse_tile_mns(value: str) -> list[tuple[int, int] | None]:
    return [parse_tile_mn(item) for item in value.split(";")]


def selected_stat(samples: list[float], stat: str) -> float:
    ordered = sorted(samples)
    if stat == "min":
        return ordered[0]
    if stat == "second-min":
        return ordered[1] if len(ordered) > 1 else ordered[0]
    if stat == "median":
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])
    raise ValueError(f"Unsupported stat: {stat}")


def benchmark_cuda_events(fn, repeats: int, warmup: int, stat: str) -> tuple[float, list[float]]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return selected_stat(samples, stat), samples


def preheat_gpu(duration_ms: int) -> None:
    if duration_ms <= 0:
        return
    deadline = time.perf_counter() + duration_ms / 1000.0
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    while time.perf_counter() < deadline:
        torch.mm(a, b)
    torch.cuda.synchronize()


def make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)


def make_varlen_lengths(seqlen: int, batch_size: int, mode: str) -> list[int]:
    if mode == "full":
        return [seqlen] * batch_size
    if mode == "staggered":
        step = max(1, seqlen // max(2, batch_size + 1))
        return [max(1, seqlen - batch_idx * step) for batch_idx in range(batch_size)]
    raise ValueError(f"Unsupported varlen mode: {mode}")


def make_paged_kv(k: torch.Tensor, v: torch.Tensor, page_size: int):
    batch_size, seqlen_k, num_heads_kv, head_dim = k.shape
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_seq
    k_paged = torch.zeros(
        num_pages, page_size, num_heads_kv, head_dim, device=k.device, dtype=k.dtype
    )
    v_paged = torch.zeros_like(k_paged)
    page_table = torch.empty(
        batch_size, num_pages_per_seq, device=k.device, dtype=torch.int32
    )
    for batch_idx in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            global_page_idx = batch_idx * num_pages_per_seq + page_idx
            page_table[batch_idx, page_idx] = global_page_idx
            start = page_idx * page_size
            end = min(start + page_size, seqlen_k)
            if start < end:
                k_paged[global_page_idx, : end - start] = k[batch_idx, start:end]
                v_paged[global_page_idx, : end - start] = v[batch_idx, start:end]
    return k_paged, v_paged, page_table


def make_tensors(args, dtype: torch.dtype):
    if args.paged_kv:
        q_lens = (
            make_varlen_lengths(args.seqlen_q, args.batch_size, args.varlen_mode)
            if args.varlen
            else [args.seqlen_q] * args.batch_size
        )
        k_lens = make_varlen_lengths(args.seqlen_k, args.batch_size, args.varlen_mode)
        q = torch.randn(sum(q_lens), args.nheads, args.headdim, device="cuda", dtype=dtype)
        k_ref = torch.randn(
            args.batch_size,
            args.seqlen_k,
            args.nheads_kv,
            args.headdim,
            device="cuda",
            dtype=dtype,
        )
        v_ref = torch.randn_like(k_ref)
        k_paged, v_paged, page_table = make_paged_kv(k_ref, v_ref, args.page_size)
        out = torch.empty_like(q)
        return {
            "q": q,
            "k": k_paged,
            "v": v_paged,
            "out": out,
            "k_ref": k_ref,
            "v_ref": v_ref,
            "cu_seqlens_q": make_cu_seqlens(q_lens),
            "seqused_k": torch.tensor(k_lens, device="cuda", dtype=torch.int32),
            "page_table": page_table,
            "q_lens": q_lens,
            "k_lens": k_lens,
        }

    if args.varlen:
        q_lens = make_varlen_lengths(args.seqlen_q, args.batch_size, args.varlen_mode)
        k_lens = make_varlen_lengths(args.seqlen_k, args.batch_size, args.varlen_mode)
        q = torch.randn(sum(q_lens), args.nheads, args.headdim, device="cuda", dtype=dtype)
        k = torch.randn(sum(k_lens), args.nheads_kv, args.headdim, device="cuda", dtype=dtype)
        v = torch.randn(sum(k_lens), args.nheads_kv, args.headdim, device="cuda", dtype=dtype)
        out = torch.empty_like(q)
        return {
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "cu_seqlens_q": make_cu_seqlens(q_lens),
            "cu_seqlens_k": make_cu_seqlens(k_lens),
            "q_lens": q_lens,
            "k_lens": k_lens,
        }

    q = torch.randn(
        args.batch_size,
        args.seqlen_q,
        args.nheads,
        args.headdim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        args.batch_size,
        args.seqlen_k,
        args.nheads_kv,
        args.headdim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        args.batch_size,
        args.seqlen_k,
        args.nheads_kv,
        args.headdim,
        device="cuda",
        dtype=dtype,
    )
    out = torch.empty_like(q)
    return {"q": q, "k": k, "v": v, "out": out}


def make_fwd_fn(args, tensors, tile_mn):
    pack_gqa = None if args.pack_gqa == "auto" else args.pack_gqa == "true"

    def fwd():
        return _flash_attn_fwd(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            cu_seqlens_q=tensors.get("cu_seqlens_q"),
            cu_seqlens_k=tensors.get("cu_seqlens_k"),
            seqused_k=tensors.get("seqused_k"),
            max_seqlen_q=args.seqlen_q if args.varlen or args.paged_kv else None,
            max_seqlen_k=None if args.paged_kv else (args.seqlen_k if args.varlen else None),
            page_table=tensors.get("page_table"),
            causal=args.causal,
            softcap=args.softcap,
            window_size_left=args.window_left,
            window_size_right=args.window_right,
            tile_mn=tile_mn,
            num_splits=1,
            pack_gqa=pack_gqa,
            return_lse=False,
            out=tensors["out"],
            _sm120_num_stages=args.num_stages,
            _sm120_q_in_regs=args.q_in_regs,
        )[0]

    return fwd


def attention_ref_paged(args, tensors):
    q = tensors["q"]
    k = tensors["k_ref"]
    v = tensors["v_ref"]
    cu_seqlens_q = tensors["cu_seqlens_q"]
    cache_seqlens = tensors["seqused_k"]
    outs = []
    for batch_idx, seqlen_k in enumerate(cache_seqlens.tolist()):
        q_start, q_end = cu_seqlens_q[batch_idx : batch_idx + 2].tolist()
        out, _ = attention_ref(
            q[q_start:q_end].unsqueeze(0),
            k[batch_idx : batch_idx + 1, :seqlen_k],
            v[batch_idx : batch_idx + 1, :seqlen_k],
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            softcap=args.softcap,
        )
        outs.append(out.squeeze(0))
    return torch.cat(outs, dim=0)


def attention_ref_varlen(args, tensors):
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    cu_seqlens_q = tensors["cu_seqlens_q"]
    cu_seqlens_k = tensors["cu_seqlens_k"]
    outs = []
    for batch_idx in range(args.batch_size):
        q_start, q_end = cu_seqlens_q[batch_idx : batch_idx + 2].tolist()
        k_start, k_end = cu_seqlens_k[batch_idx : batch_idx + 2].tolist()
        out, _ = attention_ref(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            softcap=args.softcap,
        )
        outs.append(out.squeeze(0))
    return torch.cat(outs, dim=0)


def check_output(args, tensors, actual) -> None:
    window_size = (args.window_left, args.window_right)
    if args.paged_kv:
        expected = attention_ref_paged(args, tensors)
    elif args.varlen:
        expected = attention_ref_varlen(args, tensors)
    else:
        expected, _ = attention_ref(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            causal=args.causal,
            window_size=window_size,
            softcap=args.softcap,
        )
    torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)


def profile_once(fn, warmup: int) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        cudart.cudaProfilerStop()


def tile_label(tile_mn: tuple[int, int] | None) -> str:
    return "auto" if tile_mn is None else f"{tile_mn[0]}x{tile_mn[1]}"


def describe_shape(args, tensors) -> str:
    if args.paged_kv:
        return (
            f"paged_kv page_size={args.page_size} q_lens={tensors['q_lens']} "
            f"cache_seqlens={tensors['k_lens']} nheads={args.nheads} "
            f"nheads_kv={args.nheads_kv} headdim={args.headdim}"
        )
    if not args.varlen:
        return (
            f"batch={args.batch_size} seqlen_q={args.seqlen_q} seqlen_k={args.seqlen_k} "
            f"nheads={args.nheads} nheads_kv={args.nheads_kv} headdim={args.headdim}"
        )
    return (
        f"varlen_mode={args.varlen_mode} q_lens={tensors['q_lens']} k_lens={tensors['k_lens']} "
        f"nheads={args.nheads} nheads_kv={args.nheads_kv} headdim={args.headdim}"
    )


def run_one(args, dtype: torch.dtype, tile_mn: tuple[int, int] | None) -> float | None:
    tensors = make_tensors(args, dtype)
    fwd = make_fwd_fn(args, tensors, tile_mn)
    print(
        "config "
        f"dtype={dtype} tile_mn={tile_label(tile_mn)} "
        f"{describe_shape(args, tensors)} "
        f"causal={args.causal} window=({args.window_left},{args.window_right}) "
        f"softcap={args.softcap} pack_gqa={args.pack_gqa} "
        f"num_stages={args.num_stages} q_in_regs={args.q_in_regs}"
    )
    if args.check:
        check_output(args, tensors, fwd())
        torch.cuda.synchronize()
        print("  check=passed")
    if args.profile:
        profile_once(fwd, args.profile_warmup)
        print("  profile=completed")
        return None
    ms, samples = benchmark_cuda_events(fwd, args.repeats, args.warmup, args.stat)
    nflops = flops(
        args.batch_size,
        args.nheads,
        args.seqlen_q,
        args.seqlen_k,
        args.headdim,
        args.headdim,
        causal=args.causal,
        window_size=(args.window_left, args.window_right),
    )
    nbytes = bandwidth_fwd_bytes(
        args.batch_size,
        args.nheads,
        args.nheads_kv,
        args.seqlen_q,
        args.seqlen_k,
        args.headdim,
        args.headdim,
        dtype_bytes=2,
    )
    seconds = ms / 1000.0
    print(f"  stat={args.stat} runtime={ms:.3f}ms")
    print(f"  samples_ms={[round(sample, 3) for sample in samples]}")
    print(f"  tflops={nflops / seconds / 1e12:.2f} tbps={nbytes / seconds / 1e12:.2f}")
    return ms


def print_round_summary(args, tile_mn: tuple[int, int] | None, round_ms: list[float]) -> None:
    if len(round_ms) <= 1:
        return
    summary_ms = selected_stat(round_ms, args.stat)
    ordered = sorted(round_ms)
    median = selected_stat(round_ms, "median")
    print(
        f"summary tile_mn={tile_label(tile_mn)} rounds_ms={[round(ms, 3) for ms in round_ms]} "
        f"best={ordered[0]:.3f}ms median={median:.3f}ms {args.stat}={summary_ms:.3f}ms"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark native SM120 FA4 forward kernels")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--seqlen-q", type=parse_int_k, default=1024)
    parser.add_argument("--seqlen-k", type=parse_int_k, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--nheads-kv", type=int, default=None)
    parser.add_argument("--gqa-ratio", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--window-left", type=int, default=None)
    parser.add_argument("--window-right", type=int, default=None)
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--varlen-mode", choices=VARLEN_MODES, default="staggered")
    parser.add_argument("--paged-kv", action="store_true")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--tile-mn",
        type=parse_tile_mns,
        default=[None],
        help="Semicolon-separated tile overrides, e.g. 'auto;128x64;64x128'.",
    )
    parser.add_argument("--pack-gqa", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--num-stages", type=int, choices=[1, 2], default=1)
    parser.add_argument("--q-in-regs", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Repeat each full benchmark config this many times for noisy workstation GPUs.",
    )
    parser.add_argument("--stat", choices=["min", "second-min", "median"], default="second-min")
    parser.add_argument("--preheat-ms", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SM120 forward benchmarks")
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        raise RuntimeError(f"SM120 benchmark requires compute capability 12.x, got {major}.{minor}")
    if args.nheads_kv is None:
        args.nheads_kv = args.nheads // args.gqa_ratio if args.gqa_ratio else args.nheads
    if args.nheads % args.nheads_kv != 0:
        raise ValueError("--nheads must be divisible by --nheads-kv")

    torch.manual_seed(0)
    dtype = DTYPES[args.dtype]
    print(f"device={torch.cuda.get_device_name()} capability={major}.{minor}")
    preheat_gpu(args.preheat_ms)
    for tile_mn in args.tile_mn:
        round_ms = []
        try:
            for _ in range(args.rounds):
                ms = run_one(args, dtype, tile_mn)
                if ms is not None:
                    round_ms.append(ms)
            print_round_summary(args, tile_mn, round_ms)
        except Exception as exc:
            print(f"config dtype={dtype} tile_mn={tile_label(tile_mn)} failed: {exc}")
            if args.fail_fast:
                raise


if __name__ == "__main__":
    main()
