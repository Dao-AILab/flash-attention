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


def parse_int_k(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


def parse_csv_ints(value: str) -> list[int]:
    return [parse_int_k(item) for item in value.split(",")]


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


def make_tensors(args, dtype: torch.dtype):
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
    return q, k, v, out


def make_fwd_fn(args, q, k, v, out, tile_mn):
    pack_gqa = None if args.pack_gqa == "auto" else args.pack_gqa == "true"

    def fwd():
        return _flash_attn_fwd(
            q,
            k,
            v,
            causal=args.causal,
            softcap=args.softcap,
            window_size_left=args.window_left,
            window_size_right=args.window_right,
            tile_mn=tile_mn,
            num_splits=1,
            pack_gqa=pack_gqa,
            return_lse=False,
            out=out,
        )[0]

    return fwd


def check_output(args, q, k, v, actual) -> None:
    window_size = (args.window_left, args.window_right)
    expected, _ = attention_ref(q, k, v, causal=args.causal, window_size=window_size)
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


def run_one(args, dtype: torch.dtype, tile_mn: tuple[int, int] | None) -> None:
    q, k, v, out = make_tensors(args, dtype)
    fwd = make_fwd_fn(args, q, k, v, out, tile_mn)
    print(
        "config "
        f"dtype={dtype} tile_mn={tile_label(tile_mn)} "
        f"batch={args.batch_size} seqlen_q={args.seqlen_q} seqlen_k={args.seqlen_k} "
        f"nheads={args.nheads} nheads_kv={args.nheads_kv} headdim={args.headdim} "
        f"causal={args.causal} window=({args.window_left},{args.window_right}) "
        f"pack_gqa={args.pack_gqa}"
    )
    if args.check:
        check_output(args, q, k, v, fwd())
        torch.cuda.synchronize()
        print("  check=passed")
    if args.profile:
        profile_once(fwd, args.profile_warmup)
        print("  profile=completed")
        return
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
    parser.add_argument(
        "--tile-mn",
        type=parse_tile_mns,
        default=[None],
        help="Semicolon-separated tile overrides, e.g. 'auto;128x64;64x128'.",
    )
    parser.add_argument("--pack-gqa", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
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
        try:
            run_one(args, dtype, tile_mn)
        except Exception as exc:
            print(f"config dtype={dtype} tile_mn={tile_label(tile_mn)} failed: {exc}")
            if args.fail_fast:
                raise


if __name__ == "__main__":
    main()
