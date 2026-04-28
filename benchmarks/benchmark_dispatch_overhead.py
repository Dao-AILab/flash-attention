"""Measure FA4 CuTe Python dispatch overhead.

Reproduces the methodology from
https://github.com/Dao-AILab/flash-attention/issues/2426 to compare host-side
dispatch cost (wall time minus GPU kernel time) across sequence lengths.

Run:
    python benchmarks/benchmark_dispatch_overhead.py

Outputs a table of (seqlen, wall_ms_median, gpu_ms_median, overhead_us,
overhead_pct). The overhead column is the metric optimized by #2515.
"""

import argparse
import statistics
import time

import torch

from flash_attn.cute import flash_attn_func


def bench_one(seqlen: int, batch: int, heads: int, head_dim: int, causal: bool, iters: int):
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)

    # Warmup to populate compile caches.
    for _ in range(5):
        out, _ = flash_attn_func(q, k, v, causal=causal)
    torch.cuda.synchronize()

    wall = []
    gpu = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        start_evt.record()
        out, _ = flash_attn_func(q, k, v, causal=causal)
        end_evt.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        wall.append((t1 - t0) / 1e6)  # ms
        gpu.append(start_evt.elapsed_time(end_evt))  # ms

    return statistics.median(wall), statistics.median(gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--causal", action="store_true", default=True)
    args = parser.parse_args()

    # (seqlen, batch): mirrors the OP's table from issue #2426.
    configs = [
        (64, 32), (128, 32), (256, 32), (512, 32),
        (1024, 32), (2048, 32), (4096, 16),
    ]

    print(f"{'seqlen':>6} {'batch':>5} {'wall_ms':>9} {'gpu_ms':>9} {'overhead_us':>12} {'overhead%':>10}")
    print("-" * 60)
    for seqlen, batch in configs:
        wall_ms, gpu_ms = bench_one(
            seqlen, batch, args.heads, args.head_dim, args.causal, args.iters,
        )
        overhead_us = (wall_ms - gpu_ms) * 1000
        overhead_pct = 100 * (wall_ms - gpu_ms) / wall_ms if wall_ms > 0 else 0.0
        print(f"{seqlen:>6} {batch:>5} {wall_ms:>9.3f} {gpu_ms:>9.3f} {overhead_us:>12.1f} {overhead_pct:>9.1f}%")


if __name__ == "__main__":
    main()
