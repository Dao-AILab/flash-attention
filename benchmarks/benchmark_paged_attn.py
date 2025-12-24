"""
Benchmark for paged attention with various page sizes and head dimensions.

Tests page_size in [32, 64, 128] and headdim in [64, 128].
"""

import math
from typing import NamedTuple

import torch
from einops import rearrange
from triton.testing import do_bench

from flash_attn.cute.benchmark import benchmark_forward
from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python
from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_python

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None

# Only use flash_attn_func_v3 on Hopper (SM90)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 9:
    flash_attn_func_v3 = None

Timing = NamedTuple("timing", [("mean", float)])


def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    """Benchmark forward pass using triton's do_bench."""
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3)


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False):
    """Calculate FLOPs for attention."""
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        avg_seqlen = seqlen_k
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


def generate_paged_kvcache(
    seqlen_k: int,
    page_size: int,
    batch_size: int,
    nheads_k: int,
    d: int,
    dv: int,
    device: str,
    dtype: torch.dtype,
):
    """
    Generate paged KV cache with random page table ordering.

    Returns:
        k_cache: (batch_size, seqlen_k, nheads_k, d) - unpaged view for reference
        v_cache: (batch_size, seqlen_k, nheads_k, dv) - unpaged view for reference
        page_table: (batch_size, num_blocks_per_seq) - page indices
        k_cache_paged: (num_blocks, page_size, nheads_k, d) - paged storage
        v_cache_paged: (num_blocks, page_size, nheads_k, dv) - paged storage
    """
    num_blocks_per_seq = math.ceil(seqlen_k / page_size)
    # Allocate extra blocks (3x) to simulate realistic fragmented memory
    num_blocks = num_blocks_per_seq * batch_size * 3

    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype
    )

    # Create randomized page table to simulate fragmented allocation
    page_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )[:, :num_blocks_per_seq]

    # Create unpaged view for reference computations
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]

    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged


def generate_contiguous_paged_kvcache(
    seqlen_k: int,
    page_size: int,
    batch_size: int,
    nheads_k: int,
    d: int,
    dv: int,
    device: str,
    dtype: torch.dtype,
):
    """
    Generate paged KV cache with contiguous (sequential) page table.
    This represents the best-case scenario for paged attention.
    """
    num_blocks_per_seq = math.ceil(seqlen_k / page_size)
    num_blocks = num_blocks_per_seq * batch_size

    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype
    )

    # Sequential page table (best case)
    page_table = rearrange(
        torch.arange(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )

    # Create unpaged view
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]

    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged


def run_benchmark(
    # page_sizes: list[int] = [32, 64, 128],
    page_sizes: list[int] = [64, 128],
    # headdims: list[int] = [64, 128],
    headdims: list[int] = [64],
    # batch_sizes: list[int] = [2, 4, 8],
    batch_sizes: list[int] = [8],
    # seqlens: list[int] = [2048, 4096, 8192],
    seqlens: list[int] = [8192],
    causal: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    repeats: int = 10,
    verbose: bool = True,
    # test_fragmented: bool = True,
    test_fragmented: bool = False,
):
    """
    Run paged attention benchmark across different configurations.

    Args:
        page_sizes: List of page sizes to test
        headdims: List of head dimensions to test
        batch_sizes: List of batch sizes to test
        seqlens: List of sequence lengths to test
        causal: Whether to use causal attention
        dtype: Data type for tensors
        repeats: Number of benchmark repetitions
        verbose: Whether to print detailed output
        test_fragmented: Whether to test fragmented page tables (realistic scenario)
    """
    device = "cuda"
    torch.manual_seed(42)

    results = {}

    print("=" * 100)
    print("PAGED ATTENTION BENCHMARK")
    print("=" * 100)
    print(f"Page sizes: {page_sizes}")
    print(f"Head dimensions: {headdims}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seqlens}")
    print(f"Causal: {causal}, dtype: {dtype}")
    print(f"Testing fragmented page tables: {test_fragmented}")
    print("=" * 100)

    for headdim in headdims:
        headdim_v = headdim
        nheads = 32 if headdim <= 64 else 16
        nheads_kv = nheads

        for batch_size in batch_sizes:
            for seqlen in seqlens:
                seqlen_q = seqlen
                seqlen_k = seqlen

                print(f"\n### headdim={headdim}, batch={batch_size}, seqlen={seqlen} ###")

                # Generate query
                q = torch.randn(
                    batch_size, seqlen_q, nheads, headdim,
                    device=device, dtype=dtype
                )

                # First, benchmark without paging (baseline)
                k_unpaged = torch.randn(
                    batch_size, seqlen_k, nheads_kv, headdim,
                    device=device, dtype=dtype
                )
                v_unpaged = torch.randn(
                    batch_size, seqlen_k, nheads_kv, headdim_v,
                    device=device, dtype=dtype
                )

                nFLOPS = flops(
                    batch_size, nheads, seqlen_q, seqlen_k,
                    headdim, headdim_v, causal=causal
                )

                # Baseline (no paging)
                if flash_attn_func_python is not None:
                    try:
                        m_baseline = time_fwd(
                            flash_attn_func_python, q, k_unpaged, v_unpaged,
                            causal=causal, repeats=repeats, verbose=False
                        )
                        baseline_ms = m_baseline.mean * 1e3
                        baseline_tflops = nFLOPS / m_baseline.mean * 1e-12
                        print(f"  Baseline (no paging): {baseline_ms:.3f}ms, {baseline_tflops:.1f} TFLOPS")
                        results[(headdim, batch_size, seqlen, None, "baseline")] = {
                            "time_ms": baseline_ms,
                            "tflops": baseline_tflops,
                        }
                    except Exception as e:
                        print(f"  Baseline failed: {e}")
                        baseline_ms = None

                # Benchmark each page size
                for page_size in page_sizes:
                    # Skip if seqlen is not divisible by page_size
                    if seqlen_k % page_size != 0:
                        print(f"  page_size={page_size}: SKIPPED (seqlen not divisible)")
                        continue

                    # Test with contiguous pages (best case)
                    try:
                        (
                            k_cache, v_cache, page_table,
                            k_cache_paged, v_cache_paged
                        ) = generate_contiguous_paged_kvcache(
                            seqlen_k, page_size, batch_size, nheads_kv,
                            headdim, headdim_v, device, dtype
                        )

                        m_paged = time_fwd(
                            flash_attn_varlen_func_python, q, k_cache_paged, v_cache_paged,
                            page_table=page_table, causal=causal,
                            repeats=repeats, verbose=False
                        )
                        paged_ms = m_paged.mean * 1e3
                        paged_tflops = nFLOPS / m_paged.mean * 1e-12
                        overhead = ((paged_ms / baseline_ms) - 1) * 100 if baseline_ms else 0

                        print(f"  page_size={page_size:3d} (contiguous): {paged_ms:.3f}ms, {paged_tflops:.1f} TFLOPS, overhead: {overhead:+.1f}%")

                        results[(headdim, batch_size, seqlen, page_size, "contiguous")] = {
                            "time_ms": paged_ms,
                            "tflops": paged_tflops,
                            "overhead_pct": overhead,
                        }
                    except Exception as e:
                        print(f"  page_size={page_size} (contiguous): FAILED - {e}")

                    # Test with fragmented pages (realistic case)
                    if test_fragmented:
                        try:
                            (
                                k_cache, v_cache, page_table,
                                k_cache_paged, v_cache_paged
                            ) = generate_paged_kvcache(
                                seqlen_k, page_size, batch_size, nheads_kv,
                                headdim, headdim_v, device, dtype
                            )

                            m_paged_frag = time_fwd(
                                flash_attn_varlen_func_python, q, k_cache_paged, v_cache_paged,
                                page_table=page_table, causal=causal,
                                repeats=repeats, verbose=False
                            )
                            paged_frag_ms = m_paged_frag.mean * 1e3
                            paged_frag_tflops = nFLOPS / m_paged_frag.mean * 1e-12
                            overhead_frag = ((paged_frag_ms / baseline_ms) - 1) * 100 if baseline_ms else 0

                            print(f"  page_size={page_size:3d} (fragmented): {paged_frag_ms:.3f}ms, {paged_frag_tflops:.1f} TFLOPS, overhead: {overhead_frag:+.1f}%")

                            results[(headdim, batch_size, seqlen, page_size, "fragmented")] = {
                                "time_ms": paged_frag_ms,
                                "tflops": paged_frag_tflops,
                                "overhead_pct": overhead_frag,
                            }
                        except Exception as e:
                            print(f"  page_size={page_size} (fragmented): FAILED - {e}")

    return results


def print_summary(results: dict):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    # Group by headdim
    headdims = sorted(set(k[0] for k in results.keys()))

    for headdim in headdims:
        print(f"\n### Head Dimension: {headdim} ###")
        print(f"{'Config':<30} {'Baseline':>12} {'PS=32':>12} {'PS=64':>12} {'PS=128':>12}")
        print("-" * 80)

        # Get unique (batch, seqlen) combinations
        configs = sorted(set((k[1], k[2]) for k in results.keys() if k[0] == headdim))

        for batch_size, seqlen in configs:
            baseline_key = (headdim, batch_size, seqlen, None, "baseline")
            baseline_ms = results.get(baseline_key, {}).get("time_ms", "-")

            row = f"b={batch_size}, s={seqlen:<5}"
            if isinstance(baseline_ms, float):
                row += f" {baseline_ms:>10.2f}ms"
            else:
                row += f" {'-':>12}"

            for page_size in [32, 64, 128]:
                key = (headdim, batch_size, seqlen, page_size, "contiguous")
                if key in results:
                    overhead = results[key].get("overhead_pct", 0)
                    row += f" {overhead:>+10.1f}%"
                else:
                    row += f" {'-':>12}"

            print(row)


def main():
    """Main entry point for the benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark paged attention")
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[64, 128],
                        help="Page sizes to benchmark")
    parser.add_argument("--headdims", type=int, nargs="+", default=[64],
                        help="Head dimensions to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seqlens", type=int, nargs="+", default=[8192],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of benchmark repetitions")
    parser.add_argument("--no-causal", action="store_true",
                        help="Disable causal attention")
    parser.add_argument("--fragmented", action="store_true",
                        help="Skip fragmented page table tests")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16"],
                        help="Data type")

    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    results = run_benchmark(
        page_sizes=args.page_sizes,
        headdims=args.headdims,
        batch_sizes=args.batch_sizes,
        seqlens=args.seqlens,
        causal=not args.no_causal,
        dtype=dtype,
        repeats=args.repeats,
        test_fragmented=args.fragmented,
    )

    print_summary(results)

    return results


if __name__ == "__main__":
    main()
