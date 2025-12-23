"""
Comparative benchmark: CuTe DSL vs Native PyTorch block sparsity computation.
"""

import torch
from dataclasses import dataclass
from typing import Callable, Optional, List
from tabulate import tabulate
from tqdm import tqdm
import itertools

from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark as cute_benchmark
import cutlass.cute as cute
from flash_attn.cute.compute_block_sparsity import BlockSparsityKernel
from flash_attn.cute.block_sparsity import BlockSparseTensors
from mask_mod_definitions import (
    get_mask_pair,
    random_doc_id_tensor,
    flex_document_mask,
    cute_document_mask,
)

from torch.nn.attention.flex_attention import create_block_mask
from triton.testing import do_bench

# Configure torch.compile cache to prevent memory buildup
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    batch_size: int
    num_heads: int
    seqlen_q: int
    seqlen_k: int
    mask_name: str
    tile_m: int = 128
    tile_n: int = 128
    use_fast_sampling: bool = False
    aux_tensors_cute: Optional[list] = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single benchmark run."""

    config: BenchmarkConfig
    cute_time_ms: Optional[float]
    pytorch_time_ms: Optional[float]
    error_message: Optional[str] = None


def benchmark_pytorch_block_sparsity(
    config: BenchmarkConfig,
    mask_fn: Callable,
) -> Optional[float]:
    """
    Benchmark PyTorch block mask creation (compiled).
    Returns: creation_time_ms
    """
    device = "cuda"

    try:
        cbm = torch.compile(create_block_mask)

        def run_benchmark():
            return cbm(
                mask_fn,
                config.batch_size,
                config.num_heads,
                config.seqlen_q,
                config.seqlen_k,
                device=device,
            )

        creation_time_ms = do_bench(run_benchmark, warmup=10, rep=100)

        return creation_time_ms

    except Exception as e:
        print(f"PyTorch benchmark failed ({config.mask_name}): {e}")
        import traceback

        traceback.print_exc()
        return None


def benchmark_cute_block_sparsity(
    config: BenchmarkConfig,
    mask_fn: Callable,
) -> Optional[float]:
    """
    Benchmark CuTe block sparsity kernel.
    Returns: creation_time_ms
    """
    device = "cuda"

    try:
        num_m_blocks = (config.seqlen_q + config.tile_m - 1) // config.tile_m
        num_n_blocks = (config.seqlen_k + config.tile_n - 1) // config.tile_n

        mask_block_cnt = torch.zeros(
            (config.batch_size, config.num_heads, num_m_blocks),
            device=device,
            dtype=torch.int32,
        )
        mask_block_idx = torch.zeros(
            (config.batch_size, config.num_heads, num_m_blocks, num_n_blocks),
            device=device,
            dtype=torch.int32,
        )
        full_block_cnt = torch.zeros(
            (config.batch_size, config.num_heads, num_m_blocks),
            device=device,
            dtype=torch.int32,
        )
        full_block_idx = torch.zeros(
            (config.batch_size, config.num_heads, num_m_blocks, num_n_blocks),
            device=device,
            dtype=torch.int32,
        )

        # Convert to CuTe tensors
        mask_cnt_cute = from_dlpack(
            mask_block_cnt.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=2)
        mask_idx_cute = from_dlpack(
            mask_block_idx.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=3)
        full_cnt_cute = from_dlpack(
            full_block_cnt.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=2)
        full_idx_cute = from_dlpack(
            full_block_idx.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=3)

        blocksparse_tensors = BlockSparseTensors(
            mask_block_cnt=mask_cnt_cute,
            mask_block_idx=mask_idx_cute,
            full_block_cnt=full_cnt_cute,
            full_block_idx=full_idx_cute,
        )

        # Create kernel
        use_aux = (
            config.aux_tensors_cute is not None and len(config.aux_tensors_cute) > 0
        )
        kernel = BlockSparsityKernel(
            mask_mod=mask_fn,
            tile_mn=(config.tile_m, config.tile_n),
            compute_full_blocks=True,
            use_aux_tensors=use_aux,
            use_fast_sampling=config.use_fast_sampling,
        )

        # Compile kernel
        compiled_kernel = cute.compile(
            kernel,
            blocksparse_tensors,
            config.seqlen_q,
            config.seqlen_k,
            config.aux_tensors_cute,
        )

        def generate_tensors():
            from cutlass.cute.testing import JitArguments

            return JitArguments(
                blocksparse_tensors,
                config.seqlen_q,
                config.seqlen_k,
                config.aux_tensors_cute,
            )

        creation_time_us = cute_benchmark(
            compiled_kernel,
            workspace_generator=generate_tensors,
            warmup_iterations=10,
            iterations=100,
        )

        torch.cuda.synchronize(device)
        creation_time_ms = creation_time_us / 1000.0

        return creation_time_ms

    except Exception as e:
        print(f"CuTe benchmark failed: {e}")
        return None


def run_benchmark(
    config: BenchmarkConfig,
    pytorch_mask_fn: Callable,
    cute_mask_fn: Callable,
) -> BenchmarkResult:
    """Run benchmarks for both implementations."""

    print(
        f"Benchmarking {config.mask_name} - B={config.batch_size}, H={config.num_heads}, "
        f"M={config.seqlen_q}, N={config.seqlen_k}"
    )

    # Benchmark PyTorch
    pytorch_time = benchmark_pytorch_block_sparsity(config, pytorch_mask_fn)

    # Benchmark CuTe
    cute_time = benchmark_cute_block_sparsity(config, cute_mask_fn)

    return BenchmarkResult(
        config=config,
        cute_time_ms=cute_time,
        pytorch_time_ms=pytorch_time,
    )


def generate_configs(
    batch_sizes: List[int],
    num_heads: List[int],
    seqlens: List[int],
    mask_names: List[str],
) -> List[BenchmarkConfig]:
    """Generate all benchmark configurations."""
    configs = []
    for B, H, S, mask_name in itertools.product(
        batch_sizes, num_heads, seqlens, mask_names
    ):
        configs.append(
            BenchmarkConfig(
                batch_size=B,
                num_heads=H,
                seqlen_q=S,
                seqlen_k=S,
                mask_name=mask_name,
            )
        )
    return configs


def print_results(results: List[BenchmarkResult]):
    successful_results = [
        r
        for r in results
        if r.cute_time_ms is not None and r.pytorch_time_ms is not None
    ]

    if not successful_results:
        print("No successful benchmark results to display")
        return

    headers = [
        "B",
        "H",
        "M",
        "N",
        "Mask Type",
        "CuTe Time (ms)",
        "PyTorch Time (ms)",
        "Speedup",
    ]

    rows = []
    for result in successful_results:
        speedup = (
            result.pytorch_time_ms / result.cute_time_ms
            if result.cute_time_ms > 0
            else 0
        )

        rows.append(
            [
                result.config.batch_size,
                result.config.num_heads,
                result.config.seqlen_q,
                result.config.seqlen_k,
                result.config.mask_name,
                f"{result.cute_time_ms:.4f}",
                f"{result.pytorch_time_ms:.4f}",
                f"{speedup:.2f}x",
            ]
        )

    # Sort by batch, head, seqlen, then mask type
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[4]))

    print("\n" + "=" * 100)
    print("CuTe DSL vs PyTorch Block Sparsity Benchmark Results")
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("=" * 100)


def main():
    """Run the comparative benchmark."""

    # Configuration
    batch_sizes = [1, 4, 8]
    num_heads = [8, 16]
    seqlens = [1024, 2048, 4096, 8192]
    mask_names = [
        "causal",
        "sliding_window",
        "prefix_lm",
        "dilated_sliding_window",
        "document",
    ]

    device = "cuda"
    max_seqlen = max(seqlens)
    max_batch = max(batch_sizes)
    max_heads = max(num_heads)

    # Create document IDs using the helper from mask_definitions
    doc_ids = random_doc_id_tensor(max_heads, max_batch, max_seqlen, device=device)
    doc_ids_cute = from_dlpack(doc_ids.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=2
    )

    # Generate base configurations
    base_configs = generate_configs(batch_sizes, num_heads, seqlens, mask_names)

    # Update configs with aux tensors for document masking
    configs = []
    for config in base_configs:
        if config.mask_name == "document":
            # Add aux tensors for document masking
            configs.append(
                BenchmarkConfig(
                    batch_size=config.batch_size,
                    num_heads=config.num_heads,
                    seqlen_q=config.seqlen_q,
                    seqlen_k=config.seqlen_k,
                    mask_name=config.mask_name,
                    tile_m=config.tile_m,
                    tile_n=config.tile_n,
                    use_fast_sampling=False,
                    aux_tensors_cute=[doc_ids_cute],
                )
            )
        else:
            configs.append(config)

    # Run benchmarks
    results = []
    print(f"Running {len(configs)} benchmark configurations...")
    for config in tqdm(configs, desc="Benchmarking"):
        try:
            # Get mask pair from mask_definitions
            mask_kwargs = {}
            if config.mask_name == "sliding_window":
                mask_kwargs["window_size"] = 128  # Default window size

            cute_mask_fn, pytorch_mask_fn = get_mask_pair(
                config.mask_name,
                seqlen_q=config.seqlen_q,
                seqlen_k=config.seqlen_k,
                **mask_kwargs,
            )

            # For document masking, create wrapper that captures doc_ids
            if config.mask_name == "document":
                # PyTorch wrapper
                def pytorch_mask_fn(b, h, q, kv):
                    return flex_document_mask(b, h, q, kv, doc_ids)

                # CuTe wrapper - reuse cute_document_mask with aux_tensors
                cute_mask_fn = cute_document_mask

            result = run_benchmark(config, pytorch_mask_fn, cute_mask_fn)
            results.append(result)

        except Exception as e:
            print(f"Failed to run config {config}: {e}")
            results.append(
                BenchmarkResult(
                    config=config,
                    cute_time_ms=None,
                    pytorch_time_ms=None,
                    error_message=str(e),
                )
            )
        finally:
            torch.cuda.empty_cache()
            torch._dynamo.reset()

    print_results(results)


if __name__ == "__main__":
    main()
