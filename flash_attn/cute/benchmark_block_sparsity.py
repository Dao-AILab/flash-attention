"""
Comprehensive benchmark script for BlockSparsityKernel.
Outputs results in a table format similar to PyTorch flex attention benchmarks.
"""

import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark as cute_benchmark
import cutlass.cute as cute
from compute_block_sparsity import BlockSparsityKernel, BlockSparseTensors
from tqdm import tqdm


def create_mask_mods():
    """Create various mask modification functions for benchmarking."""

    # Causal mask
    @cute.jit
    def causal(b, h, q, kv, aux):
        return q >= kv

    # Sliding window mask (window size 128)
    @cute.jit
    def sliding_window(b, h, q, kv, aux):
        return (q >= kv) & (q - kv < 128)

    # Prefix LM mask (first 512 tokens are bidirectional, rest is causal)
    @cute.jit
    def prefix_lm(b, h, q, kv, aux):
        both_in_prefix = (q < 512) & (kv < 512)
        causal_part = q >= kv
        return both_in_prefix | causal_part

    # Dilated sliding window (every other position in a window)
    @cute.jit
    def dilated_sliding_window(b, h, q, kv, aux):
        in_window = (q >= kv) & (q - kv < 256)
        dilated = ((q - kv) % 2) == 0
        return in_window & dilated

    return {
        "causal": (causal, None),
        "sliding_window": (sliding_window, None),
        "prefix_lm": (prefix_lm, None),
        "dilated_sliding_window": (dilated_sliding_window, None),
    }


def create_doc_mask_mod(batch_size, seqlen, device):
    """Create document mask with aux tensors."""
    # Create document IDs with random boundaries
    doc_ids = torch.zeros((batch_size, seqlen), device=device, dtype=torch.int32)

    for b in range(batch_size):
        pos = 0
        doc_id = 0
        while pos < seqlen:
            doc_len = torch.randint(64, 513, (1,)).item()  # Random doc length 64-512
            end_pos = min(pos + doc_len, seqlen)
            doc_ids[b, pos:end_pos] = doc_id
            pos = end_pos
            doc_id += 1

    doc_ids_cute = from_dlpack(doc_ids.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)

    @cute.jit
    def doc_mask_mod(b, h, q, kv, aux):
        doc_ids = aux[0]
        # Direct indexing with scalars
        q_doc_id = doc_ids[b, q]
        kv_doc_id = doc_ids[b, kv]
        return q_doc_id == kv_doc_id

    return doc_mask_mod, [doc_ids_cute]


def benchmark_config(
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_name,
    mask_fn,
    aux_tensors,
    tile_m=128,
    tile_n=128,
):
    """Benchmark a single configuration and return timing in microseconds."""
    device = "cuda"

    # Calculate number of blocks
    n_blocks_q = (seqlen_q + tile_m - 1) // tile_m
    n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

    # Create output tensors
    mask_block_cnt = torch.zeros(
        (batch_size, num_heads, n_blocks_q), device=device, dtype=torch.int32
    )
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
    )
    full_block_cnt = torch.zeros(
        (batch_size, num_heads, n_blocks_q), device=device, dtype=torch.int32
    )
    full_block_idx = torch.zeros(
        (batch_size, num_heads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
    )

    # Convert to cute tensors
    mask_cnt_cute = from_dlpack(mask_block_cnt.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=2
    )
    mask_idx_cute = from_dlpack(mask_block_idx.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=3
    )
    full_cnt_cute = from_dlpack(full_block_cnt.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=2
    )
    full_idx_cute = from_dlpack(full_block_idx.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=3
    )

    blocksparse_tensors = BlockSparseTensors(
        mask_block_cnt=mask_cnt_cute,
        mask_block_idx=mask_idx_cute,
        full_block_cnt=full_cnt_cute,
        full_block_idx=full_idx_cute,
    )

    # Create and compile kernel
    # Set use_aux_tensors=True if aux_tensors is not None
    use_aux = aux_tensors is not None and len(aux_tensors) > 0
    kernel = BlockSparsityKernel(
        mask_mod=mask_fn, tile_mn=(tile_m, tile_n), compute_full_blocks=True, use_aux_tensors=use_aux
    )
    compiled_kernel = cute.compile(
        kernel,
        blocksparse_tensors=blocksparse_tensors,
        aux_tensors=aux_tensors,
    )

    # Generator function for benchmark
    def generate_tensors():
        from cutlass.cute.testing import JitArguments

        return JitArguments(blocksparse_tensors, aux_tensors)

    # Run benchmark
    try:
        exec_time_us = cute_benchmark(
            compiled_kernel,
            workspace_generator=generate_tensors,
            warmup_iterations=10,
            iterations=100,
        )

        # Calculate memory usage
        mask_size_gib = (
            mask_block_cnt.element_size() * mask_block_cnt.numel()
            + mask_block_idx.element_size() * mask_block_idx.numel()
            + full_block_cnt.element_size() * full_block_cnt.numel()
            + full_block_idx.element_size() * full_block_idx.numel()
        ) / (1024**3)

        # Max construction memory is roughly the same (we don't track peak separately)
        max_memory_gib = mask_size_gib

        return exec_time_us, mask_size_gib, max_memory_gib

    except Exception as e:
        print(
            f"Error benchmarking {mask_name} for B={batch_size}, H={num_heads}, M={seqlen_q}, N={seqlen_k}: {e}"
        )
        return None, None, None


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks across multiple configurations."""

    # Test configurations
    batch_sizes = [1, 4, 8]
    num_heads_list = [8, 16]
    seqlens = [1024, 2048, 4096, 8192]

    # Count total configurations
    mask_mods = create_mask_mods()
    total_configs = (
        len(batch_sizes) * len(num_heads_list) * len(seqlens) * (len(mask_mods) + 1)
    )  # +1 for doc_mask

    print(f"Running {total_configs} benchmark configurations...")

    results = []

    # Progress bar
    pbar = tqdm(total=total_configs)

    for batch_size in batch_sizes:
        for num_heads in num_heads_list:
            for seqlen in seqlens:
                device = "cuda"

                # Benchmark standard masks
                for mask_name, (mask_fn, aux_tensors) in mask_mods.items():
                    exec_time_us, mask_size_gib, max_memory_gib = benchmark_config(
                        batch_size, num_heads, seqlen, seqlen, mask_name, mask_fn, aux_tensors
                    )

                    if exec_time_us is not None:
                        results.append(
                            {
                                "B": batch_size,
                                "H": num_heads,
                                "M": seqlen,
                                "N": seqlen,
                                "Mask Mod": mask_name,
                                "Creation Time (ms)": exec_time_us / 1000.0,  # Convert us to ms
                                "Mask Size Memory (GiB)": mask_size_gib,
                                "Max Construction Memory (GiB)": max_memory_gib,
                            }
                        )

                    pbar.update(1)

                # Benchmark document mask (with aux tensors)
                doc_mask_fn, doc_aux_tensors = create_doc_mask_mod(batch_size, seqlen, device)
                exec_time_us, mask_size_gib, max_memory_gib = benchmark_config(
                    batch_size,
                    num_heads,
                    seqlen,
                    seqlen,
                    "doc_mask_mod",
                    doc_mask_fn,
                    doc_aux_tensors,
                )

                if exec_time_us is not None:
                    results.append(
                        {
                            "B": batch_size,
                            "H": num_heads,
                            "M": seqlen,
                            "N": seqlen,
                            "Mask Mod": "doc_mask_mod",
                            "Creation Time (ms)": exec_time_us / 1000.0,
                            "Mask Size Memory (GiB)": mask_size_gib,
                            "Max Construction Memory (GiB)": max_memory_gib,
                        }
                    )

                pbar.update(1)

    pbar.close()

    # Print results in table format
    print_results_table(results)

    return results


def print_results_table(results):
    """Print results in a formatted table."""
    if not results:
        print("No results to display")
        return

    # Print header
    header = "|   B |   H |    M |    N | Mask Mod               |   Creation Time (ms) |   Mask Size Memory (GiB) |   Max Construction Memory (GiB) |"
    separator = "|-----|-----|------|------|------------------------|----------------------|--------------------------|---------------------------------|"

    print(header)
    print(separator)

    # Print each row
    for result in results:
        row = f"|{result['B']:4d} |{result['H']:4d} |{result['M']:5d} |{result['N']:5d} | {result['Mask Mod']:<22s} |"
        row += f"{result['Creation Time (ms)']:21.4f} |"
        row += f"{result['Mask Size Memory (GiB)']:25.4f} |"
        row += f"{result['Max Construction Memory (GiB)']:32.4f} |"
        print(row)


if __name__ == "__main__":
    results = run_comprehensive_benchmark()

    # Optionally save results to CSV
    try:
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv("block_sparsity_benchmark_results.csv", index=False)
        print("\nResults saved to block_sparsity_benchmark_results.csv")
    except ImportError:
        print("\nPandas not available, skipping CSV export")
