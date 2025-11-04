import math
import operator
from typing import Callable, Optional, Tuple, Type
from functools import partial
import cuda.bindings.driver as cuda
import cutlass
from cutlass import Boolean, Constexpr, Float32, Int32, const_expr, Int8
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

from block_sparsity import BlockSparseTensors
from utils import hash_callable, scalar_to_ssa, ssa_to_scalar


class BlockSparsityKernel:
    """Block sparsity kernel for FlexAttention.

    This kernel computes `mask_mod` for every token of each block
    to determine if an n block is full, masked, or neither.

    Writes block counts and indices to a BlockSparseTensors object.

    When use_fast_sampling=True, uses 5-point sampling (4 corners + center)
    which is much faster but only suitable for masks where this is sufficient.
    """

    def __init__(
        self,
        mask_mod: Callable,
        tile_mn: Tuple[int, int],
        compute_full_blocks: bool = True,
        use_aux_tensors: bool = False,
        use_fast_sampling: bool = False,
    ):
        self.mask_mod = mask_mod
        self.tile_mn = tile_mn
        self.compute_full_blocks = compute_full_blocks
        self.use_aux_tensors = use_aux_tensors
        self.use_fast_sampling = use_fast_sampling

    @cute.jit
    def __call__(
        self,
        blocksparse_tensors: BlockSparseTensors,
        aux_tensors: Optional[list] = None,
    ):
        self.mask_cnt, self.mask_idx, self.full_cnt, self.full_idx = blocksparse_tensors

        if const_expr(self.compute_full_blocks):
            assert self.full_cnt is not None and self.full_idx is not None, (
                "full block tensors must be provided when computing full blocks"
            )

        batch_size, num_heads, num_m_blocks, num_n_blocks = list(self.mask_idx.shape)
        grid = [num_m_blocks, num_heads, batch_size]

        # Fast sampling uses only 5 threads (4 corners + center), full sampling uses 1 thread per row
        if const_expr(self.use_fast_sampling):
            num_threads = 5
            self.num_warps = 1
        else:
            num_threads = self.tile_mn[0]
            self.num_warps = (num_threads + 32 - 1) // 32

        self.kernel(
            self.mask_cnt,
            self.mask_idx,
            self.full_cnt,
            self.full_idx,
            num_n_blocks,
            aux_tensors,
        ).launch(grid=grid, block=[num_threads, 1, 1])

    @cute.kernel
    def kernel(
        self,
        mask_cnt: cute.Tensor,
        mask_idx: cute.Tensor,
        full_cnt: cute.Tensor,
        full_idx: cute.Tensor,
        num_n_blocks: Int32,
        aux_tensors: Optional[list] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        ssa = partial(scalar_to_ssa, dtype=Int32)

        @cute.struct
        class SharedStorage:
            reduction_buffer_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, 2 * self.num_warps], 1024
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage, 16)

        reduction_buffer = storage.reduction_buffer_smem.get_tensor(
            cute.make_layout((self.num_warps, 2))
        )
        # if tidx == 0:
        #     print(reduction_buffer)

        num_mask_blocks = Int32(0)
        num_full_blocks = Int32(0)

        # Use runtime loop for variable sequence lengths
        for n_block in cutlass.range(num_n_blocks, unroll_full=True):
            m_base = m_block * self.tile_mn[0]
            n_base = n_block * self.tile_mn[1]

            # Branch at compile time between fast and full sampling
            if const_expr(self.use_fast_sampling):
                # Fast path: 5-point sampling (4 corners + center)
                thread_result = Boolean(False)

                if tidx == 0:
                    # Top-left corner
                    q_idx_ssa = ssa(m_base)
                    kv_idx_ssa = ssa(n_base)
                    thread_result = ssa_to_scalar(
                        self.mask_mod(
                            ssa(batch_idx), ssa(head_idx), q_idx_ssa, kv_idx_ssa, aux_tensors
                        )
                    )
                elif tidx == 1:
                    # Top-right corner
                    q_idx_ssa = ssa(m_base)
                    kv_idx_ssa = ssa(n_base + self.tile_mn[1] - 1)
                    thread_result = ssa_to_scalar(
                        self.mask_mod(
                            ssa(batch_idx), ssa(head_idx), q_idx_ssa, kv_idx_ssa, aux_tensors
                        )
                    )
                elif tidx == 2:
                    # Bottom-left corner
                    q_idx_ssa = ssa(m_base + self.tile_mn[0] - 1)
                    kv_idx_ssa = ssa(n_base)
                    thread_result = ssa_to_scalar(
                        self.mask_mod(
                            ssa(batch_idx), ssa(head_idx), q_idx_ssa, kv_idx_ssa, aux_tensors
                        )
                    )
                elif tidx == 3:
                    # Bottom-right corner
                    q_idx_ssa = ssa(m_base + self.tile_mn[0] - 1)
                    kv_idx_ssa = ssa(n_base + self.tile_mn[1] - 1)
                    thread_result = ssa_to_scalar(
                        self.mask_mod(
                            ssa(batch_idx), ssa(head_idx), q_idx_ssa, kv_idx_ssa, aux_tensors
                        )
                    )
                elif tidx == 4:
                    # Center point
                    q_idx_ssa = ssa(m_base + self.tile_mn[0] // 2)
                    kv_idx_ssa = ssa(n_base + self.tile_mn[1] // 2)
                    thread_result = ssa_to_scalar(
                        self.mask_mod(
                            ssa(batch_idx), ssa(head_idx), q_idx_ssa, kv_idx_ssa, aux_tensors
                        )
                    )

                # Use vote_any_sync to see if any thread found unmasked or masked
                has_unmasked = cute.arch.vote_any_sync(thread_result)
                has_masked = cute.arch.vote_any_sync(Boolean(not thread_result))

            else:
                # Full path: check all elements in the block
                # Track if this thread's row has any masked or unmasked elements
                thread_has_unmasked = Boolean(False)
                thread_has_masked = Boolean(False)

                # Each thread handles 1 row
                if tidx < self.tile_mn[0]:
                    q_idx_ssa = ssa(m_base + tidx)

                    # Loop over all columns in this row
                    for c in cutlass.range_constexpr(self.tile_mn[1]):
                        kv_idx_ssa = ssa(n_base + c)

                        # Direct scalar call
                        mask_val = ssa_to_scalar(
                            self.mask_mod(
                                ssa(batch_idx), ssa(head_idx), q_idx_ssa, kv_idx_ssa, aux_tensors
                            )
                        )

                        # Update tracking flags
                        if mask_val:
                            thread_has_unmasked = Boolean(True)
                        else:
                            thread_has_masked = Boolean(True)

                # Block-level reduction to combine results across all threads
                warp_has_unmasked_mask = cute.arch.vote_any_sync(thread_has_unmasked)
                warp_has_masked_mask = cute.arch.vote_any_sync(thread_has_masked)

                # lane 0 writes the ballot mask to shared memory
                lane_id = tidx % 32
                if lane_id == 0:
                    # Store as Int8
                    reduction_buffer[warp_idx, 0] = Int8(1) if warp_has_unmasked_mask else Int8(0)
                    reduction_buffer[warp_idx, 1] = Int8(1) if warp_has_masked_mask else Int8(0)

                cute.arch.sync_threads()

                # Thread 0 ORs all warp results together
                has_unmasked = Boolean(False)
                has_masked = Boolean(False)
                if tidx == 0:
                    for w in cutlass.range(self.num_warps):
                        if reduction_buffer[w, 0]:
                            has_unmasked = Boolean(True)
                        if reduction_buffer[w, 1]:
                            has_masked = Boolean(True)

            # Only thread 0 updates the output arrays (common to both paths)
            if tidx == 0:
                # Block classification based on what we found:
                # - If has_masked and has_unmasked: partial block (needs masking)
                # - If only has_unmasked: full block (no masking needed)
                # - If only has_masked: skip this block entirely
                is_partial = Boolean(has_masked and has_unmasked)
                is_full = Boolean(has_unmasked and (not has_masked))

                if is_partial:
                    mask_idx[batch_idx, head_idx, m_block, num_mask_blocks] = n_block
                    num_mask_blocks += 1
                elif is_full and const_expr(self.compute_full_blocks):
                    full_idx[batch_idx, head_idx, m_block, num_full_blocks] = n_block
                    num_full_blocks += 1

        # Only thread 0 writes back the counts
        if tidx == 0:
            mask_cnt[batch_idx, head_idx, m_block] = num_mask_blocks
            if const_expr(self.compute_full_blocks):
                full_cnt[batch_idx, head_idx, m_block] = num_full_blocks


def compute_block_sparsity(
    tile_m,
    tile_n,
    batch_size,
    num_heads,
    seqlen_q,
    seqlen_k,
    mask_mod: Callable,
    aux_tensors: Optional[list],  # list[cute.Tensor]
    device,
    compute_full_blocks: bool = True,
) -> BlockSparseTensors:
    num_m_blocks = (seqlen_q + tile_m - 1) // tile_m
    num_n_blocks = (seqlen_k + tile_n - 1) // tile_n

    mask_block_cnt = torch.zeros(
        (batch_size, num_heads, num_m_blocks), device=device, dtype=torch.int32
    )
    mask_block_idx = torch.zeros(
        (batch_size, num_heads, num_m_blocks, num_n_blocks), device=device, dtype=torch.int32
    )
    full_block_cnt = torch.zeros(
        (batch_size, num_heads, num_m_blocks), device=device, dtype=torch.int32
    )
    full_block_idx = torch.zeros(
        (batch_size, num_heads, num_m_blocks, num_n_blocks), device=device, dtype=torch.int32
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

    mask_mod_hash = hash_callable(mask_mod)

    compile_key = (tile_m, tile_n, mask_mod_hash, compute_full_blocks, aux_tensors is not None)
    if compile_key not in compute_block_sparsity.compile_cache:
        kernel = BlockSparsityKernel(
            mask_mod,
            tile_mn=(tile_m, tile_n),
            compute_full_blocks=True,
            use_aux_tensors=aux_tensors is not None,
        )

        compute_block_sparsity.compile_cache[compile_key] = cute.compile(
            kernel,
            blocksparse_tensors,
            aux_tensors,
        )

    compute_block_sparsity.compile_cache[compile_key](
        blocksparse_tensors,
        aux_tensors,
    )

    return blocksparse_tensors


compute_block_sparsity.compile_cache = {}


def run():
    """Test the BlockSparsityKernel with a simple causal mask."""

    print("Testing BlockSparsityKernel...")

    # Configuration
    batch_size = 2
    num_heads = 2
    seqlen_q = 16384
    seqlen_k = 16384
    tile_m, tile_n = 128, 128  # Use very small tiles for initial testing

    # Calculate number of blocks
    n_blocks_q = (seqlen_q + tile_m - 1) // tile_m
    n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

    print(f"Batch size: {batch_size}, Num heads: {num_heads}")
    print(f"Sequence length Q: {seqlen_q}, K: {seqlen_k}")
    print(f"Tile size: {tile_m} x {tile_n}")
    print(f"Number of blocks Q: {n_blocks_q}, K: {n_blocks_k}")

    # Create output tensors on CUDA
    device = "cuda"
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

    # Define a simple causal mask function
    @cute.jit
    def causal_mask(batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
        """Simple causal mask: only attend to positions <= current position."""
        return q_idx >= kv_idx

    # Create and run the kernel
    kernel = BlockSparsityKernel(
        mask_mod=causal_mask,
        tile_mn=(tile_m, tile_n),
        compute_full_blocks=True,
    )

    print("\nRunning kernel...")
    kernel(
        blocksparse_tensors=blocksparse_tensors,
        aux_tensors=None,
    )

    print("Kernel execution completed!")

    # # Print results
    # print("\n--- Results ---")
    # for b in range(batch_size):
    #     for h in range(num_heads):
    #         print(f"\nBatch {b}, Head {h}:")
    #         for m in range(n_blocks_q):
    #             num_mask = mask_block_cnt[b, h, m].item()
    #             num_full = full_block_cnt[b, h, m].item()
    #             print(f"  M-block {m}: {num_mask} masked blocks, {num_full} full blocks")

    #             if num_mask > 0:
    #                 mask_indices = mask_block_idx[b, h, m, :num_mask].cpu().tolist()
    #                 print(f"    Masked blocks: {mask_indices}")

    #             if num_full > 0:
    #                 full_indices = full_block_idx[b, h, m, :num_full].cpu().tolist()
    #                 print(f"    Full blocks: {full_indices}")

    print("\n✓ Test completed successfully!")


def benchmark():
    """Benchmark the BlockSparsityKernel."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.testing import benchmark as cute_benchmark

    print("=" * 60)
    print("Benchmarking BlockSparsityKernel")
    print("=" * 60)

    # Configuration
    batch_size = 2
    num_heads = 8
    seqlen_q = 4096
    seqlen_k = 4096
    tile_m, tile_n = 128, 128

    # Calculate number of blocks
    n_blocks_q = (seqlen_q + tile_m - 1) // tile_m
    n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}, Num heads: {num_heads}")
    print(f"  Sequence length Q: {seqlen_q}, K: {seqlen_k}")
    print(f"  Tile size: {tile_m} x {tile_n}")
    print(f"  Number of blocks Q: {n_blocks_q}, K: {n_blocks_k}")
    print(f"  Total blocks to check: {batch_size * num_heads * n_blocks_q * n_blocks_k}")

    # Create output tensors on CUDA
    device = "cuda"
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

    # Define a simple causal mask function
    @cute.jit
    def causal_mask(batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
        """Simple causal mask: only attend to positions <= current position."""
        return q_idx >= kv_idx

    # Create kernel with fast sampling
    kernel = BlockSparsityKernel(
        mask_mod=causal_mask,
        tile_mn=(tile_m, tile_n),
        compute_full_blocks=True,
        use_fast_sampling=True,
    )

    # Compile kernel
    compiled_kernel = cute.compile(
        kernel,
        blocksparse_tensors=blocksparse_tensors,
        aux_tensors=None,
    )

    # Generator function for benchmark - creates fresh tensors for each iteration
    def generate_tensors():
        from cutlass.cute.testing import JitArguments

        # Return fresh tensors for each benchmark iteration
        return JitArguments(blocksparse_tensors, None)

    # Run benchmark
    print("\nBenchmarking...")
    exec_time = cute_benchmark(
        compiled_kernel,
        workspace_generator=generate_tensors,
        warmup_iterations=10,
        iterations=100,
    )

    print(f"\nBenchmark Results:")
    print(f"  Execution time: {exec_time:.4f} us")
    print(f"  Execution time: {exec_time / 1000:.4f} ms")

    # Calculate throughput
    total_elements = batch_size * num_heads * n_blocks_q * n_blocks_k * tile_m * tile_n
    throughput = total_elements / (exec_time * 1e-6) / 1e9  # Billion elements per second
    print(f"\nThroughput: {throughput:.2f} billion elements/second")
    print("=" * 60)


def benchmark_with_aux_tensors():
    """Benchmark the BlockSparsityKernel with a document mask using aux_tensors."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.testing import benchmark as cute_benchmark

    print("=" * 60)
    print("Benchmarking BlockSparsityKernel with Document Mask")
    print("=" * 60)

    # Configuration
    batch_size = 2
    num_heads = 8
    seqlen_q = 4096
    seqlen_k = 4096
    tile_m, tile_n = 128, 128

    # Calculate number of blocks
    n_blocks_q = (seqlen_q + tile_m - 1) // tile_m
    n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}, Num heads: {num_heads}")
    print(f"  Sequence length Q: {seqlen_q}, K: {seqlen_k}")
    print(f"  Tile size: {tile_m} x {tile_n}")
    print(f"  Number of blocks Q: {n_blocks_q}, K: {n_blocks_k}")
    print(f"  Total blocks to check: {batch_size * num_heads * n_blocks_q * n_blocks_k}")

    # Create output tensors on CUDA
    device = "cuda"
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

    # Create document ID tensor for document masking with random lengths
    # Document lengths vary randomly between 64 and 1024 tokens
    # Each batch has different document boundaries
    doc_ids = torch.zeros((batch_size, num_heads, seqlen_q), device=device, dtype=torch.int32)

    print("\nGenerating random document boundaries per batch...")
    for b in range(batch_size):
        pos = 0
        doc_id = 0
        while pos < seqlen_q:
            # Random document length between 64 and 1024
            doc_len = torch.randint(64, 1025, (1,)).item()
            end_pos = min(pos + doc_len, seqlen_q)
            doc_ids[b, :, pos:end_pos] = doc_id
            pos = end_pos
            doc_id += 1
        print(f"  Batch {b}: {doc_id} documents (varying lengths 64-1024 tokens)")

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
    doc_ids_cute = from_dlpack(doc_ids.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=2)

    blocksparse_tensors = BlockSparseTensors(
        mask_block_cnt=mask_cnt_cute,
        mask_block_idx=mask_idx_cute,
        full_block_cnt=full_cnt_cute,
        full_block_idx=full_idx_cute,
    )

    # Define document mask function with aux_tensor access
    @cute.jit
    def document_mask(batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
        """Document mask: only attend within the same document."""
        doc_ids0 = aux_tensors[0]
        doc_ids1 = aux_tensors[0]
        doc_id_q = doc_ids0[batch_idx[0], head_idx[0], q_idx[0]]
        doc_id_kv = doc_ids1[batch_idx[0], head_idx[0], kv_idx[0]]
        q_doc = scalar_to_ssa(doc_id_q, cutlass.Int32)
        kv_doc = scalar_to_ssa(doc_id_kv, cutlass.Int32)
        return q_doc == kv_doc

    # Create kernel with fast sampling
    kernel = BlockSparsityKernel(
        mask_mod=document_mask,
        tile_mn=(tile_m, tile_n),
        compute_full_blocks=True,
        use_fast_sampling=True,
    )

    # Compile kernel
    compiled_kernel = cute.compile(
        kernel,
        blocksparse_tensors=blocksparse_tensors,
        aux_tensors=[doc_ids_cute],
    )

    # Generator function for benchmark - creates fresh tensors for each iteration
    def generate_tensors():
        from cutlass.cute.testing import JitArguments

        # Return fresh tensors for each benchmark iteration
        return JitArguments(blocksparse_tensors, [doc_ids_cute])

    # Run benchmark
    print("\nBenchmarking with document mask (uses aux_tensors)...")
    exec_time = cute_benchmark(
        compiled_kernel,
        workspace_generator=generate_tensors,
        warmup_iterations=10,
        iterations=100,
    )

    print(f"\nBenchmark Results:")
    print(f"  Execution time: {exec_time:.4f} us")
    print(f"  Execution time: {exec_time / 1000:.4f} ms")

    # Calculate throughput
    total_elements = batch_size * num_heads * n_blocks_q * n_blocks_k * tile_m * tile_n
    throughput = total_elements / (exec_time * 1e-6) / 1e9  # Billion elements per second
    print(f"\nThroughput: {throughput:.2f} billion elements/second")
    print("=" * 60)


def verify_against_pytorch():
    """Verify kernel results against PyTorch reference implementation."""
    import torch
    from cutlass.cute.runtime import from_dlpack

    print("=" * 60)
    print("Verification against PyTorch Reference")
    print("=" * 60)

    # Configuration
    batch_size = 2
    num_heads = 4
    seqlen_q = 512
    seqlen_k = 512
    tile_m, tile_n = 64, 64

    # Calculate number of blocks
    n_blocks_q = (seqlen_q + tile_m - 1) // tile_m
    n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}, Num heads: {num_heads}")
    print(f"  Sequence length Q: {seqlen_q}, K: {seqlen_k}")
    print(f"  Tile size: {tile_m} x {tile_n}")

    device = "cuda"

    # Test multiple mask types
    test_cases = []

    # Test 1: Causal mask
    @cute.jit
    def causal_mask(b, h, q, kv, aux):
        return q >= kv

    test_cases.append(("Causal mask", causal_mask, None))

    # Test 2: Document mask with random boundaries
    doc_ids = torch.zeros((batch_size, seqlen_q), device=device, dtype=torch.int32)
    for b in range(batch_size):
        pos = 0
        doc_id = 0
        while pos < seqlen_q:
            doc_len = torch.randint(32, 129, (1,)).item()
            end_pos = min(pos + doc_len, seqlen_q)
            doc_ids[b, pos:end_pos] = doc_id
            pos = end_pos
            doc_id += 1

    @cute.jit
    def document_mask(b, h, q, kv, aux):
        doc_ids = aux[0]
        # Direct scalar indexing
        q_doc_id = doc_ids[b, q]
        kv_doc_id = doc_ids[b, kv]
        return q_doc_id == kv_doc_id

    doc_ids_cute = from_dlpack(doc_ids.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
    test_cases.append(("Document mask", document_mask, [doc_ids_cute]))

    # Test 3: Sliding window mask
    @cute.jit
    def sliding_window_mask(b, h, q, kv, aux):
        return (q >= kv) & (q - kv < 128)

    test_cases.append(("Sliding window (128)", sliding_window_mask, None))

    for test_name, mask_fn, aux_tensors in test_cases:
        print(f"\n--- Testing: {test_name} ---")

        # Create output tensors for kernel
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

        # Run kernel
        kernel = BlockSparsityKernel(
            mask_mod=mask_fn, tile_mn=(tile_m, tile_n), compute_full_blocks=True
        )
        kernel(blocksparse_tensors=blocksparse_tensors, aux_tensors=aux_tensors)

        # Compute reference using PyTorch
        mask_block_cnt_ref = torch.zeros(
            (batch_size, num_heads, n_blocks_q), device=device, dtype=torch.int32
        )
        mask_block_idx_ref = torch.zeros(
            (batch_size, num_heads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
        )
        full_block_cnt_ref = torch.zeros(
            (batch_size, num_heads, n_blocks_q), device=device, dtype=torch.int32
        )
        full_block_idx_ref = torch.zeros(
            (batch_size, num_heads, n_blocks_q, n_blocks_k), device=device, dtype=torch.int32
        )

        # Build full mask matrix for reference
        for b in range(batch_size):
            for h in range(num_heads):
                # Create full mask matrix
                mask_matrix = torch.zeros((seqlen_q, seqlen_k), device=device, dtype=torch.bool)
                for q in range(seqlen_q):
                    for kv in range(seqlen_k):
                        # Call mask function
                        if test_name == "Document mask":
                            mask_val = doc_ids[b, q] == doc_ids[b, kv]
                        elif test_name == "Causal mask":
                            mask_val = q >= kv
                        elif test_name == "Sliding window (128)":
                            mask_val = (q >= kv) and (q - kv < 128)
                        mask_matrix[q, kv] = mask_val

                # Check each block
                for m_block in range(n_blocks_q):
                    num_mask = 0
                    num_full = 0
                    for n_block in range(n_blocks_k):
                        m_start = m_block * tile_m
                        m_end = min(m_start + tile_m, seqlen_q)
                        n_start = n_block * tile_n
                        n_end = min(n_start + tile_n, seqlen_k)

                        block_mask = mask_matrix[m_start:m_end, n_start:n_end]
                        has_unmasked = block_mask.any().item()
                        has_masked = (~block_mask).any().item()

                        if has_unmasked and has_masked:
                            # Partial block
                            mask_block_idx_ref[b, h, m_block, num_mask] = n_block
                            num_mask += 1
                        elif has_unmasked and not has_masked:
                            # Full block
                            full_block_idx_ref[b, h, m_block, num_full] = n_block
                            num_full += 1
                        # else: fully masked, skip

                    mask_block_cnt_ref[b, h, m_block] = num_mask
                    full_block_cnt_ref[b, h, m_block] = num_full

        # Compare results
        mask_cnt_match = torch.all(mask_block_cnt == mask_block_cnt_ref).item()
        full_cnt_match = torch.all(full_block_cnt == full_block_cnt_ref).item()

        # Check indices (only up to count)
        mask_idx_match = True
        full_idx_match = True
        for b in range(batch_size):
            for h in range(num_heads):
                for m in range(n_blocks_q):
                    num_mask = mask_block_cnt[b, h, m].item()
                    num_full = full_block_cnt[b, h, m].item()

                    if num_mask > 0:
                        mask_idx_match &= torch.all(
                            mask_block_idx[b, h, m, :num_mask]
                            == mask_block_idx_ref[b, h, m, :num_mask]
                        ).item()

                    if num_full > 0:
                        full_idx_match &= torch.all(
                            full_block_idx[b, h, m, :num_full]
                            == full_block_idx_ref[b, h, m, :num_full]
                        ).item()

        print(f"  Mask counts match: {mask_cnt_match}")
        print(f"  Full counts match: {full_cnt_match}")
        print(f"  Mask indices match: {mask_idx_match}")
        print(f"  Full indices match: {full_idx_match}")

        if mask_cnt_match and full_cnt_match and mask_idx_match and full_idx_match:
            print(f"  ✓ {test_name} PASSED")
        else:
            print(f"  ✗ {test_name} FAILED")
            # Print first mismatch for debugging
            for b in range(batch_size):
                for h in range(num_heads):
                    for m in range(n_blocks_q):
                        if mask_block_cnt[b, h, m] != mask_block_cnt_ref[b, h, m]:
                            print(
                                f"    First mask count mismatch at [{b},{h},{m}]: got {mask_block_cnt[b, h, m]}, expected {mask_block_cnt_ref[b, h, m]}"
                            )
                            # Debug: show what blocks were found
                            num_mask_kernel = mask_block_cnt[b, h, m].item()
                            num_mask_ref = mask_block_cnt_ref[b, h, m].item()
                            print(
                                f"    Kernel found mask blocks: {mask_block_idx[b, h, m, :num_mask_kernel].cpu().tolist()}"
                            )
                            print(
                                f"    Reference found mask blocks: {mask_block_idx_ref[b, h, m, :num_mask_ref].cpu().tolist()}"
                            )
                            num_full_kernel = full_block_cnt[b, h, m].item()
                            num_full_ref = full_block_cnt_ref[b, h, m].item()
                            print(
                                f"    Kernel found full blocks: {full_block_idx[b, h, m, :num_full_kernel].cpu().tolist()}"
                            )
                            print(
                                f"    Reference found full blocks: {full_block_idx_ref[b, h, m, :num_full_ref].cpu().tolist()}"
                            )

                            # Check the specific blocks in question
                            m_start = m * tile_m
                            m_end = min(m_start + tile_m, seqlen_q)
                            print(f"\n    Analyzing m_block {m} (rows {m_start}-{m_end - 1}):")

                            if test_name == "Document mask":
                                # Show document IDs for this m_block
                                print(
                                    f"    Document IDs for batch {b} in this m_block: {doc_ids[b, m_start:m_end].cpu().tolist()}"
                                )

                            return

    print("\n" + "=" * 60)


def stress_test_worst_cases():
    """Stress test with worst-case mask patterns."""
    import torch
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cute.testing import benchmark as cute_benchmark

    print("=" * 60)
    print("Stress Testing Worst-Case Mask Patterns")
    print("=" * 60)

    batch_size = 2
    num_heads = 8
    seqlen_q = 4096
    seqlen_k = 4096
    tile_m, tile_n = 128, 128

    n_blocks_q = (seqlen_q + tile_m - 1) // tile_m
    n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

    device = "cuda"

    test_cases = []

    # Worst case 1: Random mask (worst for branch prediction and caching)
    torch.manual_seed(42)
    random_mask_matrix = torch.rand((batch_size, seqlen_q, seqlen_k), device=device) > 0.5
    random_mask_cute = from_dlpack(
        random_mask_matrix.detach(), assumed_align=1
    ).mark_layout_dynamic(leading_dim=2)

    @cute.jit
    def random_mask(b, h, q, kv, aux):
        mask_matrix = aux[0]
        # Direct scalar indexing
        return mask_matrix[b, q, kv]

    test_cases.append(("Random 50% sparsity", random_mask, [random_mask_cute]))

    # Worst case 2: Checkerboard pattern (guarantees all blocks are partial)
    @cute.jit
    def checkerboard_mask(b, h, q, kv, aux):
        return ((q + kv) % 2) == 0

    test_cases.append(("Checkerboard (all partial blocks)", checkerboard_mask, None))

    # Worst case 3: Multi-aux-tensor access (stress memory bandwidth)
    aux1 = torch.randint(0, 8, (batch_size, seqlen_q), device=device, dtype=torch.int32)
    aux2 = torch.randint(0, 8, (batch_size, seqlen_k), device=device, dtype=torch.int32)
    aux3 = torch.randint(0, 16, (batch_size, num_heads), device=device, dtype=torch.int32)

    aux1_cute = from_dlpack(aux1.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
    aux2_cute = from_dlpack(aux2.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
    aux3_cute = from_dlpack(aux3.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)

    @cute.jit
    def multi_aux_mask(b, h, q, kv, aux):
        # Multiple aux tensor accesses - direct scalar indexing
        val1 = aux[0][b, q]
        val2 = aux[1][b, kv]
        val3 = aux[2][b, h]
        # Complex condition with all three - use bitwise operators
        return (val1 == val2) | (val1 < val3) | (val2 < val3)

    test_cases.append(
        ("Multi-aux-tensor (3 accesses)", multi_aux_mask, [aux1_cute, aux2_cute, aux3_cute])
    )

    # Worst case 4: Sliding window with random gaps
    gap_mask = torch.rand((batch_size, seqlen_q), device=device) > 0.3
    gap_mask_cute = from_dlpack(gap_mask.detach(), assumed_align=1).mark_layout_dynamic(
        leading_dim=1
    )

    @cute.jit
    def sliding_window_with_gaps(b, h, q, kv, aux):
        gap_mask = aux[0]
        in_window = (q >= kv) & (q - kv < 256)
        has_gap = gap_mask[b, q]
        return in_window & has_gap

    test_cases.append(("Sliding window + random gaps", sliding_window_with_gaps, [gap_mask_cute]))

    for test_name, mask_fn, aux_tensors in test_cases:
        print(f"\n--- {test_name} ---")
        print(f"  Batch size: {batch_size}, Num heads: {num_heads}")
        print(f"  Sequence length: {seqlen_q}")

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

        kernel = BlockSparsityKernel(tile_mn=(tile_m, tile_n), compute_full_blocks=True)

        # Compile
        compiled_kernel = cute.compile(
            kernel,
            mask_mod=mask_fn,
            blocksparse_tensors=blocksparse_tensors,
            aux_tensors=aux_tensors,
        )

        def generate_tensors():
            from cutlass.cute.testing import JitArguments

            return JitArguments(blocksparse_tensors, aux_tensors)

        # Benchmark
        exec_time = cute_benchmark(
            compiled_kernel,
            workspace_generator=generate_tensors,
            warmup_iterations=10,
            iterations=100,
        )

        # Analyze results
        total_blocks = batch_size * num_heads * n_blocks_q * n_blocks_k
        total_mask_blocks = mask_block_cnt.sum().item()
        total_full_blocks = full_block_cnt.sum().item()
        total_skip_blocks = total_blocks - total_mask_blocks - total_full_blocks

        print(f"  Execution time: {exec_time:.4f} us ({exec_time / 1000:.4f} ms)")
        total_elements = batch_size * num_heads * n_blocks_q * n_blocks_k * tile_m * tile_n
        throughput = total_elements / (exec_time * 1e-6) / 1e9
        print(f"  Throughput: {throughput:.2f} billion elements/second")
        print("  Block breakdown:")
        print(
            f"    Partial blocks (need masking): {total_mask_blocks} ({100 * total_mask_blocks / total_blocks:.1f}%)"
        )
        print(
            f"    Full blocks (no masking): {total_full_blocks} ({100 * total_full_blocks / total_blocks:.1f}%)"
        )
        print(
            f"    Skipped blocks (fully masked): {total_skip_blocks} ({100 * total_skip_blocks / total_blocks:.1f}%)"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run()
    print("\n")
    benchmark()
    print("\n")
    benchmark_with_aux_tensors()
    # print("\n")
    # verify_against_pytorch()
    # print("\n")
    # stress_test_worst_cases()
