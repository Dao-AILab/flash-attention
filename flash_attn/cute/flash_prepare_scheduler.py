# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_prepare_scheduler.cu
# from Cutlass C++ to Cute-DSL.

import math
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr, Constexpr

from flash_attn.cute.fast_math import FastDivmod

# @cute.jit 
def ceil_div(a, b):
    return (a + b - 1) // b


class FlashPrepareScheduler:
    def __init__(
        self,
        sort: bool = False,
    ):
        """
        Initialize the FlashPrepareScheduler with compile-time parameters.

        Args:
            sort: Whether to sort batches (compile-time parameter, corresponds to Sort template parameter)
        """
        self.sort = sort
        self.num_threads_per_warp = 32
        self.k_num_batch_per_warp = self.num_threads_per_warp - 1
        
    def _setup_attributes(self):
        self.qhead_per_khead = 1
        # if self.nheads_kv > 0:
        #     self.qhead_per_khead = ceil_div(self.nheads, self.nheads_kv)
        self.num_warps = ceil_div(self.num_batch, 31)  # max 31 batches per warp
        self.num_ctas = ceil_div(self.num_batch, 31 * 32)  # max 32 warps per CTA
        
        # Compute L2 cache size and max KV blocks
        self.size_l2_divisor = (
            1
            if self.qhead_per_khead == 1
            else (
                2
                if self.qhead_per_khead <= 2
                else (4 if self.qhead_per_khead <= 4 else (8 if self.qhead_per_khead <= 8 else 16))
            )
        )
        self.size_l2 = (32 * 1024 * 1024) / self.size_l2_divisor  # experimental
        element_size = 1 if self.is_e4m3 else 2
        self.size_one_kvblock = self.tile_n * (self.d + self.dv) * element_size
        self.max_kvblocks_in_l2 = ceil_div(self.size_l2, self.size_one_kvblock)
        

    @cute.jit
    def __call__(
        self,
        seqlen_q_static: int,
        seqlen_k_static: int,
        seqlen_k_new_static: int,
        cu_seqlens_q: Optional[cute.Tensor],
        cu_seqlens_k: Optional[cute.Tensor],
        cu_seqlens_k_new: Optional[cute.Tensor],
        seqused_q: Optional[cute.Tensor],
        seqused_k: Optional[cute.Tensor],
        leftpad_k: Optional[cute.Tensor],
        num_batch: int,
        nheads: int,
        nheads_kv: int,
        num_sm: int,
        num_splits_static: int,
        tile_m: int,
        tile_n: int,
        tile_count_semaphore: Optional[cute.Tensor],
        prepare_seqlen_q_ptr: cute.Tensor,
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx_ptr: Optional[cute.Tensor],
        num_nheads_in_l2_ptr: Optional[cute.Tensor],
        enable_pdl: bool,
        is_causal: bool,
        packgqa: bool,
        is_e4m3: Constexpr[bool],
        d: int,
        dv: int,
        stream: cuda.CUstream,
    ):
        """
        Execute the prepare scheduler kernel.

        This corresponds to the C++ prepare_varlen_num_blocks function.
        All parameters are runtime parameters extracted from Flash_fwd_params.
        """
        self.nheads = nheads
        self.nheads_kv = nheads_kv
        self.num_batch = num_batch
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.d = d
        self.dv = dv
        self.is_e4m3 = is_e4m3
        self.enable_pdl = enable_pdl
        self.is_causal = is_causal
        self.packgqa = packgqa
        self.num_splits_static = num_splits_static
        
        self._setup_attributes()

        

        self.k_smem_size = 1

        @cute.struct
        class SharedStorage:
            total_blocks_smem: cute.struct.MemRange[Int32, self.k_smem_size]
            # TODO: SMEM for sort operations (BlockMergeSort temp storage)

        self.shared_storage = SharedStorage

        self.num_head_computed = self.nheads if not self.packgqa else self.nheads_kv

        # Create FastDivmod objects for efficient division
        tile_m_divmod = FastDivmod.create(Int32(tile_m))
        tile_n_divmod = FastDivmod.create(Int32(tile_n))

        # Convert Python integers to Int32 for kernel
        qhead_per_khead_int32 = Int32(self.qhead_per_khead)
        breakpoint()
        self.kernel(
            Int32(seqlen_q_static),
            Int32(seqlen_k_static),
            Int32(seqlen_k_new_static),
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_k_new,
            seqused_q,
            seqused_k,
            leftpad_k,
            Int32(self.num_batch),
            Int32(self.num_head_computed),
            Int32(self.qhead_per_khead),
            Int32(num_sm),
            Int32(self.num_splits_static),
            tile_m_divmod,
            tile_n_divmod,
            tile_count_semaphore,
            prepare_seqlen_q_ptr,
            num_splits_dynamic_ptr,
            varlen_batch_idx_ptr,
            num_nheads_in_l2_ptr,
            enable_pdl,
            is_causal,
            packgqa,
            self.max_kvblocks_in_l2,
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[32 * self.num_warps, 1, 1],
            stream=stream,
            smem=self.shared_storage.size_in_bytes(),
        )

    @cute.kernel
    def kernel(
        self,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        seqlen_k_new_static: Int32,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        cu_seqlens_k_new: cute.Tensor,
        seqused_q: cute.Tensor,
        seqused_k: cute.Tensor,
        leftpad_k_ptr: cute.Pointer,
        num_batch: Int32,
        num_head: Int32,
        qhead_per_khead: Int32,
        num_sm: Int32,
        num_splits_static: Int32,
        tile_m_divmod: FastDivmod,
        tile_n_divmod: FastDivmod,
        tile_count_semaphore: Optional[cute.Tensor],
        prepare_seqlen_q_ptr: cute.Tensor,
        num_splits_dynamic_ptr: cute.Tensor,
        varlen_batch_idx_ptr: cute.Tensor,
        num_nheads_in_l2_ptr: cute.Tensor,
        enable_pdl: bool,
        is_causal: bool,
        packgqa: bool,
        max_kvblocks_in_l2: int,
    ):
        k_num_batch_per_warp = self.k_num_batch_per_warp
        bdimx, _, _ = cute.arch.block_dim()
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        grid_dimx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        items_per_thread = 1

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        total_blocks_smem = storage.total_blocks_smem.get_tensor((1,))

        if tidx == 0:
            total_blocks_smem[0] = 0
        cute.arch.sync_threads()

        if tidx == 0 and const_expr(tile_count_semaphore is not None):
            tile_count_semaphore[0] = 0

        batch_cta_idx_offset = bidx * 992
        bidb_start = batch_cta_idx_offset + k_num_batch_per_warp * warp_idx
        batch_idx = lane_idx + bidb_start

        num_m_blocks, seqlen_q = self.get_num_m_blocks_and_seqlen(
            lane_idx,
            batch_idx,
            k_num_batch_per_warp,
            seqused_q,
            cu_seqlens_q,
            seqlen_q_static,
            tile_m_divmod,
            num_batch,
            qhead_per_khead,
            packgqa,
        )

        num_n_blocks = self.get_num_n_blocks(
            lane_idx,
            batch_idx,
            k_num_batch_per_warp,
            seqused_k,
            cu_seqlens_k,
            cu_seqlens_k_new,
            seqlen_k_static,
            seqlen_k_new_static,
            leftpad_k_ptr,
            tile_n_divmod,
            num_batch,
        )

        if const_expr(grid_dimx > 1 or num_splits_static == 1):
            num_splits_dynamic = 1
        else:
            total_blocks = num_m_blocks * num_n_blocks
            # Warp sum
            for i in range(self.num_threads_per_warp // 2, 0, -1):
                total_blocks += cute.arch.shuffle_sync_down(total_blocks, offset=i)
            if lane_idx == 0:
                cute.arch.atomic_add(total_blocks_smem, total_blocks)
            cute.arch.sync_threads()

            total_blocks = total_blocks_smem[0]
            blocks_per_sm = cute.ceil_div(total_blocks * 1.1 * num_head / num_sm)
            num_splits_dynamic = max(
                min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1
            )
            num_n_blocks = cute.ceil_div(num_n_blocks, num_splits_dynamic)

        # TODO: sort
        if const_expr(self.sort):
            # Sort logic will be implemented later
            num_n_blocks = Int32(0)
        else:
            if batch_idx < num_batch and lane_idx < k_num_batch_per_warp:
                prepare_seqlen_q_ptr[batch_idx] = seqlen_q * (qhead_per_khead if packgqa else 1)
                if const_expr(num_splits_dynamic_ptr is not None):
                    num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic
                if const_expr(num_nheads_in_l2_ptr is not None):
                    num_nheads_in_l2_ptr[batch_idx] = self.get_num_nheads_in_l2(
                        max(num_n_blocks, 1), num_head, max_kvblocks_in_l2, packgqa, qhead_per_khead
                    )

    @cute.jit
    def get_num_m_blocks_and_seqlen(
        self,
        lane_idx: Int32,
        batch_idx: Int32,
        k_num_batch_per_warp: Int32,
        seqused_q: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        seqlen_q_static: Int32,
        tile_m_divmod: FastDivmod,
        num_batch: Int32,
        qhead_per_khead: Int32,
        packgqa: bool,
    ):
        seqlen = Int32(0)
        if const_expr(seqused_q is not None):
            seqlen = seqused_q[batch_idx] if batch_idx < num_batch else Int32(0)
        elif const_expr(cu_seqlens_q is not None):
            cur_cu_seqlen = cu_seqlens_q[batch_idx] if batch_idx <= num_batch else Int32(0)
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_q_static
        # For packgqa, multiply seqlen by qhead_per_khead; otherwise use seqlen as-is
        # This matches C++: seqlen * (packgqa ? qhead_per_khead : 1)
        seqlen_for_blocks = seqlen * (qhead_per_khead if packgqa else 1)
        num_m_blocks = (
            tile_m_divmod.div(seqlen_for_blocks + tile_m_divmod.divisor - 1)
            if batch_idx < num_batch and lane_idx < k_num_batch_per_warp
            else Int32(0)
        )
        return (num_m_blocks, seqlen)

    @cute.jit
    def get_num_n_blocks(
        self,
        lane_idx: Int32,
        batch_idx: Int32,
        k_num_batch_per_warp: Int32,
        seqused_k: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        cu_seqlens_k_new: cute.Tensor,
        seqlen_k_static: Int32,
        seqlen_k_new_static: Int32,
        leftpad_k_ptr: cute.Pointer,
        tile_n_divmod: FastDivmod,
        num_batch: Int32,
    ):
        leftpad_k = leftpad_k_ptr[batch_idx] if batch_idx < num_batch else Int32(0)
        seqlen = Int32(0)
        if const_expr(seqused_k is not None):
            seqlen = seqused_k[batch_idx] if batch_idx < num_batch else Int32(0)
        elif const_expr(cu_seqlens_k is not None):
            cur_cu_seqlen = cu_seqlens_k[batch_idx] if batch_idx <= num_batch else Int32(0)
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_k_static

        seqlen_new = Int32(0)
        if const_expr(cu_seqlens_k_new is not None):
            cur_cu_seqlen_new = cu_seqlens_k_new[batch_idx] if batch_idx <= num_batch else Int32(0)
            next_cu_seqlen_new = cute.arch.shuffle_sync_down(cur_cu_seqlen_new, offset=1)
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new
        else:
            seqlen_new = seqlen_k_new_static
        seqlen = seqlen - leftpad_k + seqlen_new
        return (
            tile_n_divmod.div(seqlen + tile_n_divmod.divisor - 1)
            if batch_idx < num_batch and lane_idx < k_num_batch_per_warp
            else Int32(0)
        )

    @cute.jit
    def get_num_nheads_in_l2(
        self,
        num_n_blocks: Int32,
        num_head: Int32,
        max_kvblocks_in_l2: int,
        packgqa: bool,
        qhead_per_khead: Int32,
    ):
        nheads_in_l2 = (
            16
            if num_n_blocks * 16 <= max_kvblocks_in_l2
            else (
                8
                if num_n_blocks * 8 <= max_kvblocks_in_l2
                else (
                    4
                    if num_n_blocks * 4 <= max_kvblocks_in_l2
                    else (2 if num_n_blocks * 2 <= max_kvblocks_in_l2 else 1)
                )
            )
        )
        if const_expr(not packgqa):
            nheads_in_l2 *= qhead_per_khead
        return min(nheads_in_l2, num_head)
