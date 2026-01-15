# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_prepare_scheduler.cu
# from CUTLASS C++ to Cute-DSL.

from typing import Tuple, Optional, Callable, List, NamedTuple
import operator
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr, Constexpr, Float32
from cutlass.cute import FastDivmodDivisor
import flash_attn.cute.utils as utils


class SchedulerMetadataTensorsTorch(NamedTuple):
    """Class to store scheduler metadata for varlen"""
    # tensors of shape (batch)
    num_m_blocks_ptr: Optional[torch.Tensor]
    num_splits_dynamic_ptr: Optional[torch.Tensor]
    varlen_batch_idx_ptr: Optional[torch.Tensor]
    num_nheads_in_l2_ptr: Optional[torch.Tensor]
    # tensor of shape (1)
    tile_count_semaphore: Optional[torch.Tensor]


class FlashPrepareScheduler:
    def __init__(
        self,
        num_warps: int,
        tile_m: int,
        tile_n: int,
        nheads: int,
        nheads_kv: int,
        headdim: int,
        headdim_v: Optional[int] = None,
        is_causal: bool = False,
        packgqa: bool = False,
        sort: bool = False,
    ):
        self.num_warps = num_warps
        self.is_causal = is_causal
        self.packgqa = packgqa
        # TODO: Implement batch sort for LPT.
        self.sort = False
        self.num_threads_per_warp = 32
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.d = headdim
        self.dv = headdim_v if headdim_v is not None else headdim
        self.k_num_batch_per_warp = 31
        self.k_smem_size = 1

        # for pack gqa, query heads per kv head is combined with seqlen_q
        self.nheads_computed = nheads if not self.packgqa else nheads_kv

        # L2 cache calculations
        self.qhead_per_khead = nheads // nheads_kv
        self.size_l2_divisor = (
            1
            if self.qhead_per_khead == 1
            else (
                2
                if self.qhead_per_khead <= 2
                else (4 if self.qhead_per_khead <= 4 else (8 if self.qhead_per_khead <= 8 else 16))
            )
        )
        self.size_l2 = (32 * 1024 * 1024) // self.size_l2_divisor
        element_size = 2
        self.size_one_kvblock = self.tile_n * (self.d + self.dv) * element_size
        self.max_kvblocks_in_l2 = (
            self.size_l2 + self.size_one_kvblock - 1
        ) // self.size_one_kvblock

    @staticmethod
    def get_grid_shape(num_batch: int) -> Tuple[int, int, int]:
        num_ctas = (num_batch + (31 * 32 - 1)) // (31 * 32)
        return (num_ctas, 1, 1)

    @cute.jit
    def __call__(
        self,
        seqlen_q_static: int,
        seqlen_k_static: int,
        seqlen_k_new_static: int,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensKNew: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mLeftPadK: Optional[cute.Tensor],
        num_batch: int,
        num_splits_static: int,
        tile_count_semaphore: Optional[cute.Tensor],
        num_m_blocks_ptr: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx_ptr: Optional[cute.Tensor],
        num_nheads_in_l2_ptr: Optional[cute.Tensor],
        n_blocks_per_split: Optional[int], # overrides heuristic
        stream: cuda.CUstream,
    ):
        tile_m_divmod = FastDivmodDivisor(self.tile_m)
        tile_n_divmod = FastDivmodDivisor(self.tile_n)

        @cute.struct
        class SharedStorage:
            total_blocks_smem: cute.struct.MemRange[Int32, self.k_smem_size]

        self.shared_storage = SharedStorage

        block = (32 * self.num_warps, 1, 1)
        grid = self.get_grid_shape(num_batch)

        hardware_info = cutlass.utils.HardwareInfo()
        num_sm = hardware_info.get_device_multiprocessor_count()

        self.kernel(
            seqlen_q_static,
            seqlen_k_static,
            seqlen_k_new_static,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensKNew,
            mSeqUsedQ,
            mSeqUsedK,
            mLeftPadK,
            num_batch,
            num_sm,
            num_splits_static,
            tile_m_divmod,
            tile_n_divmod,
            tile_count_semaphore,
            num_m_blocks_ptr,
            num_splits_dynamic_ptr,
            varlen_batch_idx_ptr,
            num_nheads_in_l2_ptr,
            n_blocks_per_split,
        ).launch(
            grid=grid,
            block=block,
            stream=stream,
            smem=self.shared_storage.size_in_bytes(),
        )

    @cute.kernel
    def kernel(
        self,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        seqlen_k_new_static: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensKNew: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mLeftPadK: Optional[cute.Tensor],
        num_batch: Int32,
        num_sm: Int32,
        num_splits_static: Int32,
        tile_m_divmod: FastDivmodDivisor,
        tile_n_divmod: FastDivmodDivisor,
        tile_count_semaphore: Optional[cute.Tensor],
        num_m_blocks_ptr: Optional[cute.Tensor],
        num_splits_dynamic_ptr: Optional[cute.Tensor],
        varlen_batch_idx_ptr: Optional[cute.Tensor],
        num_nheads_in_l2_ptr: Optional[cute.Tensor],
        n_blocks_per_split: Optional[Int32],
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        grid_dimx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        total_blocks_smem = storage.total_blocks_smem.get_tensor((1,))

        if tidx == 0:
            total_blocks_smem[0] = Int32(0)
        cute.arch.sync_threads()

        if const_expr(tile_count_semaphore is not None):
            if tidx == 0:
                tile_count_semaphore[0] = Int32(0)

        batch_cta_idx_offset = bidx * 992
        bidb_start = batch_cta_idx_offset + self.k_num_batch_per_warp * warp_idx
        batch_idx = lane_idx + bidb_start

        num_m_blocks, seqlen_q = self.get_num_m_blocks_and_seqlen(
            lane_idx,
            batch_idx,
            mSeqUsedQ,
            mCuSeqlensQ,
            seqlen_q_static,
            tile_m_divmod,
            num_batch,
        )

        num_n_blocks = self.get_num_n_blocks(
            lane_idx,
            batch_idx,
            mSeqUsedK,
            mCuSeqlensK,
            mCuSeqlensKNew,
            seqlen_k_static,
            seqlen_k_new_static,
            mLeftPadK,
            tile_n_divmod,
            num_batch,
        )

        num_splits_dynamic = Int32(0)
        if const_expr(n_blocks_per_split is not None):
            # print("n_blocks_per_splits = ", n_blocks_per_split)
            num_splits_dynamic = cutlass.max(
                cutlass.min(cute.ceil_div(num_n_blocks, n_blocks_per_split), num_splits_static), Int32(1)
            )
            num_n_blocks = cute.ceil_div(num_n_blocks, num_splits_dynamic)
        else:
            if grid_dimx > 1 or num_splits_static == 1:
                num_splits_dynamic = Int32(1)
            else:
                total_blocks = num_m_blocks * num_n_blocks
                total_blocks = utils.warp_reduce(total_blocks, operator.add)
                if lane_idx == 0:
                    utils.atomic_add_i32(total_blocks, total_blocks_smem.iterator)
                    
                cute.arch.sync_threads()

                total_blocks = total_blocks_smem[0]
                
                sm_margin = max(Float32(num_sm) / 128 + .001, 1.1) # e.g. 148/128 = 1.15625
                blocks_per_sm = cutlass.max(
                    Int32((Float32(total_blocks) * sm_margin * Float32(self.nheads_computed) / Float32(num_sm))),
                    Int32(1)
                )
                # blocks_per_sm = cute.ceil_div(total_blocks * self.nheads_computed, num_sm)
                num_splits_dynamic = cutlass.max(
                    cutlass.min(cute.ceil_div(num_n_blocks, blocks_per_sm), num_splits_static), Int32(1)
                )
                # if tidx == 0:
                #     cute.printf("num_batch = {}", num_batch)
                #     cute.printf("num_m_blocks = {}", num_m_blocks)
                #     cute.printf("num_n_blocks = {}", num_n_blocks)
                #     cute.printf("total_blocks = {}", total_blocks)
                #     cute.printf("numerator = {}", total_blocks * self.nheads_computed)
                #     cute.printf("denominator num_sm = {}", num_sm)
                #     cute.printf("blocks_per_sm = {}", blocks_per_sm)
                #     cute.printf("sm margin = {}", sm_margin)
                #     cute.printf("num_splits_dynamic = {}", num_splits_dynamic)
                num_n_blocks = cute.ceil_div(num_n_blocks, num_splits_dynamic)

        if const_expr(self.sort):
            # TODO: Implement sort logic
            pass
        
        if batch_idx < num_batch and lane_idx < self.k_num_batch_per_warp:
            if const_expr(num_m_blocks_ptr is not None):
                num_m_blocks_ptr[batch_idx] = num_m_blocks
            if const_expr(num_splits_dynamic_ptr is not None):
                num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic
            if const_expr(num_nheads_in_l2_ptr is not None):
                nheads_in_l2 = self.get_num_nheads_in_l2(num_n_blocks)
                num_nheads_in_l2_ptr[batch_idx] = nheads_in_l2


    @cute.jit
    def get_num_m_blocks_and_seqlen(
        self,
        lane_idx: Int32,
        batch_idx: Int32,
        mSeqUsedQ: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        seqlen_q_static: Int32,
        tile_m_divmod: FastDivmodDivisor,
        num_batch: Int32
    ):
        seqlen = Int32(0)
        if const_expr(mSeqUsedQ is not None):
            seqlen = mSeqUsedQ[batch_idx] if batch_idx < num_batch else Int32(0)
        elif const_expr(mCuSeqlensQ is not None):
            # Since k_num_batch_per_warp = 31, lane 31 never processes batches
            # So shuffle_down is safe: lane 30 gets lane 31's value (which is 0)
            # Only access cu_seqlens if batch_idx is valid (0 to num_batch inclusive)
            cur_cu_seqlen = Int32(0)
            if batch_idx <= num_batch:
                cur_cu_seqlen = mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_q_static

        seqlen_for_blocks = seqlen
        if const_expr(self.packgqa):
            seqlen_for_blocks = seqlen * self.qhead_per_khead
        num_m_blocks = (
            (seqlen_for_blocks + self.tile_m - 1) // tile_m_divmod
            if batch_idx < num_batch and lane_idx < self.k_num_batch_per_warp
            else Int32(0)
        )
        return (num_m_blocks, seqlen)

    @cute.jit
    def get_num_n_blocks(
        self,
        lane_idx: Int32,
        batch_idx: Int32,
        mSeqUsedK: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensKNew: Optional[cute.Tensor],
        seqlen_k_static: Int32,
        seqlen_k_new_static: Int32,
        mLeftPadK: Optional[cute.Tensor],
        tile_n_divmod: FastDivmodDivisor,
        num_batch: Int32
    ):
        leftpad_k = (
            mLeftPadK[batch_idx]
            if const_expr(mLeftPadK is not None) and batch_idx < num_batch
            else Int32(0)
        )
        seqlen = Int32(0)
        if const_expr(mSeqUsedK is not None):
            seqlen = mSeqUsedK[batch_idx] if batch_idx < num_batch else Int32(0)
        elif const_expr(mCuSeqlensK is not None):
            # Since k_num_batch_per_warp = 31, lane 31 never processes batches
            # So shuffle_down is safe: lane 30 gets lane 31's value (which is 0)
            # Only access cu_seqlens if batch_idx is valid (0 to num_batch inclusive)
            cur_cu_seqlen = Int32(0)
            if batch_idx <= num_batch:
                cur_cu_seqlen = mCuSeqlensK[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_k_static

        seqlen_new = Int32(0)
        if const_expr(mCuSeqlensKNew is not None):
            # Since k_num_batch_per_warp = 31, lane 31 never processes batches
            # So shuffle_down is safe: lane 30 gets lane 31's value (which is 0)
            # Only access cu_seqlens if batch_idx is valid (0 to num_batch inclusive)
            cur_cu_seqlen_new = Int32(0)
            if batch_idx <= num_batch:
                cur_cu_seqlen_new = mCuSeqlensKNew[batch_idx]
            next_cu_seqlen_new = cute.arch.shuffle_sync_down(cur_cu_seqlen_new, offset=1)
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new
        else:
            seqlen_new = seqlen_k_new_static
        seqlen = seqlen - leftpad_k + seqlen_new
        return (
            (seqlen + self.tile_n - 1) // tile_n_divmod
            if batch_idx < num_batch and lane_idx < self.k_num_batch_per_warp
            else Int32(0)
        )
        

    @cute.jit
    def get_num_nheads_in_l2(
        self,
        num_n_blocks: Int32,
    ):
        max_kvblocks_in_l2 = self.max_kvblocks_in_l2
        qhead_per_khead = self.qhead_per_khead
        nheads_in_l2 = Int32(16)
        if num_n_blocks * Int32(16) <= max_kvblocks_in_l2:
            nheads_in_l2 = Int32(16)
        elif num_n_blocks * Int32(8) <= max_kvblocks_in_l2:
            nheads_in_l2 = Int32(8)
        elif num_n_blocks * Int32(4) <= max_kvblocks_in_l2:
            nheads_in_l2 = Int32(4)
        elif num_n_blocks * Int32(2) <= max_kvblocks_in_l2:
            nheads_in_l2 = Int32(2)
        else:
            nheads_in_l2 = Int32(1)
        if const_expr(not self.packgqa):
            nheads_in_l2 *= qhead_per_khead
        return cutlass.min(nheads_in_l2, self.nheads_computed)