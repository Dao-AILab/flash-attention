# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_prepare_scheduler.cu
# from Cutlass C++ to Cute-DSL.

import enum
import math
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial
from dataclasses import dataclass

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, BFloat16, Numeric
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

import flash_attn.cute.utils as utils
from flash_attn.cute import copy_utils
import flash_attn.cute.pipeline as pipeline
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.pack_gqa import PackGQA
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.fast_math import FastDivmod
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)


@dataclass
class FlashFwdParams:
    h: int
    h_k: int
    b: int
    is_e4m3: bool
    varlen_sort_batches: bool
    seqlen_q: cute.Tensor
    seqlen_k: cute.Tensor
    seqlen_knew: cute.Tensor
    cu_seqlens_q: cute.Tensor
    cu_seqlens_k: cute.Tensor
    cu_seqlens_knew: cute.Tensor
    seqused_q: cute.Tensor
    seqused_k: cute.Tensor
    leftpad_k: cute.Tensor
    num_sm: int
    num_splits: int
    tile_count_semaphore: cute.Tensor
    prepare_seqlen_q_ptr: cute.Tensor
    num_splits_dynamic_ptr: cute.Tensor
    varlen_batch_idx_ptr: cute.Tensor
    num_nheads_in_l2_ptr: cute.Tensor
    enable_pdl: bool
    is_causal: bool
    max_kvblocks_in_l2: int
    d: int
    dv: int
    packgqa: bool
    tile_m: int = 128
    tile_n: int = 128
    stream: cuda.CUstream


class FlashPrepareScheduler:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric] = BFloat16,
        # TODO: add more parameters
    ):
        self.dtype = dtype

    @cute.jit
    def __call__(
        self,
        params: FlashFwdParams,
        stream: cuda.CUstream,
        packgqa: bool = False,
        tile_m: int = 128,
        tile_n: int = 128,
    ):
        qhead_per_khead = cute.ceil_div(params.h, params.h_k)
        num_warps = cute.ceil_div(params.b, 31)  # max 31 batches per warp
        num_ctas = cute.ceil_div(params.b, 31 * 32)  # max 32 warps per CTA
        size_l2_divisor = 1 if qhead_per_khead == 1 else (2 if qhead_per_khead <= 2 else (
            4 if qhead_per_khead <= 4 else (8 if qhead_per_khead <= 8 else 16)))
        size_l2 = (32 * 1024 * 1024) / size_l2_divisor  # experimental
        element_size = 1 if params.is_e4m3 else 2
        size_one_kvblock = tile_n * (params.d + params.dv) * element_size
        max_kvblocks_in_l2 = size_l2 / size_one_kvblock

        self.k_smem_size = 1

        @cute.struct
        class SharedStorage:
            total_blocks_smem: cute.struct.MemRange[Int32, self.k_smem_size]
            # TODO: SMEM for sort operations

        self.shared_storage = SharedStorage

        self.kernel(
            params.seqlen_q, params.seqlen_k, params.seqlen_knew,
            params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
            params.seqused_q, params.seqused_k, params.leftpad_k,
            params.b, not packgqa ? params.h: params.h_k, qhead_per_khead, params.num_sm, params.num_splits,
            FastDivmod(tile_m), FastDivmod(tile_n),
            params.tile_count_semaphore,
            params.prepare_seqlen_q_ptr,
            params.num_splits_dynamic_ptr,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(32 * num_warps, 1, 1),
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
        k_num_batch_per_warp = self.num_threads_per_warp - 1
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
            batch_idx,
            seqused_q,
            cu_seqlens_q,
            seqlen_q_static,
            tile_m_divmod,
            num_batch,
            qhead_per_khead,
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
            for i in range(self.num_threads_per_warp / 2, 0, -1):
                total_blocks += cute.arch.shuffle_sync_down(total_blocks, offset=i)
            if lane_idx == 0:
                cute.arch.atomic_add(total_blocks_smem, total_blocks)
            cute.arch.sync_threads()
            
            total_blocks = total_blocks_smem[0]
            blocks_per_sm = cute.ceil_div(total_blocks * 1.1 * num_head / num_sm)
            num_splits_dynamic = max(min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1)
            num_n_blocks = cute.ceil_div(num_n_blocks, num_splits_dynamic)

        # TODO: sort
        if const_expr(False):
            num_n_blocks = Int32(0)
        else:
            if batch_idx < num_batch and lane_idx < k_num_batch_per_warp:
                prepare_seqlen_q_ptr[batch_idx] = seqlen_q * (1 if qhead_per_khead == 1 else qhead_per_khead)
                if const_expr(num_splits_dynamic_ptr is not None):
                    num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic 
                if const_expr(num_nheads_in_l2_ptr is not None): 
                    num_nheads_in_l2_ptr[batch_idx] = self.get_num_nheads_in_l2(max(num_n_blocks, 1), num_head, max_kvblocks_in_l2, pack_gqa, qhead_per_khead)

    @cute.jit
    def get_num_m_blocks_and_seqlen(
        self,
        batch_idx: Int32,
        seqused_q: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        seqlen_q_static: Int32,
        tile_m_divmod: FastDivmod,
        num_batch: Int32,
        qhead_per_khead: Int32,
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
        return (tile_m_divmod.div(seqlen * (1 if qhead_per_khead == 1 else qhead_per_khead) + tile_m_divmod.divisor - 1), seqlen)

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
        return tile_n_divmod.div(seqlen + tile_n_divmod.divisor - 1) if batch_idx < num_batch and lane_idx < k_num_batch_per_warp else Int32(0)

    @cute.jit
    def get_num_nheads_in_l2(
        self,
        num_n_blocks: Int32,
        num_head: Int32,
        max_kvblocks_in_l2: int,
        packgqa: bool,
        qhead_per_khead: Int32,
    ):
        nheads_in_l2 = 16 if num_n_blocks * 16 <= max_kvblocks_in_l2 else (8 if num_n_blocks * 8 <= max_kvblocks_in_l2 else (4 if num_n_blocks * 4 <= max_kvblocks_in_l2 else (2 if num_n_blocks * 2 <= max_kvblocks_in_l2 else 1)))
        if const_expr(not packgqa):
            nheads_in_l2 *= qhead_per_khead
        return min(nheads_in_l2, num_head)
