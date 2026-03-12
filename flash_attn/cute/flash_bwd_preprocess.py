# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_preprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
import operator
from functools import partial
from typing import Callable, Type, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack import copy_utils

from flash_attn.cute import utils
from flash_attn.cute.seqlen_info import SeqlenInfo
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FlashAttentionBackwardPreprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        tile_m: int = 128,
        num_threads: int = 256,
    ):
        """
        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param num_threads: number of threads
        :type num_threads: int
        """
        self.dtype = dtype
        self.tile_m = tile_m
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.num_threads = num_threads

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param num_threads: number of threads
        :type num_threads: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < tile_m:  # For multiplying lse with log2
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        # We want kBlockKGmem to be a power of 2 so that when we do the summing,
        # it's just between threads in the same warp
        gmem_k_block_size = (
            128
            if self.head_dim_v_padded % 128 == 0
            else (
                64
                if self.head_dim_v_padded % 64 == 0
                else (32 if self.head_dim_v_padded % 32 == 0 else 16)
            )
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = gmem_k_block_size // num_copy_elems
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // Float32.width
        assert (
            self.tile_m * self.head_dim_padded // num_copy_elems_dQaccum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            Float32, self.num_threads, num_copy_elems_dQaccum
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mPdPsum.element_type not in [Float32]):
            raise TypeError("PdPsum tensor must be Float32")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if const_expr(mLSE is not None):
            assert mLSElog2 is not None, "If mLSE is provided, mLSElog2 must also be provided"
            if const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
            if const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")

        self._setup_attributes()

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_head = mO.shape[1]
            num_batch = mCuSeqlensQ.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_head = mO.shape[2]
            num_batch = mO.shape[0]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[1], self.tile_m),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[2],
            total_q=mO.shape[0],
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            mCuSeqlensQ,
            mSeqUsedQ,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m)
            mO_cur = seqlen.offset_batch(mO, batch_idx, dim=0)[None, head_idx, None]
            mdO_cur = seqlen.offset_batch(mdO, batch_idx, dim=0)[None, head_idx, None]
            offset_padded = None if const_expr(not seqlen.has_cu_seqlens) else seqlen.offset_padded
            if const_expr(not seqlen.has_cu_seqlens):
                mPdPsum_cur = mPdPsum[batch_idx, head_idx, None]
            else:
                mPdPsum_cur = cute.domain_offset((offset_padded,), mPdPsum[head_idx, None])
            headdim_v = mO.shape[cute.rank(mO) - 1]
            seqlen_q = seqlen.seqlen
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
            seqlen_limit = seqlen_q - m_block * self.tile_m

            lse = None
            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens):
                    mLSE_cur = mLSE[batch_idx, head_idx, None]
                else:
                    mLSE_cur = cute.domain_offset((seqlen.offset,), mLSE[head_idx, None])
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_limit:
                    lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = None
            if const_expr(self.check_hdim_v_oob):
                tOpO = copy_utils.predicate_k(tOcO, limit=headdim_v)
            # Each copy will use the same predicate
            copy = partial(copy_utils.copy, pred=tOpO)

            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)
            if const_expr(self.check_hdim_v_oob):
                tOrO.fill(0.0)
                tOrdO.fill(0.0)
            assert tOgO.shape == tOgdO.shape
            for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                # Instead of using tOcO, we using t0OcO and subtract the offset from the limit.
                # This is bc the entries of t0OcO are known at compile time.
                if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                    copy(tOgO[None, m, None], tOrO[None, m, None])
                    copy(tOgdO[None, m, None], tOrdO[None, m, None])
            # Sum across the "k" dimension
            pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
            )
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            pdpsum = utils.warp_reduce(pdpsum, operator.add, width=threads_per_row)
            PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            PdP_sum.store(pdpsum)

            # Write PdPsum from rmem -> gmem
            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            # Only the thread corresponding to column 0 writes out the PdPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    gPdPsum[row] = PdP_sum[m] if row < seqlen_limit else 0.0

            # Clear dQaccum
            if const_expr(mdQaccum is not None):
                if const_expr(not seqlen.has_cu_seqlens):
                    mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
                else:
                    mdQaccum_cur = cute.domain_offset(
                        (self.head_dim_padded * offset_padded,), mdQaccum[head_idx, None]
                    )
                blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens):
                    mLSElog2_cur = mLSElog2[batch_idx, head_idx, None]
                else:
                    mLSElog2_cur = cute.domain_offset((offset_padded,), mLSElog2[head_idx, None])
                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.tile_m:
                    gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0
