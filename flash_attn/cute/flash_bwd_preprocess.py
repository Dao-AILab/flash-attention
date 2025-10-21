# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_preprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
import operator
from typing import Callable, Type, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from flash_attn.cute import utils
from flash_attn.cute import copy_utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tile_scheduler import (
    ParamsBase,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FlashAttentionBackwardPreprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        m_block_size: int = 128,
        num_threads: int = 128,
    ):
        """
        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        """
        self.dtype = dtype
        self.m_block_size = m_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.num_threads = num_threads

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
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
        if num_threads < m_block_size:  # For multiplying lse with log2
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
            if self.head_dim_padded % 128 == 0
            else (
                64
                if self.head_dim_padded % 64 == 0
                else (32 if self.head_dim_padded % 32 == 0 else 16)
            )
        )
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, gmem_k_block_size, self.num_threads
        )
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // Float32.width
        assert (
            self.m_block_size * self.head_dim_padded // num_copy_elems_dQaccum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            Float32, self.num_threads, num_copy_elems_dQaccum
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mdPsum.element_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if cutlass.const_expr(mdQaccum is not None):
            if cutlass.const_expr(mdQaccum.element_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if cutlass.const_expr(mLSE is not None):
            assert mLSElog2 is not None, "If mLSE is provided, mLSElog2 must also be provided"
            if cutlass.const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
            if cutlass.const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mO, mdO, mdQaccum = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mO, mdO, mdQaccum)
        ]

        self._setup_attributes()

        if cutlass.const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_head = mO.shape[1]
            num_batch = mCuSeqlensQ.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_head = mO.shape[2]
            num_batch = mO.shape[0]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[1], self.m_block_size),
            num_head=num_head,
            num_batch=num_batch,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[2],
            total_q=mO.shape[0],
            tile_shape_mn=(self.m_block_size, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mO,
            mdO,
            mdPsum,
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
        mdPsum: cute.Tensor,
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
        m_block, num_head, batch_size = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen = SeqlenInfoQK(
                batch_size,
                mO.shape[1],
                0,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=None,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=None,
            )

            if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                mO_cur = mO[batch_size, None, num_head, None]
                mdO_cur = mdO[batch_size, None, num_head, None]
                mdPsum_cur = mdPsum[batch_size, num_head, None]
                headdim_v = mO.shape[3]
            else:
                mO_cur = cute.domain_offset((seqlen.offset_q, 0), mO[None, num_head, None])
                mdO_cur = cute.domain_offset((seqlen.offset_q, 0), mdO[None, num_head, None])

                padded_offset_q = seqlen.offset_q + batch_size * self.m_block_size
                mdPsum_cur = cute.domain_offset((padded_offset_q,), mdPsum[num_head, None])
                headdim_v = mO.shape[2]

            blkOdO_shape = (self.m_block_size, self.head_dim_padded)
            # (m_block_size, head_dim)
            gO = cute.local_tile(mO_cur, blkOdO_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blkOdO_shape, (m_block, 0))

            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////
            # Construct identity layout for KV
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = utils.predicate_k(tOcO, limit=headdim_v)
            tOpdO = utils.predicate_k(tOcO, limit=headdim_v)

            seqlen_q = seqlen.seqlen_q
            seqlen_q_rounded = cute.round_up(seqlen_q, self.m_block_size)

            if cutlass.const_expr(mLSE is not None):
                if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[batch_size, num_head, None]
                else:
                    mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[num_head, None])

                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_q - m_block * self.m_block_size:
                    lse = gLSE[tidx]

            tOrO = cute.make_fragment_like(tOgO)
            tOrdO = cute.make_fragment_like(tOgdO)
            assert cute.size(tOgO, mode=[0]) == cute.size(tOgdO, mode=[0])
            assert cute.size(tOgO, mode=[1]) == cute.size(tOgdO, mode=[1])
            assert cute.size(tOgO, mode=[2]) == cute.size(tOgdO, mode=[2])
            for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                # Instead of using tOcO, we using t0OcO and subtract the offset from the limit
                # (seqlen_q - m_block * kBlockM). This is because the entries of t0OcO are known at compile time.
                if t0OcO[0, m, 0][0] < seqlen_q - m_block * self.m_block_size - tOcO[0][0]:
                    cute.copy(
                        gmem_thr_copy_O,
                        tOgO[None, m, None],
                        tOrO[None, m, None],
                        pred=tOpO[None, m, None]
                        if cutlass.const_expr(self.check_hdim_oob)
                        else None,
                    )
                    cute.copy(
                        gmem_thr_copy_O,
                        tOgdO[None, m, None],
                        tOrdO[None, m, None],
                        pred=tOpdO[None, m, None]
                        if cutlass.const_expr(self.check_hdim_oob)
                        else None,
                    )
            # Sum across the "k" dimension
            dpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
            )
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            dpsum = utils.warp_reduce(dpsum, operator.add, width=threads_per_row)
            dP_sum = cute.make_fragment(cute.size(tOrO, mode=[1]), Float32)
            dP_sum.store(dpsum)

            # Write dPsum from rmem -> gmem
            gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (m_block,))
            # Only the thread corresponding to column 0 writes out the dPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(dP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    gdPsum[row] = dP_sum[m] if row < seqlen_q - m_block * self.m_block_size else 0.0

            # Clear dQaccum
            if cutlass.const_expr(mdQaccum is not None):
                if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                    mdQaccum_cur = mdQaccum[batch_size, num_head, None]
                else:
                    padded_offset_q = seqlen.offset_q + batch_size * self.m_block_size
                    mdQaccum_cur = cute.domain_offset(
                        (padded_offset_q * self.head_dim_padded,), mdQaccum[num_head, None]
                    )

                    # HACK: Compiler doesn't seem to recognize that padding
                    # by padded_offset_q * self.head_dim_padded keeps alignment
                    # since statically divisible by 4

                    mdQaccum_cur_ptr = cute.make_ptr(
                        dtype=mdQaccum_cur.element_type,
                        value=mdQaccum_cur.iterator.toint(),
                        mem_space=mdQaccum_cur.iterator.memspace,
                        assumed_align=mdQaccum.iterator.alignment,
                    )
                    mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)

                blkdQaccum_shape = (self.m_block_size * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tQgQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_fragment_like(tQgQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tQgQaccum)

            if cutlass.const_expr(mLSE is not None):
                if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                    mLSElog2_cur = mLSElog2[batch_size, num_head, None]
                else:
                    padded_offset_q = seqlen.offset_q + batch_size * self.m_block_size
                    mLSElog2_cur = cute.domain_offset((padded_offset_q,), mLSElog2[num_head, None])

                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.m_block_size,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.m_block_size:
                    gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0
