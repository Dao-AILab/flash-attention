# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_postprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
from typing import Callable, Optional, Type, Literal

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass import Float32, const_expr
from cutlass.utils import LayoutEnum

from flash_attn.cute import utils
from flash_attn.cute import copy_utils
from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import hopper_helpers as sm90_utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from flash_attn.cute.tile_scheduler import (
    ParamsBase,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FlashAttentionBackwardPostprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        arch: Literal[80, 90],
        tile_m: int = 128,
        num_threads: int = 256,
        AtomLayoutMdQ: int = 1,
        dQ_swapAB: bool = False,
    ):
        """
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        """
        self.dtype = dtype
        self.tile_m = tile_m
        assert arch in [80, 90], "Only Ampere (80) and Hopper (90) are supported"
        self.arch = arch
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.num_threads = num_threads
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.dQ_swapAB = dQ_swapAB

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        return True

    def _get_tiled_mma(self):
        if const_expr(self.arch == 80):
            num_mma_warps = self.num_threads // 32
            atom_layout_dQ = (
                (self.AtomLayoutMdQ, num_mma_warps // self.AtomLayoutMdQ, 1)
                if const_expr(not self.dQ_swapAB)
                else (num_mma_warps // self.AtomLayoutMdQ, self.AtomLayoutMdQ, 1)
            )
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                atom_layout_dQ,
                permutation_mnk=(atom_layout_dQ[0] * 16, atom_layout_dQ[1] * 16, 16),
            )
        else:
            num_mma_warp_groups = self.num_threads // 128
            atom_layout_dQ = (self.AtomLayoutMdQ, num_mma_warp_groups // self.AtomLayoutMdQ)
            tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
            tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.K,  # These don't matter, we only care about the accum
                warpgroup.OperandMajorMode.K,
                Float32,
                atom_layout_mnk=(atom_layout_dQ if not self.dQ_swapAB else atom_layout_dQ[::-1])
                + (1,),
                tiler_mn=tiler_mn_dQ if not self.dQ_swapAB else tiler_mn_dQ[::-1],
            )
        assert self.num_threads == tiled_mma.size
        return tiled_mma

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        # We don't do bound checking for the gmem -> smem load so we just assert here.
        assert (self.tile_m * self.tile_hdim // async_copy_elems_accum) % self.num_threads == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        num_s2r_copy_elems = 1 if const_expr(self.arch == 80) else 4
        if const_expr(self.arch == 80):
            self.s2r_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
                Float32, self.num_threads, num_s2r_copy_elems
            )
        else:
            num_threads_per_warp_group = 128
            num_mma_warp_groups = self.num_threads // 128
            self.s2r_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                cute.make_layout((num_threads_per_warp_group, num_mma_warp_groups)),  # thr_layout
                cute.make_layout(128 // Float32.width),  # val_layout
            )

        self.gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(
            self.dtype, self.tile_hdim, self.num_threads
        )
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: dQaccum / dQ
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.arch == 80):
            self.sdQaccum_layout = cute.make_layout(self.tile_m * self.tile_hdim)
        else:
            num_mma_warp_groups = self.num_threads // 128
            self.sdQaccum_layout = cute.make_layout(
                (self.tile_m * self.tile_hdim // num_mma_warp_groups, num_mma_warp_groups)
            )
        # We can't just use kHeadDim here. E.g. if MMA shape is 64 x 96 but split across 2 WGs,
        # then setting kBlockKSmem to 32 will cause "Static shape_div failure".
        # We want to treat it as 64 x 48, so kBlockKSmem should be 16.
        mma_shape_n = self.tiled_mma.get_tile_size(1)
        if const_expr(self.arch == 80):
            sdQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, mma_shape_n)
            self.sdQ_layout = cute.tile_to_shape(
                sdQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1)
            )
        else:
            self.sdQ_layout = sm90_utils.make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim)
            )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mdQaccum, mdQ = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mdQaccum, mdQ)
        ]

        self.tiled_mma = self._get_tiled_mma()
        self._setup_attributes()

        smem_size = max(
            cute.size_in_bytes(cutlass.Float32, self.sdQaccum_layout),
            cute.size_in_bytes(self.dtype, self.sdQ_layout),
        )

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_head = mdQ.shape[1]
            num_batch = mCuSeqlensQ.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_head = mdQ.shape[2]
            num_batch = mdQ.shape[0]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mdQ.shape[1], self.tile_m),
            num_head=num_head,
            num_batch=num_batch,
            seqlen_k=0,
            headdim=mdQ.shape[2],
            headdim_v=0,
            total_q=mdQ.shape[0],
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # grid_dim: (m_block, num_head, batch_size)
        self.kernel(
            mdQaccum,
            mdQ,
            mCuSeqlensQ,
            mSeqUsedQ,
            scale,
            self.tiled_mma,
            self.dQ_swapAB,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.tiled_mma.size, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        dQ_swapAB: cutlass.Constexpr,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
        sdQaccum_flat = cute.make_tensor(sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum)))
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout)
        sdQt = utils.transpose_view(sdQ)

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
                mdQ.shape[1],
                0,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=None,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=None,
            )
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQ_cur = mdQ[batch_size, None, num_head, None]
                mdQaccum_cur = mdQaccum[batch_size, num_head, None]
                head_dim = mdQ.shape[3]
            else:
                padded_offset_q = seqlen.offset_q + batch_size * self.tile_m
                mdQ_cur = cute.domain_offset((seqlen.offset_q, 0), mdQ[None, num_head, None])
                mdQaccum_cur = cute.domain_offset(
                    (padded_offset_q * self.tile_hdim,), mdQaccum[num_head, None]
                )
                head_dim = mdQ.shape[2]

                # HACK: Compiler doesn't seem to recognize that padding
                # by padded_offset_q * self.tile_hdim keeps alignment
                # since statically divisible by 4

                mdQaccum_cur_ptr = cute.make_ptr(
                    dtype=mdQaccum_cur.element_type,
                    value=mdQaccum_cur.iterator.toint(),
                    mem_space=mdQaccum_cur.iterator.memspace,
                    assumed_align=mdQaccum.iterator.alignment,
                )
                mdQaccum_cur = cute.make_tensor(mdQaccum_cur_ptr, mdQaccum_cur.layout)

            gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))
            gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))

            seqlen_q = seqlen.seqlen_q
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)

            # Step 1: load dQaccum from gmem to smem
            g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
            tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum_flat)
            cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: load dQ from smem to rmem
            s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
            tile_shape = (self.tile_m, self.tile_hdim)
            acc_shape = tiled_mma.partition_shape_C(
                tile_shape if const_expr(not dQ_swapAB) else tile_shape[::-1]
            )
            acc = cute.make_fragment(acc_shape, cutlass.Float32)
            assert cute.size(acc) == cute.size(tdQsdQaccum)
            tdQrdQaccum = cute.make_tensor(acc.iterator, cute.make_layout(tdQsdQaccum.shape))
            cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
            # Convert tdQrdQaccum from fp32 to fp16/bf16
            rdQ = cute.make_fragment_like(acc, self.dtype)
            rdQ.store((acc.load() * scale).to(self.dtype))

            # Step 3: Copy dQ from register to smem
            cute.arch.barrier()  # make sure all threads have finished loading dQaccum
            smem_copy_atom_dQ = utils.get_smem_store_atom(
                self.arch, self.dtype, transpose=self.dQ_swapAB
            )
            smem_thr_copy_dQ = cute.make_tiled_copy_C(smem_copy_atom_dQ, tiled_mma).get_slice(tidx)
            taccdQrdQ = smem_thr_copy_dQ.retile(rdQ)
            taccdQsdQ = smem_thr_copy_dQ.partition_D(
                sdQ if const_expr(not self.dQ_swapAB) else sdQt
            )
            cute.copy(smem_copy_atom_dQ, taccdQrdQ, taccdQsdQ)

            # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
            gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
            tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
            tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
            tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
            cute.arch.barrier()  # make sure all smem stores are done
            # TODO: check OOB when reading from smem if kBlockM isn't evenly tiled
            cute.autovec_copy(tdQsdQ, tdQrdQ)

            # Step 5: Copy dQ from register to gmem
            cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
            tdQpdQ = utils.predicate_k(tdQcdQ, limit=head_dim)
            for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
                if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
                    cute.copy(
                        gmem_tiled_copy_dQ,
                        tdQrdQ[None, rest_m, None],
                        tdQgdQ[None, rest_m, None],
                        pred=tdQpdQ[None, rest_m, None],
                    )


class FlashAttentionBackwardPostprocess_sm100(FlashAttentionBackwardPostprocess):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 128,
        num_threads: int = 256,
        AtomLayoutMdQ: int = 1,
        dQ_swapAB: bool = False,
    ):
        super().__init__(
            dtype=dtype,
            head_dim=head_dim,
            arch=90,  # tmp dummy placement for now
            tile_m=tile_m,
            num_threads=num_threads,
            AtomLayoutMdQ=AtomLayoutMdQ,
            dQ_swapAB=dQ_swapAB,
        )

    def _setup_attributes(self):
        self.num_stages = self.tile_hdim // 32  # 2 for D=64, 4 for D=128

        self.sdQaccum_layout = cute.make_layout(
            shape=(self.tile_m * 32, 2), stride=(1, self.tile_m * 32)
        )
        self.epi_tile_q = (self.tile_m, self.tile_hdim)
        self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            self.epi_tile_q,
            1,
        )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mdQaccum, mdQ = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mdQaccum, mdQ)
        ]
        # (b, h, s*d) -> (s*d, h, b)
        mdQaccum = cute.make_tensor(mdQaccum.iterator, cute.select(mdQaccum.layout, mode=[2, 1, 0]))
        # (b, s, h, d) -> (s, d, h, b)
        mdQ = cute.make_tensor(mdQ.iterator, cute.select(mdQ.layout, mode=[1, 3, 2, 0]))

        self._setup_attributes()

        grid_dim = [
            cute.ceil_div(mdQ.shape[0], self.tile_m),
            cute.size(mdQ.shape[2]),
            cute.size(mdQ.shape[3]),
        ]

        cta_group = tcgen05.CtaGroup.ONE
        self.mma_tiler_dsk = (self.tile_m, self.tile_hdim)

        dS_major_mode = tcgen05.OperandMajorMode.MN
        kt_major_mode_dsq = tcgen05.OperandMajorMode.MN

        tiled_mma_dsk = sm100_utils_basic.make_trivial_tiled_mma(
            cutlass.BFloat16,
            dS_major_mode,
            kt_major_mode_dsq,
            cutlass.Float32,
            cta_group,
            self.mma_tiler_dsk,
        )

        dQ_cta_v_layout = cute.composition(cute.make_identity_layout(mdQ.shape), self.mma_tiler_dsk)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_dQ, tma_tensor_dQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdQ,
            cute.select(self.sdQ_layout, mode=[0, 1]),
            dQ_cta_v_layout,
        )

        buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.sdQaccum_layout)],
                128,
            ]

            sdQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sdQ_layout)],
                buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            mdQaccum,
            tma_tensor_dQ,
            tma_atom_dQ,
            self.sdQaccum_layout,
            self.sdQ_layout,
            tiled_mma_dsk,
            scale,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        tiled_mma_dsk: cute.TiledMma,
        scale: cutlass.Float32,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        # SMEM
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        swz128 = cute.make_swizzle(3, 4, 3)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout, swizzle=swz128)

        sdQ = storage.sdQ.get_tensor(sdQ_layout.outer, swizzle=sdQ_layout.inner)

        mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
        mdQ_cur = mdQ[None, None, head_idx, batch_idx]

        thr_mma_dsk = tiled_mma_dsk.get_slice(tidx)
        dQacc_shape = thr_mma_dsk.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dsk.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator, tdQtdQ.layout)

        tmem_ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32
        )
        tiled_tmem_ld = tcgen05.make_tmem_copy(tmem_ld_atom, tdQtdQ)
        thr_tmem_ld = tiled_tmem_ld.get_slice(tidx)

        cdQ = cute.make_identity_tensor((self.mma_tiler_dsk[0], self.mma_tiler_dsk[1]))
        tdQcdQ = thr_mma_dsk.partition_C(cdQ)
        tdQcdQ_tensor = cute.make_tensor(tdQcdQ.iterator, tdQcdQ.layout)
        tdQrdQ = thr_tmem_ld.partition_D(tdQcdQ_tensor)

        gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))

        num_reduce_warps = 4
        num_reduce_threads = cute.arch.WARP_SIZE * num_reduce_warps

        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )
        tiler_mn, layout_tv = cute.make_layout_tv(
            thr_layout=cute.make_layout(shape=num_reduce_threads, stride=1),
            val_layout=cute.make_layout(shape=4, stride=1),
        )
        G2S_tiled_copy_dQaccum = cute.make_tiled_copy(
            atom_universal_copy, layout_tv=layout_tv, tiler_mn=tiler_mn
        )

        smem_thr_copy_g2s = G2S_tiled_copy_dQaccum.get_slice(tidx)

        # S->R
        tdQrdQ_t2r = cute.make_fragment(tdQrdQ.shape, cutlass.Float32)
        tiled_smem_store_s2r = cute.make_tiled_copy(
            atom_universal_copy, layout_tv=layout_tv, tiler_mn=tiler_mn
        )

        s2r_thr_copy_dQaccum = tiled_smem_store_s2r.get_slice(tidx)
        tdQsdQ_s2r = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
        tdQrdQ_s2r = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_t2r.shape)

        # R->S
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.dtype, cutlass.Float32, tiled_tmem_ld
        )
        tiled_smem_store_r2s = cute.make_tiled_copy(
            smem_copy_atom,
            layout_tv=tiled_tmem_ld.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld.tiler_mn,
        )
        tdQsdQ_r2s = thr_tmem_ld.partition_D(thr_mma_dsk.partition_C(sdQ))
        tdQrdQ_r2s = cute.make_fragment(tdQsdQ_r2s.shape, self.dtype)

        num_stages = cute.size(tdQrdQ_t2r, mode=[1])
        for stage in cutlass.range_constexpr(num_stages):
            # G->S
            gdQaccum_stage = cute.local_tile(
                gdQaccum,
                (self.tile_m * 32,),
                (stage,),
            )

            gdQaccum_layout_g2s = cute.make_layout(shape=(self.tile_m * 32, 1), stride=(1, 0))
            gdQaccum_stage_g2s = cute.make_tensor(
                cute.recast_ptr(gdQaccum_stage.iterator, swizzle_=swz128), gdQaccum_layout_g2s
            )

            tdQgdQ = smem_thr_copy_g2s.partition_S(gdQaccum_stage_g2s)
            tdQsdQ = smem_thr_copy_g2s.partition_D(sdQaccum)

            cute.copy(smem_thr_copy_g2s, tdQgdQ[None, None, 0], tdQsdQ[None, None, 0])

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(barrier_id=6, number_of_threads=num_reduce_threads)

            # S -> R
            tdQrdQ_s2r_cpy = tdQrdQ_s2r[None, stage, None, None]
            tdQsdQ_s2r_p = tdQsdQ_s2r[None, None, 0]
            tdQrdQ_r2s_cpy = cute.make_tensor(
                tdQrdQ_s2r_cpy.iterator, cute.make_layout(tdQsdQ_s2r_p.shape)
            )

            cute.copy(s2r_thr_copy_dQaccum, tdQsdQ_s2r_p, tdQrdQ_r2s_cpy)

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(barrier_id=7, number_of_threads=num_reduce_threads)

            # R->S
            tdQrdQ_r2s_cpy = cute.make_tensor(
                cute.recast_ptr(tdQrdQ_r2s_cpy.iterator),
                tdQrdQ_r2s[((None, 0), stage, 0, 0, 0)].shape,
            )
            dQ_vec = tdQrdQ_r2s_cpy.load() * scale
            tdQrdQ_r2s[((None, 0), stage, 0, 0, 0)].store(dQ_vec.to(self.dtype))

        cute.copy(
            tiled_smem_store_r2s,
            tdQrdQ_r2s[None, None, None, None, 0],
            tdQsdQ_r2s[None, None, None, None, 0],
        )
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )
        cute.arch.barrier(barrier_id=8, number_of_threads=num_reduce_threads)

        # S-> G
        gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
        tdQsdQ, tdQgdQ = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ, 0, 2),
            cute.group_modes(gdQ, 0, 2),
        )

        cute.copy(tma_atom_dQ, tdQsdQ[None, 0], tdQgdQ[None, m_block])
