from ctypes import alignment
import enum
import math
from typing import Type, Tuple, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
from cutlass._mlir.ir import _si1Attr
from cutlass.base_dsl.jit_executor import t
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05

import cutlass.utils.blackwell_helpers as sm100_utils_basic
import flash_attn.cute.utils as utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfo, SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo

from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, StaticPersistentTileScheduler, ParamsBase
from cutlass.pipeline import PipelineAsync

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from cutlass._mlir.dialects import nvvm

from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwdSm100


@dsl_user_op
def tma_reduce_add_bulk_f32(
        smem_ptr: cute.Pointer,
        gmem_ptr: cute.Pointer,
        store_bytes: cutlass.Int32,
        *, loc=None, ip=None
    ):
    cute.make_mma_atom
    smem_u32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_u32, store_bytes.ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionBackwardSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
    ):

        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        assert head_dim == head_dim_v, "head_dim and head_dim_v must be the same for now"
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        assert self.head_dim_padded == self.head_dim_v_padded, "head_dim_padded and head_dim_v_padded must be the same for now"
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded

        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        # number of tma reduce adds per dQacc mma
        self.dQaccum_reduce_stage = self.head_dim_padded // 32

        # CTA tiler
        self.cta_tiler     = (m_block_size, n_block_size, self.head_dim_padded)

        # S = K @ Q.T
        self.mma_tiler_kq  = (n_block_size, m_block_size, self.head_dim_padded)

        # dP = V @ dO.T
        self.mma_tiler_vdo = (n_block_size, m_block_size, self.head_dim_v_padded)

        # dV = P.T @ dO
        self.mma_tiler_pdo = (n_block_size, self.head_dim_v_padded, m_block_size)

        # dK = dS.T @ Q (N, M) (M, D)
        self.mma_tiler_dsq = (n_block_size, self.head_dim_v_padded, m_block_size)

        # dQ = dS @ K
        self.mma_tiler_dsk = (m_block_size, self.head_dim_v_padded, n_block_size)


        self.kq_acc_dtype  = self.vdo_acc_dtype = self.pdo_acc_dtype = self.dsq_acc_dtype =  self.dsk_acc_dtype = Float32

        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = False
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.use_tma_store = True
        self.deterministic = deterministic

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epi_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epi_warp_id,
                self.empty_warp_id,
            )
        )

        # TMEM setup
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.tmem_s_offset       = 0
        self.tmem_p_offset       = 0 # overlap with S
        self.tmem_dV_offset      = self.tmem_s_offset  + self.n_block_size
        self.tmem_dP_offset      = self.tmem_dV_offset + self.head_dim_v_padded
        self.tmem_dQaccum_offset = self.tmem_dP_offset # overlap with dP
        self.tmem_dK_offset      = self.tmem_dP_offset + self.m_block_size

        self.num_regs_reduce = 144
        self.num_regs_compute = 128
        self.num_regs_load = 96
        self.num_regs_mma = 112
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

        self.num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)

    def _setup_attributes(self):

        self.q_stage       = 2
        self.k_stage       = 1
        self.v_stage       = 1
        self.do_stage      = 1
        self.ds_stage      = 1
        self.lse_stage     = 1
        self.acc_stage     = 1
        self.s_stage       = 1
        self.dP_stage      = 1
        self.dV_stage      = 1
        self.dK_stage      = 1
        self.dS_stage      = 1
        self.dQaccum_mma_stage = 1
        self.sdQaccum_stage    = 2
        self.psum_stage        = 1
        self.p_tmem_stage      = 1
        self.sdKdVaccum_stage = 2


    @cute.jit
    def __call__(
        self,
        mQ:       cute.Tensor,
        mK:       cute.Tensor,
        mV:       cute.Tensor,
        mdO:      cute.Tensor,
        mLSE:     cute.Tensor,
        mPsum:    cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK:      cute.Tensor,
        mdV:      cute.Tensor,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
        self.q_dtype  = mQ.element_type
        self.k_dtype  = mK.element_type
        self.v_dtype  = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype  = mLSE.element_type
        self.psum_dtype = mPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        if const_expr(self.qhead_per_kvhead > 1):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        QKVdO_layout_transpose = [1, 3, 2, 0] # (b, s, n, h) --> (s, h, n, b)
        mQ, mK, mV, mdO, mdK, mdV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=QKVdO_layout_transpose))
            for t in (mQ, mK, mV, mdO, mdK, mdV)
        ]

        LSE_Psum_dQaccum_layout_transpose = [2, 1, 0] # (b, n, s) --> (s, n, b)
        mLSE, mPsum, mdQaccum = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=LSE_Psum_dQaccum_layout_transpose))
            for t in (mLSE, mPsum, mdQaccum)
        ]

        dO_transpose  = [1, 0, 2, 3]
        mdO = cute.make_tensor(mdO.iterator, cute.select(mdO.layout, mode=dO_transpose))

        semaphore_transpose = [2, 3, 1, 0] # (b, n, block, stage) -> (block, stage, n, b)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = cute.make_tensor(mdQ_semaphore.iterator, cute.select(mdQ_semaphore.layout, mode=semaphore_transpose))
        else:
            mdQ_semaphore = None

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                cute.make_tensor(t.iterator, cute.select(t.layout, mode=semaphore_transpose))
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self.q_major_mode  =  cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode  =  cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode  =  cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.do_major_mode =  cutlass.utils.LayoutEnum.from_tensor(mdO).mma_major_mode()

        self._setup_attributes()
        cta_group = tcgen05.CtaGroup.ONE

        # S = K @ Q.T
        tiled_mma_kq = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            self.k_major_mode,
            self.q_major_mode,
            self.kq_acc_dtype,
            cta_group,
            self.mma_tiler_kq[:2],
        )

        # dV += P @ dO --> (K, MN) major
        p_source = tcgen05.OperandSource.TMEM
        self.p_major_mode  = tcgen05.OperandMajorMode.K
        tiled_mma_pdo = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.p_major_mode,
            self.do_major_mode,
            self.pdo_acc_dtype,
            cta_group,
            self.mma_tiler_pdo[:2],
            p_source,
        )

        # dP = V @ dO.T
        self.dot_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_vdo = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.v_major_mode,
            self.dot_major_mode,
            self.vdo_acc_dtype,
            cta_group,
            self.mma_tiler_vdo[:2],
        )

        # dK += dS.T @ Q
        self.dSt_major_mode    = tcgen05.OperandMajorMode.K
        self.q_major_mode_dsq  = tcgen05.OperandMajorMode.MN
        tiled_mma_dsq = sm100_utils_basic.make_trivial_tiled_mma(
            self.ds_dtype,
            self.dSt_major_mode,
            self.q_major_mode_dsq,
            self.dsq_acc_dtype,
            cta_group,
            self.mma_tiler_dsq[:2],
        )

        # dQ = dS @ K
        self.dS_major_mode     = tcgen05.OperandMajorMode.MN
        self.kt_major_mode_dsq = tcgen05.OperandMajorMode.MN
        tiled_mma_dsk = sm100_utils_basic.make_trivial_tiled_mma(
            self.ds_dtype,
            self.dS_major_mode,
            self.kt_major_mode_dsq,
            self.dsk_acc_dtype,
            cta_group,
            self.mma_tiler_dsk[:2],
        )
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_kq.thr_id.shape,),
        )

        # S = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_kq, self.mma_tiler_kq, self.k_dtype, self.k_stage,
        )
        sQ_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_kq, self.mma_tiler_kq, self.q_dtype, self.q_stage,
        )

        # dV += P @ dO
        sdO_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pdo, self.mma_tiler_pdo, self.do_dtype, self.do_stage,
        )

        # dP = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_vdo, self.mma_tiler_vdo, self.v_dtype, self.v_stage,
        )

        sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_vdo, self.mma_tiler_vdo, self.do_dtype, self.do_stage,
        )

        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_dsq, self.mma_tiler_dsq, self.ds_dtype, self.ds_stage,
        )

        sQt_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_dsq, self.mma_tiler_dsq, self.q_dtype, self.q_stage,
        )

        # dQaccum = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_dsk, self.mma_tiler_dsk, self.q_dtype, self.ds_stage,
        )
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_dsk, self.mma_tiler_dsk, self.k_dtype, self.k_stage,
        )

        sdQaccum_layout = cute.make_layout(shape=(self.m_block_size * 32, self.sdQaccum_stage ),)
        sLSE_layout  = cute.make_layout(shape=(self.m_block_size, self.lse_stage),  stride=(1, cute.round_up(self.m_block_size, 64)))
        sPsum_layout = cute.make_layout(shape=(self.m_block_size, self.psum_stage), stride=(1, cute.round_up(self.m_block_size, 64)))

        self.mdK_layout_enum = cutlass.utils.LayoutEnum.from_tensor(mdK)
        self.mdV_layout_enum = cutlass.utils.LayoutEnum.from_tensor(mdV)
        self.dK_major_mode = self.mdK_layout_enum.mma_major_mode()
        self.dV_major_mode = self.mdV_layout_enum.mma_major_mode()
        if const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mdK is wrong")
        if const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mdV is wrong")
        self.sdKdV_epi_tile = (self.n_block_size, 128 // (self.dk_dtype.width // 8)) # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        sdKdV_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dk_dtype, self.mdK_layout_enum, self.sdKdV_epi_tile, self.sdKdVaccum_stage,
        )

        self.tma_copy_dKdV_bytes = cute.size_in_bytes(self.dk_dtype, cute.select(sdKdV_layout, mode=[0,1]))

        if const_expr(self.use_tma_store):
            if const_expr(self.dk_dtype.width == 32):
                tma_copy_op_dKdV = cpasync.CopyReduceBulkTensorTileS2GOp()
            else:
                tma_copy_op_dKdV = cpasync.CopyBulkTensorTileS2GOp()

            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKdV,
                mdK,
                cute.select(sdKdV_layout, mode=[0, 1]),
                self.sdKdV_epi_tile,
                1  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKdV,
                mdV,
                cute.select(sdKdV_layout, mode=[0, 1]),
                self.sdKdV_epi_tile,
                1  # no mcast
            )
        else:
            assert self.qhead_per_kvhead == 1, "Must use TMA reduce add for GQA"
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        thr_layout_r2s_dKdV = cute.make_ordered_layout((self.n_block_size, 1), order=(1,0)) # 128 threads
        val_layout_r2s_dKdV = cute.make_ordered_layout((1, 128 // self.dk_dtype.width), order=(1,0)) # 4 or 8 vals for 16 byte store
        r2s_copy_atom_r2s_dKdV = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dk_dtype, num_bits_per_copy=128,)
        tiled_copy_r2s_dKdV = cute.make_tiled_copy_tv(r2s_copy_atom_r2s_dKdV, thr_layout_r2s_dKdV, val_layout_r2s_dKdV)

        tma_load_op  = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        # S = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            tiled_mma_kq,
            self.cluster_layout_vmnk.shape,
        )

        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            tiled_mma_kq,
            self.cluster_layout_vmnk.shape,
        )

        # dV += P @ dO
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mdO,
            cute.select(sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            tiled_mma_pdo,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_LSE, tma_tensor_LSE = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            mLSE,
            cute.make_layout((self.m_block_size)),
            (self.m_block_size, ),
        )
        tma_atom_Psum, tma_tensor_Psum = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            mPsum,
            cute.make_layout((self.m_block_size)),
            (self.m_block_size, ),
        )

        # dP = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            tiled_mma_vdo,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_q_bytes    = cute.size_in_bytes(self.q_dtype,     cute.select(sQ_layout,   mode=[0, 1, 2]))
        self.tma_copy_k_bytes    = cute.size_in_bytes(self.k_dtype,     cute.select(sK_layout,   mode=[0, 1, 2]))
        self.tma_copy_v_bytes    = cute.size_in_bytes(self.v_dtype,     cute.select(sV_layout,   mode=[0, 1, 2]))
        self.tma_copy_do_bytes   = cute.size_in_bytes(self.do_dtype,    cute.select(sdO_layout,  mode=[0, 1, 2]))
        self.tma_copy_lse_bytes  = self.m_block_size * 4
        self.tma_copy_psum_bytes = self.m_block_size * 4

        TileScheduler = SingleTileScheduler
        # TODO -- optimizer scheduler for causal
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]), # num_heads = num_query_heads
            cute.size(mK.shape[3]),
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=False,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        # cute.printf("grid_dim = {}", grid_dim)

        @cute.struct
        class SharedStorage:
            q_mbar_ptr:          cute.struct.MemRange[cutlass.Int64, 2 * self.q_stage]
            k_full_mbar_ptr:     cute.struct.MemRange[cutlass.Int64,     self.k_stage]
            v_full_mbar_ptr:     cute.struct.MemRange[cutlass.Int64,     self.v_stage]
            lse_mbar_ptr:        cute.struct.MemRange[cutlass.Int64, 2 * self.lse_stage]
            do_mbar_ptr:         cute.struct.MemRange[cutlass.Int64, 2 * self.do_stage]
            lse_full_mbar_ptr:   cute.struct.MemRange[cutlass.Int64,  self.k_stage]
            lse_empty_mbar_ptr:  cute.struct.MemRange[cutlass.Int64,  self.k_stage]
            psum_full_mbar_ptr:  cute.struct.MemRange[cutlass.Int64,  self.psum_stage]
            psum_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64,  self.psum_stage]
            s_mbar_ptr:          cute.struct.MemRange[cutlass.Int64, 2 * self.s_stage]
            dP_mbar_ptr:         cute.struct.MemRange[cutlass.Int64, 2 * self.dP_stage]
            p_mbar_ptr:          cute.struct.MemRange[cutlass.Int64, 2 * self.s_stage]
            dS_mbar_ptr:         cute.struct.MemRange[cutlass.Int64, 2 * self.ds_stage]
            dV_mbar_ptr:         cute.struct.MemRange[cutlass.Int64, 2 * self.dV_stage]
            dK_mbar_ptr:         cute.struct.MemRange[cutlass.Int64, 2 * self.dK_stage]
            dQaccum_mbar_ptr:         cute.struct.MemRange[cutlass.Int64, 2 * self.dQaccum_mma_stage]
            dQaccum_reduce_mbar_ptr:  cute.struct.MemRange[cutlass.Int64, 2 * self.dQaccum_mma_stage]

            # TMEM
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr:  cute.struct.MemRange[cutlass.Int64, 1]

            # Smem tensors
            sQ:  cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                    self.buffer_align_bytes,
            ]
            sK:  cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                    self.buffer_align_bytes,
            ]
            sV:  cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)],
                    self.buffer_align_bytes,
            ]
            sdO:  cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(sdO_layout)],
                    self.buffer_align_bytes,
            ]
            sdS:  cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(sdSt_layout)],
                    128,
            ]
            sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(sLSE_layout)],
                    128,
            ]
            sPsum: cute.struct.Align[
                    cute.struct.MemRange[self.psum_dtype, cute.cosize(sPsum_layout)],
                    128,
            ]
            sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(sdQaccum_layout)],
                    self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage


        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_LSE,
            tma_tensor_Psum,
            tma_tensor_dO,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_LSE,
            tma_atom_Psum,
            tma_atom_dO,
            tma_atom_dV,
            tma_atom_dK,
            sQ_layout,
            sQt_layout,
            sK_layout,
            sV_layout,
            sLSE_layout,
            sPsum_layout,
            sdO_layout,
            sdOt_layout,
            sdSt_layout,
            sdS_layout,
            sKt_layout,
            sdQaccum_layout,
            sdKdV_layout,
            tiled_mma_kq,
            tiled_mma_pdo,
            tiled_mma_vdo,
            tiled_mma_dsq,
            tiled_mma_dsk,
            tiled_copy_r2s_dKdV,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )


    @cute.kernel
    def kernel(
        self,
        mQ:    cute.Tensor,
        mK:    cute.Tensor,
        mV:    cute.Tensor,
        mLSE:  cute.Tensor,
        mPsum: cute.Tensor,
        mdO:   cute.Tensor,
        mdV:   cute.Tensor,
        mdK:   cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        tma_atom_Q:    cute.CopyAtom,
        tma_atom_K:    cute.CopyAtom,
        tma_atom_V:    cute.CopyAtom,
        tma_atom_LSE:  cute.CopyAtom,
        tma_atom_Psum: cute.CopyAtom,
        tma_atom_dO:   cute.CopyAtom,
        tma_atom_dV:   Optional[cute.CopyAtom],
        tma_atom_dK:   Optional[cute.CopyAtom],
        sQ_layout:     cute.ComposedLayout,
        sQt_layout:    cute.ComposedLayout,
        sK_layout:     cute.ComposedLayout,
        sV_layout:     cute.ComposedLayout,
        sLSE_layout:   cute.Layout,
        sPsum_layout:  cute.Layout,
        sdO_layout:    cute.ComposedLayout,
        sdOt_layout:   cute.ComposedLayout,
        sdSt_layout:   cute.ComposedLayout,
        sdS_layout:    cute.ComposedLayout,
        sKt_layout:    cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdKdV_layout:       cute.ComposedLayout,
        tiled_mma_kq:       cute.TiledMma,
        tiled_mma_pdo:      cute.TiledMma,
        tiled_mma_vdo:      cute.TiledMma,
        tiled_mma_dsq:      cute.TiledMma,
        tiled_mma_dsk:      cute.TiledMma,
        tiled_copy_r2s_dKdV: cute.TiledCopy,
        softmax_scale:      cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        tile_sched_params: ParamsBase,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_LSE)
                cpasync.prefetch_descriptor(tma_atom_Psum)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        # Alloc
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        k_full_mbar_ptr       = storage.k_full_mbar_ptr.data_ptr()
        v_full_mbar_ptr       = storage.v_full_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        lse_full_mbar_ptr     = storage.lse_full_mbar_ptr.data_ptr()
        lse_empty_mbar_ptr    = storage.lse_empty_mbar_ptr.data_ptr()
        psum_full_mbar_ptr    = storage.psum_full_mbar_ptr.data_ptr()
        psum_empty_mbar_ptr   = storage.psum_empty_mbar_ptr.data_ptr()
        dQaccum_reduce_mbar_ptr  = storage.dQaccum_reduce_mbar_ptr.data_ptr()

        if warp_idx == self.load_warp_id:
            cute.arch.mbarrier_init(k_full_mbar_ptr,        len([self.load_warp_id]))
            cute.arch.mbarrier_init(v_full_mbar_ptr,        len([self.load_warp_id]))
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr,  cute.arch.WARP_SIZE * len(self.compute_warp_ids))
            cute.arch.mbarrier_init(lse_full_mbar_ptr,      len([self.compute_warp_ids]))
            cute.arch.mbarrier_init(lse_empty_mbar_ptr,     len([self.compute_warp_ids]))
            cute.arch.mbarrier_init(psum_full_mbar_ptr,     len([self.compute_warp_ids]))
            cute.arch.mbarrier_init(psum_empty_mbar_ptr,    len([self.compute_warp_ids]))
            cute.arch.mbarrier_init(dQaccum_reduce_mbar_ptr, 1)

        pipeline_producer_group      = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread,  len([self.load_warp_id]))
        pipeline_consumer_group      = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread,  len([self.mma_warp_id]))

        pipeline_q = cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_q_bytes,
        )

        pipeline_do = cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar_ptr.data_ptr(),
            num_stages=self.do_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_do_bytes,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread,                        len([self.mma_warp_id]))
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread,  cute.arch.WARP_SIZE * len(self.compute_warp_ids))

        pipeline_s = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.s_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.s_mbar_ptr.data_ptr(),
        )
        pipeline_dV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.dV_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dV_mbar_ptr.data_ptr(),
        )
        pipeline_dK = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.dK_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dK_mbar_ptr.data_ptr(),
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread,  cute.arch.WARP_SIZE * len(self.reduce_warp_ids), alignment=128) # Compute
        pipeline_dQaccum = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.dQaccum_mma_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQaccum_mbar_ptr.data_ptr(),
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.dP_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
        )

        # AsyncThread producers and UMMA consumers
        pipeline_pdS_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread,  cute.arch.WARP_SIZE * len(self.compute_warp_ids)) # Compute
        pipeline_pdS_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread,                        len([self.mma_warp_id]))    # MMA

        pipeline_p = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=self.s_stage,
            producer_group=pipeline_pdS_producer_group,
            consumer_group=pipeline_pdS_consumer_group,
            barrier_storage=storage.p_mbar_ptr.data_ptr(),
        )

        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=self.dS_stage,
            producer_group=pipeline_pdS_producer_group,
            consumer_group=pipeline_pdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
        )

        sQ  = storage.sQ.get_tensor(sQ_layout.outer,        swizzle=sQ_layout.inner)
        sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, swizzle_=sQt_layout.inner), sQt_layout.outer)
        sQ_pi = storage.sQ.get_tensor(sQ_layout)

        sK   = storage.sK.get_tensor(sK_layout.outer,        swizzle=sK_layout.inner)
        sKt  = cute.make_tensor(cute.recast_ptr(sK.iterator, swizzle_=sKt_layout.inner), sKt_layout.outer)

        sV   = storage.sV.get_tensor(sV_layout.outer,        swizzle=sV_layout.inner)

        sdSt    = storage.sdS.get_tensor(sdSt_layout.outer,       swizzle=sdSt_layout.inner)
        sdSt_pi = storage.sdS.get_tensor(sdSt_layout)

        sdS  = cute.make_tensor(cute.recast_ptr(sdSt.iterator, swizzle_=sdS_layout.inner), sdS_layout.outer)

        sdO  = storage.sdO.get_tensor(sdO_layout.outer,  swizzle=sdO_layout.inner)
        sdOt = cute.make_tensor(cute.recast_ptr(sdO.iterator, swizzle_=sdOt_layout.inner), sdOt_layout.outer)

        sLSE_load = storage.sLSE.get_tensor(sLSE_layout)
        sLSE_mma  = storage.sLSE.get_tensor(cute.make_layout(
                                            shape=(self.m_block_size, self.n_block_size, self.lse_stage),
                                            stride=(0, 1, 0)
                                            ))


        sPsum_load = storage.sPsum.get_tensor(sPsum_layout)
        sPsum_mma  = storage.sPsum.get_tensor(cute.make_layout(
                                            shape=(self.m_block_size, self.n_block_size, self.psum_stage),
                                            stride=(0, 1, 0)
                                            ))

        sdV = storage.sdO.get_tensor(sdKdV_layout.outer, swizzle=sdKdV_layout.inner, dtype=self.dk_dtype)
        sdK = storage.sQ.get_tensor(sdKdV_layout.outer, swizzle=sdKdV_layout.inner,  dtype=self.dk_dtype)

        assert cute.cosize(sdV) * self.dv_dtype.width <= cute.cosize(sdO) * self.do_dtype.width, "Not enough space for sdV"
        assert cute.cosize(sdK) * self.dk_dtype.width <= cute.cosize(sQ)  * self.q_dtype.width,  "Not enough space for sdK"

        swz128 = cute.make_swizzle(3, 4, 3)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout, swizzle=swz128)

        # TMEM
        # S
        thr_mma_kq      = tiled_mma_kq.get_slice(0)
        Sacc_shape      = thr_mma_kq.partition_shape_C(self.mma_tiler_kq[:2]) #(M, N)
        tStS            = thr_mma_kq.make_fragment_C(Sacc_shape)
        tStS            = cute.make_tensor(tStS.iterator, tStS.layout)

        # dV
        thr_mma_pdo = tiled_mma_pdo.get_slice(0)
        dvacc_shape = thr_mma_pdo.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV      = thr_mma_pdo.make_fragment_C(dvacc_shape)
        tdVtdV      = cute.make_tensor(tdVtdV.iterator + self.tmem_dV_offset , tdVtdV.layout)

        # dK
        thr_mma_dsq = tiled_mma_dsq.get_slice(0)
        dkacc_shape = thr_mma_dsq.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK      = thr_mma_dsq.make_fragment_C(dkacc_shape)
        tdKtdK      = cute.make_tensor(tdKtdK.iterator + self.tmem_dK_offset , tdKtdK.layout)

        # dQ
        thr_mma_dsk = tiled_mma_dsk.get_slice(0)
        dQacc_shape = thr_mma_dsk.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ      = thr_mma_dsk.make_fragment_C(dQacc_shape)
        tdQtdQ      = cute.make_tensor(tdQtdQ.iterator + self.tmem_dQaccum_offset , tdQtdQ.layout)

        # dP
        thr_mma_vdo = tiled_mma_vdo.get_slice(0)
        dPacc_shape = thr_mma_vdo.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP      = thr_mma_vdo.make_fragment_C(dPacc_shape)
        tdPtdP      = cute.make_tensor(tdPtdP.iterator + self.tmem_dP_offset , tdPtdP.layout)

        block_info = BlockInfo(
            self.m_block_size,
            self.n_block_size,
            self.is_causal, self.is_local,
            None, None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None, mCuSeqlensK=None,
            mSeqUsedQ=None, mSeqUsedK=None,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # TODO: support local
        AttentionMaskCls = partial(
            AttentionMask, self.m_block_size, self.n_block_size,
        )

        cute.arch.sync_threads()

        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        #  EPI
        # (14)
        if warp_idx == self.epi_warp_id:
            # currently no-op, could use for tma store/reduce
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            self.load(
                thr_mma_kq,
                thr_mma_pdo,
                thr_mma_vdo,
                mQ,
                mK,
                mV,
                mLSE,
                mPsum,
                mdO,
                sQ,
                sK,
                sV,
                sLSE_load,
                sPsum_load,
                sdO,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_LSE,
                tma_atom_Psum,
                tma_atom_dO,
                pipeline_q,
                lse_full_mbar_ptr,
                lse_empty_mbar_ptr,
                psum_full_mbar_ptr,
                psum_empty_mbar_ptr,
                pipeline_do,
                k_full_mbar_ptr,
                v_full_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            cute.arch.sync_warp()

            self.mma(
                tiled_mma_kq,
                tiled_mma_pdo,
                tiled_mma_vdo,
                tiled_mma_dsq,
                tiled_mma_dsk,
                thr_mma_kq,
                thr_mma_pdo,
                thr_mma_vdo,
                thr_mma_dsq,
                thr_mma_dsk,
                sQ,
                sQt,
                sK,
                sV,
                sdO,
                sdOt,
                sdSt,
                sdS,
                sKt,
                sK_layout.inner,
                sQ_layout.inner,
                tStS,
                tdVtdV,
                tdKtdK,
                tdPtdP,
                tdQtdQ,
                pipeline_q,
                pipeline_do,
                pipeline_s,
                pipeline_p,
                pipeline_dS,
                pipeline_dV,
                pipeline_dK,
                pipeline_dP,
                pipeline_dQaccum,
                k_full_mbar_ptr,
                v_full_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf)

            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, is_two_cta=False)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_compute) # 8 warps
            self.compute_loop(
                thr_mma_kq,
                thr_mma_pdo,
                thr_mma_vdo,
                thr_mma_dsq,
                tStS,
                sLSE_mma,
                sPsum_mma,
                tdVtdV,
                tdKtdK,
                mdV,
                mdK,
                sdSt,
                sdS,
                tdPtdP,
                lse_full_mbar_ptr,
                lse_empty_mbar_ptr,
                psum_full_mbar_ptr,
                psum_empty_mbar_ptr,
                pipeline_s,
                pipeline_p,
                pipeline_dS,
                pipeline_dV,
                pipeline_dK,
                pipeline_dP,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKdV,
                mdK_semaphore,
                mdV_semaphore,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)

            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dsk,
                tdQtdQ,
                pipeline_dQaccum,
                dQaccum_reduce_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
            )

        return


    @cute.jit
    def load(
        self,
        thr_mma_kq:  cute.core.ThrMma,
        thr_mma_pdo: cute.core.ThrMma,
        thr_mma_vdo: cute.core.ThrMma,
        mQ:   cute.Tensor,
        mK:   cute.Tensor,
        mV:   cute.Tensor,
        mLSE:  cute.Tensor,
        mPsum: cute.Tensor,
        mdO:   cute.Tensor,
        sQ:    cute.Tensor,
        sK:    cute.Tensor,
        sV:    cute.Tensor,
        sLSE:  cute.Tensor,
        sPsum: cute.Tensor,
        sdO:   cute.Tensor,
        tma_atom_Q:    cute.CopyAtom,
        tma_atom_K:    cute.CopyAtom,
        tma_atom_V:    cute.CopyAtom,
        tma_atom_LSE:  cute.CopyAtom,
        tma_atom_Psum: cute.CopyAtom,
        tma_atom_dO:   cute.CopyAtom,
        pipeline_q:    PipelineAsync,
        lse_full_mbar_ptr:   cute.Pointer,
        lse_empty_mbar_ptr:  cute.Pointer,
        psum_full_mbar_ptr:  cute.Pointer,
        psum_empty_mbar_ptr: cute.Pointer,
        pipeline_do:  PipelineAsync,
        k_full_mbar_ptr: cute.Pointer,
        v_full_mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]

        q_producer_state   = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer,  self.q_stage)
        do_producer_state  = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer,  self.do_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx

            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mQ_cur    = mQ[None,  None, head_idx, batch_idx]
            mK_cur    = mK[None,  None, head_idx_kv, batch_idx]
            mV_cur    = mV[None,  None, head_idx_kv, batch_idx]
            mdO_cur   = mdO[None, None, head_idx, batch_idx]
            mLSE_cur  = mLSE[None, head_idx, batch_idx]
            mPsum_cur = mPsum[None, head_idx, batch_idx]

            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
            tSgK = thr_mma_kq.partition_A(gK)

            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
            tdPgV = thr_mma_vdo.partition_A(gV)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_kq.partition_B(gQ)

            gLSE  = cute.local_tile(mLSE_cur,  (self.n_block_size, ), (None, ))
            gPsum = cute.local_tile(mPsum_cur, (self.n_block_size, ), (None, ))

            gdO    = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_pdo.partition_B(gdO)

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tdPgV, 0, 3),
            )
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ, 0, 3),
            )
            tdOsdO, tdOgdO = cpasync.tma_partition(
                tma_atom_dO,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdO, 0, 3),
                cute.group_modes(tdVgdO, 0, 3),
            )
            tLSEsLSE, tLSEgLSE = cpasync.tma_partition(
                tma_atom_LSE,
                0,
                cute.make_layout(1),
                sLSE,
                gLSE,
            )
            tPsumsPsum, tPsumgPsum = cpasync.tma_partition(
                tma_atom_Psum,
                0,
                cute.make_layout(1),
                sPsum,
                gPsum,
            )
            # K
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(k_full_mbar_ptr, self.tma_copy_k_bytes)
            cute.copy(tma_atom_K, tKgK, tKsK[None, 0], tma_bar_ptr=k_full_mbar_ptr)

            ###### Prologue
            # Q0
            pipeline_q.producer_acquire(q_producer_state)
            cute.copy(
                    tma_atom_Q,
                    tQgQ[None, m_block_max - 1],
                    tQsQ[None, q_producer_state.index],
                    tma_bar_ptr=pipeline_q.producer_get_barrier(q_producer_state)
            )
            pipeline_q.producer_commit(q_producer_state)
            q_producer_state.advance()

            # LSE
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(lse_full_mbar_ptr, self.tma_copy_lse_bytes)

            cute.copy(
                tma_atom_LSE,
                tLSEgLSE[None, m_block_max - 1],
                tLSEsLSE[None, 0],
                tma_bar_ptr=lse_full_mbar_ptr,
            )

            # V
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(v_full_mbar_ptr, self.tma_copy_v_bytes)
            cute.copy(tma_atom_V, tVgV, tVsV[None, 0], tma_bar_ptr=v_full_mbar_ptr)

            # dO
            pipeline_do.producer_acquire(do_producer_state)
            cute.copy(
                tma_atom_dO,
                tdOgdO[None, m_block_max - 1],
                tdOsdO[None, do_producer_state.index],
                tma_bar_ptr=pipeline_do.producer_get_barrier(do_producer_state)
            )
            pipeline_do.producer_commit(do_producer_state)
            do_producer_state.advance()

            # Psum
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(psum_full_mbar_ptr, self.tma_copy_psum_bytes)

            cute.copy(
                tma_atom_Psum,
                tPsumgPsum[None, m_block_max - 1],
                tPsumsPsum[None, 0],
                tma_bar_ptr=psum_full_mbar_ptr,
            )
            lse_empty_consumer_phase = cute.Int32(0)
            psum_empty_consumer_phase = cute.Int32(0)

            for i in cutlass.range(m_block_max - m_block_min - 1, unroll=1):
                m_block = m_block_max - 2 - i

                # Q
                self.load_M_tile(tma_atom_Q, tQgQ, tQsQ, pipeline_q, m_block, producer_state=q_producer_state)
                pipeline_q.producer_commit(q_producer_state)
                q_producer_state.advance()

                # LSE
                cute.arch.mbarrier_wait(lse_empty_mbar_ptr, lse_empty_consumer_phase)
                lse_empty_consumer_phase ^= 1

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(lse_full_mbar_ptr, self.tma_copy_lse_bytes)

                cute.copy(
                    tma_atom_LSE,
                    tLSEgLSE[None, m_block],
                    tLSEsLSE[None, 0],
                    tma_bar_ptr=lse_full_mbar_ptr,
                )

                # dO
                self.load_M_tile(tma_atom_dO, tdOgdO, tdOsdO, pipeline_do, m_block, producer_state=do_producer_state)
                pipeline_do.producer_commit(do_producer_state)
                do_producer_state.advance()

                # Psum
                cute.arch.mbarrier_wait(psum_empty_mbar_ptr, psum_empty_consumer_phase)
                psum_empty_consumer_phase ^= 1

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(psum_full_mbar_ptr, self.tma_copy_psum_bytes)

                cute.copy(
                    tma_atom_Psum,
                    tPsumgPsum[None, m_block],
                    tPsumsPsum[None, 0],
                    tma_bar_ptr=psum_full_mbar_ptr,
                )

            pipeline_q.producer_tail(q_producer_state)
            pipeline_do.producer_tail(do_producer_state)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


    @cute.jit
    def mma(
        self,
        tiled_mma_kq:  cute.core.TiledMma,
        tiled_mma_pdo: cute.core.TiledMma,
        tiled_mma_vdo: cute.core.TiledMma,
        tiled_mma_dsq: cute.core.TiledMma,
        tiled_mma_dsk: cute.core.TiledMma,
        thr_mma_kq:   cute.core.ThrMma,
        thr_mma_pdo:  cute.core.ThrMma,
        thr_mma_vdo:  cute.core.ThrMma,
        thr_mma_dsq:  cute.core.ThrMma,
        thr_mma_dsk:  cute.core.ThrMma,
        sQ:   cute.Tensor,
        sQt:  cute.Tensor,
        sK:   cute.Tensor,
        sV:   cute.Tensor,
        sdO:  cute.Tensor,
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sdS:  cute.Tensor,
        sKt:  cute.Tensor,
        sK_swizzle: cute.Swizzle,
        sQ_swizzle: cute.Swizzle,
        tStS: cute.Tensor,
        tdVtdV:       cute.Tensor,
        tdKtdK:       cute.Tensor,
        tdPtdP:       cute.Tensor,
        tdQacctdQacc: cute.Tensor,
        pipeline_q:  PipelineAsync,
        pipeline_do: PipelineAsync,
        pipeline_s:  PipelineAsync,
        pipeline_p:  PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dV: PipelineAsync,
        pipeline_dK: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQaccum: PipelineAsync,
        full_key_mbar_ptr:   cute.Pointer,
        full_value_mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        key_consumer_phase = cutlass.Int32(0)

        q_consumer_state     = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.q_stage)
        q_dk_consumer_state  = q_consumer_state
        do_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.do_stage)

        s_producer_state  = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.s_stage)
        dP_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dP_stage)
        p_consumer_state  = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.s_stage)
        dS_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dS_stage)
        dV_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dV_stage)
        dK_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dK_stage)
        dQaccum_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dQaccum_mma_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile  = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx) # must be seqlen_k

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            cute.arch.mbarrier_wait(full_key_mbar_ptr,     phase=key_consumer_phase)
            cute.arch.mbarrier_wait(full_value_mbar_ptr,   phase=key_consumer_phase)

            key_consumer_phase ^= 1

            # S = K @ Q.T sK and sQ
            tSrK = thr_mma_kq.make_fragment_A(sK)
            tSrQ = thr_mma_kq.make_fragment_B(sQ)

            # dP = V @ dOt
            tdPrV   = thr_mma_vdo.make_fragment_A(sV)
            tdPrdOt = thr_mma_vdo.make_fragment_B(sdOt)

            # dK = dS.T @ Q
            tdKrdS = thr_mma_dsq.make_fragment_A(sdSt)
            tdKrQ  = thr_mma_dsq.make_fragment_B(sQt)

            accumulate_dK = False

            # dV = P @ dO.T
            tdVrdO = thr_mma_pdo.make_fragment_B(sdO)
            p_tmem_layout = sm100_utils_basic.make_smem_layout_a(tiled_mma_pdo, self.mma_tiler_pdo, self.q_dtype, self.acc_stage,)

            tP    = cute.make_tensor(tStS.iterator, p_tmem_layout.outer)
            tdVrP = thr_mma_pdo.make_fragment_A(tP)[None, None, None, 0]
            tdVrP = cute.make_tensor(tdVrP.iterator, tdVrP.layout)

            # dQ = dS @ K
            tdQaccrdS = thr_mma_dsk.make_fragment_A(sdS)
            tdQaccrK  = thr_mma_dsk.make_fragment_B(sKt)


            #-----------------------------------------------------------
            ###### Prologue
            #-----------------------------------------------------------
            # 1. S  = Q0 @ K.T
            # 2. dP = V @ dO.T
            # 3. dV = P @ dO

            # 1) S  = Q0 @ K.T
            pipeline_q.consumer_wait(q_consumer_state)
            pipeline_s.producer_acquire(s_producer_state)

            num_k_phases = cute.size(tSrK, mode=[2])
            for kphase_idx in cutlass.range_constexpr(num_k_phases, unroll=1):
                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_kq,
                    tStS,
                    tSrK[(None, None, kphase_idx, 0)],
                    tSrQ[(None, None, kphase_idx, q_consumer_state.index)],
                    tStS,
                )

            q_consumer_state.advance()
            pipeline_s.producer_commit(s_producer_state)
            s_producer_state.advance()

            # 2) dP = V @ dO.T
            pipeline_do.consumer_wait(do_consumer_state)
            pipeline_dP.producer_acquire(dP_producer_state)

            pipeline_dQaccum.producer_acquire(dQaccum_producer_state)

            for kphase_idx in cutlass.range_constexpr(cute.size(tdPrV, mode=[2]), unroll=1):
                    tiled_mma_vdo.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_vdo,
                        tdPtdP,
                        tdPrV[(None, None, kphase_idx, 0)],
                        tdPrdOt[(None, None, kphase_idx, do_consumer_state.index)],
                        tdPtdP,
                    )
            pipeline_dP.producer_commit(dP_producer_state); dP_producer_state.advance()

            # 3) dV = P.T @ dO
            pipeline_p.consumer_wait(p_consumer_state)

            num_kphases = cute.size(tdVrP, mode=[2])
            for kphase_idx in cutlass.range_constexpr(num_kphases):
                tiled_mma_pdo.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_pdo,
                    tdVtdV,
                    tdVrP[(None,  None, kphase_idx)],
                    tdVrdO[(None, None, kphase_idx, do_consumer_state.index)],
                    tdVtdV,
                )
            pipeline_p.consumer_release(p_consumer_state); p_consumer_state.advance()
            pipeline_do.consumer_release(do_consumer_state); do_consumer_state.advance()
            #-----------------------------------------------------------
            ###### MAIN LOOP
            #-----------------------------------------------------------
            # 1. S  = K    @ Q.T
            # 2. dQ = dS   @ K
            # 3. dK = dS.T @ Q
            # 4. dP = V    @ dO.T
            # 5. dV = P.T  @ dO

            for i in cutlass.range(m_block_max - m_block_min - 1, unroll=1):
                # 1) S = K @ Q_i
                pipeline_q.consumer_wait(q_consumer_state)
                pipeline_s.producer_acquire(s_producer_state)
                #'''
                for kphase_idx in cutlass.range_constexpr(num_k_phases, unroll=1):
                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_kq,
                        tStS,
                        tSrK[(None, None, kphase_idx, 0)],
                        tSrQ[(None, None, kphase_idx, q_consumer_state.index)],
                        tStS,
                    )

                pipeline_s.producer_commit(s_producer_state)
                s_producer_state.advance()
                q_consumer_state.advance()

                # 2) dQ = dS @ K
                pipeline_dS.consumer_wait(dS_consumer_state)
                pipeline_dP.producer_acquire(dP_producer_state)

                num_kphases = cute.size(tdQaccrdS, mode=[2])
                for kphase_idx in cutlass.range_constexpr(num_kphases):
                    tiled_mma_dsk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        tiled_mma_dsk,
                        tdQacctdQacc,
                        tdQaccrdS[(None,  None, kphase_idx, dS_consumer_state.index)],
                        tdQaccrK[(None,   None, kphase_idx, 0)],
                        tdQacctdQacc,
                    )
                pipeline_dQaccum.producer_commit(dQaccum_producer_state) ; dQaccum_producer_state.advance()

                # 3) dK = dS.T @ Q
                num_kphases = cute.size(tdKrdS, mode=[2])
                for kphase_idx in cutlass.range_constexpr(num_kphases, unroll=1):
                    tiled_mma_dsq.set(tcgen05.Field.ACCUMULATE, accumulate_dK)
                    cute.gemm(
                        tiled_mma_dsq,
                        tdKtdK,
                        tdKrdS[(None,  None, kphase_idx, 0)],
                        tdKrQ[(None,   None, kphase_idx, q_dk_consumer_state.index)],
                        tdKtdK,
                    )
                    accumulate_dK = True

                pipeline_q.consumer_release(q_dk_consumer_state) ; q_dk_consumer_state.advance()
                pipeline_dS.consumer_release(dS_consumer_state); dS_consumer_state.advance()

                #4) dP = V @ dO.T
                pipeline_do.consumer_wait(do_consumer_state)

                pipeline_dQaccum.producer_acquire(dQaccum_producer_state)

                for kphase_idx in cutlass.range_constexpr(cute.size(tdPrV, mode=[2]), unroll=1):
                     tiled_mma_vdo.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                     cute.gemm(
                         tiled_mma_vdo,
                         tdPtdP,
                         tdPrV[(None, None, kphase_idx, 0)],
                         tdPrdOt[(None, None, kphase_idx, do_consumer_state.index)],
                         tdPtdP,
                     )
                pipeline_dP.producer_commit(dP_producer_state);  dP_producer_state.advance()

                # 5) dV += P @ dO
                pipeline_p.consumer_wait(p_consumer_state)

                num_kphases = cute.size(tdVrP, mode=[2])
                for kphase_idx in cutlass.range_constexpr(num_kphases):
                    tiled_mma_pdo.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tiled_mma_pdo,
                        tdVtdV,
                        tdVrP[(None,  None, kphase_idx)],
                        tdVrdO[(None, None, kphase_idx, do_consumer_state.index)],
                        tdVtdV,
                    )

                pipeline_p.consumer_release(p_consumer_state); p_consumer_state.advance()
                pipeline_do.consumer_release(do_consumer_state); do_consumer_state.advance()

            pipeline_dV.producer_acquire(dV_producer_state); pipeline_dV.producer_commit(dV_producer_state); dV_producer_state.advance()

            pipeline_s.producer_tail(s_producer_state)
            pipeline_dP.producer_tail(dP_producer_state)
            pipeline_dV.producer_tail(dV_producer_state)

            #-----------------------------------------------------------
            ###### Remaining 2
            #-----------------------------------------------------------
            # 1) dK += dS.T @ Q
            pipeline_dS.consumer_wait(dS_consumer_state)

            num_kphases = cute.size(tdKrdS, mode=[2])
            for kphase_idx in cutlass.range_constexpr(num_kphases):
                tiled_mma_dsq.set(tcgen05.Field.ACCUMULATE, accumulate_dK)
                cute.gemm(
                    tiled_mma_dsq,
                    tdKtdK,
                    tdKrdS[(None,  None, kphase_idx, dS_consumer_state.index)],
                    tdKrQ[(None,   None, kphase_idx, q_dk_consumer_state.index)],
                    tdKtdK,
                )
                accumulate_dK = True

            pipeline_dK.producer_acquire(dK_producer_state);
            pipeline_dK.producer_commit(dK_producer_state); dK_producer_state.advance()

            # 2) dQaccum = dS @ K
            num_kphases = cute.size(tdQaccrdS, mode=[2])
            for kphase_idx in cutlass.range_constexpr(num_kphases, unroll=1):
                tiled_mma_dsk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_dsk,
                    tdQacctdQacc,
                    tdQaccrdS[(None,  None, kphase_idx, dS_consumer_state.index)],
                    tdQaccrK[(None,   None, kphase_idx, 0)],
                    tdQacctdQacc,
                )
            pipeline_dQaccum.producer_commit(dQaccum_producer_state) ; dQaccum_producer_state.advance()
            pipeline_q.consumer_release(q_dk_consumer_state); q_dk_consumer_state.advance()
            pipeline_dS.consumer_release(dS_consumer_state);  dS_consumer_state.advance()

            pipeline_dK.producer_tail(dK_producer_state)
            pipeline_dQaccum.producer_tail(dQaccum_producer_state)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


    @cute.jit
    def split_wg(self, thr_tensor: cute.Tensor, wg_idx: cutlass.Int32, num_wg: cutlass.Constexpr[cutlass.Int32]):
        reduced_shape = cute.product_each(thr_tensor.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for thr_tensor in split_wg"
            t = cute.logical_divide(thr_tensor, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None, ) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for thr_tensor in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    thr_tensor, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg))
                coord = (None, None, (None, wg_idx), ) + (None, ) * (rank - 3)
            else:
                t = cute.logical_divide(thr_tensor, (reduced_shape[0], reduced_shape[1], reduced_shape[2], reduced_shape[3] // num_wg))
                coord = (None, None, None, (None, wg_idx), ) + (None, ) * (rank - 4)
        return t[coord]


    @cute.jit
    def compute_loop(
        self,
        thr_mma_kq:            cute.core.ThrMma,
        thr_mma_pdo:           cute.core.ThrMma,
        thr_mma_vdo:           cute.core.ThrMma,
        thr_mma_dsq:           cute.core.ThrMma,
        tStS:                  cute.Tensor,
        sLSE_2D:               cute.Tensor,
        sPsum_2D:              cute.Tensor,
        tdVtdV:                cute.Tensor,
        tdKtdK:                cute.Tensor,
        mdV:                   cute.Tensor,
        mdK:                   cute.Tensor,
        sdSt:                  cute.Tensor,
        sdSt_pi:               cute.Tensor,
        tdPtdP:                cute.Tensor,
        lse_full_mbar_ptr:     cute.Pointer,
        lse_empty_mbar_ptr:    cute.Pointer,
        psum_full_mbar_ptr:    cute.Pointer,
        psum_empty_mbar_ptr:   cute.Pointer,
        pipeline_s:            PipelineAsync,
        pipeline_p:            PipelineAsync,
        pipeline_dS:           PipelineAsync,
        pipeline_dV:           PipelineAsync,
        pipeline_dK:           PipelineAsync,
        pipeline_dP:           PipelineAsync,
        softmax_scale:         cutlass.Float32,
        softmax_scale_log2:    cutlass.Float32,
        block_info:            BlockInfo,
        SeqlenInfoCls:         Callable,
        AttentionMaskCls:      Callable,
        TileSchedulerCls:      Callable,
        sdV:                   Optional[cute.Tensor],
        sdK:                   Optional[cute.Tensor],
        mdV_tma_tensor:        Optional[cute.Tensor],
        mdK_tma_tensor:        Optional[cute.Tensor],
        tma_atom_dV:           Optional[cute.CopyAtom],
        tma_atom_dK:           Optional[cute.CopyAtom],
        tiled_copy_r2s_dKdV:   Optional[cute.TiledCopy],
        mdK_semaphore:         Optional[cute.Tensor],
        mdV_semaphore:         Optional[cute.Tensor],
    ):
        # tix: [128...384]  8 warps
        warp_idx =  cute.arch.make_warp_uniform(cute.arch.warp_idx()) # 4-11

        tidx     =  cute.arch.thread_idx()[0] % 128 # 0...128
        wg_idx   = (cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))) // 128
        num_wg   = (cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128) # 2

        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tmem_load_atom  = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        tmem_store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32)

        s_consumer_state   = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.s_stage)
        p_producer_state   = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.s_stage)
        dS_producer_state  = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.ds_stage)

        dP_consumer_state   = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dP_stage)

        lse_consumer_phase  = psum_consumer_phase =  cute.Int32(0)

        sub_packed_f32x2 = partial(cute.arch.calc_packed_f32x2_op, src_c=None, calc_func=nvvm.sub_packed_f32x2, rnd=nvvm.RoundingModeKind.RN )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                n_block=n_block, mask_seqlen=True, mask_causal=self.is_causal, mask_local=self.is_local
            )

            # Mainloop
            for i in cutlass.range(m_block_max - m_block_min, unroll=1):
                m_block = m_block_max - 1 - i

                pipeline_s.consumer_wait(s_consumer_state)
                pipeline_p.producer_acquire(p_producer_state)

                if warp_idx == self.compute_warp_ids[0]:
                    cute.arch.mbarrier_wait(lse_full_mbar_ptr, lse_consumer_phase)
                    lse_consumer_phase ^= 1

                tiled_tmem_ld = tcgen05.make_tmem_copy(tmem_load_atom,  tStS)
                thr_tmem_ld   = tiled_tmem_ld.get_slice(tidx)

                tileP_f32_like = self.mma_tiler_kq[0] // 32  * self.v_dtype.width # (128, 64)
                tStP           = cute.make_tensor(
                                    tStS.iterator,
                                    cute.composition(tStS.layout, cute.make_layout((self.m_block_size, tileP_f32_like))),
                                )

                tiled_tmem_st = tcgen05.make_tmem_copy(tmem_store_atom, tStP)
                thr_tmem_st   = tiled_tmem_st.get_slice(tidx)

                #### TMEM
                tStS_t2r_p = thr_tmem_ld.partition_S(tStS)
                tStS_t2r   = self.split_wg(tStS_t2r_p, wg_idx, num_wg)

                #### RMEM
                tScS        = thr_mma_kq.partition_C(cute.make_identity_tensor((self.mma_tiler_kq[0], self.mma_tiler_kq[1])))
                tScS_tensor = cute.make_tensor(tScS.iterator, tScS.layout)
                tScS_t2r_p  = thr_tmem_ld.partition_D(tScS_tensor)
                tScS_t2r    = self.split_wg(tScS_t2r_p, wg_idx, num_wg)

                tSrS_t2r    = cute.make_fragment(tScS_t2r.shape, Float32) # 64

                #### TMEM->RMEM (Load S from TMEM)
                cute.copy(tiled_tmem_ld, tStS_t2r, tSrS_t2r)
                cute.arch.fence_view_async_tmem_load()

                #### Sync for load fence and LSE
                cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.Compute), number_of_threads=self.num_compute_threads)

                #### APPLY MASK
                if const_expr(self.is_causal or self.is_local):
                    mask_fn(tSrS_t2r, tScS_t2r, m_block=m_block, )

                #---------------------------------------------
                #### P = exp(S - LSE)
                #---------------------------------------------

                #### RMEM (coordinates for P)
                cP_f32           =  cute.make_tensor(
                                    tScS.iterator,
                                    cute.composition(tScS.layout, cute.make_layout((self.m_block_size, tileP_f32_like)))
                                )

                tScP_r2t_p = thr_tmem_st.partition_S(cP_f32)
                tScP_r2t   = self.split_wg(tScP_r2t_p, wg_idx, num_wg)

                tStP_r2t_p = thr_tmem_st.partition_D(tStP)
                tStP_r2t   = self.split_wg(tStP_r2t_p, wg_idx, num_wg)

                #### Compute P = exp(S * scale - LSE)
                tLSE = thr_tmem_ld.partition_D(sLSE_2D)
                # split to  wg0 & wg1
                tLSErLSE_p = cute.make_tensor(cute.recast_ptr(tLSE.iterator), cute.make_layout((tScS_t2r_p.shape[0], (tScS_t2r_p.shape[1] // num_wg, num_wg), 1, 1)))
                tLSErLSE   = tLSErLSE_p[None, (None, wg_idx), None, None]


                WIDTH  = cute.arch.WARP_SIZE
                CLAMP  = WIDTH - 1
                MAC    = (0 << 8) | CLAMP
                FULL   = cute.arch.FULL_MASK

                lidx = cute.arch.lane_idx()


                tSrP_r2t_f32 = cute.make_fragment(tScP_r2t[None, None, 0].shape, Float32)  # 16
                tSrP_r2t     = cute.make_tensor(cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r[None, 0, None, None].layout)

                for i in cutlass.range_constexpr(cute.size(tStP_r2t, mode=[2]), unroll=1):

                    own0 = tLSErLSE[(lidx, 0), i, 0, 0]
                    own1 = tLSErLSE[(lidx+1, 0), i, 0, 0]
                    #own1 = cute.arch.shuffle_sync(own0, offset=((lidx + 1) & CLAMP),
                    #          mask=FULL, mask_and_clamp=MAC)

                    for j in cutlass.range_constexpr(0, cute.size(tSrP_r2t), 2, unroll=1):
                        lse_j  = cute.arch.shuffle_sync(own0, offset=j, mask=FULL, mask_and_clamp=MAC)
                        lse_j1 = cute.arch.shuffle_sync(own1, offset=j, mask=FULL, mask_and_clamp=MAC)

                        tSrS_t2r[j,   i, 0, 0], tSrS_t2r[j+1, i, 0, 0] = cute.arch.fma_packed_f32x2((
                                (tSrS_t2r[j,   i, 0, 0], tSrS_t2r[j+1, i, 0, 0])),
                                (softmax_scale_log2, softmax_scale_log2),
                                (-lse_j, -lse_j1))

                        tSrS_t2r[j,   i, 0, 0] = cute.arch.exp2(tSrS_t2r[j,   i, 0, 0])
                        tSrS_t2r[j+1, i, 0, 0] = cute.arch.exp2(tSrS_t2r[j+1, i, 0, 0])

                        tSrP_r2t[j,   0, 0] = tSrS_t2r[j,   i, 0, 0].to(self.q_dtype)
                        tSrP_r2t[j+1, 0, 0] = tSrS_t2r[j+1, i, 0, 0].to(self.q_dtype)

                    cute.copy(thr_tmem_st, tSrP_r2t_f32[None, None], tStP_r2t[None, None, i])

                cute.arch.fence_view_async_tmem_store()
                cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.Compute), number_of_threads=self.num_compute_threads)

                pipeline_p.producer_commit(p_producer_state)
                p_producer_state.advance()

                pipeline_s.consumer_release(s_consumer_state)
                s_consumer_state.advance()

                if warp_idx == self.compute_warp_ids[0]:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(lse_empty_mbar_ptr)

                #---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                #---------------------------------------------
                if warp_idx == self.compute_warp_ids[0]:
                    cute.arch.mbarrier_wait(psum_full_mbar_ptr, psum_consumer_phase)
                psum_consumer_phase ^= 1

                pipeline_dP.consumer_wait(dP_consumer_state)
                pipeline_dS.producer_acquire(dS_producer_state)

                #### TMEM->RMEM (Load dP from TMEM)
                tiled_tmem_ld_dP = tcgen05.make_tmem_copy(tmem_load_atom, tdPtdP)
                thr_tmem_ld_dP   = tiled_tmem_ld_dP.get_slice(tidx)

                tdPtdP_t2r_p = thr_tmem_ld_dP.partition_S(tdPtdP) #
                tdPtdP_t2r   = self.split_wg(tdPtdP_t2r_p, wg_idx, num_wg)

                #### TMEM->RMEM (Load dP from TMEM)
                cdP           = cute.make_identity_tensor((self.mma_tiler_vdo[0], self.mma_tiler_vdo[1]))
                tdPcdP        = thr_mma_vdo.partition_C(cdP)
                tdPcdP_tensor = cute.make_tensor(tdPcdP.iterator, tdPcdP.layout)

                tdPcdP_t2r_p = thr_tmem_ld_dP.partition_D(tdPcdP_tensor)
                tdPcdP_t2r   = self.split_wg(tdPcdP_t2r_p, wg_idx, num_wg)
                tdPrdP_t2r   = cute.make_fragment(tdPcdP_t2r[(None, 0, None, None)].shape, Float32) # ((32,1),1,1)

                #### Sync for load fence and Psum
                cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.Compute), number_of_threads=self.num_compute_threads)

                ##### dS.T = P.T * (dP.T - Psum)
                sdSt_mn = cute.make_tensor(sdSt_pi.iterator, cute.composition(sdSt_pi.layout, cute.make_layout((self.m_block_size, self.n_block_size))))
                tdKsdS =  cute.composition(sdSt_mn[(None, wg_idx), tidx], cute.make_layout(tSrS_t2r.shape))

                tSrS_t2r_bf16 = cute.make_tensor(cute.recast_ptr(tSrS_t2r.iterator, dtype=self.ds_dtype), tSrS_t2r.shape)

                tPsum = thr_tmem_ld.partition_D(sPsum_2D)
                tPsumrPsum_p = cute.make_tensor(cute.recast_ptr(tPsum.iterator), cute.make_layout((tScS_t2r_p.shape[0], (tScS_t2r_p.shape[1] // num_wg, num_wg), 1, 1)))
                tPsumrPsum   = tPsumrPsum_p[None, (None, wg_idx), None, None] # self.split_wg(tLSErLSE_p, wg_idx, num_wg)

                for i in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[1]), unroll=1):
                    cute.copy(thr_tmem_ld_dP, tdPtdP_t2r[None, i, None, None], tdPrdP_t2r)
                    cute.arch.fence_view_async_tmem_load()

                    own0 = tPsumrPsum[(lidx, 0), i, 0, 0]
                    own1 = tPsumrPsum[(lidx+1, 0), i, 0, 0]

                    for j in cutlass.range_constexpr(0, cute.size(tdPrdP_t2r), 2, unroll=1):

                        psum_j  = cute.arch.shuffle_sync(own0, offset=j, mask=FULL, mask_and_clamp=MAC)
                        psum_j1 = cute.arch.shuffle_sync(own1, offset=j, mask=FULL, mask_and_clamp=MAC)

                        tdPrdP_t2r[j, 0, 0], tdPrdP_t2r[j+1, 0, 0] = sub_packed_f32x2(
                                        (tdPrdP_t2r[j, 0, 0], tdPrdP_t2r[j+1, 0, 0]),
                                        (psum_j, psum_j1)
                                        )

                        tSrS_t2r[j, i, 0, 0], tSrS_t2r[j+1, i, 0, 0] = cute.arch.mul_packed_f32x2(
                                        (tSrS_t2r[j, i, 0, 0], tSrS_t2r[j+1, i, 0, 0]),
                                        (tdPrdP_t2r[j, 0, 0], tdPrdP_t2r[j+1, 0, 0])
                                        )

                        tSrS_t2r_bf16[j, i, 0, 0]   = tSrS_t2r[j, i, 0, 0].to(self.ds_dtype)
                        tSrS_t2r_bf16[j+1, i, 0, 0] = tSrS_t2r[j+1, i, 0, 0].to(self.ds_dtype)

                    cute.autovec_copy(tSrS_t2r_bf16[None, i, 0, 0], tdKsdS[None, i, 0, 0])

                pipeline_dP.consumer_release(dP_consumer_state)
                dP_consumer_state.advance()

                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.Compute), number_of_threads=self.num_compute_threads)

                pipeline_dS.producer_commit(dS_producer_state)
                dS_producer_state.advance()

                if warp_idx == self.compute_warp_ids[0]:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(psum_empty_mbar_ptr)

            if const_expr(not self.use_tma_store):
                self.epilogue_dKV(
                    tidx,
                    warp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_pdo,
                    thr_mma_dsq,
                    tdVtdV,
                    tdKtdK,
                    mdV,
                    mdK,
                    pipeline_dV,
                    pipeline_dK,
                    softmax_scale,
                )
            else:
                thr_copy_r2s_dKdV = tiled_copy_r2s_dKdV.get_slice(tidx)
                #### STORE dV
                self.epilogue_dK_or_dV_tma(
                    tidx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_pdo,
                    tdVtdV,
                    mdV_tma_tensor,
                    sdV,
                    tma_atom_dV,
                    thr_copy_r2s_dKdV,
                    pipeline_dV,
                    softmax_scale,
                    False, # apply scale
                    int(NamedBarrierBwdSm100.EpilogueWG1), # barrier_id
                    mdV_semaphore,
                )
                #### STORE dK
                self.epilogue_dK_or_dV_tma(
                    tidx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dsq,
                    tdKtdK,
                    mdK_tma_tensor,
                    sdK,
                    tma_atom_dK,
                    thr_copy_r2s_dKdV,
                    pipeline_dK,
                    softmax_scale,
                    True, # apply scale
                    int(NamedBarrierBwdSm100.EpilogueWG1), # barrier_id
                    mdK_semaphore,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum:              cute.Tensor,
        sdQaccum:              cute.Tensor,
        thr_mma_dsk:           cute.core.ThrMma,
        tdQtdQ:                cute.Tensor,
        pipeline_dQ:           PipelineAsync,
        dQaccum_reduce_mbar_ptr: cute.Pointer,
        block_info:            BlockInfo,
        SeqlenInfoCls:         Callable,
        TileSchedulerCls:      Callable,
        mdQ_semaphore:         Optional[cute.Tensor],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx     = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * 4)

        dQ_consumer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dQaccum_mma_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        # TMEM -> RMEM
        tmem_ld_atom  = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        tiled_tmem_ld = tcgen05.make_tmem_copy(tmem_ld_atom, tdQtdQ)
        thr_tmem_ld   = tiled_tmem_ld.get_slice(tidx)

        tdQtdQ_t2r    = thr_tmem_ld.partition_S(tdQtdQ)

        cdQ           = cute.make_identity_tensor((self.mma_tiler_dsk[0], self.mma_tiler_dsk[1]))
        tdQcdQ        = thr_mma_dsk.partition_C(cdQ)
        tdQcdQ_tensor = cute.make_tensor(tdQcdQ.iterator, tdQcdQ.layout)
        tdQrdQ        = thr_tmem_ld.partition_D(tdQcdQ_tensor)

        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)

        atom_universal_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dqaccum_dtype, num_bits_per_copy=128)
        thr_layout = cute.make_layout(shape=128, stride=1)
        val_layout = cute.make_layout(shape=4,   stride=1)

        tiler_mn, layout_tv = cute.make_layout_tv(thr_layout=thr_layout, val_layout=val_layout)
        tiled_smem_store    = cute.make_tiled_copy(atom_universal_copy, layout_tv=layout_tv, tiler_mn=tiler_mn)


        smem_thr_copy_dQaccum = tiled_smem_store.get_slice(tidx)
        tdQsdQ = smem_thr_copy_dQaccum.partition_D(sdQaccum)
        store_bytes = cutlass.Int32(self.m_block_size * 32 * 4)

        if const_expr(self.deterministic):
            read_flag = False
        else:
            read_flag = True

        reduce_phase = cutlass.Int32(0)
        if cute.arch.thread_idx()[0] == 0:
            cute.arch.mbarrier_arrive(dQaccum_reduce_mbar_ptr)

        cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.dQaccReduce), number_of_threads=num_reduce_threads)

        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]

            if const_expr(self.deterministic):
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            for i in cutlass.range(m_block_max - m_block_min, unroll=1):
                m_block = m_block_max - 1 - i

                pipeline_dQ.consumer_wait(dQ_consumer_state)

                # TMEM -> RMEM
                tdQrdQ_t2r = cute.make_fragment(tdQrdQ.shape, Float32)
                assert self.dQaccum_reduce_stage == cute.size(tdQrdQ_t2r, mode=[1]), "dQaccum reduce stage mismatch"

                cute.copy(thr_tmem_ld, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()

                pipeline_dQ.consumer_release(dQ_consumer_state); dQ_consumer_state.advance()

                # semaphore acquire
                if const_expr(self.deterministic):
                    barrier.wait_eq(mdQ_semaphore_cur[(m_block, None)].iterator, tidx, 0, n_block)
                    cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.dQaccReduce), number_of_threads=num_reduce_threads)

                for stage in cutlass.range_constexpr(cute.size(tdQrdQ_t2r, mode=[1])): # 4

                    if stage >= 2 and cute.arch.thread_idx()[0] == 0:
                        cute.arch.cp_async_bulk_wait_group(1, read=read_flag)

                    cute.arch.mbarrier_wait(dQaccum_reduce_mbar_ptr, reduce_phase)

                    tdQrdQ_r2s = tdQrdQ_t2r[None, stage, None, None]
                    tdQsdQ_r2s = tdQsdQ[None, None, reduce_phase]
                    tdQrdQ_r2s = cute.make_tensor(tdQrdQ_r2s.iterator, cute.make_layout(tdQsdQ_r2s.shape))

                    cute.copy(smem_thr_copy_dQaccum, tdQrdQ_r2s, tdQsdQ_r2s)

                    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                    cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.dQaccReduce), number_of_threads=num_reduce_threads)

                    if cute.arch.thread_idx()[0] == 0:
                        smem_ptr = sdQaccum[None, reduce_phase].iterator
                        g_stage_index_elems = m_block * (self.m_block_size *  self.head_dim_v_padded) + stage * (self.m_block_size * 32)
                        gmem_row_ptr = cute.domain_offset((g_stage_index_elems,), mdQaccum_cur).iterator

                        tma_reduce_add_bulk_f32(smem_ptr, gmem_row_ptr, store_bytes)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(1, read=read_flag)

                        cute.arch.mbarrier_arrive(dQaccum_reduce_mbar_ptr)

                    reduce_phase ^= 1

                    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                    cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.dQaccReduce), number_of_threads=num_reduce_threads)

                # semaphore release
                # NOTE: arrive_inc calls red_release which issues membar
                if const_expr(self.deterministic):
                    if cute.arch.thread_idx()[0] == 0:
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                    cute.arch.barrier(barrier_id=int(NamedBarrierBwdSm100.dQaccReduce), number_of_threads=num_reduce_threads)
                    barrier.arrive_inc(mdQ_semaphore_cur[(m_block, None)].iterator, tidx, 0, 1)


            if cute.arch.thread_idx()[0] == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


    @cute.jit
    def epilogue_dKV(
        self,
        tidx:       Int32,
        warp_idx:   Int32,
        batch_idx:  Int32,
        head_idx:   Int32,
        n_block:    Int32,
        thr_mma_pdo:   cute.core.ThrMma,
        thr_mma_dsq:   cute.core.ThrMma,
        tdVtdV:        cute.Tensor,
        tdKtdK:        cute.Tensor,
        mdV:           cute.Tensor,
        mdK:           cute.Tensor,
        pipeline_dV:   PipelineAsync,
        pipeline_dK:   PipelineAsync,
        softmax_scale: Float32,
    ):

        wg_idx = (cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))) // 128
        num_wg = (cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128)

        dV_consumer_state   = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dV_stage)
        dK_consumer_state   = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dK_stage)

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = mdV[None, None, head_idx, batch_idx]
        mdK_cur = mdK[None, None, head_idx, batch_idx]

        tmem_load_atom  = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32)

        # dV
        pipeline_dV.consumer_wait(dV_consumer_state)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV   = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r   = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV           = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV        = thr_mma_pdo.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r   = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r   = cute.make_fragment(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dv_dtype, num_bits_per_copy=universal_copy_bits,)
        tiled_gmem_store_dV = cute.make_tiled_copy(atom_universal_copy, layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled, tiler_mn=tiled_tmem_ld_dV.tiler_mn,)

        tdVrdV_r2s  = cute.make_fragment(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
        gdV_tile = gdV[None, None, n_block]

        tdVgdV       = thr_mma_pdo.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g   = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dV, tdVrdV_r2s , tdVgdV_r2g)

        pipeline_dV.consumer_release(dV_consumer_state); dV_consumer_state.advance()

        # dK
        pipeline_dK.consumer_wait(dK_consumer_state)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK   = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r   = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK            = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK         = thr_mma_dsq.partition_C(cdK)
        tdKcdK_tensor  = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r   = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r   = cute.make_fragment(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dk_dtype, num_bits_per_copy=universal_copy_bits,)

        tiled_gmem_store_dK = cute.make_tiled_copy(atom_universal_copy,layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,tiler_mn=tiled_tmem_ld_dK.tiler_mn,)

        tdKrdK_r2s  = cute.make_fragment(tdKrdK_t2r.shape, self.dk_dtype)


        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.n_block_size, self.head_dim_v_padded), (None, 0))
        gdK_tile = gdK[None, None, n_block]

        tdKgdK       = thr_mma_dsq.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g   = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dK, tdKrdK_r2s , tdKgdK_r2g)

        pipeline_dK.consumer_release(dK_consumer_state); dK_consumer_state.advance()


    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx:       Int32,
        batch_idx:  Int32,
        head_idx:   Int32,
        n_block:    Int32,
        thr_mma:    cute.core.ThrMma,
        tdKVtdKV:   cute.Tensor,
        mdKV:       cute.Tensor,
        sdKV:       cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKdV: cute.TiledCopy,
        pipeline:   PipelineAsync,
        softmax_scale : Float32,
        do_scale : cutlass.Constexpr[cutlass.Boolean],
        barrier_id : Int32,
        mdKV_semaphore : Optional[cute.Tensor],
    ):
        # assumes mma_tiler_pdo = mma_tiler_dsq = (n_block_size, head_dim)
        # head_dim = head_dim_v, dk_dtype = dv_dtype

        wg_idx = (cute.arch.thread_idx()[0] % self.num_compute_threads) // 128
        num_wg = (self.num_compute_threads // 128)
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        sdKV = sdKV[None, None, wg_idx]

        head_idx_kv = head_idx // self.qhead_per_kvhead
        mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]

        gdKV_p = cute.local_tile(mdKV_cur, (self.m_block_size, self.head_dim_v_padded), (n_block, 0))
        gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)
        gdKV_epi = cute.local_tile(gdKV, self.sdKdV_epi_tile, (0, None))

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

        # (TMA) and (TMA, EPI_STAGE)
        tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
            tma_atom_dKV,
            0, # no multicast
            cute.make_layout(1),
            cute.group_modes(sdKV, 0, 2),
            cute.group_modes(gdKV_epi, 0, 2),
        )

        assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
        assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"

        num_epi_stages = cute.size(tdKVgdKV.shape[1])
        assert num_epi_stages == 1 or num_epi_stages == 2, "Wrong number of epi stages"

        tmem_ld_atom  = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)

        if const_expr(self.deterministic):
            read_flag = False
        else:
            read_flag = True

        # TODO: maybe support more than 1 stage
        consumer_state   = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        pipeline.consumer_wait(consumer_state)

        # semaphore acquire
        if const_expr(self.deterministic):
            barrier.wait_eq(mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for s in cutlass.range_constexpr(num_epi_stages):

            # TMEM -> RMEM -- setup
            tiled_tmem_ld = tcgen05.make_tmem_copy(tmem_ld_atom, tdKVtdKV)
            thr_tmem_ld   = tiled_tmem_ld.get_slice(tidx)

            tdKVtdKV_t2r_p = thr_tmem_ld.partition_S(tdKVtdKV)
            tdKVtdKV_t2r   = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, s]

            cdKV           = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
            tdKVcdKV       = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_tmem_ld.partition_D(tdKVcdKV)
            tdKVcdKV_t2r   = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, s]

            tdKVrdKV_t2r   = cute.make_fragment(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, "RMEM<->TMEM fragment size mismatch"

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_tmem_ld, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            tdKVrdKV  = cute.make_fragment(tdKVrdKV_t2r.shape, self.dv_dtype)
            if const_expr(do_scale):
                scale = softmax_scale
            else:
                scale = Float32(1)

            dKV_vec = tdKVrdKV_t2r.load() * scale
            tdKVrdKV.store(dKV_vec.to(self.dv_dtype))

            # RMEM -> SMEM -- setup
            tdKVcdKV_r2s_p = thr_copy_r2s_dKdV.partition_S(cdKV)
            tdKVcdKV_r2s = self.split_wg(tdKVcdKV_r2s_p, wg_idx, num_wg)
            tdKVcdKV_r2s = cute.logical_divide(
                tdKVcdKV_r2s,
                (tdKVcdKV_r2s.shape[0], tdKVcdKV_r2s.shape[1], tdKVcdKV_r2s.shape[2] // num_epi_stages)
            )[((None, 0), (None, 0), (None, s))]

            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVcdKV_r2s.shape)

            tdKVsdKV_r2s = thr_copy_r2s_dKdV.partition_D(sdKV)

            assert cute.size(tdKVrdKV_r2s) == cute.size(tdKVsdKV_r2s), "RMEM<->SMEM fragment size mismatch"

            # RMEM -> SMEM -- copy, fence and barrier
            cute.copy(thr_copy_r2s_dKdV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, s])
                if s < num_epi_stages - 1:
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(self.deterministic):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        pipeline.consumer_release(consumer_state)
        consumer_state.advance()


    @cute.jit
    def load_M_tile(
        self,
        tma_atom: cute.CopyAtom,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        pipeline: PipelineAsync,
        block: cutlass.Int32,
        producer_state: cutlass.pipeline.PipelineState,
    ):
        pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tQgQ[None, block],
            tQsQ[None, producer_state.index],
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state)
        )
