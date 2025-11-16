# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SM120 (RTX 50) backward implementation using tcgen05 SuperMMA instructions.
# SM120 is a stripped-down version of SM100: it lacks the Cluster fabric (GPC-CGA), Tensor Memory, and UTCMMA instructions,
# and only supports the smaller-tile "SuperMMA" operations.
# This implementation uses shared memory for intermediate tensors (S, dP, dV, dK, dQ, P, dS) instead of Tensor Memory.
# PTX references:
# https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-sp
# https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift
import math
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import PipelineAsync, PipelineConsumer

from flash_attn.cute import utils
from flash_attn.cute import copy_utils
from flash_attn.cute import pipeline
from flash_attn.cute.sm120_helpers import gemm_w_idx, gemm_ptx_w_idx  # noqa
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,  # noqa
    ParamsBase,
)

from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwdSm120


class FlashAttentionBackwardSm120:
    arch = 120

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        cluster_size: int = 1,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        assert head_dim == head_dim_v, "head_dim and head_dim_v must be the same for now"
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        assert self.tile_hdim == self.tile_hdimv, (
            "tile_hdim and tile_hdimv must be the same for now"
        )
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n

        # CTA tiler
        self.cta_tiler = (tile_m, tile_n, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q (N, M) (M, D)
        self.mma_tiler_dsq = (tile_n, self.tile_hdimv, tile_m)
        # dQ = dS @ K
        self.mma_tiler_dsk = (tile_m, self.tile_hdimv, tile_n)

        self.acc_dtype = Float32

        # SM120: No cluster fabric support - cluster_size must be 1
        assert cluster_size == 1, "SM120 does not support clusters - cluster_size must be 1"
        self.cluster_shape_mn = (1, 1)  # SM120: Fixed to (1, 1) - no cluster support
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = False
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.use_tma_store = True
        self.deterministic = deterministic

        # Speed optimizations, does not affect correctness
        self.shuffle_LSE = False
        self.shuffle_dPsum = False

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

        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm120.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        # self.epilogue_sync_barrier = pipeline.NamedBarrier(
        #     barrier_id=2,
        #     num_threads=self.num_compute_warps * self.threads_per_warp,
        # )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm120.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )

        # TMEM setup
        # SM120: No Tensor Memory - using shared memory instead
        # self.tmem_alloc_cols removed - not needed for SM120

        # SM120: Tensor Memory offsets removed - using shared memory instead
        # These offsets were used for Tensor Memory addressing, but SM120 doesn't have Tensor Memory

        if not is_causal and not is_local:
            self.num_regs_reduce = 152
            self.num_regs_compute = 136
        else:
            self.num_regs_reduce = 136
            self.num_regs_compute = 144
        self.num_regs_other = 96 - 8
        self.num_regs_empty = 24
        assert self.num_regs_reduce + self.num_regs_compute * 2 + self.num_regs_other <= 512

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 2
        self.dO_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        # self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        self.dQ_reduce_ncol = 32
        self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
        assert self.tile_hdim % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1
        # number of tma reduce adds for dKacc and dVacc epilogue
        self.dK_reduce_ncol = 32

    def _get_tiled_mma(self):
        cta_group = tcgen05.CtaGroup.ONE
        # S = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P @ dO --> (K, MN) major
        # dV += P @ dO
        # SM120: P is in shared memory, not Tensor Memory
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.SMEM,  # SM120: P is in shared memory
        )
        # dK += dS.T @ Q
        # SM120: dS is in shared memory, not Tensor Memory
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.SMEM,  # SM120: dS is in shared memory
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdKV_epi_tile = (
            self.tile_n,
            128 // (self.dk_dtype.width // 8), # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.num_epi_stages = (self.tile_hdim // 2) // self.sdKV_epi_tile[1]
        self.sdKV_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages

        # TODO: dK and dV could have different shapes
        if const_expr(self.qhead_per_kvhead == 1):
            self.sdKV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdKV_epi_tile,
                2, # num compute wgs
            )
        else:
            self.sdKV_layout = cute.make_layout(
                (self.tile_n * self.dK_reduce_ncol, 2)
            )

        # SM120: Create shared memory layouts for accumulators (replacing Tensor Memory)
        # S = K @ Q.T, shape (tile_n, tile_m)
        self.sS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            Float32,  # accumulator dtype
            1,
        )
        # dP = V @ dO.T, shape (tile_n, tile_m)
        self.sdP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            Float32,  # accumulator dtype
            1,
        )
        # dV += P @ dO, shape (tile_n, tile_hdimv)
        self.sdV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            Float32,  # accumulator dtype
            1,
        )
        # dK += dS.T @ Q, shape (tile_n, tile_hdim)
        self.sdK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            Float32,  # accumulator dtype
            1,
        )
        # dQ = dS @ K, shape (tile_m, tile_hdim)
        self.sdQ_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            Float32,  # accumulator dtype
            1,
        )
        # P layout already exists as tP_layout
        # dS layout already exists as tdS_layout

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
        assert all(x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)), (
            "Variable sequence length is not supported yet in FlashAttentionBackwardSm100"
        )
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        if const_expr(self.qhead_per_kvhead > 1):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        (mdQaccum,) = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mdQaccum,)
        ]

        layout_transpose = [1, 3, 2, 0]  # (b, s, n, h) --> (s, h, n, b)
        mQ, mK, mV, mdO = [
            utils.select(t, mode=layout_transpose) for t in (mQ, mK, mV, mdO)
        ]
        LSE_dPsum_dQaccum_transpose = [2, 1, 0]  # (b, n, s) --> (s, n, b)
        mLSE, mdPsum, mdQaccum = [
            utils.select(t, mode=LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]
        if const_expr(self.qhead_per_kvhead == 1):
            layout_dKV_transpose = layout_transpose
        else:
            layout_dKV_transpose = LSE_dPsum_dQaccum_transpose
        mdK, mdV = [
            utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)
        ]
        dO_transpose = [1, 0, 2, 3] # (s, h, n, b) --> (h, s, n, h)
        mdO = utils.select(mdO, mode=dO_transpose)

        semaphore_transpose = [2, 3, 1, 0]  # (b, n, block, stage) -> (block, stage, n, b)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = utils.select(mdQ_semaphore.layout, mode=semaphore_transpose)

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                utils.select(t.layout, mode=semaphore_transpose)
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        cta_group = tcgen05.CtaGroup.ONE

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(self.qhead_per_kvhead == 1):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")
            
        if const_expr(self.use_tma_store and self.qhead_per_kvhead == 1):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdKV_layout, mode=[0, 1]),
                self.sdKV_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdKV_layout, mode=[0, 1]),
                self.sdKV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(self.qhead_per_kvhead == 1):
            thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_load_op_multicast = cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group)

        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            # tma_load_op if const_expr(self.cluster_shape_mnk[0] == 1) else tma_load_op_multicast,
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dP.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            # tma_load_op if const_expr(self.cluster_shape_mnk[0] == 1) else tma_load_op_multicast,
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8

        # TileScheduler = SingleTileScheduler if not self.is_causal else SingleTileLPTBwdScheduler
        TileScheduler = SingleTileScheduler
        # TODO -- optimizer scheduler for causal
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3]),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            cluster_shape_mn=self.cluster_shape_mnk[:2],
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
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.dQaccum_reduce_stage // 2
            ]
            dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.dQaccum_reduce_stage // 2
            ]
            # SM120: Tensor Memory removed - using shared memory instead
            # tmem_holding_buf and tmem_dealloc_mbar_ptr removed

            # Smem tensors
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                128,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                128,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                128,
            ]
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                self.buffer_align_bytes,
            ]
            # SM120: Shared memory buffers for accumulators (replacing Tensor Memory)
            sS: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sS_layout)],
                self.buffer_align_bytes,
            ]
            sdP: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sdP_layout)],
                self.buffer_align_bytes,
            ]
            sdV: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sdV_layout)],
                self.buffer_align_bytes,
            ]
            sdK: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sdK_layout)],
                self.buffer_align_bytes,
            ]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sdQ_layout)],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.do_dtype, cute.cosize(self.tP_layout)],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.tdS_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mLSE,
            mdPsum,
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
            tma_atom_dO,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sKt_layout,
            self.sdQaccum_layout,
            self.sdKV_layout,
            self.sS_layout,
            self.sdP_layout,
            self.sdV_layout,
            self.sdK_layout,
            self.sdQ_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdKV_layout: cute.ComposedLayout | cute.Layout,
        sS_layout: cute.ComposedLayout,
        sdP_layout: cute.ComposedLayout,
        sdV_layout: cute.ComposedLayout,
        sdK_layout: cute.ComposedLayout,
        sdQ_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
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
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # SM120: Tensor Memory deallocation barrier removed - not needed
        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()
        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        # Only 1 thread per warp will signal
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids),
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            # cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.num_mcast_ctas_b
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            # init_wait=False,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            # init_wait=False,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            init_wait=False,
        )
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            init_wait=True,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQt_layout.inner), sQt_layout.outer)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sdOt = cute.make_tensor(cute.recast_ptr(sdO.iterator, sdOt_layout.inner), sdOt_layout.outer)
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(self.qhead_per_kvhead == 1):
            sdV = storage.sdO.get_tensor(
                sdKV_layout.outer, swizzle=sdKV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sQ.get_tensor(
                sdKV_layout.outer, swizzle=sdKV_layout.inner, dtype=self.dk_dtype
            )
        else:
            sdV = storage.sdO.get_tensor(sdKV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdKV_layout, dtype=self.dk_dtype)
        assert cute.size_in_bytes(self.do_dtype, sdO_layout) >= cute.size_in_bytes(
            self.dv_dtype, sdKV_layout
        ), "Not enough space for sdV"
        assert cute.size_in_bytes(self.q_dtype, sQ_layout) >= cute.size_in_bytes(
            self.dk_dtype, sdKV_layout
        ), "Not enough space for sdK"
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # SM120: Create shared memory tensors for accumulators (replacing Tensor Memory)
        sS = storage.sS.get_tensor(sS_layout.outer, swizzle=sS_layout.inner)
        sdP = storage.sdP.get_tensor(sdP_layout.outer, swizzle=sdP_layout.inner)
        sdV = storage.sdV.get_tensor(sdV_layout.outer, swizzle=sdV_layout.inner)
        sdK = storage.sdK.get_tensor(sdK_layout.outer, swizzle=sdK_layout.inner)
        sdQ = storage.sdQ.get_tensor(sdQ_layout.outer, swizzle=sdQ_layout.inner)
        sP = storage.sP.get_tensor(tP_layout.outer, swizzle=tP_layout.inner)
        sdS = storage.sdS.get_tensor(tdS_layout.outer, swizzle=tdS_layout.inner)

        # SM120: Create register fragments for accumulators (will be stored to shared memory after GEMM)
        thr_mma_S = tiled_mma_S.get_slice(0)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)  # SM120: Register fragment, will store to smem
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)  # SM120: Register fragment
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)  # SM120: Register fragment
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)  # SM120: Register fragment
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)  # SM120: Register fragment

        block_info = BlockInfo(
            self.tile_m,
            # self.tile_n,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # TODO: support local
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            swap_AB=True,
        )

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
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                mQ,
                mK,
                mV,
                mLSE,
                mdPsum,
                mdO,
                sQ,
                sK,
                sV,
                sLSE,
                sdPsum,
                sdO,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                should_load_Q=True,
                should_load_dO=True,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # SM120: No Tensor Memory allocation needed - using shared memory instead
            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sV,
                sdO,
                sdOt,
                sdSt,
                sdS,
                sKt,
                sS,
                sdP,
                sdV,
                sdK,
                sdQ,
                sP,
                sdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                pipeline_Q.make_consumer(),
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)  # 8 warps
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                sS,  # SM120: Shared memory for S instead of register fragment
                sLSE,
                sdPsum,
                sdV,  # SM120: Shared memory for dV accumulator
                sdK,  # SM120: Shared memory for dK accumulator
                mdV,
                mdK,
                sdS,
                sdP,  # SM120: Shared memory for dP instead of register fragment
                sP,  # SM120: Shared memory for P
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
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
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
            )
            # SM120: Tensor Memory deallocation barrier removed - not needed

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                sdQ,  # SM120: Shared memory for dQ accumulator instead of Tensor Memory
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
            )

        return

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mQ_cur = mQ[None, None, head_idx, batch_idx]
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            mPsum_cur = mdPsum[None, head_idx, batch_idx]

            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
            tSgK = thr_mma_S.partition_A(gK)
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
            tdPgV = thr_mma_dP.partition_A(gV)
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_n,), (None,))
            gdPsum = cute.local_tile(mPsum_cur, (self.tile_n,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dP.partition_B(gdO)

            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), tSgK, sK, single_stage=True
            )
            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdPgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)
            # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
            # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)

            # First iteration: load K together w Q & LSE, then V together w dO & dPsum
            if const_expr(should_load_Q):
                # K & Q
                pipeline_Q.producer_acquire(
                    producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                )
                load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                load_Q(m_block_min, producer_state=producer_state_Q_LSE)
                pipeline_Q.producer_commit(producer_state_Q_LSE)
                # LSE
                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block_min],
                        sLSE[None, producer_state_Q_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                    )
                producer_state_Q_LSE.advance()
            if const_expr(should_load_dO):
                # V & dO
                pipeline_dO.producer_acquire(
                    producer_state_dO_dPsum, extra_tx_count=self.tma_copy_bytes["V"]
                )
                load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                load_dO(m_block_min, producer_state=producer_state_dO_dPsum)
                pipeline_dO.producer_commit(producer_state_dO_dPsum)
                # dPsum
                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block_min],
                        sdPsum[None, producer_state_dO_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                    )
                producer_state_dO_dPsum.advance()

            for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                if const_expr(should_load_Q):
                    # Q
                    pipeline_Q.producer_acquire(producer_state_Q_LSE)
                    load_Q(m_block, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    # dO
                    pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                    load_dO(m_block, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                        )
                    producer_state_dO_dPsum.advance()

            if const_expr(should_load_Q):
                pipeline_Q.producer_tail(
                    producer_state_Q_LSE.clone()
                )  # will hang if we don't clone
                pipeline_LSE.producer_tail(producer_state_Q_LSE)
            if const_expr(should_load_dO):
                pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        sKt: cute.Tensor,
        sS: cute.Tensor,  # SM120: Shared memory for S accumulator
        sdP: cute.Tensor,  # SM120: Shared memory for dP accumulator
        sdV: cute.Tensor,  # SM120: Shared memory for dV accumulator
        sdK: cute.Tensor,  # SM120: Shared memory for dK accumulator
        sdQ: cute.Tensor,  # SM120: Shared memory for dQ accumulator
        sP: cute.Tensor,  # SM120: Shared memory for P
        sdS_smem: cute.Tensor,  # SM120: Shared memory for dS
        tStS: cute.Tensor,  # SM120: Register fragment for S accumulator
        tdPtdP: cute.Tensor,  # SM120: Register fragment for dP accumulator
        tdVtdV: cute.Tensor,  # SM120: Register fragment for dV accumulator
        tdKtdK: cute.Tensor,  # SM120: Register fragment for dK accumulator
        tdQtdQ: cute.Tensor,  # SM120: Register fragment for dQ accumulator
        pipeline_Q_consumer: PipelineConsumer,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem tensors (SM120: no Tensor Memory)
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dO.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        # SM120: Use shared memory for dS instead of Tensor Memory
        tdKrdS = tiled_mma_dK.make_fragment_A(sdS_smem)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS_smem)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        # SM120: Use shared memory for P instead of Tensor Memory
        tdVrP = tiled_mma_dV.make_fragment_A(sP)

        # SM120: Create shared memory store operations for register accumulators
        tidx = cute.arch.thread_idx()[0]
        thr_mma_S = tiled_mma_S.get_slice(0)
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)

        # Store operations for S
        smem_store_S_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        tScS = thr_mma_S.partition_C(sS)
        tiled_smem_store_S = cute.make_tiled_copy(smem_store_S_atom, tScS.layout)
        thr_smem_store_S = tiled_smem_store_S.get_slice(tidx)
        tSrS_reg = thr_smem_store_S.partition_S(tStS)
        tSsS = thr_smem_store_S.partition_D(tScS)

        # Store operations for dP
        smem_store_dP_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        tdPcdP = thr_mma_dP.partition_C(sdP)
        tiled_smem_store_dP = cute.make_tiled_copy(smem_store_dP_atom, tdPcdP.layout)
        thr_smem_store_dP = tiled_smem_store_dP.get_slice(tidx)
        tdPrdP_reg = thr_smem_store_dP.partition_S(tdPtdP)
        tdPsdP = thr_smem_store_dP.partition_D(tdPcdP)

        # Store operations for dV
        smem_store_dV_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        tdVcdV = thr_mma_dV.partition_C(sdV)
        tiled_smem_store_dV = cute.make_tiled_copy(smem_store_dV_atom, tdVcdV.layout)
        thr_smem_store_dV = tiled_smem_store_dV.get_slice(tidx)
        tdVrdV_reg = thr_smem_store_dV.partition_S(tdVtdV)
        tdVsdV = thr_smem_store_dV.partition_D(tdVcdV)

        # Store operations for dK
        smem_store_dK_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        tdKcdK = thr_mma_dK.partition_C(sdK)
        tiled_smem_store_dK = cute.make_tiled_copy(smem_store_dK_atom, tdKcdK.layout)
        thr_smem_store_dK = tiled_smem_store_dK.get_slice(tidx)
        tdKrdK_reg = thr_smem_store_dK.partition_S(tdKtdK)
        tdKsdK = thr_smem_store_dK.partition_D(tdKcdK)

        # Store operations for dQ
        smem_store_dQ_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        tdQcdQ = thr_mma_dQ.partition_C(sdQ)
        tiled_smem_store_dQ = cute.make_tiled_copy(smem_store_dQ_atom, tdQcdQ.layout)
        thr_smem_store_dQ = tiled_smem_store_dQ.get_slice(tidx)
        tdQrdQ_reg = thr_smem_store_dQ.partition_S(tdQtdQ)
        tdQsdQ = thr_smem_store_dQ.partition_D(tdQcdQ)

        # mma_qk_fn = partial(gemm_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, zero_init=True)
        mma_qk_fn = partial(
            gemm_ptx_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, sA=sK, sB=sQ, zero_init=True
        )
        # mma_dov_fn = partial(gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, zero_init=True)
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
        )
        # mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO)
        # SM120: Use shared memory for P instead of Tensor Memory
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=sP,  # SM120: P is in shared memory
            sB=sdO,
        )
        mma_dsk_fn = partial(gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, zero_init=True)
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True
        # )
        # mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ)
        # SM120: Use shared memory for dS instead of Tensor Memory
        mma_dsq_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
            sA=sdS_smem,  # SM120: dS is in shared memory
            sB=sQt,
        )

        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # producer_state_dKV = cutlass.pipeline.make_pipeline_state(
        #     cutlass.pipeline.PipelineUserType.Producer, 2
        # )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )

            accumulate_dK = False
            # -----------------------------------------------------------
            ###### Prologue
            # -----------------------------------------------------------
            # 1. S  = Q0 @ K.T
            # 2. dP = V @ dO.T
            # 3. dV = P @ dO

            # 1) S  = Q0 @ K.T
            handle_Q = pipeline_Q_consumer.wait_and_advance()
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
            mma_qk_fn(B_idx=handle_Q.index)
            # SM120: Store S accumulator from registers to shared memory
            cute.copy(thr_smem_store_S, tSrS_reg, tSsS)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            # Don't release Q yet
            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

            # 2) dP = V @ dO.T
            pipeline_dO.consumer_wait(consumer_state_dO)
            pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
            # dQ uses the same smem as dP (SM120: both use shared memory)
            pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
            mma_dov_fn(B_idx=consumer_state_dO.index)
            # SM120: Store dP accumulator from registers to shared memory
            cute.copy(thr_smem_store_dP, tdPrdP_reg, tdPsdP)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            # Don't release dO yet
            pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

            producer_phase_acc ^= 1
            # 3) dV = P.T @ dO
            # wait for P to be ready, which uses the same smem as S (SM120: both use shared memory)
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
            # SM120: Store dV accumulator from registers to shared memory
            cute.copy(thr_smem_store_dV, tdVrdV_reg, tdVsdV)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            pipeline_dO.consumer_release(consumer_state_dO)
            consumer_state_dO.advance()
            # -----------------------------------------------------------
            ###### MAIN LOOP
            # -----------------------------------------------------------
            # 1. S  = K    @ Q.T
            # 2. dQ = dS   @ K
            # 3. dK = dS.T @ Q
            # 4. dP = V    @ dO.T
            # 5. dV = P.T  @ dO

            for _ in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                # 1) S = K @ Q_i
                handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                # Don't need to wait for S, as P must have been ready ealier, i.e., S is ready
                mma_qk_fn(B_idx=handle_Q_next.index)
                # SM120: Store S accumulator from registers to shared memory
                cute.copy(thr_smem_store_S, tSrS_reg, tSsS)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                # 2) dK = dS.T @ Q
                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                # SM120: Store dK accumulator from registers to shared memory
                cute.copy(thr_smem_store_dK, tdKrdK_reg, tdKsdK)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                accumulate_dK = True
                handle_Q.release()

                # 3) dQ = dS @ K
                # dP uses the same smem as dQ (SM120: both use shared memory)
                # However, if dS is ready, then dP must have been ready, so we don't need to wait
                # pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                mma_dsk_fn()
                # SM120: Store dQ accumulator from registers to shared memory
                cute.copy(thr_smem_store_dQ, tdQrdQ_reg, tdQsdQ)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                # 4) dP = V @ dO.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                # dQ uses the same smem as dP (SM120: both use shared memory)
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                # SM120: Store dP accumulator from registers to shared memory
                cute.copy(thr_smem_store_dP, tdPrdP_reg, tdPsdP)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                producer_phase_acc ^= 1
                # 5) dV += P @ dO
                # wait for P to be ready, which uses the same smem as S (SM120: both use shared memory)
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                # SM120: Store dV accumulator from registers to shared memory
                cute.copy(thr_smem_store_dV, tdVrdV_reg, tdVsdV)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()

                handle_Q = handle_Q_next

            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

            # signal to the epilogue that dV is ready
            # pipeline_dKV.producer_acquire(producer_state_dKV)
            pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
            # pipeline_dKV.producer_commit(producer_state_dKV)
            pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
            # producer_state_dKV.advance()
            # pipeline_dKV.producer_acquire(producer_state_dKV)
            pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

            # -----------------------------------------------------------
            ###### Remaining 2
            # -----------------------------------------------------------
            # 1) dK += dS.T @ Q
            pipeline_dS.consumer_wait(consumer_state_dS)
            mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
            # SM120: Store dK accumulator from registers to shared memory
            cute.copy(thr_smem_store_dK, tdKrdK_reg, tdKsdK)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            # signal to the epilogue that dK is ready
            # pipeline_dKV.producer_commit(producer_state_dKV)
            pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
            # producer_state_dKV.advance()
            producer_phase_dKV ^= 1

            # 2) dQ = dS @ K
            # dS is done, so dP must have been ready, we don't need to wait
            mma_dsk_fn()
            # SM120: Store dQ accumulator from registers to shared memory
            cute.copy(thr_smem_store_dQ, tdQrdQ_reg, tdQsdQ)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
            # Wait until dQ is done before releasing Q, since K and Q0 uses the same mbarrier
            handle_Q.release()
            pipeline_dS.consumer_release(consumer_state_dS)
            consumer_state_dS.advance()

            producer_phase_acc ^= 1

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        # Currently it hangs if we have this S_P.producer_tail, will need to understand why
        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        sS: cute.Tensor,  # SM120: Shared memory for S accumulator
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdV: cute.Tensor,  # SM120: Shared memory for dV accumulator
        sdK: cute.Tensor,  # SM120: Shared memory for dK accumulator
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        sdP: cute.Tensor,  # SM120: Shared memory for dP accumulator
        sP: cute.Tensor,  # SM120: Shared memory for P
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV_epi: Optional[cute.Tensor],
        sdK_epi: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        if const_expr(True):
            sLSE_2D = utils.transpose_view(sLSE_2D)
            sdPsum_2D = utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.mma_tiler_kq[0] // 32 * self.v_dtype.width  # (128, 64)
        # SM120: Create shared memory partitions instead of Tensor Memory
        tScS = thr_mma_S.partition_C(sS)  # Partition shared memory S
        tScP = thr_mma_S.partition_C(sP)  # Partition shared memory P
        tdPcdP = thr_mma_dP.partition_C(sdP)  # Partition shared memory dP
        tdPcdS = thr_mma_dP.partition_C(sdS)  # Partition shared memory dS (overlaps with dP)

        # SM120: Create shared memory load/store operations instead of Tensor Memory
        smem_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )

        # smem -> rmem
        thr_copy_s2r = cute.make_tiled_copy(smem_load_atom, tScS.layout).get_slice(tidx)
        tScS_s2r = thr_copy_s2r.partition_S(tScS)  # Source: shared memory S
        tScP_s2r = thr_copy_s2r.partition_S(tScP)  # Source: shared memory P
        tdPcdP_s2r = thr_copy_s2r.partition_S(tdPcdP)  # Source: shared memory dP
        tdPcdS_s2r = thr_copy_s2r.partition_S(tdPcdS)  # Source: shared memory dS
        
        tScS_t2r = thr_copy_s2r.partition_D(tScS)  # Destination: register fragment
        t0ScS_t2r = thr_copy_s2r.get_slice(0).partition_D(tScS)  # For masking
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_s2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_s2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        
        # rmem -> smem
        thr_copy_r2s = cute.make_tiled_copy(smem_store_atom, tScP.layout).get_slice(tidx)
        tScP_r2s = thr_copy_r2s.partition_S(tScP)  # Source: register fragment
        tScP_r2s_dst = thr_copy_r2s.partition_D(tScP)  # Destination: shared memory P
        tdPcdS_r2s = thr_copy_r2s.partition_S(tdPcdS)  # Source: register fragment
        tdPcdS_r2s_dst = thr_copy_r2s.partition_D(tdPcdS)  # Destination: shared memory dS
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_s2r
        )
        thr_copy_r2s_dS = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_s2r).get_slice(tidx)
        # We assume the swizzle (i.e. layout.inner) stays the same
        sdS_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        ).outer  # ((8,16), (64,2), (1, 1))
        sdS_layout = cute.slice_(sdS_layout, (None, None, 0))  # ((8,16), (64,2))
        # Need to group into 1 mode to be compatible w thr_copy_r2s
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s_dS.partition_D(sdS_epi)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        # consumer_state_dPsum = cutlass.pipeline.make_pipeline_state(
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
            )

            # prefetch_LSE = not self.is_causal
            prefetch_LSE = False

            # Mainloop
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                # Prefetch 1 stage of LSE
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_fragment(tScS_t2r[None, 0, 0, 0].shape, Float32)
                if const_expr(prefetch_LSE and not self.shuffle_LSE):
                    cute.autovec_copy(tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r)

                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                #### SMEM->RMEM (Load S from shared memory)
                tSrS_t2r = cute.make_fragment(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_s2r, tScS_s2r, tSrS_t2r)

                #### APPLY MASK
                mask_fn(tSrS_t2r, m_block=m_block)

                num_stages = cute.size(tScS_t2r, mode=[1])

                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------
                lane_idx = cute.arch.lane_idx()
                tSrP_r2s_f32 = cute.make_fragment(tScP_r2s_dst[None, 0, 0, 0].shape, Float32)  # 64
                tSrP_r2s = cute.recast_tensor(tSrP_r2s_f32, self.q_dtype)
                for stage in cutlass.range_constexpr(num_stages):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    if const_expr(not self.shuffle_LSE):
                        if const_expr(stage > 0 or not prefetch_LSE):
                            cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                        tSrLSE = tSrLSE_s2r
                    else:
                        tSrLSE = tSsLSE_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                        if const_expr(not self.shuffle_LSE):
                            lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                        else:
                            lse_pair = (
                                utils.shuffle_sync(tSrLSE, offset=2 * v),
                                utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                            )
                        tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = utils.fma_packed_f32x2(
                            ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                            (softmax_scale_log2, softmax_scale_log2),
                            (-lse_pair[0], -lse_pair[1]),
                        )
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                    utils.cvt_f16(tSrS_cur, tSrP_r2s[None, stage, 0, 0])
                    if const_expr(stage == 0):
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                        )
                        # Without this barrier, we could have 1 warp writing to P in smem while
                        # another warp is still reading S from smem.
                        self.compute_sync_barrier.arrive_and_wait()
                    # SM120: Store P to shared memory instead of Tensor Memory
                    cute.copy(
                        thr_copy_r2s,
                        tSrP_r2s_f32[None, stage, None, None],
                        tScP_r2s_dst[None, stage, None, None],
                    )

                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )

                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                    # pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                pipeline_LSE.consumer_release(consumer_state_LSE)
                # consumer_state_S_P_dP.advance()
                consumer_state_LSE.advance()

                # ---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                # ---------------------------------------------
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)

                pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
                consumer_state_S_P_dP.advance()
                # consumer_phase_S_P_dP ^= 1

                ##### dS.T = P.T * (dP.T - Psum)
                for stage in cutlass.range_constexpr(num_stages):
                    tdPrdP_t2r = cute.make_fragment(tScS_t2r[None, 0, None, None].shape, Float32)
                    # SM120: Load dP from shared memory instead of Tensor Memory
                    cute.copy(thr_copy_s2r, tdPcdP_s2r[None, stage, None, None], tdPrdP_t2r)
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                    )
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                    if const_expr(not self.shuffle_dPsum):
                        tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                        cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    else:
                        tSrdPsum = tSsdPsum_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                        if const_expr(not self.shuffle_dPsum):
                            dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                        else:
                            dPsum_pair = (
                                utils.shuffle_sync(tSrdPsum, offset=2 * v),
                                utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                            )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = utils.sub_packed_f32x2(
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                        )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = utils.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        )
                    tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                    utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)
                    cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                    # SM120: Store dS to shared memory instead of Tensor Memory
                    tdPrdS_r2s = cute.make_tensor(
                        tdPrdS_cvt.iterator,
                        tdPcdS_r2s_dst[None, stage, 0, 0].shape
                    )
                    tdPrdS_r2s_f32 = cute.recast_tensor(tdPrdS_r2s, Float32)
                    cute.copy(thr_copy_r2s, tdPrdS_r2s_f32, tdPcdS_r2s_dst[None, stage, 0, 0])

                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )

                cute.arch.sync_warp()
                # with cute.arch.elect_one():
                # The mma warp no longer waits for dP (it waits for dS), so we don't have to arrive
                # pipeline_dP.sync_object_empty.arrive(0, pipeline_dP.consumer_mask)
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()

                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

            if const_expr(not self.use_tma_store):
                consumer_state_dKV = self.epilogue_dKV(
                    dp_idx,
                    warp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dV,
                    thr_mma_dK,
                    sdV,  # SM120: Shared memory for dV accumulator
                    sdK,  # SM120: Shared memory for dK accumulator
                    mdV,
                    mdK,
                    pipeline_dKV,
                    consumer_state_dKV,
                    softmax_scale,
                )
            else:
                thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                #### STORE dV
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dV,
                    sdV,  # SM120: Shared memory for dV accumulator
                    mdV_tma_tensor,
                    sdV_epi,
                    tma_atom_dV,
                    thr_copy_r2s_dKV,
                    pipeline_dKV,
                    consumer_state_dKV,
                    None,  # Don't scale
                    int(NamedBarrierBwdSm120.EpilogueWG1),  # barrier_id
                    mdV_semaphore,
                )
                #### STORE dK
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dK,
                    sdK,  # SM120: Shared memory for dK accumulator
                    mdK_tma_tensor,
                    sdK_epi,
                    tma_atom_dK,
                    thr_copy_r2s_dKV,
                    pipeline_dKV,
                    consumer_state_dKV,
                    softmax_scale if const_expr(self.qhead_per_kvhead == 1) else None,
                    int(NamedBarrierBwdSm120.EpilogueWG1),  # barrier_id
                    mdK_semaphore,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.core.ThrMma,
        sdQ: cute.Tensor,  # SM120: Shared memory for dQ accumulator instead of Tensor Memory
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        # SM120: Load dQ from shared memory instead of Tensor Memory
        smem_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        tdQcdQ = thr_mma_dQ.partition_C(sdQ)  # SM120: Partition shared memory dQ
        tiled_smem_ld = cute.make_tiled_copy(smem_load_atom, tdQcdQ.layout)
        thr_copy_s2r = tiled_smem_ld.get_slice(tidx)
        tdQcdQ_s2r = thr_copy_s2r.partition_S(tdQcdQ)
        tdQcdQ_reg = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_s2r.partition_D(tdQcdQ_reg).shape
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == self.dQaccum_reduce_stage, (
            "dQaccum reduce stage mismatch"
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
            )
            mdQ_semaphore_cur = None
            if const_expr(self.deterministic):
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                pipeline_dQ.consumer_wait(dQ_consumer_state)
                # SMEM -> RMEM
                tdQrdQ_t2r = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_s2r, tdQcdQ_s2r, tdQrdQ_t2r)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                # semaphore acquire
                if const_expr(self.deterministic):
                    barrier.wait_eq(mdQ_semaphore_cur[m_block, None].iterator, tidx, 0, n_block)
                    self.reduce_sync_barrier.arrive_and_wait()

                gdQaccum_cur = gdQaccum[None, None, m_block]

                for stage in cutlass.range_constexpr(cute.size(tdQrdQ_t2r, mode=[1])):  # 4
                    smem_idx = dQ_tma_store_producer_state.index
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                    tdQrdQ_r2s = cute.make_tensor(
                        tdQrdQ_t2r[None, stage, None, None].iterator, tdQsdQ_r2s.shape
                    )
                    cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                    )
                    self.reduce_sync_barrier.arrive_and_wait()
                    # Copy from shared memory to global memory
                    if is_tma_warp:
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQaccum[None, smem_idx].iterator,
                                gdQaccum_cur[None, stage].iterator,
                                self.tma_copy_bytes["dQ"] // 1,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=read_flag)
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()
                    # Directly add to gmem, much slower
                    # tdQgdQ = thr_copy_dQaccum_r2s.partition_D(gdQaccum[None, stage, m_block])
                    # assert cute.size(tdQrdQ_r2s) == cute.size(tdQgdQ)
                    # for i in cutlass.range(cute.size(tdQrdQ_r2s) // 4, unroll_full=True):
                    #     copy_utils.atomic_add_fp32x4(
                    #         tdQrdQ_r2s[4 * i],
                    #         tdQrdQ_r2s[4 * i + 1],
                    #         tdQrdQ_r2s[4 * i + 2],
                    #         tdQrdQ_r2s[4 * i + 3],
                    #         utils.elem_pointer(tdQgdQ, 4 * i),
                    #     )

                # semaphore release
                # NOTE: arrive_inc calls red_release which issues membar
                if const_expr(self.deterministic):
                    if tidx == 0:
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                    self.reduce_sync_barrier.arrive_and_wait()
                    barrier.arrive_inc(mdQ_semaphore_cur[m_block, None].iterator, tidx, 0, 1)

            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        sdV: cute.Tensor,  # SM120: Shared memory for dV accumulator
        sdK: cute.Tensor,  # SM120: Shared memory for dK accumulator
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
    ):
        wg_idx = (
            cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = mdV[None, None, head_idx, batch_idx]
        mdK_cur = mdK[None, None, head_idx, batch_idx]

        # SM120: Create shared memory load operations instead of Tensor Memory
        smem_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )

        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # SM120: Load dV from shared memory
        tdVcdV = thr_mma_dV.partition_C(sdV)
        tiled_smem_ld_dV = cute.make_tiled_copy(smem_load_atom, tdVcdV.layout)
        thr_smem_ld_dV = tiled_smem_ld_dV.get_slice(tidx)

        tdVcdV_s2r_p = thr_smem_ld_dV.partition_S(tdVcdV)
        tdVcdV_s2r = self.split_wg(tdVcdV_s2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV_reg = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV_reg.iterator, tdVcdV_reg.layout)

        tdVcdV_t2r_p = thr_smem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_fragment(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_smem_ld_dV, tdVcdV_s2r, tdVrdV_t2r)
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tdVcdV_reg.layout,
            tiler_mn=(self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]),
        )

        tdVrdV_r2s = cute.make_fragment(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.tile_m, self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, n_block]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_smem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # SM120: Load dK from shared memory
        tdKcdK = thr_mma_dK.partition_C(sdK)
        tiled_smem_ld_dK = cute.make_tiled_copy(smem_load_atom, tdKcdK.layout)
        thr_smem_ld_dK = tiled_smem_ld_dK.get_slice(tidx)

        tdKcdK_s2r_p = thr_smem_ld_dK.partition_S(tdKcdK)
        tdKcdK_s2r = self.split_wg(tdKcdK_s2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK_reg = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK_reg.iterator, tdKcdK_reg.layout)

        tdKcdK_t2r_p = thr_smem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_fragment(tdKcdK_t2r.shape, Float32)

        cute.copy(thr_smem_ld_dK, tdKcdK_s2r, tdKrdK_t2r)
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tdKcdK_reg.layout,
            tiler_mn=(self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]),
        )

        tdKrdK_r2s = cute.make_fragment(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdimv), (None, 0))
        gdK_tile = gdK[None, None, n_block]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_smem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dK, tdKrdK_r2s, tdKgdK_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma: cute.core.ThrMma,
        sdKV: cute.Tensor,  # SM120: Shared memory for dK or dV accumulator
        mdKV: cute.Tensor,
        sdKV_epi: cute.Tensor,  # SM120: Shared memory for epilogue staging
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
    ) -> cutlass.pipeline.PipelineState:
        # assumes mma_tiler_pdo = mma_tiler_dsq = (tile_n, head_dim)
        # head_dim = head_dim_v, dk_dtype = dv_dtype
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        if const_expr(self.qhead_per_kvhead == 1):
            sdKV_epi = sdKV_epi[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV_epi = sdKV_epi[None, wg_idx] # (tile_n * 32) for fp32
        
        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV_epi)

        head_idx_kv = head_idx // self.qhead_per_kvhead
        if const_expr(self.qhead_per_kvhead == 1):
            mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx] # (seqlen, hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n, self.tile_hdim), (n_block, 0)
            ) # (tile_n, hdim)
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg) # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(
                gdKV, self.sdKV_epi_tile, (0, None)
            ) # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            mdKV_cur = mdKV[None, head_idx_kv, batch_idx] # (seqlen * hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n * self.tile_hdim, ), (n_block, )
            ) # (tile_n * hdim)
            gdKV = cute.logical_divide(
                gdKV_p, (self.tile_n * self.tile_hdim // num_wg, )
            )[((None, wg_idx), )] # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(
                gdKV, (self.sdKV_flat_epi_tile, )
            ) # (tile_n * hdim / 2 / epi_stage, epi_stage)

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

        if const_expr(self.qhead_per_kvhead == 1):
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV_epi, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            ) # (TMA) and (TMA, EPI_STAGE)
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            assert num_epi_stages == self.num_epi_stages, "Epi stage calculation is wrong"
        else:
            num_epi_stages = self.num_epi_stages

        # SM120: Create shared memory load operations instead of Tensor Memory
        smem_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )

        read_flag = const_expr(not self.deterministic)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # semaphore acquire
        if const_expr(self.deterministic):
            barrier.wait_eq(
                mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # SMEM -> RMEM -- setup
            # SM120: Load from shared memory accumulator
            tdKVcdKV = thr_mma.partition_C(sdKV)
            tiled_smem_ld = cute.make_tiled_copy(smem_load_atom, tdKVcdKV.layout)
            thr_copy_s2r = tiled_smem_ld.get_slice(tidx)
            
            tdKVcdKV_s2r_p = thr_copy_s2r.partition_S(tdKVcdKV)
            tdKVcdKV_s2r = self.split_wg(tdKVcdKV_s2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_s2r = tdKVcdKV_s2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tdKVcdKV_reg = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_s2r.partition_D(tdKVcdKV_reg)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_fragment(tdKVcdKV_t2r.shape, Float32)

            # SMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_s2r, tdKVcdKV_s2r, tdKVrdKV_t2r)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = utils.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_fragment(tdKVrdKV_t2r.shape, self.dv_dtype) # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(self.dv_dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(self.qhead_per_kvhead == 1):
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                else:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV_epi.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            self.tma_copy_bytes["dKacc"],
                        )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(self.deterministic):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
