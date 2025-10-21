# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
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
from flash_attn.cute.blackwell_helpers import gemm_w_idx, gemm_ptx_w_idx  # noqa
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    ParamsBase,
)

from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwdSm100


class FlashAttentionBackwardSm100:
    arch = 100

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

        self.kq_acc_dtype = self.vdo_acc_dtype = self.pdo_acc_dtype = self.dsq_acc_dtype = (
            self.dsk_acc_dtype
        ) = Float32

        self.cluster_shape_mn = (1, 1)
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

        # TMEM setup
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.tmem_S_offset = 0
        self.tmem_P_offset = 0  # overlap with S
        self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dQ_offset = self.tmem_dP_offset  # overlap with dP
        self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m

        self.num_regs_reduce = 160
        self.num_regs_compute = 128
        self.num_regs_other = 80
        self.num_regs_empty = 24
        assert self.num_regs_reduce + self.num_regs_compute * 2 + self.num_regs_other <= 512

        self.buffer_align_bytes = 1024

        self.num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)

    def _setup_attributes(self):
        self.Q_stage = 2
        self.k_stage = self.v_stage = 1
        self.dO_stage = 1
        self.LSE_stage = 1
        self.sdQaccum_stage = 2
        self.dPsum_stage = 1
        self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        self.dQaccum_reduce_stage = self.tile_hdim // 32

    def _get_tiled_mma(self):
        cta_group = tcgen05.CtaGroup.ONE
        # S = K @ Q.T, dP = V @ dO.T
        tiled_mma_SdP = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.kq_acc_dtype,
            cta_group,
            self.mma_tiler_kq[:2],
        )
        # dV += P @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.pdo_acc_dtype,
            cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.pdo_acc_dtype,
            cta_group,
            self.mma_tiler_dsq[:2],
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.dsk_acc_dtype,
            cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S = K @ Q.T
        self.sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_SdP,
            self.mma_tiler_kq,
            self.k_dtype,
            self.k_stage,
        )
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_SdP,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dV += P @ dO
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dP = V @ dO.T
        self.sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_SdP,
            self.mma_tiler_vdo,
            self.v_dtype,
            self.v_stage,
        )
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_SdP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        self.sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQaccum = dS @ K
        self.sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.q_dtype,
            1,
        )
        self.sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            self.k_stage,
        )

        self.sdQaccum_layout = cute.make_layout((self.tile_m * 32, self.sdQaccum_stage))
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.LSE_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dPsum_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )

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
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
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

        layout_transpose = [1, 3, 2, 0]  # (b, s, n, h) --> (s, h, n, b)
        mQ, mK, mV, mdO, mdK, mdV = [
            utils.select(t, mode=layout_transpose) for t in (mQ, mK, mV, mdO, mdK, mdV)
        ]
        LSE_dPsum_dQaccum_transpose = [2, 1, 0]  # (b, n, s) --> (s, n, b)
        mLSE, mdPsum, mdQaccum = [
            utils.select(t, mode=LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]
        dO_transpose = [1, 0, 2, 3]
        mdO = utils.select(mdO, mode=dO_transpose)

        semaphore_transpose = [2, 3, 1, 0]  # (b, n, block, stage) -> (block, stage, n, b)
        mdQ_semaphore = None
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
        self.tiled_mma_SdP, self.tiled_mma_dK, self.tiled_mma_dV, self.tiled_mma_dQ = (
            self._get_tiled_mma()
        )
        self._setup_smem_layout()

        cta_group = tcgen05.CtaGroup.ONE

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_SdP.thr_id.shape,),
        )

        self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
        self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
        dK_major_mode = self.mdK_layout_enum.mma_major_mode()
        dV_major_mode = self.mdV_layout_enum.mma_major_mode()
        if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mdK is wrong")
        if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mdV is wrong")
        self.sdKV_epi_tile = (
            self.tile_n,
            128 // (self.dk_dtype.width // 8),
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        sdKV_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dk_dtype,
            self.mdK_layout_enum,
            self.sdKV_epi_tile,
            self.sdKVaccum_stage,
        )

        if const_expr(self.use_tma_store):
            if const_expr(self.dk_dtype.width == 32):
                tma_copy_op_dKV = cpasync.CopyReduceBulkTensorTileS2GOp()
            else:
                tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()

            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(sdKV_layout, mode=[0, 1]),
                self.sdKV_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(sdKV_layout, mode=[0, 1]),
                self.sdKV_epi_tile,
                1,  # no mcast
            )
        else:
            assert self.qhead_per_kvhead == 1, "Must use TMA reduce add for GQA"
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        thr_layout_r2s_dKV = cute.make_ordered_layout((self.tile_n, 1), order=(1, 0))  # 128 threads
        val_layout_r2s_dKV = cute.make_ordered_layout(
            (1, 128 // self.dk_dtype.width), order=(1, 0)
        )  # 4 or 8 vals for 16 byte store
        r2s_copy_atom_r2s_dKV = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=128,
        )
        tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
            r2s_copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        # S = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_SdP,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_SdP,
            self.cluster_layout_vmnk.shape,
        )
        # dV += P @ dO
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        # dP = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_SdP,
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
        self.tma_copy_bytes["dQ"] = self.tile_m * 32 * Float32.width // 8

        TileScheduler = SingleTileScheduler
        # TODO -- optimizer scheduler for causal
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
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
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.LSE_stage]
            LSE_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.LSE_stage]
            dPsum_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dPsum_stage]
            dPsum_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dPsum_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            dQaccum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]

            # TMEM
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]

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
            # tma_atom_Psum,
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
            sdKV_layout,
            self.tiled_mma_SdP,
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
        sdKV_layout: cute.ComposedLayout,
        tiled_mma_SdP: cute.TiledMma,
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

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        LSE_full_mbar_ptr = storage.LSE_full_mbar_ptr.data_ptr()
        LSE_empty_mbar_ptr = storage.LSE_empty_mbar_ptr.data_ptr()
        dPsum_full_mbar_ptr = storage.dPsum_full_mbar_ptr.data_ptr()
        dPsum_empty_mbar_ptr = storage.dPsum_empty_mbar_ptr.data_ptr()

        if warp_idx == 1:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE * len(self.compute_warp_ids)
            )
        if warp_idx == 2:
            cute.arch.mbarrier_init(LSE_full_mbar_ptr, 1)
            cute.arch.mbarrier_init(LSE_empty_mbar_ptr, len(self.compute_warp_ids))
        if warp_idx == 3:
            cute.arch.mbarrier_init(dPsum_full_mbar_ptr, 1)
            cute.arch.mbarrier_init(dPsum_empty_mbar_ptr, len(self.compute_warp_ids))

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            init_wait=False,
        )
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            init_wait=False,
        )

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
            barrier_storage=storage.dQaccum_mbar_ptr.data_ptr(),
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

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQt = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, swizzle_=sQt_layout.inner), sQt_layout.outer
        )

        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sKt = cute.make_tensor(
            cute.recast_ptr(sK.iterator, swizzle_=sKt_layout.inner), sKt_layout.outer
        )

        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)

        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(
            cute.recast_ptr(sdSt.iterator, swizzle_=sdS_layout.inner), sdS_layout.outer
        )

        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sdOt = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, swizzle_=sdOt_layout.inner), sdOt_layout.outer
        )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)

        sdV = storage.sdO.get_tensor(
            sdKV_layout.outer, swizzle=sdKV_layout.inner, dtype=self.dv_dtype
        )
        sdK = storage.sQ.get_tensor(
            sdKV_layout.outer, swizzle=sdKV_layout.inner, dtype=self.dk_dtype
        )

        assert cute.cosize(sdV) * self.dv_dtype.width <= cute.cosize(sdO) * self.do_dtype.width, (
            "Not enough space for sdV"
        )
        assert cute.cosize(sdK) * self.dk_dtype.width <= cute.cosize(sQ) * self.q_dtype.width, (
            "Not enough space for sdK"
        )

        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # TMEM
        # S
        thr_mma_SdP = tiled_mma_SdP.get_slice(0)
        Sacc_shape = thr_mma_SdP.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_SdP.make_fragment_C(Sacc_shape)
        tStS = cute.make_tensor(tStS.iterator, tStS.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tdVtdV.iterator + self.tmem_dV_offset, tdVtdV.layout)
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tdKtdK.iterator + self.tmem_dK_offset, tdKtdK.layout)
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator + self.tmem_dQ_offset, tdQtdQ.layout)
        # dP
        dPacc_shape = thr_mma_SdP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_SdP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_dP_offset, tdPtdP.layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK,
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
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_SdP,
                thr_mma_dV,
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
                LSE_full_mbar_ptr,
                LSE_empty_mbar_ptr,
                dPsum_full_mbar_ptr,
                dPsum_empty_mbar_ptr,
                pipeline_dO,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            cute.arch.sync_warp()

            self.mma(
                tiled_mma_SdP,
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
                tStS,
                tdVtdV,
                tdKtdK,
                tdPtdP,
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
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
            )

            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, is_two_cta=False)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_compute)  # 8 warps
            self.compute_loop(
                thr_mma_SdP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                sLSE,
                sdPsum,
                tdVtdV,
                tdKtdK,
                mdV,
                mdK,
                sdSt,
                sdS,
                tdPtdP,
                LSE_full_mbar_ptr,
                LSE_empty_mbar_ptr,
                dPsum_full_mbar_ptr,
                dPsum_empty_mbar_ptr,
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
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)

            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
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
        thr_mma_SdP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
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
        LSE_full_mbar_ptr: cute.Pointer,
        LSE_empty_mbar_ptr: cute.Pointer,
        dPsum_full_mbar_ptr: cute.Pointer,
        dPsum_empty_mbar_ptr: cute.Pointer,
        pipeline_dO: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        producer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mQ_cur = mQ[None, None, head_idx, batch_idx]
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            mPsum_cur = mdPsum[None, head_idx, batch_idx]

            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
            tSgK = thr_mma_SdP.partition_A(gK)
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
            tdPgV = thr_mma_SdP.partition_A(gV)
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_SdP.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_n,), (None,))
            gdPsum = cute.local_tile(mPsum_cur, (self.tile_n,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)

            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), tSgK, sK[None, None, None, 0], single_stage=True
            )
            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV[None, None, None, 0],
                single_stage=True,
            )
            load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ)
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO, 0, cute.make_layout(1), tdVgdO, sdO
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
            copy_atom_stats = cute.make_copy_atom(
                cpasync.CopyBulkG2SOp(), Float32, num_bits_per_copy=self.tma_copy_bytes["LSE"] * 8
            )
            copy_stats = partial(cute.copy, copy_atom_stats)

            # First iteration: load K together w Q & LSE, then V together w dO & dPsum
            # K & Q
            pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
            load_Q(m_block_min, producer_state=producer_state_Q)
            pipeline_Q.producer_commit(producer_state_Q)
            producer_state_Q.advance()
            # LSE
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    LSE_full_mbar_ptr, self.tma_copy_bytes["LSE"]
                )
                copy_stats(gLSE[None, m_block_min], sLSE[None, 0], mbar_ptr=LSE_full_mbar_ptr)
            # V & dO
            pipeline_dO.producer_acquire(producer_state_dO, extra_tx_count=self.tma_copy_bytes["V"])
            load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO))
            load_dO(m_block_min, producer_state=producer_state_dO)
            pipeline_dO.producer_commit(producer_state_dO)
            producer_state_dO.advance()
            # dPsum
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    dPsum_full_mbar_ptr, self.tma_copy_bytes["dPsum"]
                )
                copy_stats(gdPsum[None, m_block_min], sdPsum[None, 0], mbar_ptr=dPsum_full_mbar_ptr)

            lse_empty_consumer_phase = cute.Int32(0)
            dpsum_empty_consumer_phase = cute.Int32(0)

            for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                # Q
                pipeline_Q.producer_acquire(producer_state_Q)
                load_Q(m_block, producer_state=producer_state_Q)
                pipeline_Q.producer_commit(producer_state_Q)
                producer_state_Q.advance()
                # LSE
                cute.arch.mbarrier_wait(LSE_empty_mbar_ptr, lse_empty_consumer_phase)
                lse_empty_consumer_phase ^= 1
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        LSE_full_mbar_ptr, self.tma_copy_bytes["LSE"]
                    )
                    copy_stats(gLSE[None, m_block], sLSE[None, 0], mbar_ptr=LSE_full_mbar_ptr)
                # dO
                pipeline_dO.producer_acquire(producer_state_dO)
                load_dO(m_block, producer_state=producer_state_dO)
                pipeline_dO.producer_commit(producer_state_dO)
                producer_state_dO.advance()
                # dPsum
                cute.arch.mbarrier_wait(dPsum_empty_mbar_ptr, dpsum_empty_consumer_phase)
                dpsum_empty_consumer_phase ^= 1
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        dPsum_full_mbar_ptr, self.tma_copy_bytes["dPsum"]
                    )
                    copy_stats(gdPsum[None, m_block], sdPsum[None, 0], mbar_ptr=dPsum_full_mbar_ptr)

            pipeline_Q.producer_tail(producer_state_Q)
            pipeline_dO.producer_tail(producer_state_dO)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
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
        tStS: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdQtdQ: cute.Tensor,
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
        thr_mma_SdP = tiled_mma_SdP.get_slice(0)
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        # Partition smem / tmem tensors
        # S = K @ Q.T
        tSrK = thr_mma_SdP.make_fragment_A(sK)
        tSrQ = thr_mma_SdP.make_fragment_B(sQ)
        # dP = V @ dO.T
        tdPrV = thr_mma_SdP.make_fragment_A(sV)
        tdPrdOt = thr_mma_SdP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        tdKrdS = thr_mma_dK.make_fragment_A(sdSt)
        tdKrQ = thr_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = thr_mma_dQ.make_fragment_A(sdS)
        tdQrK = thr_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = thr_mma_dV.make_fragment_B(sdO)
        p_tmem_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_dV,
            self.mma_tiler_pdo,
            self.q_dtype,
            1,
        )
        tP = cute.make_tensor(tStS.iterator, p_tmem_layout.outer)
        tdVrP = thr_mma_dV.make_fragment_A(tP)[None, None, None, 0]
        tdVrP = cute.make_tensor(tdVrP.iterator, tdVrP.layout)

        mma_qk_fn = partial(gemm_w_idx, tiled_mma_SdP, tStS, tSrK, tSrQ, A_idx=0, zero_init=True)
        # mma_qk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_SdP, tStS, tSrK, tSrQ, sA=sK, sB=sQ, A_idx=0, zero_init=True
        # )
        mma_dov_fn = partial(
            gemm_w_idx, tiled_mma_SdP, tdPtdP, tdPrV, tdPrdOt, A_idx=0, zero_init=True
        )
        # mma_dov_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_SdP, tdPtdP, tdPrV, tdPrdOt, sA=sV, sB=sdOt, A_idx=0, zero_init=True
        # )
        mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO, A_idx=None)
        # mma_pdo_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO, sA=None, sB=sdO, A_idx=None
        # )
        mma_dsk_fn = partial(
            gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, A_idx=0, B_idx=0, zero_init=True
        )
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, A_idx=0, B_idx=0, zero_init=True
        # )
        mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ, A_idx=0)
        # mma_dsq_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ, sA=sdSt, sB=sQt, A_idx=0
        # )

        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        # producer_state_S_P = cutlass.pipeline.make_pipeline_state(
        #     cutlass.pipeline.PipelineUserType.Producer, 1
        # )
        producer_phase_S_P = Int32(1)
        # producer_state_dP = cutlass.pipeline.make_pipeline_state(
        #     cutlass.pipeline.PipelineUserType.Producer, 1
        # )
        producer_phase_dP = Int32(1)
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # producer_state_dQ = cutlass.pipeline.make_pipeline_state(
        #     cutlass.pipeline.PipelineUserType.Producer, 1
        # )
        producer_phase_dQ = Int32(1)
        # producer_state_dKV = cutlass.pipeline.make_pipeline_state(
        #     cutlass.pipeline.PipelineUserType.Producer, 2
        # )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            accumulate_dK = False
            # -----------------------------------------------------------
            ###### Prologue
            # -----------------------------------------------------------
            # 1. S  = Q0 @ K.T
            # 2. dP = V @ dO.T
            # 3. dV = P @ dO

            # 1) S  = Q0 @ K.T
            handle_Q = pipeline_Q_consumer.wait_and_advance()
            # pipeline_S_P.producer_acquire(producer_state_S_P)
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_S_P)
            mma_qk_fn(B_idx=handle_Q.index)
            # Don't release Q yet
            # pipeline_S_P.producer_commit(producer_state_S_P)
            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
            # producer_state_S_P.advance()
            producer_phase_S_P ^= 1

            # 2) dP = V @ dO.T
            pipeline_dO.consumer_wait(consumer_state_dO)
            # pipeline_dP.producer_acquire(producer_state_dP)
            pipeline_dP.sync_object_empty.wait(0, producer_phase_dP)
            # pipeline_dQ.producer_acquire(producer_state_dQ)  # dQ uses the same tmem as dP
            pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
            mma_dov_fn(B_idx=consumer_state_dO.index)
            # Don't release dO yet
            # pipeline_dP.producer_commit(producer_state_dP)
            pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
            # producer_state_dP.advance()
            producer_phase_dP ^= 1

            # 3) dV = P.T @ dO
            # wait for P to be ready, which uses the same tmem as S
            # pipeline_S_P.producer_acquire(producer_state_S_P)
            pipeline_S_P.sync_object_empty.wait(0, producer_phase_S_P)
            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
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
                # pipeline_S_P.producer_commit(producer_state_S_P)
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                # producer_state_S_P.advance()
                producer_phase_S_P ^= 1

                # 2) dQ = dS @ K
                pipeline_dS.consumer_wait(consumer_state_dS)
                # pipeline_dP.producer_acquire(producer_state_dP)  # dP uses the same tmem as dQ
                pipeline_dP.sync_object_empty.wait(0, producer_phase_dP)
                mma_dsk_fn()
                # pipeline_dQ.producer_commit(producer_state_dQ)
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                # producer_state_dQ.advance()
                producer_phase_dQ ^= 1

                # 3) dK = dS.T @ Q
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                accumulate_dK = True
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                # 4) dP = V @ dO.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                # pipeline_dQ.producer_acquire(producer_state_dQ)  # dQ uses the same tmem as dP
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                # pipeline_dP.producer_commit(producer_state_dP)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                # producer_state_dP.advance()
                producer_phase_dP ^= 1

                # 5) dV += P @ dO
                # wait for P to be ready, which uses the same tmem as S
                # pipeline_S_P.producer_acquire(producer_state_S_P)
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_S_P)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()

                handle_Q = handle_Q_next

            # pipeline_S_P.producer_commit(producer_state_S_P)
            pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
            # producer_state_S_P.advance()
            producer_phase_S_P ^= 1

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
            # signal to the epilogue that dK is ready
            # pipeline_dKV.producer_commit(producer_state_dKV)
            pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
            # producer_state_dKV.advance()
            producer_phase_dKV ^= 1

            # 2) dQ = dS @ K
            mma_dsk_fn()
            # pipeline_dQ.producer_commit(producer_state_dQ)
            pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
            # producer_state_dQ.advance()
            producer_phase_dQ ^= 1
            # Wait until dQ is done before releasing Q, since K and Q0 uses the same mbarrier
            handle_Q.release()
            pipeline_dS.consumer_release(consumer_state_dS)
            consumer_state_dS.advance()

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
        thr_tensor: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[cutlass.Int32],
    ):
        reduced_shape = cute.product_each(thr_tensor.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for thr_tensor in split_wg"
            t = cute.logical_divide(thr_tensor, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for thr_tensor in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    thr_tensor, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    thr_tensor,
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
        thr_mma_SdP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tStS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        tdPtdP: cute.Tensor,
        LSE_full_mbar_ptr: cute.Pointer,
        LSE_empty_mbar_ptr: cute.Pointer,
        dPsum_full_mbar_ptr: cute.Pointer,
        dPsum_empty_mbar_ptr: cute.Pointer,
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
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
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
                (self.tile_m, self.tile_n, self.LSE_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dPsum_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        if const_expr(True):
            sLSE_2D = utils.transpose_view(sLSE_2D)
            sdPsum_2D = utils.transpose_view(sdPsum_2D)
        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % 128  # 0...128
        wg_idx = (
            cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        wg_idx = cute.arch.make_warp_uniform(wg_idx)
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128  # 2

        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.mma_tiler_kq[0] // 32 * self.v_dtype.width  # (128, 64)
        tStP = cute.composition(tStS, cute.make_layout((self.tile_m, tileP_f32_like)))
        tScS = thr_mma_SdP.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, cute.make_layout((self.tile_m, tileP_f32_like)))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS).get_slice(tidx)
        tStS_t2r_p = thr_tmem_load.partition_S(tStS)
        tStS_t2r = self.split_wg(tStS_t2r_p, wg_idx, num_wg)
        tdPtdP_t2r_p = thr_tmem_load.partition_S(tdPtdP)
        tdPtdP_t2r = self.split_wg(tdPtdP_t2r_p, wg_idx, num_wg)
        tScS_t2r_p = thr_tmem_load.partition_D(tScS)
        tScS_t2r = self.split_wg(tScS_t2r_p, wg_idx, num_wg)
        tSsLSE_p = thr_tmem_load.partition_D(thr_mma_SdP.partition_C(sLSE_2D))
        tSsLSE = self.split_wg(tSsLSE_p, wg_idx, num_wg)  # ((32, 1), 2, 1, 1, STAGE)
        tSsdPsum_p = thr_tmem_load.partition_D(thr_mma_SdP.partition_C(sdPsum_2D))
        tSsdPsum = self.split_wg(tSsdPsum_p, wg_idx, num_wg)

        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tScP_r2t_p = thr_tmem_store.partition_S(tScP)
        tScP_r2t = self.split_wg(tScP_r2t_p, wg_idx, num_wg)
        tStP_r2t_p = thr_tmem_store.partition_D(tStP)
        tStP_r2t = self.split_wg(tStP_r2t_p, wg_idx, num_wg)

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

        consumer_phase_LSE = consumer_phase_dPsum = cute.Int32(0)

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
                n_block=n_block,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
            )

            # Mainloop
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                #### TMEM->RMEM (Load S from TMEM)
                tSrS_t2r = cute.make_fragment(tScS_t2r.shape, Float32)  # 64
                cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
                cute.arch.fence_view_async_tmem_load()

                # Without this barrier, we could have 1 warp writing to P in tmem while
                # another warp is still reading S from tmem.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwdSm100.Compute),
                    number_of_threads=self.num_compute_threads,
                )

                cute.arch.mbarrier_wait(LSE_full_mbar_ptr, consumer_phase_LSE)
                consumer_phase_LSE ^= 1

                #### APPLY MASK
                if const_expr(self.is_causal or self.is_local):
                    mask_fn(tSrS_t2r, tScS_t2r, m_block=m_block)

                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------

                lane_idx = cute.arch.lane_idx()

                tSrP_r2t_f32 = cute.make_fragment(tScP_r2t[None, None, 0].shape, Float32)  # 16
                tSrP_r2t = cute.make_tensor(
                    cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
                    tSrS_t2r[None, 0, None, None].layout,
                )

                for stage in cutlass.range_constexpr(cute.size(tStP_r2t, mode=[2]), unroll=1):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, 0]  # TODO: have stages
                    if const_expr(not self.shuffle_LSE):
                        tSrLSE = cute.make_fragment_like(tSsLSE_cur, Float32)
                        cute.autovec_copy(tSsLSE_cur, tSrLSE)
                    else:
                        tSrLSE = tSsLSE_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tSrP_r2t) // 2, unroll_full=True):
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
                    utils.cvt_f16(tSrS_cur, tSrP_r2t[None, 0, 0])
                    cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t[None, None, stage])

                cute.arch.fence_view_async_tmem_store()

                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                    # pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                    cute.arch.mbarrier_arrive(LSE_empty_mbar_ptr)
                # consumer_state_S_P_dP.advance()

                # ---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                # ---------------------------------------------
                cute.arch.mbarrier_wait(dPsum_full_mbar_ptr, consumer_phase_dPsum)
                consumer_phase_dPsum ^= 1

                pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
                pipeline_dS.producer_acquire(producer_state_dS)

                #### TMEM->RMEM (Load dP from TMEM)
                # ((32,1),1,1)
                tdPrdP_t2r = cute.make_fragment(tScS_t2r[None, 0, None, None].shape, Float32)

                ##### dS.T = P.T * (dP.T - Psum)
                sdSt_mn = cute.composition(sdS, cute.make_layout((self.tile_m, self.tile_n)))
                tdKsdS = cute.composition(
                    sdSt_mn[(None, wg_idx), tidx], cute.make_layout(tSrS_t2r.shape)
                )
                tSrS_t2r_bf16 = cute.make_tensor(
                    cute.recast_ptr(tSrS_t2r.iterator, dtype=self.ds_dtype), tSrS_t2r.shape
                )

                for stage in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[1]), unroll=1):
                    cute.copy(thr_tmem_load, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                    cute.arch.fence_view_async_tmem_load()
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, 0]  # TODO: have stages
                    if const_expr(not self.shuffle_dPsum):
                        tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                        cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    else:
                        tSrdPsum = tSsdPsum_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r) // 2, unroll=1):
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
                    utils.cvt_f16(tdPrdP_cur, tSrS_t2r_bf16[None, stage, 0, 0])
                    cute.autovec_copy(tSrS_t2r_bf16[None, stage, 0, 0], tdKsdS[None, stage, 0, 0])

                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    # pipeline_dP.consumer_release(consumer_state_dP)
                    pipeline_dP.sync_object_empty.arrive(0, pipeline_dP.consumer_mask)
                    cute.arch.mbarrier_arrive(dPsum_empty_mbar_ptr)
                consumer_state_S_P_dP.advance()
                # consumer_phase_S_P_dP ^= 1

                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

            if const_expr(not self.use_tma_store):
                consumer_state_dKV = self.epilogue_dKV(
                    tidx,
                    warp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dV,
                    thr_mma_dK,
                    tdVtdV,
                    tdKtdK,
                    mdV,
                    mdK,
                    pipeline_dKV,
                    consumer_state_dKV,
                    softmax_scale,
                )
            else:
                thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(tidx)
                #### STORE dV
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    tidx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dV,
                    tdVtdV,
                    mdV_tma_tensor,
                    sdV,
                    tma_atom_dV,
                    thr_copy_r2s_dKV,
                    pipeline_dKV,
                    consumer_state_dKV,
                    None,  # Don't scale
                    int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                    mdV_semaphore,
                )
                #### STORE dK
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    tidx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dK,
                    tdKtdK,
                    mdK_tma_tensor,
                    sdK,
                    tma_atom_dK,
                    thr_copy_r2s_dKV,
                    pipeline_dKV,
                    consumer_state_dKV,
                    softmax_scale,
                    int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
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
        tdQtdQ: cute.Tensor,
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
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_tmem_load.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_tmem_load.partition_D(tdQcdQ).shape
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == self.dQaccum_reduce_stage, (
            "dQaccum reduce stage mismatch"
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(not self.deterministic)

        # TODO: reduce_phase is currently hardcoded for 2 stages
        reduce_phase = cutlass.Int32(0)

        dQacc_reduce_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=num_reduce_threads,
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
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
                # TMEM -> RMEM
                tdQrdQ_t2r = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_tmem_load, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                # semaphore acquire
                if const_expr(self.deterministic):
                    barrier.wait_eq(mdQ_semaphore_cur[(m_block, None)].iterator, tidx, 0, n_block)
                    dQacc_reduce_barrier.arrive_and_wait()

                # We could delay the TMA store by 1 epi tile to better overlap the non-TMA ops
                delay_tma_store = False

                def tma_store_fn(src_idx, dst_idx):
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                    )
                    dQacc_reduce_barrier.arrive_and_wait()
                    # Copy from shared memory to global memory
                    if is_tma_warp:
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQaccum[None, src_idx].iterator,
                                gdQaccum[None, dst_idx, m_block].iterator,
                                self.tma_copy_bytes["dQ"],
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(1, read=read_flag)
                    dQacc_reduce_barrier.arrive_and_wait()

                reduce_phase_prev, stage_prev = None, -1
                for stage in cutlass.range_constexpr(cute.size(tdQrdQ_t2r, mode=[1])):  # 4
                    tdQsdQ_r2s = tdQsdQ[None, None, reduce_phase]
                    tdQrdQ_r2s = cute.make_tensor(
                        tdQrdQ_t2r[None, stage, None, None].iterator, tdQsdQ_r2s.shape
                    )
                    if const_expr(delay_tma_store):
                        if const_expr(stage > 0):
                            tma_store_fn(src_idx=reduce_phase_prev, dst_idx=stage_prev)
                        reduce_phase_prev, stage_prev = reduce_phase, stage
                    cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                    if const_expr(not delay_tma_store):
                        tma_store_fn(reduce_phase, stage)
                    reduce_phase ^= 1
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
                if const_expr(delay_tma_store):
                    tma_store_fn(src_idx=reduce_phase_prev, dst_idx=stage_prev)

                # semaphore release
                # NOTE: arrive_inc calls red_release which issues membar
                if const_expr(self.deterministic):
                    if tidx == 0:
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                    dQacc_reduce_barrier.arrive_and_wait()
                    barrier.arrive_inc(mdQ_semaphore_cur[(m_block, None)].iterator, tidx, 0, 1)

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
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
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

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_fragment(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_fragment(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.tile_m, self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, n_block]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_fragment(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_fragment(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdimv), (None, 0))
        gdK_tile = gdK[None, None, n_block]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
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
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
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

        wg_idx = (cute.arch.thread_idx()[0] % self.num_compute_threads) // 128
        num_wg = self.num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        sdKV = sdKV[None, None, wg_idx]

        head_idx_kv = head_idx // self.qhead_per_kvhead
        mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]

        gdKV_p = cute.local_tile(mdKV_cur, (self.tile_m, self.tile_hdimv), (n_block, 0))
        gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)
        gdKV_epi = cute.local_tile(gdKV, self.sdKV_epi_tile, (0, None))

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

        # (TMA) and (TMA, EPI_STAGE)
        tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
            tma_atom_dKV,
            0,  # no multicast
            cute.make_layout(1),
            cute.group_modes(sdKV, 0, 2),
            cute.group_modes(gdKV_epi, 0, 2),
        )

        assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
        assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"

        num_epi_stages = cute.size(tdKVgdKV.shape[1])
        assert num_epi_stages == 1 or num_epi_stages == 2, "Wrong number of epi stages"

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )

        read_flag = const_expr(not self.deterministic)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # semaphore acquire
        if const_expr(self.deterministic):
            barrier.wait_eq(
                mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for s in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_tmem_load.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, s]

            cdKV = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_tmem_load.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, s]

            tdKVrdKV_t2r = cute.make_fragment(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_tmem_load, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = utils.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_fragment(tdKVrdKV_t2r.shape, self.dv_dtype)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(self.dv_dtype))

            # RMEM -> SMEM -- setup
            tdKVcdKV_r2s_p = thr_copy_r2s_dKV.partition_S(cdKV)
            tdKVcdKV_r2s = self.split_wg(tdKVcdKV_r2s_p, wg_idx, num_wg)
            tdKVcdKV_r2s = cute.logical_divide(
                tdKVcdKV_r2s,
                (
                    tdKVcdKV_r2s.shape[0],
                    tdKVcdKV_r2s.shape[1],
                    tdKVcdKV_r2s.shape[2] // num_epi_stages,
                ),
            )[((None, 0), (None, 0), (None, s))]

            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVcdKV_r2s.shape)

            tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

            assert cute.size(tdKVrdKV_r2s) == cute.size(tdKVsdKV_r2s), (
                "RMEM<->SMEM fragment size mismatch"
            )

            # RMEM -> SMEM -- copy, fence and barrier
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, s])
                if s < num_epi_stages - 1:
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
