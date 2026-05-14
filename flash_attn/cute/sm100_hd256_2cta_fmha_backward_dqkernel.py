# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

from functools import partial
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import math
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.nvgpu import cpasync
from cutlass import const_expr
import cutlass.utils as cutlass_utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.blackwell_helpers import gemm_w_idx
from flash_attn.cute import pipeline
from flash_attn.cute import copy_utils
from flash_attn.cute import utils
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.cute.typing import Int32, Int64, Float32
import quack.activation
from quack import layout_utils

from cutlass.utils import ClcDynamicPersistentTileScheduler
from flash_attn.cute.tile_scheduler import (
    ClcState,
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    SingleTileLPTBwdScheduler,
    compute_sm100_fmha_grid as compute_grid,
    compute_sm100_fmha_grid_clc as compute_grid_clc,
    Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
    Sm100FmhaClcDynamicTileScheduler as FmhaClcDynamicTileScheduler,
    Sm100FmhaClcDynamicTileSchedulerParams as FmhaClcDynamicTileSchedulerParams,
)
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK


class BlackwellFusedMultiHeadAttentionBackwardDQKernel:
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
        spt: Optional[bool] = None,
        cluster_size: int = 2,
        use_2cta_instrs: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = tile_m
        self.n_block_size = tile_n
        self.q_stage = 1
        assert self.q_stage in [1, 2]
        self.use_2cta_instrs = use_2cta_instrs

        assert head_dim == 256
        assert head_dim_v == 256
        assert self.head_dim_padded == 256
        assert self.head_dim_v_padded == 256
        assert self.m_block_size == 128 and self.n_block_size == 128, "Only 128x128 tile impl is supported"
        assert not is_persistent
        assert not deterministic
        assert cluster_size == 2
        assert self.use_2cta_instrs
        assert score_mod is None
        assert score_mod_bwd is None
        assert mask_mod is None
        assert not has_aux_tensors

        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
        self.cta_tiler = (self.q_stage * self.m_block_size, self.n_block_size, self.head_dim_padded)
        # With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
        self.mma_tiler_qk = (
            self.cta_group_size * self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        # dP = dO @ V.T
        self.mma_tiler_dov = (
            self.cta_group_size * self.m_block_size,
            self.n_block_size,
            self.head_dim_v_padded,
        )
        # dQ = dS @ K
        self.mma_tiler_dsk = (
            self.cta_group_size * self.m_block_size,
            self.head_dim_padded,
            self.n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.dov_acc_dtype = Float32
        self.dsk_acc_dtype = Float32
        self.acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.use_correction_warps_for_epi = False
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = False
        self.pack_gqa = False
        self.q_subtile_factor = None
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = False
        self.use_clc_scheduler = False
        self.sched_stages = 1
        self.shuffle_LSE = False
        self.shuffle_dPsum = False

        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.deterministic = deterministic
        self.spt = spt
        self.cluster_size = cluster_size
        self.has_aux_tensors = has_aux_tensors
        self.subtile_factor = subtile_factor
        self.window_size_left = None
        self.window_size_right = cutlass.Int32(0) if self.is_causal else None

        self.tile_m = self.m_block_size
        self.tile_n = self.n_block_size
        self.tile_hdim = self.head_dim_padded
        self.tile_hdimv = self.head_dim_v_padded
        self.mma_tiler = self.cta_tiler
        self.qk_mma_tiler = self.mma_tiler_qk
        self.dov_mma_tiler = self.mma_tiler_dov
        self.dsk_mma_tiler = self.mma_tiler_dsk
        self.dsk_block_tiler = (
            self.dsk_mma_tiler[0] // 2,
            self.dsk_mma_tiler[1],
            self.dsk_mma_tiler[2],
        )
        self.tmem_warp_shape_mn = (4, 1)
        self.use_semantic_trip_range = self.is_causal or self.is_local

        # Warp layout (HD256 dQ-only): 12 warps -> 384 threads
        self.compute_warp_ids = (0, 1, 2, 3)  # 4 warps
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (4, 5, 6, 7, 10, 11)
        self.softmax0_warp_ids = self.compute_warp_ids
        self.softmax1_warp_ids = ()
        self.correction_warp_ids = ()
        self.load_warp_ids = (self.load_warp_id,)
        self.empty_warp_ids = self.empty_warp_id
        self.sched_warp_id = self.empty_warp_id[0] if self.use_clc_scheduler else None
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.num_compute_warps = len(self.compute_warp_ids)

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2
        self.epilogue_sync_bar_id = 3

        self.threads_per_warp = cute.arch.WARP_SIZE
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_id,
            )
        )

        self.tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.compute_warp_ids)),
        )

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_dq_offset = self.tmem_s_offset[-1] + self.n_block_size  # e.g., 256
        self.tmem_total = self.tmem_dq_offset + self.head_dim_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_dp_offset = self.n_block_size
        self.tmem_dp_offset = self.tmem_s_offset[0] + self.tmem_s_to_dp_offset

        self.num_regs_compute = 256
        self.num_regs_other = 32

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up pipeline stages for the HD256 dQ kernel."""

        self.q_stage = 1
        self.k_stage = 1
        self.do_stage = 1
        self.v_stage = 1
        self.kt_stage = 1
        self.Q_stage = self.q_stage
        self.dO_stage = self.do_stage
        self.single_stage = 1

        self.qk_acc_stage = 1
        self.dov_acc_stage = 1
        self.dsk_acc_stage = 1
        self.epi_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        if cutlass.const_expr(self.use_clc_scheduler):
            self.num_clc_stage = 1
            self.num_clc_response_bytes = 16

    def _get_tiled_mma(self):
        self.cta_group = tcgen05.CtaGroup.TWO

        # S = Q @ K.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.qk_mma_tiler[:2],
        )
        # dP = dO @ V.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dov_mma_tiler[:2],
        )
        # dQ = dS @ K. HD256 2CTA keeps dS as TMEM A and K.T as MN-major B;
        # this is the intentional swapAB divergence from flash_bwd_sm100.py.
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dsk_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S = Q @ K.T
        self.sQ_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        self.sK_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )

        # dP = dO @ V.T
        self.sdO_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.dov_mma_tiler,
            self.do_dtype,
            self.do_stage,
        )
        self.sV_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.dov_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )

        # dQ = dS @ K
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.dsk_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.dsk_mma_tiler,
            self.k_dtype,
            self.dsk_acc_stage,
        )

        self.sLSE_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_LSE_stage))
        self.sdPsum_layout = cute.make_layout(
            (self.cta_tiler[0], self.load_compute_sum_OdO_stage)
        )

        self.epi_cols_dQ = math.gcd(128 // (self.dq_dtype.width // 8), self.epi_tile[1])
        self.epi_tile_dQ = (self.epi_tile[0], self.epi_cols_dQ)
        self.num_epi_stages_dQ = max(1, self.tile_hdim // self.epi_tile_dQ[1])
        self.sdQ_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dq_dtype,
            self.dq_layout,
            self.epi_tile_dQ,
            1,
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
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        # Block-sparse tensors (Q direction - for iterating m_blocks per n_block):
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        assert mSeqUsedQ is None
        assert mSeqUsedK is None
        assert mdQ_semaphore is None
        assert mdK_semaphore is None
        assert mdV_semaphore is None
        assert aux_tensors is None
        assert blocksparse_tensors is None

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dq_dtype = mdQaccum.element_type
        self.ds_dtype = self.q_dtype

        self.is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mdQaccum = assume_tensor_aligned(mdQaccum)

        # (b, s, n, h) --> (s, h, n, b) or (t, n, h) -> (t, h, n)
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdO, mdQaccum = [
            layout_utils.select(t, mode=Q_layout_transpose) for t in (mQ, mdO, mdQaccum)
        ]

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        # (b, n, s) --> (s, n, b) or (n, t) --> (t, n)
        LSE_dPsum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum = [
            layout_utils.select(t, mode=LSE_dPsum_transpose) for t in (mLSE, mdPsum)
        ]
        mdQ = mdQaccum
        self.dq_layout = cutlass_utils.LayoutEnum.from_tensor(mdQ)

        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if const_expr(self.q_dtype != self.do_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.do_dtype}")

        # Transposes for 2-CTA K paths (K follows K seqlens)
        transpose_sh_k = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]

        self.epi_tile = self.dsk_block_tiler[:2]

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dQ
        ) = self._get_tiled_mma()
        self._setup_smem_layout()
        self.dQ_reduce_ncol = self.epi_tile_dQ[1]

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        # dQ epilogue TMA store. Use a single (M, 64) SMEM stage and rotate it
        # across the hd=256 N dimension to stay within the SMEM budget.
        tma_copy_op_dQ = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_dQ, mdQ_tma = cpasync.make_tiled_tma_atom(
            tma_copy_op_dQ,
            mdQ,
            cute.select(self.sdQ_epi_layout, mode=[0, 1]),
            self.epi_tile_dQ,
        )
        thr_layout_r2s_dQ = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
        val_layout_r2s_dQ = cute.make_ordered_layout(
            (1, 128 // self.dq_dtype.width), order=(1, 0)
        )  # 4 or 8 vals for 16 byte store
        copy_atom_r2s_dQ = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dq_dtype,
            num_bits_per_copy=128,
        )
        tiled_copy_r2s_dQ = cute.make_tiled_copy_tv(
            copy_atom_r2s_dQ, thr_layout_r2s_dQ, val_layout_r2s_dQ
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # S = Q @ K.T
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_dO, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.dov_mma_tiler,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )

        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        K_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            K_tma_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.qk_mma_tiler,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        V_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dP.thr_id
        )
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_B(
            V_tma_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.dov_mma_tiler,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id
        )
        tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
            Kt_tma_op,
            layout_utils.select(mK, mode=transpose_sh_k),
            cute.select(self.sKt_layout, mode=[0, 1, 2]),
            self.mma_tiler_dsk,
            self.tiled_mma_dQ,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
                ("Kt", layout_utils.select(mK, mode=transpose_sh_k), self.sKt_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8

        # TileScheduler = SingleTileScheduler
        if const_expr(self.is_varlen_q):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler

        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),  # num_blocks
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),  # num_batches
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mQ.shape[0])  # pass total_k for total_q
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,  # pack_gqa disabled for bwd
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,  # persistent mode not tested
            lpt=self.is_causal or self.is_local,
            use_cluster_idx=False,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.do_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.load_compute_LSE_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, 2 * self.load_compute_sum_OdO_stage
            ]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.qk_acc_stage]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dov_acc_stage]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dsk_acc_stage]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.epi_stage]
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr: cutlass.Int64

            # 2-CTA
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.k_stage]
            Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.kt_stage]
            V_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.v_stage]

            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]

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
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                self.buffer_align_bytes,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                128,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                128,
            ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            # Without score_mod: bake scale into log2
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            # With score_mod: score_mod applied to S * softmax_scale, then use LOG2_E only
            softmax_scale_log2 = LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        # Launch the kernel synchronously
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_do,
            mdQ,
            mdQ_tma,
            mCuSeqlensQ,
            mCuSeqlensK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dQ,
            self.sQ_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.tdS_layout,
            self.sdQ_epi_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dQ,
            softmax_scale,
            window_size_left,
            window_size_right,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdQ: cute.Tensor,
        mdQ_tma: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dQ: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        sdQ_epi_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dQ: cute.TiledCopy,
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params: SingleTileScheduler.Params
        | SingleTileVarlenScheduler.Params
        | SingleTileLPTBwdScheduler.Params,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_dO)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Kt)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_dQ)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        clc = None

        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
        )

        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.dov_acc_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cluster_shape_mnk[0],
        )

        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=self.epi_stage,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=self.dsk_acc_stage,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
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
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            defer_sync=True,
        )

        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_K = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.K_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_V = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.V_mbar_ptr.data_ptr(),
            num_stages=self.v_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_Kt = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
            num_stages=self.single_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        # Cluster arrive after barrier init
        cutlass.pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        sKT = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        # Alias the dQ TMA epilogue staging buffer onto sdO.  A standalone
        # (128, 64) bf16 allocation exceeds B200's per-block SMEM limit for
        # this kernel, while sdO is no longer needed once the dQ accumulator
        # is ready to store.
        sdQ = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, sdQ_epi_layout.inner, self.dq_dtype),
            sdQ_epi_layout.outer,
        )

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)

        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)  # default 1sm
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)  # default 1sm
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)  # default 1sm
        qk_acc_shape = thr_mma_S.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = thr_mma_S.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        dov_acc_shape = thr_mma_dP.partition_shape_C(
            (self.dov_mma_tiler[0], self.dov_mma_tiler[1])
        )
        tdPtdP = thr_mma_dP.make_fragment_C(cute.append(dov_acc_shape, self.dov_acc_stage))
        dsk_acc_shape = thr_mma_dQ.partition_shape_C(
            (self.dsk_mma_tiler[0], self.dsk_mma_tiler[1])
        )
        tdQtdQ = thr_mma_dQ.make_fragment_C(dsk_acc_shape)
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
        tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_dp_offset, tdPtdP.layout)
        tdQtdQ = cute.make_tensor(tdQtdQ.iterator + self.tmem_dq_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.qk_mma_tiler[0],
            self.qk_mma_tiler[1],
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )

        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=None,
            mSeqUsedK=None,
            tile_m=self.cta_tiler[0],
            tile_n=self.cta_tiler[1],
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n * self.cta_group_size,
            swap_AB=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

        #  EMPTY
        # (4, 5, 6, 7, 10, 11)
        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        # Cluster wait
        cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #  LOAD
        # (9)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                tiled_mma_S,
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dQ,
                mQ,
                mK,
                mKt,
                mV,
                mdO,
                mLSE,
                mdPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                sQ,
                sK,
                sKT,
                sV,
                sdO,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_K,
                pipeline_Kt,
                pipeline_V,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
            )

        #  MMA
        # (8)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dQ,
                sQ,
                sK,
                sKT,
                sV,
                sdO,
                tStS,
                tdPtdP,
                tdQtdQ,
                tdS_layout,
                pipeline_Q,
                pipeline_K,
                pipeline_V,
                pipeline_dO,
                pipeline_Kt,
                pipeline_S_P,
                pipeline_dP,
                pipeline_dS,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            self.tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (0, 1, 2, 3) --> 4 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem.retrieve_ptr(self.acc_dtype)
            self.compute_loop(
                tiled_mma_S,
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dQ,
                scale_softmax,
                tStS,
                tdPtdP,
                tdQtdQ,
                mdQ,
                mdQ_tma,
                sdQ,
                tma_atom_dQ,
                tiled_copy_r2s_dQ,
                sLSE,
                sdPsum,
                pipeline_S_P,
                pipeline_dP,
                pipeline_dS,
                pipeline_dQ,
                pipeline_LSE,
                pipeline_dPsum,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            self.tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        tiled_mma_S,
        thr_mma_S,
        thr_mma_dP,
        thr_mma_dQ,
        mQ,
        mK,
        mKt,
        mV,
        mdO,
        mLSE,
        mdPsum,
        cluster_layout_vmnk,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
        sQ,
        sK,
        sKT,
        sV,
        sdO,
        sLSE,
        sdPsum,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_Kt,
        tma_atom_V,
        tma_atom_dO,
        pipeline_Q,
        pipeline_K,
        pipeline_Kt,
        pipeline_V,
        pipeline_dO,
        pipeline_LSE,
        pipeline_dPsum,
    ):
        producer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_K = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.k_stage
        )
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_V = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.v_stage
        )
        producer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage
        )
        producer_state_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage
        )

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            head_idx_kv = head_idx // self.qhead_per_kvhead
            m_block_cta_group = m_block // self.cta_group_size
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block_cta_group, Int32(0), Int32(1)
            )

            # GMEM tensors (varlen-aware)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]
            mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
                mKt_cur = mKt[None, None, head_idx_kv, batch_idx]
            else:
                mKt_cur = cute.domain_offset((0, seqlen.offset_k, 0), mKt)[
                    None, None, head_idx_kv
                ]

            # (1) S = Q @ K.T
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.qk_mma_tiler, mode=[0, 2]), (m_block_cta_group, 0)
            )
            tSgQ = thr_mma_S.partition_A(gQ)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[2],
                cta_layout=a_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
                single_stage=True,
            )

            gK = cute.local_tile(
                mK_cur, cute.select(self.qk_mma_tiler, mode=[1, 2]), (None, 0)
            )
            tSgK = thr_mma_S.partition_B(gK)

            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgK,
                dst_tensor=sK,
            )
            load_K = copy_utils.tma_producer_copy_fn(load_K, pipeline_K)

            # (2) dP = dO @ V.T
            gdO = cute.local_tile(
                mdO_cur, cute.select(self.dov_mma_tiler, mode=[0, 2]), (m_block_cta_group, 0)
            )
            tdPgdO = thr_mma_dP.partition_A(gdO)

            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[2],
                cta_layout=a_cta_layout,
                src_tensor=tdPgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
                single_stage=True,
            )

            gV = cute.local_tile(
                mV_cur, cute.select(self.dov_mma_tiler, mode=[1, 2]), (None, 0)
            )
            tSgV = thr_mma_dP.partition_B(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgV,
                dst_tensor=sV,
            )
            load_V = copy_utils.tma_producer_copy_fn(load_V, pipeline_V)

            # (3) dQ = dS @ K
            gKt = cute.local_tile(
                mKt_cur, cute.select(self.dsk_mma_tiler, mode=[1, 2]), (0, None)
            )
            tdQgK = thr_mma_dQ.partition_B(gKt)

            load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Kt,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdQgK,
                dst_tensor=sKT,
            )
            load_Kt = copy_utils.tma_producer_copy_fn(load_Kt, pipeline_Kt)

            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)

            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q and not self.is_varlen_k)
                or n_block_min < n_block_max
            )

            if process_tile:
                first_n_block = n_block_min

                #### Prologue ####
                # Q (for S)
                pipeline_Q.producer_acquire(producer_state_Q)
                load_Q(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                pipeline_Q.producer_commit(producer_state_Q)
                producer_state_Q.advance()

                # K (for S)
                pipeline_K.producer_acquire(producer_state_K)
                load_K(first_n_block, producer_state=producer_state_K)
                pipeline_K.producer_commit(producer_state_K)
                producer_state_K.advance()

                # LSE
                pipeline_LSE.producer_acquire(producer_state_LSE)
                with cute.arch.elect_one():
                    copy_stats(
                        gLSE[None, m_block],
                        sLSE[None, producer_state_LSE.index],
                        mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                    )
                producer_state_LSE.advance()

                # dO (for dP)
                pipeline_dO.producer_acquire(producer_state_dO)
                load_dO(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO))
                pipeline_dO.producer_commit(producer_state_dO)
                producer_state_dO.advance()

                # V (for dP)
                pipeline_V.producer_acquire(producer_state_V)
                load_V(first_n_block, producer_state=producer_state_V)
                pipeline_V.producer_commit(producer_state_V)
                producer_state_V.advance()

                # dPsum
                pipeline_dPsum.producer_acquire(producer_state_dPsum)
                with cute.arch.elect_one():
                    copy_stats(
                        gdPsum[None, m_block],
                        sdPsum[None, producer_state_dPsum.index],
                        mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                    )
                producer_state_dPsum.advance()

                # Kt (for dQ)
                pipeline_Kt.producer_acquire(producer_state_Kt)
                load_Kt(first_n_block, producer_state=producer_state_Kt)
                pipeline_Kt.producer_commit(producer_state_Kt)
                producer_state_Kt.advance()

                #### Main Loop ####
                for n_block in cutlass.range(n_block_min + 1, n_block_max, unroll=1):
                    # K (for S)
                    pipeline_K.producer_acquire(producer_state_K)
                    load_K(n_block, producer_state=producer_state_K)
                    pipeline_K.producer_commit(producer_state_K)
                    producer_state_K.advance()

                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block],
                            sLSE[None, producer_state_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                        )
                    producer_state_LSE.advance()

                    # V (for dP)
                    pipeline_V.producer_acquire(producer_state_V)
                    load_V(n_block, producer_state=producer_state_V)
                    pipeline_V.producer_commit(producer_state_V)
                    producer_state_V.advance()

                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block],
                            sdPsum[None, producer_state_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                producer_state_dPsum
                            ),
                        )
                    producer_state_dPsum.advance()

                    # Kt (for dQ)
                    pipeline_Kt.producer_acquire(producer_state_Kt)
                    load_Kt(n_block, producer_state=producer_state_Kt)
                    pipeline_Kt.producer_commit(producer_state_Kt)
                    producer_state_Kt.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop
        pipeline_K.producer_tail(producer_state_K)
        pipeline_V.producer_tail(producer_state_V)
        pipeline_Kt.producer_tail(producer_state_Kt)
        pipeline_Q.producer_tail(producer_state_Q)
        pipeline_dO.producer_tail(producer_state_dO)
        pipeline_LSE.producer_tail(producer_state_LSE)
        pipeline_dPsum.producer_tail(producer_state_dPsum)


    @cute.jit
    def mma(
        self,
        tiled_mma_S,
        tiled_mma_dP,
        tiled_mma_dQ,
        sQ,
        sK,
        sKT,
        sV,
        sdO,
        tStS,
        tdPtdP,
        tdQtdQ,
        tdS_layout,
        pipeline_Q,
        pipeline_K,
        pipeline_V,
        pipeline_dO,
        pipeline_Kt,
        pipeline_S_P,
        pipeline_dP,
        pipeline_dS,
        pipeline_dQ,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        # S = Q @ K.T
        tSrQ = thr_mma_S.make_fragment_A(sQ)[None, None, None, 0]
        tSrK = thr_mma_S.make_fragment_B(sK)
        # dP = dO @ V.T
        tdPrdO = thr_mma_dP.make_fragment_A(sdO)[None, None, None, 0]
        tdPrV = thr_mma_dP.make_fragment_B(sV)
        # dQ = dS @ K
        tdQrKT = thr_mma_dQ.make_fragment_B(sKT)

        mma_qk_fn = partial(
            gemm_w_idx,
            tiled_mma_S,
            tStS[None, None, None, 0],
            tSrQ,
            tSrK,
            zero_init=True,
            num_unroll_groups=2,
        )
        mma_dov_fn = partial(
            gemm_w_idx,
            tiled_mma_dP,
            tdPtdP[None, None, None, 0],
            tdPrdO,
            tdPrV,
            zero_init=True,
            num_unroll_groups=2,
        )
        num_unroll_groups = 2 if const_expr(self.use_2cta_instrs) else 1
        mma_dsk_fn = partial(
            gemm_w_idx,
            tiled_mma_dQ,
            tCrB=tdQrKT,
            num_unroll_groups=num_unroll_groups,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_K = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_V = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        producer_phase_acc = Int32(1)  # For S and dP
        producer_phase_dQ = Int32(1)
        consumer_state_dS = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.dsk_acc_stage
        )
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            n_block_cta_group = n_block // cute.size(tiled_mma_S.thr_id.shape)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, n_block_cta_group, Int32(0), Int32(1)
            )
            block_iter_count = n_block_max - n_block_min
            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q and not self.is_varlen_k)
                or (
                    n_block_cta_group < cute.ceil_div(seqlen.seqlen_q, self.qk_mma_tiler[0])
                    and n_block_min < n_block_max
                )
            )

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_cta = cta_rank_in_cluster % 2 == 0

            if is_leader_cta and process_tile:
                consumer_state_Q_releaser = consumer_state_Q.clone()
                consumer_state_dO_releaser = consumer_state_dO.clone()
                accumulate_dQ = False

                pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)

                # -----------------------------------------------------------
                ###### Prologue
                # -----------------------------------------------------------
                # 1. S  = Q @ K.T
                # 2. dP = dO @ V.T

                # 1) S = Q @ K.T
                pipeline_Q.consumer_wait(consumer_state_Q)
                consumer_state_Q.advance()
                pipeline_K.consumer_wait(consumer_state_K)
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_qk_fn(B_idx=consumer_state_K.index)
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                pipeline_K.consumer_release(consumer_state_K)
                consumer_state_K.advance()

                # 2) dP = dO @ V.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                consumer_state_dO.advance()
                pipeline_V.consumer_wait(consumer_state_V)
                pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_V.index)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                pipeline_V.consumer_release(consumer_state_V)
                consumer_state_V.advance()
                producer_phase_acc ^= 1

                # -----------------------------------------------------------
                ###### Main Loop
                # -----------------------------------------------------------
                # 1. S  = Q    @ K.T  (next)
                # 2. dQ = dS   @ K    (cur)
                # 3. dP = dO   @ V.T  (next)

                main_loop_iters = block_iter_count - 1
                for _ in cutlass.range(main_loop_iters, unroll=1):
                    # (1) S = Q @ K.T (next)
                    pipeline_K.consumer_wait(consumer_state_K)
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_qk_fn(B_idx=consumer_state_K.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    pipeline_K.consumer_release(consumer_state_K)
                    consumer_state_K.advance()

                    # (2) dQ += dS @ K (cur)
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    pipeline_Kt.consumer_wait(consumer_state_Kt)
                    tdStdS = tdPtdP[None, None, None, consumer_state_dS.index]
                    tdS = cute.make_tensor(tdStdS.iterator, tdS_layout.outer)
                    tdQrdS = thr_mma_dQ.make_fragment_A(tdS)
                    tdQrdS = cute.make_tensor(
                        cute.recast_ptr(tdStdS.iterator, dtype=self.q_dtype),
                        tdQrdS.layout,
                    )
                    mma_dsk_fn(
                        tdQtdQ,
                        tdQrdS,
                        B_idx=consumer_state_Kt.index,
                        zero_init=not accumulate_dQ,
                    )
                    accumulate_dQ = True
                    pipeline_Kt.consumer_release(consumer_state_Kt)
                    consumer_state_Kt.advance()
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    # (3) dP = dO @ V.T (next)
                    pipeline_V.consumer_wait(consumer_state_V)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_V.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                    pipeline_V.consumer_release(consumer_state_V)
                    consumer_state_V.advance()
                    producer_phase_acc ^= 1

                # Release Q / dO after the final S and dP that reuse them are issued.
                pipeline_Q.consumer_release(consumer_state_Q_releaser)
                consumer_state_Q_releaser.advance()
                pipeline_dO.consumer_release(consumer_state_dO_releaser)
                consumer_state_dO_releaser.advance()

                # -----------------------------------------------------------
                # Tail: remaining dQ
                # -----------------------------------------------------------
                pipeline_dS.consumer_wait(consumer_state_dS)
                pipeline_Kt.consumer_wait(consumer_state_Kt)
                tdStdS = tdPtdP[None, None, None, consumer_state_dS.index]
                tdS = cute.make_tensor(tdStdS.iterator, tdS_layout.outer)
                tdQrdS = thr_mma_dQ.make_fragment_A(tdS)
                tdQrdS = cute.make_tensor(
                    cute.recast_ptr(tdStdS.iterator, dtype=self.q_dtype),
                    tdQrdS.layout,
                )
                mma_dsk_fn(
                    tdQtdQ,
                    tdQrdS,
                    B_idx=consumer_state_Kt.index,
                    zero_init=not accumulate_dQ,
                )
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                producer_phase_dQ ^= 1
                pipeline_Kt.consumer_release(consumer_state_Kt)
                consumer_state_Kt.advance()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

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
        tiled_mma_S,
        thr_mma_S,
        thr_mma_dP,
        thr_mma_dQ,
        scale_softmax,
        tStS,
        tdPtdP,
        tdQtdQ,
        mdQ,
        mdQ_tma,
        sdQ,
        tma_atom_dQ,
        tiled_copy_r2s_dQ,
        sLSE,
        sdPsum,
        pipeline_S_P,
        pipeline_dP,
        pipeline_dS,
        pipeline_dQ,
        pipeline_LSE,
        pipeline_dPsum,
        block_info,
        SeqlenInfoCls,
        TileSchedulerCls,
    ):
        # This dQ kernel computes S = Q @ K.T, so accumulator coordinates are (q, k).
        # Keep LSE/dPsum indexed by q; flash_bwd_sm100.py transposes these views because
        # its main loop computes S.T = K @ Q.T.
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

        # tix: [128...384] 8 warps
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        num_wg = len(self.compute_warp_ids) // 4
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_rank_offset = cta_rank_in_cluster * self.cta_tiler[0]
        # Stats SMEM is CTA-local, while 2CTA MMA C coordinates cover both CTA rows.
        sLSE_2D = cute.domain_offset((Int32(0) - cta_rank_offset, Int32(0), Int32(0)), sLSE_2D)
        sdPsum_2D = cute.domain_offset(
            (Int32(0) - cta_rank_offset, Int32(0), Int32(0)), sdPsum_2D
        )

        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
        # tdS overlaps with tdP in TMEM. dQ-only does not materialize P in TMEM.
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.qk_mma_tiler[:2]))
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.dov_mma_tiler[:2]))
        tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        # 2-CTA assumes: repetition should always be 32 & 16.
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)

        consumer_state_S_P = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.qk_acc_stage
        )
        consumer_state_dP = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dov_acc_stage
        )
        producer_state_dS = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dsk_acc_stage
        )
        consumer_state_dQ = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.epi_stage
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        softmax_scale_log2 = scale_softmax * cutlass.Float32(math.log2(math.e))

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            m_block_cta_group = m_block // cute.size(tiled_mma_S.thr_id.shape)
            seqlen = SeqlenInfoCls(batch_idx)
            seqlen_q = seqlen.seqlen_q
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block_cta_group, Int32(0), Int32(1)
            )
            mask = AttentionMask(
                self.qk_mma_tiler[0],
                self.qk_mma_tiler[1],
                seqlen,
                window_size_left=block_info.window_size_left,
                window_size_right=block_info.window_size_right,
            )
            mask_fn = partial(
                mask.apply_mask_sm100,
                m_block=m_block_cta_group,
                thr_mma=thr_mma_S,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                mask_mod=None,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=None,
                fastdiv_mods=(None, None),
                head_divmod=None,
                r2p=True,
                rBitmask=None,
            )

            prefetch_LSE = False
            loop_count = n_block_max - n_block_min
            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q and not self.is_varlen_k)
                or n_block_min < n_block_max
            )

            # Mainloop
            for iter_idx in cutlass.range(loop_count, unroll=1):
                n_block = n_block_min + iter_idx

                # S = Q @ K.T in this kernel, so S/dP accumulator coordinates are (q, k).
                # flash_bwd_sm100.py consumes S.T as (k, q); do not transpose the row-indexed
                # LSE/dPsum path when mirroring its loop structure here.
                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_fragment(tScS_t2r[None, 0, 0, 0].shape, Float32)
                if const_expr(prefetch_LSE and not self.shuffle_LSE):
                    cute.autovec_copy(
                        tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r
                    )

                pipeline_S_P.consumer_wait(consumer_state_S_P)
                tSrS_t2r = cute.make_fragment(tScS_t2r.shape, Float32)
                cute.copy(
                    thr_copy_t2r,
                    tStS_t2r[None, None, None, None, consumer_state_S_P.index],
                    tSrS_t2r,
                )
                cute.arch.fence_view_async_tmem_load()
                pipeline_S_P.consumer_release(consumer_state_S_P)
                consumer_state_S_P.advance()
                check_q_boundary = (m_block_cta_group + 1) * self.qk_mma_tiler[0] > seqlen_q
                mask_fn(
                    tSrS_t2r,
                    n_block=n_block,
                    thr_tmem_load=thr_copy_t2r,
                    check_q_boundary=check_q_boundary,
                )
                num_stages = cute.size(tScS_t2r, mode=[1])
                lane_idx = cute.arch.lane_idx()
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
                        tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (softmax_scale_log2, softmax_scale_log2),
                            (-lse_pair[0], -lse_pair[1]),
                        )
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()

                # ---------------------------------------------
                #### dS = P * (dP - dPsum)
                # ---------------------------------------------
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_dP)
                pipeline_dS.producer_acquire(producer_state_dS)

                for stage in cutlass.range_constexpr(num_stages):
                    tdPrdP_t2r = cute.make_fragment(tScS_t2r[None, 0, None, None].shape, Float32)
                    cute.copy(
                        thr_copy_t2r,
                        tdPtdP_t2r[None, stage, None, None, consumer_state_dP.index],
                        tdPrdP_t2r,
                    )
                    cute.arch.fence_view_async_tmem_load()
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
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                            quack.activation.sub_packed_f32x2(
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                            )
                        )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                            (scale_softmax, scale_softmax),
                        )

                    tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                    utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                    tdPrdS_r2t = cute.recast_tensor(tdPrdS_cvt, Float32)
                    cute.copy(thr_copy_r2t, tdPrdS_r2t, tdPtdS_r2t[None, stage, 0, 0])
                cute.arch.fence_view_async_tmem_store()
                pipeline_dP.consumer_release(consumer_state_dP)
                consumer_state_dP.advance()
                pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()

            # Epilogue
            if process_tile:
                if const_expr(not seqlen.has_cu_seqlens_q):
                    thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
                    consumer_state_dQ = self.epilogue_dQ_tma(
                        tidx,
                        batch_idx,
                        head_idx,
                        m_block,
                        seqlen,
                        thr_mma_dQ,
                        tdQtdQ,
                        mdQ_tma,
                        sdQ,
                        tma_atom_dQ,
                        thr_copy_r2s_dQ,
                        pipeline_dQ,
                        consumer_state_dQ,
                        None,  # Don't scale
                        self.epilogue_sync_bar_id,
                        None,
                    )
                else:
                    consumer_state_dQ = self.epilogue_dQ(
                        tidx,
                        batch_idx,
                        head_idx,
                        m_block,
                        seqlen,
                        thr_mma_dQ,
                        tdQtdQ,
                        mdQ,
                        pipeline_dQ,
                        consumer_state_dQ,
                    )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        pipeline_dS.producer_tail(producer_state_dS)

    @cute.jit
    def epilogue_dQ_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        thr_mma: cute.core.ThrMma,
        tdQtdQ: cute.Tensor,
        mdQ: cute.Tensor,
        sdQ: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        thr_copy_r2s_dQ: cute.TiledCopy,
        pipeline_dQ,
        consumer_state_dQ,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdQ_semaphore: Optional[cute.Tensor],
    ) -> cutlass.pipeline.PipelineState:
        tile_hdim = self.tile_hdim
        dtype = self.dq_dtype
        epi_tile = self.epi_tile_dQ
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        assert scale is None
        assert mdQ_semaphore is None

        sdQ = sdQ[None, None, wg_idx]  # (tile_m, 64) for bf16

        # (8, tile_m / 128, 64 / 8) = (8, 1, 8)
        tdQsdQ_r2s = thr_copy_r2s_dQ.partition_D(sdQ)

        assert not seqlen.has_cu_seqlens_q, "varlen uses non tma store path"
        mdQ_cur = mdQ[None, None, head_idx, batch_idx]  # (seqlen, hdim)
        gdQ_p = cute.local_tile(mdQ_cur, (self.tile_m, tile_hdim), (m_block, 0))
        gdQ = self.split_wg(gdQ_p, wg_idx, num_wg)  # (tile_m, hdim)
        gdQ_epi = cute.local_tile(gdQ, epi_tile, (0, None))  # (tile_m, 64, epi_stage)

        tdQsdQ, tdQgdQ = cpasync.tma_partition(
            tma_atom_dQ,
            0,  # no multicast
            cute.make_layout(1),
            cute.group_modes(sdQ, 0, 2),
            cute.group_modes(gdQ_epi, 0, 2),
        )  # (TMA) and (TMA, EPI_STAGE)
        assert len(tdQsdQ.shape) == 1, "Wrong rank for SMEM fragment tdQsdQ"
        assert len(tdQgdQ.shape) == 2, "Wrong rank for GMEM fragment tdQgdQ"
        num_epi_stages = cute.size(tdQgdQ.shape[1])
        assert num_epi_stages == self.num_epi_stages_dQ, "Epi stage calculation is wrong (Q)"

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32
        )

        read_flag = const_expr(True)

        pipeline_dQ.consumer_wait(consumer_state_dQ)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
            tdQtdQ_t2r_p = thr_copy_t2r.partition_S(tdQtdQ)
            tdQtdQ_t2r = self.split_wg(tdQtdQ_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdQtdQ_t2r = tdQtdQ_t2r[None, epi_stage]

            cdQ = cute.make_identity_tensor((self.tile_m, tile_hdim))
            tdQcdQ = thr_mma.partition_C(cdQ)
            tdQcdQ_t2r_p = thr_copy_t2r.partition_D(tdQcdQ)
            tdQcdQ_t2r = self.split_wg(tdQcdQ_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdQcdQ_t2r = tdQcdQ_t2r[None, epi_stage]

            tdQrdQ_t2r = cute.make_fragment(tdQcdQ_t2r.shape, Float32)

            assert cute.size(tdQrdQ_t2r) == cute.size(tdQtdQ_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- convert
            tdQrdQ = cute.make_fragment(tdQrdQ_t2r.shape, dtype)  # (64 columns)
            tdQrdQ.store(tdQrdQ_t2r.load().to(dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdQrdQ_r2s = cute.make_tensor(tdQrdQ.iterator, tdQsdQ_r2s.shape)
            cute.copy(thr_copy_r2s_dQ, tdQrdQ_r2s, tdQsdQ_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ[None, epi_stage])
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dQ.consumer_release(consumer_state_dQ)
        consumer_state_dQ.advance()
        return consumer_state_dQ

    @cute.jit
    def epilogue_dQ(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        thr_mma: cute.core.ThrMma,
        tdQtdQ: cute.Tensor,
        mdQ: cute.Tensor,
        pipeline_dQ,
        consumer_state_dQ,
    ) -> cutlass.pipeline.PipelineState:
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        thread_idx = tidx % 128
        epi_tile = self.epi_tile
        mdQ_cur = seqlen.offset_batch_Q(mdQ, batch_idx, dim=3)[None, None, head_idx]
        gdQ = cute.local_tile(
            mdQ_cur, cute.select(self.dsk_block_tiler, mode=[0, 1]), (m_block, 0)
        )
        cdQ = cute.local_tile(
            cute.make_identity_tensor(mdQ_cur.shape),
            cute.select(self.dsk_block_tiler, mode=[0, 1]),
            (m_block, 0),
        )
        tdQtdQ = tdQtdQ[(None, None), 0, 0]
        tdQtdQ_epi = cute.zipped_divide(tdQtdQ, epi_tile)
        gdQ_epi = cute.zipped_divide(gdQ, epi_tile)
        cdQ_epi = cute.zipped_divide(cdQ, epi_tile)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ_epi).get_slice(thread_idx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ_epi)
        tdQgdQ_t2r = thr_copy_t2r.partition_D(gdQ_epi)
        tdQcdQ_t2r = thr_copy_t2r.partition_D(cdQ_epi)

        pipeline_dQ.consumer_wait(consumer_state_dQ)

        for epi_stage in cutlass.range(cute.size(tdQtdQ_t2r, mode=[1]), unroll_full=True):
            # TMEM -> RMEM -- copy and fence
            tdQtdQ_t2r_cur = tdQtdQ_t2r[None, epi_stage, 0]
            tdQgdQ_t2r_cur = tdQgdQ_t2r[None, epi_stage, 0]
            tdQcdQ_t2r_cur = tdQcdQ_t2r[None, epi_stage, 0]
            tdQrdQ_t2r = cute.make_fragment(tdQcdQ_t2r_cur.shape, Float32)
            cute.copy(thr_copy_t2r, tdQtdQ_t2r_cur, tdQrdQ_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -> GMEM
            tdQrdQ = cute.make_fragment(tdQrdQ_t2r.shape, self.dq_dtype)
            tdQrdQ.store(tdQrdQ_t2r.load().to(self.dq_dtype))
            if cute.elem_less(tdQcdQ_t2r_cur[0][0], seqlen.seqlen_q):
                cute.autovec_copy(tdQrdQ, tdQgdQ_t2r_cur)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dQ.consumer_release(consumer_state_dQ)
        consumer_state_dQ.advance()
        return consumer_state_dQ
