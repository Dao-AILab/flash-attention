# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* cta_tiler_mn must be 64,128
* Batch size must be the same for Q, K, and V tensors
"""

import math
from functools import partial
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import LayoutEnum
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.cute.typing import Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from flash_attn.cute import copy_utils
from flash_attn.cute import pipeline
from flash_attn.cute import utils
from quack import layout_utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
)
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.blackwell_helpers import gemm_w_idx
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.named_barrier import NamedBarrierBwdSm100


class BlackwellFusedMultiHeadAttentionBackwardDKDVKernel:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 64,
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
        """Initialization."""
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n

        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 HD256 dK/dV kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert self.tile_hdim == 256 and self.tile_hdimv == 256
        assert self.tile_m == 128 and self.tile_n == 64, (
            "SM100 HD256 dK/dV kernel only supports tile_m=128 and tile_n=64"
        )

        self.use_2cta_instrs = bool(
            use_2cta_instrs
            and cluster_size == 2
            and score_mod is None
            and score_mod_bwd is None
            and mask_mod is None
        )
        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        assert self.use_2cta_instrs, "SM100 HD256 dK/dV kernel requires use_2cta_instrs=True"

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (self.cta_group_size * tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (self.cta_group_size * tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (self.cta_group_size * tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (self.cta_group_size * tile_n, self.tile_hdim, tile_m)
        # dQ = dS @ K
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n * self.cta_group_size)

        self.acc_dtype = Float32

        assert cluster_size == 2, "SM100 HD256 dK/dV kernel only supports cluster_size=2"
        self.cluster_shape_mn = (cluster_size, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.deterministic = deterministic
        self.spt_override = spt

        # Score mod and mask mod support
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.subtile_factor = subtile_factor
        # For score_mod, use vec_size=1 (like forward) to handle per-element indices
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32

        # Speed optimizations, does not affect correctness
        self.shuffle_LSE = False
        self.shuffle_dPsum = False
        # Generally slower to use store dS in smem for dK, and doesn't work for 2cta
        self.use_smem_dS_for_mma_dK = False

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15

        self.num_compute_warps = len(self.compute_warp_ids)

        # 16 warps -> 512 threads
        self.threads_per_warp = cute.arch.WARP_SIZE
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )

        self.tmem_alloc_sync_bar_id = int(NamedBarrierBwdSm100.TmemPtr)
        self.epilogue_sync_bar_id = int(NamedBarrierBwdSm100.EpilogueWG1)
        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )

        # TMEM setup
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_dK_offset = 0
        self.tmem_dV_offset = self.tmem_dK_offset + self.tile_hdim // 2
        self.tmem_S_offset = self.tmem_dK_offset + self.tile_hdim
        self.tmem_dP_offset = self.tmem_S_offset + self.tile_m // 2

        if (not is_causal and not is_local) or deterministic:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 152
            self.num_regs_compute = 136
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        else:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 136
            self.num_regs_compute = 136 if self.use_2cta_instrs else 144
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        self.num_regs_empty = 24

        assert (
            self.num_regs_reduce
            + self.num_regs_compute * 2
            + max(self.num_regs_load, self.num_regs_mma)
            <= 512
        )
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 1 if self.use_2cta_instrs else 2
        self.dO_stage = 1
        self.single_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        # CTA group for MMA operations
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        tiled_mma_S = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO
        tiled_mma_dV = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            tcgen05.OperandSource.SMEM,
        )
        # dK += dS.T @ Q
        tiled_mma_dK = sm100_utils.make_trivial_tiled_mma(
            self.ds_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
        )
        # dQ = dS @ K. HD256 dK/dV keeps this only for the staged dS layout.
        tiled_mma_dQ = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        self.sK_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sQ_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        self.sV_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sdOt_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        self.sdSt_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            self.single_stage,
        )
        self.sQt_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dV += P.T @ dO
        self.sP_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.ds_dtype,
            self.single_stage,
        )
        self.sdO_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )

        num_compute_wgs = self.num_compute_warps // 4
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // num_compute_wgs),
        )
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // num_compute_wgs),
        )
        num_epi_stages = (self.tile_hdim // num_compute_wgs) // self.sdK_epi_tile[1]
        num_epi_stages_v = (self.tile_hdimv // num_compute_wgs) // self.sdV_epi_tile[1]
        total_epi_stages = num_compute_wgs * num_epi_stages
        total_epi_stages_v = num_compute_wgs * num_epi_stages_v
        self.sdK_layout = sm100_utils.make_smem_layout_epi(
            self.dk_dtype,
            self.dk_layout_enum,
            self.sdK_epi_tile,
            total_epi_stages,
        )
        self.sdV_layout = sm100_utils.make_smem_layout_epi(
            self.dv_dtype,
            self.dv_layout_enum,
            self.sdV_epi_tile,
            total_epi_stages_v,
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
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        self.is_varlen_k = mCuSeqlensK is not None
        self.is_varlen_q = mCuSeqlensQ is not None
        self.use_tma_store = True
        self.dKV_postprocess = False

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        mdK, mdV = [assume_tensor_aligned(t) for t in (mdK, mdV)]

        # (b, s, n, h) --> (s, h, n, b) or (t, n, h) -> (t, h, n)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        # (b, n, s) --> (s, n, b) or (n, t) --> (t, n)
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum = [
            layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum)
        ]

        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
        mdK, mdV = [layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        # (s, h, n, b) --> (h, s, n, b) or (t, h, n) -> (h, t, b)
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose

        self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
        self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
        self.dk_layout_enum = self.mdK_layout_enum
        self.dv_layout_enum = self.mdV_layout_enum

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
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
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
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
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdK,
            cute.select(self.sdK_layout, mode=[0, 1]),
            self.sdK_epi_tile,
        )
        tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdV,
            cute.select(self.sdV_layout, mode=[0, 1]),
            self.sdV_epi_tile,
        )
        tma_tensor_dK = mdK_tma_tensor
        tma_tensor_dV = mdV_tma_tensor

        # TileScheduler = SingleTileScheduler
        if const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        if const_expr(self.spt_override is None):
            self.spt = (self.is_causal or self.is_local) and self.deterministic
        else:
            assert self.spt_override is not None
            self.spt = self.spt_override and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),  # num K-blocks
            cute.size(mK.shape[2]),  # num KV heads
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batches
            1,  # num_splits
            cute.size(mK.shape[0]),  # pass seqlen_k/static_k
            mK.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])  # pass total_k for total_q
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.spt,
            head_swizzle=self.deterministic,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.single_stage * 2]
            V_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.single_stage * 2]
            Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            LSE_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.Q_stage * 2
            ]
            dPsum_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.dO_stage * 2
            ]
            S_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.single_stage * 2
            ]
            dP_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.single_stage * 2
            ]
            P_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.single_stage * 2
            ]
            dS_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.single_stage * 2
            ]
            dKV_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.sdKVaccum_stage * 2
            ]
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar_ptr: cutlass.Int64
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            # only used in 2cta
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                self.buffer_align_bytes,
            ]
            sQt: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQt_layout)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                self.buffer_align_bytes,
            ]
            sdOt: cute.struct.Align[
                cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdOt_layout)],
                self.buffer_align_bytes,
            ]
            # only used in 2cta
            # dishengbin checked whether we need sP
            sP: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sP_layout)],
                self.buffer_align_bytes,
            ]
            sdSt: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                self.buffer_align_bytes,
            ]

            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                self.buffer_align_bytes,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                self.buffer_align_bytes,
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

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            tma_tensor_dV,
            tma_tensor_dK,
            mCuSeqlensQ,
            mCuSeqlensK,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
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
            self.sP_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
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
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdV_tma_tensor: cute.Tensor,
        mdK_tma_tensor: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params,
    ):
        """Core CuTeDSL backward kernel."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        varlen = mCuSeqlensQ is not None or mCuSeqlensK is not None

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                if const_expr(tma_atom_Qt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dOt is not None):
                    cpasync.prefetch_descriptor(tma_atom_dOt)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        pipeline_S = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_P = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.P_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
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
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_K = pipeline.PipelineTmaUmma.create(
            num_stages=self.single_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            barrier_storage=storage.K_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_V = pipeline.PipelineTmaUmma.create(
            num_stages=self.single_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
            barrier_storage=storage.V_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_Qt = pipeline.PipelineTmaUmma.create(
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        # setup mma
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner
        )
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        tmem_holding_buf = storage.tmem_holding_buf
        # for 2cta, Qt use different mem from Q

        sQt = storage.sQt.get_tensor(
            sQt_layout.outer, swizzle=sQt_layout.inner
        )
        sdSt = storage.sdSt.get_tensor(
            sdSt_layout.outer, swizzle=sdSt_layout.inner
        )
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)

        sdOt = storage.sdOt.get_tensor(
            sdOt_layout.outer, swizzle=sdOt_layout.inner
        )

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)

        blk_coord_k, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = blk_coord_k % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0

        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)

        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)

        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        # tdVrP shape : (MMA, MMA_M, MMA_K, STAGE)
        tdVrP = tiled_mma_dV.make_fragment_A(sP)

        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)

        # Logical m_block range helper (same role as flash_bwd_sm100).
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n * self.cluster_shape_mn[0],
            self.is_causal,
            self.is_local,
            False,
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
            tile_m=self.tile_m,
            tile_n=self.tile_n * self.cluster_shape_mn[0],
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
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  RELAY
        # (14)
        if warp_idx == self.relay_warp_id:
            cute.arch.setmaxregister_decrease(
                self.num_regs_mma if self.use_2cta_instrs else self.num_regs_empty
            )

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                mQ,
                mK,
                mV,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q,
                pipeline_Qt,
                pipeline_K,
                pipeline_V,
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
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                sQ,
                sQt,
                sK,
                sV,
                sdO,
                sdOt,
                tdVrP,
                sdSt,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                pipeline_Q,
                pipeline_Qt,
                pipeline_K,
                pipeline_V,
                pipeline_dO,
                pipeline_S,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_P,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                is_leader_cta,
            )

            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)


        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdSt,
                sdOt,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S,
                pipeline_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                varlen,
                tdVrP,
                sP,
                sK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                sdV_layout,
                sdK_layout,
            )
            tmem_alloc_barrier.arrive()


        # Reduce
        # (0, 1, 2, 3) - dQ placeholder disabled for the dK/dV-only kernel.
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tmem_alloc_barrier.arrive()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mQt: cute.Tensor,
        mdOt: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: cute.CopyAtom,
        tma_atom_dOt: cute.CopyAtom,
        pipeline_Q,
        pipeline_Qt,
        pipeline_K,
        pipeline_V,
        pipeline_dO,
        pipeline_LSE,
        pipeline_dPsum,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls,
        TileSchedulerCls,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        """TMA load."""
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_K = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_V = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(
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
            n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mn[0]
            )
            n_block_cta_group = n_block // self.cta_group_size

            # GMEM tensors (varlen-aware)
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tSgK = thr_mma_S.partition_A(gK)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK,
                single_stage=True,
            )

            # (2) dP = V @ dO.T
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tdPgV,
                sV,
                single_stage=True,
            )

            # some tiles might be empty due to local/window masking
            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )

            if process_tile:
                if const_expr(should_load_Q):
                    # K, for S.T = K @ Q.T
                    pipeline_K.producer_acquire(producer_state_K)
                    load_K(tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_K))
                    producer_state_K.advance()

                if const_expr(should_load_dO):
                    # V, for dP = V @ dO.T
                    pipeline_V.producer_acquire(producer_state_V)
                    load_V(tma_bar_ptr=pipeline_V.producer_get_barrier(producer_state_V))
                    producer_state_V.advance()

                b_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                )
                for head_idx_r in cutlass.range(self.qhead_per_kvhead, unroll=1):
                    head_idx = head_idx_kv * self.qhead_per_kvhead + head_idx_r
                    mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                    if const_expr(not seqlen.has_cu_seqlens_q):
                        mdO_cur = mdO[None, None, head_idx, batch_idx]
                        mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                        mQt_cur = mQt[None, None, head_idx, batch_idx]
                    else:
                        mdO_cur = cute.domain_offset(
                            (0, seqlen.offset_q), mdO[None, None, head_idx]
                        )
                        mdOt_cur = cute.domain_offset((seqlen.offset_q, 0, 0), mdOt)[
                            None, None, head_idx
                        ]
                        mQt_cur = cute.domain_offset((0, seqlen.offset_q, 0), mQt)[
                            None, None, head_idx
                        ]
                    mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[
                        None, head_idx
                    ]
                    mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[
                        None, head_idx
                    ]
                    gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                    gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                    gQ = cute.local_tile(
                        mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0)
                    )
                    tSgQ = thr_mma_S.partition_B(gQ)

                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q,
                        cta_coord=block_in_cluster_coord_vmnk[1],
                        cta_layout=b_cta_layout,
                        src_tensor=tSgQ,
                        dst_tensor=sQ,
                        mcast_mask=q_do_mcast_mask,
                    )
                    load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

                    # (2) dP = V @ dO.T
                    gdOt = cute.local_tile(
                        mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                    )
                    tdPgdO = thr_mma_dP.partition_B(gdOt)
                    load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_dOt,
                        cta_coord=block_in_cluster_coord_vmnk[1],
                        cta_layout=b_cta_layout,
                        src_tensor=tdPgdO,
                        dst_tensor=sdOt,
                        mcast_mask=q_do_mcast_mask,
                    )
                    load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

                    # (3) dV += P.T @ dO
                    gdO = cute.local_tile(
                        mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None)
                    )
                    tdVgdO = thr_mma_dV.partition_B(gdO)
                    load_dO, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_dO,
                        cta_coord=block_in_cluster_coord_vmnk[1],
                        cta_layout=b_cta_layout,
                        src_tensor=tdVgdO,
                        dst_tensor=sdO,
                        mcast_mask=q_do_mcast_mask,
                    )
                    load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

                    # (4) dK += dS.T @ Q
                    gQt = cute.local_tile(
                        mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                    )
                    tdKgQt = thr_mma_dK.partition_B(gQt)
                    load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Qt,
                        cta_coord=block_in_cluster_coord_vmnk[1],
                        cta_layout=b_cta_layout,
                        src_tensor=tdKgQt,
                        dst_tensor=sQt,
                        mcast_mask=q_do_mcast_mask,
                    )
                    load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

                    copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
                    copy_stats = partial(cute.copy, copy_atom_stats)

                    first_m_block = m_block_min
                    #### Prologue ####
                    if const_expr(should_load_Q):
                        # Q (for S)
                        pipeline_Q.producer_acquire(producer_state_Q_LSE)
                        load_Q(first_m_block, producer_state=producer_state_Q_LSE)
                        pipeline_Q.producer_commit(producer_state_Q_LSE)

                        # LSE
                        pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                        with cute.arch.elect_one():
                            copy_stats(
                                gLSE[None, first_m_block],
                                sLSE[None, producer_state_Q_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                            )
                        producer_state_Q_LSE.advance()

                    if const_expr(should_load_dO):
                        # dO + dOt
                        pipeline_dO.producer_acquire(
                            producer_state_dO_dPsum,
                            extra_tx_count=self.tma_copy_bytes["dO"],
                        )
                        load_dO(first_m_block, producer_state=producer_state_dO_dPsum)
                        load_dOt(first_m_block, producer_state=producer_state_dO_dPsum)
                        pipeline_dO.producer_commit(producer_state_dO_dPsum)

                        # dPsum
                        pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                        with cute.arch.elect_one():
                            copy_stats(
                                gdPsum[None, first_m_block],
                                sdPsum[None, producer_state_dO_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                    producer_state_dO_dPsum
                                ),
                            )
                        producer_state_dO_dPsum.advance()

                    #### Main Loop ####
                    for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                        if const_expr(should_load_Q):
                            # Qt, for dK = dS.T @ Q
                            pipeline_Qt.producer_acquire(producer_state_Qt)
                            load_Qt(m_block - 1, producer_state=producer_state_Qt)
                            pipeline_Qt.producer_commit(producer_state_Qt)
                            producer_state_Qt.advance()

                            # Q (for S)
                            pipeline_Q.producer_acquire(producer_state_Q_LSE)
                            load_Q(m_block, producer_state=producer_state_Q_LSE)
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gLSE[None, m_block],
                                    sLSE[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(
                                        producer_state_Q_LSE
                                    ),
                                )
                            producer_state_Q_LSE.advance()

                        if const_expr(should_load_dO):
                            # dO + dOt
                            pipeline_dO.producer_acquire(
                                producer_state_dO_dPsum,
                                extra_tx_count=self.tma_copy_bytes["dO"],
                            )
                            load_dO(m_block, producer_state=producer_state_dO_dPsum)
                            load_dOt(m_block, producer_state=producer_state_dO_dPsum)
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            with cute.arch.elect_one():
                                copy_stats(
                                    gdPsum[None, m_block],
                                    sdPsum[None, producer_state_dO_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                )
                            producer_state_dO_dPsum.advance()

                    #### Tail ####
                    if const_expr(should_load_Q):
                        pipeline_Qt.producer_acquire(producer_state_Qt)
                        load_Qt(m_block_max - 1, producer_state=producer_state_Qt)
                        pipeline_Qt.producer_commit(producer_state_Qt)
                        producer_state_Qt.advance()

                if cutlass.const_expr(True):
                    if const_expr(should_load_Q):
                        pipeline_K.producer_tail(producer_state_K)
                        pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                        pipeline_LSE.producer_tail(producer_state_Q_LSE)
                        pipeline_Qt.producer_tail(producer_state_Qt)
                    if const_expr(should_load_dO):
                        pipeline_V.producer_tail(producer_state_V)
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
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tdVrP: cute.Tensor,
        sdSt: cute.Tensor,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        pipeline_Q,
        pipeline_Qt,
        pipeline_K,
        pipeline_V,
        pipeline_dO,
        pipeline_S,
        pipeline_dS,
        pipeline_dKV,
        pipeline_dP,
        pipeline_P,
        block_info: BlockInfo,
        SeqlenInfoCls,
        TileSchedulerCls,
        is_leader_cta: cutlass.Boolean,
    ):
        """CuTeDSL kernel for mma pipeline."""
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dOt.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        # dK = dS.T @ Q
        tdKrdS = tiled_mma_dK.make_fragment_A(sdSt)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)

        mma_qk_fn = partial(
            gemm_w_idx,
            tiled_mma_S,
            tSTtST,
            tSrK,
            tSrQ,
            zero_init=True,
        )
        mma_dov_fn = partial(
            gemm_w_idx,
            tiled_mma_dP,
            tdPTtdPT,
            tdPrV,
            tdPrdOt,
            zero_init=True,
        )
        mma_pdo_fn = partial(
            gemm_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
        )
        mma_dsq_fn = partial(
            gemm_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
        )

        consumer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        producer_phase_acc = Int32(1)  # For S and dP
        producer_phase_dKV = Int32(1)
        consumer_phase_KV = Int32(0)
        consumer_phase_P = Int32(0)
        cta_group = pipeline_S.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mn[0]
            )
            process_tile = (
                cutlass.const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )

            if process_tile:
                loop_count = (m_block_max - m_block_min) * self.qhead_per_kvhead
                accumulate_dK = False

                if is_leader_cta and process_tile:
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = K @ Q
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    pipeline_S.sync_object_empty.wait(0, producer_phase_acc)
                    pipeline_K.sync_object_full.wait(0, consumer_phase_KV)
                    pipeline_Q.consumer_wait(consumer_state_Q)

                    mma_qk_fn(A_idx=0, B_idx=consumer_state_Q.index)
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                    cute.arch.fence_view_async_tmem_store()
                    pipeline_S.sync_object_full.arrive(0, pipeline_S.producer_mask, cta_group)

                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_V.sync_object_full.wait(0, consumer_phase_KV)

                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    # 2) dP = V @ dOt.T
                    mma_dov_fn(A_idx=0, B_idx=consumer_state_dO.index)

                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                    # V only produced once by load(); hold it until end, release there via release state.

                    pipeline_P.sync_object_full.wait(0, consumer_phase_P)

                    # 3) dV = P.T @ dO
                    producer_phase_acc ^= 1
                    mma_pdo_fn(A_idx=0, B_idx=consumer_state_dO.index, zero_init=True)

                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()
                    pipeline_P.sync_object_empty.arrive(0, pipeline_P.consumer_mask, pipeline_P.cta_group)
                    consumer_phase_P ^= 1

                # -----------------------------------------------------------
                ###### MAIN LOOP
                # -----------------------------------------------------------
                # 1. S.T  = K    @ Q.T
                # 2. dK   = dS.T @ Q
                # 3. dP.T = V    @ dO.T
                # 4. dV   = P.T  @ dO

                main_loop_iters = loop_count - 1
                for _ in cutlass.range(main_loop_iters, unroll=1):
                    if is_leader_cta and process_tile:
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_S.sync_object_empty.wait(0, producer_phase_acc)

                        # (1) S.T = K @ Q.T (next)
                        mma_qk_fn(A_idx=0, B_idx=consumer_state_Q.index)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()
                        pipeline_S.sync_object_full.arrive(0, pipeline_S.producer_mask, cta_group)

                        pipeline_Qt.consumer_wait(consumer_state_Qt)
                        pipeline_dS.consumer_wait(consumer_state_dS)

                        # (2) dK += dS.T @ Q (cur)
                        mma_dsq_fn(A_idx=consumer_state_dS.index, B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()

                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        # (3) dP.T = V @ dO.T (next)
                        # V only produced once by load(); reuse same V (index 0) for all loop iterations
                        mma_dov_fn(A_idx=0, B_idx=consumer_state_dO.index)

                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                        pipeline_P.sync_object_full.wait(0, consumer_phase_P)

                        # (4) dV += P.T @ dO (next)
                        producer_phase_acc ^= 1
                        mma_pdo_fn(A_idx=0, B_idx=consumer_state_dO.index, zero_init=False)

                        pipeline_P.sync_object_empty.arrive(0, pipeline_P.consumer_mask, pipeline_P.cta_group)
                        consumer_phase_P ^= 1
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                if is_leader_cta and process_tile:
                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)

                    pipeline_K.sync_object_empty.arrive(0, pipeline_K.consumer_mask, pipeline_K.cta_group)
                    pipeline_V.sync_object_empty.arrive(0, pipeline_V.consumer_mask, pipeline_V.cta_group)
                    consumer_phase_KV ^= 1

                    # -----------------------------------------------------------
                    # Tail: Remaining dK
                    # -----------------------------------------------------------
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    pipeline_dS.consumer_wait(consumer_state_dS)
                    pipeline_Qt.consumer_wait(consumer_state_Qt)

                    # dK += dS.T @ Q
                    mma_dsq_fn(A_idx=consumer_state_dS.index, B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)

                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            t = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        (num_wg, cute.size(t, mode=[2]) // num_wg),
                    )
                ),
            )
            return t[None, None, (wg_idx, None)]
        else:
            t = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        t.shape[2],
                        (num_wg, cute.size(t, mode=[3]) // num_wg),
                    )
                ),
            )
            return t[None, None, None, (wg_idx, None)]

    @cute.jit
    def reg_to_smem_mma128x128_2cta(
        self,
        regs: cute.Tensor,
        smem: cute.Tensor,
        index: Int32,
        tiler_mn: tuple[Int32, Int32],
        dp_idx: Int32,
        wg_idx: Int32,
        smem_RowMajor: bool = True,
    ):
        smem_slice = smem[None, None, None, index]
        # K>> smem_slice:  tensor<ptr<f16, smem, align<1024>, S<3,4,3>> o ((64,16),1,(4,2)):((64,1),0,(16,4096))>
        thread_layout = cute.make_ordered_layout(
            # (tileN, tileM)
            tiler_mn,
            (0, 1),
        )
        # K>> thread_layout:  (64,128):(128,1)
        smem_slice_tmp = cute.composition(smem_slice, thread_layout)

        # NOTE: hardcode for tcgen05.ld.32x32b.x8 & mma128x64+2cta
        # tmp_shape = ((32, 2), (8, 2, 2, 2)) # for 64x64 tile
        # tmp_stride = ((64, 32*64), (1, 8, 16, 32))
        # NOTE: hardcode for tcgen05.ld.32x32b.x16 & mma128x64+2cta
        tmp_shape = ((32, 2), (16, 2, 2, 2))  # for 128x64 tile
        tmp_stride = ((64, 32 * 64), (1, 16, 32, 64 * 64))
        # smem_copy = cute.composition(smem_slice_tmp, cute.make_layout(tmp_shape, stride=tmp_stride))
        smem_copy = cute.make_tensor(
            smem_slice_tmp.iterator, cute.make_layout(tmp_shape, stride=tmp_stride)
        )

        warp_idx = dp_idx // 32
        warp_row_idx = warp_idx % 2
        warp_col_idx = warp_idx // 2  # corresponding to the second 64 cols in smem
        lane_idx = dp_idx % 32
        reg_shape = (
            regs.shape
        )  # ((8,1),1,2):((1,0),0,8) for 64x64, ((16,1),1,2):((1,0),0,16) for 128x64
        block_loops = reg_shape[2]

        # TODO: maybe can use cp.async for optimization
        for ib in cutlass.range(block_loops):
            regs_copy = regs[(None, 0), 0, ib]
            smem_copy_slice = smem_copy[(lane_idx, warp_row_idx), (None, wg_idx, ib, warp_col_idx)]
            cute.autovec_copy(regs_copy, smem_copy_slice)

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdSt: cute.Tensor,
        sdOt: cute.Tensor,
        pipeline_LSE,
        pipeline_dPsum,
        pipeline_S,
        pipeline_P,
        pipeline_dS,
        pipeline_dKV,
        pipeline_dP,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls,
        AttentionMaskCls,
        TileSchedulerCls,
        varlen: bool,
        tdVrP: cute.Tensor,
        sP: cute.Tensor,
        sK: cute.Tensor,
        mdV_tma_tensor: cute.Tensor,
        mdK_tma_tensor: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        sdV_layout: cute.ComposedLayout,
        sdK_layout: cute.ComposedLayout,
    ):
        """CuTeDSL kernel for recomputing softmax and producing dk and dv."""
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
        if const_expr(True):
            sLSE_2D = layout_utils.transpose_view(sLSE_2D)
            sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
        # tP overlaps with tS in TMEM.
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlaps with tdP in TMEM.
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
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
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)

        consumer_state_S = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        producer_state_P = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        consumer_state_dP = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        producer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.sdKVaccum_stage
        )

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cS_smem = cute.make_identity_tensor((self.cta_group_size * self.tile_n, self.tile_m))
        cS_smem = cute.local_tile(
            cS_smem,
            cute.select(self.cta_tiler, mode=[0, 1]),
            (cta_rank_in_cluster, 0),
        )
        tmem_load_atom_smem = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype
        )
        thr_copy_t2r_smem = tcgen05.make_tmem_copy(
            tmem_load_atom_smem, tStS[(None, None), 0, 0]
        ).get_slice(dp_idx)
        tScS_t2r = thr_copy_t2r_smem.partition_D(cS_smem)
        tScS_t2r = self.split_wg(tScS_t2r, wg_idx, num_wg)
        t0ScS_t2r = thr_copy_t2r_smem.get_slice(0).partition_D(cS_smem)
        t0ScS_t2r = self.split_wg(t0ScS_t2r, wg_idx, num_wg)
        tStS_t2r = thr_copy_t2r_smem.partition_S(tStS[(None, None), 0, 0])
        tStS_t2r = self.split_wg(tStS_t2r, wg_idx, num_wg)
        tdPcdP_t2r = thr_copy_t2r_smem.partition_D(cS_smem)
        tdPcdP_t2r = self.split_wg(tdPcdP_t2r, wg_idx, num_wg)
        tdPtdP_t2r = thr_copy_t2r_smem.partition_S(tdPtdP[(None, None), 0, 0])
        tdPtdP_t2r = self.split_wg(tdPtdP_t2r, wg_idx, num_wg)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            blk_coord_k, head_idx_kv, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            seqlen_q_cur_batch = seqlen.seqlen_q
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, blk_coord_k // self.cluster_shape_mn[0]
            )
            mask = AttentionMaskCls(seqlen)
            n_block_for_cluster = blk_coord_k // self.cluster_shape_mn[0]
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block_for_cluster,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                mask_mod=None,
                batch_idx=batch_idx,
                head_idx=head_idx_kv,
                aux_tensors=None,
                fastdiv_mods=(None, None),
            )
            process_tile = (
                cutlass.const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )
            m_block_count = m_block_max - m_block_min
            loop_count = m_block_count * self.qhead_per_kvhead

            # Mainloop
            for iter_idx in cutlass.range(loop_count, unroll=1):
                head_idx_r = iter_idx // m_block_count
                m_block = m_block_min + iter_idx - head_idx_r * m_block_count
                head_idx = head_idx_kv * self.qhead_per_kvhead + head_idx_r
                pipeline_S.consumer_wait(consumer_state_S)
                pipeline_P.producer_acquire(producer_state_P)
                pipeline_LSE.consumer_wait(consumer_state_LSE)

                #### TMEM->RMEM (Load S from TMEM)
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.acc_dtype)
                cute.copy(thr_copy_t2r_smem, tStS_t2r, tSrS_t2r)
                cute.arch.fence_view_async_tmem_load()

                #### APPLY MASK (after score_mod, matching forward pass order)
                check_m_boundary = (m_block + 1) * self.tile_m > seqlen_q_cur_batch
                mask_fn(
                    tSrS_t2r,
                    m_block=m_block,
                    is_full_block=False,
                    check_m_boundary=check_m_boundary,
                )

                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------
                for v in cutlass.range(0, cute.size(tSrS_t2r), 2, unroll_full=True):
                    lse = (
                        -sLSE[
                            cute.get(tScS_t2r[v], mode=[1]),
                            consumer_state_LSE.index,
                        ],
                        -sLSE[
                            cute.get(tScS_t2r[v + 1], mode=[1]),
                            consumer_state_LSE.index,
                        ],
                    )
                    tSrS_t2r[v], tSrS_t2r[v + 1] = cute.arch.fma_packed_f32x2(
                        (tSrS_t2r[v], tSrS_t2r[v + 1]),
                        (softmax_scale_log2, softmax_scale_log2),
                        lse,
                    )
                    tSrS_t2r[v] = cute.math.exp2(tSrS_t2r[v], fastmath=True)
                    tSrS_t2r[v + 1] = cute.math.exp2(tSrS_t2r[v + 1], fastmath=True)

                tSrP = cute.make_rmem_tensor(tSrS_t2r.shape, mdV.element_type)
                utils.cvt_f16(tSrS_t2r, tSrP)
                self.reg_to_smem_mma128x128_2cta(
                    tSrP,
                    sP,
                    producer_state_P.index,
                    (self.tile_n, self.tile_m),
                    dp_idx,
                    wg_idx,
                )
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    pipeline_P.producer_commit(producer_state_P)
                producer_state_P.advance()

                with cute.arch.elect_one():
                    pipeline_S.consumer_release(consumer_state_S)
                consumer_state_S.advance()
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()

                # ---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                # ---------------------------------------------
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_dP)
                pipeline_dS.producer_acquire(producer_state_dS)

                ##### dS.T = P.T * (dP.T - Psum)
                tdPrdP_t2r = cute.make_rmem_tensor(tdPcdP_t2r.shape, self.acc_dtype)
                cute.copy(thr_copy_t2r_smem, tdPtdP_t2r, tdPrdP_t2r)

                for v in cutlass.range(0, cute.size(tdPrdP_t2r), 2, unroll_full=True):
                    dPsum_pair = (
                        -sdPsum[
                            cute.get(tdPcdP_t2r[v], mode=[1]),
                            consumer_state_dPsum.index,
                        ],
                        -sdPsum[
                            cute.get(tdPcdP_t2r[v + 1], mode=[1]),
                            consumer_state_dPsum.index,
                        ],
                    )
                    if cutlass.const_expr(varlen):
                        if not cute.elem_less(
                            cute.get(tdPcdP_t2r[v], mode=[1]), seqlen_q_cur_batch
                        ):
                            dPsum_pair = (0.0, dPsum_pair[1])
                        if not cute.elem_less(
                            cute.get(tdPcdP_t2r[v + 1], mode=[1]), seqlen_q_cur_batch
                        ):
                            dPsum_pair = (dPsum_pair[0], 0.0)
                    tdPrdP_t2r[v], tdPrdP_t2r[v + 1] = cute.arch.add_packed_f32x2(
                        (tdPrdP_t2r[v], tdPrdP_t2r[v + 1]), dPsum_pair
                    )
                    tdPrdP_t2r[v], tdPrdP_t2r[v + 1] = cute.arch.mul_packed_f32x2(
                        (tSrS_t2r[v], tSrS_t2r[v + 1]),
                        (tdPrdP_t2r[v], tdPrdP_t2r[v + 1]),
                    )
                tdPrdS = cute.make_rmem_tensor(tdPrdP_t2r.shape, mdV.element_type)
                utils.cvt_f16(tdPrdP_t2r, tdPrdS)

                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    pipeline_dP.consumer_release(consumer_state_dP)
                consumer_state_dP.advance()

                self.reg_to_smem_mma128x128_2cta(
                    tdPrdS,
                    sdSt,
                    producer_state_dS.index,
                    (self.tile_n, self.tile_m),
                    dp_idx,
                    wg_idx,
                )
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()

            # Epilogue
            if process_tile:
                consumer_state_dKV = self.epilogue_dKV(
                    tidx,
                    batch_idx,
                    head_idx_kv,
                    blk_coord_k,
                    seqlen,
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

                if cutlass.const_expr(True):
                    pipeline_P.producer_tail(producer_state_P)
                    pipeline_dS.producer_tail(producer_state_dS)

                cute.arch.barrier(
                    barrier_id=self.epilogue_sync_bar_id,
                    number_of_threads=self.num_compute_warps * self.threads_per_warp,
                )
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        return

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        for i in cutlass.range(cute.size(coord, mode=[2]), unroll_full=True):
            coord_i = coord[None, 0, i]
            gmem_i = gmem[None, 0, i]
            regs_i = regs[None, 0, i]
            if cute.elem_less(coord_i[0], tensor_shape):
                gmem_i.store(regs_i.load().to(gmem.element_type))

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV,
        consumer_state_dKV,
        softmax_scale: cutlass.Float32,
    ):
        num_wg = self.num_compute_warps // 4
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )

        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tdVtdV = tdVtdV[(None, None), 0, 0]
        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx % 128)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.domain_offset(
            (n_block * self.tile_n, 0),
            cute.make_identity_tensor((self.cta_tiler[0], self.cta_tiler[2])),
        )
        gdV = cute.local_tile(mdV_cur, (self.cta_tiler[0], self.cta_tiler[2]), (n_block, 0))
        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(cdV)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVgdV_t2r_p = thr_tmem_ld_dV.partition_D(gdV)
        tdVgdV_t2r = self.split_wg(tdVgdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_rmem_tensor(tdVcdV_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()
        self.store(tdVgdV_t2r, tdVrdV_t2r, tdVcdV_t2r, (seqlen.seqlen_k, mdV.shape[1]))

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tdKtdK = tdKtdK[(None, None), 0, 0]
        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx % 128)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.domain_offset(
            (n_block * self.tile_n, 0),
            cute.make_identity_tensor((self.cta_tiler[0], self.cta_tiler[2])),
        )
        gdK = cute.local_tile(mdK_cur, (self.cta_tiler[0], self.cta_tiler[2]), (n_block, 0))
        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(cdK)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKgdK_t2r_p = thr_tmem_ld_dK.partition_D(gdK)
        tdKgdK_t2r = self.split_wg(tdKgdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_rmem_tensor(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        for i in cutlass.range(cute.size(tdKrdK_t2r), unroll_full=True):
            tdKrdK_t2r[i] = softmax_scale * tdKrdK_t2r[i]
        cute.arch.fence_view_async_tmem_load()
        self.store(tdKgdK_t2r, tdKrdK_t2r, tdKcdK_t2r, (seqlen.seqlen_k, mdK.shape[1]))

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        return consumer_state_dKV

    @staticmethod
    def get_workspace_size(s_q: int, d: int, h: int, b: int, acc_dtype: type[cutlass.Numeric]):
        """Get workspace size."""
        d = (d + 7) // 8 * 8  # round up to 8
        s_q = (s_q + 7) // 8 * 8  # round up to 8
        workspace_bytes = 0
        # OdO vector
        workspace_bytes += acc_dtype.width // 8
        # scaled LSE vector
        workspace_bytes += acc_dtype.width // 8
        # FP32 versions of outputs that are churned (start off with Q only)
        workspace_bytes += d * acc_dtype.width // 8
        return (b, s_q, h, workspace_bytes)
