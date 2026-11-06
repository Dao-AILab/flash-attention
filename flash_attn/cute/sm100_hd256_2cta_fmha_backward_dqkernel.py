# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

from typing import Type, Tuple, Optional

import cuda.bindings.driver as cuda

import math
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Int64, Float32

from cutlass.utils import ClcDynamicPersistentTileScheduler
from flash_attn.cute.tile_scheduler import (
    SchedulerState,
    SingleTileVarlenScheduler,
    compute_sm100_fmha_varlen_grid,
    sm100_fmha_block_coord,
    compute_sm100_fmha_grid as compute_grid,
    compute_sm100_fmha_grid_clc as compute_grid_clc,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
    Sm100FmhaClcDynamicTileScheduler as FmhaClcDynamicTileScheduler,
    Sm100FmhaClcDynamicTileSchedulerParams as FmhaClcDynamicTileSchedulerParams,
)
from flash_attn.cute.mask import (
    Sm100FusedMask as FusedMask,
)
from flash_attn.cute.tile_scheduler import SM100_TMEM_CAPACITY_COLUMNS
import flash_attn.cute.copy_utils as fa_copy_utils
import cutlass.utils.blockscaled_layout as blockscaled_layout
from flash_attn.cute import sm100_hd256_mx_utils as mxu


class BlackwellFusedMultiHeadAttentionBackwardDQKernel:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_causal: bool,
        window_size_left: int | None,
        window_size_right: int | None,
        is_persistent: bool,
        split_head: bool,
        use_clc_scheduler: bool = False,
        mxfp8: bool = False,
    ):
        """mxfp8: NOTE [MXFP8 hd256 backward] in sm100_hd256_2cta_fmha_backward.py."""
        self.acc_dtype = acc_dtype
        self.mxfp8 = mxfp8
        assert not (mxfp8 and (use_clc_scheduler or split_head or is_causal))
        # MXFP8: the dK / dV kernel (independent of dQ) is launched as this kernel's programmatic
        # dependent, so its CTAs fill the SMs this kernel's last wave leaves idle
        self.pdl_primary = mxfp8
        self.mma_tiler = mma_tiler
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        # Keep original behavior (known-good in this repo)
        window_size_left = (
            None
            if (window_size_left is None or window_size_left < 0)
            else cutlass.Int32(window_size_left)
        )
        window_size_right = (
            None
            if (window_size_right is None or window_size_right < 0)
            else cutlass.Int32(window_size_right)
        )
        self.window_size_left = None if self.is_causal else window_size_left
        self.window_size_right = cutlass.Int32(0) if self.is_causal else window_size_right
        self.is_local = (not self.is_causal) and (
            self.window_size_left is not None or self.window_size_right is not None
        )
        assert mma_tiler[0] == 128 and mma_tiler[1] == 128, "Only 128x128 tile impl is supported"
        assert mma_tiler[2] == 256, "Only 256 is supported for 128x128 tile impl"
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = (
            2 * mma_tiler[0],
            mma_tiler[1],
            min(self.cta_tiler[2], 128) if split_head else self.cta_tiler[2],
        )
        self.dov_mma_tiler = self.qk_mma_tiler
        self.dsk_mma_tiler = (
            2 * mma_tiler[0],
            min(self.cta_tiler[2], 128) if split_head else self.cta_tiler[2],
            mma_tiler[1],
        )

        self.dsk_block_tiler = (
            self.dsk_mma_tiler[0] // 2,
            self.dsk_mma_tiler[1],
            self.dsk_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_dov = self.cta_tiler[2] // self.dov_mma_tiler[2]
        self.iterations_dsk = self.cta_tiler[2] // self.dsk_mma_tiler[1]
        self.cluster_shape_mn = (2, 1)
        self.tmem_warp_shape_mn = (4, 1)
        self.is_persistent = is_persistent
        self.use_clc_scheduler = use_clc_scheduler
        self.use_semantic_trip_range = self.is_causal or self.is_local

        # MXFP8 (one tile per CTA): the epilogue warps also run the S / dP -> dS step as a second compute warp
        # group (each group half of the 128 key columns), then store dQ; otherwise they idle until the epilogue.
        # Only on sm_107 (B300, B1/B4 S4k-S16k H16: 0.6% slower backward).
        self.wg2_compute = mxfp8 and mxu.is_sm107() and not is_persistent and not use_clc_scheduler
        self.compute_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7) if self.wg2_compute else (0, 1, 2, 3)
        self.epilogue_warp_ids = (4, 5, 6, 7)
        self.num_compute_wgs = len(self.compute_warp_ids) // 4
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (10, 11)
        self.sched_warp_id = self.empty_warp_id[0] if use_clc_scheduler else None
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.num_compute_warps = len(self.compute_warp_ids)

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.compute_warp_ids[:4],  # this is to get a round num threads
                *self.epilogue_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_id,
            )
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

        self.tmem_s_offset = 0
        self.tmem_dp_offset = 128
        self.tmem_dq_offset = 256
        # sm_107 (576-column exclusive TMEM allocation): the S / dP scales get dedicated columns
        # [512, 544), so QK(i) no longer waits for dP(i-1) to be read, nor dP(i) for S(i)
        self.dedicated_sf = mxfp8 and mxu.is_sm107()
        self.tmem_arch = "sm_107" if self.dedicated_sf else "sm_100"
        if self.dedicated_sf:
            self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.tmem_arch)
            self.tmem_sfq_offset = 512
            self.tmem_sfk_offset = 520
            self.tmem_sfdo_offset = 528
            self.tmem_sfv_offset = 536
            assert self.tmem_sfv_offset + 8 <= self.tmem_alloc_cols
        elif mxfp8:
            # No free TMEM columns: the S scales (Q, K) sit in the tail of dP (dS uses its first
            # 64 columns), copied once dP(i-1) has been read; the dP scales (dO, V) in the tail
            # of S, copied once S(i) has been read (NOTE [MXFP8 hd256 backward]).
            self.tmem_sfq_offset = self.tmem_dp_offset + 112
            self.tmem_sfk_offset = self.tmem_dp_offset + 120
            self.tmem_sfdo_offset = self.tmem_s_offset + 112
            self.tmem_sfv_offset = self.tmem_s_offset + 120

        self.num_regs_compute = (
            232 if self.wg2_compute else 256
        )  # 3 warp groups: 232 + 232 + 32 <= 512
        self.num_regs_epilogue = 160
        self.num_regs_other = 32

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = self.iterations_qk
        self.k_stage = self.iterations_qk
        self.do_stage = self.iterations_dov
        self.v_stage = self.iterations_dov
        self.kt_stage = 1
        self.qk_acc_stage = 1
        self.dov_acc_stage = 1
        self.dsk_acc_stage = 1
        self.epi_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        if cutlass.const_expr(self.use_clc_scheduler):
            self.num_clc_stage = 1
            self.num_clc_response_bytes = 16

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        dq_tensor: cute.Tensor,
        do_tensor: cute.Tensor,
        lse_tensor: cute.Tensor,
        sum_odo_tensor: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        seqused_q: Optional[cute.Tensor],
        seqused_k: Optional[cute.Tensor],
        scale_softmax: cutlass.Float32,
        max_seqlen_q: Optional[Int32],
        stream: cuda.CUstream,
        q_mx: Optional[cute.Tensor] = None,
        sfq_mx: Optional[cute.Tensor] = None,
        k_mx: Optional[cute.Tensor] = None,
        sfk_mx: Optional[cute.Tensor] = None,
        v_mx: Optional[cute.Tensor] = None,
        sfv_mx: Optional[cute.Tensor] = None,
        do_mx: Optional[cute.Tensor] = None,
        sfdo_mx: Optional[cute.Tensor] = None,
    ):
        """mxfp8: E4M3 Q / K / V / dO (b, s, h_k, h_r, d) with standard interleaved scales
        along d (raw buffers) for S = Q K^T and dP = dO V^T."""
        varlen = cum_seqlen_q is not None or cum_seqlen_k is not None
        # Infer shape metadata from normalized 5D tensors (B, S, H_k, H_r, D).
        s_q = q_tensor.shape[1]
        s_k = k_tensor.shape[1]
        d = q_tensor.shape[4]
        h_k = q_tensor.shape[2]
        h_r = q_tensor.shape[3]
        if cutlass.const_expr(cum_seqlen_q is not None):
            b = cum_seqlen_q.shape[0] - 1
        elif cutlass.const_expr(cum_seqlen_k is not None):
            b = cum_seqlen_k.shape[0] - 1
        else:
            b = q_tensor.shape[0]
        # `lse_tensor` / `sum_odo_tensor` are preallocated FP32 buffers (lse_log2 / dpsum)
        # whose sequence dimension is padded (rounded up) by the caller.
        # Use their leading dimension as the LSE length to ensure correct batch strides.
        s_lse = lse_tensor.shape[0]
        s_q64 = Int64(s_q)
        s_k64 = Int64(s_k)
        s_lse64 = Int64(s_lse)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        b64 = Int64(b)
        # Packed-varlen representation uses batch-dim = 1 and sequence-dim = total_{q,k}.
        # Keep the *physical* sequence extent in the tensor layouts so that applying
        # `cuseqlen_*` offsets stays within the tensor domain.
        s_q_total = q_tensor.shape[1] if cum_seqlen_q is not None else s_q64
        s_k_total = k_tensor.shape[1] if cum_seqlen_k is not None else s_k64
        b_lse = b64 if cum_seqlen_q is None else 1
        stride_b_lse = h_r64 * h_k64 * s_lse64 if cum_seqlen_q is None else 0

        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q_total, d, ((h_r, h_k), b)),
            stride=(
                cute.assume(q_tensor.stride[1], divby=64),
                q_tensor.stride[4],
                (
                    (q_tensor.stride[3], q_tensor.stride[2]),
                    0 if cum_seqlen_q is not None else cute.assume(q_tensor.stride[0], divby=64),
                ),
            ),
        )
        q = cute.make_tensor(q_tensor.iterator, q_layout)
        # (s, d, ((h_r, h_k), b))
        do_layout = cute.make_layout(
            (s_q_total, d, ((h_r, h_k), b)),
            stride=(
                cute.assume(do_tensor.stride[1], divby=64),
                do_tensor.stride[4],
                (
                    (do_tensor.stride[3], do_tensor.stride[2]),
                    0 if cum_seqlen_q is not None else cute.assume(do_tensor.stride[0], divby=64),
                ),
            ),
        )
        do = cute.make_tensor(do_tensor.iterator, do_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k_total, d, ((h_r, h_k), b)),
            stride=(
                cute.assume(k_tensor.stride[1], divby=64),
                k_tensor.stride[4],
                (
                    (0, k_tensor.stride[2]),
                    0 if cum_seqlen_k is not None else cute.assume(k_tensor.stride[0], divby=64),
                ),
            ),
        )
        k = cute.make_tensor(k_tensor.iterator, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        kt_layout = cute.make_layout(
            (d, s_k_total, ((h_r, h_k), b)),
            stride=(
                k_tensor.stride[4],
                cute.assume(k_tensor.stride[1], divby=64),
                (
                    (0, k_tensor.stride[2]),
                    0 if cum_seqlen_k is not None else cute.assume(k_tensor.stride[0], divby=64),
                ),
            ),
        )
        kt = cute.make_tensor(k_tensor.iterator, kt_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (s_k_total, d, ((h_r, h_k), b)),
            stride=(
                cute.assume(v_tensor.stride[1], divby=64),
                v_tensor.stride[4],
                (
                    (0, v_tensor.stride[2]),
                    0 if cum_seqlen_k is not None else cute.assume(v_tensor.stride[0], divby=64),
                ),
            ),
        )
        v = cute.make_tensor(v_tensor.iterator, v_layout)
        # (s, ((h_r, h_k), b))
        lse = cute.make_tensor(lse_tensor.iterator, lse_tensor.layout)
        # (s, ((h_r, h_k), b))
        sum_odo_layout = cute.make_layout(
            (s_lse64, ((h_r, h_k), b_lse)),
            stride=(1, ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
        )
        sum_odo = cute.make_tensor(sum_odo_tensor.iterator, sum_odo_layout)
        # (s, d, ((h_r, h_k), b))
        dq_layout = cute.make_layout(
            (s_q_total, d, ((h_r, h_k), b)),
            stride=(
                cute.assume(dq_tensor.stride[1], divby=64),
                dq_tensor.stride[4],
                (
                    (dq_tensor.stride[3], dq_tensor.stride[2]),
                    0 if cum_seqlen_q is not None else cute.assume(dq_tensor.stride[0], divby=64),
                ),
            ),
        )
        dq = cute.make_tensor(dq_tensor.iterator, dq_layout)

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.do_dtype = do.element_type
        self.dq_dtype = dq.element_type
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width

        self.use_varlen_scheduler = cum_seqlen_q is not None and max_seqlen_q is None
        assert not (self.use_varlen_scheduler and self.use_clc_scheduler), (
            "SM100 hd256 varlen dQ requires max_seqlen_q for grid sizing"
        )
        grid_q_extent = (
            max_seqlen_q if cutlass.const_expr(cum_seqlen_q is not None) else cute.size(dq.shape[0])
        )
        if cutlass.const_expr(self.use_varlen_scheduler):
            self.tile_sched_params, grid = compute_sm100_fmha_varlen_grid(
                dq.shape, cum_seqlen_q, self.cta_tiler[:2]
            )
        elif cutlass.const_expr(self.use_clc_scheduler):
            self.tile_sched_params, grid = compute_grid_clc(
                (grid_q_extent, dq.shape[1], dq.shape[2]),
                self.cta_tiler,
                (*self.cluster_shape_mn, 1),
            )
        else:
            self.tile_sched_params, grid = compute_grid(
                (grid_q_extent, dq.shape[1], dq.shape[2]),
                self.cta_tiler,
                self.is_persistent,
            )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.do_major_mode = utils.LayoutEnum.from_tensor(do).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.dq_layout = utils.LayoutEnum.from_tensor(dq)

        if cutlass.const_expr(self.q_major_mode != cute.nvgpu.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != cute.nvgpu.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != cute.nvgpu.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.do_major_mode != cute.nvgpu.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if cutlass.const_expr(self.q_dtype != self.do_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.do_dtype}")

        self._setup_attributes()

        # MXFP8: the S / dP operands are E4M3 copies (dsk keeps BF16 K^T and dS)
        self.sdp_dtype = self.q_dtype
        mSFQ_tma = mSFK_tma = mSFdO_tma = mSFV_tma = None
        if cutlass.const_expr(self.mxfp8):
            assert not varlen and seqused_q is None and seqused_k is None
            self.sdp_dtype = mxu.MX_DTYPE
            hb = q.shape[2]
            q, do = mxu.mx_sd_view(q_mx, hb, False), mxu.mx_sd_view(do_mx, hb, False)
            k, v = mxu.mx_sd_view(k_mx, hb, True), mxu.mx_sd_view(v_mx, hb, True)
            n_q = cute.ceil_div(q.shape[0], 128)
            n_k = cute.ceil_div(k.shape[0], 128)
            mSFQ_tma, mSFdO_tma = [
                mxu.mx_sf_tile_view(t, 64, n_q, hb, h_r, h_k, False) for t in (sfq_mx, sfdo_mx)
            ]
            mSFK_tma, mSFV_tma = [
                mxu.mx_sf_tile_view(t, 64, n_k, hb, h_r, h_k, True) for t in (sfk_mx, sfv_mx)
            ]

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from tmem & k-major
        ds_source = tcgen05.OperandSource.TMEM
        ds_major_mode = cute.nvgpu.OperandMajorMode.K
        k_trans_major_mode = cute.nvgpu.OperandMajorMode.MN
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        dov_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_dtype,
            self.do_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.dov_mma_tiler[:2],
        )
        dsk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            ds_major_mode,
            k_trans_major_mode,
            self.acc_dtype,
            cta_group,
            self.dsk_mma_tiler[:2],
            ds_source,
        )

        sSFA_layout = sSFB_layout = tSFA_layout = tSFB_layout = None
        if cutlass.const_expr(self.mxfp8):
            qk_tiled_mma, dov_tiled_mma = [
                mxu.make_mx_tiled_mma(
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                    cta_group,
                    tiler[:2],
                )
                for tiler in (self.qk_mma_tiler, self.dov_mma_tiler)
            ]
            # standard 128-row scale chunks (M=256 pair MMA, N=128): 1024 B per operand tile
            sSFA_layout = blockscaled_layout.make_smem_layout_sfa(
                qk_tiled_mma, self.qk_mma_tiler, mxu.SF_VEC, 1
            )
            sSFB_layout = blockscaled_layout.make_smem_layout_sfb(
                qk_tiled_mma, self.qk_mma_tiler, mxu.SF_VEC, self.k_stage
            )
            assert self.k_stage == self.v_stage
            tSFA_layout = blockscaled_layout.make_tmem_layout_sfa(
                qk_tiled_mma,
                self.qk_mma_tiler,
                mxu.SF_VEC,
                cute.slice_(sSFA_layout, (None, None, None, 0)),
            )
            tSFB_layout = blockscaled_layout.make_tmem_layout_sfb(
                qk_tiled_mma,
                self.qk_mma_tiler,
                mxu.SF_VEC,
                cute.slice_(sSFB_layout, (None, None, None, 0)),
            )
            assert cute.cosize(sSFA_layout) == 1024
            assert cute.cosize(sSFB_layout) == 1024 * self.k_stage

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.dsk_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.sdp_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.sdp_dtype,
            self.k_stage,
        )
        do_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dov_tiled_mma,
            self.dov_mma_tiler,
            self.sdp_dtype,
            self.do_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dov_tiled_mma,
            self.dov_mma_tiler,
            self.sdp_dtype,
            self.v_stage,
        )
        ds_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            dsk_tiled_mma,
            self.dsk_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        ds_tmem_layout = cute.select(ds_tmem_layout_staged, mode=[0, 1, 2])
        kt_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dsk_tiled_mma,
            self.dsk_mma_tiler,
            self.k_dtype,
            self.dsk_acc_stage,
        )

        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for dO
        do_smem_layout = cute.select(do_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            do,
            do_smem_layout,
            self.dov_mma_tiler,
            dov_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.dov_mma_tiler,
            dov_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for KT
        kt_smem_layout = cute.select(kt_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kt,
            kt_smem_layout,
            self.dsk_mma_tiler,
            dsk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        lse_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_LSE_stage))
        sum_odo_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_sum_OdO_stage))

        q_copy_size = cute.size_in_bytes(self.sdp_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.sdp_dtype, k_smem_layout)
        do_copy_size = cute.size_in_bytes(self.sdp_dtype, do_smem_layout)
        v_copy_size = cute.size_in_bytes(self.sdp_dtype, v_smem_layout)
        tma_atom_sfq = tma_tensor_sfq = tma_atom_sfk = tma_tensor_sfk = None
        tma_atom_sfdo = tma_tensor_sfdo = tma_atom_sfv = tma_tensor_sfv = None
        if cutlass.const_expr(self.mxfp8):
            # one plain 2CTA TMA per CTA and 1024 B scale tile; Q / dO: this CTA's 128 queries,
            # K / V: all 128 keys (each CTA's TMEM needs the full N)
            sf_sl = cute.make_layout((16, 64), stride=(1, 16))
            tma_atom_sfq, tma_tensor_sfq = cpasync.make_tiled_tma_atom(
                tma_load_op, mSFQ_tma, sf_sl, (16, 64)
            )
            tma_atom_sfk, tma_tensor_sfk = cpasync.make_tiled_tma_atom(
                tma_load_op, mSFK_tma, sf_sl, (16, 64)
            )
            tma_atom_sfdo, tma_tensor_sfdo = cpasync.make_tiled_tma_atom(
                tma_load_op, mSFdO_tma, sf_sl, (16, 64)
            )
            tma_atom_sfv, tma_tensor_sfv = cpasync.make_tiled_tma_atom(
                tma_load_op, mSFV_tma, sf_sl, (16, 64)
            )
            q_copy_size += 1024
            k_copy_size += 1024
            do_copy_size += 1024
            v_copy_size += 1024
        kt_copy_size = cute.size_in_bytes(self.k_dtype, kt_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_k_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_do_bytes = do_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_v_bytes = v_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kt_bytes = kt_copy_size * cute.size(qk_tiled_mma.thr_id.shape)

        # dQ epilogue TMA store. Use a single (M, 64) SMEM stage and rotate it
        # across the hd=256 N dimension to stay within the SMEM budget.
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        epi_cols_dQ = math.gcd(128 // (dq.element_type.width // 8), self.epi_tile[1])
        epi_tile_dQ = (self.epi_tile[0], epi_cols_dQ)
        sdQ_epi_layout = sm100_utils.make_smem_layout_epi(
            dq.element_type,
            self.dq_layout,
            epi_tile_dQ,
            1,
        )
        tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dq,
            cute.select(sdQ_epi_layout, mode=[0, 1]),
            epi_tile_dQ,
        )

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[
                Int64, self.q_stage * 2
            ]  # load_q_{producer,consumer}
            load_do_mbar_ptr: cute.struct.MemRange[
                Int64, self.do_stage * 2
            ]  # load_do_{producer,consumer}
            load_k_mbar_ptr: cute.struct.MemRange[
                Int64, self.k_stage * 2
            ]  # load_k_{producer,consumer}
            load_kt_mbar_ptr: cute.struct.MemRange[
                Int64, self.kt_stage * 2
            ]  # load_kt_{producer,consumer}
            load_v_mbar_ptr: cute.struct.MemRange[
                Int64, self.v_stage * 2
            ]  # load_v_{producer,consumer}
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            mma_dp_mbar_ptr: cute.struct.MemRange[Int64, self.dov_acc_stage * 2]
            mma_dq_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            ds_mma_mbar_ptr: cute.struct.MemRange[Int64, self.dsk_acc_stage * 2]
            lse_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_LSE_stage * 2]
            sum_odo_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_sum_OdO_stage * 2
            ]
            # A CTA-wide "TMEM lifetime" barrier used to safely deallocate TMEM after all users finish.
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # CLC pipeline barriers and response buffer
            clc_mbar_ptr: cute.struct.MemRange[Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            dov_tiled_mma,
            dsk_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_kt,
            tma_tensor_kt,
            tma_atom_dQ,
            tma_tensor_dQ,
            lse,
            sum_odo,
            dq,
            cum_seqlen_q,
            cum_seqlen_k,
            seqused_q,
            seqused_k,
            scale_softmax,
            self.window_size_left,
            self.window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            v_smem_layout_staged,
            do_smem_layout_staged,
            kt_smem_layout_staged,
            ds_tmem_layout,
            sdQ_epi_layout,
            lse_smem_layout,
            sum_odo_smem_layout,
            self.tile_sched_params,
            tma_atom_sfq,
            tma_tensor_sfq,
            tma_atom_sfk,
            tma_tensor_sfk,
            tma_atom_sfdo,
            tma_tensor_sfdo,
            tma_atom_sfv,
            tma_tensor_sfv,
            sSFA_layout,
            sSFB_layout,
            tSFA_layout,
            tSFB_layout,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        dov_tiled_mma: cute.TiledMma,
        dsk_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        mdO_qdl: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        mK_dkl: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        mdQ_tma: cute.Tensor,
        mLSE: cute.Tensor,
        mSum_OdO: cute.Tensor,
        mdQ_qdl: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        seqused_q: Optional[cute.Tensor],
        seqused_k: Optional[cute.Tensor],
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        do_smem_layout_staged: cute.ComposedLayout,
        kt_smem_layout_staged: cute.ComposedLayout,
        ds_tmem_layout_staged: cute.ComposedLayout,
        sdQ_epi_layout: cute.ComposedLayout,
        lse_smem_layout: cute.Layout,
        sum_odo_smem_layout: cute.Layout,
        tile_sched_params: (
            FmhaStaticTileSchedulerParams
            | FmhaClcDynamicTileSchedulerParams
            | SingleTileVarlenScheduler.Params
        ),
        tma_atom_sfq: Optional[cute.CopyAtom] = None,
        mSFQ: Optional[cute.Tensor] = None,
        tma_atom_sfk: Optional[cute.CopyAtom] = None,
        mSFK: Optional[cute.Tensor] = None,
        tma_atom_sfdo: Optional[cute.CopyAtom] = None,
        mSFdO: Optional[cute.Tensor] = None,
        tma_atom_sfv: Optional[cute.CopyAtom] = None,
        mSFV: Optional[cute.Tensor] = None,
        sSFA_layout: Optional[cute.Layout] = None,
        sSFB_layout: Optional[cute.Layout] = None,
        tSFA_layout: Optional[cute.Layout] = None,
        tSFB_layout: Optional[cute.Layout] = None,
    ):
        # llvm.inline_asm(
        #     None,
        #     [],
        #     '.pragma "global knob CommonIntoMultiBlockLoop=1";',
        #     "",
        #     has_side_effects=True,
        #     is_align_stack=False,
        #     asm_dialect=llvm.AsmDialect.AD_ATT,
        # )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if cutlass.const_expr(self.pdl_primary):
            cute.arch.griddepcontrol_launch_dependents()
        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_do)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_dQ)
            if cutlass.const_expr(self.mxfp8):
                for atom in (tma_atom_sfq, tma_atom_sfk, tma_atom_sfdo, tma_atom_sfv):
                    cute.nvgpu.cpasync.prefetch_descriptor(atom)

        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        varlen = cum_seqlen_q is not None or cum_seqlen_k is not None
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_do_producer, load_do_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.do_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_do_bytes,
            barrier_storage=storage.load_do_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_kt_producer, load_kt_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kt_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kt_bytes,
            barrier_storage=storage.load_kt_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_dp_producer, mma_dp_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.dov_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_dp_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        ds_mma_producer, ds_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.dsk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.ds_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_dq_producer, mma_dq_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.epilogue_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_dq_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        load_lse_producer, load_lse_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_LSE_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * self.num_compute_warps
            ),
            barrier_storage=storage.lse_mbar_ptr.data_ptr(),
        ).make_participants()
        load_sum_odo_producer, load_sum_odo_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * self.num_compute_warps
            ),
            barrier_storage=storage.sum_odo_mbar_ptr.data_ptr(),
        ).make_participants()

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.tmem_arch,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # Initialize CLC state if using dynamic scheduler
        if cutlass.const_expr(self.use_clc_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cluster_size = cute.size(self.cluster_shape_mnk)
            num_clc_consumer_threads = self.threads_per_warp * (
                1  # sched_warp (CTA 0 only)
                + cluster_size
                * (
                    len(self.compute_warp_ids)
                    + len(self.epilogue_warp_ids)
                    + 1  # mma_warp
                    + 1  # load_warp
                )
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc = SchedulerState.create_clc(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_sched_params.clc_hw_params(),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                    num_stages=self.num_clc_stage,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=self.num_clc_response_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_clc_stage
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_clc_stage
                ),
            )
        else:
            clc = None
            clc_response_ptr = None

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.sdp_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.sdp_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        # K and V now use separate memory since we removed the transform stage
        sV = smem.allocate_tensor(
            element_type=self.sdp_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sdO = smem.allocate_tensor(
            element_type=self.sdp_dtype,
            layout=do_smem_layout_staged.outer,
            swizzle=do_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sKT = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=kt_smem_layout_staged.outer,
            swizzle=kt_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sLSE = smem.allocate_tensor(
            element_type=self.acc_dtype,
            layout=lse_smem_layout,
            byte_alignment=128,
        )
        sSum_OdO = smem.allocate_tensor(
            element_type=self.acc_dtype,
            layout=sum_odo_smem_layout,
            byte_alignment=128,
        )
        sSFQ = sSFK = sSFdO = sSFV = None
        if cutlass.const_expr(self.mxfp8):
            sSFQ, sSFdO = [
                smem.allocate_tensor(
                    element_type=mxu.SF_DTYPE, layout=sSFA_layout, byte_alignment=128
                )
                for _ in range(2)
            ]
            sSFK, sSFV = [
                smem.allocate_tensor(
                    element_type=mxu.SF_DTYPE, layout=sSFB_layout, byte_alignment=128
                )
                for _ in range(2)
            ]
        # Alias the dQ TMA epilogue staging buffer onto sdO.  A standalone
        # (128, 64) bf16 allocation exceeds B200's per-block SMEM limit for
        # this kernel, while sdO is no longer needed by the epilogue warp once
        # the dQ accumulator is ready to store.
        s_epi_dQ = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, sdQ_epi_layout.inner, self.dq_dtype),
            sdQ_epi_layout.outer,
        )
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        dov_thr_mma = dov_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        dsk_thr_mma = dsk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tdPrdO = dov_thr_mma.make_fragment_A(sdO)
        tdPrV = dov_thr_mma.make_fragment_B(sV)
        tdQrKT = dsk_thr_mma.make_fragment_B(sKT)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        dov_acc_shape = dov_thr_mma.partition_shape_C(
            (self.dov_mma_tiler[0], self.dov_mma_tiler[1])
        )
        tdPtdP = dov_thr_mma.make_fragment_C(cute.append(dov_acc_shape, self.dov_acc_stage))
        dsk_acc_shape = dsk_thr_mma.partition_shape_C(
            (self.dsk_mma_tiler[0], self.dsk_mma_tiler[1])
        )
        tdQtdQ = dsk_thr_mma.make_fragment_C(dsk_acc_shape)
        tdQtdQ_layout = cute.append(
            tdQtdQ.layout,
            cute.make_layout(
                self.iterations_dsk,
                stride=self.dsk_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_dp_offset, tdPtdP.layout)
        tdQtdQ_staged = cute.make_tensor(tdQtdQ.iterator + self.tmem_dq_offset, tdQtdQ_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        if cutlass.const_expr(self.use_clc_scheduler):
            tile_sched = FmhaClcDynamicTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
                clc,
            )
        elif cutlass.const_expr(self.use_varlen_scheduler):
            tile_sched = SingleTileVarlenScheduler.create(tile_sched_params)
        else:
            blk_idx = cute.arch.block_idx()
            tile_sched = FmhaStaticTileScheduler(
                tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
            )
        work_tile = tile_sched.initial_work_tile_info()

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            while work_tile.is_valid_tile:
                curr_block_coord = sm100_fmha_block_coord(work_tile, self.use_varlen_scheduler)
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                is_valid_q = True
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                # Used lengths control masks and work, while cu_seqlens retain
                # physical offsets. All four warp roles must agree on has_work.
                if cutlass.const_expr(seqused_q is not None):
                    seqlen_q = seqused_q[batch_coord]
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                if cutlass.const_expr(cum_seqlen_q is not None or seqused_q is not None):
                    is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                    FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                )
                is_valid_k = seqlen_kv_loop_steps > 0
                has_work = is_valid_q and is_valid_k

                if has_work:
                    block_offset = (
                        cuseqlen_q,
                        cuseqlen_k,
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    mQ_qdl_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl)
                    mK_kdl_ = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), mK_kdl)
                    mdO_qdl_ = cute.domain_offset(
                        cute.select(block_offset, mode=[0, 2, 3]), mdO_qdl
                    )
                    mV_dkl_ = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), mV_dkl)
                    mK_dkl_ = cute.domain_offset(cute.select(block_offset, mode=[2, 1, 3]), mK_dkl)
                    block_offset_stats = block_offset
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        cuseqlen_q_stats = cute.assume(
                            (cuseqlen_q + batch_coord * self.cta_tiler[0])
                            // self.cta_tiler[0]
                            * self.cta_tiler[0],
                            divby=self.cta_tiler[0],
                        )
                        block_offset_stats = (
                            cuseqlen_q_stats,
                            block_offset[1],
                            block_offset[2],
                            block_offset[3],
                        )
                    LSE = cute.domain_offset(cute.select(block_offset_stats, mode=[0, 3]), mLSE)
                    sum_OdO = cute.domain_offset(
                        cute.select(block_offset_stats, mode=[0, 3]), mSum_OdO
                    )

                    # Local tile partition global tensors
                    q_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                    )
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        block_in_cluster_coord_vmnk[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    k_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    )
                    gK_kdl = cute.flat_divide(mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        block_in_cluster_coord_vmnk[1],
                        k_cta_layout,
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    do_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                    )
                    # (bM, bK, loopM, loopK, loopL)
                    gdO_qdl = cute.flat_divide(
                        mdO_qdl_, cute.select(self.dov_mma_tiler, mode=[0, 2])
                    )
                    tdPgdO_qdl = dov_thr_mma.partition_A(gdO_qdl)
                    tdOsdO, tdOgdO_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_do,
                        block_in_cluster_coord_vmnk[2],
                        do_cta_layout,
                        cute.group_modes(sdO, 0, 3),
                        cute.group_modes(tdPgdO_qdl, 0, 3),
                    )
                    v_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                    )
                    gV_dkl = cute.flat_divide(mV_dkl_, cute.select(self.dov_mma_tiler, mode=[1, 2]))
                    tSgV_dkl = dov_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        block_in_cluster_coord_vmnk[1],
                        v_cta_layout,
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_dkl, 0, 3),
                    )
                    # kt layout
                    kt_cta_layout = cute.make_layout(
                        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                    )
                    gK_dkl = cute.flat_divide(mK_dkl_, cute.select(self.dsk_mma_tiler, mode=[1, 2]))
                    tdQgK_dkl = dsk_thr_mma.partition_B(gK_dkl)
                    tKTsKT, tKgK_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_kt,
                        block_in_cluster_coord_vmnk[1],
                        kt_cta_layout,
                        cute.group_modes(sKT, 0, 3),
                        cute.group_modes(tdQgK_dkl, 0, 3),
                    )
                    load_sfq = load_sfk = load_sfdo = load_sfv = None
                    if cutlass.const_expr(self.mxfp8):
                        # MXFP8: one plain TMA per CTA and 1024 B scale tile
                        sf_loads = []
                        for atom, mSF, sSF in (
                            (tma_atom_sfq, mSFQ, sSFQ),
                            (tma_atom_sfk, mSFK, sSFK),
                            (tma_atom_sfdo, mSFdO, sSFdO),
                            (tma_atom_sfv, mSFV, sSFV),
                        ):
                            sSF_bytes = cute.make_tensor(
                                cute.recast_ptr(sSF.iterator, dtype=cutlass.Uint8),
                                cute.make_layout(
                                    (16, 64, cute.size(sSF.layout, mode=[3])),
                                    stride=(1, 16, 1024),
                                ),
                            )
                            load_sf, _, _ = fa_copy_utils.tma_get_copy_fn(
                                atom,
                                cta_coord=0,
                                cta_layout=cute.make_layout(1),
                                src_tensor=mSF[None, None, None, curr_block_coord[2]],
                                dst_tensor=sSF_bytes,
                            )
                            sf_loads.append(load_sf)
                        load_sfq, load_sfk, load_sfdo, load_sfv = sf_loads
                    # ((atom_v, rest_v), RestK)
                    tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestK)
                    tdOgdO = tdOgdO_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tKTgKT = tKgK_dkl[None, None, None, mma_block_coord[2]]

                    # LSE
                    lse_handle = load_lse_producer.acquire_and_advance()
                    # 32 threads loading 128 values of 32b each
                    # so 4*32b = 128b
                    thread_idx = tidx % self.threads_per_warp
                    async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
                    atom_async_copy = cute.make_copy_atom(
                        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )
                    sLSE_for_copy = cute.flat_divide(sLSE, (1,))
                    LSE_for_copy = cute.flat_divide(LSE, (1,))
                    for i in cutlass.range_constexpr(async_copy_num_elts):
                        LSE_idx = (
                            self.cta_tiler[0] * curr_block_coord[0]
                            + thread_idx * async_copy_num_elts
                        )
                        if cute.elem_less(LSE_idx + i, seqlen_q):
                            cute.copy(
                                atom_async_copy,
                                LSE_for_copy[None, LSE_idx + i, curr_block_coord[2]],
                                sLSE_for_copy[
                                    None,
                                    thread_idx * async_copy_num_elts + i,
                                    lse_handle.index,
                                ],
                            )
                        else:
                            sLSE_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                lse_handle.index,
                            ].fill(0.0)
                    lse_handle.commit()

                    sum_odo_handle = load_sum_odo_producer.acquire_and_advance()
                    sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
                    sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
                    for i in cutlass.range_constexpr(async_copy_num_elts):
                        sum_OdO_idx = (
                            self.cta_tiler[0] * curr_block_coord[0]
                            + thread_idx * async_copy_num_elts
                        )
                        if cute.elem_less(sum_OdO_idx + i, seqlen_q):
                            cute.copy(
                                atom_async_copy,
                                sum_OdO_for_copy[None, sum_OdO_idx + i, curr_block_coord[2]],
                                sSum_OdO_for_copy[
                                    None,
                                    thread_idx * async_copy_num_elts + i,
                                    sum_odo_handle.index,
                                ],
                            )
                        else:
                            sSum_OdO_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                sum_odo_handle.index,
                            ].fill(0.0)
                    sum_odo_handle.commit()

                    # Q
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, iter],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )
                        if cutlass.const_expr(self.mxfp8):
                            load_sfq(curr_block_coord[0], 0, tma_bar_ptr=q_handle.barrier)
                    # dO
                    for iter in cutlass.range(self.iterations_dov, unroll=1):
                        do_handle = load_do_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_do,
                            tdOgdO[None, iter],
                            tdOsdO[None, do_handle.index],
                            tma_bar_ptr=do_handle.barrier,
                        )
                        if cutlass.const_expr(self.mxfp8):
                            load_sfdo(curr_block_coord[0], 0, tma_bar_ptr=do_handle.barrier)

                    kv_coord = seqlen_kv_loop_start
                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            k_handle = load_k_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, kv_coord, iter],
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                            if cutlass.const_expr(self.mxfp8):
                                load_sfk(kv_coord, k_handle.index, tma_bar_ptr=k_handle.barrier)
                        # Vi
                        for iter in cutlass.range(self.iterations_dov, unroll=1):
                            v_handle = load_v_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, kv_coord, iter],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                            if cutlass.const_expr(self.mxfp8):
                                load_sfv(kv_coord, v_handle.index, tma_bar_ptr=v_handle.barrier)
                        # KTi
                        for iter in cutlass.range(self.iterations_dsk, unroll=1):
                            kt_handle = load_kt_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_kt,
                                tKTgKT[None, iter, kv_coord],
                                tKTsKT[None, kt_handle.index],
                                tma_bar_ptr=kt_handle.barrier,
                            )
                        kv_coord += 1

                work_tile = tile_sched.advance_to_next_work()
                # End of persistent scheduler loop
            load_k_producer.tail()
            load_v_producer.tail()
            load_kt_producer.tail()
            load_q_producer.tail()
            load_do_producer.tail()
            load_lse_producer.tail()
            load_sum_odo_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_cta = cta_rank_in_cluster % 2 == 0
            sf_qk = sf_dov = None
            if cutlass.const_expr(self.mxfp8):
                cta_group = tcgen05.CtaGroup.TWO
                sf_qk = (
                    mxu.make_sf_s2t(sSFQ, tSFA_layout, self.tmem_sfq_offset, cta_group),
                    mxu.make_sf_s2t(sSFK, tSFB_layout, self.tmem_sfk_offset, cta_group),
                )
                sf_dov = (
                    mxu.make_sf_s2t(sSFdO, tSFA_layout, self.tmem_sfdo_offset, cta_group),
                    mxu.make_sf_s2t(sSFV, tSFB_layout, self.tmem_sfv_offset, cta_group),
                )

            while work_tile.is_valid_tile:
                curr_block_coord = sm100_fmha_block_coord(work_tile, self.use_varlen_scheduler)
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                batch_coord = curr_block_coord[2][1]
                is_valid_q = True
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                if cutlass.const_expr(seqused_q is not None):
                    seqlen_q = seqused_q[batch_coord]
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                if cutlass.const_expr(cum_seqlen_q is not None or seqused_q is not None):
                    is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                    FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                )
                is_valid_k = seqlen_kv_loop_steps > 0
                has_work = is_valid_q and is_valid_k

                if has_work:
                    # dq_handle = mma_dq_producer.acquire_and_advance()
                    load_q_releaser = load_q_consumer.clone()
                    load_do_releaser = load_do_consumer.clone()
                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    num_innerloop = 8

                    if is_leader_cta:
                        dq_handle = mma_dq_producer.acquire_and_advance()
                        if seqlen_kv_loop_steps > 1:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]

                                k_handle = load_k_consumer.wait_and_advance()
                                if cutlass.const_expr(self.mxfp8):
                                    # dP(i-1) read: the S scales go to its tail
                                    if cutlass.const_expr(not self.dedicated_sf):
                                        mma_dp_producer.acquire()
                                    self.copy_sf(sf_qk, k_handle.index)
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(qk_tiled_mma, sf_qk, kphase_coord[2])
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        if cutlass.const_expr(self.mxfp8):
                                            self.set_sf(qk_tiled_mma, sf_qk, kphase_coord[2])
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            cute.arch.fence_view_async_tmem_store()
                            s_handle.commit()

                            # dOV0
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_dov, unroll=1):
                                load_do_consumer.wait_and_advance()
                                tdPrdO_slice = tdPrdO[None, None, None, iter]
                                v_handle = load_v_consumer.wait_and_advance()
                                if cutlass.const_expr(self.mxfp8):
                                    # S(i) read: the dP scales go to its tail
                                    if cutlass.const_expr(not self.dedicated_sf):
                                        mma_s_producer.acquire()
                                    self.copy_sf(sf_dov, v_handle.index)
                                tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(dov_tiled_mma, sf_dov, kphase_coord[2])
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        if cutlass.const_expr(self.mxfp8):
                                            self.set_sf(dov_tiled_mma, sf_dov, kphase_coord[2])
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                            cute.arch.fence_view_async_tmem_store()
                            dp_handle.commit()

                            for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                                # QKi
                                s_handle = mma_s_producer.acquire_and_advance()

                                tStS_slice = tStS[None, None, None, s_handle.index]
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_qk, unroll=1):
                                    tSrQ_slice = tSrQ[None, None, None, iter]
                                    k_handle = load_k_consumer.wait_and_advance()
                                    if cutlass.const_expr(self.mxfp8):
                                        # dP(i-1) read: the S scales go to its tail
                                        if cutlass.const_expr(not self.dedicated_sf):
                                            mma_dp_producer.acquire()
                                        self.copy_sf(sf_qk, k_handle.index)
                                    tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                    num_kphases = cute.size(tSrQ_slice, mode=[2])
                                    if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                        num_outer_iter = num_kphases // num_innerloop
                                        for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                            for kphase_idx in cutlass.range(
                                                num_innerloop, unroll_full=True
                                            ):
                                                kphase_coord = (
                                                    None,
                                                    None,
                                                    outer_iter * num_innerloop + kphase_idx,
                                                )
                                                if cutlass.const_expr(self.mxfp8):
                                                    self.set_sf(
                                                        qk_tiled_mma, sf_qk, kphase_coord[2]
                                                    )
                                                cute.gemm(
                                                    qk_tiled_mma,
                                                    tStS_slice,
                                                    tSrQ_slice[kphase_coord],
                                                    tSrK_trans_slice[kphase_coord],
                                                    tStS_slice,
                                                )
                                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    else:
                                        for kphase_idx in cutlass.range(
                                            num_kphases, unroll_full=True
                                        ):
                                            kphase_coord = (None, None, kphase_idx)
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(qk_tiled_mma, sf_qk, kphase_coord[2])
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    k_handle.release()
                                s_handle.commit()

                                # dSKTi
                                ds_handle = ds_mma_consumer.wait_and_advance()
                                dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                                for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                    kt_handle = load_kt_consumer.wait_and_advance()
                                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                    tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                    tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                    tdS = cute.make_tensor(
                                        tdStdS_slice.iterator, ds_tmem_layout_staged.outer
                                    )
                                    tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                    tdQrdS_slice = cute.make_tensor(
                                        cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                        tdQrdS.layout,
                                    )

                                    tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                    num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                    if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                        num_outer_iter = num_kphases // num_innerloop
                                        for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                            for kphase_idx in cutlass.range(
                                                num_innerloop, unroll_full=True
                                            ):
                                                kphase_coord = (
                                                    None,
                                                    None,
                                                    outer_iter * num_innerloop + kphase_idx,
                                                )
                                                cute.gemm(
                                                    dsk_tiled_mma,
                                                    tdQtdQ_slice,
                                                    tdQrdS_slice[kphase_coord],
                                                    tdQrKT_slice[kphase_coord],
                                                    tdQtdQ_slice,
                                                )
                                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    else:
                                        for kphase_idx in cutlass.range(
                                            num_kphases, unroll_full=True
                                        ):
                                            kphase_coord = (None, None, kphase_idx)
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    kt_handle.release()
                                ds_handle.release()

                                # dOVi
                                dp_handle = mma_dp_producer.acquire_and_advance()
                                tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                                dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_dov, unroll=1):
                                    tdPrdO_slice = tdPrdO[None, None, None, iter]
                                    v_handle = load_v_consumer.wait_and_advance()
                                    if cutlass.const_expr(self.mxfp8):
                                        # S(i) read: the dP scales go to its tail
                                        if cutlass.const_expr(not self.dedicated_sf):
                                            mma_s_producer.acquire()
                                        self.copy_sf(sf_dov, v_handle.index)
                                    tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                    num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                    if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                        num_outer_iter = num_kphases // num_innerloop
                                        for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                            for kphase_idx in cutlass.range(
                                                num_innerloop, unroll_full=True
                                            ):
                                                kphase_coord = (
                                                    None,
                                                    None,
                                                    outer_iter * num_innerloop + kphase_idx,
                                                )
                                                if cutlass.const_expr(self.mxfp8):
                                                    self.set_sf(
                                                        dov_tiled_mma, sf_dov, kphase_coord[2]
                                                    )
                                                cute.gemm(
                                                    dov_tiled_mma,
                                                    tdPtdP_slice,
                                                    tdPrdO_slice[kphase_coord],
                                                    tdPrV_trans_slice[kphase_coord],
                                                    tdPtdP_slice,
                                                )
                                                dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    else:
                                        for kphase_idx in cutlass.range(
                                            num_kphases, unroll_full=True
                                        ):
                                            kphase_coord = (None, None, kphase_idx)
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(dov_tiled_mma, sf_dov, kphase_coord[2])
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    v_handle.release()
                                dp_handle.commit()

                            # QKend
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_k_consumer.wait_and_advance()
                                if cutlass.const_expr(self.mxfp8):
                                    # dP(i-1) read: the S scales go to its tail
                                    if cutlass.const_expr(not self.dedicated_sf):
                                        mma_dp_producer.acquire()
                                    self.copy_sf(sf_qk, k_handle.index)

                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]

                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(qk_tiled_mma, sf_qk, kphase_coord[2])
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        if cutlass.const_expr(self.mxfp8):
                                            self.set_sf(qk_tiled_mma, sf_qk, kphase_coord[2])
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            # dSKTend - 1
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                kt_handle = load_kt_consumer.wait_and_advance()
                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                tdS = cute.make_tensor(
                                    tdStdS_slice.iterator, ds_tmem_layout_staged.outer
                                )
                                tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                tdQrdS_slice = cute.make_tensor(
                                    cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                    tdQrdS.layout,
                                )

                                tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                kt_handle.release()
                            ds_handle.release()

                            # dOVend
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_dov, unroll=1):
                                tdPrdO_slice = tdPrdO[None, None, None, iter]
                                v_handle = load_v_consumer.wait_and_advance()
                                if cutlass.const_expr(self.mxfp8):
                                    # S(i) read: the dP scales go to its tail
                                    if cutlass.const_expr(not self.dedicated_sf):
                                        mma_s_producer.acquire()
                                    self.copy_sf(sf_dov, v_handle.index)
                                tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(dov_tiled_mma, sf_dov, kphase_coord[2])
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        if cutlass.const_expr(self.mxfp8):
                                            self.set_sf(dov_tiled_mma, sf_dov, kphase_coord[2])
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                                load_do_releaser.release()
                                load_do_releaser.advance()
                            dp_handle.commit()
                            # dSKTend
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                kt_handle = load_kt_consumer.wait_and_advance()
                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                tdS = cute.make_tensor(
                                    tdStdS_slice.iterator, ds_tmem_layout_staged.outer
                                )
                                tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                tdQrdS_slice = cute.make_tensor(
                                    cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                    tdQrdS.layout,
                                )

                                tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                kt_handle.release()
                            ds_handle.release()
                        else:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_k_consumer.wait_and_advance()
                                if cutlass.const_expr(self.mxfp8):
                                    # dP(i-1) read: the S scales go to its tail
                                    if cutlass.const_expr(not self.dedicated_sf):
                                        mma_dp_producer.acquire()
                                    self.copy_sf(sf_qk, k_handle.index)
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(qk_tiled_mma, sf_qk, kphase_coord[2])
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        if cutlass.const_expr(self.mxfp8):
                                            self.set_sf(qk_tiled_mma, sf_qk, kphase_coord[2])
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            # dOV0
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_dov, unroll=1):
                                load_do_consumer.wait_and_advance()
                                tdPrdO_slice = tdPrdO[None, None, None, iter]
                                v_handle = load_v_consumer.wait_and_advance()
                                if cutlass.const_expr(self.mxfp8):
                                    # S(i) read: the dP scales go to its tail
                                    if cutlass.const_expr(not self.dedicated_sf):
                                        mma_s_producer.acquire()
                                    self.copy_sf(sf_dov, v_handle.index)
                                tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            if cutlass.const_expr(self.mxfp8):
                                                self.set_sf(dov_tiled_mma, sf_dov, kphase_coord[2])
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        if cutlass.const_expr(self.mxfp8):
                                            self.set_sf(dov_tiled_mma, sf_dov, kphase_coord[2])
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                                load_do_releaser.release()
                                load_do_releaser.advance()
                            dp_handle.commit()

                            # dSKT0
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_dsk, unroll=1):
                                kt_handle = load_kt_consumer.wait_and_advance()
                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                tdQtdQ_slice = tdQtdQ_staged[None, None, None, iter]
                                tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                tdS = cute.make_tensor(
                                    tdStdS_slice.iterator, ds_tmem_layout_staged.outer
                                )
                                tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                tdQrdS_slice = cute.make_tensor(
                                    cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                    tdQrdS.layout,
                                )

                                tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % num_innerloop == 0):
                                    num_outer_iter = num_kphases // num_innerloop
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(
                                            num_innerloop, unroll_full=True
                                        ):
                                            kphase_coord = (
                                                None,
                                                None,
                                                outer_iter * num_innerloop + kphase_idx,
                                            )
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                kt_handle.release()
                            ds_handle.release()
                        dq_handle.commit()
                work_tile = tile_sched.advance_to_next_work()
            # End of persistent scheduler loop
            mma_s_producer.tail()
            mma_dp_producer.tail()
            mma_dq_producer.tail()

        # Softmax and dSoftmax warp
        work_tile_epi = work_tile
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
            while work_tile.is_valid_tile:
                curr_block_coord = sm100_fmha_block_coord(work_tile, self.use_varlen_scheduler)
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                cuseqlen_q = Int32(0)
                is_valid_q = True
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                if cutlass.const_expr(seqused_q is not None):
                    seqlen_q = seqused_q[batch_coord]
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                if cutlass.const_expr(cum_seqlen_q is not None or seqused_q is not None):
                    is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                start_count, trip_count = FusedMask.get_trip_start_count_via_block_info(
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    self.is_causal,
                    self.is_local,
                    window_size_left,
                    window_size_right,
                )
                is_valid_k = trip_count > 0
                has_work = is_valid_q and is_valid_k

                if has_work:
                    end_count = start_count + trip_count
                    if cutlass.const_expr(self.use_semantic_trip_range):
                        n_block_min_causal_local_mask, n_block_min_before_local_mask = (
                            FusedMask.get_trip_mask_bounds_via_block_info(
                                mma_block_coord,
                                self.qk_mma_tiler,
                                seqlen_q,
                                seqlen_k,
                                self.is_causal,
                                self.is_local,
                                window_size_left,
                                window_size_right,
                            )
                        )

                    cS_base = cute.make_identity_tensor(
                        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                    )
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    tScS = qk_thr_mma.partition_C(cS)

                    cdP_base = cute.make_identity_tensor(
                        (self.dov_mma_tiler[0], self.dov_mma_tiler[1])
                    )
                    cdP = cute.domain_offset(
                        (mma_block_coord[0] * self.dov_mma_tiler[0], 0), cdP_base
                    )
                    tdPcdP = dov_thr_mma.partition_C(cdP)

                    lse_handle = load_lse_consumer.wait_and_advance()
                    sum_odo_handle = load_sum_odo_consumer.wait_and_advance()
                    for step in cutlass.range(start_count, end_count, 1, unroll=1):
                        cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)

                        cdP_iter = cute.domain_offset((0, step * self.dov_mma_tiler[1]), cdP)

                        tdPcdP_iter = dov_thr_mma.partition_C(cdP_iter)

                        # Si, dPi -> dSi
                        if cutlass.const_expr(self.use_semantic_trip_range):
                            need_apply_mask = (
                                step >= n_block_min_causal_local_mask
                                or step < n_block_min_before_local_mask
                            )
                        else:
                            need_apply_mask = step == end_count - 1
                        mma_s_consumer, mma_dp_consumer, ds_mma_producer = self.compute_step(
                            (need_apply_mask, window_size_left, window_size_right),
                            (
                                seqlen_q,
                                seqlen_k,
                                scale_softmax,
                                batch_coord,
                                curr_block_coord[0],
                                varlen,
                            ),
                            (tStS, tScS_iter, tdPtdP, tdPcdP_iter, sLSE, sSum_OdO),
                            (
                                mma_s_consumer,
                                mma_dp_consumer,
                                ds_mma_producer,
                                lse_handle,
                                sum_odo_handle,
                            ),
                            step,
                        )
                    lse_handle.release()
                    sum_odo_handle.release()

                work_tile = tile_sched.advance_to_next_work()
            ds_mma_producer.tail()
            if cutlass.const_expr(self.wg2_compute):
                work_tile = work_tile_epi  # the epilogue warps store this tile's dQ next

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_epilogue)

            while work_tile.is_valid_tile:
                curr_block_coord = sm100_fmha_block_coord(work_tile, self.use_varlen_scheduler)
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                # cute.printf("batch_coord={}", batch_coord)
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                cuseqlen_q = Int32(0)
                is_valid_q = True
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                if cutlass.const_expr(seqused_q is not None):
                    seqlen_q = seqused_q[batch_coord]
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                if cutlass.const_expr(cum_seqlen_q is not None or seqused_q is not None):
                    is_valid_q = FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q,
                    )
                seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                    FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                    )
                )
                is_valid_k = seqlen_kv_loop_steps > 0
                has_work = is_valid_q and is_valid_k

                mdQ_qdl_eff = mdQ_qdl
                if cutlass.const_expr(cum_seqlen_q is not None):
                    block_offset_dQ = (cuseqlen_q,) + (None,) * 2
                    offset_dQ = cute.assume(
                        cuseqlen_q * mdQ_qdl.stride[0],
                        divby=64,
                    )
                    mdQ_qdl_eff = cute.make_tensor(
                        mdQ_qdl.iterator + offset_dQ,
                        mdQ_qdl.layout,
                    )

                # (bM, bN, loopM, loopN, loopL)
                gdQ_qdl = cute.flat_divide(
                    mdQ_qdl_eff, cute.select(self.dsk_block_tiler, mode=[0, 1])
                )
                cdQ_qdl = cute.flat_divide(
                    cute.make_identity_tensor(mdQ_qdl_eff.shape),
                    cute.select(self.dsk_block_tiler, mode=[0, 1]),
                )

                gdQ_staged = gdQ_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                cdQ_staged = cdQ_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                mdQ_tma_eff = mdQ_tma
                if cutlass.const_expr(varlen):
                    mdQ_tma_eff = cute.domain_offset(
                        block_offset_dQ,
                        mdQ_tma,
                    )
                gdQ_tma_qdl = cute.flat_divide(
                    mdQ_tma_eff,
                    cute.select(self.dsk_block_tiler, mode=[0, 1]),
                )
                gdQ_tma_staged = gdQ_tma_qdl[
                    None, None, curr_block_coord[0], None, curr_block_coord[2]
                ]

                if has_work:
                    # dQ TMEM to GMEM
                    mma_dq_consumer = self.dQ_epilogue(
                        seqlen_q,
                        curr_block_coord[0],
                        (mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged),
                        self.epi_tile,
                        (tma_atom_dQ, gdQ_tma_staged, s_epi_dQ, varlen),
                    )
                else:
                    self.dQ_epilogue_write_zero(
                        seqlen_q,
                        gdQ_staged,
                        cdQ_staged,
                    )

                work_tile = tile_sched.advance_to_next_work()
            # NOTE: tmem.free() moved to kernel end to enable cluster-wide sync

        # ///////////////////////////////////////////////////////////////////////////////
        #  Scheduler Warp (only for CLC dynamic scheduler)
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            is_first_cta_in_cluster = cta_rank_in_cluster == 0

            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                while work_tile.is_valid_tile:
                    tile_sched.prefetch_next_work()
                    work_tile = tile_sched.advance_to_next_work()
                tile_sched.producer_tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty warps reg dealloc
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            if warp_idx > self.load_warp_id:
                if not (warp_idx == self.sched_warp_id and is_first_cta_in_cluster):
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
        else:
            if warp_idx > self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cooperative TMEM Deallocation (2CTA)
        # ///////////////////////////////////////////////////////////////////////////////
        # All warps (including scheduler) have finished by this point.
        # Cluster-wide sync ensures both CTAs reach here before dealloc.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

        return

    @cute.jit
    def copy_sf(self, sf, B_idx: Int32):
        """MXFP8: SMEM -> TMEM copies of one block-scaled MMA's A (stage 0) and B (stage B_idx)
        scales, issued by the leader before the MMA (same-thread tcgen05 issue order)."""
        (_, copy_a, src_a, dst_a), (_, copy_b, src_b, dst_b) = sf
        cute.copy(copy_a, src_a[None, None, None, None, 0], dst_a)
        cute.copy(copy_b, src_b[None, None, None, None, B_idx], dst_b)

    @cute.jit
    def set_sf(self, tiled_mma: cute.TiledMma, sf, k_block: Int32):
        """MXFP8: point the block-scaled MMA at k_block's A / B scales in TMEM."""
        (tSFA, _, _, _), (tSFB, _, _, _) = sf
        tiled_mma.set(tcgen05.Field.SFA, tSFA[None, None, k_block].iterator)
        tiled_mma.set(tcgen05.Field.SFB, tSFB[None, None, k_block].iterator)

    @cute.jit
    def compute_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        step: Int32,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        seqlen_q, seqlen_k, scale_softmax, batch_coord, block_m_idx, varlen = value_args
        tStS, tScS, tdPtdP, tdPcdP, sLSE, sSum_OdO = tensor_args
        mma_s_consumer, mma_dp_consumer, ds_mma_producer, lse_handle, sum_odo_handle = pipeline_args

        bidx = block_m_idx
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * 4)
        wg_idx = cute.arch.make_warp_uniform(tidx // (self.threads_per_warp * 4))
        nwg = self.num_compute_wgs
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), self.acc_dtype
        )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        if cutlass.const_expr(nwg > 1):
            tTMEM_LOADtS = mxu.split_wg_contiguous(tTMEM_LOADtS, nwg, wg_idx)
            tTMEM_LOADcS = mxu.split_wg_contiguous(tTMEM_LOADcS, nwg, wg_idx)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if need_apply_mask:
            FusedMask.apply_mask_via_causal_local(
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                self.use_semantic_trip_range,
                self.is_causal,
                self.is_local,
                window_size_left,
                window_size_right,
            )

        log2_e = cutlass.Float32(math.log2(math.e))
        softmax_scale_log2_e = scale_softmax * log2_e
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        for k in cutlass.range(0, cute.size(tTMEM_LOADrS), 2, unroll_full=True):
            lse = (
                -sLSE[
                    cute.get(tTMEM_LOADcS[k], mode=[0]) - bidx * self.cta_tiler[0],
                    lse_handle.index,
                ],
                -sLSE[
                    cute.get(tTMEM_LOADcS[k + 1], mode=[0]) - bidx * self.cta_tiler[0],
                    lse_handle.index,
                ],
            )
            tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1] = cute.arch.fma_packed_f32x2(
                (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1]),
                (softmax_scale_log2_e, softmax_scale_log2_e),
                lse,
            )
            tTMEM_LOADrS[k] = cute.math.exp2(tTMEM_LOADrS[k], fastmath=True)
            tTMEM_LOADrS[k + 1] = cute.math.exp2(tTMEM_LOADrS[k + 1], fastmath=True)

        dp_handle = mma_dp_consumer.wait_and_advance()
        tdPtdP_slice = tdPtdP[(None, None), 0, 0, dp_handle.index]
        tdPcdP_slice = tdPcdP[(None, None), 0, 0]
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtdP = thr_load.partition_S(tdPtdP_slice)
        tTMEM_LOADcdP = thr_load.partition_D(tdPcdP_slice)
        if cutlass.const_expr(nwg > 1):
            tTMEM_LOADtdP = mxu.split_wg_contiguous(tTMEM_LOADtdP, nwg, wg_idx)
            tTMEM_LOADcdP = mxu.split_wg_contiguous(tTMEM_LOADcdP, nwg, wg_idx)
        tTMEM_LOADrdP = cute.make_rmem_tensor(tTMEM_LOADcdP.shape, self.acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtdP, tTMEM_LOADrdP)
        cute.arch.fence_view_async_tmem_load()
        dp_handle.release()
        tTMEM_STORErdP = cute.make_rmem_tensor(tTMEM_LOADrdP.shape, self.q_dtype)

        for k in cutlass.range(0, cute.size(tTMEM_LOADrdP), 2, unroll_full=True):
            dpsum_0 = -sSum_OdO[
                cute.get(tTMEM_LOADcdP[k], mode=[0]) - bidx * self.cta_tiler[0],
                sum_odo_handle.index,
            ]
            dpsum_1 = -sSum_OdO[
                cute.get(tTMEM_LOADcdP[k + 1], mode=[0]) - bidx * self.cta_tiler[0],
                sum_odo_handle.index,
            ]
            if cutlass.const_expr(varlen):
                if not cute.elem_less(cute.get(tTMEM_LOADcdP[k], mode=[0]), seqlen_q):
                    dpsum_0 = 0.0
                if not cute.elem_less(cute.get(tTMEM_LOADcdP[k + 1], mode=[0]), seqlen_q):
                    dpsum_1 = 0.0
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.add_packed_f32x2(
                (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]),
                (dpsum_0, dpsum_1),
            )
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.mul_packed_f32x2(
                (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]), (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1])
            )
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.mul_packed_f32x2(
                (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]), (scale_softmax, scale_softmax)
            )
        dp_vec = tTMEM_LOADrdP.load()
        tTMEM_STORErdP.store(dp_vec.to(self.q_dtype))
        if cutlass.const_expr(nwg > 1):
            # dS (BF16, 128 keys) is stored into dP's first 64 columns, which hold the other group's dP keys
            # 32..63: both groups must have read their dP before either stores
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id, number_of_threads=self.threads_per_warp * 8
            )

        ds_handle = ds_mma_producer.acquire_and_advance()
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.acc_dtype
        )
        tilePlikeFP32 = tdPtdP_slice.shape[1] // Float32.width * self.q_dtype.width
        tdPtdP_dS_layout = cute.composition(
            tdPtdP_slice.layout, cute.make_layout((tdPtdP_slice.shape[0], tilePlikeFP32))
        )
        tdPtdP_dS = cute.make_tensor(tdPtdP_slice.iterator, tdPtdP_dS_layout)
        tdPcdP_dS_layout = cute.composition(
            tdPcdP_slice.layout, cute.make_layout((tdPcdP_slice.shape[0], tilePlikeFP32))
        )
        tdPcdP_dS = cute.make_tensor(tdPcdP_slice.iterator, tdPcdP_dS_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tdPtdP_dS)

        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtdS = thr_store.partition_D(tdPtdP_dS)
        tTMEM_STOREcdP = thr_store.partition_S(tdPcdP_dS)
        if cutlass.const_expr(nwg > 1):
            tTMEM_STOREtdS = mxu.split_wg_contiguous(tTMEM_STOREtdS, nwg, wg_idx)
            tTMEM_STOREcdP = mxu.split_wg_contiguous(tTMEM_STOREcdP, nwg, wg_idx)
        tTMEM_STORErdS_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErdP.iterator, dtype=self.acc_dtype),
            tTMEM_STOREcdP.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErdS_, tTMEM_STOREtdS)
        cute.arch.fence_view_async_tmem_store()
        ds_handle.commit()
        return mma_s_consumer, mma_dp_consumer, ds_mma_producer

    @cute.jit
    def dQ_epilogue(
        self,
        seqlen_q: int,
        q_block: int,
        dq_args: Tuple,
        epi_tile: cute.Tile,
        tma_args: Tuple,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged) = dq_args
        tma_atom_dQ, gdQ_tma_staged, s_epi_dQ, varlen = tma_args
        dq_handle = mma_dq_consumer.wait_and_advance()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cute.arch.fence_view_async_shared()

        epi_cols_dQ = math.gcd(128 // (self.dq_dtype.width // 8), epi_tile[1])
        num_epi_stages_dQ = epi_tile[1] // epi_cols_dQ
        epi_tile_dQ = (epi_tile[0], epi_cols_dQ)
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        for iter in cutlass.range(self.iterations_dsk):
            gdQ = gdQ_staged[None, None, iter]
            cdQ = cdQ_staged[None, None, iter]
            tdQtdQ = tdQtdQ_staged[(None, None), 0, 0, iter]
            tdQtdQ_epi = cute.zipped_divide(tdQtdQ, epi_tile)
            cdQ_epi = cute.zipped_divide(cdQ, epi_tile)
            gdQ_epi = cute.zipped_divide(gdQ, epi_tile)
            cdQ_local = cute.make_identity_tensor(epi_tile)
            cdQ_local_epi = cute.zipped_divide(cdQ_local, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.epilogue_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tdQtdQ_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtdQ = thr_tmem_load.partition_S(tdQtdQ_epi)
            tTMEM_LOADgdQ = thr_tmem_load.partition_D(gdQ_epi)
            tTMEM_LOADcdQ = thr_tmem_load.partition_D(cdQ_epi)
            tTMEM_LOADcdQ_local = thr_tmem_load.partition_D(cdQ_local_epi)

            use_tma_store = True
            if cutlass.const_expr(varlen):
                use_tma_store = (
                    q_block * self.dsk_block_tiler[0] + self.dsk_block_tiler[0] <= seqlen_q
                )
            if use_tma_store:
                gdQ_tma = gdQ_tma_staged[None, None, iter]
                gdQ_tma_epi = cute.local_tile(gdQ_tma, epi_tile_dQ, (0, None))
                sdQ_stage = s_epi_dQ[None, None, 0]

                for stage_k in cutlass.range_constexpr(num_epi_stages_dQ):
                    for i in cutlass.range(cute.size(tTMEM_LOADtdQ, mode=[1]), unroll_full=True):
                        tTMEM_LOADtdQ_i = tTMEM_LOADtdQ[None, i, 0]
                        tTMEM_LOADcdQ_i_local = tTMEM_LOADcdQ_local[None, i, 0]
                        tTMrdQ = cute.make_rmem_tensor(tTMEM_LOADcdQ_i_local.shape, self.acc_dtype)
                        cute.copy(tiled_tmem_load, tTMEM_LOADtdQ_i, tTMrdQ)
                        tSMrdQ = cute.make_rmem_tensor(tTMrdQ.shape, self.q_dtype)
                        dq_vec = tTMrdQ.load()
                        tSMrdQ.store(dq_vec.to(self.q_dtype))
                        for j in cutlass.range_constexpr(cute.size(tTMEM_LOADcdQ_i_local)):
                            c = tTMEM_LOADcdQ_i_local[j]
                            m_pos = c[0]
                            n_pos = c[1]
                            if n_pos // epi_cols_dQ == stage_k:
                                s_epi_dQ[m_pos, n_pos % epi_cols_dQ, 0] = tSMrdQ[j]

                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=3, number_of_threads=128)

                    if leader_warp:
                        gdQ_stage = gdQ_tma_epi[None, None, stage_k]
                        td_sdQ, td_gdQ = cpasync.tma_partition(
                            tma_atom_dQ,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sdQ_stage, 0, 2),
                            cute.group_modes(gdQ_stage, 0, 2),
                        )
                        cute.copy(tma_atom_dQ, td_sdQ, td_gdQ)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    # Non-issuing threads must not rotate the SMEM buffer
                    # until the leader warp's TMA read has completed.
                    cute.arch.barrier(barrier_id=3, number_of_threads=128)
            else:
                for i in cutlass.range(cute.size(tTMEM_LOADtdQ, mode=[1]), unroll_full=True):
                    tTMEM_LOADtdQ_i = tTMEM_LOADtdQ[None, i, 0]
                    tTMEM_LOADgdQ_i = tTMEM_LOADgdQ[None, i, 0]
                    tTMEM_LOADcdQ_i = tTMEM_LOADcdQ[None, i, 0]
                    tTMrdQ = cute.make_rmem_tensor(tTMEM_LOADcdQ[None, 0, i].shape, self.acc_dtype)
                    cute.copy(tiled_tmem_load, tTMEM_LOADtdQ_i, tTMrdQ)
                    tSMrdQ = cute.make_rmem_tensor(tTMrdQ.shape, self.q_dtype)
                    dq_vec = tTMrdQ.load()
                    tSMrdQ.store(dq_vec.to(self.q_dtype))
                    if cute.elem_less(tTMEM_LOADcdQ_i[0][0], seqlen_q):
                        cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)
        dq_handle.release()
        return mma_dq_consumer

    @cute.jit
    def dQ_epilogue_write_zero(
        self,
        seqlen_q,
        gdQ_staged,
        cdQ_staged,
    ):
        num_epi_threads = self.threads_per_warp * len(self.epilogue_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_epi_threads

        tiled_copy_r2g = fa_copy_utils.tiled_copy_2d(
            self.dq_dtype, cute.size(gdQ_staged.shape[1]), num_epi_threads
        )

        thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)
        tdQgdQ_staged = thr_copy_r2g.partition_D(gdQ_staged)
        tdQcdQ_staged = thr_copy_r2g.partition_D(cdQ_staged)

        tdQrdQ = cute.make_rmem_tensor_like(tdQgdQ_staged[None, 0, None, 0])
        tdQrdQ.fill(self.dq_dtype(0.0))

        for iter in cutlass.range(self.iterations_dsk, unroll_full=True):
            tdQgdQ = tdQgdQ_staged[None, None, None, iter]
            tdQcdQ = tdQcdQ_staged[None, None, None, iter]
            for m in cutlass.range(cute.size(tdQgdQ.shape[1]), unroll_full=True):
                if cute.elem_less(tdQcdQ[0, m, 0][0], seqlen_q):
                    cute.copy(tiled_copy_r2g, tdQrdQ, tdQgdQ[None, m, None])
