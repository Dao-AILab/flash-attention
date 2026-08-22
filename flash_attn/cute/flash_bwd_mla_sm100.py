# Copyright (c) 2026, Colfax International.

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Uint32, Boolean, const_expr
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import ClcDynamicPersistentTileScheduler

from quack import copy_utils, layout_utils

from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
import flash_attn.cute.blackwell_helpers as fa_sm100_utils
from flash_attn.cute.tile_scheduler import (
    SchedulerState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)
from flash_attn.cute.fa_logging import fa_log, fa_printf
from flash_attn.cute.utils import smid, elem_pointer, get_batch_from_cu_tensor
from flash_attn.cute.copy_utils import tiled_copy_2d, atomic_add_fp32x4

from flash_attn.cute.topk_gather_kv import CpasyncGatherKVManager


from flash_attn.cute.named_barrier import NamedBarrierBwdSm100_MLA2CTA


class FlashAttentionSparseMLABackwardSm100:
    def __init__(
        self,
        is_causal: bool = False,
        topk_length: int = 2048,
        qhead_per_kvhead: int = 1,
        nheads_kv: int = 1,
        hdim: int = 64,
        hdimv: int = 512,
        has_seqused_q: bool = False,
        disable_bitmask: bool = False,
        use_clc_scheduler: bool = True,
        recompute_P: bool = False,
    ):
        use_cpasync_load_KV = True
        # recompute_P: instead of loading the fwd-saved p (rescaled by scale_p),
        # recompute S^T = V_latent @ Qv^T (+ K_rope @ Q_rope^T) with an extra
        # UMMA and take P^T = exp2(softmax_scale_log2 * S^T - lse * log2(e)).
        # The fwd then saves nothing but out/lse.
        self.recompute_P = recompute_P
        self.is_causal = is_causal
        self.is_local = False
        self.pack_gqa = True
        self.qhead_per_kvhead = qhead_per_kvhead
        self.nheads_kv = nheads_kv
        self.has_seqused_q = has_seqused_q
        self.use_tma_O = True
        self.use_cpasync_load_KV = True
        self.use_tma_KV = False
        self.topk_length = topk_length
        self.is_topk_gather = True
        assert qhead_per_kvhead == 128 or qhead_per_kvhead == 64

        # user-provided option if topk indices guaranteed in bounds
        self.disable_bitmask = disable_bitmask
        # In recompute mode the bitmask is the ONLY masking of invalid topk
        # slots (there is no fwd-saved p carrying zeros to fall back on): it
        # forces the exponent to -inf on sentinel / causal-out-of-range rows
        # and gates the non-finite-dP hardening. Without it, recomputed P is
        # garbage on those slots and fully-masked rows (lse = -inf) go NaN.
        assert not (self.recompute_P and self.disable_bitmask), (
            "recompute_P requires the KV bitmask (disable_bitmask=False)"
        )

        # ==== tile scheduler ====
        self.static_persistent = False
        self.use_clc_scheduler = use_clc_scheduler
        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )

        if const_expr(has_seqused_q):
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        fa_log(
            1,
            f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}",
        )

        # ==== thread info ====
        self.num_softmax_threads = 128
        self.num_epilogue_threads = 128
        self.num_load_threads = 32
        self.num_mma_threads = 32
        self.num_empty_threads = 0
        self.num_relay_threads = 32
        self.num_cpasync_load_threads = 128
        self.num_threads = 512
        self.num_warps = self.num_threads // 32
        self.softmax_warp_indices = (0, 1, 2, 3)
        self.epilogue_warp_indices = (4, 5, 6, 7)
        self.load_warp_id = 8
        self.mma_warp_id = 9
        self.clc_scheduler_warp_id = 10
        self.relay_warp_id = 11
        self.cpasync_load_warp_indices = (12, 13, 14, 15)
        self.empty_warp_ids = ()

        # ==== register usage ====
        assert self.num_warps == 16

        self.num_regs_load = 128
        self.num_regs_mma = 128
        self.num_regs_softmax = 128
        self.num_regs_epilogue = 128
        self.num_regs_cpasync = 128
        self.num_regs_other = 128

        # self.num_regs_load = 128 - 32
        # self.num_regs_mma = 128 - 32
        # self.num_regs_softmax = 128 + 32
        # self.num_regs_epilogue = 128 + 32
        # self.num_regs_cpasync = 128 - 32
        # self.num_regs_other = 48

        self.num_regs_per_thread = 128
        self.num_regs_total = 512

        assert (
            self.num_regs_mma
            + self.num_regs_softmax
            + self.num_regs_epilogue
            + self.num_regs_cpasync
            <= self.num_regs_total
        )

        # ==== 2cta info ====
        self.use_2cta_instrs = True
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cta_group_size = 2
        self.cluster_shape_mn = (2, 1)
        self.cluster_shape_mnk = (2, 1, 1)

        # ==== problem shape info ====
        self.hdim = hdim  # ignored
        self.hdimv = hdimv
        self.tile_m = qhead_per_kvhead
        self.tile_n = 64
        self.cta_tiler_mn = (self.tile_m // self.cta_group_size, self.tile_n)
        self.cluster_tile_n = self.cta_group_size * self.tile_n
        self.num_hdimv_splits = 2  # split hdimv in half for our Qv @ V^T and P @ V mmas.

        self.tile_P = (self.tile_m, self.tile_n)
        self.tile_Pt = (self.tile_n, self.tile_m)
        self.tile_dS = (self.tile_m, self.tile_n)
        self.tile_dSt = (self.tile_n, self.tile_m)
        self.tile_dV = (self.tile_n, 32)

        # ==== MMA info ====
        # dP.T = V    @ dO.T , N x M x dv
        # dV  += P.T  @ dO   , N x dv x M
        # dV  += dS.T @ Qv   , N x dv x M
        self.mma_tiler_VdO = (
            self.cluster_tile_n,
            self.tile_m,
            self.hdimv // self.num_hdimv_splits,
        )
        self.mma_tiler_PtdOt = (
            self.cluster_tile_n,
            self.hdimv // self.num_hdimv_splits,
            self.tile_m,
        )
        self.mma_tiler_dStQvt = (
            self.cluster_tile_n,
            self.hdimv // self.num_hdimv_splits,
            self.tile_m,
        )
        # recompute_P: S^T = V_latent @ Qv^T + K_rope @ Q_rope^T. The latent
        # chunks reuse mma_tiler_VdO (identical M/N/majorness as dP^T = V@dO^T,
        # B = Qv instead of dO); the rope chunk only differs in K.
        self.mma_tiler_Kr = (self.cluster_tile_n, self.tile_m, self.hdim)
        # note: store P.T, dS.T as tile_n major (i.e., as P and dS)
        self.major_mode_V = tcgen05.OperandMajorMode.K
        self.major_mode_dO = tcgen05.OperandMajorMode.K
        self.major_mode_Pt = tcgen05.OperandMajorMode.MN
        self.major_mode_dOt = tcgen05.OperandMajorMode.MN
        self.major_mode_dSt = tcgen05.OperandMajorMode.MN
        self.major_mode_Qvt = tcgen05.OperandMajorMode.MN
        self.operand_source_V = tcgen05.OperandSource.SMEM
        self.operand_source_Pt = tcgen05.OperandSource.SMEM
        self.operand_source_dSt = tcgen05.OperandSource.SMEM

        # ==== pipeline info ====
        # stationary: dOi
        # mainloop:
        # *) P, scaleP => Pt
        # *) dSt
        # *) Vi, i = {0, 1}
        # *) dOti => Qvi => dVi, i = {0, 1}

        # redundant names for ease-of-use
        self.num_stages_V = 2
        self.num_stages_dO = 2
        self.num_stages_P = 1
        self.num_stages_Pt = 1
        self.num_stages_dS = 1
        self.num_stages_dSt = 1
        self.num_stages_dOt = 2
        self.num_stages_Qv = 2
        self.num_stages_Qvt = 2

        self.num_stages_dP = 1
        self.num_stages_dPt = 1
        self.num_stages_dV = 2  # == hdimv splits, for Umma <-> Async
        self.num_epi_stages_dV = 8  # == 2 splits x 4 slots/split

        self.num_stages_scaleP = 1
        self.num_stages_dPsum = 1

        # recompute_P pipeline stages. QvB (the K-major Qv B-operand of the
        # S^T gemm) rides the existing 2-stage dOt/Qvt multiplex pipeline
        # (6 loads/group instead of 4), so it needs no stage count of its own.
        self.num_stages_Kr = 1
        self.num_stages_Qr = 1
        self.num_stages_lse = 1
        self.num_stages_St = 1
        self.num_stages_bitmask = 2

        # ==== dtype info ====
        self.dtype_acc = Float32

        # ==== TMEM info ====
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_cols_dP = self.tile_m // self.cta_group_size
        self.tmem_cols_dVi = (self.hdimv // self.num_hdimv_splits) // self.cta_group_size
        self.tmem_offset_dV0 = 0
        self.tmem_offset_dV1 = self.tmem_offset_dV0 + self.tmem_cols_dVi
        self.tmem_offsets_dV = [self.tmem_offset_dV0, self.tmem_offset_dV1]
        self.tmem_offset_dP = self.tmem_offset_dV1 + self.tmem_cols_dVi
        self.total_tmem = self.tmem_offset_dP + self.tmem_cols_dP
        if self.recompute_P:
            # S^T accumulator: same footprint as dP^T (one (128,128) fp32
            # 2-CTA accumulator = 64 columns per CTA).
            self.tmem_cols_S = self.tile_m // self.cta_group_size
            self.tmem_offset_S = self.tmem_offset_dP + self.tmem_cols_dP
            self.total_tmem = self.tmem_offset_S + self.tmem_cols_S * self.num_stages_St
        assert self.total_tmem <= self.tmem_alloc_cols, (
            f"Total TMEM columns allocated {self.total_tmem} exceeds capacity {self.tmem_alloc_cols}"
        )

    def _get_shared_storage_cls(self):
        self.buffer_align_bytes = 1024

        def smem_struct_align(dtype, staged_layout, disabled=False):
            # disabled fields are zero-size MemRanges: they add no bytes, and
            # (element alignment still applies to zero-size fields) every
            # insertion point in the struct below already satisfies their
            # element alignment, so they add no padding either — same pattern
            # as the forward kernel
            if disabled:
                return cute.struct.MemRange[dtype, 0]
            return cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(staged_layout)],
                self.buffer_align_bytes,
            ]

        def mbar_struct(num_stages):
            return cute.struct.MemRange[Int64, 2 * num_stages]

        # sV, sdO, sP = sPt, sdSt = sdS, sdOt = sQvt = sdV
        # recompute_P: the sdO slot holds the stationary QvB (S^T gemm
        # B-operand); dO rides the sdOt = sQvt = sdV multiplex; with q/k rope
        # (has_qk) dS additionally merges into sP (sQr/sKr take its 16 KiB).
        (
            sV_struct,
            sdO_struct,
            sP_struct,
            sdS_struct,
            sQvt_struct,
            sScaleP_struct,
            sdPsum_struct,
        ) = (
            smem_struct_align(dtype, layout, disabled)
            for dtype, layout, disabled in [
                (self.dtype, self.sV_layout_staged, False),
                (self.dtype, self.sdO_layout_staged, False),
                (self.dtype, self.sPt_layout_staged, False),
                # merged_dS: dS shares sP's buffer (P is consumed by the
                # dV += P^T @ dO mma strictly before dS overwrites it)
                (self.dtype, self.sdSt_layout_staged, self.merged_dS),
                (self.dtype, self.sQvt_layout_staged, False),
                # recompute_P replaces the fwd-saved scaleP by lse
                (self.dtype_scale, self.sScaleP_layout_staged, self.recompute_P),
                (self.dtype_scale, self.sdPsum_layout_staged, False),
            ]
        )

        # Fields absent from a compile mode get 0 stages => zero-size
        # MemRanges that add no bytes and no padding (all header fields in a
        # run of Int64s), so the single struct below reproduces each mode's
        # previous hand-written layout exactly.
        rp = self.recompute_P
        has_rope = rp and self.has_qk
        (
            mbar_ptr_V_struct,  # load V
            mbar_ptr_dO_struct,  # load dO (recompute_P: stationary QvB)
            mbar_ptr_dOt_Qvt_struct,  # load dOt => Qvt
            mbar_ptr_dSt_struct,  # store dS
            mbar_ptr_P_struct,  # load P (default mode only)
            mbar_ptr_Pt_struct,  # store Pt
            mbar_ptr_dPt_struct,  # dP mma
            mbar_ptr_dV_struct,  # dV mma
            mbar_ptr_scaleP_struct,  # load scaleP (default mode only)
            mbar_ptr_dPsum_struct,  # load dPsum
            mbar_ptr_St_struct,  # S^T mma (recompute_P only)
            mbar_ptr_Kr_struct,  # gather K_rope (recompute_P + has_qk only)
            mbar_ptr_Qr_struct,  # load Q_rope (recompute_P + has_qk only)
            mbar_ptr_lse_struct,  # load lse (recompute_P only)
            mbar_ptr_bitmask_struct,  # bitmask (recompute_P only)
        ) = (
            mbar_struct(n)
            for n in [
                self.num_stages_V,
                self.num_stages_dO,
                self.num_stages_Qvt,
                self.num_stages_dSt,
                self.num_stages_P if not rp else 0,
                self.num_stages_Pt,
                self.num_stages_dPt,
                self.num_stages_dV,
                self.num_stages_scaleP if not rp else 0,
                self.num_stages_dPsum,
                self.num_stages_St if rp else 0,
                self.num_stages_Kr if has_rope else 0,
                self.num_stages_Qr if has_rope else 0,
                self.num_stages_lse if rp else 0,
                self.num_stages_bitmask if rp else 0,
            ]
        )
        tmem_dealloc_mbar_struct = Int64
        tmem_holding_buf_struct = Int32

        self.sched_stages = 1
        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        # ==== Unified SharedStorage (all three compile modes) ====
        # recompute_P deltas vs the default layout (kernel sits at the exact
        # 232448 B cap):
        # - with rope (has_qk) sdS is zero-size: dS shares sP's 16 KiB, and
        #   sQr (stationary Q_rope B-operand) + sKr (gathered K_rope A
        #   stages) take the 16 KiB freed by the merge.
        # - sScaleP (default) and sLse (recompute) are mutually exclusive
        #   (both 128 fp32).
        # - the recompute-only mbarriers + sBitmask fit in the header padding
        #   before the first 1024-aligned buffer.
        sBitmask_struct = cute.struct.MemRange[
            Uint32, cute.cosize(self.sBitmask_layout) if rp else 0
        ]
        sLse_struct = smem_struct_align(self.dtype_scale, self.sLse_layout_staged, disabled=not rp)
        sQr_struct = smem_struct_align(
            self.dtype, self.sQr_layout_staged if has_rope else None, disabled=not has_rope
        )
        sKr_struct = smem_struct_align(
            self.dtype, self.sKr_layout_staged if has_rope else None, disabled=not has_rope
        )

        @cute.struct
        class SharedStorage:
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_V_cpasync: mbar_ptr_V_struct
            mbar_ptr_dO: mbar_ptr_dO_struct
            mbar_ptr_dOt_Qvt: mbar_ptr_dOt_Qvt_struct
            mbar_ptr_P: mbar_ptr_P_struct
            mbar_ptr_Pt: mbar_ptr_Pt_struct
            mbar_ptr_dSt: mbar_ptr_dSt_struct
            mbar_ptr_dPt: mbar_ptr_dPt_struct
            mbar_ptr_dV: mbar_ptr_dV_struct
            mbar_ptr_dV_epi: mbar_ptr_dV_struct
            mbar_ptr_scaleP: mbar_ptr_scaleP_struct
            mbar_ptr_dPsum: mbar_ptr_dPsum_struct
            mbar_ptr_St: mbar_ptr_St_struct
            mbar_ptr_Kr: mbar_ptr_Kr_struct
            mbar_ptr_Kr_cpasync: mbar_ptr_Kr_struct
            mbar_ptr_Qr: mbar_ptr_Qr_struct
            mbar_ptr_lse: mbar_ptr_lse_struct
            mbar_ptr_bitmask: mbar_ptr_bitmask_struct
            sBitmask: sBitmask_struct
            tmem_dealloc_mbar: tmem_dealloc_mbar_struct
            tmem_holding_buf: tmem_holding_buf_struct
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]

            sScaleP: sScaleP_struct
            sLse: sLse_struct
            sdPsum: sdPsum_struct
            sV: sV_struct
            sdO: sdO_struct
            sP: sP_struct
            sdS: sdS_struct
            sQv: sQvt_struct
            sQr: sQr_struct
            sKr: sKr_struct

        # print("smem bytes = ", SharedStorage.size_in_bytes())

        return SharedStorage

    # fmt: off
    @cute.jit
    def __call__(
        self,
        mdO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mV: cute.Tensor,   # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
        mQv: cute.Tensor,  # == mdO
        mP: cute.Tensor,   # (b, s_q, h, topk) or (total_q, h, topk)
        mdV: cute.Tensor,  # == mV
        mdS: cute.Tensor,  # == mP
        mIndexTopk: cute.Tensor,  # (b, s_q, topk) or (total_q, topk) if there is cu_seqlens_q
        softmax_scale: Float32,
        mScaleP: Optional[cute.Tensor] = None,      # (b, s_q, topk//128, h) or (total_q, topk//128, h)
        mdPsum: Optional[cute.Tensor] = None,       # (b, s_q, h) or (total_q, h) if there is cu_seqlens_q
        mQ: Optional[cute.Tensor] = None,           # (b, s_q, h, d) or (total_q, h, d); recompute_P only
        mK: Optional[cute.Tensor] = None,           # (b_k, s_k, h_k, d) or (total_k, h_k, d); recompute_P only
        mLseLog2: Optional[cute.Tensor] = None,     # (b, s_q, h) or (total_q, h); recompute_P only
        mCuSeqlensQ: Optional[cute.Tensor] = None,  # (b + 1)
        mCuSeqlensK: Optional[cute.Tensor] = None,  # (b + 1)
        mSeqUsedQ: Optional[cute.Tensor] = None,    # (b)
        mSeqUsedK: Optional[cute.Tensor] = None,    # (b)
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # fmt: on
        # ==== dtype info ====
        self.dtype = mdO.element_type
        self.dtype_dV = mdV.element_type
        self.dtype_scale = Float32
        self.dtype_index = mIndexTopk.element_type
        assert self.dtype.width == 16
        assert self.dtype_dV.width == 32
        assert self.dtype_index == Int32
        if const_expr(mScaleP is not None):
            assert mScaleP.element_type == self.dtype_scale
        if const_expr(mdPsum is not None):
            assert mdPsum.element_type == self.dtype_scale
        if const_expr(self.recompute_P):
            assert mP is None and mScaleP is None, "recompute_P replaces the saved p/scale_p"
            assert mLseLog2 is not None, "recompute_P requires lse (log2 units)"
            assert mLseLog2.element_type == self.dtype_scale
            self.has_qk = mQ is not None
            if const_expr(self.has_qk):
                assert mK is not None, "recompute_P with q_rope requires k_rope"
        else:
            assert mP is not None and mScaleP is not None
            self.has_qk = False
            mQ = mK = mLseLog2 = None
        # P and dS share one smem buffer only when the rope buffers (sQr/sKr)
        # eat the 16 KiB that dS would otherwise use; without q/k rope, dS
        # keeps its own buffer and the baseline dSt gating applies.
        self.merged_dS = self.recompute_P and self.has_qk

        # ==== Prepare Tensors ====
        new_stride = lambda mX: (
            *(cute.assume(s, divby=128 // mX.element_type.width) for s in mX.stride[:-1]),
            mX.stride[-1],
        )
        mQv, mV, mdV, mdO, mP, mdS, mScaleP, mdPsum, mQ, mK, mLseLog2 = [
            cute.make_tensor(mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX)))
            if mX is not None
            else None
            for mX in (mQv, mV, mdV, mdO, mP, mdS, mScaleP, mdPsum, mQ, mK, mLseLog2)
        ]
        # (b, s, h, d)  -> (s, d, h, b)  or
        # (total, h, d) -> (total, d, h)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQv, mdO, mP, mdS, mQ = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            if mX is not None
            else None
            for mX in (mQv, mdO, mP, mdS, mQ)
        ]
        mV, mdV, mK = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=KV_layout_transpose))
            if mX is not None
            else None
            for mX in (mV, mdV, mK)
        ]

        # (b, s, topk//128, h) -> (s, topk//128, h, b) or
        # (total, topk//128, h) -> (total, topk//128, h)
        ScaleP_layout_transpose = [1, 2, 3, 0] if const_expr(mCuSeqlensQ is None) else [0, 1, 2]
        if const_expr(mScaleP is not None):
            mScaleP = cute.make_tensor(
                mScaleP.iterator, cute.select(mScaleP.layout, mode=ScaleP_layout_transpose)
            )

        # (b, s, h) -> (s, h, b) or
        # (total, h) -> (total, h)
        dPsum_layout_transpose = [1, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 1]
        mdPsum = cute.make_tensor(
            mdPsum.iterator, cute.select(mdPsum.layout, mode=dPsum_layout_transpose)
        )
        if const_expr(mLseLog2 is not None):
            mLseLog2 = cute.make_tensor(
                mLseLog2.iterator, cute.select(mLseLog2.layout, mode=dPsum_layout_transpose)
            )

        # (b, s_q, topk) -> (topk, s_q, b) or (total_q, topk) -> (topk, total_q)
        topk_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mIndexTopk = cute.make_tensor(
            mIndexTopk.iterator, cute.select(mIndexTopk.layout, mode=topk_layout_transpose)
        )
        topk_length_dynamic = mIndexTopk.shape[0]

        if const_expr(self.pack_gqa):
            mQv, mdO, mP, mdS, mQ = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                if mX is not None
                else None
                for mX in (mQv, mdO, mP, mdS, mQ)
            ]
            if const_expr(mScaleP is not None):
                mScaleP = pack_gqa_layout(mScaleP, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
            if const_expr(mdPsum is not None):
                mdPsum = pack_gqa_layout(mdPsum, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)
            if const_expr(mLseLog2 is not None):
                mLseLog2 = pack_gqa_layout(mLseLog2, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)

        # ((h/h_k, s_q), dv, h_k, b) -> (dv, (h/h_k, s_q), h_k, b)
        # or ((h/h_k, total_q), dv, h_k) -> (dv, (h/h_k, total_q), h_k)
        mma_operand_layout_transpose = (
            [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        )
        mQvt, mdOt = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=mma_operand_layout_transpose))
            for mX in (mQv, mdO)
        ]

        # fmt: off
        # ==== Prepare MMAs ====
        # (local_var, dtype_a, major_a, major_b, mma_tiler, operand_source_a)
        _mma_specs = [
            ("tiled_mma_VdO",    self.dtype, self.major_mode_V,   self.major_mode_dO,  self.mma_tiler_VdO,    self.operand_source_V),
            ("tiled_mma_PtdOt",  self.dtype, self.major_mode_Pt,  self.major_mode_dOt, self.mma_tiler_PtdOt,  self.operand_source_Pt),
            ("tiled_mma_dStQvt", self.dtype, self.major_mode_dSt, self.major_mode_Qvt, self.mma_tiler_dStQvt, self.operand_source_dSt),
        ]
        tiled_mma_VdO, tiled_mma_PtdOt, tiled_mma_dStQvt = (
            sm100_utils.make_trivial_tiled_mma(
                dtype_a, major_a, major_b, self.dtype_acc, self.cta_group, mma_tiler[:2], operand_source_a,
            )
            for _, dtype_a, major_a, major_b, mma_tiler, operand_source_a in _mma_specs
        )

        # ==== Prepare SMEM layouts and TMAs ====
        # (attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages)
        _smem_layout_specs = [
            ("sV_layout",   sm100_utils.make_smem_layout_a, tiled_mma_VdO,    self.mma_tiler_VdO,    self.dtype, self.num_stages_V),
            ("sdO_layout",  sm100_utils.make_smem_layout_b, tiled_mma_VdO,    self.mma_tiler_VdO,    self.dtype, self.num_stages_dO),
            ("sPt_layout",  sm100_utils.make_smem_layout_a, tiled_mma_PtdOt,  self.mma_tiler_PtdOt,  self.dtype, self.num_stages_Pt),
            ("sdOt_layout", sm100_utils.make_smem_layout_b, tiled_mma_PtdOt,  self.mma_tiler_PtdOt,  self.dtype, self.num_stages_dOt),
            ("sdSt_layout", sm100_utils.make_smem_layout_a, tiled_mma_dStQvt, self.mma_tiler_dStQvt, self.dtype, self.num_stages_dSt),
            ("sQvt_layout", sm100_utils.make_smem_layout_b, tiled_mma_dStQvt, self.mma_tiler_dStQvt, self.dtype, self.num_stages_Qvt),
        ]
        if const_expr(self.recompute_P):
            # K-major Qv B-operand for the S^T gemm; the 2 "stages" are the
            # two hdimv splits, cycling through the dOt/Qvt multiplex buffer.
            _smem_layout_specs.append(
                ("sQvB_layout", sm100_utils.make_smem_layout_b, tiled_mma_VdO, self.mma_tiler_VdO, self.dtype, self.num_hdimv_splits)
            )
            if const_expr(self.has_qk):
                _smem_layout_specs += [
                    ("sKr_layout", sm100_utils.make_smem_layout_a, tiled_mma_VdO, self.mma_tiler_Kr, self.dtype, self.num_stages_Kr),
                    ("sQr_layout", sm100_utils.make_smem_layout_b, tiled_mma_VdO, self.mma_tiler_Kr, self.dtype, self.num_stages_Qr),
                ]
        for attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages in _smem_layout_specs:
            ab_kwarg = "a_dtype" if make_fn is sm100_utils.make_smem_layout_a else "b_dtype"
            staged = make_fn(
                tiled_mma=tiled_mma,
                mma_tiler_mnk=mma_tiler,
                num_stages=num_stages,
                **{ab_kwarg: dtype},
            )
            setattr(self, f"{attr}_staged", staged)
            setattr(self, attr, cute.select(staged, mode=[0, 1, 2]))

        # Prepare additional SMEM load layouts
        if const_expr(not self.recompute_P):
            self.P_layout_major = cutlass.utils.LayoutEnum.from_tensor(mP)
        else:
            self.P_layout_major = cutlass.utils.LayoutEnum.from_tensor(mdS)
        self.sP_layout_staged = sm100_utils.make_smem_layout_epi(
            self.dtype, self.P_layout_major, self.tile_P, self.num_stages_P
        )
        self.sP_layout = cute.select(self.sP_layout_staged, mode=[0, 1])
        self.sScaleP_layout_staged = cute.make_layout((self.tile_m, self.num_stages_scaleP))
        self.sScaleP_layout = cute.select(self.sScaleP_layout_staged, mode=[0])
        self.sdPsum_layout_staged = cute.make_layout((self.tile_m, self.num_stages_dPsum))
        self.sdPsum_layout = cute.select(self.sdPsum_layout_staged, mode=[0])
        self.sLse_layout_staged = cute.make_layout((self.tile_m, self.num_stages_lse))
        self.sLse_layout = cute.select(self.sLse_layout_staged, mode=[0])
        self.sBitmask_layout = cute.make_layout(
            (self.cluster_tile_n // 32, self.num_stages_bitmask)
        )

        # ==== TMA load ====
        _tma_bytes_specs = [
            ("tma_copy_bytes_V",   self.dtype, self.sV_layout),
            ("tma_copy_bytes_dO",  self.dtype, self.sdO_layout),
            ("tma_copy_bytes_dOt", self.dtype, self.sdOt_layout),
            ("tma_copy_bytes_Qvt", self.dtype, self.sQvt_layout),
        ]
        if const_expr(self.recompute_P):
            _tma_bytes_specs.append(("tma_copy_bytes_QvB", self.dtype, self.sQvB_layout))
            if const_expr(self.has_qk):
                _tma_bytes_specs.append(("tma_copy_bytes_Qr", self.dtype, self.sQr_layout))
        for attr, dtype, layout in _tma_bytes_specs:
            setattr(self, attr, cute.size_in_bytes(dtype, layout) * self.cta_group_size)

        assert self.tma_copy_bytes_dOt == self.tma_copy_bytes_Qvt
        if const_expr(self.recompute_P):
            # Role swap (see load()): QvB rides pipeline_dO (tx_count is the
            # dO stage bytes) and dO rides the dOt/Qvt multiplex (tx_count is
            # the dOt stage bytes) — each pairing must match its pipeline's
            # mbarrier expected-transaction count or the kernel hangs/races.
            assert self.tma_copy_bytes_QvB == self.tma_copy_bytes_dO
            assert self.tma_copy_bytes_dO == self.tma_copy_bytes_dOt
        self.tma_copy_bytes_P = cute.size_in_bytes(self.dtype, self.sP_layout)
        self.tma_copy_bytes_scaleP = cute.size_in_bytes(self.dtype_scale, self.sScaleP_layout)
        self.tma_copy_bytes_dPsum = cute.size_in_bytes(self.dtype_scale, self.sdPsum_layout)
        self.tma_copy_bytes_lse = cute.size_in_bytes(self.dtype_scale, self.sLse_layout)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_VdO.thr_id.shape,)
        )
        cta_shape = cta_layout_vmnk.shape

        def make_tma(make_fn, mX, smem_layout, mma_tiler, tiled_mma):
            return make_fn(tma_load_op, mX, smem_layout, mma_tiler, tiled_mma, cta_shape)

        A, B = cute.nvgpu.make_tiled_tma_atom_A, cute.nvgpu.make_tiled_tma_atom_B

        # (atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma)
        _tma_specs = [
            ("tma_atom_dO",  "tma_tensor_dO",  B, mdO,  self.sdO_layout,  self.mma_tiler_VdO,    tiled_mma_VdO),
            ("tma_atom_dOt", "tma_tensor_dOt", B, mdOt, self.sdOt_layout, self.mma_tiler_PtdOt,  tiled_mma_PtdOt),
            ("tma_atom_Qvt", "tma_tensor_Qvt", B, mQvt, self.sQvt_layout, self.mma_tiler_dStQvt, tiled_mma_dStQvt),
        ]
        _tmas = {}
        for atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma in _tma_specs:
            _tmas[atom_name], _tmas[tensor_name] = (
                make_tma(make_fn, m, smem_layout, mma_tiler, tiled_mma)
            )

        (tma_atom_dO,  tma_tensor_dO,
         tma_atom_dOt, tma_tensor_dOt,
         tma_atom_Qvt, tma_tensor_Qvt) = _tmas.values()

        tma_atom_P = tma_tensor_P = None
        tma_atom_QvB = tma_tensor_QvB = None
        tma_atom_Qr = tma_tensor_Qr = None
        if const_expr(not self.recompute_P):
            # Make TMA load for P separately
            tma_atom_P, tma_tensor_P = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mP,
                self.sP_layout,
                self.tile_P,
            )
        else:
            tma_atom_QvB, tma_tensor_QvB = make_tma(
                B, mQv, self.sQvB_layout, self.mma_tiler_VdO, tiled_mma_VdO
            )
            if const_expr(self.has_qk):
                tma_atom_Qr, tma_tensor_Qr = make_tma(
                    B, mQ, self.sQr_layout, self.mma_tiler_Kr, tiled_mma_VdO
                )

        # ==== TMA store ====
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()  

        self.dS_layout_major = cutlass.utils.LayoutEnum.from_tensor(mdS)
        self.dV_layout_major = cutlass.utils.LayoutEnum.from_tensor(mdV)
        # (tile_m, tile_n, dS_stages) = (nheads, 64, dS_stage)
        sdS_layout_staged = sm100_utils.make_smem_layout_epi(
            self.dtype, self.dS_layout_major, self.tile_dS, self.num_stages_dSt
        )
        # (tile_n, 32, dV_epi_stages) = (64, 32, 4 x 2)
        sdV_layout_staged = sm100_utils.make_smem_layout_epi(
            self.dtype_dV, self.dV_layout_major, self.tile_dV, self.num_epi_stages_dV
        )
        tma_atom_dS, tma_tensor_dS = cpasync.make_tiled_tma_atom(
            tma_store_op, mdS, cute.select(sdS_layout_staged, mode=[0, 1]), self.tile_dS
        )
        # fmt: on

        # ==== Allocate shared memory ====
        SharedStorage = self._get_shared_storage_cls()

        # ==== Tile scheduler ====
        TileScheduler = self.TileScheduler

        fa_printf(1, "mdO = {}", mdO.layout)
        batch_size_for_sched = cute.size(mdO.shape[3]) if const_expr(mCuSeqlensQ is None) else 1
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mdO.shape[0]), self.tile_m),
            num_head=cute.size(mdO.shape[2]),
            num_batch=batch_size_for_sched,
            num_splits=1,
            seqlen_k=cute.size(mV.shape[0]),
            headdim=self.hdim,
            headdim_v=self.hdimv,
            total_q=cute.size(mdO.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mdO.shape[0]) * cute.size(mdO.shape[3]),
            tile_shape_mn=self.cta_tiler_mn,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=self.static_persistent,
            lpt=False,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=True,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        fa_printf(1, "grid = {}", grid_dim)

        # ==== Named Barrier ====
        self.cpasync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.Cpasync),
            num_threads=self.num_cpasync_load_threads,
        )
        self.softmax_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.Softmax),
            num_threads=self.num_softmax_threads,
        )
        self.epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.Epilogue),
            num_threads=self.num_epilogue_threads,
        )

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        # ==== Launch kernel ====
        block_dim = (self.num_threads, 1, 1)
        self.kernel(
            mV,
            mdV,
            tma_tensor_dO,
            tma_tensor_dOt,
            tma_tensor_Qvt,
            tma_tensor_P,
            tma_tensor_dS,
            mScaleP,
            mdPsum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mIndexTopk,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_Qvt,
            tma_atom_P,
            tma_atom_dS,
            self.sV_layout_staged,
            self.sdO_layout_staged,
            self.sdOt_layout_staged,
            self.sQvt_layout_staged,
            self.sP_layout_staged,  # load P
            sdS_layout_staged,  # store dS
            sdV_layout_staged,
            self.sPt_layout_staged,  # mma Pt
            self.sdSt_layout_staged,  # mma dSt
            self.sScaleP_layout_staged,
            self.sdPsum_layout_staged,
            tiled_mma_VdO,
            tiled_mma_PtdOt,
            tiled_mma_dStQvt,
            softmax_scale,
            softmax_scale_log2,
            topk_length_dynamic,
            tile_sched_params,
            SharedStorage,
            # ==== recompute_P extras ====
            tma_tensor_QvB,
            tma_tensor_Qr,
            mK,
            mLseLog2,
            tma_atom_QvB,
            tma_atom_Qr,
            getattr(self, "sQvB_layout_staged", None),
            getattr(self, "sKr_layout_staged", None),
            getattr(self, "sQr_layout_staged", None),
            self.sLse_layout_staged,
            self.sBitmask_layout,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mV: cute.Tensor,
        mdV: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: cute.Tensor,
        mQvt: cute.Tensor,
        mP: Optional[cute.Tensor],
        mdS: cute.Tensor,
        mScaleP: Optional[cute.Tensor],
        mdPsum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mIndexTopk: Optional[cute.Tensor],
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: cute.CopyAtom,
        tma_atom_Qvt: cute.CopyAtom,
        tma_atom_P: Optional[cute.CopyAtom],
        tma_atom_dS: cute.CopyAtom,
        sV_layout_staged: cute.ComposedLayout,
        sdO_layout_staged: cute.ComposedLayout,
        sdOt_layout_staged: cute.ComposedLayout,
        sQvt_layout_staged: cute.ComposedLayout,
        sP_layout_staged: cute.ComposedLayout,
        sdS_layout_staged: cute.ComposedLayout,
        sdV_layout_staged: cute.ComposedLayout,
        sPt_layout_staged: cute.ComposedLayout,
        sdSt_layout_staged: cute.ComposedLayout,
        sScaleP_layout_staged: cute.Layout,
        sdPsum_layout_staged: cute.Layout,
        tiled_mma_VdO: cute.TiledMma,
        tiled_mma_PtdOt: cute.TiledMma,
        tiled_mma_dStQvt: cute.TiledMma,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        topk_length_dynamic: Optional[Int32],
        tile_sched_params: ParamsBase,
        SharedStorage: cutlass.Constexpr[Callable],
        # ==== recompute_P extras ====
        mQvB: Optional[cute.Tensor] = None,
        mQr: Optional[cute.Tensor] = None,
        mKr: Optional[cute.Tensor] = None,
        mLseLog2: Optional[cute.Tensor] = None,
        tma_atom_QvB: Optional[cute.CopyAtom] = None,
        tma_atom_Qr: Optional[cute.CopyAtom] = None,
        sQvB_layout_staged: Optional[cute.ComposedLayout] = None,
        sKr_layout_staged: Optional[cute.ComposedLayout] = None,
        sQr_layout_staged: Optional[cute.ComposedLayout] = None,
        sLse_layout_staged: Optional[cute.Layout] = None,
        sBitmask_layout: Optional[cute.Layout] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_VdO.thr_id.shape,)
        )
        mma_tile_coord_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = mma_tile_coord_v == 0

        # ==== Allocate SMEM ====
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ==== Prepare TMEM allocator ====
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100_MLA2CTA.TmemPtr),
            num_threads=self.num_mma_threads + self.num_softmax_threads + self.num_epilogue_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # ==== Prefetch TMA descriptors ====
        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dOt)
            cpasync.prefetch_descriptor(tma_atom_Qvt)
            if const_expr(tma_atom_P is not None):
                cpasync.prefetch_descriptor(tma_atom_P)
            if const_expr(tma_atom_QvB is not None):
                cpasync.prefetch_descriptor(tma_atom_QvB)
            if const_expr(tma_atom_Qr is not None):
                cpasync.prefetch_descriptor(tma_atom_Qr)
            cpasync.prefetch_descriptor(tma_atom_dS)

        # ==== Construct pipelines ====
        tma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        mma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_warps = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads // 32)
        store_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads)
        epi_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilogue_threads)
        sm_threads_cluster = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_softmax_threads * self.cta_group_size
        )
        epi_threads_cluster = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_epilogue_threads * self.cta_group_size
        )
        cpasync_load_threads = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_cpasync_load_threads
        )
        relay_warps_cluster = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.cta_group_size)
        relay_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_relay_threads)

        TmaUmma = pipeline.PipelineTmaUmma
        TmaAsync = pipeline.PipelineTmaAsync
        AsyncUmma = pipeline.PipelineAsyncUmma
        UmmaAsync = pipeline.PipelineUmmaAsync
        Async = pipeline.PipelineAsync

        def make_pipeline(cls, mbar_ptr, num_stages, producer, consumer, tx_count=None):
            return cls.create(
                barrier_storage=mbar_ptr.data_ptr(),
                num_stages=num_stages,
                producer_group=producer,
                consumer_group=consumer,
                defer_sync=True,
                **(
                    {"cta_layout_vmnk": cta_layout_vmnk}
                    if cls is not Async and cls is not TmaAsync
                    else {}
                ),
                **({"tx_count": tx_count} if tx_count is not None else {}),
            )

        # Unconditional pipelines
        # fmt: off
        # TmaUmma: dO, dOt & Qvt
        pipeline_dO = make_pipeline(TmaUmma, storage.mbar_ptr_dO, self.num_stages_dO, tma_warp, mma_warp, self.tma_copy_bytes_dO)
        pipeline_dOt_Qvt = make_pipeline(TmaUmma, storage.mbar_ptr_dOt_Qvt, self.num_stages_Qvt, tma_warp, mma_warp, self.tma_copy_bytes_dOt)
        pipeline_dPsum = make_pipeline(TmaAsync, storage.mbar_ptr_dPsum, self.num_stages_dPsum, tma_warp, sm_warps, self.tma_copy_bytes_dPsum)
        # AsyncUmma: Pt => dV mma, dSt => dV mma
        pipeline_Pt = make_pipeline(AsyncUmma, storage.mbar_ptr_Pt, self.num_stages_Pt, sm_threads_cluster, mma_warp)
        pipeline_dSt = make_pipeline(AsyncUmma, storage.mbar_ptr_dSt, self.num_stages_dSt, sm_threads_cluster, mma_warp)
        # UmmaAsync: dPt, dV
        pipeline_dPt = make_pipeline(UmmaAsync, storage.mbar_ptr_dPt, self.num_stages_dPt, mma_warp, sm_threads_cluster)
        pipeline_dV = make_pipeline(UmmaAsync, storage.mbar_ptr_dV, self.num_stages_dV, mma_warp, epi_threads_cluster)
        # Async: dV_epi
        pipeline_dV_epi = make_pipeline(Async, storage.mbar_ptr_dV_epi, self.num_stages_dV, tma_warp, store_warp)

        pipeline_V         = make_pipeline(AsyncUmma, storage.mbar_ptr_V,         self.num_stages_V, relay_warps_cluster,  mma_warp)
        pipeline_V_cpasync = make_pipeline(Async,     storage.mbar_ptr_V_cpasync, self.num_stages_V, cpasync_load_threads, relay_threads)

        pipeline_P = pipeline_scaleP = None
        pipeline_St = pipeline_lse = pipeline_bitmask = None
        pipeline_Kr = pipeline_Kr_cpasync = pipeline_Qr = None
        if const_expr(not self.recompute_P):
            # TmaAsync: P, scaleP
            pipeline_P = make_pipeline(TmaAsync, storage.mbar_ptr_P, self.num_stages_P, tma_warp, sm_warps, self.tma_copy_bytes_P)
            pipeline_scaleP = make_pipeline(TmaAsync, storage.mbar_ptr_scaleP, self.num_stages_scaleP, tma_warp, sm_warps, self.tma_copy_bytes_scaleP)
        else:
            # UmmaAsync: S^T from the mma warp to the softmax warps (both CTAs)
            pipeline_St = make_pipeline(UmmaAsync, storage.mbar_ptr_St, self.num_stages_St, mma_warp, sm_threads_cluster)
            # TmaAsync: lse (log2 units), once per work tile like dPsum
            pipeline_lse = make_pipeline(TmaAsync, storage.mbar_ptr_lse, self.num_stages_lse, tma_warp, sm_warps, self.tma_copy_bytes_lse)
            if const_expr(not self.disable_bitmask):
                # Async: gather warps -> softmax warps (per CTA)
                pipeline_bitmask = make_pipeline(Async, storage.mbar_ptr_bitmask, self.num_stages_bitmask, cpasync_load_threads, sm_threads)
            if const_expr(self.has_qk):
                pipeline_Kr         = make_pipeline(AsyncUmma, storage.mbar_ptr_Kr,         self.num_stages_Kr, relay_warps_cluster,  mma_warp)
                pipeline_Kr_cpasync = make_pipeline(Async,     storage.mbar_ptr_Kr_cpasync, self.num_stages_Kr, cpasync_load_threads, relay_threads)
                pipeline_Qr = make_pipeline(TmaUmma, storage.mbar_ptr_Qr, self.num_stages_Qr, tma_warp, mma_warp, self.tma_copy_bytes_Qr)
        # fmt: on

        # ==== Zero-fill the gathered-V smem stages ====
        # Masked top-k slots skip their cp.async in the gather, so the FIRST
        # fill round of each sV stage otherwise feeds whatever smem held at
        # launch into the dP (and recompute-S) gemms. P is 0 on masked slots,
        # but if the stale bits decode to NaN/Inf then dS = 0 * NaN = NaN.
        # After the first round, stale rows hold finite V values from earlier
        # groups, so this one-time zero-fill suffices for both the load-p and
        # the recompute paths. The non-relaxed (release) init arrive below +
        # pipeline_init_wait order it cluster-wide before any pipeline
        # traffic; the proxy fence covers the UMMA reads.
        sV_fill_words = cute.cosize(sV_layout_staged) * self.dtype.width // 32
        sV_words = cute.make_tensor(
            cute.recast_ptr(storage.sV.data_ptr(), dtype=Int32),
            cute.make_layout(sV_fill_words),
        )
        tidx_fill = cute.arch.thread_idx()[0]
        for j in cutlass.range_constexpr(cute.ceil_div(sV_fill_words, self.num_threads)):
            fill_idx = tidx_fill + j * self.num_threads
            if fill_idx < sV_fill_words:
                sV_words[fill_idx] = Int32(0)
        cute.arch.fence_view_async_shared()

        pipeline.pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=False)

        # ==== Get SMEM tensors ====
        # fmt: off
        if const_expr(not self.merged_dS):
            sdS_storage = storage.sdS
        else:
            sdS_storage = storage.sP
        if const_expr(not self.recompute_P):
            sdO_storage = storage.sdO
        else:
            # Role swap: the stationary 64 KiB buffer (storage.sdO) holds the
            # K-major Qv B-operand of the S^T gemm; dO reloads per group
            # through the dOt/Qvt multiplex buffer instead (L2-resident).
            sdO_storage = storage.sQv
        sV, sdO, sP, sPt, sdOt, sdS, sdSt, sQvt = (
            store.get_tensor(layout.outer, swizzle=layout.inner)
            for store, layout in [
                (storage.sV,  sV_layout_staged),
                (sdO_storage, sdO_layout_staged),   # recompute_P: {dO, dOt, Qvt, dV} overlap
                (storage.sP,  sP_layout_staged),    # P & Pt overlap
                (storage.sP,  sPt_layout_staged),   # P & Pt overlap
                (storage.sQv, sdOt_layout_staged),  # {dOt, Qvt, dV} overlap
                (sdS_storage, sdS_layout_staged),   # dS & dSt overlap (recompute_P: also P)
                (sdS_storage, sdSt_layout_staged),  # dS & dSt overlap (recompute_P: also P)
                (storage.sQv, sQvt_layout_staged),  # {dOt, Qvt, dV} overlap
            ]
        )
        sdV = cute.make_tensor(
            cute.recast_ptr(sdOt.iterator, sdV_layout_staged.inner, self.dtype_acc), sdV_layout_staged.outer
        )
        assert cute.cosize(sdV) * self.dtype_acc.width // self.dtype.width == cute.cosize(sdOt)

        sScaleP = sdPsum = sQvB = sLse = sQr = sKr = sBitmask = None
        if const_expr(not self.recompute_P):
            sScaleP = storage.sScaleP.get_tensor(sScaleP_layout_staged)
        else:
            sQvB = storage.sdO.get_tensor(sQvB_layout_staged.outer, swizzle=sQvB_layout_staged.inner)  # stationary
            assert cute.cosize(sQvB) == cute.cosize(sdO)
            sLse = storage.sLse.get_tensor(sLse_layout_staged)
            if const_expr(not self.disable_bitmask):
                sBitmask = storage.sBitmask.get_tensor(sBitmask_layout)
            if const_expr(self.has_qk):
                sQr = storage.sQr.get_tensor(sQr_layout_staged.outer, swizzle=sQr_layout_staged.inner)
                sKr = storage.sKr.get_tensor(sKr_layout_staged.outer, swizzle=sKr_layout_staged.inner)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout_staged)
        # fmt: on

        # ==== Get thread MMAs and accumulator fragments ====
        thr_mma_VdO = tiled_mma_VdO.get_slice(mma_tile_coord_v)
        thr_mma_PtdOt = tiled_mma_PtdOt.get_slice(mma_tile_coord_v)
        thr_mma_dStQvt = tiled_mma_dStQvt.get_slice(mma_tile_coord_v)

        acc_shape_dPt = thr_mma_VdO.partition_shape_C(self.mma_tiler_VdO[:2])
        acc_shape_dVi = thr_mma_PtdOt.partition_shape_C(self.mma_tiler_PtdOt[:2])
        tdPtdP_fake = thr_mma_VdO.make_fragment_C(acc_shape_dPt)
        tdVtdV0_fake = thr_mma_PtdOt.make_fragment_C(acc_shape_dVi)
        tdVtdV1_fake = thr_mma_PtdOt.make_fragment_C(acc_shape_dVi)
        # tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_offset_dP, tdPtdP.layout)
        # tdVtdV0 = cute.make_tensor(tdVtdV0.iterator + self.tmem_offset_dV0, tdVtdV0.layout)
        # tdVtdV1 = cute.make_tensor(tdVtdV1.iterator + self.tmem_offset_dV1, tdVtdV1.layout)

        block_info = BlockInfo(
            self.tile_m * self.cta_group_size,
            self.tile_n,
            is_causal=self.is_causal,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mdO.shape[0] if const_expr(not self.pack_gqa) else mdO.shape[0][1],
            seqlen_k_static=mV.shape[0],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_clc_consumer_warps_per_cta = self.num_threads // cute.arch.WARP_SIZE
            num_clc_consumer_warps = num_clc_consumer_warps_per_cta * self.cta_group_size
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cute.arch.WARP_SIZE * num_clc_consumer_warps
            )
            sched_ctx = SchedulerState.create_clc(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, ctx=sched_ctx)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), (
            f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"
        )

        pipeline.pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        if const_expr(self.use_cpasync_load_KV):
            if warp_idx == self.relay_warp_id:
                if const_expr(self.num_regs_load < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_load)
                self.relay(
                    pipeline_V,
                    pipeline_V_cpasync,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler=tile_scheduler,
                    pipeline_Kr=pipeline_Kr,
                    pipeline_Kr_cpasync=pipeline_Kr_cpasync,
                )

            if warp_idx in self.cpasync_load_warp_indices:
                if const_expr(self.num_regs_cpasync < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_cpasync)
                self.load_cpasync(
                    mIndexTopk,
                    mV,
                    sV,
                    pipeline_V,
                    pipeline_V_cpasync,
                    topk_length_dynamic,
                    block_info,
                    SeqlenInfoCls,
                    mCuSeqlensQ,
                    tile_scheduler=tile_scheduler,
                    mKr=mKr,
                    sKr=sKr,
                    pipeline_Kr=pipeline_Kr,
                    pipeline_Kr_cpasync=pipeline_Kr_cpasync,
                    sBitmask=sBitmask,
                    pipeline_bitmask=pipeline_bitmask,
                )

        if warp_idx == self.load_warp_id:
            if const_expr(self.num_regs_load < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                mdO,
                mP,
                mdOt,
                mQvt,
                mScaleP,
                mdPsum,
                sdO,
                sP,
                sdOt,
                sQvt,
                sScaleP,
                sdPsum,
                tma_atom_dO,
                tma_atom_P,
                tma_atom_dOt,
                tma_atom_Qvt,
                pipeline_dO,
                pipeline_P,
                pipeline_dOt_Qvt,
                pipeline_Pt,
                pipeline_dV_epi,
                pipeline_scaleP,
                pipeline_dPsum,
                thr_mma_VdO,
                thr_mma_PtdOt,
                thr_mma_dStQvt,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
                mQvB=mQvB,
                mQr=mQr,
                mLseLog2=mLseLog2,
                sQvB=sQvB,
                sQr=sQr,
                sLse=sLse,
                tma_atom_QvB=tma_atom_QvB,
                tma_atom_Qr=tma_atom_Qr,
                pipeline_Qr=pipeline_Qr,
                pipeline_lse=pipeline_lse,
            )

        if warp_idx == self.mma_warp_id:
            if const_expr(self.num_regs_mma < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # ==== Allocate TMEM ====
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_offset_dP, tdPtdP_fake.layout)
            tdVtdV0 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV0, tdVtdV0_fake.layout)
            tdVtdV1 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV1, tdVtdV1_fake.layout)
            tdStS = None
            if const_expr(self.recompute_P):
                # S^T accumulator shares the dP^T fragment layout
                tdStS = cute.make_tensor(tmem_ptr + self.tmem_offset_S, tdPtdP_fake.layout)
            self.mma(
                sV,
                sdO,
                sPt,
                sdOt,
                sdSt,
                sQvt,
                tdPtdP,
                tdVtdV0,
                tdVtdV1,
                tiled_mma_VdO,
                tiled_mma_PtdOt,
                tiled_mma_dStQvt,
                pipeline_V,
                pipeline_dO,
                pipeline_dPt,
                pipeline_Pt,
                pipeline_dOt_Qvt,
                pipeline_dSt,
                pipeline_dV,
                is_leader_cta,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
                tdStS=tdStS,
                sQvB=sQvB,
                sKr=sKr,
                sQr=sQr,
                pipeline_St=pipeline_St,
                pipeline_Kr=pipeline_Kr,
                pipeline_Qr=pipeline_Qr,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx in self.softmax_warp_indices:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_offset_dP, tdPtdP_fake.layout)
            tdStS = None
            if const_expr(self.recompute_P):
                tdStS = cute.make_tensor(tmem_ptr + self.tmem_offset_S, tdPtdP_fake.layout)
            self.compute_loop(
                softmax_scale,
                softmax_scale_log2,
                thr_mma_VdO,
                tdPtdP,
                sP,
                sdS,
                sScaleP,
                sdPsum,
                mdS,
                tma_atom_dS,
                pipeline_P,
                pipeline_Pt,
                pipeline_dPt,
                pipeline_dSt,
                pipeline_scaleP,
                pipeline_dPsum,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
                tdStS=tdStS,
                sLse=sLse,
                sBitmask=sBitmask,
                pipeline_St=pipeline_St,
                pipeline_lse=pipeline_lse,
                pipeline_bitmask=pipeline_bitmask,
            )
            tmem_alloc_barrier.arrive()

        if warp_idx in self.epilogue_warp_indices:
            if const_expr(self.num_regs_epilogue < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            elif const_expr(self.num_regs_epilogue > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tdVtdV0 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV0, tdVtdV0_fake.layout)
            tdVtdV1 = cute.make_tensor(tmem_ptr + self.tmem_offset_dV1, tdVtdV1_fake.layout)
            self.dVacc_store(
                mIndexTopk,
                mdV,
                sdV,
                tdVtdV0,
                tdVtdV1,
                thr_mma_PtdOt,
                pipeline_dV,
                pipeline_dV_epi,
                topk_length_dynamic,
                block_info,
                SeqlenInfoCls,
                mCuSeqlensQ,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            cta_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if cute.arch.thread_idx()[0] == self.clc_scheduler_warp_id * cute.arch.WARP_SIZE:
                fa_printf(
                    3,
                    "[CLC] query sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def relay(
        self,
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_V_cpasync: pipeline.PipelineAsync,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        pipeline_Kr: Optional[pipeline.PipelineAsyncUmma] = None,
        pipeline_Kr_cpasync: Optional[pipeline.PipelineAsync] = None,
    ):
        # ==== Make pipeline states ====
        # pipeline_V producer
        # pipeline_V_cpasync consumer
        producer_state_V = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_V
        )
        consumer_state_V = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_V
        )
        relay_V_fn = partial(self.relay_inner, pipeline_V_cpasync, pipeline_V)
        if const_expr(pipeline_Kr is not None):
            producer_state_Kr = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_Kr
            )
            consumer_state_Kr = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages=self.num_stages_Kr
            )
            relay_Kr_fn = partial(self.relay_inner, pipeline_Kr_cpasync, pipeline_Kr)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            # seqlen = SeqlenInfoCls(batch_idx)

            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            # ==== Mainloop ====
            for _ in cutlass.range(num_n_block_groups, unroll=1):
                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_state_V, producer_state_V = relay_V_fn(
                        consumer_state_V, producer_state_V
                    )
                if const_expr(pipeline_Kr is not None):
                    consumer_state_Kr, producer_state_Kr = relay_Kr_fn(
                        consumer_state_Kr, producer_state_Kr
                    )

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_V.producer_tail(producer_state_V)
        if const_expr(pipeline_Kr is not None):
            pipeline_Kr.producer_tail(producer_state_Kr)

    @cute.jit
    def relay_inner(
        self,
        pipeline_cpasync: pipeline.PipelineAsync,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        consumer_state: pipeline.PipelineState,
        producer_state: pipeline.PipelineState,
    ):
        pipeline_cpasync.consumer_wait(consumer_state)
        with cute.arch.elect_one():
            pipeline_mma.producer_commit(producer_state)
        consumer_state.advance()
        producer_state.advance()
        return consumer_state, producer_state

    @cute.jit
    def load_cpasync(
        self,
        mIndexTopk: cute.Tensor,
        mV: cute.Tensor,
        sV: cute.Tensor,
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_V_cpasync: pipeline.PipelineAsync,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
        mKr: Optional[cute.Tensor] = None,
        sKr: Optional[cute.Tensor] = None,
        pipeline_Kr: Optional[pipeline.PipelineAsyncUmma] = None,
        pipeline_Kr_cpasync: Optional[pipeline.PipelineAsync] = None,
        sBitmask: Optional[cute.Tensor] = None,
        pipeline_bitmask: Optional[pipeline.PipelineAsync] = None,
    ):
        # ==== cpasync load warpgroup ====
        # Description: loads tiles of V (and K_rope with recompute_P) from gmem
        # to smem using cpasync; with recompute_P also emits the validity
        # bitmask for the softmax warps.
        # produces: V (, Kr, bitmask)
        # consumes: -

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % self.num_cpasync_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_cpasync_load_threads // 32
        )

        # ==== Make pipeline states ====
        # producer: acquire PipelineAsyncUmma <- mma
        # producer: commit  PipelineAsync     -> relay
        producer_state_V = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, stages=self.num_stages_V
        )
        if const_expr(pipeline_Kr is not None):
            producer_state_Kr = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_Kr
            )
        if const_expr(pipeline_bitmask is not None):
            producer_state_bitmask = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_bitmask
            )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)

            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            if const_expr(seqlen.has_cu_seqlens_q):
                # m_block means absolute m_idx
                mIndexTopk_cur = mIndexTopk[None, m_block]
            else:
                mIndexTopk_cur = mIndexTopk[None, m_block, batch_idx]

            if const_expr(self.is_causal):
                m_local_idx = (
                    m_block - seqlen.offset_q if const_expr(seqlen.has_cu_seqlens_q) else m_block
                )
                # NOTE: the token-chunked backward wrapper (interface.py,
                # _flash_attn_bwd_sparse_mla) reproduces this exact limit per
                # chunk by shrinking the kernel's K extent (non-varlen: sliced
                # v/dv/k views; varlen: clamped cu_seqlens_k end offsets). If
                # this formula changes (e.g. local windows, seqused), the
                # chunked wrapper must change with it, or chunked recompute-P
                # gradients silently diverge from the forward's mask.
                seqlen_k_limit = m_local_idx + 1 + seqlen.seqlen_k - seqlen.seqlen_q
            else:
                seqlen_k_limit = seqlen.seqlen_k
            cpasync_gather_kv_manager = CpasyncGatherKVManager.create(
                mIndexTopk_cur,
                cta_rank_in_cluster,
                tidx,
                warp_idx,
                self.topk_length,
                seqlen_k_limit,
                self.cluster_tile_n,
                self.hdim,
                self.hdimv,
                self.num_hdimv_splits,
                self.num_cpasync_load_threads,
                mV.element_type,
                self.cta_group_size,
                self.cpasync_barrier,
                self.disable_bitmask,
                sBitmask,
                pipeline_bitmask,
            )

            # (seqlen_k, hdimv)
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx]

            load_V = partial(
                self.cpasync_gather_load_KV,
                cpasync_gather_kv_manager,
                pipeline_V,
                pipeline_V_cpasync,
                sV,
                False,
                "V",
                mV_cur,
            )
            if const_expr(pipeline_Kr is not None):
                # (seqlen_k, hdim)
                mKr_cur = seqlen.offset_batch_K(mKr, batch_idx, dim=3)[None, None, head_idx]
                load_Kr = partial(
                    self.cpasync_gather_load_KV,
                    cpasync_gather_kv_manager,
                    pipeline_Kr,
                    pipeline_Kr_cpasync,
                    sKr,
                    False,
                    "K",
                    mKr_cur,
                )

            # ==== Mainloop ====
            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                cpasync_gather_kv_manager.load_index_topk(n_block_group, transpose=False)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_V = load_V(producer_state_V, d_offset=split * self.hdimv // 2)
                if const_expr(pipeline_Kr is not None):
                    producer_state_Kr = load_Kr(producer_state_Kr)
                if const_expr(pipeline_bitmask is not None):
                    producer_state_bitmask = cpasync_gather_kv_manager.compute_bitmask(
                        producer_state_bitmask
                    )

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_V.producer_tail(producer_state_V)
        if const_expr(pipeline_Kr is not None):
            pipeline_Kr.producer_tail(producer_state_Kr)
        # note: no producer_tail for the *_cpasync pipelines (their
        # producer_acquire is never used), but compute_bitmask DOES use
        # pipeline_bitmask.producer_acquire, so it must be drained — same
        # invariant as the forward kernel's gather loader.
        if const_expr(pipeline_bitmask is not None):
            pipeline_bitmask.producer_tail(producer_state_bitmask)

    @cute.jit
    def cpasync_gather_load_KV(
        self,
        cpasync_gather_kv_manager: CpasyncGatherKVManager,
        pipeline_mma: pipeline.PipelineAsyncUmma,
        pipeline_cpasync: pipeline.PipelineAsync,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
        mX: cute.Tensor,
        producer_state: pipeline.PipelineState,
        d_offset: int = 0,
    ):
        stage = producer_state.index
        pipeline_mma.producer_acquire(producer_state)
        cpasync_gather_kv_manager.load_X(
            mX, sX[None, None, None, stage], transpose, K_or_V, d_offset
        )
        cute.arch.cp_async_commit_group()
        pipeline_cpasync.sync_object_full.arrive_cp_async_mbarrier(stage)
        producer_state.advance()
        return producer_state

    @cute.jit
    def load(
        self,
        mdO: cute.Tensor,
        mP: cute.Tensor,
        mdOt: cute.Tensor,
        mQvt: cute.Tensor,
        mScaleP: Optional[cute.Tensor],
        mdPsum: Optional[cute.Tensor],
        sdO: cute.Tensor,
        sP: cute.Tensor,
        sdOt: cute.Tensor,
        sQvt: cute.Tensor,
        sScaleP: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_P: cute.CopyAtom,
        tma_atom_dOt: cute.CopyAtom,
        tma_atom_Qvt: cute.CopyAtom,
        pipeline_dO: pipeline.PipelineAsync,  # TmaUmma
        pipeline_P: pipeline.PipelineAsync,  # TmaAsync
        pipeline_dOt_Qvt: pipeline.PipelineAsync,  # TmaUmma
        pipeline_Pt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dV_epi: pipeline.PipelineAsync,  # Async
        pipeline_scaleP: pipeline.PipelineAsync,  # TmaAsync
        pipeline_dPsum: pipeline.PipelineAsync,  # TmaAsync
        thr_mma_VdO: cute.ThrMma,
        thr_mma_PtdOt: cute.ThrMma,
        thr_mma_dStQvt: cute.ThrMma,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
        mQvB: Optional[cute.Tensor] = None,
        mQr: Optional[cute.Tensor] = None,
        mLseLog2: Optional[cute.Tensor] = None,
        sQvB: Optional[cute.Tensor] = None,
        sQr: Optional[cute.Tensor] = None,
        sLse: Optional[cute.Tensor] = None,
        tma_atom_QvB: Optional[cute.CopyAtom] = None,
        tma_atom_Qr: Optional[cute.CopyAtom] = None,
        pipeline_Qr: Optional[pipeline.PipelineAsync] = None,  # TmaUmma
        pipeline_lse: Optional[pipeline.PipelineAsync] = None,  # TmaAsync
    ):
        # ==== Load warp ====
        # Description: loads tiles of dO, P, dOt, Qvt from gmem to smem using TMA
        # (recompute_P: QvB instead of P, plus stationary Qr and lse)
        # produces: dO, P/QvB, dOt, Qvt (, Qr, lse)
        # consumes: -
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        lane_idx = cute.arch.lane_idx()

        # ==== Make pipeline states ====
        Producer = pipeline.PipelineUserType.Producer
        producer_state_dO = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dO)
        producer_state_P = pipeline.make_pipeline_state(Producer, stages=self.num_stages_P)
        producer_state_dOt_Qvt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dOt)
        producer_state_dV_epi = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dV)
        producer_state_scaleP = pipeline.make_pipeline_state(
            Producer, stages=self.num_stages_scaleP
        )
        producer_state_dPsum = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dPsum)
        if const_expr(self.recompute_P):
            producer_state_lse = pipeline.make_pipeline_state(
                Producer, stages=self.num_stages_lse
            )
            if const_expr(self.has_qk):
                producer_state_Qr = pipeline.make_pipeline_state(
                    Producer, stages=self.num_stages_Qr
                )

        copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(mCuSeqlensQ is not None):
                m_block -= seqlen.offset_q
            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            # ==== Partition GMEM tensors ====
            # (seqlen_q, topk or hdimv)
            mP_cur = None
            if const_expr(mP is not None):
                mP_cur = seqlen.offset_batch_Q(mP, batch_idx, dim=3)[None, None, head_idx]
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
            mQvB_cur = None
            if const_expr(mQvB is not None):
                mQvB_cur = seqlen.offset_batch_Q(mQvB, batch_idx, dim=3)[None, None, head_idx]
            mQr_cur = None
            if const_expr(mQr is not None):
                mQr_cur = seqlen.offset_batch_Q(mQr, batch_idx, dim=3)[None, None, head_idx]

            # (hdimv, seqlen_q)
            offset = (
                (0, seqlen.offset_q) if const_expr(not self.pack_gqa) else (0, (0, seqlen.offset_q))
            )
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                mQvt_cur = mQvt[None, None, head_idx, batch_idx]
            else:
                mdOt_cur = cute.domain_offset(offset, mdOt[None, None, head_idx])
                mQvt_cur = cute.domain_offset(offset, mQvt[None, None, head_idx])

            gScaleP = None
            if const_expr(mScaleP is not None):
                mScaleP_cur = seqlen.offset_batch_Q(mScaleP, batch_idx, dim=3)[None, None, head_idx]
                # (tile_m, topk//128)
                gScaleP = cute.local_tile(mScaleP_cur, (self.tile_m,), (m_block, None))
            gdPsum = None
            if const_expr(mdPsum is not None):
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2)[None, head_idx]
                # (tile_m)
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (m_block,))
            gLse = None
            if const_expr(mLseLog2 is not None):
                mLse_cur = seqlen.offset_batch_Q(mLseLog2, batch_idx, dim=2)[None, head_idx]
                # (tile_m)
                gLse = cute.local_tile(mLse_cur, (self.tile_m,), (m_block,))

            # (tile_m, tile_n, n_blocks)
            gP = None
            if const_expr(mP_cur is not None):
                gP = cute.local_tile(
                    mP_cur,
                    (self.tile_m, self.tile_n),
                    (m_block, None),
                )
            # (tile_m, hdimv//2, 2)
            gdO = cute.local_tile(
                mdO_cur,
                (self.mma_tiler_VdO[1], self.mma_tiler_VdO[2]),
                (m_block, None),
            )
            # (hdimv//2, tile_m, 2)
            gdOt = cute.local_tile(
                mdOt_cur,
                (self.mma_tiler_PtdOt[1], self.mma_tiler_PtdOt[2]),
                (None, m_block),
            )
            gQvt = cute.local_tile(
                mQvt_cur,
                (self.mma_tiler_dStQvt[1], self.mma_tiler_dStQvt[2]),
                (None, m_block),
            )

            tdPgdO = thr_mma_VdO.partition_B(gdO)
            tdVgdOt = thr_mma_PtdOt.partition_B(gdOt)
            tdVgQvt = thr_mma_dStQvt.partition_B(gQvt)

            # (V, REST)
            if const_expr(mP_cur is not None):
                tPsP, tPgP = cpasync.tma_partition(
                    atom=tma_atom_P,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sP, 0, 2),
                    gmem_tensor=cute.group_modes(gP, 0, 2),
                )
            if const_expr(mQvB_cur is not None):
                # (hdimv//2 tile of Qv as K-major B of the S^T gemm), 2 splits
                gQvB = cute.local_tile(
                    mQvB_cur,
                    (self.mma_tiler_VdO[1], self.mma_tiler_VdO[2]),
                    (m_block, None),
                )
                tSgQvB = thr_mma_VdO.partition_B(gQvB)
                tQvBsQvB, tQvBgQvB = cpasync.tma_partition(
                    atom=tma_atom_QvB,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sQvB, 0, 3),
                    gmem_tensor=cute.group_modes(tSgQvB, 0, 3),
                )
            if const_expr(mQr_cur is not None):
                gQr = cute.local_tile(
                    mQr_cur,
                    (self.mma_tiler_Kr[1], self.mma_tiler_Kr[2]),
                    (m_block, None),
                )
                tSgQr = thr_mma_VdO.partition_B(gQr)
                tQrsQr, tQrgQr = cpasync.tma_partition(
                    atom=tma_atom_Qr,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sQr, 0, 3),
                    gmem_tensor=cute.group_modes(tSgQr, 0, 3),
                )
            tdOsdO, tdOgdO = cpasync.tma_partition(
                atom=tma_atom_dO,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sdO, 0, 3),
                gmem_tensor=cute.group_modes(tdPgdO, 0, 3),
            )
            tdOtsdOt, tdOtgdOt = cpasync.tma_partition(
                atom=tma_atom_dOt,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sdOt, 0, 3),
                gmem_tensor=cute.group_modes(tdVgdOt, 0, 3),
            )
            tQvtsQvt, tQvtgQvt = cpasync.tma_partition(
                atom=tma_atom_Qvt,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sQvt, 0, 3),
                gmem_tensor=cute.group_modes(tdVgQvt, 0, 3),
            )

            if const_expr(not self.recompute_P):
                load_P = partial(self.load_inner, tma_atom_P, tPgP, tPsP, pipeline_P)
                load_scaleP = partial(
                    self.load_inner, copy_atom_stats, gScaleP, sScaleP, pipeline_scaleP, bulk_copy=True
                )
                load_dO = partial(self.load_inner, tma_atom_dO, tdOgdO, tdOsdO, pipeline_dO)
            else:
                # Role swap: QvB is stationary (through pipeline_dO into the
                # storage.sdO buffer); dO reloads per group through the
                # dOt/Qvt multiplex (its sdO view lives in storage.sQv).
                load_dO = partial(self.load_inner, tma_atom_dO, tdOgdO, tdOsdO, pipeline_dOt_Qvt)
                load_QvB = partial(
                    self.load_inner, tma_atom_QvB, tQvBgQvB, tQvBsQvB, pipeline_dO
                )
                load_lse = partial(
                    self.load_inner, copy_atom_stats, gLse, sLse, pipeline_lse, bulk_copy=True
                )
                if const_expr(self.has_qk):
                    load_Qr = partial(self.load_inner, tma_atom_Qr, tQrgQr, tQrsQr, pipeline_Qr)
            load_dOt = partial(self.load_inner, tma_atom_dOt, tdOtgdOt, tdOtsdOt, pipeline_dOt_Qvt)
            load_Qvt = partial(self.load_inner, tma_atom_Qvt, tQvtgQvt, tQvtsQvt, pipeline_dOt_Qvt)
            load_dPsum = partial(
                self.load_inner, copy_atom_stats, gdPsum, sdPsum, pipeline_dPsum, bulk_copy=True
            )

            # ==== Load stationary operands ====
            if const_expr(not self.recompute_P):
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_dO = load_dO(producer_state_dO, block=split)
            else:
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_state_dO = load_QvB(producer_state_dO, block=split)

            producer_state_dPsum = load_dPsum(producer_state_dPsum)
            if const_expr(self.recompute_P):
                producer_state_lse = load_lse(producer_state_lse)
                if const_expr(self.has_qk):
                    producer_state_Qr = load_Qr(producer_state_Qr, block=0)

            # ==== Mainloop ====
            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                n_block = 2 * n_block_group + cta_rank_in_cluster
                if const_expr(not self.recompute_P):
                    # load ScaleP
                    if const_expr(mScaleP is not None):
                        producer_state_scaleP = load_scaleP(producer_state_scaleP, block=n_block_group)
                    pipeline_Pt.producer_acquire(producer_state_P)
                    producer_state_P = load_P(producer_state_P, block=n_block)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        pipeline_dV_epi.producer_acquire(producer_state_dV_epi)
                        producer_state_dV_epi.advance()
                        producer_state_dOt_Qvt = load_dOt(producer_state_dOt_Qvt, block=split)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_dOt_Qvt = load_Qvt(producer_state_dOt_Qvt, block=split)
                else:
                    # 6 loads/group through the 2-stage multiplex buffer, in mma
                    # consumption order: dO (dP gemm), dOt (dV += P^T@dO),
                    # Qvt (dV += dS^T@Qv). dO is the first writer of each
                    # 32 KiB half per group, so the dV-staging guard sits here.
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        pipeline_dV_epi.producer_acquire(producer_state_dV_epi)
                        producer_state_dV_epi.advance()
                        producer_state_dOt_Qvt = load_dO(producer_state_dOt_Qvt, block=split)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_dOt_Qvt = load_dOt(producer_state_dOt_Qvt, block=split)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_dOt_Qvt = load_Qvt(producer_state_dOt_Qvt, block=split)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(not self.recompute_P):
            pipeline_P.producer_tail(producer_state_P)
        else:
            pipeline_lse.producer_tail(producer_state_lse)
            if const_expr(self.has_qk):
                pipeline_Qr.producer_tail(producer_state_Qr)
        pipeline_dO.producer_tail(producer_state_dO)
        pipeline_dOt_Qvt.producer_tail(producer_state_dOt_Qvt)

    @cute.jit
    def load_inner(
        self,
        copy_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        load_pipeline: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        block: Optional[Int32] = None,
        bulk_copy: bool = False,
    ):
        if const_expr(block is not None):
            tXgX = tXgX[(None, block)]
        if const_expr(cute.rank(tXsX) != 1):
            assert cute.rank(tXsX) == 2, f"wrong rank for tXsX, got {cute.rank(tXsX)}"
            stage = producer_state.index
            tXsX = tXsX[(None, stage)]

        load_pipeline.producer_acquire(producer_state)
        mbar_ptr = load_pipeline.producer_get_barrier(producer_state)
        if const_expr(bulk_copy):
            with cute.arch.elect_one():
                cute.copy(copy_atom, tXgX, tXsX, mbar_ptr=mbar_ptr)
        else:
            cute.copy(copy_atom, tXgX, tXsX, tma_bar_ptr=mbar_ptr)
        producer_state.advance()
        return producer_state

    @cute.jit
    def mma(
        self,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sPt: cute.Tensor,
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sQvt: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV0: cute.Tensor,
        tdVtdV1: cute.Tensor,
        tiled_mma_VdO: cute.TiledMma,
        tiled_mma_PtdOt: cute.TiledMma,
        tiled_mma_dStQvt: cute.TiledMma,
        pipeline_V: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dO: pipeline.PipelineAsync,  # TmaUmma
        pipeline_dPt: pipeline.PipelineAsync,  # UmmaAsync
        pipeline_Pt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dOt_Qvt: pipeline.PipelineAsync,  # TmaUmma
        pipeline_dSt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dV: pipeline.PipelineAsync,  # UmmaAsync
        is_leader_cta: Boolean,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
        tdStS: Optional[cute.Tensor] = None,
        sQvB: Optional[cute.Tensor] = None,
        sKr: Optional[cute.Tensor] = None,
        sQr: Optional[cute.Tensor] = None,
        pipeline_St: Optional[pipeline.PipelineAsync] = None,  # UmmaAsync
        pipeline_Kr: Optional[pipeline.PipelineAsync] = None,  # AsyncUmma
        pipeline_Qr: Optional[pipeline.PipelineAsync] = None,  # TmaUmma
    ):
        # ==== mma warp ====
        # Description: Computes dP = V @ dO^T, dV = P^T @ dO, and dV += dS^T @ Qv
        # i.e. dP = gemm(V, dO), dV += gemm(P.T, dO.T), dV += gemm(dS.T, Qv.T)
        # recompute_P additionally computes S^T = V @ Qv^T (+ Kr @ Qr^T) first.
        # Produces: dP, dV (, S^T)
        # Consumes: V, dO, P.T, dO.T, dS.T, Qv.T (, QvB, Kr, Qr)
        lane_idx = cute.arch.lane_idx()

        tdVtdVs = [tdVtdV0, tdVtdV1]

        # Set accumulate = True for dS^T@Qv since we are accumulating on the P^T@dO result
        tiled_mma_dStQvt.set(tcgen05.Field.ACCUMULATE, True)

        # Operands for dP=V@dO^T
        tdPrV = tiled_mma_VdO.make_fragment_A(sV)
        tdPrdO = tiled_mma_VdO.make_fragment_B(sdO)

        # Operands for dVi=P^T@dOi
        tdVrPt = tiled_mma_PtdOt.make_fragment_A(sPt)
        tdVrdOt = tiled_mma_PtdOt.make_fragment_B(sdOt)

        # Operands for dVi+=dS^T@Qvi
        tdVrdSt = tiled_mma_dStQvt.make_fragment_A(sdSt)
        tdVrQvt = tiled_mma_dStQvt.make_fragment_B(sQvt)

        # Operands for S^T = V@Qv^T (+ Kr@Qr^T): A is the same gathered V
        tSrQvB = tSrKr = tSrQr = gemm_S_rope = None
        if const_expr(self.recompute_P):
            tSrQvB = tiled_mma_VdO.make_fragment_B(sQvB)
            if const_expr(self.has_qk):
                tSrKr = tiled_mma_VdO.make_fragment_A(sKr)
                tSrQr = tiled_mma_VdO.make_fragment_B(sQr)
                gemm_S_rope = partial(fa_sm100_utils.gemm, tiled_mma_VdO, tdStS)

        use_ptx_gemm_VdO = False
        use_ptx_gemm_PtdOt = False
        use_ptx_gemm_dStQvt = False

        # GEMM functions
        if const_expr(use_ptx_gemm_VdO):
            gemm_VdO = partial(
                fa_sm100_utils.gemm_ptx_partial,
                tiled_mma_VdO.op,
                self.tmem_offset_dP,
                zero_init=True,
                cta_group=self.cta_group_size,
            )
        else:
            gemm_VdO = partial(
                fa_sm100_utils.gemm,
                tiled_mma_VdO,
                tdPtdP,
            )
        if const_expr(use_ptx_gemm_PtdOt):
            gemm_PtdOt = [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_PtdOt.op,
                    self.tmem_offsets_dV[split],
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                for split in range(self.num_hdimv_splits)
            ]
        else:
            gemm_PtdOt = [
                partial(
                    fa_sm100_utils.gemm,
                    tiled_mma_PtdOt,
                    tdVtdVs[split],
                    zero_init=True,
                )
                for split in range(self.num_hdimv_splits)
            ]
        if const_expr(use_ptx_gemm_dStQvt):
            gemm_dStQvt = [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_dStQvt.op,
                    self.tmem_offsets_dV[split],
                    zero_init=False,
                    cta_group=self.cta_group_size,
                )
                for split in range(self.num_hdimv_splits)
            ]
        else:
            gemm_dStQvt = [
                partial(
                    fa_sm100_utils.gemm,
                    tiled_mma_dStQvt,
                    tdVtdVs[split],
                    zero_init=False,
                )
                for split in range(self.num_hdimv_splits)
            ]

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        consumer_state_V = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        consumer_state_dO = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dO)
        consumer_state_Pt = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Pt)
        consumer_state_dOt_Qvt = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dOt)
        consumer_state_dSt = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dSt)
        producer_state_dPt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dPt)
        producer_state_dV = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dV)
        consumer_state_Kr = consumer_state_Qr = None
        if const_expr(self.recompute_P):
            producer_state_St = pipeline.make_pipeline_state(Producer, stages=self.num_stages_St)
            if const_expr(self.has_qk):
                consumer_state_Kr = pipeline.make_pipeline_state(
                    Consumer, stages=self.num_stages_Kr
                )
                consumer_state_Qr = pipeline.make_pipeline_state(
                    Consumer, stages=self.num_stages_Qr
                )

        mma_VdO = partial(
            self.mma_inner, gemm_VdO, pipeline_V, tdPrV, sV, tdPrdO, sdO, swap_AB_stage=True, use_ptx=use_ptx_gemm_VdO
        )
        mma_PtdOt = partial(
            self.mma_inner, gemm_PtdOt, pipeline_dOt_Qvt, tdVrPt, sPt, tdVrdOt, sdOt, use_ptx=use_ptx_gemm_PtdOt
        )
        mma_dStQvt = partial(
            self.mma_inner, gemm_dStQvt, pipeline_dOt_Qvt, tdVrdSt, sdSt, tdVrQvt, sQvt, use_ptx=use_ptx_gemm_dStQvt
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            # if const_expr(mCuSeqlensQ is not None):
            #     batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            # seqlen = SeqlenInfoCls(batch_idx)
            # All warps must agree on the group count. Every other warp (load,
            # gather, softmax, epilogue) uses the static topk_length, so use
            # it here too: the runtime index width must equal the compiled
            # value anyway (gather_kv_length is in the host-side compile key),
            # and a mismatch would otherwise desynchronize the pipelines into
            # a device hang instead of merely wrong results.
            num_n_block_groups = self.topk_length // self.cluster_tile_n

            if is_leader_cta:
                # ==== Prologue ====
                # (recompute_P: pipeline_dO carries the stationary QvB)
                consumer_wait_state_dO = consumer_state_dO.clone()
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_dO.consumer_wait(consumer_wait_state_dO)
                    consumer_wait_state_dO.advance()
                if const_expr(self.recompute_P and self.has_qk):
                    consumer_wait_state_Qr = consumer_state_Qr.clone()
                    pipeline_Qr.consumer_wait(consumer_wait_state_Qr)

                if const_expr(not self.recompute_P):
                    # ==== Mainloop ====
                    for _ in cutlass.range(num_n_block_groups, unroll=1):
                        # 1. dP = V @ dO^T
                        # mma inner waits for V
                        pipeline_dPt.producer_acquire(producer_state_dPt)
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            consumer_state_V = mma_VdO(
                                consumer_state_V, a_stage=split, zero_init=split == 0
                            )
                        pipeline_dPt.producer_commit(producer_state_dPt)
                        producer_state_dPt.advance()

                        # 2. dV = P^T @ dO
                        # mma inner waits for dOt
                        consumer_state_Pt, consumer_state_dOt_Qvt = self.mma_dV_leg1(
                            mma_PtdOt, pipeline_Pt, pipeline_dV,
                            consumer_state_Pt, consumer_state_dOt_Qvt, producer_state_dV,
                        )

                        # 3. dV += dS^T @ Qv
                        # mma inner waits for Qvt
                        consumer_state_dSt, consumer_state_dOt_Qvt, producer_state_dV = self.mma_dV_leg2(
                            mma_dStQvt, pipeline_dSt, pipeline_dV,
                            consumer_state_dSt, consumer_state_dOt_Qvt, producer_state_dV,
                        )
                else:
                    # ==== Mainloop (recompute_P) ====
                    # Per group g the order is S(g), dP(g), dV-leg1(g),
                    # dV-leg2(g). S(g) is issued at the TOP of the iteration so
                    # the softmax turnaround (exp2 + Pt stmatrix) overlaps the
                    # in-flight dP/dV gemms instead of delaying leg1's Pt wait.
                    # S consumes the V stages through a wait-only clone of the
                    # consumer state; the dP gemm performs the actual releases.
                    mma_S_next = partial(
                        self.mma_recompute_S,
                        tiled_mma_VdO, tdStS, tdPrV, tSrQvB,
                        gemm_S_rope, sKr, tSrKr, sQr, tSrQr,
                        pipeline_St, pipeline_V, pipeline_Kr,
                    )
                    mma_dP_group = partial(
                        self.mma_recompute_dP,
                        tiled_mma_VdO, tdPtdP, tdPrV, tdPrdO,
                        pipeline_dPt, pipeline_dOt_Qvt, pipeline_V,
                    )
                    mma_dV_leg1 = partial(
                        self.mma_dV_leg1, mma_PtdOt, pipeline_Pt, pipeline_dV
                    )
                    mma_dV_leg2 = partial(
                        self.mma_dV_leg2, mma_dStQvt, pipeline_dSt, pipeline_dV
                    )

                    for _ in cutlass.range(num_n_block_groups, unroll=1):
                        # S(g) first: consumes V(g) via a wait-only clone (the
                        # dP gemm below releases the stages), then the softmax
                        # turnaround overlaps dP(g); the dV legs are never
                        # delayed by the S gemm.
                        consumer_wait_state_V = consumer_state_V.clone()
                        producer_state_St, consumer_wait_state_V, consumer_state_Kr = mma_S_next(
                            producer_state_St, consumer_wait_state_V, consumer_state_Kr
                        )
                        producer_state_dPt, consumer_state_dOt_Qvt, consumer_state_V = mma_dP_group(
                            producer_state_dPt, consumer_state_dOt_Qvt, consumer_state_V
                        )
                        consumer_state_Pt, consumer_state_dOt_Qvt = mma_dV_leg1(
                            consumer_state_Pt, consumer_state_dOt_Qvt, producer_state_dV
                        )
                        consumer_state_dSt, consumer_state_dOt_Qvt, producer_state_dV = mma_dV_leg2(
                            consumer_state_dSt, consumer_state_dOt_Qvt, producer_state_dV
                        )

                # ==== Epilogue ====
                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()
                if const_expr(self.recompute_P and self.has_qk):
                    pipeline_Qr.consumer_release(consumer_state_Qr)
                    consumer_state_Qr.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        pipeline_dPt.producer_tail(producer_state_dPt)
        pipeline_dV.producer_tail(producer_state_dV)
        if const_expr(self.recompute_P):
            pipeline_St.producer_tail(producer_state_St)

    @cute.jit
    def mma_inner(
        self,
        gemm,
        load_pipeline,
        tCrA,
        sA,
        tCrB,
        sB,
        consumer_state: pipeline.PipelineState,
        acc_stage: Optional[Int32] = None,
        a_stage: Int32 = 0,
        zero_init: Optional[bool] = None,
        swap_AB_stage: bool = False,
        use_ptx: bool = True,
    ):
        if const_expr(acc_stage is not None):
            gemm = gemm[acc_stage]

        smem_stage = consumer_state.index

        if const_expr(not swap_AB_stage):
            a_stage = a_stage
            b_stage = smem_stage
        else:
            a_stage = smem_stage
            b_stage = a_stage

        tCrA_cur = tCrA[None, None, None, a_stage]
        sA_cur = sA[None, None, None, a_stage]
        tCrB_cur = tCrB[None, None, None, b_stage]
        sB_cur = sB[None, None, None, b_stage]

        kwargs = dict(tCrA=tCrA_cur, tCrB=tCrB_cur)
        if const_expr(use_ptx):
            kwargs |= dict(sA=sA_cur, sB=sB_cur)
        if const_expr(zero_init is not None):
            kwargs["zero_init"] = zero_init

        load_pipeline.consumer_wait(consumer_state)
        gemm(**kwargs)
        load_pipeline.consumer_release(consumer_state)
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def mma_recompute_S(
        self,
        tiled_mma_VdO: cute.TiledMma,
        tdStS: cute.Tensor,
        tdPrV: cute.Tensor,
        tSrQvB: cute.Tensor,
        gemm_S_rope,
        sKr: Optional[cute.Tensor],
        tSrKr: Optional[cute.Tensor],
        sQr: Optional[cute.Tensor],
        tSrQr: Optional[cute.Tensor],
        pipeline_St: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        pipeline_Kr: Optional[pipeline.PipelineAsync],
        producer_state_St: pipeline.PipelineState,
        consumer_wait_state_V: pipeline.PipelineState,
        consumer_state_Kr: Optional[pipeline.PipelineState],
    ):
        # S^T = V @ Qv^T (+ Kr @ Qr^T); QvB/Qr stationary. V is consumed via a
        # wait-only state (one group ahead) and released later by the dP gemm.
        pipeline_St.producer_acquire(producer_state_St)
        for split in cutlass.range_constexpr(self.num_hdimv_splits):
            pipeline_V.consumer_wait(consumer_wait_state_V)
            fa_sm100_utils.gemm(
                tiled_mma_VdO,
                tdStS,
                tCrA=tdPrV[None, None, None, consumer_wait_state_V.index],
                tCrB=tSrQvB[None, None, None, split],
                zero_init=split == 0,
            )
            consumer_wait_state_V.advance()
        if const_expr(self.has_qk):
            consumer_state_Kr = self.mma_inner(
                gemm_S_rope,
                pipeline_Kr,
                tSrKr,
                sKr,
                tSrQr,
                sQr,
                consumer_state_Kr,
                zero_init=False,
                use_ptx=False,
            )
        pipeline_St.producer_commit(producer_state_St)
        producer_state_St.advance()
        return producer_state_St, consumer_wait_state_V, consumer_state_Kr

    @cute.jit
    def mma_recompute_dP(
        self,
        tiled_mma_VdO: cute.TiledMma,
        tdPtdP: cute.Tensor,
        tdPrV: cute.Tensor,
        tdPrdO: cute.Tensor,
        pipeline_dPt: pipeline.PipelineAsync,
        pipeline_dOt_Qvt: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        producer_state_dPt: pipeline.PipelineState,
        consumer_state_dOt_Qvt: pipeline.PipelineState,
        consumer_state_V: pipeline.PipelineState,
    ):
        # dP = V @ dO^T; dO comes through the multiplex; V (already consumed
        # by the S gemm) is released here.
        pipeline_dPt.producer_acquire(producer_state_dPt)
        for split in cutlass.range_constexpr(self.num_hdimv_splits):
            pipeline_dOt_Qvt.consumer_wait(consumer_state_dOt_Qvt)
            pipeline_V.consumer_wait(consumer_state_V)
            fa_sm100_utils.gemm(
                tiled_mma_VdO,
                tdPtdP,
                tCrA=tdPrV[None, None, None, consumer_state_V.index],
                tCrB=tdPrdO[None, None, None, consumer_state_dOt_Qvt.index],
                zero_init=split == 0,
            )
            pipeline_dOt_Qvt.consumer_release(consumer_state_dOt_Qvt)
            consumer_state_dOt_Qvt.advance()
            pipeline_V.consumer_release(consumer_state_V)
            consumer_state_V.advance()
        pipeline_dPt.producer_commit(producer_state_dPt)
        producer_state_dPt.advance()
        return producer_state_dPt, consumer_state_dOt_Qvt, consumer_state_V

    @cute.jit
    def mma_dV_leg1(
        self,
        mma_PtdOt,
        pipeline_Pt: pipeline.PipelineAsync,
        pipeline_dV: pipeline.PipelineAsync,
        consumer_state_Pt: pipeline.PipelineState,
        consumer_state_dOt_Qvt: pipeline.PipelineState,
        producer_state_dV: pipeline.PipelineState,
    ):
        # dV = P^T @ dO (shared by the default and recompute mainloops)
        pipeline_Pt.consumer_wait(consumer_state_Pt)
        producer_acquire_state_dV = producer_state_dV.clone()
        for split in cutlass.range_constexpr(self.num_hdimv_splits):
            pipeline_dV.producer_acquire(producer_acquire_state_dV)
            producer_acquire_state_dV.advance()
            consumer_state_dOt_Qvt = mma_PtdOt(consumer_state_dOt_Qvt, acc_stage=split)
        pipeline_Pt.consumer_release(consumer_state_Pt)
        consumer_state_Pt.advance()
        return consumer_state_Pt, consumer_state_dOt_Qvt

    @cute.jit
    def mma_dV_leg2(
        self,
        mma_dStQvt,
        pipeline_dSt: pipeline.PipelineAsync,
        pipeline_dV: pipeline.PipelineAsync,
        consumer_state_dSt: pipeline.PipelineState,
        consumer_state_dOt_Qvt: pipeline.PipelineState,
        producer_state_dV: pipeline.PipelineState,
    ):
        # dV += dS^T @ Qv (shared by the default and recompute mainloops)
        pipeline_dSt.consumer_wait(consumer_state_dSt)
        for split in cutlass.range_constexpr(self.num_hdimv_splits):
            consumer_state_dOt_Qvt = mma_dStQvt(consumer_state_dOt_Qvt, acc_stage=split)
            pipeline_dV.producer_commit(producer_state_dV)
            producer_state_dV.advance()
        pipeline_dSt.consumer_release(consumer_state_dSt)
        consumer_state_dSt.advance()
        return consumer_state_dSt, consumer_state_dOt_Qvt, producer_state_dV

    @cute.jit
    def compute_loop(
        self,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        thr_mma_VdO: cute.ThrMma,
        tdPtdP: cute.Tensor,
        sP: cute.Tensor,
        sdS: cute.Tensor,
        sScaleP: cute.Tensor,
        sdPsum: cute.Tensor,
        mdS: cute.Tensor,
        tma_atom_dS: cute.CopyAtom,
        pipeline_P: pipeline.PipelineAsync,  # TmaAsync
        pipeline_Pt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_dPt: pipeline.PipelineAsync,  # UmmaAsync
        pipeline_dSt: pipeline.PipelineAsync,  # AsyncUmma
        pipeline_scaleP: pipeline.PipelineAsync,  # TmaAsync
        pipeline_dPsum: pipeline.PipelineAsync,  # TmaAsync
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
        tdStS: Optional[cute.Tensor] = None,
        sLse: Optional[cute.Tensor] = None,
        sBitmask: Optional[cute.Tensor] = None,
        pipeline_St: Optional[pipeline.PipelineAsync] = None,  # UmmaAsync
        pipeline_lse: Optional[pipeline.PipelineAsync] = None,  # TmaAsync
        pipeline_bitmask: Optional[pipeline.PipelineAsync] = None,  # Async
    ):
        tidx = cute.arch.thread_idx()[0] % self.num_softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_softmax_threads // 32
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        leader_warp = warp_idx == 0

        # 256b // 32 = 8 values, 128 mqa // 2 => tmem_rep = 8
        tmem_rep = self.tile_m // self.cta_group_size // 8
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(tmem_rep)),
            self.dtype_acc,
        )
        # ((64,(64,2)),1,1):((65536,(1,4194304 = 65536*64)),0,0)
        tdPtdP = tdPtdP[(None, None), 0, 0]
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tdPtdP)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N)
        # (((64,16),1),2,1):(((1,65536),0),1048576,0)>, 1048576/65536 = 16
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)

        cdP = cute.make_identity_tensor(self.mma_tiler_VdO[:2])  # (128, 128)
        tdPcdP = thr_mma_VdO.partition_C(cdP)[(None, None), 0, 0]  # (64,128):(1@0,1@1)
        # (((2,2,8),1),2,1):(((1@1,8@0,8@1),0),16@0,0)
        tdPcdP_t2r = thr_copy_t2r.partition_D(tdPcdP)
        assert tdPcdP_t2r.shape[0][1] == 1, f"unexpected tdPcdP_t2r shape, got {tdPcdP_t2r.shape}"

        tStS_t2r = None
        if const_expr(self.recompute_P):
            # S^T accumulator: identical layout to dP^T, same t2r tiling
            tdStS_local = tdStS[(None, None), 0, 0]
            tStS_t2r = thr_copy_t2r.partition_S(tdStS_local)

        smem_load_op = cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4)  # ldsm x num_matrices = ldsm x 4
        smem_store_op = cute.nvgpu.warp.StMatrix8x8x16bOp(True, 4)  # stsm x num_matrices = stsm x 4
        smem_load_atom = cute.make_copy_atom(smem_load_op, self.dtype)
        smem_store_atom = cute.make_copy_atom(smem_store_op, self.dtype)
        tiled_copy_r2s = cute.make_tiled_copy_D(smem_store_atom, tiled_copy_t2r)
        tiled_copy_s2r = cute.make_tiled_copy_D(smem_load_atom, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)

        sPt_load_layout = cute.make_ordered_layout(
            self.tile_Pt + (self.num_stages_P,), order=(1, 0, 2)
        )
        # (tile_n, tile_m, stages_P)
        sPt = cute.composition(sP, sPt_load_layout)
        sdSt = cute.composition(sdS, sPt_load_layout)

        # (R2S, R2S_M, R2S_N, PIPE_D)
        # ((8,4),2,1,1):((1,1024),16,0,0)
        tSR_sPt = thr_copy_s2r.partition_S(sPt)
        tRS_sdSt = thr_copy_r2s.partition_D(sdSt)

        # ((2,2),(2,8,1),stage):((0,0),(1,8,0),_)
        tPsScaleP_nm = None
        if const_expr(not self.recompute_P):
            tPsScaleP_nm = self.broadcast_tensor_nm_view(sScaleP, thr_mma_VdO, thr_copy_t2r)
        tdPsdPsum_nm = self.broadcast_tensor_nm_view(sdPsum, thr_mma_VdO, thr_copy_t2r)
        tdPsLse_nm = None
        if const_expr(self.recompute_P):
            tdPsLse_nm = self.broadcast_tensor_nm_view(sLse, thr_mma_VdO, thr_copy_t2r)

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer

        consumer_state_P = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_P)
        consumer_state_dPt = pipeline.make_pipeline_state(Consumer, stages=1)
        consumer_state_scaleP = pipeline.make_pipeline_state(
            Consumer, stages=self.num_stages_scaleP
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_dPsum)

        producer_state_Pt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Pt)
        producer_state_dSt = pipeline.make_pipeline_state(Producer, stages=self.num_stages_dSt)

        if const_expr(self.recompute_P):
            consumer_state_St = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_St)
            consumer_state_lse = pipeline.make_pipeline_state(
                Consumer, stages=self.num_stages_lse
            )
            if const_expr(not self.disable_bitmask):
                consumer_state_bitmask = pipeline.make_pipeline_state(
                    Consumer, stages=self.num_stages_bitmask
                )
            producer_state_Pt_empty = pipeline.make_pipeline_state(
                Producer, stages=self.num_stages_Pt
            )
            if const_expr(self.merged_dS):
                # P and dS share one smem buffer. The P stmatrix is gated by
                # pipeline_dSt (previous group's dSt consumed + its TMA store
                # drained), and the dS stmatrix is gated by pipeline_Pt (THIS
                # group's Pt consumed by the dV += P^T@dO mma). The latter is
                # an off-by-one-tighter coupling than the usual producer
                # pattern, so claim the pipeline's initial empty credit once
                # up front; every producer_acquire below then waits for the
                # mma release of the current group.
                pipeline_Pt.producer_acquire(producer_state_Pt_empty)
                producer_state_Pt_empty.advance()
            # unmerged: producer_state_Pt_empty follows the standard offset
            # (acquire before each P stmatrix waits the PREVIOUS group's Pt
            # consumption).

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(mCuSeqlensQ is not None):
                m_block -= seqlen.offset_q
            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            mdS_cur = seqlen.offset_batch_Q(mdS, batch_idx, dim=3)[None, None, head_idx]

            gdS = cute.local_tile(mdS_cur, (self.tile_m, self.tile_n), (m_block, None))
            store_dS, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dS,
                0,
                cute.make_layout(1),
                sdS,
                gdS,
            )

            pipeline_dPsum.consumer_wait(consumer_state_dPsum)

            tdPsdPsum_cur = tdPsdPsum_nm[0, None, consumer_state_dPsum.index]
            tdPrdPsum_cur_f32 = cute.make_rmem_tensor(tdPsdPsum_cur.shape, dtype=self.dtype_scale)
            cute.autovec_copy(tdPsdPsum_cur, tdPrdPsum_cur_f32)

            if const_expr(self.recompute_P):
                # lse (log2 units), per q-head, loaded once per work tile
                pipeline_lse.consumer_wait(consumer_state_lse)
                tdPsLse_cur = tdPsLse_nm[0, None, consumer_state_lse.index]
                tdPrLse_cur_f32 = cute.make_rmem_tensor(
                    tdPsLse_cur.shape, dtype=self.dtype_scale
                )
                cute.autovec_copy(tdPsLse_cur, tdPrLse_cur_f32)

            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                n_block = 2 * n_block_group + cta_rank_in_cluster

                # (((2,2,8),1),2,1):(((1,2,4),0),32,0)
                rPt = cute.make_rmem_tensor(tdPcdP_t2r.shape, self.dtype)
                # (S2R, S2R_M, S2R_N)
                rPt_copy_view = tiled_copy_s2r.retile(rPt)

                if const_expr(not self.recompute_P):
                    pipeline_P.consumer_wait(consumer_state_P)
                    # todo: ablate wait -> try_wait
                    pipeline_scaleP.consumer_wait(consumer_state_scaleP)

                    tSR_sPt_cur = tSR_sPt[None, None, None, consumer_state_P.index]
                    cute.copy(tiled_copy_s2r, tSR_sPt_cur, rPt_copy_view)

                    # ((2,2),(2,8,1)):((2,32),(1,4,0))
                    rP_nm = layout_utils.reshape_acc_to_mn(rPt[(None, 0), None, None])

                    tPsScaleP_cur = tPsScaleP_nm[0, None, consumer_state_scaleP.index]
                    tPrScaleP_cur_f32 = cute.make_rmem_tensor(
                        tPsScaleP_cur.shape, dtype=self.dtype_scale
                    )
                    tPrScaleP_cur = cute.make_rmem_tensor(tPsScaleP_cur.shape, dtype=self.dtype)
                    cute.autovec_copy(tPsScaleP_cur, tPrScaleP_cur_f32)
                    tPrScaleP_cur.store(tPrScaleP_cur_f32.load().to(self.dtype))

                    # scale P
                    for n in cutlass.range_constexpr(cute.size(rP_nm.shape[0])):
                        rP_cur = rP_nm[n, None]
                        rP_cur.store(rP_cur.load() * tPrScaleP_cur.load())
                    cute.arch.sync_warp()

                    cute.copy(tiled_copy_r2s, rPt_copy_view, tSR_sPt_cur)
                    cute.arch.fence_view_async_shared()
                    self.softmax_barrier.arrive_and_wait()

                    pipeline_scaleP.consumer_release(consumer_state_scaleP)
                    consumer_state_scaleP.advance()

                    pipeline_Pt.producer_commit(producer_state_Pt)
                    producer_state_Pt.advance()

                    # note: mma also signals Pt free, signal acquired in tma warp
                    pipeline_P.consumer_release(consumer_state_P)
                    consumer_state_P.advance()
                else:
                    tSR_sPt_cur = tSR_sPt[None, None, None, 0]

                    # ==== 1) S^T from TMEM ====
                    pipeline_St.consumer_wait(consumer_state_St)
                    rS_t2r = cute.make_rmem_tensor(tdPcdP_t2r.shape, self.dtype_acc)
                    cute.copy(tiled_copy_t2r, tStS_t2r, rS_t2r)
                    cute.arch.fence_view_async_tmem_load()
                    self.softmax_barrier.arrive_and_wait()
                    pipeline_St.consumer_release(consumer_state_St)
                    consumer_state_St.advance()

                    # ==== 2) exponent = softmax_scale_log2 * S - lse_log2[head] ====
                    rS_nm = layout_utils.reshape_acc_to_mn(rS_t2r[(None, 0), None, None])
                    for n in cutlass.range_constexpr(cute.size(rS_nm.shape[0])):
                        rS_cur = rS_nm[n, None]
                        rS_cur.store(
                            rS_cur.load() * softmax_scale_log2 - tdPrLse_cur_f32.load()
                        )

                    # ==== 3) mask invalid topk rows (sentinel / OOB) ====
                    # Overwriting the exponent handles garbage S from stale
                    # smem rows (incl. NaN/Inf) and the all-masked lse=-inf
                    # case (exponent forced to -inf => P = 0).
                    num_rows = cute.size(rS_nm.shape[0])
                    rRowValid = cute.make_rmem_tensor((num_rows,), Int32)
                    rRowValid.fill(Int32(1))
                    if const_expr(not self.disable_bitmask):
                        pipeline_bitmask.consumer_wait(consumer_state_bitmask)
                        rBitmask = cute.make_rmem_tensor(
                            (self.cluster_tile_n // 32,), dtype=Uint32
                        )
                        for w in cutlass.range_constexpr(cute.size(rBitmask)):
                            rBitmask[w] = sBitmask[w, consumer_state_bitmask.index]
                        self.softmax_barrier.arrive_and_wait()
                        pipeline_bitmask.consumer_release(consumer_state_bitmask)
                        consumer_state_bitmask.advance()

                        cS_nm = layout_utils.reshape_acc_to_mn(
                            tdPcdP_t2r[(None, 0), None, None]
                        )
                        for n in cutlass.range_constexpr(num_rows):
                            # topk row within the 128-wide cluster group
                            m_coord = cS_nm[n, 0][0]
                            valid = Boolean(False)
                            for w in cutlass.range_constexpr(cute.size(rBitmask)):
                                in_word = Boolean((m_coord >> 5) == w)
                                bit_set = Boolean(
                                    ((rBitmask[w] >> Uint32(m_coord % 32)) & 1) != 0
                                )
                                valid = valid | (in_word & bit_set)
                            rRowValid[n] = Int32(1) if valid else Int32(0)
                            rS_cur = rS_nm[n, None]
                            for i in cutlass.range_constexpr(cute.size(rS_cur)):
                                rS_cur[i] = rS_cur[i] if valid else -Float32.inf

                    # ==== 4) P^T = exp2(exponent) -> bf16 ====
                    for i in cutlass.range_constexpr(0, cute.size(rS_t2r), 2):
                        rS_t2r[i] = cute.math.exp2(rS_t2r[i], fastmath=True)
                        rS_t2r[i + 1] = cute.math.exp2(rS_t2r[i + 1], fastmath=True)
                    rPt.store(rS_t2r.load().to(self.dtype))

                    # ==== 5) stage P^T into the Pt operand buffer ====
                    if const_expr(self.merged_dS):
                        # Gate: previous group's dSt consumed by the mma AND
                        # its TMA store drained (P and dS share this buffer).
                        if leader_warp:
                            cute.arch.cp_async_bulk_wait_group(1 - self.num_stages_dSt, read=True)
                        self.softmax_barrier.arrive_and_wait()
                        pipeline_dSt.producer_acquire(producer_state_dSt)
                    else:
                        # Gate: previous group's Pt consumed by the mma.
                        pipeline_Pt.producer_acquire(producer_state_Pt_empty)
                        producer_state_Pt_empty.advance()

                    cute.copy(tiled_copy_r2s, rPt_copy_view, tSR_sPt_cur)
                    cute.arch.fence_view_async_shared()
                    self.softmax_barrier.arrive_and_wait()

                    pipeline_Pt.producer_commit(producer_state_Pt)
                    producer_state_Pt.advance()

                pipeline_dPt.consumer_wait(consumer_state_dPt)

                # (((2,2,8),1),2,1):(((1,2,4),0),32,0)
                tdPrdP_t2r = cute.make_rmem_tensor(tdPcdP_t2r.shape, self.dtype_acc)
                cute.copy(tiled_copy_t2r, tdPtdP_t2r, tdPrdP_t2r)
                cute.arch.fence_view_async_tmem_load()
                self.softmax_barrier.arrive_and_wait()

                pipeline_dPt.consumer_release(consumer_state_dPt)
                consumer_state_dPt.advance()

                # dS = P o (dP - dPsum)
                rdP_nm = layout_utils.reshape_acc_to_mn(tdPrdP_t2r[(None, 0), None, None])
                for n in cutlass.range_constexpr(cute.size(rdP_nm.shape[0])):
                    rdP_cur = rdP_nm[n, None]
                    rdP_cur.store(rdP_cur.load() - tdPrdPsum_cur_f32.load())

                if const_expr(self.recompute_P and not self.disable_bitmask):
                    # Harden dS = P * (dP - dPsum) against non-finite dP on
                    # masked rows (stale-smem V rows make dP garbage; P is 0
                    # there but 0 * NaN = NaN).
                    for n in cutlass.range_constexpr(cute.size(rdP_nm.shape[0])):
                        valid = Boolean(rRowValid[n] != 0)
                        rdP_cur = rdP_nm[n, None]
                        for i in cutlass.range_constexpr(cute.size(rdP_cur)):
                            rdP_cur[i] = rdP_cur[i] if valid else Float32(0.0)

                rPt.store(rPt.load() * (tdPrdP_t2r.load() * softmax_scale).to(self.dtype))

                if const_expr(not self.merged_dS):
                    # wait for tma store to free dSt buffer
                    if leader_warp:
                        cute.arch.cp_async_bulk_wait_group(1 - self.num_stages_dSt, read=True)
                    self.softmax_barrier.arrive_and_wait()

                    # note: dS guaranteed free as mma operand
                    pipeline_dSt.producer_acquire(producer_state_dSt)
                else:
                    # P and dS share the buffer: wait for THIS group's Pt to be
                    # consumed by the dV += P^T@dO mma before overwriting.
                    # (pipeline_dSt was already acquired before the P stmatrix;
                    # the TMA-store drain was waited there too.)
                    pipeline_Pt.producer_acquire(producer_state_Pt_empty)
                    producer_state_Pt_empty.advance()

                tRS_sdSt_cur = tRS_sdSt[None, None, None, producer_state_dSt.index]
                cute.copy(tiled_copy_r2s, rPt_copy_view, tRS_sdSt_cur)

                cute.arch.fence_view_async_shared()
                self.softmax_barrier.arrive_and_wait()
                pipeline_dSt.producer_commit(producer_state_dSt)

                # tma store
                if leader_warp:
                    store_dS(src_idx=producer_state_dSt.index, dst_idx=n_block)
                    cute.arch.cp_async_bulk_commit_group()

                producer_state_dSt.advance()

            pipeline_dPsum.consumer_release(consumer_state_dPsum)
            consumer_state_dPsum.advance()
            if const_expr(self.recompute_P):
                pipeline_lse.consumer_release(consumer_state_lse)
                consumer_state_lse.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        # producer tails

    @cute.jit
    def broadcast_tensor_nm_view(
        self,
        sX: cute.Tensor,  # (tile_m, num_stages)
        thr_mma: cute.ThrMma,
        thr_copy_t2r: cute.ThrCopy,
    ):
        assert cute.size(sX.shape[0]) == self.tile_m
        num_stages = sX.shape[1] if const_expr(cute.rank(sX) > 1) else 1
        sX_2D_cluster = cute.make_tensor(
            sX.iterator,
            cute.make_layout(
                (self.tile_m, self.cluster_tile_n, num_stages),
                stride=(1, 0, self.tile_m),
            ),
        )
        sXt_2D_cluster = layout_utils.transpose_view(sX_2D_cluster)
        sXt_2D = thr_mma.partition_C(sXt_2D_cluster)[(None, None), 0, 0, None]
        tXsXt_2D = thr_copy_t2r.partition_D(sXt_2D)[(None, 0), None, None, None]
        tXsXt_nm = layout_utils.make_acc_tensor_mn_view(tXsXt_2D)
        return tXsXt_nm

    @cute.jit
    def dVacc_store(
        self,
        mIndexTopk: cute.Tensor,
        mdV: cute.Tensor,
        sdV: cute.Tensor,
        tdVtdV0: cute.Tensor,
        tdVtdV1: cute.Tensor,
        thr_mma_PtdOt: cute.ThrMma,
        pipeline_dV: pipeline.PipelineAsync,  # UmmaAsync
        pipeline_dV_epi: pipeline.PipelineAsync,  # Async
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        mCuSeqlensQ: Optional[cute.Tensor],
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== dVaccum store warpgroup ====
        # produces: -
        # consumes: dV

        tdVtdV0 = tdVtdV0[(None, None), 0, 0]
        tdVtdV1 = tdVtdV1[(None, None), 0, 0]

        num_epi_warps = self.num_epilogue_threads // 32
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % self.num_epilogue_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % num_epi_warps
        leader_warp = warp_idx == 0
        wg_half = warp_idx // 2

        consumer_state_dV = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, stages=self.num_stages_dV
        )

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.dtype_acc,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tdVtdV0)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tdVtdV0_t2r = thr_copy_t2r.partition_S(tdVtdV0)
        tdVtdV1_t2r = thr_copy_t2r.partition_S(tdVtdV1)
        tdVtdVs_t2r = [tdVtdV0_t2r, tdVtdV1_t2r]

        cdVmma = cute.make_identity_tensor(self.mma_tiler_PtdOt[:2])
        tdVcdVmma = thr_mma_PtdOt.partition_C(cdVmma)[(None, None), 0, 0]
        tdVcdVmma_t2r = thr_copy_t2r.partition_D(tdVcdVmma)

        # 64 threads x 4 values to tile over tile_dV = (64, 32)
        tiled_copy_r2s = tiled_copy_2d(self.dtype_acc, 4, 64)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx % 64)

        # ((4,1),1,8,(1,8)):((1,0),0,4,(0,2048))
        tRS_sdV = thr_copy_r2s.partition_D(sdV)

        tiled_copy_s2r = copy_utils.tiled_copy_2d(self.dtype_acc, 8, self.num_epilogue_threads, 4)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        # (V, M, N, STAGE)
        tSR_sdV = thr_copy_s2r.partition_S(sdV)

        cdV = cute.make_identity_tensor(cute.product_each(sdV.shape[:2]))
        # (V, M, N)
        tdVcdV = thr_copy_s2r.partition_S(cdV)

        gmem_rows_per_thread = cute.size(tSR_sdV.shape[1])

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(mCuSeqlensQ is not None):
                batch_idx = get_batch_from_cu_tensor(m_block, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)
            num_n_block_groups = self.topk_length // self.cluster_tile_n
            # num_n_block_groups = topk_length_dynamic // self.cluster_tile_n

            # (seqlen_k, hdimv)
            mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]

            # (topk, dv)
            if const_expr(seqlen.has_cu_seqlens_q):
                # m_block means absolute m_idx
                mIndexTopk_cur = mIndexTopk[None, m_block]
            else:
                mIndexTopk_cur = mIndexTopk[None, m_block, batch_idx]

            # ==== Mainloop ====
            for n_block_group in cutlass.range(num_n_block_groups, unroll=1):
                n_block = 2 * n_block_group + cta_rank_in_cluster

                rIdxTopK = cute.make_rmem_tensor((gmem_rows_per_thread,), dtype=self.dtype_index)
                for j in cutlass.range_constexpr(gmem_rows_per_thread):
                    n_idx = n_block * self.tile_n + tdVcdV[0, j, 0][0]
                    rIdxTopK[j] = mIndexTopk_cur[n_idx]

                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    tdVtdV_t2r = tdVtdVs_t2r[split]

                    pipeline_dV.consumer_wait(consumer_state_dV)

                    # TODO: record meaning of hard-coded values
                    num_cols_per_store = self.tile_dV[1] * 2
                    num_epi_subtiles = (self.hdimv // self.num_hdimv_splits) // num_cols_per_store
                    assert num_cols_per_store == 64
                    assert num_epi_subtiles == 4
                    assert cute.size(tdVtdV_t2r.shape[2]) == num_epi_subtiles

                    tdVrdV_cur_shape = tdVcdVmma_t2r[None, None, 0].shape
                    tRS_rdV_cur_shape = tRS_sdV[None, None, None, 0].shape
                    assert cute.size(tdVrdV_cur_shape) == cute.size(tRS_rdV_cur_shape)

                    tdVrdV_out_shape = tSR_sdV[None, None, None, 0].shape + (2,)

                    for i in cutlass.range_constexpr(num_epi_subtiles):
                        tdVrdV_cur = cute.make_rmem_tensor(tdVrdV_cur_shape, self.dtype_acc)
                        cute.copy(tiled_copy_t2r, tdVtdV_t2r[None, None, i], tdVrdV_cur)

                        tRS_rdV_cur = cute.make_tensor(tdVrdV_cur.iterator, tRS_rdV_cur_shape)

                        stage = 4 * split + 2 * wg_half + (i % 2)
                        cute.copy(tiled_copy_r2s, tRS_rdV_cur, tRS_sdV[None, None, None, stage])
                        cute.arch.fence_view_async_shared()
                        self.epi_barrier.arrive_and_wait()

                        tSR_rdV = cute.make_rmem_tensor(tdVrdV_out_shape, dtype=self.dtype_acc)

                        for w in cutlass.range_constexpr(2):
                            stage_out = 4 * split + 2 * w + (i % 2)
                            cute.copy(
                                tiled_copy_s2r,
                                tSR_sdV[None, None, None, stage_out],
                                tSR_rdV[None, None, None, w],
                            )

                        for j in cutlass.range_constexpr(gmem_rows_per_thread):
                            gmem_n_idx = rIdxTopK[j]
                            # Skip -1 sentinel slots (invalid top-k entries)
                            # and slots at/beyond the K extent: the chunked
                            # backward shrinks the dv view to k_end, so
                            # without the upper bound these would be writes
                            # past the view's logical extent (their dV
                            # contribution is exactly zero, so skipping also
                            # saves the atomics).
                            if (gmem_n_idx >= 0) & (gmem_n_idx < seqlen.seqlen_k):
                                for w in cutlass.range_constexpr(2):
                                    dv_offset = (
                                        self.hdimv // self.num_hdimv_splits * split  # 256 * split
                                        + (self.hdimv // self.num_hdimv_splits // 2) * w  # 128 * w
                                        + 32 * i
                                    )
                                    dv_offset += tdVcdV[0, j, 0][1]
                                    gmem_coord = (gmem_n_idx, dv_offset)
                                    dV_gmem_ptr = elem_pointer(mdV_cur, gmem_coord)

                                    a = tSR_rdV[0, j, 0, w]
                                    b = tSR_rdV[1, j, 0, w]
                                    c = tSR_rdV[2, j, 0, w]
                                    d = tSR_rdV[3, j, 0, w]
                                    atomic_add_fp32x4(a, b, c, d, dV_gmem_ptr)

                    cute.arch.fence_view_async_tmem_load()
                    self.epi_barrier.arrive_and_wait()
                    pipeline_dV.consumer_release(consumer_state_dV)

                    if leader_warp:
                        with cute.arch.elect_one():
                            pipeline_dV_epi.consumer_release(consumer_state_dV)

                    consumer_state_dV.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
