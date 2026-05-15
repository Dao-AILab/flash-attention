# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
# Unsupported features that will be added later:
# - page size != 128
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import enum
import math
import os
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial
from dataclasses import dataclass
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, Float8E4M3FN, Float8E5M2, Float8E8M0FNU, Float4E2M1FN
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from flash_attn.cute.modified_utils.block_scaled_layout_test import make_smem_layout_sfa, make_smem_layout_sfb
import cutlass.utils.blockscaled_layout as blockscaled_utils
from flash_attn.cute.paged_kv import PagedKVManager
import flash_attn.cute.utils as utils
from flash_attn.cute import copy_utils
import flash_attn.cute.pipeline as pipeline
from flash_attn.cute.mask import AttentionMask
# from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from flash_attn.cute.pack_gqa import PackGQA
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.blackwell_helpers import packed_float_to_ue4m3, packed_float_to_e2m1
from flash_attn.cute.fast_math import FastDivmod
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)

# === TUNING KNOBS ===
# Keys: (is_causal: bool, head_dim_padded: int)
# FP4 kernel is always 1-CTA.
# Values:
#   ex2_emu_freq: int — exp2 emulation period; higher = more hardware MUFU.EX2,
#       fewer ALU-emulated exp2. 0 = all hardware. With ex2_emu_res=4, freq=N
#       means N-4 of every N positions use MUFU and 4 use polynomial emulation.
#   ex2_emu_start_frg: int — fragment index to start emulation from
#   num_regs_softmax: int — register count for softmax warps (multiple of 8)
#   num_regs_correction: int — register count for correction warps (multiple of 8)
#   num_regs_other: derived as 512 - num_regs_softmax * 2 - num_regs_correction
_FP4_TUNING_CONFIG = {
    # BF16 PV: ex2_emu_freq=16 verified via bench_fp4.py (1921 TF peak).
    # Only applied for NVFP4 (sf_vec_size=16); MXFP8 BF16 uses freq=0.
    (False, 128): {"ex2_emu_freq": 16, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 80},
    (True, 128):  {"ex2_emu_freq": 16, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 80},
}
# FP8 PV overrides: when v_dtype.width == 8 and quant_pv == False
_NVFP4_FP8PV_TUNING_CONFIG = {
    # Verified via bench_fp4.py (2018 TF peak).
    (False, 128): {"ex2_emu_freq": 9, "ex2_emu_start_frg": 0},
    (True, 128):  {"ex2_emu_freq": 9, "ex2_emu_start_frg": 0},
    (False, 64):  {"ex2_emu_freq": 16, "ex2_emu_start_frg": 0},
    (True, 64):   {"ex2_emu_freq": 16, "ex2_emu_start_frg": 0},
}
_MXFP8_FP8PV_TUNING_CONFIG = {
    # Verified via bench_fp4.py (1948 TF peak).
    (False, 128): {"ex2_emu_freq": 10, "ex2_emu_start_frg": 0},
    (True, 128):  {"ex2_emu_freq": 10, "ex2_emu_start_frg": 0},
    (False, 64):  {"ex2_emu_freq": 10, "ex2_emu_start_frg": 0},
    (True, 64):   {"ex2_emu_freq": 10, "ex2_emu_start_frg": 0},
}
# === END TUNING KNOBS ===


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


class FlashAttentionForwardSm100:
    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        sf_dtype: Optional[Type[cutlass.Numeric]] = None,
        sf_vec_size: Optional[int] = None,
    ):
        assert (sf_dtype, sf_vec_size) in {
            (cutlass.Float8E4M3FN, 16),
            (cutlass.Float8E8M0FNU, 32),
        }, f"Unsupported block-scaled configuration: sf_dtype={sf_dtype}, sf_vec_size={sf_vec_size}"
        self.use_tma_KV = not paged_kv_non_tma
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = 2 if not is_split_kv else 1
        assert self.q_stage in [1, 2]
        # 2 Q tile per CTA
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, (
                "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
            )
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 2
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or
            (self.head_dim_v_padded >= 128 and self.is_split_kv)
        )
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        self.softmax0_warp_ids = (0, 1, 2, 3) # stage 0
        self.softmax1_warp_ids = (4, 5, 6, 7) # stage 1
        # self.correction_warp_ids = (8, 9)
        self.correction_warp_ids = (8, 9, 10, 11)
        # self.mma_warp_id = 10
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15, )
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()
        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q: # fallback
            self.epilogue_warp_ids = (13, 14)

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        # self.tmem_o_offset = self.tmem_s_offset
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_s_to_p_offset = self.n_block_size // 2

        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # e.g., 64, 192

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        _tune_key = (self.is_causal, self.head_dim_padded)
        self._tune = _FP4_TUNING_CONFIG.get(_tune_key, {})
        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        elif "num_regs_softmax" in self._tune:
            self.num_regs_softmax = self._tune["num_regs_softmax"]
            self.num_regs_correction = self._tune["num_regs_correction"]
            self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        else:
            self.num_regs_softmax = 192
            self.num_regs_correction = 80
            self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        self.num_regs_empty = 24
        self.buffer_align_bytes = 1024
        
        # Scale factor parameters for block-scaled quantization (FP4)
        self.sf_dtype = sf_dtype
        self.sf_vec_size = sf_vec_size
        self.debug_skip_sfq_s2t = os.getenv("FA4_DEBUG_SKIP_SFQ_S2T", "0") == "1"
        # FA4_SFQK_TMEM_SLOT: "s" (default staggered-S), "o" (o-offsets),
        # "o_stagger" (staggered-O). Experiment knob for SFQK TMEM placement.
        self.sfqk_tmem_slot = os.getenv("FA4_SFQK_TMEM_SLOT", "s")
        self.debug_skip_sfk_s2t = os.getenv("FA4_DEBUG_SKIP_SFK_S2T", "0") == "1"
        self.debug_force_generic_mxfp8_qk = os.getenv("FA4_DEBUG_FORCE_GENERIC_MXFP8_QK", "0") == "1"
        # Force MXFP8 QK through the inline-PTX helper for perf A/B. Measured
        # after the tmem_s_offset SFQ collision fix:
        #   MXFP8+BF16 (1,32768,24,128): inline 1718 TF, generic 1670 TF (+2.9%)
        #   MXFP8+BF16 (1,32768,12,128): inline 1643 TF, generic 1650 TF (tie)
        # Mixed results; keep default generic, toggleable for per-shape tuning.
        self.debug_mxfp8_use_inline_ptx = os.getenv("FA4_MXFP8_USE_INLINE_PTX", "0") == "1"
        self.fp8_pv_use_explicit_pack = os.getenv("FA4_FP8_PV_USE_EXPLICIT_PACK", "1") == "1"
        # exp2 emulation: NVFP4 BF16 uses freq from tuning config, MXFP8 BF16 disables (freq=0)
        _ex2_freq = self._tune.get("ex2_emu_freq", 0)
        self.ex2_emu_freq = _ex2_freq if self.sf_vec_size == 16 else 0
        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 0)
        self.fp8_pv_use_fused_pack = os.getenv("FA4_FP8_PV_USE_FUSED_PACK", "0") == "1"
        self.fp8_pv_zero_fill_regs = True
        self.mma_inst_bits_k = 256
        if self.sf_vec_size == 16:
            # NVFP4 / MXF4NVF4: 256-bit operand tile covers 64 logical K elements.
            self.mma_inst_tile_k = self.head_dim_padded // (self.mma_inst_bits_k // 8 * 2)
        elif self.sf_vec_size == 32:
            # MXFP8: 256-bit operand tile covers 32 logical K elements.
            self.mma_inst_tile_k = self.head_dim_padded // (self.mma_inst_bits_k // 8)
        else:
            raise ValueError(f"Unsupported sf_vec_size={self.sf_vec_size}")

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """
        self.acc_stage = 1
        self.epi_stage = 2
        # Compute kv_stage from SMEM budget.
        # Blackwell: 228KB per SM, 227KB optin per block.
        # K and V alias when same dtype or when K is smaller (FP4 K in BF16 V).
        smem_budget = 227 * 1024
        align = self.buffer_align_bytes  # 128B struct field alignment
        def align_up(x, a): return (x + a - 1) // a * a
        # Fixed fields (not scaled by kv_stage): mbar, tmem_holding_buf, sScale, sO, sQ, SFQ, SFP
        # mbar_total depends on kv_stage but is small (~40 barriers * 8B = 320B); use upper bound
        smem_mbar = 512  # generous upper bound for mbarrier storage
        smem_tmem = 4  # Int32
        smem_sScale = align_up(self.q_stage * self.m_block_size * 2 * 4, align)  # Float32
        smem_q_per_stage = self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_o_per_stage = self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        smem_sO = align_up(smem_o_per_stage * self.epi_stage, align) if not self.overlap_sO_sQ else 0
        smem_sQ = align_up(smem_q_per_stage * self.q_stage, align)
        # SFQ/SFP are per q_stage, SF layout cosize: m_block * head_dim / sf_vec_size
        sfq_per_stage = self.m_block_size * self.head_dim_padded // self.sf_vec_size
        sfp_per_stage = self.m_block_size * self.head_dim_v_padded // self.sf_vec_size
        smem_sSFQ = align_up(sfq_per_stage * self.q_stage, align) if self.quant_qk else 0
        smem_sSFP = align_up(sfp_per_stage * self.q_stage, align) if self.quant_pv else 0
        smem_fixed = smem_mbar + smem_tmem + smem_sScale + smem_sO + smem_sQ + smem_sSFQ + smem_sSFP
        # Per-kv_stage fields: sK (or aliased), sV, SFK, SFV
        smem_k_per_stage = self.m_block_size * self.head_dim_padded * self.k_dtype.width // 8
        smem_v_per_stage = self.m_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        if self.v_dtype == self.k_dtype or self.k_dtype.width < self.v_dtype.width:
            smem_kv_per_stage = max(smem_k_per_stage, smem_v_per_stage)
        else:
            smem_kv_per_stage = smem_k_per_stage + smem_v_per_stage
        # SF layout cosize per stage: n_block * head_dim / sf_vec_size (MMA-tiled)
        sfk_per_stage = self.n_block_size * self.head_dim_padded // self.sf_vec_size
        sfv_per_stage = self.n_block_size * self.head_dim_v_padded // self.sf_vec_size
        if self.quant_qk:
            smem_kv_per_stage += sfk_per_stage
        if self.quant_pv:
            smem_kv_per_stage += sfv_per_stage
        # Add per-stage padding for swizzle/layout cosize inflation (~128B per staged field)
        num_kv_staged_fields = 2  # sK + sV (or 1 if aliased, but sV still has cosize overhead)
        if self.quant_qk:
            num_kv_staged_fields += 1  # sSFK
        if self.quant_pv:
            num_kv_staged_fields += 1  # sSFV
        smem_kv_per_stage += num_kv_staged_fields * 128
        self.kv_stage = (smem_budget - smem_fixed) // smem_kv_per_stage
        # Pure FP8 PV: cap KV pipeline depth to 4 for hdim >= 128.
        fp8_pv_kv_stage_cap = 4 if self.head_dim_v_padded >= 128 else 0
        if const_expr(not self.quant_pv and self.v_dtype.width == 8 and fp8_pv_kv_stage_cap > 0):
            self.kv_stage = min(self.kv_stage, fp8_pv_kv_stage_cap)
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        # self.uneven_kv_smem = (
            # self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        # )
        self.uneven_kv_smem = False
        self.uneven_kv_smem_offset = (
            self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ,  # cute.Tensor or cute.Pointer (b, s_q, h, d)
        mK,  # cute.Tensor or cute.Pointer (b_k, s_k, h_k, d)
        mV,  # cute.Tensor or cute.Pointer (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        # For pointer-based Q/K: separate shapes to handle cross-attention (seqlen_q != seqlen_k)
        q_ptr_shape: tuple = (),
        k_ptr_shape: tuple = (),
        # For pointer-based V (FP4 K-major): full headdim shape (b, s, h, d)
        v_ptr_shape: tuple = (),
        compute_sp1: cutlass.Constexpr[bool] = False,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        For FP4, mQ/mK can be cute.Pointer with q/k_ptr_shape providing (b, s, h, d).
        The kernel builds tensors from the pointer using make_ordered_layout.
        """
        # Build Q/K tensors from pointer/tensor + shape
        # For pointers: mQ is a Pointer, .iterator not needed
        # For tensors: mQ is a Tensor, use .iterator to extract pointer
        q_iter = mQ.iterator if hasattr(mQ, 'iterator') else mQ
        k_iter = mK.iterator if hasattr(mK, 'iterator') else mK
        mQ = cute.make_tensor(q_iter, cute.make_ordered_layout(
            q_ptr_shape, order=tuple(range(len(q_ptr_shape) - 1, -1, -1))
        ))
        mK = cute.make_tensor(k_iter, cute.make_ordered_layout(
            k_ptr_shape, order=tuple(range(len(k_ptr_shape) - 1, -1, -1))
        ))
        # FP4 K-major V: build from pointer with explicit (b, s, h, d) shape and
        # K-major strides (S*H*D, 1, S, S*H). The host transposes V's underlying
        # buffer so that seqlen has stride 1 in the FP4 byte buffer.
        if const_expr(len(v_ptr_shape) > 0):
            v_iter = mV.iterator if hasattr(mV, 'iterator') else mV
            # K-major V: nvfp4_quantize on `v.permute(0,2,3,1).reshape(b*h*d, s)`
            # produces an FP4 byte buffer of physical shape (b, h, d, s/2) row-major,
            # where each int8 byte holds two seqlen-adjacent FP4 in the high/low
            # nibble. Logically the V tensor has FP4 shape (b, s, h, d) with element
            # strides (h*d*s, 1, h*d, d) — order=(3, 0, 2, 1):
            #   s (dim 1) order 0 → stride 1
            #   d (dim 3) order 1 → stride s
            #   h (dim 2) order 2 → stride s*d
            #   b (dim 0) order 3 → stride s*d*h
            # Int64 shape so make_ordered_layout produces Int64 strides
            # (tma_partition for SFV requires Int64).
            from cutlass import Int64
            v_b, v_s, v_h, v_d = v_ptr_shape
            mV = cute.make_tensor(v_iter, cute.make_ordered_layout(
                (Int64(v_b), Int64(v_s), Int64(v_h), Int64(v_d)),
                order=(3, 0, 2, 1),
            ))
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        self.compute_sp1 = const_expr(compute_sp1)
        if const_expr(mSFQ is None):
            assert self.q_dtype.width >= 8
            assert const_expr(mSFK is None), "Must provide both QK sfs or None"
        self.quant_qk = const_expr(mSFQ is not None)
        self.quant_pv = const_expr(mSFV is not None)
        assert not (not self.quant_qk and self.quant_pv)

        # Assume all strides are divisible by 128 bits except the last stride
        def _assume_strides(t):
            divby = 128 // t.element_type.width
            return tuple(
                s if isinstance(s, int) else cute.assume(s, divby=divby)
                for s in t.stride[:-1]
            ) + (t.stride[-1],)
        mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=_assume_strides(t)))
            for t in (mV, mO)
        ]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)) # (s_q, d, h, b)
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        # For FP4 block-scaled MMA, B (V) must be K-major (K=seqlen contiguous).
        # Skip V transpose when V is already K-major (headdim contiguous = N contiguous after transpose).
        # Without transpose: mV = (s, d, h, b) → mode 0=s(K), mode 1=d(N) with d contiguous → K-major: NO!
        # With transpose:    mV = (d, s, h, b) → mode 0=d(N), mode 1=s(K) with d contiguous → MN-major
        # For FP4 we need K-major: mode 1 (K=seqlen) contiguous. Need V physically transposed on host.
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        # FP4 block-scaled MMA requires K-major B; standard MMA uses MN-major
        if const_expr(mSFV is not None):
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mV must be K-major for FP4 block-scaled MMA")
        else:
            if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
                raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype == cutlass.Int8):
            assert self.q_dtype == self.k_dtype
            self.q_dtype = self.k_dtype = cutlass.Float4E2M1FN
            if const_expr(mSFV is not None):
                self.v_dtype = cutlass.Float4E2M1FN

        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(mSFV is not None and self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype} (V quantization requires matching dtype)")
        self._setup_attributes()
        self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None and mSeqUsedQ is None
        # Apply FP8 PV tuning overrides when v_dtype is FP8
        if const_expr(not self.quant_pv and self.v_dtype.width == 8):
            _fp8_cfg = _NVFP4_FP8PV_TUNING_CONFIG if self.sf_vec_size == 16 else _MXFP8_FP8PV_TUNING_CONFIG
            _fp8_tune = _fp8_cfg.get((self.is_causal, self.head_dim_padded), {})
            if const_expr("ex2_emu_freq" in _fp8_tune):
                self.ex2_emu_freq = _fp8_tune["ex2_emu_freq"]
                self.ex2_emu_start_frg = _fp8_tune.get("ex2_emu_start_frg", self.ex2_emu_start_frg)

        use_2cta_instrs = self.mma_tiler_qk[0] == 256
        assert use_2cta_instrs == False, "Two-CTA instructions not supported yet"
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        
        # Use block-scaled MMA for PV only if V is being quantized (mSFV is provided)
        if const_expr(self.quant_qk):
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.qk_acc_dtype,
                self.cta_group,
                self.mma_tiler_qk[:2],
            )

        if const_expr(self.quant_pv):
            tiled_mma_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )
        else:
            tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.pv_acc_dtype,
                self.cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.mma_tiler_pv[:2]
        
        # ((Atom_Inst_M, Atom_Inst_K), MMA_M, MMA_K, STAGE)
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )
        
        sfv_smem_layout_staged = None
        sfp_smem_layout_staged = None
        # # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
        # Always thread our computed mma_inst_tile_k through. Default is 4 in the
        # helper, which only happens to match MXFP8 d=128. MXFP8 d=64 needs 2 and
        # was failing to compile via TMA SFQ atom shape mismatch.
        sf_layout_kwargs = {"mma_tile_inst_k": self.mma_inst_tile_k}
        sfq_smem_layout_staged = make_smem_layout_sfa(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.sf_vec_size,
            self.q_stage,
            **sf_layout_kwargs,
        )
        sfk_smem_layout_staged = make_smem_layout_sfb(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.sf_vec_size,
            self.kv_stage,
            **sf_layout_kwargs,
        )
        # Create P scale factor layout for P*V operation (P is the A matrix)
        if const_expr(self.quant_pv):
            sfp_smem_layout_staged = make_smem_layout_sfa(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                self.q_stage,
                **sf_layout_kwargs,
            )

            sfv_smem_layout_staged = make_smem_layout_sfb(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                self.kv_stage,
                **sf_layout_kwargs,
            )
        
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )


        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (self.qhead_per_kvhead, mQ.shape[0]), # (qhead_per_kvhead, sq)
                mQ.shape[1], # d
                mK.shape[2], # h_k
                *mQ.shape[3:], # b
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed)
                )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        # Add scale factor copy bytes to Q/K/V since they use the same barrier
        if const_expr(self.quant_qk):
            self.tma_copy_bytes["Q"] += cute.size_in_bytes(mSFQ.element_type, cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]))
            self.tma_copy_bytes["K"] += cute.size_in_bytes(mSFK.element_type, cute.select(sfk_smem_layout_staged, mode=[0, 1, 2]))
        if const_expr(self.quant_pv):
            self.tma_copy_bytes["V"] += cute.size_in_bytes(mSFV.element_type, cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]))

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        mQ_shape = mQ.shape
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        tma_atom_K = None
        tma_atom_V = None
        mK_shape = mK.shape
        mV_shape = mV.shape
        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                self.cluster_layout_vmnk.shape,
            )

        # TMA load for scale factors
        tma_atom_sfq = None
        tma_tensor_sfq = None
        tma_atom_sfk = None
        tma_tensor_sfk = None
        tma_atom_sfv = None
        tma_tensor_sfv = None
        # Setup TMA load for SFQ
        if const_expr(self.quant_qk):
            sfq_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, mQ_shape, (2, 1, 3, 4))
            sfq_op = sm100_utils_basic.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma_qk.thr_id
            )
            mSFQ = cute.make_tensor(mSFQ.iterator, sfq_layout)
            tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_A(
                sfq_op,
                mSFQ,
                cute.select(sfq_smem_layout_staged, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # Setup TMA load for SFK (scale factor for K, like SFB)
            sfk_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma_qk.thr_id
            )

        if const_expr(self.quant_qk):
            sfk_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, mK_shape, (2, 1, 3, 4))
            mSFK = cute.make_tensor(mSFK.iterator, sfk_layout)

            # For SFB, compute mma_inst_shape_mnk_sfb: (M // (2 if use_2cta_instrs else 1), round_up(N, 128), K)
            mma_inst_shape_mnk_qk = (
                self.mma_tiler_qk[0],
                self.mma_tiler_qk[1],
                self.mma_inst_bits_k // self.k_dtype.width,
            )

            mma_inst_shape_mnk_sfb_qk = (
                mma_inst_shape_mnk_qk[0] // (2 if use_2cta_instrs else 1),
                cute.round_up(mma_inst_shape_mnk_qk[1], 128),
                mma_inst_shape_mnk_qk[2],
            )
        
            mma_tiler_sfb_qk = (
                mma_inst_shape_mnk_sfb_qk[0],
                mma_inst_shape_mnk_sfb_qk[1],
                mma_inst_shape_mnk_sfb_qk[2] * self.mma_inst_tile_k,
            )
            # For SFB, we need a separate tiled_mma_sfb with CtaGroup.ONE
            tiled_mma_sfb_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.k_dtype,
                self.k_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                mma_inst_shape_mnk_sfb_qk[:2],
            )
            cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_sfb_qk.thr_id.shape,),
            )
            tma_atom_sfk, tma_tensor_sfk = cute.nvgpu.make_tiled_tma_atom_B(
                sfk_op,
                mSFK,
                cute.select(sfk_smem_layout_staged, mode=[0, 1, 2]),
                mma_tiler_sfb_qk,
                tiled_mma_sfb_qk,
                cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
    
        if const_expr(self.quant_pv):
            # Setup TMA load for SFV (scale factor for V, like SFB)
            sfv_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma_pv.thr_id
            )
            # Setup scale factor tensor layout
            sfv_layout = cute.tile_to_shape(blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size).layout, mV_shape, (2, 1, 3, 4))
            mSFV = cute.make_tensor(mSFV.iterator, sfv_layout)
            # For SFB, compute mma_inst_shape_mnk_sfb: (M // (2 if use_2cta_instrs else 1), round_up(N, 128), K)
            mma_inst_shape_mnk_pv = ( # the same processed by one tcgen05.mma instruction
                self.mma_tiler_pv[0],
                self.mma_tiler_pv[1],
                self.mma_inst_bits_k // self.v_dtype.width,
            )
            use_2cta_instrs = self.mma_tiler_pv[0] == 256
            mma_inst_shape_mnk_sfb_pv = (
                mma_inst_shape_mnk_pv[0] // (2 if use_2cta_instrs else 1),
                cute.round_up(mma_inst_shape_mnk_pv[1], 128),
                mma_inst_shape_mnk_pv[2],
            )
            mma_tiler_sfb_pv = (
                mma_inst_shape_mnk_sfb_pv[0],
                mma_inst_shape_mnk_sfb_pv[1],
                mma_inst_shape_mnk_sfb_pv[2] * self.mma_inst_tile_k,
            )
            # For SFB, we need a separate tiled_mma_sfb with CtaGroup.ONE
            tiled_mma_sfb_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
                mma_inst_shape_mnk_sfb_pv[:2],
                p_source,
            )
            cluster_layout_sfb_pv_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_sfb_pv.thr_id.shape,),
            )
            tma_atom_sfv, tma_tensor_sfv = cute.nvgpu.make_tiled_tma_atom_B(
                sfv_op,
                mSFV,
                cute.select(sfv_smem_layout_staged, mode=[0, 1, 2]),
                mma_tiler_sfb_pv,
                tiled_mma_sfb_pv,
                cluster_layout_sfb_pv_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            # For sub-byte dtypes (FP4), width=4 rounds to 0 under integer division;
            # clamp so size_one_head * element_size in the scheduler's L2 swizzle calc
            # doesn't become 0 and divide by zero in the else branch of the ifexp.
            element_size=max(self.k_dtype.width // 8, 1),
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + self.q_stage
        self.mbar_O_full_offset = self.mbar_S_full_offset + self.q_stage
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + self.q_stage
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + self.q_stage
        self.mbar_corr_epi_full_offset = self.mbar_softmax_corr_empty_offset + self.epi_stage
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.epi_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + self.q_stage
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        # QK and PV SF tmem load wait for softmax t2r store
        self.mbar_sfqk_load_offset = self.mbar_P_full_2_offset + self.q_stage 
        self.mbar_sfpv_load_offset = self.mbar_sfqk_load_offset + self.q_stage
        self.mbar_total = self.mbar_sfpv_load_offset + self.q_stage
        # self.mbar_total = self.mbar_P_full_2_offset + self.q_stage
        # Pure FP8 PV benefits from releasing P to the PV MMA earlier than the
        # FP4 quantized path; keep env overrides for additional tuning sweeps.
        fp8_pv_split_num = int(os.getenv("FA4_FP8_PV_P_SPLIT_NUM", "1"))
        fp8_pv_split_den = int(os.getenv("FA4_FP8_PV_P_SPLIT_DEN", "2"))
        fp8_pv_small_d = self.head_dim_v_padded <= 64
        self.mbar_p_split = lambda k: (
            max(
                1,
                min(
                    k - 1,
                    k
                    * (3 if cutlass.const_expr(fp8_pv_small_d) else fp8_pv_split_num)
                    // (4 if cutlass.const_expr(fp8_pv_small_d) else fp8_pv_split_den),
                ),
            )
            if cutlass.const_expr(self.v_dtype.width == 8 and k > 1) else
            ((k // 4 * 3) if cutlass.const_expr(self.v_dtype.width > 8) else k // 2)
        )
        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 1
        sQ_size = (
            cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ) else
            cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width)
        )
        
        # Calculate scale factor shared memory sizes
        # Use size 1 as minimum to avoid alignment issues when size is 0
        sfq_smem_size = cute.cosize(sfq_smem_layout_staged) if const_expr(self.quant_qk) else 1
        sfk_smem_size = cute.cosize(sfk_smem_layout_staged) if const_expr(self.quant_qk) else 1
        sfp_smem_size = cute.cosize(sfp_smem_layout_staged) if const_expr(self.quant_pv) else 1
        sfv_smem_size = cute.cosize(sfv_smem_layout_staged) if const_expr(self.quant_pv) else 1

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2], self.buffer_align_bytes]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            # K reuses V's buffer when K is smaller (FP4 K in BF16 V), or same dtype
            k_aliases_v = self.k_dtype.width < self.v_dtype.width
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, 1] if const_expr(k_aliases_v) else cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                1 if const_expr(k_aliases_v) else self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)] if not const_expr(self.v_dtype == self.k_dtype) else cute.struct.MemRange[self.k_dtype, 1],
                self.buffer_align_bytes if not const_expr(self.v_dtype == self.k_dtype) else 1,
            ]
            # Scale factor shared memory (if block-scaled quantization is used)
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfq_smem_size],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfk_smem_size],
                self.buffer_align_bytes,
            ]
            sSFP: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfp_smem_size],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfv_smem_size],
                self.buffer_align_bytes,
            ]
        
        # Remove scale factors to avoid OOM. Seems I can't set their size to 0
        @cute.struct
        class SharedStorageBF16:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.Align[cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2], self.buffer_align_bytes]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage if const_expr(self.quant_qk) or const_expr(self.quant_pv) else SharedStorageBF16
        
        # Verify shared memory fits within budget
        total_smem_bytes = self.shared_storage.size_in_bytes()
        assert total_smem_bytes <= 227 * 1024, (
            f"SharedStorage {total_smem_bytes // 1024}KB exceeds 227KB limit. "
            f"Reduce kv_stage (currently {self.kv_stage})."
        )
        print(f"Total shared memory used: {total_smem_bytes / 1024:.2f} KB")
        sO_bytes = cute.size_in_bytes(self.o_dtype, sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_bytes = cute.size_in_bytes(self.q_dtype, sQ_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, sK_layout)
        sV_bytes = cute.size_in_bytes(self.v_dtype, sV_layout)
        sfq_bytes = cute.size_in_bytes(cutlass.Uint8, sfq_smem_layout_staged)
        sfk_bytes = cute.size_in_bytes(cutlass.Uint8, sfk_smem_layout_staged)
        sfp_bytes = cute.size_in_bytes(cutlass.Uint8, sfp_smem_layout_staged) if const_expr(sfp_smem_layout_staged is not None) else 0
        sfv_bytes = cute.size_in_bytes(cutlass.Uint8, sfv_smem_layout_staged) if const_expr(sfv_smem_layout_staged is not None) else 0
        print(f"sO_size: {sO_bytes / 1024:.2f} KB")
        print(f"sQ_size: {sQ_bytes / 1024:.2f} KB")
        print(f"sK_size: {sK_bytes / 1024:.2f} KB")
        if const_expr(self.v_dtype != self.k_dtype):
            print(f"sV_size: {sV_bytes / 1024:.2f} KB")
        print(f"sfq_smem_size: {sfq_bytes / 1024:.2f} KB")
        print(f"sfk_smem_size: {sfk_bytes / 1024:.2f} KB")
        print(f"sfp_smem_size: {sfp_bytes / 1024:.2f} KB")
        print(f"sfv_smem_size: {sfv_bytes / 1024:.2f} KB")
        
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softmax_scale = None
        else:
            # NB: If a users passes in a score mod, we want to apply the score-mod in the sm_scaled qk
            # But in the original base 10. We hijack softmax_scale_log2 to just be the change of base
            # and correctly apply the softmax_scale prior to score_mod in the softmax step
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmod.create(seqlen_q)
            seqlen_k_divmod = FastDivmod.create(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            tma_atom_sfq,
            tma_tensor_sfq,
            tma_atom_sfk,
            tma_tensor_sfk,
            tma_atom_sfv,
            tma_tensor_sfv,
            sfq_smem_layout_staged,
            sfk_smem_layout_staged,
            sfp_smem_layout_staged,
            sfv_smem_layout_staged,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        tma_atom_sfq: Optional[cute.CopyAtom] = None,
        tma_tensor_sfq: Optional[cute.Tensor] = None,
        tma_atom_sfk: Optional[cute.CopyAtom] = None,
        tma_tensor_sfk: Optional[cute.Tensor] = None,
        tma_atom_sfv: Optional[cute.CopyAtom] = None,
        tma_tensor_sfv: Optional[cute.Tensor] = None,
        sfq_smem_layout_staged: Optional[cute.Layout] = None,
        sfk_smem_layout_staged: Optional[cute.Layout] = None,
        sfp_smem_layout_staged: Optional[cute.Layout] = None,
        sfv_smem_layout_staged: Optional[cute.Layout] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(tma_atom_K is not None):
                cpasync.prefetch_descriptor(tma_atom_K)
            if const_expr(tma_atom_V is not None):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)
            if const_expr(tma_atom_sfq is not None):
                cpasync.prefetch_descriptor(tma_atom_sfq)
            if const_expr(tma_atom_sfk is not None):
                cpasync.prefetch_descriptor(tma_atom_sfk)
            if const_expr(tma_atom_sfv is not None):
                cpasync.prefetch_descriptor(tma_atom_sfv)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Use the first N warps to initialize barriers
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_full_offset + i, 1
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 2:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + i, cute.arch.WARP_SIZE * 4
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + i, cute.arch.WARP_SIZE * 4
                )
        if warp_idx == 3:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE
                    )
        if const_expr(not self.use_correction_warps_for_epi) and warp_idx == 4:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_full_offset + i,
                    cute.arch.WARP_SIZE * len(self.correction_warp_ids),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_corr_epi_empty_offset + i,
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids),
                )
        if warp_idx == 5:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_O_rescaled_offset + i,
                    cute.arch.WARP_SIZE
                    * (len(self.softmax0_warp_ids) + len(self.correction_warp_ids)),
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id])
                )
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id])
                )
        if warp_idx == 6:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_P_full_2_offset + i,
                    cute.arch.WARP_SIZE *  len(self.softmax0_warp_ids),
                )
        if warp_idx == 7:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        if warp_idx == 8:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(
                    mbar_ptr + self.mbar_sfqk_load_offset + i,
                    cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
                )
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # K and V share physical SMEM:
        # - same dtype: V aliases K (V uses K's base pointer)
        # - FP4 K + BF16 V: K aliases V (K uses V's base pointer, stride scaled by dtype ratio)
        if const_expr(self.v_dtype == self.k_dtype):
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        elif const_expr(self.k_dtype.width < self.v_dtype.width):
            # K aliases V's buffer — K is smaller, fits inside V's stage.
            # K's stage stride must match V's stage stride in bytes so they align.
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stride_sK_aligned = const_expr(stride_sV * self.v_dtype.width // self.k_dtype.width)
            sK_outer_aligned = cute.make_layout(
                sK_layout.outer.shape,
                stride=(*sK_layout.outer.stride[:-1], stride_sK_aligned),
            )
            sK = storage.sV.get_tensor(sK_outer_aligned, swizzle=sK_layout.inner, dtype=self.k_dtype)
        else:
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))

        # Get scale factor shared memory tensors if they exist
        sSFV = None
        sSFP = None
        sSFQ = None
        sSFK = None
        if const_expr(self.quant_qk):
            sSFQ = storage.sSFQ.get_tensor(sfq_smem_layout_staged)
            sSFK = storage.sSFK.get_tensor(sfk_smem_layout_staged)
        if const_expr(sfp_smem_layout_staged is not None):
            sSFP = storage.sSFP.get_tensor(sfp_smem_layout_staged)
        if const_expr(sfv_smem_layout_staged is not None):
            sSFV = storage.sSFV.get_tensor(sfv_smem_layout_staged)

        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)


        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
            for stage in range(2)
        )
        tOtOs = tuple(
            cute.make_tensor(tOtO.iterator + self.tmem_o_offset[stage], tOtO.layout)
            for stage in range(self.q_stage)
        )

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrPs = [
            cute.make_tensor(
                tOrP.iterator
                + self.qk_acc_dtype.width // tOrP._dtype.width * self.tmem_p_offset[stage],
                tOrP.layout,
            )
            for stage in range(2)
        ]
        # Setup scale factor TMEM tensors and S2T copy operations.
        # For QK, park the block-scaled metadata on the opposite S stage so it
        # doesn't alias the accumulator tile currently being produced.
        align = 16 # required for tcgen05.cp 4x32dp128bit TODO: may be just 4, 4 * 32 = 128 bits
        tCtSFQs = [None] * self.q_stage
        tCtSFKs = [None] * self.q_stage
        if const_expr(self.quant_qk):
            # SFQK tmem placement experiment (FA4_SFQK_TMEM_SLOT env):
            #   "s"  (default) — use tmem_s_offset (staggered): SF for stage k
            #                    lives where S for stage (q_stage-1-k) lives.
            #                    Safe by construction; current baseline.
            #   "o"            — use tmem_o_offset slots. Frees S's tmem from
            #                    SFQK traffic so softmax warp + SFQK MMA copy
            #                    don't compete. Risk: collides with O acc.
            #   "o_stagger"    — o offsets with opposite-stage staggering
            #                    (stage k → tmem_o_offset[q_stage-1-k]).
            if const_expr(self.sfqk_tmem_slot == "o"):
                sfq_base_offsets = self.tmem_o_offset
                sfq_stage_order = tuple(range(self.q_stage))
            elif const_expr(self.sfqk_tmem_slot == "o_stagger"):
                sfq_base_offsets = self.tmem_o_offset
                sfq_stage_order = tuple(
                    self.q_stage - 1 - stage for stage in range(self.q_stage)
                )
            else:  # "s" default — original staggered S-offset
                sfq_base_offsets = self.tmem_s_offset
                sfq_stage_order = tuple(
                    self.q_stage - 1 - stage for stage in range(self.q_stage)
                )
            sfq_tmem_ptrs_f32 = [
                cute.make_ptr(
                    Float32,
                    sfq_base_offsets[sfq_stage_order[stage]],
                    mem_space=cute.AddressSpace.tmem,
                    assumed_align=align,
                )
                for stage in range(self.q_stage)
            ]
            sfq_tmem_ptrs = [
                cute.recast_ptr(sfq_tmem_ptrs_f32[stage], dtype=self.sf_dtype)
                for stage in range(self.q_stage)
            ]

            # (MMA, MMA_M, MMA_K)
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                cute.slice_(sfq_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFQs = [cute.make_tensor(sfq_tmem_ptrs[stage], tCtSFQ_layout) for stage in range(self.q_stage)]

            # Make SFK tmem tensor
            # TMEM offsets are expressed in u32 columns. Match the dense
            # blockscaled kernels instead of deriving a smaller offset from the
            # recast FP8 tensor view.
            sf_atom_mn = 32
            sfq_tmem_cols = (self.mma_tiler_qk[0] // sf_atom_mn) * self.mma_inst_tile_k
            sfk_tmem_ptrs = [
                cute.recast_ptr(sfq_tmem_ptrs_f32[stage] + sfq_tmem_cols, dtype=self.sf_dtype)
                for stage in range(self.q_stage)
            ]

            # (MMA, MMA_N, MMA_K)
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                cute.slice_(sfk_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFKs = [cute.make_tensor(sfk_tmem_ptrs[stage], tCtSFK_layout) for stage in range(self.q_stage)]
        
        # Setup SFP and SFV TMEM tensors
        # Reuse the TMEM of S
        tCtSFPs = [None] * self.q_stage
        tCtSFVs = [None] * self.q_stage
        if const_expr(self.quant_pv):
            sfp_tmem_ptrs = [cute.recast_ptr(
                            cute.make_ptr(Float32, self.tmem_s_offset[stage],
                            mem_space=cute.AddressSpace.tmem, assumed_align=align),
                            dtype=self.sf_dtype) for stage in range(self.q_stage)]
            # (MMA, MMA_M, MMA_K) 
            tCtSFP_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                cute.slice_(sfp_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFPs = [cute.make_tensor(sfp_tmem_ptrs[stage], tCtSFP_layout) for stage in range(self.q_stage)]
            
            # Make SFV tmem tensor.
            # find_tmem_tensor_col_offset returns u32 cols; sfp_tmem_ptrs are
            # typed as `self.sf_dtype` (1 byte for E4M3/E8M0), so we convert
            # u32 cols → SF-element offset via `sf_dtype_per_u32 = 32 / sf_width`.
            # Both E4M3 and E8M0 are 8 bits → 4 SFs per u32 col.
            sf_dtype_per_u32 = 32 // self.sf_dtype.width
            sfp_offset = math.ceil(tcgen05.find_tmem_tensor_col_offset(tCtSFPs[0]) * sf_dtype_per_u32 / align) * align
            sfv_tmem_ptrs = [sfp_tmem_ptrs[stage] + sfp_offset for stage in range(self.q_stage)]

            # (MMA, MMA_N, MMA_K) for P*V operation (V is the B matrix)
            tCtSFV_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.sf_vec_size,
                cute.slice_(sfv_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFVs = [cute.make_tensor(sfv_tmem_ptrs[stage], tCtSFV_layout) for stage in range(self.q_stage)]

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if const_expr(len(self.empty_warp_ids) > 1):
            if warp_idx == self.empty_warp_ids[1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        assert len(self.empty_warp_ids) <= 2

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_sfq,
                tma_tensor_sfq,
                tma_atom_sfk,
                tma_tensor_sfk,
                sSFQ,
                sSFK,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                tma_atom_sfv,
                tma_tensor_sfv,
                sfv_smem_layout_staged,
                sSFV,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
                cute.arch.sync_warp()

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStSs,
                tOtOs,
                tOrPs,
                pipeline_kv,
                mbar_ptr,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                sSFQ,
                sSFK,
                sSFV,
                sSFP,
                tCtSFQs,
                tCtSFKs,
                tCtSFPs,
                tCtSFVs,
            )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    mbar_ptr,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                blocksparse_tensors=blocksparse_tensors,
                sSFP=sSFP,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                if const_expr(self.quant_pv):
                    tCtSFP = tCtSFPs[0] if stage == 0 else tCtSFPs[1]
                else:
                    tCtSFP = None
                softmax_loop(
                    stage=stage,
                    tStSi=cute.make_tensor(
                        tStS.iterator
                        + (self.tmem_s_offset[0] if stage == 0 else self.tmem_s_offset[1]),
                        tStS.layout,
                    ),
                    tCtSFP=tCtSFP, # need to copy P sf to tmem after exp
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
                    softmax_loop(stage=0, tStSi=tStSi, tCtSFP=tCtSFPs[0])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[1], tStS.layout)
                    softmax_loop(stage=1, tStSi=tStSi, tCtSFP=tCtSFPs[1])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtOs,
                sScale,
                mO,
                mLSE,
                sO,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                mbar_ptr,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
            )
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_sfq: Optional[cute.CopyAtom],
        tma_tensor_sfq: Optional[cute.Tensor],
        tma_atom_sfk: Optional[cute.CopyAtom],
        tma_tensor_sfk: Optional[cute.Tensor],
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tma_atom_sfv: Optional[cute.CopyAtom] = None,
        tma_tensor_sfv: Optional[cute.Tensor] = None,
        sfv_smem_layout_staged: Optional[cute.ComposedLayout] = None,
        sSFV: Optional[cute.Tensor] = None,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            # mQ: [s, d, h, b]
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)) # (bM, hdim/bK, RestM)

            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)

            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            if const_expr(self.use_tma_KV):
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
                    cute.group_modes(tOgV, 0, 3),
                )
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmod.create(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            # Partition SFQ similar to Q - index batch and head first, then use select like gQ
            load_SFQ_fn = None
            if const_expr(self.quant_qk):
                tma_tensor_sfq_cur = seqlen.offset_batch_Q(tma_tensor_sfq, batch_idx, dim=3)[None, None, head_idx]
                gSFQ = cute.local_tile(tma_tensor_sfq_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))
                tSgSFQ = thr_mma_qk.partition_A(gSFQ)
                load_SFQ_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_sfq, 0, cute.make_layout(1), tSgSFQ, sSFQ, filter_zeros=True
                )
            
            # Partition SFK similar to K - index batch and head first like mK_cur
            tKsSFK = None
            tKgSFK = None
            if const_expr(self.quant_qk):
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, batch_idx]
                        gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                    else:
                        tma_tensor_sfk_cur = cute.domain_offset((seqlen.offset_k, 0), tma_tensor_sfk[None, None, head_idx_kv])
                        gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                else:
                    tma_tensor_sfk_cur = tma_tensor_sfk[None, None, head_idx_kv, None]
                    gSFK = cute.local_tile(tma_tensor_sfk_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None))
                tSgSFK = thr_mma_qk.partition_B(gSFK)
                # Group only the first 3 modes (static MMA modes) to avoid grouping dynamic Rest modes
                tKsSFK, tKgSFK = cpasync.tma_partition(
                    tma_atom_sfk,
                    0, 
                    cute.make_layout(1),
                    cute.group_modes(sSFK, 0, 3),
                    cute.group_modes(tSgSFK, 0, 3),
                )
                tKsSFK = cute.filter_zeros(tKsSFK)
                tKgSFK = cute.filter_zeros(tKgSFK)
            
            # Partition SFV similar to V
            tVsSFV, tVgSFV = None, None
            if const_expr(self.quant_pv):
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        tma_tensor_sfv_cur = tma_tensor_sfv[None, None, head_idx_kv, batch_idx]
                        gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                    else:
                        tma_tensor_sfv_cur = cute.domain_offset((0, seqlen.offset_k), tma_tensor_sfv[None, None, head_idx_kv])
                        gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                else:
                    tma_tensor_sfv_cur = tma_tensor_sfv[None, None, head_idx_kv, None]
                    gSFV = cute.local_tile(tma_tensor_sfv_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
                tOgSFV = thr_mma_pv.partition_B(gSFV)
                # Group only the first 3 modes (static MMA modes) to avoid grouping dynamic Rest modes
                tVsSFV, tVgSFV = cpasync.tma_partition(
                    tma_atom_sfv,
                    0,  
                    cute.make_layout(1),
                    cute.group_modes(sSFV, 0, 3),
                    cute.group_modes(tOgSFV, 0, 3),
                )
                tVsSFV = cute.filter_zeros(tVsSFV)
                tVgSFV = cute.filter_zeros(tVgSFV)

            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                mbar_ptr + self.mbar_load_q_full_offset,
                mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
                load_SFQ_fn=load_SFQ_fn if const_expr(tma_atom_sfq is not None) else None,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
                tma_atom_sf=tma_atom_sfk,
                tXgSF=tKgSFK,
                tXsSF=tKsSFK,
            )

            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                mbar_ptr + self.mbar_load_kv_full_offset,
                mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
                tma_atom_sf=tma_atom_sfv,
                tXgSF=tVgSFV,
                tXsSF=tVsSFV,
            )

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    if const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE:
                        load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0 + SFQ0
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        mPageTable[batch_idx, n_block_first]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0 + SFK0
                    kv_producer_state.advance()
                    if const_expr(self.q_stage == 2) and (const_expr(self.use_tma_KV) or tidx < cute.arch.WARP_SIZE):
                        load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1 + SFQ1
                    q_producer_phase ^= 1
                    load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # V0 + SFV0
                    kv_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki + SFKi
                        kv_producer_state.advance()
                        load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi + SFVi
                        kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                )


            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        # In smem
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        # In tmem - per-stage scale factors
        tCtSFQs: Tuple[cute.Tensor, ...],
        tCtSFKs: Tuple[cute.Tensor, ...],
        tCtSFPs: Tuple[cute.Tensor, ...],
        tCtSFVs: Tuple[cute.Tensor, ...],
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        if const_expr(self.quant_qk):
            # For MXFP8 QK the generic cute.gemm path is slightly faster than
            # our inline-PTX helper (+1-5% across shapes, +4.5% on
            # (1,32768,24,128)) because the helper's per-K SF-address
            # plumbing adds register pressure. Use generic by default for
            # block-scaled FP8; keep inline-PTX for NVFP4 where it wins.
            # The debug flag lets us force generic on NVFP4 too for A/B tests.
            if const_expr(
                self.debug_force_generic_mxfp8_qk
                or (qk_mma_op.a_dtype in (Float8E4M3FN, Float8E5M2)
                    and not self.debug_mxfp8_use_inline_ptx)
            ):
                gemm_Si = [
                    partial(
                        sm100_utils.gemm_blockscaled_generic,
                        tiled_mma_qk,
                        tStSs[stage],
                        tSrQs[stage],
                        tScaleA=tCtSFQs[stage],
                        tScaleB=tCtSFKs[stage],
                        zero_init=True,
                    )
                    for stage in range(self.q_stage)
                ]
            else:
                gemm_Si = [
                    partial(
                        sm100_utils.gemm_ptx_partial_fp4,
                        qk_mma_op,
                        self.tmem_s_offset[stage],
                        tSrQs[stage],
                        sA=sQ[None, None, None, stage],
                        zero_init=True,
                        tScaleA=tCtSFQs[stage],
                        tScaleB=tCtSFKs[stage],
                    )
                    for stage in range(self.q_stage)
                ]
        else:
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    qk_mma_op,
                    self.tmem_s_offset[stage],
                    tSrQs[stage],
                    sA=sQ[None, None, None, stage],
                    zero_init=True,
                )
                for stage in range(self.q_stage)
            ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial_fp4,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
                tScaleA=tCtSFPs[stage],
                tScaleB=tCtSFVs[stage],
                pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])),
                tA_addr=self.tmem_p_offset[stage],
            ) if const_expr(self.quant_pv) else
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage if self.q_stage == 2 else 0],
                tOrPs[stage],
                sA=None,
                pre_mbar_tiles=self.mbar_p_split(cute.size(tOrPs[stage].shape[2])),
            )
            for stage in range(self.q_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        # Partition for S2T copy of SFQ/SFK - change addr per q_stage to avoid overwriting
        if const_expr(self.quant_qk):
            tiled_copy_s2t_sfq_staged = [
                self.mainloop_s2t_copy_and_partition(sSFQ, tCtSFQs[stage])
                for stage in range(self.q_stage)
            ] 
            tiled_copy_s2t_sfk_staged = [
                self.mainloop_s2t_copy_and_partition(sSFK, tCtSFKs[stage])
                for stage in range(self.q_stage)
            ] 
            tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, _ = tiled_copy_s2t_sfq_staged[0]
            tiled_copy_s2t_sfk, tCsSFK_compact_s2t, _ = tiled_copy_s2t_sfk_staged[0]
        else:
            # Dummy values when quant_qk is False - these won't be used
            tiled_copy_s2t_sfq_staged = []
            tiled_copy_s2t_sfk_staged = []
            tiled_copy_s2t_sfq = None
            tCsSFQ_compact_s2t = None
            tiled_copy_s2t_sfk = None
            tCsSFK_compact_s2t = None

        if const_expr(self.quant_pv):
            tiled_copy_s2t_sfv_staged = [
                self.mainloop_s2t_copy_and_partition(sSFV, tCtSFVs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfv, tCsSFV_compact_s2t, _ = tiled_copy_s2t_sfv_staged[0]
            # S2T copy setup for SFP (P scale factors, computed by softmax warp via R2S)
            tiled_copy_s2t_sfp_staged = [
                self.mainloop_s2t_copy_and_partition(sSFP, tCtSFPs[stage])
                for stage in range(self.q_stage)
            ]
            tiled_copy_s2t_sfp, tCsSFP_compact_s2t, _ = tiled_copy_s2t_sfp_staged[0]
        else:
            tiled_copy_s2t_sfv_staged = []
            tiled_copy_s2t_sfv = None
            tCsSFV_compact_s2t = None
            tiled_copy_s2t_sfp_staged = []
            tiled_copy_s2t_sfp = None
            tCsSFP_compact_s2t = None

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        
        
        mma_sfqk_producer_phase = Int32(0)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase
                    )
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    
                    # Copy SFQ 
                    # only tmem changes per q_stage.
                    if const_expr(self.quant_qk):
                        sm100_utils.tcgen05_after_thread_sync()
                        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + stage, mma_sfqk_producer_phase)
                        _, _, tCtSFQ_compact_s2t = tiled_copy_s2t_sfq_staged[stage]
                        tCsSFQ_compact_s2t_staged = tCsSFQ_compact_s2t[None, None, None, None, stage]
                        if const_expr(not self.debug_skip_sfq_s2t):
                            cute.copy(
                                tiled_copy_s2t_sfq,
                                tCsSFQ_compact_s2t_staged,
                                tCtSFQ_compact_s2t,
                            )

                        _, _, tCtSFK_compact_s2t = tiled_copy_s2t_sfk_staged[stage]
                        tCsSFK_compact_s2t_staged = tCsSFK_compact_s2t[None, None, None, None, mma_kv_consumer_state.index]
                        if const_expr(not self.debug_skip_sfk_s2t):
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_compact_s2t_staged,
                                tCtSFK_compact_s2t,
                            )

                    # 3. gemm
                    # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(
                            sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        )

                    gemm_Si[stage](
                        tCrB=tSrKi,  # tCrB
                        sB=sK_cur,  # sB
                    )

                    # 4. release S0 / S1
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                if const_expr(self.quant_qk):
                    mma_sfqk_producer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile has finished.
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage,
                            P_full_O_rescaled_phase,
                        )
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        
                        # No need for mbar.wait because it depends on the same Si as the prev qk mma
                        if const_expr(self.quant_pv):
                            # S2T copy SFP: sSFP (smem) -> tCtSFPs (tmem)
                            _, _, tCtSFP_compact_s2t = tiled_copy_s2t_sfp_staged[stage]
                            tCsSFP_compact_s2t_cur = tCsSFP_compact_s2t[None, None, None, None, stage]
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_compact_s2t_cur,
                                tCtSFP_compact_s2t,
                            )
                            # S2T copy SFV: sSFV (smem) -> tCtSFVs (tmem)
                            _, _, tCtSFV_compact_s2t = tiled_copy_s2t_sfv_staged[stage]
                            tCsSFV_compact_s2t_staged = tCsSFV_compact_s2t[None, None, None, None, Vi_index]
                            cute.copy(
                                tiled_copy_s2t_sfv,
                                tCsSFV_compact_s2t_staged,
                                tCtSFV_compact_s2t,
                            )

                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)

                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                            mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                            mbar_phase=P_full_O_rescaled_phase,
                        )

                        # 4. release accumulated O0_partial / O1_partial
                        # Don't need to signal O_full to the correction warps anymore since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # with cute.arch.elect_one():
                        #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                        # 5. release V(i-1)
                        if const_expr(stage == 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # tiled_mma_qk = sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrK[None, None, None, Ki_index], zero_init=True)

                        if const_expr(self.quant_qk):
                            _, _, tCtSFQ_compact_s2t = tiled_copy_s2t_sfq_staged[stage]
                            tCsSFQ_compact_s2t_staged = tCsSFQ_compact_s2t[None, None, None, None, stage]
                            sm100_utils.tcgen05_after_thread_sync()
                            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_sfqk_load_offset + stage, mma_sfqk_producer_phase)
                            if const_expr(not self.debug_skip_sfq_s2t):
                                cute.copy(
                                    tiled_copy_s2t_sfq,
                                    tCsSFQ_compact_s2t_staged,
                                    tCtSFQ_compact_s2t,
                                )
                            _, _, tCtSFK_compact_s2t = tiled_copy_s2t_sfk_staged[stage]
                            tCsSFK_compact_s2t_staged = tCsSFK_compact_s2t[None, None, None, None, mma_kv_consumer_state.index]
                            if const_expr(not self.debug_skip_sfk_s2t):
                                cute.copy(
                                    tiled_copy_s2t_sfk,
                                    tCsSFK_compact_s2t_staged,
                                    tCtSFK_compact_s2t,
                                )

                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)

                        gemm_Si[stage](
                            tCrB=tSrK[None, None, None, Ki_index],  # tCrB
                            sB=sK_cur,  # sB
                        )
                        # 3. release S0
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    if const_expr(self.quant_qk):
                        mma_sfqk_producer_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.q_stage):
                        tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase
                    )
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)

                    if const_expr(self.quant_pv):
                        # S2T copy SFP: sSFP (smem) -> tCtSFPs (tmem)
                        _, _, tCtSFP_compact_s2t = tiled_copy_s2t_sfp_staged[stage]
                        tCsSFP_compact_s2t_cur = tCsSFP_compact_s2t[None, None, None, None, stage]
                        cute.copy(
                            tiled_copy_s2t_sfp,
                            tCsSFP_compact_s2t_cur,
                            tCtSFP_compact_s2t,
                        )
                        # S2T copy SFV: sSFV (smem) -> tCtSFVs (tmem)
                        _, _, tCtSFV_compact_s2t = tiled_copy_s2t_sfv_staged[stage]
                        tCsSFV_compact_s2t_staged = tCsSFV_compact_s2t[None, None, None, None, Vi_index]
                        cute.copy(
                            tiled_copy_s2t_sfv,
                            tCsSFV_compact_s2t_staged,
                            tCtSFV_compact_s2t,
                        )
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    _zi_post = not O_should_accumulate
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=_zi_post,
                        mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished compute
                    # the row sum of the current tile. It does not guarantee that the 1st tile
                    # of the next work tile has been computed yet.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop


    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tCtSFP: Optional[Tuple[cute.Tensor, ...]] = None,
        sSFP: Optional[cute.Tensor] = None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )

        tStScale = cute.composition(tStSi, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(
            tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)
        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
            Float32,
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(tidx)

        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        fp8_pv_store_rep = int(os.getenv("FA4_FP8_PV_TMEM_STORE_REP", "8"))
        fp8_pv_default_store_rep = 4 if self.head_dim_v_padded <= 64 else 8
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(
                    fp8_pv_store_rep if const_expr(
                        not self.quant_pv and self.v_dtype.width == 8 and "FA4_FP8_PV_TMEM_STORE_REP" in os.environ
                    ) else
                    fp8_pv_default_store_rep if const_expr(not self.quant_pv and self.v_dtype.width == 8) else
                    8 if const_expr(self.quant_pv) else
                    16
                )
            ),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)
        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)
        
        # First iter: no need for wait correction for sfqk1
        if stage == 1 and const_expr(self.quant_qk): 
            sfqk_stage = 0
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)
        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg
        
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=self.q_stage * m_block + stage,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
            )
            block_mask_mod = self.mask_mod if const_expr(self.use_block_sparsity) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=block_mask_mod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            fp8_pv_p_log2_offset = float(
                os.getenv(
                    "FA4_FP8_PV_P_LOG2_OFFSET",
                    "0.0" if self.head_dim_v_padded <= 64 else "8.0",
                )
            )
            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0,
                # rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0, # (Wenxuan) disable skipping rescale until FP4 precision is verified
                softmax_scale=softmax_scale,
                p_log2_offset=fp8_pv_p_log2_offset if const_expr(not self.quant_pv and self.v_dtype.width == 8) else 0.0,
                quant_pv=self.quant_pv,
                compute_sp1=self.compute_sp1,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=self.q_stage * m_block + stage,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                mask_fn=partial(mask_fn, mask_seqlen=False),
                tCtSFP=tCtSFP,
                sSFP=sSFP,
            )

            if has_work:
                # Softmax acts as the producer: wait until correction signals the stage is empty
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
                )
                si_corr_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mbar_ptr,
                    self.mbar_softmax_corr_full_offset,
                    self.mbar_softmax_corr_empty_offset,
                    self.mbar_P_full_O_rescaled_offset,
                    self.mbar_P_full_2_offset,
                    self.q_stage,
                    Int32(stage),
                )
                if not empty_tile:
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                        mma_si_consumer_phase,
                        si_corr_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    # The remaining iterations have no masking
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(
                        mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block
                    )
                    # Separate iterations with local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    si_corr_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    # Dense path always writes scale / signals
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.m_block_size * 2
                        ] = softmax.row_max[0]
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + utils.log2f(softmax.row_sum[0])) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop
        
    @cute.jit
    def _quant_fp4(self, 
                   # src
                   tSrP_f32: cute.Tensor,
                   tSrPSF_f32: cute.Tensor,
                   # dst
                   tSrP: cute.Tensor,
                   tSrPSF: cute.Tensor,
                   ):
        tSrP_f32_frag = cute.logical_divide(tSrP_f32, cute.make_layout(self.sf_vec_size))
        assert cute.size(tSrP_f32_frag, mode=[1]) == cute.size(tSrPSF_f32)
        tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(self.sf_vec_size))
        tSrPSF_u32_view = cute.recast_tensor(tSrPSF, cute.Int32)

        # Process in groups of 4 for UE4M3 conversion
        assert cute.size(tSrPSF_f32) % 4 == 0
        for i in cutlass.range_constexpr(0, cute.size(tSrPSF_f32) // 4, unroll=1):
        # for i in cutlass.range_constexpr(0, 2):
            # Pack 4 FP32 values into UE4M3 format
            packed_ue4m3 = packed_float_to_ue4m3(
                tSrPSF_f32[i * 4],
                tSrPSF_f32[i * 4 + 1], 
                tSrPSF_f32[i * 4 + 2],
                tSrPSF_f32[i * 4 + 3]
            )
            tSrPSF_u32_view[i] = packed_ue4m3
    
        # Quantize main tensor to E2M1 format (8 values per uint32_t)
        # Process in groups of 8 for E2M1 conversion
        for i in cutlass.range_constexpr(0, cute.size(tSrP_frag, mode=[1])):
            tSrP_u32_view = cute.recast_tensor(tSrP_frag[None, i], cute.Int32)
            for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                packed_e2m1 = packed_float_to_e2m1(
                    tSrP_f32_frag[k * 8, i],
                    tSrP_f32_frag[k * 8 + 1, i],
                    tSrP_f32_frag[k * 8 + 2, i],
                    tSrP_f32_frag[k * 8 + 3, i],
                    tSrP_f32_frag[k * 8 + 4, i],
                    tSrP_f32_frag[k * 8 + 5, i],
                    tSrP_f32_frag[k * 8 + 6, i],
                    tSrP_f32_frag[k * 8 + 7, i]
                )
                tSrP_u32_view[k] = packed_e2m1

    @cute.jit
    def _apply_exp2_pack_fp8(
        self,
        softmax,
        tSrS: cute.Tensor,
        tSrP: cute.Tensor,
    ):
        """Per-fragment interleaved exp2 + packed E4M3.

        Processes one fragment at a time: exp2 all elements in the fragment,
        then F2FP pack that fragment, before moving to the next. This
        interleaves MUFU.EX2 (MIO pipe) with F2FP (also MIO) per fragment,
        preventing long back-to-back MUFU bursts that saturate MIO dispatch.
        """
        assert self.v_dtype == Float8E4M3FN, "only E4M3 packed path implemented"
        frg_tile = min(32, cute.size(tSrS))
        frg_cnt = cute.size(tSrS) // frg_tile
        tSrS_frg = cute.logical_divide(tSrS, cute.make_layout(frg_tile))
        tSrP_frg = cute.logical_divide(tSrP, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, frg_tile, 2):
                tSrS_frg[k, j] = cute.arch.exp2(tSrS_frg[k, j])
                tSrS_frg[k + 1, j] = cute.arch.exp2(tSrS_frg[k + 1, j])
            u32_view = cute.recast_tensor(tSrP_frg[None, j], cute.Int32)
            for k in cutlass.range_constexpr(0, cute.size(u32_view, mode=[0])):
                u32_view[k] = packed_float_to_ue4m3(
                    tSrS_frg[k * 4, j],
                    tSrS_frg[k * 4 + 1, j],
                    tSrS_frg[k * 4 + 2, j],
                    tSrS_frg[k * 4 + 3, j],
                )

    @cute.jit
    def _pack_fp8(self, tSrP_f32: cute.Tensor, tSrP: cute.Tensor):
        # Pack FP32 probabilities directly into FP8 lanes so the pure-FP8 PV path
        # avoids the generic elementwise convert-and-pack sequence.
        if const_expr(self.v_dtype == Float8E4M3FN):
            tSrP_f32_frag = cute.logical_divide(tSrP_f32, cute.make_layout(4))
            tSrP_frag = cute.logical_divide(tSrP, cute.make_layout(4))
            for i in cutlass.range_constexpr(0, cute.size(tSrP_frag, mode=[1])):
                tSrP_u32_view = cute.recast_tensor(tSrP_frag[None, i], cute.Int32)
                for k in cutlass.range_constexpr(0, cute.size(tSrP_u32_view, mode=[0])):
                    tSrP_u32_view[k] = packed_float_to_ue4m3(
                        tSrP_f32_frag[k * 4, i],
                        tSrP_f32_frag[k * 4 + 1, i],
                        tSrP_f32_frag[k * 4 + 2, i],
                        tSrP_f32_frag[k * 4 + 3, i],
                    )
        else:
            # Keep the generic path for E5M2 until we add a dedicated pack helper.
            softmax_like = cute.logical_divide(tSrP, cute.make_layout(min(32, cute.size(tSrP))))
            src_like = cute.logical_divide(tSrP_f32, cute.make_layout(min(32, cute.size(tSrP_f32))))
            for j in cutlass.range_constexpr(cute.size(src_like, mode=[1])):
                softmax_like[None, j].store(src_like[None, j].load().to(tSrP.element_type))

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        tCtSFP: Optional[cute.Tensor] = None,
        sSFP: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        pure_fp8_pv = const_expr(not self.quant_pv and self.v_dtype.width == 8)
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        # P size when in FP32
        tScP = cute.composition(tScS, cute.make_layout((self.m_block_size, tilePlikeFP32)))

        # Wait for Si
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        
        # unblock sfqk load
        cute.arch.fence_view_async_tmem_load()
        sfqk_stage = self.q_stage - 1 - stage
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_sfqk_load_offset + sfqk_stage)

        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                aux_tensors,
                fastdiv_mods,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block) 

        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        tSrPSF_f32 = None
        tSrPSF = None

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_fragment(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase
            )
        tSrP_r2t_f32 = cute.make_fragment(thr_tmem_store.partition_S(tScP).shape, Float32)
        # The FP8 P path recasts this Float32 backing storage into packed FP8 lanes.
        # Zero-initialize first so any padding / partially covered bytes cannot leak
        # stale register contents into the TMEM store on later configs.
        if const_expr(self.fp8_pv_zero_fill_regs):
            tSrP_r2t_f32.fill(0.0)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_dtype),
            tSrS_t2r.layout, # shape of S owned by this thread
        )


        if const_expr(self.quant_pv):
            softmax.apply_exp2_convert(
                tSrS_t2r,
                ex2_emu_freq=self.ex2_emu_freq,
                ex2_emu_start_frg=self.ex2_emu_start_frg,
            )
            # update_row_sum BEFORE scale_groupwise so it uses original P values
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
            tSrPSF_f32 = softmax.compute_group_max(tSrS_t2r, sf_size=self.sf_vec_size)
            tSrPSF = cute.make_rmem_tensor(tSrPSF_f32.layout, self.sf_dtype)
            softmax.scale_groupwise(tSrS_t2r, tSrPSF_f32, sf_size=self.sf_vec_size)
            self._quant_fp4(tSrS_t2r, tSrPSF_f32, tSrP_r2t, tSrPSF)
            # R2S: Copy tSrPSF (registers) to sSFP (shared memory).
            # The SFP smem layout is BlockScaledBasicChunk(sf_vec_size).layout
            # tile_to_shape((M=128, K=128)). The atom is ((32,4),(sf_vec,4))
            # with stride ((16,4),(0,1)) — those strides are independent of
            # sf_vec_size, so `base_offset = lane_id*16 + (warp_id%4)*4` is
            # correct for both NVFP4 (sf_vec=16) and MXFP8 (sf_vec=32).
            # What changes is the K-group count per row:
            #   sf_vec=16, K=128 → 8 SFs per row → 4 inner + 2 outer (rest_k stride 512)
            #   sf_vec=32, K=128 → 4 SFs per row → 4 inner + 1 outer (no rest_k)
            if const_expr(sSFP is not None):
                thread_idx = thr_tmem_load.thr_idx
                lane_id = thread_idx % 32
                warp_id = thread_idx // 32
                base_offset = lane_id * 16 + (warp_id % 4) * 4
                # PV is (M=128, N=head_dim_v, K=n_block_size). SFP scales the
                # P (=A) operand along the K dim, so per-row SF count is
                # n_block_size // sf_vec_size.
                k_groups_per_row = const_expr(self.mma_tiler_pv[2] // self.sf_vec_size)
                k_inner = const_expr(min(k_groups_per_row, 4))
                k_outer = const_expr(k_groups_per_row // k_inner)
                # cute.cosize of one atom = 512 bytes for both sf_vec sizes.
                sfp_thread_layout = cute.make_layout((k_inner, k_outer), stride=(1, 512))
                sSFP_stage_ptr = sSFP[None, None, None, stage].iterator
                sSFP_thread = cute.make_tensor(sSFP_stage_ptr + base_offset, sfp_thread_layout)
                tSrPSF_2d = cute.logical_divide(tSrPSF, cute.make_layout(k_inner))
                cute.autovec_copy(tSrPSF_2d, sSFP_thread)
        else:
            # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
            if const_expr(pure_fp8_pv and self.v_dtype == Float8E4M3FN and self.fp8_pv_use_explicit_pack):
                if const_expr(self.fp8_pv_use_fused_pack):
                    self._apply_exp2_pack_fp8(
                        softmax,
                        tSrS_t2r,
                        tSrP_r2t,
                    )
                else:
                    softmax.apply_exp2_convert(
                        tSrS_t2r,
                        ex2_emu_freq=self.ex2_emu_freq,
                        ex2_emu_start_frg=self.ex2_emu_start_frg,
                    )
                    self._pack_fp8(tSrS_t2r, tSrP_r2t)
            else:
                softmax.apply_exp2_convert(
                    tSrS_t2r,
                    tSrP_r2t,
                    ex2_emu_freq=self.ex2_emu_freq,
                    ex2_emu_start_frg=self.ex2_emu_start_frg,
                )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)
        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_r2t.shape[2]))):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready (and SFP is in SMEM)
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(self.mbar_p_split(cute.size(tStP_r2t.shape[2])), cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        cute.arch.mbarrier_wait(
            mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase
        )

        if const_expr(not self.quant_pv):
            softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.arch.exp2(acc_scale_)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtOs: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
            self.qk_acc_dtype,
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0)
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1)

        softmax_corr_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))

            # Default LSE to -inf for invalid split_idx tiles
            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(blocksparse_tensors, batch_idx, head_idx, m_block)
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + 0, softmax_corr_consumer_phase
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)
                cute.arch.mbarrier_wait(
                    mbar_ptr + self.mbar_softmax_corr_full_offset + 1, softmax_corr_consumer_phase
                )
                softmax_corr_consumer_phase ^= 1

                tSrScale_t2r = cute.make_fragment(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait for S0 / S1
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                            softmax_corr_consumer_phase,
                        )
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv, tOtOs[stage if self.q_stage == 2 else 0], tidx, scale
                            )
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
                        cute.arch.mbarrier_arrive(
                            mbar_ptr + self.mbar_softmax_corr_empty_offset + (1 - stage)
                        )
                    softmax_corr_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                (self.q_stage * m_block + stage) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_softmax_corr_full_offset + stage,
                        softmax_corr_consumer_phase,
                    )
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[tidx + stage * self.m_block_size + self.m_block_size * 2]
                    else:
                        row_max = None
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage)
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(1.0)
                            else:
                                row_sum += utils.exp2f(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    cute.arch.mbarrier_wait(
                        mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_empty_offset + stage, corr_epi_producer_phase
                        )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtOs[stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O,
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_full_offset + stage)
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)

                o_corr_consumer_phase ^= 1
                softmax_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # WARNING: we need some code before the const_expr, see https://github.com/NVIDIA/cutlass/issues/2781
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                else:
                    gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_block_sparsity):
                    (
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtOs,
                        sO,
                        mbar_ptr,
                        self.mbar_softmax_corr_full_offset,
                        self.mbar_softmax_corr_empty_offset,
                        self.mbar_P_full_O_rescaled_offset,
                        self.mbar_P_full_2_offset,
                        self.mbar_corr_epi_full_offset,
                        self.mbar_corr_epi_empty_offset,
                        softmax_corr_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (self.q_stage * m_block + stage,)
                    )
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            row_max * softmax_scale_log2
                            + utils.log2f(row_sum)
                            - (
                                fp8_pv_p_log2_offset
                                if const_expr(not self.quant_pv and self.v_dtype.width == 8)
                                else 0.0
                            )
                        ) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if tidx < seqlen_q - (self.q_stage * m_block + stage) * self.m_block_size:
                        # This actually just works with PackGQA too
                        gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)
        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor, # tmem
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor(softmax denominator) to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0]).get_slice(
            tidx
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = utils.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            tOrO_frg_cvt = cute.make_fragment(tOrO_frg.shape, self.o_dtype)
            tOrO_frg_cvt.store(tOrO_frg.load().to(self.o_dtype))
            cute.copy(tiled_smem_store, tOrO_frg_cvt, tOsO_r2s_i)
        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        if const_expr(self.use_correction_warps_for_epi):
            assert(not self.use_tma_O)
            assert(gmem_tiled_copy_O is not None)
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                              number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE)
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = utils.predicate_k(tOcO, limit=mO_cur.shape[1])
            # TODO: the packgqa case isn't correct rn (sometimes IMA), disabling it
            assert not self.pack_gqa
            pack_gqa = PackGQA(
                self.m_block_size,
                self.head_dim_v_padded,
                self.check_hdim_v_oob,
                self.qhead_per_kvhead,
            )
        
            # load acc O from smem to rmem for wider vectorization
            tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
            cute.autovec_copy(tOsO, tOrO)
            # copy acc O from rmem to gmem
            if const_expr(not self.pack_gqa):
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen_q
                        - (self.q_stage * m_block + stage) * self.m_block_size
                        - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(
                    mO_cur,
                    tOrO,
                    gmem_tiled_copy_O,
                    tidx,
                    self.q_stage * m_block + stage,
                    seqlen_q,
                )

    @cute.jit
    def add_delta_s(self, acc: cute.Tensor, sDeltaS: cute.Tensor, stage: int):
        """Add delta_s smoothing factors (computed from avg pooled qkv attn) to attention accumulator.
        
        This function implements the delta_s addition exactly like SageAttention:
        1. Recast delta_s to float4 for efficient processing
        2. Use quad-based indexing with thread coordination
        3. Apply delta_s values using complex coordinate indexing
        
        :param acc: Attention accumulator tensor to modify
        :type acc: cute.Tensor
        :param sDeltaS: Shared memory tensor containing delta_s values
        :type sDeltaS: cute.Tensor
        :param stage: Processing stage (0 or 1)
        :type stage: int
        """
        if const_expr(sDeltaS is None):
            return
            
        # Get thread index for quad-based processing (matches SageAttention)
        tidx, _, _ = cute.arch.thread_idx()
        quad_id = (tidx % 4) * 2
        
        # Recast delta_s to float4 for efficient processing (matches SageAttention)
        sDeltaS_stage = sDeltaS[None, None, stage]
        tSsDS_stage = cute.recast(sDeltaS_stage, Float32)
        
        # Recast accumulator to float4 for efficient processing
        acc_float4 = cute.recast(acc, Float32)
        
        # Process in groups of 4 float4 values (matches SageAttention pattern)
        for i in cutlass.range(0, 4, unroll=True):
            num = quad_id + i * 8
            
            # Load delta_s values for current quad using coordinate indexing
            # This matches the SageAttention pattern exactly
            delta_s_0 = tSsDS_stage[0, num]
            delta_s_1 = tSsDS_stage[0, num + 1]
            
            # Apply delta_s to accumulator using quad-based indexing
            # This follows the exact SageAttention coordinate pattern
            acc_float4[0, 0, i] += delta_s_0
            acc_float4[0, 1, i] += delta_s_0
            acc_float4[1, 0, i] += delta_s_1
            acc_float4[1, 1, i] += delta_s_1

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
                if const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=self.q_stage * m_block + stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                    tOsO = gmem_thr_copy_O.partition_S(sO)
                    cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                    tOgO = gmem_thr_copy_O.partition_D(gO)
                    tOcO = gmem_thr_copy_O.partition_S(cO)
                    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                    tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                    # TODO: the packgqa case isn't correct rn (sometimes IMA), disabling it
                    assert not self.pack_gqa
                    pack_gqa = PackGQA(
                        self.m_block_size,
                        self.head_dim_v_padded,
                        self.check_hdim_v_oob,
                        self.qhead_per_kvhead,
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        cute.arch.mbarrier_wait(
                            mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        # load acc O from smem to rmem for wider vectorization
                        tOrO = cute.make_fragment_like(tOsO[None, None, None, 0], self.o_dtype)
                        cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                        # copy acc O from rmem to gmem
                        if const_expr(not self.pack_gqa):
                            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                                if (
                                    t0OcO[0, rest_m, 0][0]
                                    < seqlen.seqlen_q
                                    - (self.q_stage * m_block + stage) * self.m_block_size
                                    - tOcO[0][0]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_O,
                                        tOrO[None, rest_m, None],
                                        tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                                        pred=tOpO[None, rest_m, None]
                                        if const_expr(self.check_hdim_v_oob)
                                        else None,
                                    )
                        else:
                            pack_gqa.store_O(
                                mO_cur,
                                tOrO,
                                gmem_tiled_copy_O,
                                tidx,
                                self.q_stage * m_block + stage,
                                seqlen.seqlen_q,
                            )
                        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)

                epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
        load_SFQ_fn: Optional[Callable] = None,
    ):
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_bytes["Q"])
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)
        
        # Load scale factor for Q if provided
        if const_expr(load_SFQ_fn is not None):
            load_SFQ_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        tma_atom_sf: Optional[cute.CopyAtom] = None,
        tXgSF: Optional[cute.Tensor] = None,
        tXsSF: Optional[cute.Tensor] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)

        if const_expr(self.use_tma_KV):
            assert (
                tXgX is not None and
                tXsX is not None and
                tma_atom is not None
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full_ptr + stage, self.tma_copy_bytes[K_or_V],
                )
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)
        else:
            assert paged_kv_manager is not None
            paged_kv_manager.load_KV(block, sX[None, None, None, stage], K_or_V)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_mbarrier_arrive_noinc(mbar_full_ptr + stage)
        
        # Load scale factor for K or V if provided (uses same barrier as K/V)
        if const_expr(tma_atom_sf is not None and tXgSF is not None and tXsSF is not None):
            tXsSF_cur = tXsSF[None, stage]
            # After tma_partition with rank-1 grouping, tXgSF has structure: ((atom_v, rest_v), RestL)
            if const_expr(page_idx is None):
                tXgSF_cur = tXgSF[None, block]
            else:
                tXgSF_cur = tXgSF[None, page_idx]
            cute.copy(tma_atom_sf, tXgSF_cur, tXsSF_cur, tma_bar_ptr=mbar_full_ptr + stage)

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tCtSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr, use_k_bytes: bool = True):
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        if self.use_tma_KV:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids)
            )
            return cutlass.pipeline.PipelineTmaUmma.create(
                barrier_storage=load_kv_mbar_ptr,
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                tx_count=self.tma_copy_bytes["K"] if use_k_bytes else self.tma_copy_bytes["V"],
            )
        else:
            load_kv_producer_group = cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, len(self.load_warp_ids) * cute.arch.WARP_SIZE
            )
            return cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                producer_group=load_kv_producer_group,
                consumer_group=load_kv_consumer_group,
                barrier_storage=load_kv_mbar_ptr,
            )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical = q_physical // self.qhead_per_kvhead
            head_offset = q_physical - q_idx_logical * self.qhead_per_kvhead
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = seqlen_q_divmod.divmod(q_idx_logical)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
        )
