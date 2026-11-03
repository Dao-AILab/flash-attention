# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.


"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* mma_tiler_mn must be 64,64
* Batch size must be the same for Q, K, and V tensors
"""

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.utils import AuxData, as_bshkrd_tensor, as_shhb_tensor


# NOTE [MXFP8 hd256 backward]
# With `mxfp8`, the dedicated split kernels run every GEMM except dQ = dS K as an SS block-scaled
# MMA on E4M3 operands with one UE8M0 scale per 32 elements:
#   dQ kernel:    S = Q K^T and dP = dO V^T (scales along head_dim); dQ = dS K stays BF16.
#   dK/dV kernel: S^T = K Q^T and dP^T = V dO^T (scales along head_dim), dV = P^T dO and
#                 dK = dS^T Q (scales along queries).
# It is the backward of the MXFP8 forward (BlockScales with native PV) and differentiates the
# function that forward computed, with P recomputed from the final statistics: the quantizers are
# straight-through and the row max is a constant. That P matches the forward's only up to its online
# rounding and rescaling, bitwise where the forward used the final row max (BlockScales.stats_out).
# With c = softmax_scale * log2(e), offset off (8 "exact", 4 "fast"), the forward's row max M (log2
# units) and normalizer L (BlockScales.stats_out), P~ = 2^(S c - (M - off)) and Phat = e4m3(P~):
#     dV = (Phat / L)^T dO,   dS = (P~ / L) (dO Vhat^T - D),   D = rowsum(dO * O),
#     dQ = softmax_scale dS Khat,   dK = softmax_scale dS^T Qhat.
# The caller passes MxHd256BwdOperands (interface.py):
#   * S operands: the forward's Q / K payloads and scales, unchanged;
#   * dP's V: V-hat requantized along head_dim; dK's Q^T: Q-hat requantized along queries; dQ's K:
#     K-hat in BF16 (exact). Requantizing an E4M3 payload with power-of-two scales along another
#     axis is exact down to 2^-14.8 of the new group's maximum (E4M3's normal range); smaller
#     values round to subnormals, and below 2^-17.8 to zero;
#   * dO'' = 2^8 dO / L in both dO copies (along head_dim and along queries) and
#     D'' = rowsum(dequant(dO''_d) * O): the 1 / L of dS and dV folded into dO, times the 2^8 that
#     the dK / dV kernel removes (its P^T scale 2^-8 and its dS^T scale bytes minus 8);
#   * the LSE slots M - off (dK / dV kernel: P~ <= 2^off fits E4M3 with the constant P^T scale
#     2^-8, and is bitwise the forward's P wherever the forward used the final row max) and
#     M - off + 8 (dQ kernel).
#
# Scale layouts. The dQ kernel's pair MMAs are M=256 with standard 128-row interleave_sf chunks.
# The dK / dV kernel's pair MMAs are M=128 (64 keys per CTA; accumulator layout B: lanes 64..127
# hold columns N/2..), which CUTLASS's scale-layout helpers do not cover:
#   * A scales (K, V, dS^T, the constant P^T): the 128-row chunk of an M=256 MMA with key k in rows
#     k and k + 64, copied to all four lane quadrants (tcgen05.cp .32x128b.warpx4). Host tile per
#     64 keys (1024 B): row r holds key r % 64, group g at byte
#     16 (r % 32) + 4 (r // 32) + g % 4 + 512 (g // 4).
#   * B scales (Q, dO with N=128; Q_T, dO_T with N=256): CUTLASS's TMEM SFB layout of this MMA,
#     second half of N in lane quadrants 2/3, one tcgen05.cp .64x128b.warpx2::01_23 copy. Host
#     tile per 128 queries (Q, dO; 1024 B): query n, group g at byte
#     16 (n % 32) + 4 ((n % 64) // 32) + 512 (n // 64) + g % 4 + 8 (g // 4).
#     Q_T / dO_T use interleave_sf buffers (MN = channels), loaded half per CTA and multicast.
# A dS^T compute thread holds one key and 32 adjacent queries (contiguous t2r chunks per warp
# group), so its 1x32 amax is thread-local. Scales reach TMEM by tcgen05.cp from SMEM, issued by
# the MMA warp right before the MMA that reads them. The dQ kernel's TMEM is full on sm_100 / sm_103
# (S, dP, dQ): the S scales sit in the tail of dP (copied once dP(i-1) has been read; dS uses dP's
# first 64 columns) and the dP scales in the tail of S, so QK(i) also waits for the dP(i-1) release;
# sm_107 gives them dedicated columns of its 576-column allocation.
# Scope: non-causal, MHA, dense (no varlen / seqused), seqlen_q = seqlen_k, static scheduler.


class BlackwellFusedMultiHeadAttentionBackward:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int | None = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_persistent: bool = False,
        deterministic: bool = False,
        cluster_size: int = 1,
        use_2cta_instrs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        q_subtile_factor: cutlass.Constexpr[int] = 1,
        tile_m_dq: int = 128,
        tile_n_dq: int = 128,
        tile_m_dkdv: int = 128,
        tile_n_dkdv: int = 64,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        use_clc_scheduler: bool = False,
        mxfp8: bool = False,
    ):
        """Initialization. mxfp8: NOTE [MXFP8 hd256 backward]."""
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 dedicated backward kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert not is_local, "SM100 backward with head_dim=256 does not support local attention"
        assert tile_m_dq == 128 and tile_n_dq == 128, (
            "SM100 dedicated backward kernel only supports tile_m_dq=128 and tile_n_dq=128"
        )
        assert tile_m_dkdv == 128 and tile_n_dkdv == 64, (
            "SM100 dedicated backward kernel only supports tile_m_dkdv=128 and tile_n_dkdv=64"
        )
        assert score_mod is None and score_mod_bwd is None and mask_mod is None, (
            "SM100 backward with head_dim=256 does not support score_mod/mask_mod"
        )
        assert not has_aux_tensors, "SM100 backward with head_dim=256 does not support aux_tensors"
        assert cluster_size in (1, 2), (
            "SM100 backward with head_dim=256 only supports cluster_size in {1, 2}"
        )
        assert use_2cta_instrs, "SM100 backward with head_dim=256 requires use_2cta_instrs=True"
        # q_subtile_factor is accepted for interface parity with FlashAttentionBackwardSm100,
        # but this dedicated kernel uses fixed internal behavior.

        self.acc_dtype = cutlass.Float32
        self.is_causal = is_causal
        self.window_size_left = (
            None if (window_size_left is None or window_size_left < 0) else window_size_left
        )
        self.window_size_right = (
            None if (window_size_right is None or window_size_right < 0) else window_size_right
        )
        self.tile_m_dq = tile_m_dq
        self.tile_n_dq = tile_n_dq
        self.tile_m_dkdv = tile_m_dkdv
        self.tile_n_dkdv = tile_n_dkdv
        self.use_clc_scheduler = use_clc_scheduler
        self.mxfp8 = mxfp8

        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            self.acc_dtype,
            (self.tile_m_dq, self.tile_n_dq, 256),
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            False,  # is_persistent
            False,  # split_head
            use_clc_scheduler=self.use_clc_scheduler,
            mxfp8=mxfp8,
        )
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            self.acc_dtype,
            (self.tile_m_dkdv, self.tile_n_dkdv, 256),
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            use_clc_scheduler=self.use_clc_scheduler,
            mxfp8=mxfp8,
        )

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dO: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        dQ_accum: cute.Tensor | None,
        dK: cute.Tensor,
        dV: cute.Tensor,
        scale_softmax: cutlass.Float32,
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        seqused_q: cute.Tensor | None = None,
        seqused_k: cute.Tensor | None = None,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
        dQ_semaphore: cute.Tensor | None = None,
        dK_semaphore: cute.Tensor | None = None,
        dV_semaphore: cute.Tensor | None = None,
        aux_data: AuxData = AuxData(),
        block_sparse_tensors: cute.Tensor | None = None,
        max_seqlen_q: Int32 | None = None,
        max_seqlen_k: Int32 | None = None,
        # NOTE [MXFP8 hd256 backward]: the dK / dV kernel's LSE slot, E4M3 (b, s, h, d) copies, and
        # raw UE8M0 scale buffers: Q_T / dO_T (scales along queries), Q / K / V / dO (scales along d;
        # interleaved for the dQ kernel, *_dkdv pair tiles for the dK / dV kernel)
        lse_log2_mx: cute.Tensor | None = None,
        QT_mx: cute.Tensor | None = None,
        SFQT_mx: cute.Tensor | None = None,
        dOT_mx: cute.Tensor | None = None,
        SFdOT_mx: cute.Tensor | None = None,
        Q_mx: cute.Tensor | None = None,
        SFQ_mx: cute.Tensor | None = None,
        SFQ_dkdv: cute.Tensor | None = None,
        K_mx: cute.Tensor | None = None,
        SFK_mx: cute.Tensor | None = None,
        SFK_dkdv: cute.Tensor | None = None,
        V_mx: cute.Tensor | None = None,
        SFV_mx: cute.Tensor | None = None,
        SFV_dkdv: cute.Tensor | None = None,
        dO_mx: cute.Tensor | None = None,
        SFdO_mx: cute.Tensor | None = None,
        SFdO_dkdv: cute.Tensor | None = None,
        stream: cuda.CUstream = None,
    ):
        """Host function to launch CuTeDSL kernel."""
        assert window_size_left is None and window_size_right is None, (
            "SM100 backward with head_dim=256 uses constructor-provided window sizes"
        )
        assert dQ_semaphore is None and dK_semaphore is None and dV_semaphore is None, (
            "SM100 backward with head_dim=256 does not use semaphores"
        )
        assert block_sparse_tensors is None, (
            "SM100 backward with head_dim=256 does not support block sparse tensors"
        )
        assert aux_data.tensors is None or len(aux_data.tensors) == 0, (
            "SM100 backward with head_dim=256 does not support aux_tensors"
        )
        assert aux_data.scalars is None or len(aux_data.scalars) == 0, (
            "SM100 backward with head_dim=256 does not support aux_scalars"
        )
        assert dQ_accum is not None, (
            "SM100 backward with head_dim=256 expects dQ tensor at dQ_accum slot"
        )
        dQ = dQ_accum
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        q_rank = cute.rank(Q.layout)
        k_rank = cute.rank(K.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = Q.shape[2] * Q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = Q.shape[2]
        else:
            h_q = Q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = K.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = K.shape[2]
        else:
            h_k = K.shape[1]
        h_r = h_q // h_k
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]

        Q, K, V, dQ, dK, dV, dO = [assume_tensor_aligned(t) for t in (Q, K, V, dQ, dK, dV, dO)]

        Q = as_bshkrd_tensor(Q, h_k, h_r, varlen)
        K = as_bshkrd_tensor(K, h_k, 1, varlen)
        V = as_bshkrd_tensor(V, h_k, 1, varlen)
        dQ = as_bshkrd_tensor(dQ, h_k, h_r, varlen)
        dK = as_bshkrd_tensor(dK, h_k, 1, varlen)
        dV = as_bshkrd_tensor(dV, h_k, 1, varlen)
        dO = as_bshkrd_tensor(dO, h_k, h_r, varlen)
        scaled_LSE = as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = as_shhb_tensor(dpsum, h_k, h_r, b, varlen)
        scaled_LSE_dkdv = scaled_LSE
        if cutlass.const_expr(self.mxfp8):
            assert not varlen
            scaled_LSE_dkdv = as_shhb_tensor(lse_log2_mx, h_k, h_r, b, varlen)
            QT_mx, dOT_mx, Q_mx, dO_mx = [
                as_bshkrd_tensor(assume_tensor_aligned(t), h_k, h_r, varlen)
                for t in (QT_mx, dOT_mx, Q_mx, dO_mx)
            ]
            K_mx, V_mx = [
                as_bshkrd_tensor(assume_tensor_aligned(t), h_k, 1, varlen) for t in (K_mx, V_mx)
            ]

        # Keep original order: dQ first, then dKdV.
        self.dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            seqused_q,
            seqused_k,
            scale_softmax,
            max_seqlen_q,
            stream,
            Q_mx,
            SFQ_mx,
            K_mx,
            SFK_mx,
            V_mx,
            SFV_mx,
            dO_mx,
            SFdO_mx,
        )
        self.dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            scaled_LSE_dkdv,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            seqused_q,
            seqused_k,
            scale_softmax,
            max_seqlen_k,
            stream,
            QT_mx,
            SFQT_mx,
            dOT_mx,
            SFdOT_mx,
            Q_mx,
            SFQ_dkdv,
            K_mx,
            SFK_dkdv,
            V_mx,
            SFV_dkdv,
            dO_mx,
            SFdO_dkdv,
        )
