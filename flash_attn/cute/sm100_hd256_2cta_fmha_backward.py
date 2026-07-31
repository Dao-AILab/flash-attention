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

from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.utils import AuxData, as_bshkrd_tensor, as_shhb_tensor


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
    ):
        """Initialization."""
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 dedicated backward kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert tile_m_dq == 128 and tile_n_dq == 128, (
            "SM100 dedicated backward kernel only supports tile_m_dq=128 and tile_n_dq=128"
        )
        assert tile_m_dkdv == 128 and tile_n_dkdv == 64, (
            "SM100 dedicated backward kernel only supports tile_m_dkdv=128 and tile_n_dkdv=64"
        )
        assert cluster_size in (1, 2), (
            "SM100 backward with head_dim=256 only supports cluster_size in {1, 2}"
        )
        assert use_2cta_instrs, "SM100 backward with head_dim=256 requires use_2cta_instrs=True"
        # q_subtile_factor is accepted for interface parity with FlashAttentionBackwardSm100,
        # but this dedicated kernel uses fixed internal behavior.

        self.acc_dtype = cutlass.Float32
        self.is_causal = is_causal
        self.is_local = is_local
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
        self.qhead_per_kvhead = qhead_per_kvhead
        self.use_clc_scheduler = use_clc_scheduler

        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            head_dim,
            head_dim_v,
            is_causal=self.is_causal,
            is_local=self.is_local,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=self.tile_m_dq,
            tile_n=self.tile_n_dq,
            is_persistent=False,
            deterministic=deterministic,
            spt=None,
            cluster_size=2,
            use_2cta_instrs=use_2cta_instrs,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            mask_mod=mask_mod,
            has_aux_tensors=has_aux_tensors,
            subtile_factor=q_subtile_factor,
        )
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            head_dim,
            head_dim_v,
            is_causal=self.is_causal,
            is_local=self.is_local,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=self.tile_m_dkdv,
            tile_n=self.tile_n_dkdv,
            is_persistent=False,
            deterministic=deterministic,
            spt=None,
            cluster_size=cluster_size,
            use_2cta_instrs=use_2cta_instrs,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            mask_mod=mask_mod,
            has_aux_tensors=has_aux_tensors,
            subtile_factor=q_subtile_factor,
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
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        dQ_semaphore: cute.Tensor | None = None,
        dK_semaphore: cute.Tensor | None = None,
        dV_semaphore: cute.Tensor | None = None,
        aux_data: AuxData = AuxData(),
        block_sparse_tensors: BlockSparseTensors | None = None,
        stream: cuda.CUstream = None,
    ):
        """Host function to launch CuTeDSL kernel."""
        assert dQ_accum is not None, (
            "SM100 backward with head_dim=256 expects dQ tensor at dQ_accum slot"
        )
        dQ = dQ_accum
        mQ, mK, mV = Q, K, V
        mdO, mLSE, mdPsum = dO, lse_log2, dpsum
        mdK, mdV = dK, dV
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
        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            dK = as_bshkrd_tensor(dK, h_k, 1, varlen)
            dV = as_bshkrd_tensor(dV, h_k, 1, varlen)
        dO = as_bshkrd_tensor(dO, h_k, h_r, varlen)
        scaled_LSE = as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = as_shhb_tensor(dpsum, h_k, h_r, b, varlen)

        # Keep original order: dQ first, then dKdV.
        self.dq_kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            dQ_accum,
            scale_softmax,
            mCuSeqlensQ=cumulative_s_q,
            mCuSeqlensK=cumulative_s_k,
            mSeqUsedQ=seqused_q,
            mSeqUsedK=seqused_k,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            mdQ_semaphore=None,
            mdK_semaphore=None,
            mdV_semaphore=None,
            aux_data=aux_data,
            stream=stream,
        )
        self.dkdv_kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdK,
            mdV,
            scale_softmax,
            mCuSeqlensQ=cumulative_s_q,
            mCuSeqlensK=cumulative_s_k,
            mSeqUsedQ=seqused_q,
            mSeqUsedK=seqused_k,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            mdK_semaphore=dK_semaphore,
            mdV_semaphore=dV_semaphore,
            aux_data=aux_data,
            stream=stream,
        )
