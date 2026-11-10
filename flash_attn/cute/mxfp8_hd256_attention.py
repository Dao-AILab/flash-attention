"""MXFP8 hdim-256 attention for training: the native MXFP8 PV forward and its mirrored MXFP8 backward.

NOTE [MXFP8 hd256 attention]
`mxfp8_hd256_attn_func(q, k, v, causal, pv_softmax=...)` is an opt-in autograd function:
  forward  quantizes the BF16 Q / K / V once (MXFP8 1x32: Q / K along head_dim, V along keys), runs the native
           MXFP8 PV forward with `BlockScales(..., stats_out=...)`, and saves O, LSE, the row max m and normalizer
           L, and the backward's forward-only operands, which the same prep pass emits (Q_T, BF16 K-hat, V-hat
           along head_dim, and the causal band residual);
  backward differentiates the function that forward computed, with P recomputed from the final (m, L)
           (the forward's P up to its online rounding and rescaling, BlockScales.stats_out; NOTE [MXFP8
           hd256 backward] in sm100_hd256_2cta_fmha_backward.py). It reuses the forward's payloads instead
           of requantizing the BF16 inputs; its prep is one pass over dO.
The prep is fused Triton (NOTE [MXFP8 hd256 prep] in mxfp8_hd256_prep.py). `mxfp8_hd256_forward` /
`mxfp8_hd256_backward` expose the two halves. The scope is that of the dedicated hdim-256 backward kernels:
dense, BF16, MHA (num_heads_q = num_heads_kv), seqlen_q = seqlen_k, a multiple of 32 (256 when causal).
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch.autograd.function import once_differentiable

from flash_attn.cute import mxfp8_hd256_prep as prep
from flash_attn.cute.interface import BlockScales, _flash_attn_bwd, _flash_attn_fwd


@dataclass
class MxHd256Saved:
    """What mxfp8_hd256_backward reads from the forward. stats: (B, H, S, 2) FP32 BlockScales.stats_out (row
    max m, normalizer L)."""

    bwd_operands: prep.ForwardOnlyOperands
    k_hat: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    stats: torch.Tensor
    causal: bool
    softmax_scale: float
    pv_softmax: Literal["exact", "fast"]


def _check_inputs(q, k, v, causal):
    if any(t.dtype != torch.bfloat16 or t.ndim != 4 for t in (q, k, v)):
        raise ValueError("mxfp8_hd256 attention takes BF16 (batch, seqlen, heads, 256) Q / K / V")
    if q.shape[-1] != 256 or k.shape != q.shape or v.shape != q.shape:
        raise NotImplementedError(
            "mxfp8_hd256 attention needs head_dim 256, seqlen_q = seqlen_k and num_heads_q = num_heads_kv"
        )
    if q.shape[1] % (256 if causal else 32) != 0:
        raise NotImplementedError(
            "mxfp8_hd256 attention needs seqlen a multiple of 32 (256 when causal)"
        )


def mxfp8_hd256_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    pv_softmax: Literal["exact", "fast"] = "exact",
) -> tuple[torch.Tensor, torch.Tensor, MxHd256Saved]:
    """Native MXFP8 PV forward. Returns (O, LSE, saved state for mxfp8_hd256_backward)."""
    _check_inputs(q, k, v, causal)
    q, k, v = (t.detach().contiguous() for t in (q, k, v))
    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    ops = prep.quantize_forward(q, k, v, causal)
    b, s, h, _ = q.shape
    stats = torch.empty(b, h, s, 2, device=q.device, dtype=torch.float32)
    # causal native PV reads the original V on the BF16 band
    scales = BlockScales(
        ops.sfq, ops.sfk, ops.sfv, v if causal else None, pv_softmax=pv_softmax, stats_out=stats
    )
    out, lse = _flash_attn_fwd(
        ops.q8,
        ops.k8,
        ops.v8,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
        scales=scales,
    )[:2]
    saved = MxHd256Saved(ops.bwd, ops.k_hat, out, lse, stats, causal, softmax_scale, pv_softmax)
    return out, lse, saved


def mxfp8_hd256_backward(
    saved: MxHd256Saved, dout: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirrored MXFP8 backward of mxfp8_hd256_forward. Returns (dq, dk, dv) in BF16."""
    dout = dout.contiguous()
    operands = prep.backward_operands(
        saved.bwd_operands, saved.out, dout, saved.stats, saved.softmax_scale, saved.pv_softmax
    )
    # The MXFP8 kernels read only the shapes of the BF16 Q / V slots and K-hat for dQ = dS K.
    k_hat = saved.k_hat
    dq, dk, dv = _flash_attn_bwd(
        k_hat,
        k_hat,
        k_hat,
        saved.out,
        dout,
        saved.lse,
        softmax_scale=saved.softmax_scale,
        causal=saved.causal,
        mx_hd256=operands,
    )[:3]
    return dq, dk, dv


class MxHd256AttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale, pv_softmax):
        out, _, saved = mxfp8_hd256_forward(q, k, v, causal, softmax_scale, pv_softmax)
        ctx.save_for_backward(*saved.bwd_operands, saved.k_hat, saved.out, saved.lse, saved.stats)
        ctx.config = (saved.causal, saved.softmax_scale, saved.pv_softmax)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        *bwd, k_hat, out, lse, stats = ctx.saved_tensors
        saved = MxHd256Saved(prep.ForwardOnlyOperands(*bwd), k_hat, out, lse, stats, *ctx.config)
        return (*mxfp8_hd256_backward(saved, dout), None, None, None)


def mxfp8_hd256_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    pv_softmax: Literal["exact", "fast"] = "exact",
) -> torch.Tensor:
    """MXFP8 hdim-256 attention with the mirrored MXFP8 backward (NOTE [MXFP8 hd256 attention])."""
    return MxHd256AttnFunc.apply(q, k, v, causal, softmax_scale, pv_softmax)
