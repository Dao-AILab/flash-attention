"""Trainable MXFP8 attention: the MXFP8 forward and the backward that mirrors it.

`mxfp8_attn_func(q, k, v, causal, softmax_scale, pv_normalizer, deterministic)` takes BF16
(batch, seqlen, heads, head_dim) Q / K / V. Its forward quantizes them once to MXFP8 (Q / K along
head_dim; V along keys for native PV) and runs the MXFP8 forward with `BlockScales(...,
stats_out=...)`. Its backward differentiates the function that forward computed: the forward's
quantized operands and P roundings, straight-through, with the saved row max and normalizer in
place of recomputing them. Gradients are with respect to the BF16 inputs.

- `pv_normalizer=None`: QK-only MXFP8 (BF16 P and V), head dims 64, 128, 256.
- `pv_normalizer="rounded_p" | "fp32"`: native MXFP8 PV with that normalizer (BlockScales).
  - head dims 64 / 128: the mirrored BF16 backward (NOTE [MXFP8 mirrored BF16 backward] in
    flash_bwd_sm100.py) on the forward's dequantized payloads, saved by the forward.
  - head dim 256: the MXFP8 backward (NOTE [MXFP8 hd256 backward]) on the forward-only operands
    the fused prep emits with the forward's payloads (mxfp8_hd256_prep).

Scope: SM100 family, dense, causal or not; the hdim-256 native PV backward needs MHA,
seqlen_q = seqlen_k, and seqlen a multiple of 32 (256 causal). See flash_attn_func for BF16
attention outside it.

Known limit: P is recomputed at the forward's final row max, so blocks the forward visited under
an earlier running max are reproduced only up to that rounding (BlockScales.stats_out). Against
the FP64 derivative of the online forward this dominates native PV's dV error, which is at the
percent level where dQ and dK stay at BF16 level.
"""

from typing import Literal

import torch
from torch.autograd.function import once_differentiable

from flash_attn.cute import mxfp8_hd256_prep
from flash_attn.cute.interface import BlockScales, _flash_attn_bwd, _flash_attn_fwd
from flash_attn.cute.mxfp8_mirror import MirrorOperands, mxfp8_mirror_backward, quantize_for_mirror

PvNormalizer = Literal["rounded_p", "fp32"] | None


def _uses_hd256_backward(q: torch.Tensor, pv_normalizer: PvNormalizer) -> bool:
    return q.shape[-1] == 256 and pv_normalizer is not None


def _check_inputs(q, k, v, causal, pv_normalizer) -> None:
    if any(t.dtype != torch.bfloat16 or t.ndim != 4 for t in (q, k, v)):
        raise ValueError(
            "mxfp8_attn_func takes BF16 (batch, seqlen, heads, head_dim) Q / K / V; use "
            "flash_attn_func for other dtypes"
        )
    if pv_normalizer not in (None, "rounded_p", "fp32"):
        raise ValueError(
            f'pv_normalizer must be "rounded_p", "fp32", or None, got {pv_normalizer!r}'
        )
    if not _uses_hd256_backward(q, pv_normalizer):
        return
    # The backward's config owns this scope (config.mxfp8_mirror_rejection); checking it here
    # fails before the forward runs instead of in the backward, and names the alternatives.
    if k.shape != q.shape or v.shape != q.shape:
        raise NotImplementedError(
            "mxfp8_attn_func with head_dim 256 and native PV needs MHA and seqlen_q = seqlen_k; "
            "use pv_normalizer=None (QK-only MXFP8) or flash_attn_func with BF16"
        )
    if q.shape[1] % (256 if causal else 32) != 0:
        raise NotImplementedError(
            "mxfp8_attn_func with head_dim 256 and native PV needs seqlen a multiple of 32 "
            "(256 when causal); use pv_normalizer=None (QK-only MXFP8) or flash_attn_func with BF16"
        )


def _stats_buffer(q: torch.Tensor) -> torch.Tensor:
    batch, seqlen, heads, _ = q.shape
    return torch.empty(batch, heads, seqlen, 2, dtype=torch.float32, device=q.device)


def _mirror_forward(q, k, v, causal, softmax_scale, pv_normalizer):
    """Quantize once for both directions and run the MXFP8 forward for the BF16 mirror."""
    inputs = quantize_for_mirror(q, k, v, causal, pv_normalizer)
    stats = _stats_buffer(q)
    out = _flash_attn_fwd(
        inputs.q8,
        inputs.k8,
        inputs.v,
        softmax_scale=softmax_scale,
        causal=causal,
        scales=inputs.scales._replace(stats_out=stats),
    )[0]
    return out, stats, inputs.operands


def _hd256_forward(q, k, v, causal, softmax_scale, pv_normalizer):
    """The fused prep and the hdim-256 native PV forward; the prep's forward-only backward
    operands."""
    ops = mxfp8_hd256_prep.quantize_forward(q, k, v, causal)
    stats = _stats_buffer(q)
    scales = BlockScales(
        ops.sfq,
        ops.sfk,
        ops.sfv,
        v if causal else None,
        pv_normalizer=pv_normalizer,
        stats_out=stats,
    )
    out = _flash_attn_fwd(
        ops.q8, ops.k8, ops.v8, softmax_scale=softmax_scale, causal=causal, scales=scales
    )[0]
    return out, stats, ops.bwd


def _hd256_backward(dout, out, stats, fwd_ops, causal, softmax_scale, deterministic):
    operands = mxfp8_hd256_prep.backward_operands(fwd_ops, out, dout, stats, softmax_scale)
    # q / v / dout only give the backward its shapes and dtypes (NOTE [MXFP8 hd256 backward])
    return _flash_attn_bwd(
        out,
        operands.k_hat,
        out,
        out,
        dout,
        None,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
        mx_hd256=operands,
    )[:3]


class MxAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale, pv_normalizer, deterministic):
        q, k, v = (t.detach().contiguous() for t in (q, k, v))
        softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        if _uses_hd256_backward(q, pv_normalizer):
            out, stats, fwd_ops = _hd256_forward(q, k, v, causal, softmax_scale, pv_normalizer)
            ctx.save_for_backward(out, stats, *fwd_ops)
            ctx.mirror_mode = None
        else:
            out, stats, operands = _mirror_forward(q, k, v, causal, softmax_scale, pv_normalizer)
            ctx.save_for_backward(out, stats, *operands[:4])
            ctx.mirror_mode = operands.mode
        ctx.causal, ctx.softmax_scale, ctx.deterministic = causal, softmax_scale, deterministic
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        out, stats, *saved = ctx.saved_tensors
        dout = dout.contiguous()
        if ctx.mirror_mode is None:
            fwd_ops = mxfp8_hd256_prep.ForwardOnlyOperands(*saved)
            dq, dk, dv = _hd256_backward(
                dout, out, stats, fwd_ops, ctx.causal, ctx.softmax_scale, ctx.deterministic
            )
        else:
            dq, dk, dv = mxfp8_mirror_backward(
                dout,
                out,
                MirrorOperands(*saved, ctx.mirror_mode),
                stats,
                ctx.causal,
                ctx.softmax_scale,
                ctx.deterministic,
            )
        return dq, dk, dv, None, None, None, None


def mxfp8_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    pv_normalizer: PvNormalizer = "rounded_p",
    deterministic: bool = False,
) -> torch.Tensor:
    """MXFP8 attention of BF16 (batch, seqlen, heads, head_dim) Q / K / V with the backward that
    mirrors its forward (module docstring).

    pv_normalizer: None runs QK-only MXFP8 (BF16 P and V); "rounded_p" / "fp32" runs native
    MXFP8 PV with that normalizer (BlockScales.pv_normalizer). deterministic: bitwise
    reproducible dQ (the backward's FP32 dQ accumulation in a fixed order).
    """
    _check_inputs(q, k, v, causal, pv_normalizer)
    return MxAttnFunc.apply(q, k, v, causal, softmax_scale, pv_normalizer, deterministic)
