"""The mirrored BF16 backward of the MXFP8 forwards (NOTE [MXFP8 mirrored BF16 backward] in
flash_bwd_sm100.py).

The backward differentiates the function the MXFP8 forward computed: its quantized operands, its P
roundings straight-through, and its saved row max and normalizer (BlockScales.stats_out) in place
of recomputing them, with BF16 GEMMs on the general SM100 backward kernel. It reads the forward's
payloads in BF16, which is exact for E4M3 x 2^k:

- `quantize_for_mirror(q, k, v)` quantizes BF16 inputs once for both directions (mxfp8_quant);
- `mxfp8_mirror_backward(dout, out, operands, stats, causal)` runs the backward.
"""

from typing import NamedTuple

import torch

from flash_attn.cute.config import Mxfp8Mirror
from flash_attn.cute.interface import BlockScales, MxMirrorBwd, _flash_attn_bwd
from flash_attn.cute.mxfp8_quant import quantize_rows


class MirrorOperands(NamedTuple):
    """The forward-only operands of the mirrored backward in BF16: Q-hat, K-hat, and the dP
    operand (V for QK-only). mode is the forward it mirrors."""

    q_hat: torch.Tensor
    k_hat: torch.Tensor
    v_hat: torch.Tensor
    mode: Mxfp8Mirror


class MirrorInputs(NamedTuple):
    """The MXFP8 forward's operands (its payloads and BlockScales, to which the caller adds
    stats_out) and the mirrored backward's, from one quantization pass per input."""

    q8: torch.Tensor
    k8: torch.Tensor
    v: torch.Tensor  # the forward's V: BF16 for QK-only
    scales: BlockScales
    operands: MirrorOperands


def quantize_for_mirror(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> MirrorInputs:
    """QK-only MXFP8 of contiguous BF16 (batch, seqlen, heads, dim) Q / K (V stays BF16)."""
    q8, sfq, q_hat = quantize_rows(q)
    k8, sfk, k_hat = quantize_rows(k)
    operands = MirrorOperands(q_hat, k_hat, v, Mxfp8Mirror.QK)
    return MirrorInputs(q8, k8, v, BlockScales(sfq, sfk), operands)


def mxfp8_mirror_backward(
    dout: torch.Tensor,
    out: torch.Tensor,
    operands: MirrorOperands,
    stats: torch.Tensor,
    causal: bool,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dQ, dK, dV) of the MXFP8 forward that produced out and stats (its BlockScales.stats_out),
    with respect to its BF16 inputs (straight-through quantization)."""
    return _flash_attn_bwd(
        operands.q_hat,
        operands.k_hat,
        operands.v_hat,
        out,
        dout,
        None,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
        mx_mirror=MxMirrorBwd(stats, operands.mode),
    )[:3]
