"""mxfp8_attn_func (flash_attn/cute/mxfp8_attention.py): trainable MXFP8 attention whose
backward differentiates the MXFP8 forward it ran."""

import math

import pytest
import torch

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute import mxfp8_attn_func  # noqa: E402
from flash_attn.cute.blockscaled_utils import (  # noqa: E402
    dequantize_fp8_with_sf,
    dequantize_v_fp8_with_sf,
    interleave_sf,
    quantize_to_fp8_with_sf,
    quantize_v_to_fp8_with_sf,
)
from flash_attn.cute.interface import BlockScales, flash_attn_func  # noqa: E402

from mxfp8_mirror_reference import autograd_backward, ideal_backward, rel_l2  # noqa: E402

NORMALIZERS = [None, "rounded_p", "fp32"]


@pytest.fixture(autouse=True)
def requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("requires SM100 / SM110")


def make_inputs(b, s, hq, hk, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, s, hq, d, generator=g, device="cuda")
    k = torch.randn(b, s, hk, d, generator=g, device="cuda")
    v = torch.randn(b, s, hk, d, generator=g, device="cuda")
    q = q * torch.exp(0.5 * torch.randn(b, s, hq, 1, generator=g, device="cuda"))
    v = v * torch.exp2((torch.arange(s, device="cuda") // 32 % 4 - 2).float())[None, :, None, None]
    dout = torch.randn(b, s, hq, d, generator=g, device="cuda")
    return [t.bfloat16() for t in (q, k, v, dout)]


def quantized_operands(q, k, v, pv_normalizer):
    """The values the MXFP8 forward consumes: Q-hat, K-hat, and V-hat (native PV) or V."""
    (q8, _, sq), (k8, _, sk) = quantize_to_fp8_with_sf(q), quantize_to_fp8_with_sf(k)
    vh = v
    if pv_normalizer is not None:
        v8, _, sv = quantize_v_to_fp8_with_sf(v)
        vh = dequantize_v_fp8_with_sf(v8, sv)
    return dequantize_fp8_with_sf(q8, sq), dequantize_fp8_with_sf(k8, sk), vh


def straight_through_reference(q, k, v, dout, out, causal, pv_normalizer, row_max):
    """FP64 torch.autograd of the quantized forward at the final row max (row_max: the forward's
    (batch, heads, seqlen) score maxima), straight through every rounding: dQ / dK / dV with
    respect to the BF16 inputs, per (batch, query head)."""
    qh, kh, vh = quantized_operands(q, k, v, pv_normalizer)
    rep = q.shape[2] // k.shape[2]
    grads = [torch.zeros(t.shape, dtype=torch.float64, device="cuda") for t in (q, k, v)]
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            g = h // rep
            (dq, dk, dv), _, _ = autograd_backward(
                qh[b, :, h], kh[b, :, g], vh[b, :, g], v[b, :, g], dout[b, :, h],
                row_max[b, h], q.shape[-1] ** -0.5, causal=causal,
                native_pv=pv_normalizer is not None, pv_normalizer=pv_normalizer or "fp32",
            )
            grads[0][b, :, h] = dq
            grads[1][b, :, g] += dk
            grads[2][b, :, g] += dv
    return grads


def forward_row_max(q, k, v, causal, pv_normalizer):
    """The plain MXFP8 forward with stats_out on the same inputs: (O, final row max m)."""
    (q8, sfq, _), (k8, sfk, _) = quantize_to_fp8_with_sf(q), quantize_to_fp8_with_sf(k)
    stats = torch.empty(q.shape[0], q.shape[2], q.shape[1], 2, device="cuda")
    if pv_normalizer is None:
        v_fwd, scales = v, BlockScales(interleave_sf(sfq), interleave_sf(sfk), stats_out=stats)
    else:
        v_fwd, sfv, _ = quantize_v_to_fp8_with_sf(v)
        scales = BlockScales(interleave_sf(sfq), interleave_sf(sfk), sfv, v if causal else None,
                             pv_normalizer=pv_normalizer, stats_out=stats)
    out, _ = flash_attn_func(q8, k8, v_fwd, causal=causal, scales=scales)
    return out, stats[..., 0]


# (batch, seqlen, heads_q, heads_kv, head_dim); hdim 256 native PV runs the MXFP8 backward
# (MHA, seqlen a multiple of 256), everything else the mirrored BF16 backward
CASES = [(2, 384, 4, 2, 64), (1, 512, 2, 2, 128), (1, 512, 2, 2, 256), (1, 384, 4, 2, 256)]


@pytest.mark.parametrize("shape", CASES, ids=lambda s: "b{}_s{}_h{}x{}_d{}".format(*s))
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pv_normalizer", NORMALIZERS)
def test_gradients_are_the_quantized_forwards(shape, causal, pv_normalizer):
    """mxfp8_attn_func's forward is the MXFP8 forward, and its gradients are the derivative of
    that forward (FP64 autograd at the final max, straight through the roundings) to BF16-class
    error: the BF16 mirror rounds dO / L and dS to BF16 (bound 1.5e-2); hdim 256 native PV also
    rounds dO'' and dS'' to MXFP8 for its block-scaled GEMMs (6e-2). Both are empirical,
    2x above B200. The ideal FP64 gradient is reported, loosely bounded by the MXFP8 forward."""
    b, s, hq, hk, d = shape
    if d == 256 and pv_normalizer is not None and hq != hk:
        pytest.skip("hdim 256 native PV backward is MHA only")
    q, k, v, dout = make_inputs(b, s, hq, hk, d)
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = mxfp8_attn_func(*leaves, causal=causal, pv_normalizer=pv_normalizer)
    out.backward(dout)
    out_ref, row_max = forward_row_max(q, k, v, causal, pv_normalizer)
    assert torch.equal(out.detach().view(torch.int16), out_ref.view(torch.int16))
    reference = straight_through_reference(q, k, v, dout, out_ref, causal, pv_normalizer, row_max)
    ideal = ideal_backward(q, k, v, dout, d**-0.5, causal)
    bound = 6e-2 if d == 256 and pv_normalizer is not None else 1.5e-2
    errors = {
        name: (rel_l2(t.grad, r), rel_l2(t.grad, i))
        for name, t, r, i in zip("qkv", leaves, reference, ideal)
    }
    for name, (vs_reference, vs_ideal) in errors.items():
        assert torch.isfinite(leaves["qkv".index(name)].grad).all()
        assert vs_reference <= bound, f"d{name}: vs straight-through {vs_reference:.2e} ({errors})"
        assert vs_ideal <= 0.2, f"d{name}: vs ideal {vs_ideal:.2e} ({errors})"


@pytest.mark.parametrize("d", [128, 256])
def test_deterministic_gradients_repeat(d):
    q, k, v, dout = make_inputs(1, 512, 2, 2, d, seed=3)
    grads = []
    for _ in range(2):
        leaves = [t.clone().requires_grad_() for t in (q, k, v)]
        mxfp8_attn_func(*leaves, causal=True, deterministic=True).backward(dout)
        grads.append([t.grad for t in leaves])
    for a, b in zip(*grads):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_unsupported_inputs_name_the_alternative():
    q, k, v, _ = make_inputs(1, 384, 4, 2, 256)
    with pytest.raises(NotImplementedError, match="pv_normalizer=None .* or flash_attn_func"):
        mxfp8_attn_func(q, k, v)  # hdim 256 native PV: MHA only
    with pytest.raises(NotImplementedError, match="multiple of 32"):
        mxfp8_attn_func(*(t[:, :, :2][:, :300].contiguous() for t in (q, q, q)))
    with pytest.raises(ValueError, match="flash_attn_func"):
        mxfp8_attn_func(*(t.half() for t in (q, k, v)))
    with pytest.raises(ValueError, match="pv_normalizer"):
        mxfp8_attn_func(q, k, v, pv_normalizer="exact")
    assert math.isfinite(mxfp8_attn_func(q, k, v, pv_normalizer=None).float().sum().item())
