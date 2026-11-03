"""Native PV: sequence-axis scales, quantized-P normalization, and persistent reuse."""

import math
import os

import pytest
import torch

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute.blockscaled_utils import (
    dequantize_fp8_with_sf,
    dequantize_v_fp8_with_sf,
    interleave_sf,
    quantize_to_fp8_with_sf,
    quantize_v_to_fp8_with_sf,
)
from flash_attn.cute.interface import BlockScales, flash_attn_func


def inputs(batch, sq, sk, hq, hk, dim, dim_v=None):
    dim_v = dim if dim_v is None else dim_v
    g = torch.Generator(device="cuda").manual_seed(41)
    q = torch.randn(batch, sq, hq, dim, generator=g, device="cuda")
    k = torch.randn(batch, sk, hk, dim, generator=g, device="cuda")
    v = torch.randn(batch, sk, hk, dim_v, generator=g, device="cuda")
    # Nonuniform in every scale dimension, including consecutive 32-key groups.
    v *= torch.exp2((torch.arange(sk, device="cuda") // 32 % 4 - 2).float())[
        None, :, None, None
    ]
    v *= torch.exp2((torch.arange(dim_v, device="cuda") % 3 - 1).float())
    q8, sfq, qs = quantize_to_fp8_with_sf(q)
    k8, sfk, ks = quantize_to_fp8_with_sf(k)
    v8, sfv, vs = quantize_v_to_fp8_with_sf(v)
    return (q8, k8, v8, interleave_sf(sfq), interleave_sf(sfk), sfv), (
        dequantize_fp8_with_sf(q8, qs),
        dequantize_fp8_with_sf(k8, ks),
        dequantize_v_fp8_with_sf(v8, vs),
    )


def reference(q, k, v, causal=False, v_bf16=None):
    """Independent reverse-block online attention with e4m3 P and FP32 statistics.

    Causal attention runs every KV block that intersects a query tile's diagonal (128 rows, or
    256 when seqlen_q > 128) with BF16 P and v_bf16, as the kernel does, and the fully visible
    blocks with E4M3 P and v. O is normalized by the sum of the P each PV consumed.
    """
    rep = q.shape[2] // k.shape[2]
    k, v = [x.repeat_interleave(rep, dim=2).transpose(1, 2).double() for x in (k, v)]
    q = q.transpose(1, 2).double()
    logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    seqlen_q, seqlen_k = q.shape[-2], k.shape[-2]
    if causal:
        v_bf16 = v_bf16.repeat_interleave(rep, dim=2).transpose(1, 2).double()
        rows = torch.arange(seqlen_q, device=q.device)[:, None] + seqlen_k - seqlen_q
        masked = torch.arange(seqlen_k, device=q.device)[None, :] > rows
        logits = logits.masked_fill(masked, -math.inf)
        tile_rows = 128 if seqlen_q <= 128 else 256
        tile_start = torch.arange(seqlen_q, device=q.device) // tile_rows * tile_rows
        # Block n is BF16 for a row iff n >= (tile_start + seqlen_k - seqlen_q) // 128.
        first_bf16_block = (tile_start + seqlen_k - seqlen_q) // 128
    ref64 = (torch.softmax(logits, -1).nan_to_num() @ v).transpose(1, 2)
    maximum = torch.full(logits.shape[:-1], -math.inf, device=q.device)
    rowsum = torch.zeros_like(maximum)
    out = torch.zeros((*maximum.shape, v.shape[-1]), device=q.device)
    # FP32 accumulator values rounded at the QK output, as in the device MMA.
    scores = (q @ k.transpose(-1, -2)).float()
    if causal:
        scores = scores.masked_fill(masked, -math.inf)
    scale_log2 = torch.tensor(
        math.log2(math.e) / math.sqrt(q.shape[-1]), device=q.device
    )
    for start in reversed(range(0, seqlen_k, 128)):
        s = scores[..., start : start + 128]
        new_max = torch.maximum(maximum, s.amax(-1))
        # Rows with no visible key yet (causal) use max 0, as the kernel does.
        safe_max = new_max.masked_fill(new_max == -math.inf, 0.0)
        alpha = torch.exp2((maximum.masked_fill(maximum == -math.inf, 0.0) - safe_max) * scale_log2)
        alpha = alpha.masked_fill(maximum == -math.inf, 0.0)
        bias = 8 - safe_max * scale_log2
        # Round the affine transform once, matching the packed FFMA.
        p32 = torch.exp2(
            (s.double() * scale_log2.double() + bias.double()[..., None]).float()
        )
        p = p32.to(torch.float8_e4m3fn).float()
        pv = p @ v[..., start : start + 128, :].float()
        psum = p.sum(-1)
        if causal:
            bf16 = (start // 128 >= first_bf16_block)[:, None]  # per query row
            p16 = p32.to(torch.bfloat16).float()
            pv = torch.where(bf16, p16 @ v_bf16[..., start : start + 128, :].float(), pv)
            psum = torch.where(bf16[..., 0], p16.sum(-1), psum)
        out = out * alpha[..., None] + pv
        rowsum = rowsum * alpha + psum
        maximum = new_max
    expected = (out / rowsum[..., None]).nan_to_num().transpose(1, 2)
    return expected, ref64, torch.logsumexp(logits, -1)


@pytest.mark.parametrize(
    "dim,dim_v,sq,sk,hq,hk",
    [
        (128, 128, 128, 128, 2, 2),
        (128, 128, 257, 385, 4, 1),
        (64, 64, 257, 385, 4, 2),
        (64, 128, 257, 385, 4, 1),
        (128, 64, 257, 385, 4, 1),
    ],
)
def test_mxfp8_pv(dim, dim_v, sq, sk, hq, hk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    (q, k, v, sfq, sfk, sfv), deq = inputs(2, sq, sk, hq, hk, dim, dim_v)
    expected, fp64, lse_ref = reference(*deq)
    actual, lse = flash_attn_func(q, k, v, scales=BlockScales(sfq, sfk, sfv), return_lse=True)
    assert torch.isfinite(actual).all()
    # 2e-3: BF16 output rounding (2^-9 per element, about 1.1e-3 RMS) plus FP32 summation order.
    assert (actual.float() - expected).norm() / expected.norm() <= 2e-3
    assert (actual.double() - fp64).norm() <= (
        expected.double() - fp64
    ).norm() + 2e-3 * fp64.norm()
    torch.testing.assert_close(lse, lse_ref.float(), atol=2e-4, rtol=2e-5)
    ones, sfones, _ = quantize_v_to_fp8_with_sf(torch.ones_like(deq[2]))
    normalized, _ = flash_attn_func(q, k, ones, scales=BlockScales(sfq, sfk, sfones))
    torch.testing.assert_close(
        normalized.float(), torch.ones_like(normalized.float()), atol=1e-5, rtol=0
    )


@pytest.mark.skipif(
    os.environ.get("FA_RUN_MXFP8_STRESS") != "1",
    reason="opt-in persistent/replay stress",
)
@pytest.mark.parametrize("scale_operand", ["q", "k", "v"])
def test_mxfp8_pv_persistent_replay(scale_operand):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    # 1024 logical Q tiles: >4 per physical CTA; 17 KV blocks wrap odd phases.
    (q, k, v, sfq, sfk, sfv), (qd, kd, vd) = inputs(4, 2048, 2049, 32, 4, 128)
    starts = torch.arange(0, 2048, 128, device=q.device)
    rows = torch.stack((starts, starts + 127), dim=1).flatten()
    expected, _, _ = reference(qd[:, rows], kd, vd)
    scales = [sfq, sfk, sfv]
    scale_index = ("q", "k", "v").index(scale_operand)
    # A finite E8M0 exponent increment doubles just this operand; payload bytes stay fixed.
    scaled_sf = (scales[scale_index].view(torch.uint8) + 1).view(scales[scale_index].dtype)
    if scale_operand == "v":
        expected_scaled = expected * 2
    else:
        scaled_deq = [qd[:, rows], kd, vd]
        scaled_deq[scale_index] = scaled_deq[scale_index] * 2
        expected_scaled, _, _ = reference(*scaled_deq)
    negative = (-v.float()).to(v.dtype)
    for i in range(10):
        doubled = i % 2 == 1
        sign = 1 if (i // 2) % 2 == 0 else -1
        current_scales = scales.copy()
        if doubled:
            current_scales[scale_index] = scaled_sf
        # Each pair changes only scales; the next pair also exercises the opposite V sign.
        actual, _ = flash_attn_func(
            q, k, v if sign == 1 else negative, scales=BlockScales(*current_scales)
        )
        sampled = actual[:, rows].double() * sign
        target = expected_scaled if doubled else expected
        error = (sampled - target).square().sum(dim=(1, 3)).sqrt()
        assert (error / target.square().sum(dim=(1, 3)).sqrt() <= 2e-3).all()


@pytest.mark.parametrize(
    "dim,sq,sk,hq,hk",
    [
        (64, 257, 386, 4, 2),  # unaligned seqlen_k - seqlen_q, GQA
        (128, 385, 129, 2, 2),  # leading query tile fully masked
        (128, 1000, 2049, 4, 1),  # 17 KV blocks, unaligned offset
        (128, 128, 300, 2, 2),  # one Q stage (128-row tile)
    ],
)
def test_mxfp8_pv_causal_no_future_leakage(dim, sq, sk, hq, hk):
    """Scaling V at future positions must leave every earlier row bitwise unchanged."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(1, sq, hq, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, sk, hk, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, sk, hk, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    q8, sfq, qs = quantize_to_fp8_with_sf(q)
    k8, sfk, ks = quantize_to_fp8_with_sf(k)
    sfq, sfk = interleave_sf(sfq), interleave_sf(sfk)

    def run(v_bf16):
        v8, sfv, vs = quantize_v_to_fp8_with_sf(v_bf16)
        out, lse = flash_attn_func(
            q8, k8, v8, scales=BlockScales(sfq, sfk, sfv, v_bf16=v_bf16), causal=True,
            return_lse=True,
        )
        return out, lse, dequantize_v_fp8_with_sf(v8, vs)

    out, lse, v_deq = run(v)
    assert torch.isfinite(out).all()
    expected, fp64, lse_ref = reference(
        dequantize_fp8_with_sf(q8, qs), dequantize_fp8_with_sf(k8, ks), v_deq, causal=True,
        v_bf16=v,
    )
    # Same recipe as the kernel (E4M3 P and MXFP8 V off the diagonal, BF16 on it), so the only
    # slack is BF16 output rounding and FP32 summation order; same bound as the non-causal test.
    assert (out.float() - expected).norm() / expected.norm() <= 2e-3
    assert (out.double() - fp64).norm() <= (expected.double() - fp64).norm() + 2e-3 * fp64.norm()
    torch.testing.assert_close(lse, lse_ref.float(), atol=2e-4, rtol=2e-5, equal_nan=True)
    # O is normalized by the sum of the P each PV consumed (BF16 or E4M3), so V = 1 gives 1 on
    # every row that sees a key.
    normalized, _, _ = run(torch.ones_like(v))
    visible = normalized[:, max(sq - sk, 0):].float()
    torch.testing.assert_close(visible, torch.ones_like(visible), atol=1e-5, rtol=0)
    for t0 in (sk // 4 + 17, sk // 2 + 5, 3 * sk // 4 + 29):
        rows = t0 - (sk - sq)  # rows [0, rows) attend only keys < t0
        if rows <= 0 or t0 >= sk:
            continue
        for factor in (16.0, 1024.0):
            v2 = v.clone()
            v2[:, t0:] = (v2[:, t0:].float() * factor).to(torch.bfloat16)
            out2, lse2, _ = run(v2)
            assert torch.equal(out2[:, :rows].view(torch.uint8), out[:, :rows].view(torch.uint8))
            assert torch.equal(lse2[..., :rows].view(torch.uint8), lse[..., :rows].view(torch.uint8))
