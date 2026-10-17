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
    return (q8, k8, v8, interleave_sf(sfq, 32), interleave_sf(sfk, 32), sfv), (
        dequantize_fp8_with_sf(q8, qs),
        dequantize_fp8_with_sf(k8, ks),
        dequantize_v_fp8_with_sf(v8, vs),
    )


def reference(q, k, v, pv_softmax="exact"):
    """Independent reverse-block online attention with e4m3 P and FP32 statistics.

    "exact" sums the E4M3 P with a 2^8 bias. "fast" sums the FP32 P with a 2^4 bias and keeps
    the row max until a block raises it by more than 4 (log2 units).
    """
    fast = pv_softmax == "fast"
    offset, threshold = (4, 4.0) if fast else (8, 0.0)
    rep = q.shape[2] // k.shape[2]
    k, v = [x.repeat_interleave(rep, dim=2).transpose(1, 2).double() for x in (k, v)]
    q = q.transpose(1, 2).double()
    logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    ref64 = (torch.softmax(logits, -1) @ v).transpose(1, 2)
    maximum = torch.full(logits.shape[:-1], -math.inf, device=q.device)
    rowsum = torch.zeros_like(maximum)
    out = torch.zeros((*maximum.shape, v.shape[-1]), device=q.device)
    # FP32 accumulator values rounded at the QK output, as in the device MMA.
    scores = (q @ k.transpose(-1, -2)).float()
    scale_log2 = torch.tensor(
        math.log2(math.e) / math.sqrt(q.shape[-1]), device=q.device
    )
    for start in reversed(range(0, k.shape[-2], 128)):
        s = scores[..., start : start + 128]
        new_max = torch.maximum(maximum, s.amax(-1))
        new_max = torch.where((maximum - new_max) * scale_log2 >= -threshold, maximum, new_max)
        alpha = torch.exp2((maximum - new_max) * scale_log2)
        bias = offset - new_max * scale_log2
        # Round the affine transform once, matching the packed FFMA.
        p32 = torch.exp2(
            (s.double() * scale_log2.double() + bias.double()[..., None]).float()
        )
        p = p32.to(torch.float8_e4m3fn).float()
        out = out * alpha[..., None] + p @ v[..., start : start + 128, :].float()
        rowsum = rowsum * alpha + (p32 if fast else p).sum(-1)
        maximum = new_max
    expected = (out / rowsum[..., None]).transpose(1, 2)
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
@pytest.mark.parametrize("pv_softmax", ["exact", "fast"])
def test_mxfp8_pv(dim, dim_v, sq, sk, hq, hk, pv_softmax):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    (q, k, v, sfq, sfk, sfv), deq = inputs(2, sq, sk, hq, hk, dim, dim_v)
    expected, fp64, lse_ref = reference(*deq, pv_softmax=pv_softmax)
    actual, lse = flash_attn_func(
        q, k, v, scales=BlockScales(sfq, sfk, sfv), return_lse=True, mxfp8_pv_softmax=pv_softmax
    )
    assert torch.isfinite(actual).all()
    assert (actual.float() - expected).norm() / expected.norm() <= 2e-3
    assert (actual.double() - fp64).norm() <= (
        expected.double() - fp64
    ).norm() + 2e-3 * fp64.norm()
    torch.testing.assert_close(lse, lse_ref.float(), atol=2e-4, rtol=2e-5)
    ones, sfones, _ = quantize_v_to_fp8_with_sf(torch.ones_like(deq[2]))
    normalized, _ = flash_attn_func(
        q, k, ones, scales=BlockScales(sfq, sfk, sfones), mxfp8_pv_softmax=pv_softmax
    )
    # "exact" divides by the sum of the P that PV consumed. "fast" divides by the FP32 sum, so
    # each row is off by at most the E4M3 rounding of P (2^-4 relative).
    atol = 1e-5 if pv_softmax == "exact" else 2**-4
    torch.testing.assert_close(
        normalized.float(), torch.ones_like(normalized.float()), atol=atol, rtol=0
    )


@pytest.mark.skipif(
    os.environ.get("FA_RUN_MXFP8_STRESS") != "1",
    reason="opt-in persistent/replay stress",
)
def test_mxfp8_pv_persistent_replay():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    # 1024 logical Q tiles: >4 per physical CTA; 17 KV blocks wrap odd phases.
    (q, k, v, sfq, sfk, sfv), (qd, kd, vd) = inputs(4, 2048, 2049, 32, 4, 128)
    starts = torch.arange(0, 2048, 128, device=q.device)
    rows = torch.stack((starts, starts + 127), dim=1).flatten()
    expected, _, _ = reference(qd[:, rows], kd, vd)
    negative = (-v.float()).to(v.dtype)
    for i in range(10):
        actual, _ = flash_attn_func(
            q, k, v if i % 2 == 0 else negative, scales=BlockScales(sfq, sfk, sfv)
        )
        sampled = actual[:, rows].double() * (1 if i % 2 == 0 else -1)
        error = (sampled - expected).square().sum(dim=(1, 3)).sqrt()
        assert (error / expected.square().sum(dim=(1, 3)).sqrt() <= 2e-3).all()


def causal_reference(q, k, v, causal_offset):
    """FP64 causal attention on the dequantized operands (sanity bound, not bitwise)."""
    rep = q.shape[2] // k.shape[2]
    k, v = [x.repeat_interleave(rep, dim=2).transpose(1, 2).double() for x in (k, v)]
    q = q.transpose(1, 2).double()
    logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    rows = torch.arange(q.shape[-2], device=q.device)[:, None] + causal_offset
    logits = logits.masked_fill(torch.arange(k.shape[-2], device=q.device)[None, :] > rows, -math.inf)
    return (torch.softmax(logits, -1).nan_to_num() @ v).transpose(1, 2)


@pytest.mark.parametrize(
    "dim,sq,sk,hq,hk",
    [
        (64, 257, 386, 4, 2),  # unaligned seqlen_k - seqlen_q, GQA
        (128, 385, 129, 2, 2),  # leading query tile fully masked
        (128, 1000, 2049, 4, 1),  # 17 KV blocks, unaligned offset
    ],
)
@pytest.mark.parametrize("pv_softmax", ["exact", "fast"])
def test_mxfp8_pv_causal_no_future_leakage(dim, sq, sk, hq, hk, pv_softmax):
    """Scaling V at future positions must leave every earlier row bitwise unchanged."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(1, sq, hq, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, sk, hk, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, sk, hk, dim, generator=g, device="cuda", dtype=torch.bfloat16)
    q8, sfq, qs = quantize_to_fp8_with_sf(q)
    k8, sfk, ks = quantize_to_fp8_with_sf(k)
    sfq, sfk = interleave_sf(sfq, 32), interleave_sf(sfk, 32)

    def run(v_bf16):
        v8, sfv, _ = quantize_v_to_fp8_with_sf(v_bf16)
        return flash_attn_func(
            q8, k8, v8, scales=BlockScales(sfq, sfk, sfv), v_bf16=v_bf16, causal=True,
            return_lse=True, mxfp8_pv_softmax=pv_softmax,
        )

    out, lse = run(v)
    assert torch.isfinite(out).all()
    fp64 = causal_reference(
        dequantize_fp8_with_sf(q8, qs), dequantize_fp8_with_sf(k8, ks), v.float(), sk - sq
    )
    # E4M3 P / MXFP8 V error on fully visible blocks; BF16 on the diagonal.
    assert (out.double() - fp64).norm() <= 6e-2 * fp64.norm()
    for t0 in (sk // 4 + 17, sk // 2 + 5, 3 * sk // 4 + 29):
        rows = t0 - (sk - sq)  # rows [0, rows) attend only keys < t0
        if rows <= 0 or t0 >= sk:
            continue
        for factor in (16.0, 1024.0):
            v2 = v.clone()
            v2[:, t0:] = (v2[:, t0:].float() * factor).to(torch.bfloat16)
            out2, lse2 = run(v2)
            assert torch.equal(out2[:, :rows].view(torch.uint8), out[:, :rows].view(torch.uint8))
            assert torch.equal(lse2[..., :rows].view(torch.uint8), lse[..., :rows].view(torch.uint8))
