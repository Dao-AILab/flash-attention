"""QK block-scale layout, mixed K/V ring, and BF16-P numerical contracts."""

import math
import os

import pytest
import torch

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute.blockscaled_utils import (
    dequantize_fp8_with_sf,
    interleave_sf,
    quantize_to_fp8_with_sf,
)
from flash_attn.cute.interface import BlockScales, flash_attn_func


def inputs(batch, sq, sk, hq, hk, dim, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch, sq, hq, dim, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, sk, hk, dim, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch, sk, hk, dim, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    # Different scales across both rows and K-groups catch permuted/duplicated SF bytes.
    q *= torch.tensor([0.5, 2.0, 1.0, 0.25], device="cuda").repeat_interleave(32)[:dim]
    k *= torch.tensor([2.0, 0.25, 0.5, 1.0], device="cuda").repeat_interleave(32)[:dim]
    q8, sq8, qs = quantize_to_fp8_with_sf(q)
    k8, sk8, ks = quantize_to_fp8_with_sf(k)
    return (
        q8,
        k8,
        v,
        interleave_sf(sq8),
        interleave_sf(sk8),
        dequantize_fp8_with_sf(q8, qs),
        dequantize_fp8_with_sf(k8, ks),
    )


def reference(q, k, v, causal):
    k, v = [x.repeat_interleave(q.shape[2] // k.shape[2], dim=2) for x in (k, v)]
    scores = torch.einsum("bqhd,bkhd->bhqk", q.double(), k.double()) / math.sqrt(
        q.shape[-1]
    )
    if causal:
        qp = torch.arange(q.shape[1], device=q.device) + k.shape[1] - q.shape[1]
        mask = torch.arange(k.shape[1], device=q.device)[None, :] > qp[:, None]
        scores.masked_fill_(mask, -math.inf)
    lse = torch.logsumexp(scores, -1)
    probs = torch.softmax(scores, -1).nan_to_num()
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.double())
    eager = torch.einsum(
        "bhqk,bkhd->bqhd", probs.to(torch.bfloat16).double(), v.double()
    ).to(torch.bfloat16)
    return out, lse, eager


@pytest.mark.parametrize(
    "dim,causal,sq,sk,hq,hk",
    [
        (64, False, 128, 128, 2, 2),
        (64, True, 257, 385, 4, 2),
        (128, False, 257, 385, 4, 1),
        (128, True, 257, 385, 2, 2),
        (64, False, 257, 385, 4, 1),
        (64, True, 128, 128, 2, 2),
        (128, False, 128, 128, 2, 2),
        (128, True, 257, 129, 4, 1),
        (128, True, 513, 129, 4, 1),  # The first two-stage Q tile is entirely masked.
    ],
)
def test_mxfp8_qk(dim, causal, sq, sk, hq, hk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    q, k, v, sfq, sfk, qd, kd = inputs(2, sq, sk, hq, hk, dim)
    expected, lse_ref, eager = reference(qd, kd, v, causal)
    for _ in range(2):
        actual, lse = flash_attn_func(
            q, k, v, scales=BlockScales(sfq, sfk), causal=causal, return_lse=True
        )
        error = torch.linalg.vector_norm(actual.double() - expected) / expected.norm()
        eager_error = (
            torch.linalg.vector_norm(eager.double() - expected) / expected.norm()
        )
        assert torch.isfinite(actual).all()
        if causal and sq > sk:
            assert torch.count_nonzero(actual[:, : sq - sk]) == 0
            assert torch.isneginf(lse[..., : sq - sk]).all()
        # BF16 P and BF16 O roundings plus online accumulation: compare to the eager budget.
        assert error <= eager_error + 2e-3, (error.item(), eager_error.item())
        assert (
            actual.double() - expected
        ).abs().max() <= expected.abs().max() * 0.015 + 1e-5
        torch.testing.assert_close(lse, lse_ref.float(), atol=2e-4, rtol=2e-5)


@pytest.mark.skipif(
    os.environ.get("FA_RUN_MXFP8_STRESS") != "1",
    reason="opt-in persistent/replay stress",
)
@pytest.mark.parametrize("scale_operand", ["q", "k"])
def test_mxfp8_qk_persistent_replay(scale_operand):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")
    # 1024 logical Q tiles: several per CTA, so persistent stage reuse is exercised.
    # Seventeen KV blocks exercise odd phase flips across persistent work tiles.
    q, k, v, sfq, sfk, qd, kd = inputs(4, 2048, 2049, 32, 32, 128, seed=7)
    starts = torch.arange(0, 2048, 128, device=q.device)
    rows = torch.stack((starts, starts + 127), dim=1).flatten()
    scales = {"q": sfq, "k": sfk}
    # A finite E8M0 exponent increment doubles just this operand; payload bytes stay fixed.
    scaled_sf = (scales[scale_operand].view(torch.uint8) + 1).view(torch.float8_e8m0fnu)
    qd2, kd2 = (qd * 2, kd) if scale_operand == "q" else (qd, kd * 2)
    # Every batch/head and both Q stages of every tile, not just the first scheduled work.
    references = [reference(qd[:, rows], kd, v, False), reference(qd2[:, rows], kd2, v, False)]
    for i in range(10):
        # Each pair changes only one scale binding; successive pairs flip V's sign.
        doubled = i % 2 == 1
        sign = 1 if (i // 2) % 2 == 0 else -1
        current = {**scales, scale_operand: scaled_sf} if doubled else scales
        actual, _ = flash_attn_func(
            q, k, v if sign == 1 else -v, scales=BlockScales(current["q"], current["k"])
        )
        assert torch.isfinite(actual).all()
        actual = actual[:, rows].double() * sign
        expected, _, eager = references[doubled]
        # Bound each batch/head separately so earlier work cannot dilute a later-tile failure.
        ref_norm = expected.square().sum(dim=(1, 3)).sqrt()
        error = (actual - expected).square().sum(dim=(1, 3)).sqrt() / ref_norm
        eager_error = (eager.double() - expected).square().sum(
            dim=(1, 3)
        ).sqrt() / ref_norm
        assert (error <= eager_error + 2e-3).all()
