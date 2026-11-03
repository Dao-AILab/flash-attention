"""hd256 MXFP8 forward (2CTA, S ping-pong) with BF16 V (QK-only)."""

import pytest
import torch

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute.blockscaled_utils import (
    dequantize_fp8_with_sf,
    interleave_sf,
    quantize_to_fp8_with_sf,
)
from flash_attn.cute.interface import BlockScales, flash_attn_func
import test_mxfp8_pv
import test_mxfp8_qk

DIM = 256


def requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")


def qk_inputs(batch, sq, sk, hq, hk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(batch, sq, hq, DIM, generator=g, device="cuda")
    k = torch.randn(batch, sk, hk, DIM, generator=g, device="cuda")
    v = torch.randn(batch, sk, hk, DIM, generator=g, device="cuda", dtype=torch.bfloat16)
    # Different scales across all eight 32-channel groups of both K chunks.
    groups = torch.tensor([0.5, 2.0, 1.0, 0.25, 4.0, 0.125, 1.0, 2.0], device="cuda")
    q *= groups.repeat_interleave(32)
    k *= groups.flip(0).repeat_interleave(32)
    q8, sfq, qs = quantize_to_fp8_with_sf(q.bfloat16())
    k8, sfk, ks = quantize_to_fp8_with_sf(k.bfloat16())
    return (q8, k8, v, interleave_sf(sfq), interleave_sf(sfk)), (
        dequantize_fp8_with_sf(q8, qs),
        dequantize_fp8_with_sf(k8, ks),
    )


@pytest.mark.parametrize(
    "causal,sq,sk,hq,hk",
    [
        (False, 128, 128, 2, 2),
        (False, 257, 385, 4, 1),
        (True, 257, 385, 4, 2),
        (True, 513, 129, 4, 1),  # leading 256-row work tiles entirely masked
        (False, 100, 1000, 2, 2),  # one partly empty work tile
    ],
)
def test_mxfp8_hd256_qk(causal, sq, sk, hq, hk):
    requires_sm100()
    (q, k, v, sfq, sfk), (qd, kd) = qk_inputs(2, sq, sk, hq, hk)
    expected, lse_ref, eager = test_mxfp8_qk.reference(qd, kd, v, causal)
    for _ in range(2):
        actual, lse = flash_attn_func(
            q, k, v, scales=BlockScales(sfq, sfk), causal=causal, return_lse=True
        )
        error = (actual.double() - expected).norm() / expected.norm()
        eager_error = (eager.double() - expected).norm() / expected.norm()
        assert torch.isfinite(actual).all()
        if causal and sq > sk:
            assert torch.count_nonzero(actual[:, : sq - sk]) == 0
            assert torch.isneginf(lse[..., : sq - sk]).all()
        # BF16 P and BF16 O roundings plus online accumulation: compare to the eager budget.
        assert error <= eager_error + 2e-3, (error.item(), eager_error.item())
        torch.testing.assert_close(lse, lse_ref.float(), atol=2e-4, rtol=2e-5)


def per_head_error(actual, expected):
    """Relative L2 error per (batch, head) of (batch, seqlen, heads, dim) outputs."""
    error = (actual.double() - expected).square().sum(dim=(1, 3)).sqrt()
    return error / expected.square().sum(dim=(1, 3)).sqrt()


def test_mxfp8_hd256_persistent_tiles():
    """Many work tiles per CTA (noncausal persistent): sampled rows of every tile match."""
    requires_sm100()
    (q, k, _, sfq, sfk, _), (qd, kd, vd) = test_mxfp8_pv.inputs(4, 2048, 2049, 32, 8, DIM)
    starts = torch.arange(0, 2048, 128, device=q.device)
    rows = torch.stack((starts, starts + 127), dim=1).flatten()
    v = vd.bfloat16()
    expected, _, eager = test_mxfp8_qk.reference(qd[:, rows], kd, v, causal=False)
    # BF16 P and O roundings: compare to the eager (BF16 P) reference's own error.
    head_bound = per_head_error(eager, expected) + 2e-3
    total_bound = (eager.double() - expected).norm() / expected.norm() + 2e-3
    for _ in range(3):
        actual, _ = flash_attn_func(q, k, v, scales=BlockScales(sfq, sfk))
        sampled = actual[:, rows].double()
        assert (per_head_error(sampled, expected) <= head_bound).all()
        assert (sampled - expected).norm() / expected.norm() <= total_bound
