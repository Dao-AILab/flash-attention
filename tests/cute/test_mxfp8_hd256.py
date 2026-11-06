"""hd256 MXFP8 forward (2CTA, S ping-pong): QK-only, native PV, causal leakage, stats_out."""

import math

import pytest
import torch

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute.blockscaled_utils import (
    dequantize_fp8_with_sf,
    interleave_sf,
    quantize_to_fp8_with_sf,
    quantize_v_to_fp8_with_sf,
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


@pytest.mark.parametrize(
    "causal,sq,sk,hq,hk",
    [
        (False, 128, 128, 2, 2),
        (False, 257, 385, 4, 1),
        (True, 257, 385, 4, 1),
        (True, 513, 129, 4, 1),
        (False, 100, 1000, 2, 2),
        (False, 256, 256, 2, 2),  # two KV blocks: the first step sums its own P
        (False, 300, 384, 2, 1),  # three KV blocks: one sum from the ones MMA
    ],
)
@pytest.mark.parametrize("pv_softmax", ["exact", "fast"])
def test_mxfp8_hd256_pv(causal, sq, sk, hq, hk, pv_softmax):
    requires_sm100()
    (q, k, v, sfq, sfk, sfv), deq = test_mxfp8_pv.inputs(2, sq, sk, hq, hk, DIM)
    v_bf16 = deq[2].bfloat16() if causal else None
    expected, fp64, lse_ref = test_mxfp8_pv.reference(
        *deq, pv_softmax=pv_softmax, causal=causal, v_bf16=v_bf16
    )
    actual, lse = flash_attn_func(
        q, k, v, scales=BlockScales(sfq, sfk, sfv, v_bf16, pv_softmax=pv_softmax),
        causal=causal, return_lse=True,
    )
    assert torch.isfinite(actual).all()
    # Same recipe as the kernel: BF16 output rounding plus FP32 summation order.
    assert (actual.float() - expected).norm() / expected.norm() <= 2e-3
    assert (actual.double() - fp64).norm() <= (
        expected.double() - fp64
    ).norm() + 2e-3 * fp64.norm()
    torch.testing.assert_close(lse, lse_ref.float(), atol=2e-4, rtol=2e-5, equal_nan=True)
    ones = torch.ones_like(deq[2])
    ones8, sfones, _ = quantize_v_to_fp8_with_sf(ones)
    normalized, _ = flash_attn_func(
        q, k, ones8,
        scales=BlockScales(sfq, sfk, sfones, ones.bfloat16() if causal else None,
                           pv_softmax=pv_softmax),
        causal=causal,
    )
    # "exact" divides by the sum of the P that PV consumed; "fast" by the FP32 sum.
    visible = normalized[:, max(sq - sk, 0):].float() if causal else normalized.float()
    atol = 1e-5 if pv_softmax == "exact" else 2**-4
    torch.testing.assert_close(visible, torch.ones_like(visible), atol=atol, rtol=0)


@pytest.mark.parametrize(
    "sq,sk,hq,hk",
    [
        (257, 386, 4, 2),  # unaligned seqlen_k - seqlen_q, GQA
        (385, 129, 2, 2),  # leading work tile fully masked
        (1000, 2049, 4, 1),  # 17 KV blocks, unaligned offset
        (300, 300, 2, 2),  # partial last work tile
    ],
)
@pytest.mark.parametrize("pv_softmax", ["exact", "fast"])
def test_mxfp8_hd256_causal_no_future_leakage(sq, sk, hq, hk, pv_softmax):
    """Scaling V at future positions must leave every earlier row bitwise unchanged."""
    requires_sm100()
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(1, sq, hq, DIM, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, sk, hk, DIM, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, sk, hk, DIM, generator=g, device="cuda", dtype=torch.bfloat16)
    q8, sfq, _ = quantize_to_fp8_with_sf(q)
    k8, sfk, _ = quantize_to_fp8_with_sf(k)
    sfq, sfk = interleave_sf(sfq), interleave_sf(sfk)

    def run(v_bf16):
        v8, sfv, _ = quantize_v_to_fp8_with_sf(v_bf16)
        return flash_attn_func(
            q8, k8, v8, scales=BlockScales(sfq, sfk, sfv, v_bf16=v_bf16, pv_softmax=pv_softmax),
            causal=True, return_lse=True,
        )

    out, lse = run(v)
    assert torch.isfinite(out).all()
    for t0 in (sk // 4 + 17, sk // 2 + 5, 3 * sk // 4 + 29, sk - 1):
        rows = t0 - (sk - sq)  # rows [0, rows) attend only keys < t0
        if rows <= 0 or t0 >= sk:
            continue
        for factor in (16.0, 1024.0):
            v2 = v.clone()
            v2[:, t0:] = (v2[:, t0:].float() * factor).to(torch.bfloat16)
            out2, lse2 = run(v2)
            assert torch.equal(out2[:, :rows].view(torch.uint8), out[:, :rows].view(torch.uint8))
            assert torch.equal(lse2[..., :rows].view(torch.uint8), lse[..., :rows].view(torch.uint8))


@pytest.mark.parametrize("dim", [64, 128, DIM])
@pytest.mark.parametrize("sq,sk,hq,hk", [(385, 513, 4, 2), (513, 129, 4, 1)])
@pytest.mark.parametrize("mode", ["qk", "exact", "fast"])
@pytest.mark.parametrize("causal", [False, True])
def test_mxfp8_stats_out(mode, causal, sq, sk, hq, hk, dim):
    """stats_out leaves O and LSE bitwise unchanged and reproduces them from (m, L), at every
    MXFP8 head dim. Causal 513 x 129 has rows without a visible key."""
    requires_sm100()
    (q, k, v8, sfq, sfk, sfv), (qd, kd, vd) = test_mxfp8_pv.inputs(2, sq, sk, hq, hk, dim)
    if mode == "qk":
        v, scales = vd.bfloat16(), BlockScales(sfq, sfk)
    else:
        v, scales = v8, BlockScales(sfq, sfk, sfv, vd.bfloat16() if causal else None, pv_softmax=mode)
    out, lse = flash_attn_func(q, k, v, scales=scales, causal=causal, return_lse=True)
    stats = torch.full((2, hq, sq, 2), math.nan, device="cuda")
    out2, lse2 = flash_attn_func(
        q, k, v, scales=scales._replace(stats_out=stats), causal=causal, return_lse=True
    )
    assert torch.equal(out.view(torch.uint8), out2.view(torch.uint8))
    assert torch.equal(lse.view(torch.uint8), lse2.view(torch.uint8))
    c = torch.tensor(math.log2(math.e) / math.sqrt(dim), device="cuda")
    m, normalizer = stats[..., 0] * c, stats[..., 1]
    visible = torch.isfinite(lse)
    assert torch.isneginf(m[~visible]).all() and (normalizer[~visible] == 0).all()
    # The true row max in log2 units; BF16 P keeps a stale max up to 8 below it (lazy rescale).
    rep = hq // hk
    scores = torch.einsum("bqhd,bkhd->bhqk", qd.double(), kd.repeat_interleave(rep, dim=2).double())
    if causal:
        qpos = torch.arange(sq, device="cuda")[:, None] + sk - sq
        scores = scores.masked_fill(torch.arange(sk, device="cuda")[None, :] > qpos, -math.inf)
    true_max = scores.amax(-1) * c.double()
    lag = (true_max - m.double())[visible]
    max_lag = {"qk": 8.0, "exact": 0.0, "fast": 4.0}[mode]
    assert lag.min() >= -1e-3 and lag.max() <= max_lag + 1e-3
    # LSE uses the FP32 sum, which is L except for "exact" (the E4M3 sum normalizes O there).
    offset = {"qk": 0, "exact": 8, "fast": 4}[mode]
    lse_from_stats = (m + torch.log2(normalizer) - offset) * math.log(2)
    tol = 2e-2 if mode == "exact" else 2e-5
    torch.testing.assert_close(lse_from_stats[visible], lse[visible], atol=tol, rtol=0)



def test_mxfp8_stats_out_without_keys():
    """With no keys the forward returns before any launch; stats_out still gets (-inf, 0)."""
    requires_sm100()
    q = torch.zeros(2, 128, 4, 64, dtype=torch.float8_e4m3fn, device="cuda")
    k = torch.zeros(2, 0, 2, 64, dtype=torch.float8_e4m3fn, device="cuda")
    v = torch.zeros(2, 0, 2, 64, dtype=torch.bfloat16, device="cuda")
    sfq = torch.ones(2 * 4 * 512, device="cuda").to(torch.float8_e8m0fnu)
    sfk = torch.ones(0, device="cuda").to(torch.float8_e8m0fnu)
    stats = torch.full((2, 4, 128, 2), math.nan, device="cuda")
    flash_attn_func(q, k, v, scales=BlockScales(sfq, sfk, stats_out=stats))
    assert torch.isneginf(stats[..., 0]).all() and (stats[..., 1] == 0).all()

def per_head_error(actual, expected):
    """Relative L2 error per (batch, head) of (batch, seqlen, heads, dim) outputs."""
    error = (actual.double() - expected).square().sum(dim=(1, 3)).sqrt()
    return error / expected.square().sum(dim=(1, 3)).sqrt()


@pytest.mark.parametrize("mode", ["qk", "exact"])
def test_mxfp8_hd256_persistent_tiles(mode):
    """Many work tiles per CTA (noncausal persistent): sampled rows of every tile match."""
    requires_sm100()
    (q, k, v8, sfq, sfk, sfv), (qd, kd, vd) = test_mxfp8_pv.inputs(4, 2048, 2049, 32, 8, DIM)
    starts = torch.arange(0, 2048, 128, device=q.device)
    rows = torch.stack((starts, starts + 127), dim=1).flatten()
    if mode == "qk":
        v, scales = vd.bfloat16(), BlockScales(sfq, sfk)
        expected, _, eager = test_mxfp8_qk.reference(qd[:, rows], kd, v, causal=False)
        # BF16 P and O roundings: compare to the eager (BF16 P) reference's own error.
        head_bound = per_head_error(eager, expected) + 2e-3
        total_bound = (eager.double() - expected).norm() / expected.norm() + 2e-3
    else:
        v, scales = v8, BlockScales(sfq, sfk, sfv)
        expected, _, _ = test_mxfp8_pv.reference(qd[:, rows], kd, vd)
        # A wrong or stale tile is off by O(1). Single rows can sit at an E4M3 rounding knife edge,
        # where a 1-ulp difference in the MMA's S moves O by a few percent, so the per-(batch, head)
        # bound has slack and the 2e-3 recipe bound applies to the whole set.
        head_bound, total_bound = 5e-3, 2e-3
    for _ in range(3):
        actual, _ = flash_attn_func(q, k, v, scales=scales)
        sampled = actual[:, rows].double()
        assert (per_head_error(sampled, expected) <= head_bound).all()
        assert (sampled - expected).norm() / expected.norm() <= total_bound
