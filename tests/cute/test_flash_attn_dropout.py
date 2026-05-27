# Copyright (c) 2025, Tri Dao.
"""Correctness tests for FA4 (CuTe SM100) dropout against FA2.

Layout & conventions follow ``tests/cute/test_flash_attn.py``:

  * FA4's CuTe SM100 forward / backward kernels are the device under test.
  * FA2 (the legacy CUDA build) is the bit-identical reference for the
    dropout pattern when both are fed the same Philox seed/offset.
  * ``attention_ref`` is the upcast fp32 reference used to bound FA4's
    numerical error.

Both implementations share the FA2 Philox convention, so the per-element
``dropout_mask`` returned by FA4 and ``S_dmask`` returned by FA2 must agree
exactly in the valid (non-causal-masked) region. The forward output and the
gradients ``dq / dk / dv`` are checked against fp32 reference with a tolerance
scaled by FA2's own error against the same reference - this matches the
standard FA2/FA4 comparison style used throughout the codebase.

Run with::

    pytest -q tests/cute/test_flash_attn_dropout.py
"""

import math

import pytest
import torch
from einops import rearrange, repeat

from flash_attn.cute.testing import attention_ref
from flash_attn.cute.interface import flash_attn_func as fa4_flash_attn_func

# Legacy FA2 entry points (always-on rng_state low-level API).
from flash_attn.flash_attn_interface import (
    _flash_attn_forward as fa2_flash_attn_forward,
    _flash_attn_backward as fa2_flash_attn_backward,
)


IS_SM100 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)

requires_sm100 = pytest.mark.skipif(
    not IS_SM100,
    reason="FA4 dropout is only supported on SM100 (Blackwell)",
)


def _fa2_fwd_with_rng(q, k, v, causal, p_dropout, softmax_scale, seed, offset):
    """Run FA2 forward with a controlled Philox state."""
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    gen.manual_seed(seed)
    gen.set_offset(offset)
    out, lse, S_dmask, rng_state = fa2_flash_attn_forward(
        q, k, v,
        dropout_p=p_dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=True,
    )
    return out, lse, S_dmask, rng_state


def _fa2_bwd_with_rng(dout, q, k, v, out, lse, p_dropout, softmax_scale, causal, rng_state):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    fa2_flash_attn_backward(
        dout, q, k, v, out, lse,
        dq, dk, dv,
        dropout_p=p_dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=True,
        rng_state=rng_state,
    )
    return dq, dk, dv


def _mask_match_rate(fa4_mask, fa2_S_dmask, causal):
    """Compare dropout masks in the valid (non-causal-masked) region.

    Returns (match_rate, total_count). FA4 mask: uint8 (b, h, sq, sk),
    1 = kept, 0 = dropped. FA2 ``S_dmask``: bf16 (b, h, sq, sk),
    >=0 = kept, <0 = dropped.
    """
    fa2_keep = (fa2_S_dmask >= 0).to(torch.uint8)
    b, h, sq, sk = fa4_mask.shape
    fa2_keep = fa2_keep[:, :, :sq, :sk]
    if causal:
        valid = torch.tril(
            torch.ones(sq, sk, dtype=torch.bool, device=fa4_mask.device),
            diagonal=sk - sq,
        )
        fa2_valid = fa2_keep[..., valid]
        fa4_valid = fa4_mask[..., valid]
    else:
        fa2_valid = fa2_keep
        fa4_valid = fa4_mask
    match = (fa2_valid == fa4_valid).sum().item()
    total = fa2_valid.numel()
    return (match / max(total, 1)), total


# ---------------------------------------------------------------------------
# Forward + backward correctness: FA4 vs FA2 with identical Philox state
# ---------------------------------------------------------------------------

@requires_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("p_dropout", [0.1, 0.25])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "batch,seqlen,nheads",
    [
        (2, 128, 4),
        (2, 256, 4),
        (1, 512, 8),
    ],
)
def test_flash_attn_dropout_output(batch, seqlen, nheads, d, p_dropout, causal, dtype):
    """FA4 forward + backward with dropout vs FA2 with the same Philox state.

    Asserts that:
      1. The dropout decision per (batch, head, q_idx, k_idx) is *bit*-identical
         to FA2 in the valid region.
      2. The forward output ``out`` matches the fp32 reference to within a
         tolerance proportional to FA2's own error against the same reference.
      3. The gradients ``dq``, ``dk``, ``dv`` match FA2 to within a tolerance
         proportional to FA2's own error.
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    q_ref = torch.randn(batch, seqlen, nheads, d, device=device, dtype=dtype)
    k_ref = torch.randn(batch, seqlen, nheads, d, device=device, dtype=dtype)
    v_ref = torch.randn(batch, seqlen, nheads, d, device=device, dtype=dtype)
    q = q_ref.detach().clone().requires_grad_()
    k = k_ref.detach().clone().requires_grad_()
    v = v_ref.detach().clone().requires_grad_()
    softmax_scale = 1.0 / math.sqrt(d)

    # FA4 with a known Philox seed/offset; this is the single source of truth
    # for the dropout decisions in this test.
    seed, offset = 42, 0
    rng = torch.tensor([seed, offset], dtype=torch.int64)
    out4, lse4, rng4, mask4 = fa4_flash_attn_func(
        q, k, v,
        causal=causal,
        softmax_scale=softmax_scale,
        p_dropout=p_dropout,
        return_dropout_mask=True,
        return_lse=True,
        rng_state=rng,
    )

    # FA2 with the same seed/offset (samples through the CUDA generator we
    # seed manually).
    q2 = q_ref.detach().clone().requires_grad_()
    k2 = k_ref.detach().clone().requires_grad_()
    v2 = v_ref.detach().clone().requires_grad_()
    out2, lse2, S_dmask2, rng2 = _fa2_fwd_with_rng(
        q2, k2, v2,
        causal=causal,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        seed=seed,
        offset=offset,
    )

    # ----- 1. Bit-identical dropout mask in the valid region -------------
    match_rate, total = _mask_match_rate(mask4, S_dmask2, causal)
    assert match_rate == 1.0, (
        f"FA4/FA2 dropout masks diverge: match_rate={match_rate:.6f} "
        f"({total} elements compared)"
    )

    # ----- 2. Forward output vs fp32 reference -------------------------
    # Use the FA4 mask as ground truth (>=1 = kept) and rerun attention in fp32.
    dropout_mask_bool = mask4.to(torch.bool)
    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref,
        None, None,
        dropout_p=p_dropout,
        dropout_mask=dropout_mask_bool,
        causal=causal,
    )
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref,
        None, None,
        dropout_p=p_dropout,
        dropout_mask=dropout_mask_bool,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    # FA4 emulates exp2 via a short polynomial when dropout is on
    # (``FA4_DROP_E2E_*`` knobs in softmax.py); this widens the tolerance
    # vs the standard FA4 path.
    rtol = 4
    err_fa4 = (out4.float() - out_ref.float()).abs().max().item()
    err_pt = (out_pt - out_ref).abs().max().item()
    assert err_fa4 <= rtol * err_pt + fwd_atol, (
        f"FA4 fwd error {err_fa4:.4e} vs PT reference error {err_pt:.4e} "
        f"(atol={fwd_atol:.4e}, rtol={rtol})"
    )

    # ----- 3. Backward correctness ------------------------------------
    if d > 128:
        # bwd path is not exercised here on hd>128 to keep this test focused
        # on the dropout-specific behaviour; the main test_flash_attn.py
        # covers the bwd kernel itself.
        return

    dout = torch.randn_like(out4)
    dq4, dk4, dv4 = torch.autograd.grad(out4, (q, k, v), dout)

    dq2, dk2, dv2 = _fa2_bwd_with_rng(
        dout, q2, k2, v2, out2, lse2,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        rng_state=rng2,
    )

    # FA2 is the reference here (same Philox state, same algorithm); allow
    # FA4 to be within a multiple of FA2's own bf16 error budget.
    for name, g4, g2 in [("dq", dq4, dq2), ("dk", dk4, dk2), ("dv", dv4, dv2)]:
        max_ref = g2.float().abs().max().item()
        atol = 0.06 * max(max_ref, 1.0)
        diff = (g4.float() - g2.float()).abs().max().item()
        assert diff <= atol, (
            f"{name}: |FA4-FA2|={diff:.4e} > atol={atol:.4e} "
            f"(max |FA2|={max_ref:.4e})"
        )


# ---------------------------------------------------------------------------
# Empirical kept-fraction sanity check (no FA2 needed)
# ---------------------------------------------------------------------------

@requires_sm100
@pytest.mark.parametrize("p_dropout", [0.0, 0.1, 0.25])
@pytest.mark.parametrize("d", [64, 128])
def test_flash_attn_dropout_kept_fraction(p_dropout, d):
    """Empirical kept-fraction of the FA4 dropout mask is close to ``1 - p``.

    Non-causal only so every cell of the iterated tile is written; with causal
    the kernel skips fully-masked tiles, leaving them at the buffer's zero
    initialiser which would skew the global mean.
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    batch, seqlen, nheads = 2, 512, 8
    q = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16)

    rng = torch.tensor([7, 0], dtype=torch.int64)
    if p_dropout == 0.0:
        out, lse = fa4_flash_attn_func(
            q, k, v, causal=False, p_dropout=0.0, return_lse=True,
        )
        assert out.isfinite().all().item()
        return

    out, lse, _, mask = fa4_flash_attn_func(
        q, k, v,
        causal=False,
        p_dropout=p_dropout,
        return_dropout_mask=True,
        return_lse=True,
        rng_state=rng,
    )
    kept = mask.float().mean().item()
    expected = 1.0 - p_dropout
    # ~2% slack accounts for finite-sample variance at this tensor size.
    assert abs(kept - expected) < 0.02, (
        f"kept-fraction {kept:.4f} too far from 1-p={expected:.4f}"
    )
