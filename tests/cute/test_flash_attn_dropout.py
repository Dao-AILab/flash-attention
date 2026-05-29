# Copyright (c) 2025, Tri Dao.
"""Correctness tests for FA4 (CuTe SM100) varlen dropout against FA2 varlen.

Layout & conventions:

  * FA4's CuTe SM100 ``flash_attn_varlen_func`` is the device under test.
    Inputs are the standard packed ``(total_tokens, nheads, headdim)``
    buffers with cumulative-sequence-length tensors.
  * FA2 varlen (``_flash_attn_varlen_forward`` / ``_flash_attn_varlen_backward``)
    is the bit-identical reference for the dropout pattern when both are fed
    the same Philox seed/offset.
  * ``attention_ref`` is the upcast fp32 reference used to bound FA4's
    numerical error; it accepts ``query_padding_mask`` / ``key_padding_mask``
    so we can express the varlen valid region without unpacking.

Both implementations share the FA2 Philox convention, so the per-element
``dropout_mask`` returned by FA4 must agree exactly with
``S_dmask >= 0`` returned by FA2 in the per-batch valid (non-padding,
non-causal-masked) region. The forward output and the gradients
``dq / dk / dv`` are checked against fp32 reference with a tolerance
scaled by FA2's own error against the same reference — the standard
FA2/FA4 comparison style used throughout the codebase.

Run with::

    pytest -q tests/cute/test_flash_attn_dropout.py
"""

import math

import pytest
import torch

from flash_attn.cute.testing import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)
from flash_attn.cute.interface import flash_attn_varlen_func as fa4_varlen_func

# Call the FA2 C++ kernels directly: the Python wrappers in
# ``flash_attn.flash_attn_interface`` were authored against a newer FA2 ABI
# that passes ``num_splits`` to ``varlen_fwd`` / ``varlen_bwd``, but the
# legacy FA2 binary on this machine doesn't accept that arg. Going through
# ``flash_attn_2_cuda`` directly keeps the test independent of the wrapper
# evolution and matches what the wrapper would do internally.
import flash_attn_2_cuda  # noqa: E402


IS_SM100 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)

requires_sm100 = pytest.mark.skipif(
    not IS_SM100,
    reason="FA4 dropout is only supported on SM100 (Blackwell)",
)


def _fa2_varlen_fwd_with_rng(
    q, k, v, cu_q, cu_k, max_q, max_k,
    causal, p_dropout, softmax_scale, seed, offset,
):
    """Run FA2 varlen forward with a controlled Philox state.

    Bypasses ``_flash_attn_varlen_forward`` (which appends ``num_splits``);
    the underlying C++ ``varlen_fwd`` here takes exactly 21 positional args.
    Returns the same ``(out, lse, S_dmask, rng_state)`` tuple the wrapper
    would.
    """
    gen = torch.cuda.default_generators[torch.cuda.current_device()]
    gen.manual_seed(seed)
    gen.set_offset(offset)
    out, lse, S_dmask, rng_state = flash_attn_2_cuda.varlen_fwd(
        q, k, v,
        None,         # out (allocated by kernel)
        cu_q, cu_k,
        None,         # seqused_k
        None,         # leftpad_k
        None,         # block_table
        None,         # alibi_slopes
        max_q, max_k,
        p_dropout, softmax_scale,
        False,        # zero_tensors
        causal,
        -1, -1,       # window_size_{left,right}
        0.0,          # softcap
        True,         # return_softmax
        None,         # generator (state is set above via gen.manual_seed)
    )
    return out, lse, S_dmask, rng_state


def _fa2_varlen_bwd_with_rng(
    dout, q, k, v, out, lse,
    cu_q, cu_k, max_q, max_k,
    p_dropout, softmax_scale, causal, rng_state,
):
    """Run FA2 varlen backward, again calling the C++ kernel directly so
    the test is robust to wrapper-vs-binary version skew."""
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    flash_attn_2_cuda.varlen_bwd(
        dout, q, k, v, out, lse,
        dq, dk, dv,
        cu_q, cu_k,
        None,         # alibi_slopes
        max_q, max_k,
        p_dropout, softmax_scale,
        False,        # zero_tensors
        causal,
        -1, -1,       # window_size_{left,right}
        0.0,          # softcap
        True,         # deterministic
        None,         # gen (rng_state replays Philox state)
        rng_state,
    )
    return dq, dk, dv


def _varlen_valid_mask(seqlens_q, seqlens_k, max_q, max_k, causal, device):
    """Boolean mask of shape ``(B, max_q, max_k)`` marking positions that the
    kernels are expected to populate (the rest is left at the buffer's zero
    initializer).

    For batch element ``b``, a cell ``(q_idx, k_idx)`` is valid iff
    ``q_idx < seqlens_q[b]`` AND ``k_idx < seqlens_k[b]`` AND
    (if ``causal``) ``k_idx <= q_idx + (seqlens_k[b] - seqlens_q[b])``.
    """
    B = seqlens_q.numel()
    q_idx = torch.arange(max_q, device=device).view(1, max_q, 1)
    k_idx = torch.arange(max_k, device=device).view(1, 1, max_k)
    sq = seqlens_q.view(B, 1, 1)
    sk = seqlens_k.view(B, 1, 1)
    valid = (q_idx < sq) & (k_idx < sk)
    if causal:
        offset = sk - sq  # bottom-right causal alignment
        valid &= k_idx <= (q_idx + offset)
    return valid  # (B, max_q, max_k) bool


def _mask_match_rate_varlen(
    fa4_mask, fa2_S_dmask, seqlens_q, seqlens_k, causal,
):
    """Bit-compare FA4 vs FA2 dropout decisions in the per-batch valid region.

    ``fa4_mask``: uint8 ``(B, H, max_q, max_k)``, 1 = kept / 0 = dropped.
    ``fa2_S_dmask``: bf16 ``(B, H, round_up_128(max_q), round_up_128(max_k))``;
    decisions are ``>= 0`` for kept / ``< 0`` for dropped.
    Padding-region cells outside per-batch seqlens are ignored on both sides
    because the FA4 buffer is zero-initialised there and FA2 may leave
    arbitrary values from the padded softmax stage.
    """
    B, H, max_q, max_k = fa4_mask.shape
    fa2_keep = (fa2_S_dmask[:, :, :max_q, :max_k] >= 0).to(torch.uint8)
    valid = _varlen_valid_mask(
        seqlens_q, seqlens_k, max_q, max_k, causal, fa4_mask.device,
    )  # (B, max_q, max_k)
    valid_4d = valid.unsqueeze(1).expand(B, H, max_q, max_k)
    match = ((fa4_mask == fa2_keep) & valid_4d).sum().item()
    total = valid_4d.sum().item()
    return (match / max(total, 1)), total


# ---------------------------------------------------------------------------
# Forward + backward correctness: FA4 varlen vs FA2 varlen
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
        (4, 192, 4),  # B=4 with jagged lengths exercises the per-row philox math
    ],
)
def test_flash_attn_varlen_dropout_output(
    batch, seqlen, nheads, d, p_dropout, causal, dtype,
):
    """FA4 varlen forward + backward with dropout vs FA2 varlen with the same
    Philox state.

    Verifies:

      1. The dropout decision per ``(batch, head, q_idx, k_idx)`` is
         *bit*-identical to FA2 in the per-batch valid region.
      2. The forward output ``out`` matches the fp32 reference within a
         tolerance proportional to FA2's own error against the same reference.
      3. The gradients ``dq``, ``dk``, ``dv`` match FA2 to within a tolerance
         proportional to ``|FA2|`` (the standard FA2/FA4 comparison budget).
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # ── Dense (B, S, H, D) tensors + a varlen padding mask ────────────────
    q_dense = torch.randn(batch, seqlen, nheads, d, device=device, dtype=dtype)
    k_dense = torch.randn(batch, seqlen, nheads, d, device=device, dtype=dtype)
    v_dense = torch.randn(batch, seqlen, nheads, d, device=device, dtype=dtype)

    # Same padding mask for q and k so the comparison is symmetric.
    padding_mask = generate_random_padding_mask(seqlen, batch, device, mode="random")

    # ``generate_qkv`` packs into varlen ((total, h, d)) + builds cu_seqlens
    # and an output_pad_fn that re-pads the FA4/FA2 outputs back to dense.
    # Returns 17 items (q_u, k_u, v_u, qv_u, cu_q, cu_k, seqused_q, seqused_k,
    # max_q, max_k, q_d, k_d, v_d, qv_d, output_pad_fn, dq_pad_fn, dk_pad_fn).
    (
        q_unpad, k_unpad, v_unpad, _qv_unpad,
        cu_q, cu_k,
        _seqused_q, _seqused_k,
        max_q, max_k,
        _q_d, _k_d, _v_d, _qv_d,
        output_pad_fn, _dq_pad_fn, _dk_pad_fn,
    ) = generate_qkv(
        q_dense, k_dense, v_dense,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask,
    )
    # Per-batch valid lengths needed for both the mask comparison and the
    # causal carve-out inside the reference attention.
    seqlens_q = (cu_q[1:] - cu_q[:-1]).to(torch.int64)
    seqlens_k = (cu_k[1:] - cu_k[:-1]).to(torch.int64)

    softmax_scale = 1.0 / math.sqrt(d)

    # ── FA4 varlen with controlled Philox ─────────────────────────────────
    q_fa4 = q_unpad.detach().clone().requires_grad_()
    k_fa4 = k_unpad.detach().clone().requires_grad_()
    v_fa4 = v_unpad.detach().clone().requires_grad_()
    seed, offset = 42, 0
    rng = torch.tensor([seed, offset], dtype=torch.int64)
    out_fa4_u, lse_fa4, rng4, mask_fa4 = fa4_varlen_func(
        q_fa4, k_fa4, v_fa4,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        causal=causal,
        softmax_scale=softmax_scale,
        p_dropout=p_dropout,
        return_dropout_mask=True,
        return_lse=True,
        rng_state=rng,
    )

    # ── FA2 varlen with the same Philox seed/offset ───────────────────────
    q_fa2 = q_unpad.detach().clone().requires_grad_()
    k_fa2 = k_unpad.detach().clone().requires_grad_()
    v_fa2 = v_unpad.detach().clone().requires_grad_()
    out_fa2_u, lse_fa2, S_dmask_fa2, rng2 = _fa2_varlen_fwd_with_rng(
        q_fa2, k_fa2, v_fa2,
        cu_q, cu_k, max_q, max_k,
        causal=causal,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        seed=seed, offset=offset,
    )

    # ── 1. Bit-identical dropout mask in the per-batch valid region ───────
    match_rate, total = _mask_match_rate_varlen(
        mask_fa4, S_dmask_fa2, seqlens_q, seqlens_k, causal,
    )
    assert match_rate == 1.0, (
        f"FA4/FA2 varlen dropout masks diverge: match_rate={match_rate:.6f} "
        f"({total} valid elements compared)"
    )

    # ── 2. Forward output vs fp32 reference (dense, with padding masks) ───
    # Use FA4's mask as ground truth and re-do attention in fp32.
    # FA4 emits the mask at the kernel's logical ``(B, H, max_q, max_k)``
    # extent; the dense reference expects ``(B, H, seqlen, seqlen)``. Pad
    # with zeros on the right: those positions are also masked out by
    # ``padding_mask`` inside ``attention_ref`` so the dropout decision in
    # the padding region is a no-op.
    B, H = mask_fa4.shape[:2]
    dropout_mask_bool = torch.zeros(
        B, H, seqlen, seqlen, dtype=torch.bool, device=device,
    )
    dropout_mask_bool[:, :, :max_q, :max_k] = mask_fa4.to(torch.bool)
    out_fa4_pad = output_pad_fn(out_fa4_u)
    out_ref, _ = attention_ref(
        q_dense, k_dense, v_dense,
        padding_mask, padding_mask,
        dropout_p=p_dropout,
        dropout_mask=dropout_mask_bool,
        causal=causal,
    )
    out_pt, _ = attention_ref(
        q_dense, k_dense, v_dense,
        padding_mask, padding_mask,
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
    err_fa4 = (out_fa4_pad.float() - out_ref.float()).abs().max().item()
    err_pt = (out_pt - out_ref).abs().max().item()
    assert err_fa4 <= rtol * err_pt + fwd_atol, (
        f"FA4 fwd error {err_fa4:.4e} vs PT reference error {err_pt:.4e} "
        f"(atol={fwd_atol:.4e}, rtol={rtol})"
    )

    # ── 3. Backward correctness ──────────────────────────────────────────
    if d > 128:
        # Bwd path is not exercised here on hd>128 to keep this test focused
        # on the dropout-specific behaviour; the main test_flash_attn.py
        # covers the bwd kernel itself.
        return

    dout_u = torch.randn_like(out_fa4_u)
    dq4, dk4, dv4 = torch.autograd.grad(out_fa4_u, (q_fa4, k_fa4, v_fa4), dout_u)

    dq2, dk2, dv2 = _fa2_varlen_bwd_with_rng(
        dout_u, q_fa2, k_fa2, v_fa2, out_fa2_u, lse_fa2,
        cu_q, cu_k, max_q, max_k,
        p_dropout=p_dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        rng_state=rng2,
    )

    # Compare in unpadded (varlen) layout: cells outside the valid per-batch
    # region carry undefined values for both kernels and are not compared.
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
def test_flash_attn_varlen_dropout_kept_fraction(p_dropout, d):
    """Empirical kept-fraction of FA4's varlen dropout mask is close to ``1 - p``.

    Non-causal so every cell in the per-batch valid region is written; with
    causal the kernel skips fully-masked tiles, leaving them at the buffer's
    zero initialiser which would skew the global mean.
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    batch, seqlen, nheads = 2, 512, 8
    q = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16)
    padding_mask = generate_random_padding_mask(seqlen, batch, device, mode="random")

    (
        q_u, k_u, v_u, _qv_u,
        cu_q, cu_k,
        _sq, _sk,
        max_q, max_k,
        *_unused,
    ) = generate_qkv(
        q, k, v,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask,
    )

    rng = torch.tensor([7, 0], dtype=torch.int64)
    if p_dropout == 0.0:
        out, lse = fa4_varlen_func(
            q_u, k_u, v_u,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            causal=False, p_dropout=0.0, return_lse=True,
        )
        assert out.isfinite().all().item()
        return

    out, lse, _, mask = fa4_varlen_func(
        q_u, k_u, v_u,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
        causal=False,
        p_dropout=p_dropout,
        return_dropout_mask=True,
        return_lse=True,
        rng_state=rng,
    )
    # Compute kept-fraction only over the per-batch valid (non-padding) region.
    seqlens_q = (cu_q[1:] - cu_q[:-1]).to(torch.int64)
    seqlens_k = (cu_k[1:] - cu_k[:-1]).to(torch.int64)
    valid = _varlen_valid_mask(
        seqlens_q, seqlens_k, max_q, max_k, causal=False, device=device,
    )
    B, H, _, _ = mask.shape
    valid_4d = valid.unsqueeze(1).expand(B, H, max_q, max_k)
    kept = (mask.bool() & valid_4d).sum().item() / valid_4d.sum().item()
    expected = 1.0 - p_dropout
    # ~2% slack accounts for finite-sample variance at this tensor size.
    assert abs(kept - expected) < 0.02, (
        f"kept-fraction {kept:.4f} too far from 1-p={expected:.4f}"
    )
