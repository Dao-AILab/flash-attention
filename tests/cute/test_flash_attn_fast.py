# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Fast subset of test_flash_attn.py for quick iteration.
# Covers: causal/noncausal, varlen/not varlen, MHA/GQA, split/not split, fwd+bwd.

import os
import random

import pytest
import torch

from einops import rearrange

from flash_attn.cute.testing import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
    maybe_fake_tensor_mode,
    is_fake_mode,
)
from flash_attn.cute.interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_combine,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_SM90 = torch.cuda.get_device_capability()[0] == 9


# ---------------------------------------------------------------------------
# Forward + backward (non-varlen)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (256, 256),
        (113, 203),
        (1024, 1024),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_output(seqlen_q, seqlen_k, d, causal, num_splits, mha_type, dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    random.seed(0)
    torch.cuda.empty_cache()
    batch_size = 4
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else 3

    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()

    q = q_ref.detach().to(dtype).requires_grad_()
    k = k_ref.detach().to(dtype).requires_grad_()
    v = v_ref.detach().to(dtype).requires_grad_()

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, causal=causal)
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None, causal=causal, upcast=False, reorder_ops=True,
    )

    out, lse = flash_attn_func(q, k, v, causal=causal, num_splits=num_splits)

    if is_fake_mode():
        return

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol

    # Backward (only for non-split, matching d)
    can_bwd = (
        num_splits == 1
        and d <= 128
        and not (causal and seqlen_k < seqlen_q)
    )
    if IS_SM90 and d == 64 and not causal:
        can_bwd = False  # SM90 d=64 non-causal xfail
    if not can_bwd:
        return

    g = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)

    dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
    dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item()
    dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item()
    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item() + dq_atol
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item() + dk_atol
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item() + dv_atol


# ---------------------------------------------------------------------------
# Forward + backward (varlen with cu_seqlens)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 256, 1024])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_output(seqlen, d, causal, mha_type, dtype):
    """Varlen test with cu_seqlens (packed): equal seqlens so we can compare with non-varlen ref."""
    device = "cuda"
    seed = seqlen + d + int(causal) * 2
    torch.random.manual_seed(seed)
    random.seed(seed)
    batch_size = 9
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else 3

    q_ref = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype).to(dtype).requires_grad_()

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, causal=causal)
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None, causal=causal, upcast=False, reorder_ops=True,
    )

    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32)
    q_varlen = rearrange(q_ref.detach(), "b s h d -> (b s) h d").requires_grad_()
    k_varlen = rearrange(k_ref.detach(), "b s h d -> (b s) h d").requires_grad_()
    v_varlen = rearrange(v_ref.detach(), "b s h d -> (b s) h d").requires_grad_()

    out_varlen, lse = flash_attn_varlen_func(
        q_varlen, k_varlen, v_varlen,
        cu_seqlens, cu_seqlens,
        seqlen, seqlen,
        causal=causal,
    )

    if is_fake_mode():
        return

    out_reshaped = rearrange(out_varlen, "(b s) h d -> b s h d", b=batch_size)
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out_reshaped - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol

    # Backward (original test skips all SM90 varlen backward)
    can_bwd = d <= 128 and not IS_SM90
    if not can_bwd:
        return

    g = torch.randn_like(out_varlen)
    dq_varlen, dk_varlen, dv_varlen = torch.autograd.grad(out_varlen, (q_varlen, k_varlen, v_varlen), g)

    assert dq_varlen.isfinite().all(), "dq contains non-finite values"
    assert dk_varlen.isfinite().all(), "dk contains non-finite values"
    assert dv_varlen.isfinite().all(), "dv contains non-finite values"
    assert dq_varlen.abs().max().item() > 0, "dq is all zeros"
    assert dk_varlen.abs().max().item() > 0, "dk is all zeros"
    assert dv_varlen.abs().max().item() > 0, "dv is all zeros"


# ---------------------------------------------------------------------------
# Forward + backward (varlen with padding masks — all unpad combinations)
# Covers 4 compile-key-distinct paths:
#   (unpad_q, unpad_kv) = (T,T): cu_seqlens for both Q and K
#   (unpad_q, unpad_kv) = (F,F): seqused for both Q and K
#   (unpad_q, unpad_kv) = (T,F): cu_seqlens_q + seqused_k
#   (unpad_q, unpad_kv) = (F,T): seqused_q + cu_seqlens_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize(
    "unpad_q,unpad_kv",
    [(True, True), (False, False), (True, False), (False, True)],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_unpad_output(seqlen, d, causal, mha_type, unpad_q, unpad_kv, dtype):
    """Varlen test with all 4 (unpad_q, unpad_kv) combos: cu_seqlens vs seqused."""
    device = "cuda"
    seed = seqlen + d + int(causal) * 2 + int(unpad_q) * 7 + int(unpad_kv) * 13
    torch.random.manual_seed(seed)
    random.seed(seed)
    batch_size = 9
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else 3

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype)
    q_ref = q.detach().to(dtype).requires_grad_()
    k_ref = k.detach().to(dtype).requires_grad_()
    v_ref = v.detach().to(dtype).requires_grad_()

    query_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    key_padding_mask = query_padding_mask if causal else generate_random_padding_mask(
        seqlen, batch_size, device, mode="random"
    )

    (
        q_unpad_t, k_unpad_t, v_unpad_t, _qv_unpad,
        cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        q_padded, k_padded, v_padded, _qv_padded,
        output_pad_fn, dq_pad_fn, dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask, causal=causal,
    )
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask, causal=causal,
        upcast=False, reorder_ops=True,
    )

    # Select Q input: packed (unpad) or padded (seqused)
    if unpad_q:
        q_in = q_unpad_t.detach().to(dtype).requires_grad_()
    else:
        q_in = q.detach().to(dtype).requires_grad_()
    # Select KV input: packed (unpad) or padded (seqused)
    if unpad_kv:
        k_in = k_unpad_t.detach().to(dtype).requires_grad_()
        v_in = v_unpad_t.detach().to(dtype).requires_grad_()
    else:
        k_in = k.detach().to(dtype).requires_grad_()
        v_in = v.detach().to(dtype).requires_grad_()

    out_unpad, lse = flash_attn_varlen_func(
        q_in, k_in, v_in,
        cu_seqlens_q=cu_seqlens_q if unpad_q else None,
        cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        seqused_q=seqused_q if not unpad_q else None,
        seqused_k=seqused_k if not unpad_kv else None,
        causal=causal,
    )

    if is_fake_mode():
        return

    # Reshape output to (batch, seqlen, nheads, d) for comparison
    out = output_pad_fn(out_unpad) if unpad_q else out_unpad

    # Mask out padding positions — kernel output at padding positions is undefined
    q_mask = rearrange(query_padding_mask, "b s -> b s 1 1")
    out_masked = out.clone().masked_fill_(~q_mask, 0.0)
    out_ref_masked = out_ref.clone().masked_fill_(~q_mask, 0.0)
    out_pt_masked = out_pt.clone().masked_fill_(~q_mask, 0.0)

    fwd_atol = 2 * (out_ref_masked + 0.3 - 0.3 - out_ref_masked).abs().max().item()
    assert (out_masked - out_ref_masked).abs().max().item() <= 2 * (out_pt_masked - out_ref_masked).abs().max().item() + fwd_atol

    # Backward (original test skips all SM90 varlen backward)
    can_bwd = d <= 128 and not IS_SM90
    if not can_bwd:
        return

    g = torch.randn_like(out_unpad)
    dq_in, dk_in, dv_in = torch.autograd.grad(out_unpad, (q_in, k_in, v_in), g)

    assert dq_in.isfinite().all(), "dq contains non-finite values"
    assert dk_in.isfinite().all(), "dk contains non-finite values"
    assert dv_in.isfinite().all(), "dv contains non-finite values"
    assert dq_in.abs().max().item() > 0, "dq is all zeros"
    assert dk_in.abs().max().item() > 0, "dk is all zeros"
    assert dv_in.abs().max().item() > 0, "dv is all zeros"


# ---------------------------------------------------------------------------
# Combine kernel
# ---------------------------------------------------------------------------

def attention_combine_ref(out_partial, lse_partial):
    lse = torch.logsumexp(lse_partial, dim=0)
    scale = torch.exp(lse_partial - lse)
    scale = torch.where(torch.isinf(scale) | torch.isnan(scale), torch.zeros_like(scale), scale)
    out = (scale.unsqueeze(-1) * out_partial).sum(0)
    return out, lse


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [32, 256])
@pytest.mark.parametrize("num_splits", [2, 5, 17])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_combine(num_splits, seqlen, d, dtype):
    device = "cuda"
    torch.random.manual_seed(1)
    batch_size = 3
    nheads = 8

    # out_partial: (num_splits, batch, seqlen, nheads, d) with stride(-1)==1
    # lse_partial: (num_splits, batch, seqlen, nheads) with stride(-2)==1 (seqlen contiguous)
    out_partial = torch.randn(
        num_splits, batch_size, seqlen, nheads, d, device=device, dtype=torch.float32,
    )
    lse_partial = torch.randn(
        num_splits, batch_size, nheads, seqlen, device=device, dtype=torch.float32,
    ).transpose(-1, -2)
    lse_partial[num_splits // 2 :, : batch_size // 3] = -float("inf")

    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype, return_lse=True)
    if is_fake_mode():
        return
    out_ref, lse_ref = attention_combine_ref(out_partial, lse_partial)
    out_pt = out_ref.to(dtype)

    assert torch.allclose(lse, lse_ref, atol=1e-5, rtol=1e-5)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() or torch.allclose(out, out_pt, atol=1e-5, rtol=1e-5)
