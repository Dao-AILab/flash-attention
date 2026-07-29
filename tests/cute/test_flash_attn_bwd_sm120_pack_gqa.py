"""SM120 backward PackGQA regression coverage."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func


def _sm120_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = torch.cuda.get_device_capability(0)
    if cc[0] != 12:  # consumer Blackwell sm_12x (sm_120, sm_121 DGX Spark)
        pytest.skip(f"SM120-only test (got sm_{cc[0]}{cc[1]})")


def _sdpa_ref_grads(q, k, v, dout, causal):
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    repeat = q.shape[2] // k.shape[2]
    qh = q_ref.transpose(1, 2)
    kh = k_ref.repeat_interleave(repeat, dim=2).transpose(1, 2)
    vh = v_ref.repeat_interleave(repeat, dim=2).transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            qh.float(), kh.float(), vh.float(), is_causal=causal,
        ).transpose(1, 2).to(q.dtype)
    out.backward(dout)
    return q_ref.grad, k_ref.grad, v_ref.grad


@pytest.mark.parametrize("causal", [False, True])
def test_sm120_bwd_pack_gqa_odd_seqlen(causal):
    """Odd seqlen leaves OOB packed rows in the final m-block.

    The backward PackGQA Q/dO loads must zero those rows; otherwise stale smem
    pollutes dK/dV.  Use qh=7 to cover non-divisible packed row groups.
    """
    _sm120_only()
    torch.manual_seed(1700 + int(causal))
    batch, seqlen, nheads, nheads_kv, head_dim = 1, 65, 28, 4, 64
    dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, nheads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen, nheads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    dout = torch.randn_like(q)

    out = flash_attn_func(q, k, v, causal=causal, pack_gqa=True)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(dout)
    ref_dq, ref_dk, ref_dv = _sdpa_ref_grads(q, k, v, dout, causal)

    max_diff = max(
        (q.grad.float() - ref_dq.float()).abs().max().item(),
        (k.grad.float() - ref_dk.float()).abs().max().item(),
        (v.grad.float() - ref_dv.float()).abs().max().item(),
    )
    assert max_diff < 0.12


def test_sm120_bwd_pack_gqa_varlen_batch_offset():
    """Varlen PackGQA dQ atomics must write into each batch's padded Q slot."""
    _sm120_only()
    torch.manual_seed(1731)
    seqlens = [17, 91]
    cu_seqlens = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    total, max_seqlen = sum(seqlens), max(seqlens)
    nheads, nheads_kv, head_dim = 28, 4, 64
    dtype = torch.bfloat16

    q0 = torch.randn(total, nheads, head_dim, device="cuda", dtype=dtype)
    k0 = torch.randn(total, nheads_kv, head_dim, device="cuda", dtype=dtype)
    v0 = torch.randn(total, nheads_kv, head_dim, device="cuda", dtype=dtype)
    dout = torch.randn(total, nheads, head_dim, device="cuda", dtype=dtype)
    dout[: seqlens[0]].zero_()

    q_pack = q0.detach().clone().requires_grad_(True)
    k_pack = k0.detach().clone().requires_grad_(True)
    v_pack = v0.detach().clone().requires_grad_(True)
    out_pack = flash_attn_varlen_func(
        q_pack,
        k_pack,
        v_pack,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
        pack_gqa=True,
    )
    if isinstance(out_pack, tuple):
        out_pack = out_pack[0]
    out_pack.backward(dout)

    q_base = q0.detach().clone().requires_grad_(True)
    k_base = k0.detach().clone().requires_grad_(True)
    v_base = v0.detach().clone().requires_grad_(True)
    out_base = flash_attn_varlen_func(
        q_base,
        k_base,
        v_base,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
        pack_gqa=False,
    )
    if isinstance(out_base, tuple):
        out_base = out_base[0]
    out_base.backward(dout)

    assert q_pack.grad[: seqlens[0]].abs().max().item() < 1e-5
    assert (q_pack.grad.float() - q_base.grad.float()).abs().max().item() < 0.12
