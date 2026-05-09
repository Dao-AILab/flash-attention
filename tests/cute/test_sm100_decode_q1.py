# Copyright (c) 2025, Tri Dao.

import math

import pytest
import torch
from einops import rearrange

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import attention_ref


HDIM_PAIRS = [(64, 64), (96, 96), (128, 128), (192, 128)]


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(
    not _is_sm100(),
    reason="SM100 q1 decode fast path requires Blackwell SM100/SM110",
)


def _assert_close_to_reference(out, out_ref, out_pt):
    assert torch.isfinite(out).all()
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    max_err = (out - out_ref).abs().max().item()
    pytorch_err = (out_pt - out_ref).abs().max().item()
    assert max_err <= 2 * pytorch_err + fwd_atol


def _make_decode_inputs(
    *,
    batch_size=2,
    seqlen_q=1,
    seqlen_k=512,
    nheads=64,
    nheads_kv=8,
    headdim=128,
    headdim_v=128,
    dtype=torch.bfloat16,
):
    torch.manual_seed(42 + seqlen_k + batch_size)
    device = "cuda"
    q_ref = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype)
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype)
    q, k, v = [x.detach().clone() for x in (q_ref, k_ref, v_ref)]
    return q, k, v, q_ref, k_ref, v_ref


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("enable_sm100_decode_q1_opt", [False, True])
@pytest.mark.parametrize("headdim,headdim_v", HDIM_PAIRS)
def test_sm100_decode_q1_pack_gqa_dense(causal, enable_sm100_decode_q1_opt, headdim, headdim_v):
    q, k, v, q_ref, k_ref, v_ref = _make_decode_inputs(headdim=headdim, headdim_v=headdim_v)

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_pt, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, upcast=False, reorder_ops=True)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_splits=1,
        pack_gqa=True,
        return_lse=True,
        enable_sm100_decode_q1_opt=enable_sm100_decode_q1_opt,
    )

    assert lse is not None
    assert torch.isfinite(lse).all()
    _assert_close_to_reference(out, out_ref, out_pt)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("headdim,headdim_v", HDIM_PAIRS)
def test_sm100_decode_q1_pack_gqa_tma_paged_kv(causal, headdim, headdim_v):
    page_size = 128
    seqlen_k = 512
    q, _, _, q_ref, _, _ = _make_decode_inputs(
        batch_size=1,
        seqlen_k=seqlen_k,
        headdim=headdim,
        headdim_v=headdim_v,
    )

    num_pages = math.ceil(seqlen_k / page_size)
    device = q.device
    dtype = q.dtype
    nheads_kv = 8

    torch.manual_seed(2026 + int(causal))
    k_paged = torch.randn(num_pages, page_size, nheads_kv, headdim, device=device, dtype=dtype)
    v_paged = torch.randn(num_pages, page_size, nheads_kv, headdim_v, device=device, dtype=dtype)
    page_table = torch.randperm(num_pages, dtype=torch.int32, device=device).view(1, num_pages)

    k_ref = rearrange(k_paged[page_table.flatten()], "n p h d -> 1 (n p) h d")[:, :seqlen_k]
    v_ref = rearrange(v_paged[page_table.flatten()], "n p h d -> 1 (n p) h d")[:, :seqlen_k]
    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_pt, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, upcast=False, reorder_ops=True)

    out, lse = flash_attn_varlen_func(
        q,
        k_paged,
        v_paged,
        page_table=page_table,
        max_seqlen_q=1,
        max_seqlen_k=seqlen_k,
        causal=causal,
        num_splits=1,
        pack_gqa=True,
        return_lse=True,
        enable_sm100_decode_q1_opt=True,
    )

    assert lse is not None
    assert torch.isfinite(lse).all()
    _assert_close_to_reference(out, out_ref, out_pt)


@pytest.mark.parametrize("causal", [False, True])
def test_sm100_decode_q1_pack_gqa_splitkv_uses_fallback(causal):
    q, k, v, q_ref, k_ref, v_ref = _make_decode_inputs(headdim=64, headdim_v=64)

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    out_pt, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, upcast=False, reorder_ops=True)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_splits=3,
        pack_gqa=True,
        return_lse=True,
        enable_sm100_decode_q1_opt=True,
    )

    assert lse is not None
    assert torch.isfinite(lse).all()
    _assert_close_to_reference(out, out_ref, out_pt)
