"""Shared paged KV on Ampere, Ada, and RTX Blackwell."""

import pytest
import torch

from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute.interface import _flash_attn_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (8, 12),
    reason="Requires Ampere/Ada (SM8X) or RTX Blackwell (SM120)",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "d,dv,page_size",
    [
        (24, 40, 16),
        (64, 64, 128),
        (96, 96, 64),
        (128, 64, 256),
        (192, 128, 16),
        (256, 256, 128),
        (64, 96, 3),
    ],
)
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("mode", ["decode", "causal", "varlen_local"])
def test_paged(dtype, d, dv, page_size, pack_gqa, mode):
    torch.manual_seed(42)
    device = "cuda"
    batch, heads_kv, heads_q = 3, 2, 6
    q_lengths = [1] * batch if mode == "decode" else [129] * batch
    if mode == "varlen_local":
        q_lengths = [3, 17, 129]
    k_lengths = [0, 37, 173]
    capacity = ((max(k_lengths) + page_size - 1) // page_size + 1) * page_size
    pages_per_seq = capacity // page_size
    # Physical pool capacity deliberately exceeds the logical page-table extent.
    num_pages = batch * pages_per_seq + 7
    page_table = torch.randperm(num_pages, device=device, dtype=torch.int32)[
        : batch * pages_per_seq
    ].reshape(batch, pages_per_seq)
    k = torch.randn(batch, capacity, heads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch, capacity, heads_kv, dv, device=device, dtype=dtype)
    k_paged = torch.full(
        (num_pages, page_size, heads_kv, d), float("nan"), device=device, dtype=dtype
    )
    v_paged = torch.full(
        (num_pages, page_size, heads_kv, dv), float("nan"), device=device, dtype=dtype
    )
    for b, length in enumerate(k_lengths):
        for row in range(length):
            page = page_table[b, row // page_size]
            k_paged[page, row % page_size] = k[b, row]
            v_paged[page, row % page_size] = v[b, row]
        # Unused entries must never be dereferenced, even for an empty sequence.
        page_table[b, (length + page_size - 1) // page_size :] = -1

    varlen = mode == "varlen_local"
    q = torch.randn(
        (sum(q_lengths), heads_q, d) if varlen else (batch, q_lengths[0], heads_q, d),
        device=device,
        dtype=dtype,
    )
    cu_q = (
        torch.tensor([0, 3, 20, 149], device=device, dtype=torch.int32)
        if varlen
        else None
    )
    seqused_k = torch.tensor(k_lengths, device=device, dtype=torch.int32)
    window = (23, 0) if varlen else (None, None)
    causal = mode == "causal"
    out, lse = flash_attn_varlen_func(
        q,
        k_paged,
        v_paged,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=capacity,
        page_table=page_table,
        causal=causal,
        window_size=window,
        pack_gqa=pack_gqa,
        num_splits=1,
        return_lse=True,
    )

    offset = 0
    for b, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
        qb = q[offset : offset + sq] if varlen else q[b]
        kb = k[b, :sk].repeat_interleave(heads_q // heads_kv, dim=1)
        vb = v[b, :sk].repeat_interleave(heads_q // heads_kv, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qb.float(), kb.float()) * d**-0.5
        qpos = torch.arange(sq, device=device)[:, None] + sk - sq
        kpos = torch.arange(sk, device=device)[None, :]
        if causal or varlen:
            scores.masked_fill_((kpos > qpos)[None], -torch.inf)
        if varlen:
            scores.masked_fill_((kpos < qpos - window[0])[None], -torch.inf)
        ref_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).nan_to_num()
        ref = torch.einsum("hqk,khd->qhd", probs, vb.float())
        actual = out[offset : offset + sq] if varlen else out[b]
        actual_lse = lse[:, offset : offset + sq] if varlen else lse[b]
        torch.testing.assert_close(actual.float(), ref, atol=6e-3, rtol=6e-3)
        torch.testing.assert_close(actual_lse, ref_lse, atol=2e-3, rtol=2e-3)
        offset += sq


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("heads_kv", [1, 3])
@pytest.mark.parametrize("softcap", [0.0, 5.0])
def test_paged_capacity(page_size, heads_kv, softcap):
    """Without seqused_k, use the page-table row's capacity, not the pool's."""
    torch.manual_seed(7)
    batch, sq, heads_q, d = 2, 7, 3, 64
    pages_per_seq = 3
    sk = pages_per_seq * page_size
    num_pages = batch * pages_per_seq + 11
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32)[
        : batch * pages_per_seq
    ].reshape(batch, pages_per_seq)
    k_pool = torch.randn(
        num_pages, page_size, heads_kv, d, device="cuda", dtype=torch.bfloat16
    )
    v_pool = torch.randn_like(k_pool)
    q = torch.randn(batch, sq, heads_q, d, device="cuda", dtype=torch.bfloat16)
    seqused_q = torch.tensor([3, sq], device="cuda", dtype=torch.int32)
    out, lse = flash_attn_varlen_func(
        q,
        k_pool,
        v_pool,
        page_table=page_table,
        seqused_q=seqused_q,
        softcap=softcap,
        num_splits=0 if torch.cuda.get_device_capability()[0] == 12 else 1,
        return_lse=True,
    )
    k = k_pool[page_table.long()].reshape(batch, sk, heads_kv, d)
    v = v_pool[page_table.long()].reshape(batch, sk, heads_kv, d)
    k = k.repeat_interleave(heads_q // heads_kv, dim=2)
    v = v.repeat_interleave(heads_q // heads_kv, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * d**-0.5
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    ref = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())
    ref_lse = scores.logsumexp(-1)
    for b, length in enumerate([3, sq]):
        torch.testing.assert_close(
            out[b, :length].float(), ref[b, :length], atol=6e-3, rtol=6e-3
        )
        torch.testing.assert_close(
            lse[b, :, :length], ref_lse[b, :, :length], atol=2e-3, rtol=2e-3
        )


def test_paged_partial_loader_tile():
    """A 96-row KV tile uses a partial wave of 128 page-table loader threads."""
    torch.manual_seed(3)
    q = torch.randn(2, 7, 3, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(5, 16, 1, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    page_table = torch.tensor([[3, 1], [4, 2]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([27, 31], device="cuda", dtype=torch.int32)
    out, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        page_table=page_table,
        seqused_k=lengths,
        tile_mn=(64, 96),
        pack_gqa=True,
        return_lse=True,
    )
    for b, length in enumerate([27, 31]):
        kb = k[page_table[b].long()].reshape(32, 1, 64)[:length].expand(-1, 3, -1)
        vb = v[page_table[b].long()].reshape(32, 1, 64)[:length].expand(-1, 3, -1)
        scores = torch.einsum("qhd,khd->hqk", q[b].float(), kb.float()) / 8
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vb.float())
        torch.testing.assert_close(out[b].float(), ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(lse[b], scores.logsumexp(-1), atol=2e-3, rtol=2e-3)
