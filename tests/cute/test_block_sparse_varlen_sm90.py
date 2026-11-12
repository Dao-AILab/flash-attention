"""Regression tests for the packed block-index row stride on SM90."""

from itertools import accumulate

import cutlass.cute as cute
import pytest
import torch

from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
from flash_attn.cute.interface import _flash_attn_fwd


@cute.jit
def causal_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    return n_idx <= m_idx


@cute.jit
def partial_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    return (n_idx <= m_idx) & (n_idx % 2 == 0)


@cute.jit
def full_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    return n_idx // 128 == m_idx // 128


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90-only regression",
)
@pytest.mark.parametrize(
    "head_dim,tile_n",
    [(256, 64), (128, 64), (128, 128)],
    ids=["hd256-kv64", "hd128-kv64", "hd128-kv128"],
)
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [([384], [384]), ([257, 385, 129], [321, 513, 193])],
    ids=["aligned", "ragged-packed"],
)
@pytest.mark.parametrize("mask_kind", ["partial", "full", "mixed"])
def test_varlen_block_sparse_tile_stride(head_dim, tile_n, seqlens_q, seqlens_k, mask_kind):
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    tile_m, num_heads = 128, 2
    mask_mod = {"partial": partial_mask, "full": full_mask, "mixed": causal_mask}[mask_kind]
    q = torch.randn(sum(seqlens_q), num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(sum(seqlens_k), num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)

    offsets_q = list(accumulate(seqlens_q, initial=0))
    offsets_k = list(accumulate(seqlens_k, initial=0))
    cu_seqlens_q = torch.tensor(offsets_q, device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor(offsets_k, device=device, dtype=torch.int32)
    m_blocks = [(sq + tile_m - 1) // tile_m for sq in seqlens_q]
    cu_total_m_blocks = torch.tensor(
        list(accumulate(m_blocks, initial=0)), device=device, dtype=torch.int32
    )
    sparse = compute_block_sparsity(
        tile_m=tile_m,
        tile_n=tile_n,
        batch_size=len(seqlens_q),
        num_heads=num_heads,
        seqlen_q=max(seqlens_q),
        seqlen_k=max(seqlens_k),
        mask_mod=mask_mod,
        aux_tensors=None,
        device=device,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_total_m_blocks=cu_total_m_blocks,
    )

    # Initialize unused index slots to a valid KV block. A wrong row stride then
    # reads defined, in-bounds indices and fails numerically, rather than relying
    # on uninitialized memory or an illegal access to expose the bug.
    m_offset, idx_offset = 0, 0
    for num_m, sk in zip(m_blocks, seqlens_k):
        num_n = (sk + tile_n - 1) // tile_n
        for counts, indices in (
            (sparse.mask_block_cnt, sparse.mask_block_idx),
            (sparse.full_block_cnt, sparse.full_block_idx),
        ):
            if indices is None:
                continue
            rows = indices[:, idx_offset : idx_offset + num_m * num_n].view(num_heads, num_m, num_n)
            unused = (
                torch.arange(num_n, device=device) >= counts[:, m_offset : m_offset + num_m, None]
            )
            rows.masked_fill_(unused, 0)
        m_offset += num_m
        idx_offset += num_m * num_n

    if mask_kind != "full":
        assert (sparse.mask_block_cnt > 0).any()
    else:
        assert (sparse.mask_block_cnt == 0).all()
    if mask_kind != "partial":
        assert (sparse.full_block_cnt > 0).any()
    else:
        assert (sparse.full_block_cnt == 0).all()

    out, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        tile_mn=(tile_m, tile_n),
        pack_gqa=False,
        mask_mod=mask_mod,
        block_sparse_tensors=sparse,
    )

    # Independent FP32 attention, using token indices rather than sparse metadata.
    for b, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        q_slice = q[offsets_q[b] : offsets_q[b + 1]].float().transpose(0, 1)
        k_slice = k[offsets_k[b] : offsets_k[b + 1]].float().transpose(0, 1)
        v_slice = v[offsets_k[b] : offsets_k[b + 1]].float().transpose(0, 1)
        scores = q_slice @ k_slice.transpose(-1, -2) / head_dim**0.5
        q_idx = torch.arange(sq, device=device)[:, None]
        k_idx = torch.arange(sk, device=device)[None, :]
        if mask_kind == "full":
            mask = q_idx // tile_m == k_idx // tile_m
        else:
            mask = k_idx <= q_idx
            if mask_kind == "partial":
                mask = mask & (k_idx % 2 == 0)
        probs = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        expected = (probs @ v_slice).transpose(0, 1)
        torch.testing.assert_close(
            out[offsets_q[b] : offsets_q[b + 1]].float(), expected, atol=1e-2, rtol=1e-2
        )
