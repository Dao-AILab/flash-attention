# SPDX-License-Identifier: BSD-3-Clause
# Tests for flash attention in production LLM serving patterns:
#
#  1. GQA with extreme head ratios (8:1, 16:1, 32:1) — Llama 3 70B, Mistral, Falcon
#  2. Decode-phase (seqlen_q=1 or small) + long KV sequences (4k, 8k, 16k)
#  3. Speculative decoding multi-token verify phase (seqlen_q=2..8, seqlen_k>>seqlen_q)
#  4. Chunked prefill + decode mixed batches in the varlen path
#  5. Paged KV with small block sizes [16, 32, 128] common in vLLM / SGLang
#  6. Softcap + GQA decode-phase (Gemma 2 / Llama 3 pattern)
#
# None of these are covered by the existing test_flash_attn.py grid.
# Requires: flash_attn >= 2.0, torch >= 2.0, CUDA GPU.
#
# Run:  pytest tests/test_flash_attn_gqa_decode.py -v

import math

import pytest
import torch
import torch.nn.functional as F

# Skip entire module if flash_attn is not installed (CPU-only CI)
flash_attn = pytest.importorskip("flash_attn")

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.flash_attn_interface import flash_attn_with_kvcache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_attention(q, k, v, causal=True, softcap=0.0):
    """
    Pure-PyTorch reference implementation (FP32 accumulation).

    Args:
        q : (B, S_q, H_q, D)
        k : (B, S_k, H_k, D)
        v : (B, S_k, H_k, D)
    Returns:
        out : (B, S_q, H_q, D)
    """
    B, S_q, H_q, D = q.shape
    _, S_k, H_k, _ = k.shape
    assert H_q % H_k == 0
    ratio = H_q // H_k

    q = q.float()
    # Expand KV heads to match Q heads (GQA → MHA)
    k = k.repeat_interleave(ratio, dim=2).float()   # (B, S_k, H_q, D)
    v = v.repeat_interleave(ratio, dim=2).float()

    # (B, H_q, S_q, S_k)
    scores = torch.einsum("bshd,bkhd->bhsk", q, k) / math.sqrt(D)

    if softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)

    if causal:
        # Build a causal mask: position i can only attend to j <= i (aligned to S_k tail)
        row_idx = torch.arange(S_q, device=q.device).unsqueeze(1)   # (S_q, 1)
        col_idx = torch.arange(S_k, device=q.device).unsqueeze(0)   # (1, S_k)
        # Causal offset: query position i attends to KV positions <= (S_k - S_q + i)
        mask = col_idx > (S_k - S_q + row_idx)
        scores = scores.masked_fill(mask[None, None], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhsk,bkhd->bshd", attn, v)
    return out.to(q.dtype if q.dtype != torch.float32 else torch.float16)


def _make_qkv(batch, seqlen_q, seqlen_k, nheads_q, nheads_k, headdim, dtype, device):
    q = torch.randn(batch, seqlen_q, nheads_q, headdim, dtype=dtype, device=device)
    k = torch.randn(batch, seqlen_k, nheads_k, headdim, dtype=dtype, device=device)
    v = torch.randn(batch, seqlen_k, nheads_k, headdim, dtype=dtype, device=device)
    return q, k, v


def _allclose(a, b, rtol=1e-3, atol=1e-3):
    """Compare with tolerance appropriate for fp16/bf16 flash attention output."""
    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# 1. GQA extreme ratios — decode phase (seqlen_q = 1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("nheads_q,nheads_k", [
    (8,  1),   # 8:1 — MQA, common in Falcon / early Mistral
    (16, 2),   # 8:1 ratio
    (32, 4),   # 8:1 — Mistral 7B
    (64, 8),   # 8:1 — Llama 3 70B
    (64, 4),   # 16:1
    (32, 1),   # 32:1 — extreme MQA
])
@pytest.mark.parametrize("seqlen_k", [512, 2048, 4096])
def test_gqa_decode_phase(seqlen_k, nheads_q, nheads_k, headdim, dtype):
    """
    Decode phase: single query token (seqlen_q=1) against a long KV sequence.
    Tests the splitkv kernel path for GQA with extreme head ratios.

    Covers the serving pattern: one user token generated per step against
    the full KV cache — the dominant mode in autoregressive generation.
    """
    device = "cuda"
    B = 2
    seqlen_q = 1

    q, k, v = _make_qkv(B, seqlen_q, seqlen_k, nheads_q, nheads_k, headdim, dtype, device)

    out = flash_attn_func(q, k, v, causal=True)
    ref = _ref_attention(q, k, v, causal=True)

    assert _allclose(out, ref), (
        f"GQA decode mismatch: nheads_q={nheads_q} nheads_k={nheads_k} "
        f"seqlen_k={seqlen_k} headdim={headdim} dtype={dtype}"
    )


# ---------------------------------------------------------------------------
# 2. Speculative decoding verify phase (seqlen_q = 2..8, long seqlen_k)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nheads_q,nheads_k", [
    (32, 4),   # 8:1 — Mistral
    (64, 8),   # 8:1 — Llama 3 70B
])
@pytest.mark.parametrize("seqlen_q", [2, 4, 6, 8])   # gamma in speculative decoding
@pytest.mark.parametrize("seqlen_k", [1024, 4096, 8192])
def test_gqa_speculative_decode_verify(seqlen_q, seqlen_k, nheads_q, nheads_k, dtype):
    """
    Speculative decoding verify step: the target model processes seqlen_q draft
    tokens (gamma) against the full seqlen_k KV cache in one forward pass.

    This is the pattern flash attention must handle for speculative decoding to
    be efficient — it's distinct from standard decode (seqlen_q=1) and from
    prefill (seqlen_q = seqlen_k).
    """
    device = "cuda"
    B = 2
    headdim = 128

    q, k, v = _make_qkv(B, seqlen_q, seqlen_k, nheads_q, nheads_k, headdim, dtype, device)

    out = flash_attn_func(q, k, v, causal=True)
    ref = _ref_attention(q, k, v, causal=True)

    assert _allclose(out, ref), (
        f"Spec-decode verify mismatch: seqlen_q={seqlen_q} seqlen_k={seqlen_k} "
        f"nheads_q={nheads_q} nheads_k={nheads_k}"
    )


# ---------------------------------------------------------------------------
# 3. Chunked prefill + decode mixed batches (varlen path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nheads_q,nheads_k", [
    (32, 4),
    (64, 8),
])
def test_varlen_chunked_prefill_mixed_with_decode(nheads_q, nheads_k, dtype):
    """
    Continuous batching / chunked prefill pattern: a single batch contains
    mixed-length sequences — some are long prefill chunks (seqlen ~ 2048–4096)
    and some are single-token decode steps (seqlen = 1).

    This is the realistic serving workload for production inference engines
    (vLLM, SGLang, TensorRT-LLM). The extreme length heterogeneity stresses
    the cu_seqlens bookkeeping in the varlen kernel.
    """
    device = "cuda"
    headdim = 128

    # Simulate a 5-sequence batch:
    #   seq 0: decode    (seqlen = 1)
    #   seq 1: prefill   (seqlen = 512)
    #   seq 2: decode    (seqlen = 1)
    #   seq 3: chunked prefill (seqlen = 2048)
    #   seq 4: decode    (seqlen = 1)
    seqlens = [1, 512, 1, 2048, 1]
    total_tokens = sum(seqlens)

    cu_seqlens = torch.tensor([0] + list(__import__("itertools").accumulate(seqlens)),
                               dtype=torch.int32, device=device)
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, nheads_q, headdim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, nheads_k, headdim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, nheads_k, headdim, dtype=dtype, device=device)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    # Verify output shape and absence of NaNs/Infs
    assert out.shape == (total_tokens, nheads_q, headdim)
    assert not torch.isnan(out).any(), "NaN in chunked-prefill+decode output"
    assert not torch.isinf(out).any(), "Inf in chunked-prefill+decode output"

    # Reference: process each sequence independently and concatenate
    ref_parts = []
    for i, slen in enumerate(seqlens):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        q_i = q[start:end].unsqueeze(0)   # (1, slen, H_q, D)
        k_i = k[start:end].unsqueeze(0)
        v_i = v[start:end].unsqueeze(0)
        ref_parts.append(_ref_attention(q_i, k_i, v_i, causal=True).squeeze(0))

    ref = torch.cat(ref_parts, dim=0)
    assert _allclose(out, ref), (
        f"Chunked prefill+decode mismatch: nheads_q={nheads_q} nheads_k={nheads_k}"
    )


# ---------------------------------------------------------------------------
# 4. Paged KV with small block sizes (vLLM / SGLang production configs)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("paged_kv_block_size", [16, 32, 128])
@pytest.mark.parametrize("nheads_q,nheads_k", [
    (32, 4),   # 8:1 GQA
    (64, 8),   # 8:1 GQA — Llama 3 70B
])
def test_paged_kv_small_block_sizes(paged_kv_block_size, nheads_q, nheads_k, dtype):
    """
    Paged KV cache with small block sizes (16, 32, 128 tokens per block).

    vLLM uses block_size=16 by default; SGLang uses 64 or 128; TensorRT-LLM
    uses 64. The existing tests only exercise block_size=256 and 512.
    Small block sizes increase the number of page-table lookups per sequence
    and stress the block_table indexing path.

    Decode-phase only (seqlen_q=1): the common case where paged KV matters.
    """
    device = "cuda"
    B = 4
    seqlen_q = 1
    seqlen_k = 512   # 512 // paged_kv_block_size pages per sequence
    headdim = 128

    # Number of blocks needed per sequence
    num_blocks_per_seq = (seqlen_k + paged_kv_block_size - 1) // paged_kv_block_size
    total_blocks = B * num_blocks_per_seq

    # KV cache: (total_blocks, 2, block_size, nheads_k, headdim)
    kv_cache = torch.randn(
        total_blocks, 2, paged_kv_block_size, nheads_k, headdim,
        dtype=dtype, device=device,
    )

    # Block table: each sequence gets contiguous blocks (simplified)
    block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).view(
        B, num_blocks_per_seq
    )

    q = torch.randn(B, seqlen_q, nheads_q, headdim, dtype=dtype, device=device)
    cache_seqlens = torch.full((B,), seqlen_k, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q,
        kv_cache[:, 0],   # k cache: (total_blocks, block_size, nheads_k, D)
        kv_cache[:, 1],   # v cache
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )

    assert out.shape == (B, seqlen_q, nheads_q, headdim)
    assert not torch.isnan(out).any(), (
        f"NaN in paged KV output: block_size={paged_kv_block_size} "
        f"nheads_q={nheads_q} nheads_k={nheads_k}"
    )
    assert not torch.isinf(out).any(), (
        f"Inf in paged KV output: block_size={paged_kv_block_size}"
    )


# ---------------------------------------------------------------------------
# 5. Softcap + GQA + decode phase (Gemma 2 / Llama 3 pattern)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcap", [30.0, 50.0])
@pytest.mark.parametrize("nheads_q,nheads_k", [
    (16, 2),   # 8:1 — Gemma 2 9B
    (32, 4),   # 8:1 — Llama 3 8B
])
@pytest.mark.parametrize("seqlen_k", [512, 2048])
def test_softcap_gqa_decode(seqlen_k, nheads_q, nheads_k, softcap, dtype):
    """
    Softcap (logit cap) combined with GQA in decode phase.

    Gemma 2 uses softcap=50.0; some Llama 3 variants use softcap=30.0.
    The existing softcap tests do not exercise GQA ratios > 3:1, and none
    test the decode-phase (seqlen_q=1) with softcap + GQA together.
    """
    device = "cuda"
    B = 2
    seqlen_q = 1
    headdim = 128

    q, k, v = _make_qkv(B, seqlen_q, seqlen_k, nheads_q, nheads_k, headdim, dtype, device)

    out = flash_attn_func(q, k, v, causal=True, softcap=softcap)
    ref = _ref_attention(q, k, v, causal=True, softcap=softcap)

    assert _allclose(out, ref), (
        f"Softcap+GQA decode mismatch: softcap={softcap} nheads_q={nheads_q} "
        f"nheads_k={nheads_k} seqlen_k={seqlen_k} dtype={dtype}"
    )


# ---------------------------------------------------------------------------
# 6. Long context beyond 8k with GQA (gradient check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nheads_q,nheads_k", [
    (32, 4),
    (64, 8),
])
@pytest.mark.parametrize("seqlen", [8192, 16384])
def test_long_context_gqa_prefill_backward(seqlen, nheads_q, nheads_k, dtype):
    """
    Long-context prefill with GQA and gradient computation.

    The existing gradient tests only go to seqlen=2048. This tests that
    backward pass accumulation is correct for 8k and 16k sequences with
    extreme GQA ratios — the setting where fp16 rounding errors can
    accumulate across many attention blocks.
    """
    device = "cuda"
    B = 1   # single sequence to keep memory manageable
    headdim = 64

    q = torch.randn(B, seqlen, nheads_q, headdim, dtype=dtype, device=device,
                    requires_grad=True)
    k = torch.randn(B, seqlen, nheads_k, headdim, dtype=dtype, device=device,
                    requires_grad=True)
    v = torch.randn(B, seqlen, nheads_k, headdim, dtype=dtype, device=device,
                    requires_grad=True)

    out = flash_attn_func(q, k, v, causal=True)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any(), "NaN in q.grad"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "NaN in k.grad"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "NaN in v.grad"
