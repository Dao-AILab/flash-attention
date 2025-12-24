# Copyright (c) 2025, Anthropic.
# Tests for cute-based paged attention functionality.

import math
import pytest
import torch
from einops import rearrange

# Import directly from cute module to avoid flash_attn_2_cuda dependency
from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def generate_paged_kvcache(
    seqlen_k: int,
    page_size: int,
    batch_size: int,
    nheads_k: int,
    d: int,
    dv: int,
    device: str,
    dtype: torch.dtype,
    fragmented: bool = True,
):
    """
    Generate paged KV cache with optional fragmentation.

    Args:
        seqlen_k: Total sequence length for keys/values
        page_size: Size of each page
        batch_size: Batch size
        nheads_k: Number of KV heads
        d: Head dimension for keys
        dv: Head dimension for values
        device: Device to create tensors on
        dtype: Data type for tensors
        fragmented: If True, randomize page table order (realistic scenario)
                   If False, use sequential pages (best-case scenario)

    Returns:
        k_cache: (batch_size, seqlen_k, nheads_k, d) - unpaged view for reference
        v_cache: (batch_size, seqlen_k, nheads_k, dv) - unpaged view for reference
        page_table: (batch_size, num_blocks_per_seq) - page indices
        k_cache_paged: (num_blocks, page_size, nheads_k, d) - paged storage
        v_cache_paged: (num_blocks, page_size, nheads_k, dv) - paged storage
    """
    num_blocks_per_seq = math.ceil(seqlen_k / page_size)

    if fragmented:
        # Allocate extra blocks (3x) to simulate realistic fragmented memory
        num_blocks = num_blocks_per_seq * batch_size * 3
    else:
        num_blocks = num_blocks_per_seq * batch_size

    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype
    )

    if fragmented:
        # Randomized page table to simulate fragmented allocation
        page_table = rearrange(
            torch.randperm(num_blocks, dtype=torch.int32, device=device),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )[:, :num_blocks_per_seq]
    else:
        # Sequential page table (best case)
        page_table = rearrange(
            torch.arange(num_blocks, dtype=torch.int32, device=device),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )

    # Create unpaged view for reference computations
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]

    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 512, 1024])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
def test_paged_attn_correctness(
    dtype,
    causal,
    page_size,
    headdim,
    seqlen,
    mha_type,
):
    """Test that paged attention produces the same output as non-paged attention."""
    if seqlen % page_size != 0:
        pytest.skip("seqlen must be divisible by page_size")

    device = "cuda"
    torch.manual_seed(42)

    batch_size = 4
    nheads = 8
    nheads_k = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    headdim_v = headdim

    # Generate query
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    # Generate paged KV cache
    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim_v,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    # Run paged attention using varlen interface
    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        causal=causal,
    )

    # Run non-paged attention for reference
    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=causal,
    )

    # Check outputs match
    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Paged attention output differs from reference. "
        f"Max diff: {(out_paged - out_ref).abs().max().item()}, "
        f"Mean diff: {(out_paged - out_ref).abs().mean().item()}"
    )


@pytest.mark.parametrize("fragmented", [True, False])
@pytest.mark.parametrize("page_size", [32, 64, 128])
def test_paged_attn_fragmented_vs_contiguous(fragmented, page_size):
    """Test paged attention with fragmented vs contiguous page tables."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(123)

    batch_size = 4
    seqlen = 512
    nheads = 8
    nheads_k = 8
    headdim = 64

    if seqlen % page_size != 0:
        pytest.skip("seqlen must be divisible by page_size")

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    # Generate KV cache with specified fragmentation
    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim,
        device=device,
        dtype=dtype,
        fragmented=fragmented,
    )

    # Run paged attention
    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        causal=True,
    )

    # Run non-paged attention for reference
    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=True,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Paged attention ({'fragmented' if fragmented else 'contiguous'}) differs from reference. "
        f"Max diff: {(out_paged - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("page_size", [16, 32, 64, 128, 256])
def test_paged_attn_various_page_sizes(page_size):
    """Test paged attention with various page sizes."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(456)

    batch_size = 2
    seqlen = 1024
    nheads = 8
    nheads_k = 8
    headdim = 64

    if seqlen % page_size != 0:
        pytest.skip("seqlen must be divisible by page_size")

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        causal=True,
    )

    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=True,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Paged attention with page_size={page_size} differs from reference. "
        f"Max diff: {(out_paged - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (1, 128),      # Single query token (decode)
    (64, 512),     # Short query, longer KV
    (128, 128),    # Equal lengths
    (256, 1024),   # Prefill scenario
])
def test_paged_attn_different_seqlens(seqlen_q, seqlen_k):
    """Test paged attention with different query and key sequence lengths."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(789)

    batch_size = 2
    nheads = 8
    nheads_k = 8
    headdim = 64
    page_size = 64

    if seqlen_k % page_size != 0:
        pytest.skip("seqlen_k must be divisible by page_size")

    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen_k,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    # For non-equal lengths, use seqused_k to indicate actual sequence length
    seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)

    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        seqused_k=seqused_k,
        causal=True if seqlen_q <= seqlen_k else False,
    )

    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=True if seqlen_q <= seqlen_k else False,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Paged attention with seqlen_q={seqlen_q}, seqlen_k={seqlen_k} differs. "
        f"Max diff: {(out_paged - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_paged_attn_batch_sizes(batch_size):
    """Test paged attention with various batch sizes."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(321)

    seqlen = 256
    nheads = 8
    nheads_k = 8
    headdim = 64
    page_size = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        causal=True,
    )

    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=True,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Paged attention with batch_size={batch_size} differs. "
        f"Max diff: {(out_paged - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("headdim,headdim_v", [
    (64, 64),
    (128, 128),
    (64, 128),  # Different K and V head dimensions
])
def test_paged_attn_head_dimensions(headdim, headdim_v):
    """Test paged attention with various head dimensions."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(654)

    batch_size = 2
    seqlen = 256
    nheads = 8
    nheads_k = 8
    page_size = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim_v,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        causal=True,
    )

    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=True,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Paged attention with headdim={headdim}, headdim_v={headdim_v} differs. "
        f"Max diff: {(out_paged - out_ref).abs().max().item()}"
    )


def test_paged_attn_single_page():
    """Test paged attention when sequence fits in a single page."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(111)

    batch_size = 2
    seqlen = 64
    nheads = 8
    nheads_k = 8
    headdim = 64
    page_size = 64  # Same as seqlen - single page

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        causal=True,
    )

    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=True,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Single-page attention differs. Max diff: {(out_paged - out_ref).abs().max().item()}"
    )


def test_paged_attn_many_pages():
    """Test paged attention with many small pages."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(222)

    batch_size = 2
    seqlen = 2048
    nheads = 8
    nheads_k = 8
    headdim = 64
    page_size = 32  # Many pages

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        causal=True,
    )

    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        causal=True,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Many-page attention differs. Max diff: {(out_paged - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("softmax_scale", [None, 0.1, 0.5])
def test_paged_attn_softmax_scale(softmax_scale):
    """Test paged attention with different softmax scales."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(333)

    batch_size = 2
    seqlen = 256
    nheads = 8
    nheads_k = 8
    headdim = 64
    page_size = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged = generate_paged_kvcache(
        seqlen_k=seqlen,
        page_size=page_size,
        batch_size=batch_size,
        nheads_k=nheads_k,
        d=headdim,
        dv=headdim,
        device=device,
        dtype=dtype,
        fragmented=True,
    )

    out_paged, _ = flash_attn_varlen_func(
        q,
        k_cache_paged,
        v_cache_paged,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=True,
    )

    out_ref, _ = flash_attn_func(
        q,
        k_cache,
        v_cache,
        softmax_scale=softmax_scale,
        causal=True,
    )

    atol = 1e-2
    rtol = 1e-2
    assert torch.allclose(out_paged.float(), out_ref.float(), atol=atol, rtol=rtol), (
        f"Paged attention with softmax_scale={softmax_scale} differs. "
        f"Max diff: {(out_paged - out_ref).abs().max().item()}"
    )
