#!/usr/bin/env python3
"""
Semantic validation tests for block sparsity computation.

Instead of comparing against PyTorch's create_block_mask (which may use different heuristics),
these tests validate that the kernel's block classification is **semantically correct**:

1. FULL blocks: all in-bounds values are unmasked
2. PARTIAL blocks: mix of masked and unmasked in-bounds values
3. SKIP blocks: all in-bounds values are masked
"""

import pytest
import torch
from flash_attn.cute.mask_definitions import get_mask_pair
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity


def call_compute_block_sparsity(
    batch_size, nheads, seqlen_q, seqlen_k, tile_m, tile_n, mask_name,
    window_size=None, aux_tensors=None, use_fast_sampling=False
):
    """Call compute_block_sparsity and return torch tensors."""
    cute_mask, _ = get_mask_pair(
        mask_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k, window_size=window_size
    )

    blocksparse_tensors, torch_tensors = compute_block_sparsity(
        tile_m=tile_m,
        tile_n=tile_n,
        batch_size=batch_size,
        num_heads=nheads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        mask_mod=cute_mask,
        aux_tensors=aux_tensors,
        device="cuda",
        use_fast_sampling=use_fast_sampling,
    )

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = torch_tensors
    return full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx


def validate_block_sparsity_semantics(
    mask_block_cnt,
    mask_block_idx,
    full_block_cnt,
    full_block_idx,
    seqlen_q,
    seqlen_k,
    tile_m,
    tile_n,
    batch_size,
    nheads,
    mask_mod_flex,
    verbose=False
):
    """
    Validate that kernel's block classifications are semantically correct.

    Returns:
        tuple: (is_valid, error_msg)
    """
    errors = []
    n_blocks_q = mask_block_cnt.shape[2]
    n_blocks_k = (seqlen_k + tile_n - 1) // tile_n

    for b in range(batch_size):
        for h in range(nheads):
            for m in range(n_blocks_q):
                m_start = m * tile_m
                m_end = min((m + 1) * tile_m, seqlen_q)

                # Get block indices for this m_block
                num_mask = mask_block_cnt[b, h, m].item()
                num_full = full_block_cnt[b, h, m].item()

                mask_blocks = set()
                if num_mask > 0:
                    mask_blocks = set(mask_block_idx[b, h, m, :num_mask].cpu().tolist())

                full_blocks = set()
                if num_full > 0:
                    full_blocks = set(full_block_idx[b, h, m, :num_full].cpu().tolist())

                # Check each n_block
                for n in range(n_blocks_k):
                    n_start = n * tile_n
                    n_end = min((n + 1) * tile_n, seqlen_k)

                    # Count unmasked vs masked in this block
                    unmasked = 0
                    masked = 0

                    for q_idx in range(m_start, m_end):
                        for kv_idx in range(n_start, n_end):
                            mask_val = mask_mod_flex(0, 0, q_idx, kv_idx)
                            if mask_val:
                                unmasked += 1
                            else:
                                masked += 1

                    # Determine expected classification
                    if masked == 0 and unmasked > 0:
                        expected = "FULL"
                    elif masked > 0 and unmasked > 0:
                        expected = "PARTIAL"
                    elif masked > 0 and unmasked == 0:
                        expected = "SKIP"
                    else:
                        expected = "ERROR"  # No elements?

                    # Get actual classification from kernel
                    if n in full_blocks:
                        actual = "FULL"
                    elif n in mask_blocks:
                        actual = "PARTIAL"
                    else:
                        actual = "SKIP"

                    # Check if they match
                    if expected != actual:
                        errors.append(
                            f"[{b},{h},{m},{n}]: expected {expected}, got {actual} "
                            f"(unmasked={unmasked}, masked={masked})"
                        )

    if verbose and not errors:
        print("✓ All block sparsity classifications are semantically correct!")

    return len(errors) == 0, "\n".join(errors) if errors else ""


# Test parameters
SEQLEN_PAIRS = [
    (128, 128),
    (256, 256),
    (128, 256),
    (256, 128),
    (113, 203),
    (1024, 1024),
    (1023, 1024),
]

TILE_SIZES = [
    (64, 64),
    (128, 128),
    (64, 128),
]


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS)
@pytest.mark.parametrize("tile_m,tile_n", TILE_SIZES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("mask_name", ["block_diagonal", "mini_causal"])
def test_fixed_length_masks_semantic(
    seqlen_q, seqlen_k, tile_m, tile_n, batch_size, nheads, mask_name
):
    """Test that fixed-length masks are classified correctly according to semantics."""
    print(f"\nTesting {mask_name}: B={batch_size}, H={nheads}, "
          f"Q={seqlen_q}, K={seqlen_k}, tile={tile_m},{tile_n}")

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = call_compute_block_sparsity(
        batch_size=batch_size,
        nheads=nheads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
        mask_name=mask_name,
    )

    _, mask_mod_flex = get_mask_pair(mask_name)

    is_valid, error_msg = validate_block_sparsity_semantics(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        seqlen_q,
        seqlen_k,
        tile_m,
        tile_n,
        batch_size,
        nheads,
        mask_mod_flex,
        verbose=True,
    )

    assert is_valid, f"Semantic validation failed:\n{error_msg}"


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS)
@pytest.mark.parametrize("tile_m,tile_n", [(128, 128), (64, 128)])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize(
    "mask_name,window_size",
    [
        ("causal", None),
        ("sliding_window", 128),
        ("sliding_window", 256),
    ],
)
def test_parameterized_masks_semantic(
    seqlen_q, seqlen_k, tile_m, tile_n, batch_size, nheads, mask_name, window_size
):
    """Test that parameterized masks are classified correctly according to semantics."""
    if mask_name == "sliding_window" and seqlen_q > seqlen_k:
        pytest.skip("Sliding window not supported for seqlen_q > seqlen_k")

    print(f"\nTesting {mask_name}: B={batch_size}, H={nheads}, "
          f"Q={seqlen_q}, K={seqlen_k}, tile={tile_m},{tile_n}, window={window_size}")

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = call_compute_block_sparsity(
        batch_size=batch_size,
        nheads=nheads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
        mask_name=mask_name,
        window_size=window_size,
    )

    _, mask_mod_flex = get_mask_pair(
        mask_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k, window_size=window_size
    )

    is_valid, error_msg = validate_block_sparsity_semantics(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        seqlen_q,
        seqlen_k,
        tile_m,
        tile_n,
        batch_size,
        nheads,
        mask_mod_flex,
        verbose=True,
    )

    assert is_valid, f"Semantic validation failed:\n{error_msg}"
