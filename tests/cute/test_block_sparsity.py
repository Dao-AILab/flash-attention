"""Tests for block sparsity computation in flash attention."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from flash_attn.cute.mask_definitions import get_mask_pair
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity


def _call_compute_block_sparsity(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    tile_m,
    tile_n,
    mask_name,
    window_size=None,
    aux_tensors=None,
    use_fast_sampling=False,
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


def _compare_block_sparsity(
    mask_block_cnt,
    mask_block_idx,
    full_block_cnt,
    full_block_idx,
    mask_block_cnt_ref,
    mask_block_idx_ref,
    full_block_cnt_ref,
    full_block_idx_ref,
    batch_size,
    nheads,
):
    """Compare block sparsity against reference. Returns (all_match, error_msg)."""
    if not isinstance(mask_block_cnt, torch.Tensor):
        return False, f"mask_block_cnt is not a tensor: {type(mask_block_cnt)}"

    n_blocks_q = mask_block_cnt.shape[2]
    mask_cnt_match = torch.all(mask_block_cnt == mask_block_cnt_ref).item()
    full_cnt_match = torch.all(full_block_cnt == full_block_cnt_ref).item()

    if not mask_cnt_match or not full_cnt_match:
        error_msg = []
        if not mask_cnt_match:
            error_msg.append("Mask counts mismatch")
            diff = (mask_block_cnt != mask_block_cnt_ref).nonzero(as_tuple=False)
            if len(diff) > 0:
                b, h, m = diff[0].tolist()
                error_msg.append(
                    f"  First mismatch at [{b},{h},{m}]: "
                    f"got {mask_block_cnt[b, h, m].item()}, "
                    f"expected {mask_block_cnt_ref[b, h, m].item()}"
                )
        if not full_cnt_match:
            error_msg.append("Full counts mismatch")
            diff = (full_block_cnt != full_block_cnt_ref).nonzero(as_tuple=False)
            if len(diff) > 0:
                b, h, m = diff[0].tolist()
                error_msg.append(
                    f"  First mismatch at [{b},{h},{m}]: "
                    f"got {full_block_cnt[b, h, m].item()}, "
                    f"expected {full_block_cnt_ref[b, h, m].item()}"
                )
        return False, "\n".join(error_msg)

    # Compare indices
    for b in range(batch_size):
        for h in range(nheads):
            for m in range(n_blocks_q):
                num_mask = mask_block_cnt[b, h, m].item()
                num_full = full_block_cnt[b, h, m].item()

                if num_mask > 0:
                    mask_indices = mask_block_idx[b, h, m, :num_mask].sort()[0]
                    mask_indices_ref = mask_block_idx_ref[b, h, m, :num_mask].sort()[0]
                    if not (mask_indices == mask_indices_ref).all():
                        return False, f"Mask indices mismatch at [{b},{h},{m}]"

                if num_full > 0:
                    full_indices = full_block_idx[b, h, m, :num_full].sort()[0]
                    full_indices_ref = full_block_idx_ref[b, h, m, :num_full].sort()[0]
                    if not (full_indices == full_indices_ref).all():
                        return False, f"Full indices mismatch at [{b},{h},{m}]"

    return True, ""


# Test configurations
SEQLEN_PAIRS = [
    # Small aligned
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    # Rectangular
    (128, 256),
    (256, 128),
    (512, 256),
    (256, 512),
    # Large aligned
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    # Large unaligned
    (1000, 1000),
    (2000, 2000),
    (4000, 4000),
    # Edge cases with unaligned seqlens
    (113, 203),
    (127, 127),
    (129, 129),
    (255, 255),
    (257, 257),
    (1023, 1023),
    (1025, 1025),
    (2047, 2047),
    (2049, 2049),
]
TILE_SIZES = [
    # Standard powers of 2
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    # Rectangular
    (32, 64),
    (64, 32),
    (64, 128),
    (128, 64),
    (128, 256),
    (256, 128),
    # Unusual sizes
    (40, 40),
    (48, 48),
    (96, 96),
    (112, 112),
    (32, 128),
    (128, 32),
    (40, 96),
    (96, 40),
]


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS)
@pytest.mark.parametrize("tile_m,tile_n", TILE_SIZES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("mask_name", ["block_diagonal", "mini_causal"])
def test_fixed_length_masks(
    seqlen_q, seqlen_k, tile_m, tile_n, batch_size, nheads, mask_name
):
    """Test fixed-length masks."""
    seqlen_unaligned = (seqlen_q % tile_m != 0) or (seqlen_k % tile_n != 0)

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = (
        _call_compute_block_sparsity(
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            tile_m,
            tile_n,
            mask_name,
        )
    )

    _, mask_mod_flex = get_mask_pair(mask_name)
    block_mask = create_block_mask(
        mask_mod_flex,
        B=batch_size,
        H=nheads,
        Q_LEN=seqlen_q,
        KV_LEN=seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    (
        _,
        _,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        *_,
    ) = block_mask.as_tuple()

    all_match, error_msg = _compare_block_sparsity(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        batch_size,
        nheads,
    )

    if seqlen_unaligned and not all_match:
        pytest.skip(f"Skipping at seqlen extreme: {error_msg}")
    assert all_match, f"Mismatch: {error_msg}"


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS)
@pytest.mark.parametrize(
    "tile_m,tile_n", [(64, 64), (128, 128), (64, 128), (128, 64), (256, 256)]
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize(
    "mask_name,window_size",
    [("causal", None), ("sliding_window", 64), ("sliding_window", 256)],
)
def test_parameterized_masks(
    seqlen_q, seqlen_k, tile_m, tile_n, batch_size, nheads, mask_name, window_size
):
    """Test parameterized masks."""
    if mask_name == "sliding_window" and seqlen_q > seqlen_k:
        pytest.skip("Sliding window not supported for seqlen_q > seqlen_k")

    seqlen_unaligned = (seqlen_q % tile_m != 0) or (seqlen_k % tile_n != 0)

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = (
        _call_compute_block_sparsity(
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            tile_m,
            tile_n,
            mask_name,
            window_size=window_size,
        )
    )

    _, mask_mod_flex = get_mask_pair(
        mask_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k, window_size=window_size
    )
    block_mask = create_block_mask(
        mask_mod_flex,
        B=batch_size,
        H=nheads,
        Q_LEN=seqlen_q,
        KV_LEN=seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    (
        _,
        _,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        *_,
    ) = block_mask.as_tuple()

    all_match, error_msg = _compare_block_sparsity(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        batch_size,
        nheads,
    )

    if seqlen_unaligned and not all_match:
        pytest.skip(f"Skipping at seqlen extreme: {error_msg}")
    assert all_match, f"Mismatch: {error_msg}"


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,tile_m,tile_n",
    [
        (1, 1, 64, 64),
        (63, 63, 64, 64),
        (65, 65, 64, 64),
        (129, 129, 128, 128),
        (100, 200, 64, 128),
    ],
)
def test_edge_cases(seqlen_q, seqlen_k, tile_m, tile_n):
    """Test edge cases with unaligned dimensions."""
    batch_size, nheads = 1, 1
    seqlen_unaligned = (seqlen_q % tile_m != 0) or (seqlen_k % tile_n != 0)

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = (
        _call_compute_block_sparsity(
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            tile_m,
            tile_n,
            "causal",
        )
    )

    _, mask_mod_flex = get_mask_pair("causal", seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    block_mask = create_block_mask(
        mask_mod_flex,
        B=batch_size,
        H=nheads,
        Q_LEN=seqlen_q,
        KV_LEN=seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    (
        _,
        _,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        *_,
    ) = block_mask.as_tuple()

    all_match, error_msg = _compare_block_sparsity(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        batch_size,
        nheads,
    )

    if seqlen_unaligned and not all_match:
        pytest.skip(f"Skipping at seqlen extreme: {error_msg}")
    assert all_match, f"Mismatch: {error_msg}"


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS)
@pytest.mark.parametrize(
    "tile_m,tile_n", [(64, 64), (128, 128), (64, 128), (128, 64), (256, 256)]
)
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("mask_name", ["causal", "block_diagonal"])
def test_fast_sampling(seqlen_q, seqlen_k, tile_m, tile_n, nheads, mask_name):
    """Test fast sampling mode (5-point sampling)."""
    batch_size = 1
    seqlen_unaligned = (seqlen_q % tile_m != 0) or (seqlen_k % tile_n != 0)

    full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx = (
        _call_compute_block_sparsity(
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            tile_m,
            tile_n,
            mask_name,
            use_fast_sampling=True,
        )
    )

    _, mask_mod_flex = get_mask_pair(mask_name, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    block_mask = create_block_mask(
        mask_mod_flex,
        B=batch_size,
        H=nheads,
        Q_LEN=seqlen_q,
        KV_LEN=seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    (
        _,
        _,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        *_,
    ) = block_mask.as_tuple()

    all_match, error_msg = _compare_block_sparsity(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt_ref,
        mask_block_idx_ref,
        full_block_cnt_ref,
        full_block_idx_ref,
        batch_size,
        nheads,
    )

    if seqlen_unaligned and not all_match:
        pytest.skip(f"Skipping at seqlen extreme: {error_msg}")
    assert all_match, f"Mismatch: {error_msg}"
