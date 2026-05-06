"""Tests for block sparsity computation in flash attention."""

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from mask_mod_definitions import get_mask_pair
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
    torch_tensors = compute_block_sparsity(
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
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, *_ = torch_tensors
    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


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
    seqlen_q,
    seqlen_k,
    tile_m,
    tile_n,
):
    """Compare block sparsity against reference, handling boundary block semantics.

    PyTorch treats OOB regions as masked, so boundary blocks with all in-bounds
    elements unmasked appear as "partial" in PyTorch but "full" in CuTe.

    This applies to BOTH boundary m_blocks (OOB q_idx) and boundary n_blocks (OOB kv_idx).
    """
    if not isinstance(mask_block_cnt, torch.Tensor):
        return False, f"mask_block_cnt is not a tensor: {type(mask_block_cnt)}"

    n_blocks_q = mask_block_cnt.shape[2]

    # Identify boundary blocks
    last_m_block = (seqlen_q - 1) // tile_m
    last_n_block = (seqlen_k - 1) // tile_n
    m_is_boundary = seqlen_q % tile_m != 0
    n_is_boundary = seqlen_k % tile_n != 0

    def is_boundary_n_block(n_block):
        return n_is_boundary and n_block == last_n_block

    def is_boundary_m_block(m_block):
        return m_is_boundary and m_block == last_m_block

    for b in range(batch_size):
        for h in range(nheads):
            for m in range(n_blocks_q):
                cute_mask_cnt = mask_block_cnt[b, h, m].item()
                cute_full_cnt = full_block_cnt[b, h, m].item()
                ref_mask_cnt = mask_block_cnt_ref[b, h, m].item()
                ref_full_cnt = full_block_cnt_ref[b, h, m].item()

                cute_mask_set = set(mask_block_idx[b, h, m, :cute_mask_cnt].tolist())
                cute_full_set = set(full_block_idx[b, h, m, :cute_full_cnt].tolist())
                ref_mask_set = set(mask_block_idx_ref[b, h, m, :ref_mask_cnt].tolist())
                ref_full_set = set(full_block_idx_ref[b, h, m, :ref_full_cnt].tolist())

                # A block is "boundary-affected" if EITHER the m_block OR n_block is at boundary
                def is_boundary_affected(n_block):
                    return is_boundary_m_block(m) or is_boundary_n_block(n_block)

                # Blocks that are full in CuTe but not in ref
                full_in_cute_not_ref = cute_full_set - ref_full_set

                for n_block in full_in_cute_not_ref:
                    if not is_boundary_affected(n_block):
                        return False, (
                            f"Non-boundary block mismatch at [{b},{h},{m}]: "
                            f"n_block {n_block} is full in CuTe but not in ref"
                        )
                    # Boundary-affected: CuTe says full, ref should say partial
                    if n_block not in ref_mask_set:
                        # Check if ref skipped it entirely (all masked)
                        # This is valid for boundary blocks
                        pass

                # Blocks that are partial in CuTe but full in ref (would be a bug)
                partial_in_cute_full_in_ref = cute_mask_set & ref_full_set
                if partial_in_cute_full_in_ref:
                    return False, (
                        f"Block mismatch at [{b},{h},{m}]: "
                        f"n_blocks {sorted(partial_in_cute_full_in_ref)} are partial in CuTe but full in ref"
                    )

                # Check non-boundary blocks match exactly
                non_boundary_cute_full = {
                    n for n in cute_full_set if not is_boundary_affected(n)
                }
                non_boundary_ref_full = {
                    n for n in ref_full_set if not is_boundary_affected(n)
                }
                if non_boundary_cute_full != non_boundary_ref_full:
                    return False, (
                        f"Non-boundary full block mismatch at [{b},{h},{m}]: "
                        f"CuTe={sorted(non_boundary_cute_full)}, ref={sorted(non_boundary_ref_full)}"
                    )

                non_boundary_cute_mask = {
                    n for n in cute_mask_set if not is_boundary_affected(n)
                }
                non_boundary_ref_mask = {
                    n for n in ref_mask_set if not is_boundary_affected(n)
                }
                if non_boundary_cute_mask != non_boundary_ref_mask:
                    return False, (
                        f"Non-boundary partial block mismatch at [{b},{h},{m}]: "
                        f"CuTe={sorted(non_boundary_cute_mask)}, ref={sorted(non_boundary_ref_mask)}"
                    )

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
    (8192, 8192),
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

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = (
        _call_compute_block_sparsity(
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            tile_m,
            tile_n,
            mask_name,
            use_fast_sampling=False,
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

    print("CuTe results:")
    print(f"    mask_block_cnt: {mask_block_cnt}")
    print(f"    full_block_cnt: {full_block_cnt}")
    print(f"    mask_block_idx: {mask_block_idx}")
    print(f"    full_block_idx: {full_block_idx}")
    print("Torch results:")
    print(f"    mask_block_cnt: {mask_block_cnt_ref}")
    print(f"    full_block_cnt: {full_block_cnt_ref}")
    print(f"    mask_block_idx: {mask_block_idx_ref}")
    print(f"    full_block_idx: {full_block_idx_ref}")

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
        seqlen_q,
        seqlen_k,
        tile_m,
        tile_n,
    )
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

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = (
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
        seqlen_q,
        seqlen_k,
        tile_m,
        tile_n,
    )

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

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = (
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
        seqlen_q,
        seqlen_k,
        tile_m,
        tile_n,
    )
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

    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx = (
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
        seqlen_q,
        seqlen_k,
        tile_m,
        tile_n,
    )

    assert all_match, f"Mismatch: {error_msg}"


def _compare_block_sparsity_varlen(
    mask_block_cnt,
    mask_block_idx,
    full_block_cnt,
    full_block_idx,
    cu_total_m_blocks,
    cu_total_n_blocks,
    seqlens_q,
    seqlens_k,
    nheads,
    tile_m,
    tile_n,
    mask_name,
    window_size=None,
):
    """Compare varlen block sparsity against per-sequence fixed-length references."""
    batch_size = len(seqlens_q)
    cu_m = cu_total_m_blocks.cpu().tolist()
    cu_n = cu_total_n_blocks.cpu().tolist()

    for b in range(batch_size):
        sq, sk = seqlens_q[b], seqlens_k[b]
        num_m = (sq + tile_m - 1) // tile_m
        num_n = (sk + tile_n - 1) // tile_n
        m_off = cu_m[b]
        n_off = cu_n[b]

        _, mask_mod_flex = get_mask_pair(
            mask_name, seqlen_q=sq, seqlen_k=sk, window_size=window_size
        )
        block_mask = create_block_mask(
            mask_mod_flex,
            B=1,
            H=nheads,
            Q_LEN=sq,
            KV_LEN=sk,
            device="cuda",
            BLOCK_SIZE=(tile_m, tile_n),
        )
        _, _, ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx, *_ = (
            block_mask.as_tuple()
        )

        for h in range(nheads):
            for m in range(num_m):
                global_m = m_off + m
                n_base = n_off + m * num_n

                vl_mask_cnt = mask_block_cnt[h, global_m].item()
                vl_full_cnt = full_block_cnt[h, global_m].item()
                vl_mask_set = set(
                    mask_block_idx[h, n_base : n_base + vl_mask_cnt].tolist()
                )
                vl_full_set = set(
                    full_block_idx[h, n_base : n_base + vl_full_cnt].tolist()
                )

                r_mask_cnt = ref_mask_cnt[0, h, m].item()
                r_full_cnt = ref_full_cnt[0, h, m].item()
                r_mask_set = set(ref_mask_idx[0, h, m, :r_mask_cnt].tolist())
                r_full_set = set(ref_full_idx[0, h, m, :r_full_cnt].tolist())

                last_m_block = (sq - 1) // tile_m
                last_n_block = (sk - 1) // tile_n
                m_is_boundary = sq % tile_m != 0 and m == last_m_block
                n_is_boundary = sk % tile_n != 0

                def is_boundary_affected(
                    n_block,
                    _m_bnd=m_is_boundary,
                    _n_bnd=n_is_boundary,
                    _ln=last_n_block,
                ):
                    return _m_bnd or (_n_bnd and n_block == _ln)

                non_boundary_vl_full = {
                    n for n in vl_full_set if not is_boundary_affected(n)
                }
                non_boundary_ref_full = {
                    n for n in r_full_set if not is_boundary_affected(n)
                }
                if non_boundary_vl_full != non_boundary_ref_full:
                    return False, (
                        f"Varlen full block mismatch at batch={b}, head={h}, m_block={m} "
                        f"(sq={sq}, sk={sk}): "
                        f"varlen={sorted(non_boundary_vl_full)}, ref={sorted(non_boundary_ref_full)}"
                    )

                non_boundary_vl_mask = {
                    n for n in vl_mask_set if not is_boundary_affected(n)
                }
                non_boundary_ref_mask = {
                    n for n in r_mask_set if not is_boundary_affected(n)
                }
                if non_boundary_vl_mask != non_boundary_ref_mask:
                    return False, (
                        f"Varlen partial block mismatch at batch={b}, head={h}, m_block={m} "
                        f"(sq={sq}, sk={sk}): "
                        f"varlen={sorted(non_boundary_vl_mask)}, ref={sorted(non_boundary_ref_mask)}"
                    )

    return True, ""


# ---- Varlen test configurations ----

VARLEN_SEQLEN_CONFIGS = [
    # (seqlens_q, seqlens_k) - lists of per-batch lengths
    # Uniform lengths (should match fixed-length behavior)
    ([128, 128], [128, 128]),
    ([256, 256], [256, 256]),
    # Different lengths per batch
    ([64, 128], [64, 128]),
    ([128, 256], [128, 256]),
    ([256, 512], [256, 512]),
    ([64, 128, 256], [64, 128, 256]),
    # Unaligned
    ([113, 203], [113, 203]),
    ([127, 255], [127, 255]),
    ([100, 200, 300], [100, 200, 300]),
    # Asymmetric Q/K
    ([128, 256], [256, 128]),
    ([64, 128], [128, 256]),
    # Single element sequences
    ([1, 128], [1, 128]),
    ([64, 1], [64, 1]),
    # Large spread
    ([32, 512, 128], [32, 512, 128]),
    ([1024, 64], [1024, 64]),
]


def _generate_varlen_inputs(
    seqlens_q,
    seqlens_k,
    tile_m,
    tile_n,
    device="cuda",
):
    """Generate cu_seqlens and cu_total_*_blocks for a varlen batch.

    Args:
        seqlens_q: list of per-batch query sequence lengths
        seqlens_k: list of per-batch key sequence lengths
        tile_m, tile_n: tile sizes
    Returns:
        cu_seqlens_q, cu_seqlens_k, cu_total_m_blocks, cu_total_n_blocks
    """
    batch_size = len(seqlens_q)
    assert len(seqlens_k) == batch_size

    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    cu_total_m_blocks = [0]
    cu_total_n_blocks = [0]

    for b in range(batch_size):
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[b])
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[b])
        num_m = (seqlens_q[b] + tile_m - 1) // tile_m
        num_n = (seqlens_k[b] + tile_n - 1) // tile_n
        cu_total_m_blocks.append(cu_total_m_blocks[-1] + num_m)
        cu_total_n_blocks.append(cu_total_n_blocks[-1] + num_m * num_n)

    return (
        torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32),
        torch.tensor(cu_seqlens_k, device=device, dtype=torch.int32),
        torch.tensor(cu_total_m_blocks, device=device, dtype=torch.int32),
        torch.tensor(cu_total_n_blocks, device=device, dtype=torch.int32),
    )


def _call_compute_block_sparsity_varlen(
    seqlens_q,
    seqlens_k,
    nheads,
    tile_m,
    tile_n,
    mask_name,
    window_size=None,
    aux_tensors=None,
    use_fast_sampling=False,
):
    """Call compute_block_sparsity with varlen inputs."""
    batch_size = len(seqlens_q)
    # Use max seqlens for mask_mod compilation (the kernel uses per-batch seqlens at runtime)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    cute_mask, _ = get_mask_pair(
        mask_name, seqlen_q=max_seqlen_q, seqlen_k=max_seqlen_k, window_size=window_size
    )

    cu_seqlens_q, cu_seqlens_k, cu_total_m_blocks, cu_total_n_blocks = (
        _generate_varlen_inputs(seqlens_q, seqlens_k, tile_m, tile_n)
    )

    torch_tensors = compute_block_sparsity(
        tile_m=tile_m,
        tile_n=tile_n,
        batch_size=batch_size,
        num_heads=nheads,
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        mask_mod=cute_mask,
        aux_tensors=aux_tensors,
        device="cuda",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_total_m_blocks=cu_total_m_blocks,
        cu_total_n_blocks=cu_total_n_blocks,
        use_fast_sampling=use_fast_sampling,
    )
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, *_ = torch_tensors
    return (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        cu_total_m_blocks,
        cu_total_n_blocks,
    )


@pytest.mark.parametrize("seqlens_q,seqlens_k", VARLEN_SEQLEN_CONFIGS)
@pytest.mark.parametrize("tile_m,tile_n", [(64, 64), (128, 128), (64, 128), (128, 64)])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("mask_name", ["causal", "block_diagonal"])
def test_varlen(seqlens_q, seqlens_k, tile_m, tile_n, nheads, mask_name):
    """Test variable-length sequence support."""
    (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        cu_total_m_blocks,
        cu_total_n_blocks,
    ) = _call_compute_block_sparsity_varlen(
        seqlens_q, seqlens_k, nheads, tile_m, tile_n, mask_name
    )

    all_match, error_msg = _compare_block_sparsity_varlen(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        cu_total_m_blocks,
        cu_total_n_blocks,
        seqlens_q,
        seqlens_k,
        nheads,
        tile_m,
        tile_n,
        mask_name,
    )
    assert all_match, f"Mismatch: {error_msg}"


@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [
        ([128, 128], [128, 128]),
        ([64, 128, 256], [64, 128, 256]),
        ([100, 200], [100, 200]),
    ],
)
@pytest.mark.parametrize("tile_m,tile_n", [(64, 64), (128, 128)])
@pytest.mark.parametrize("nheads", [1])
@pytest.mark.parametrize(
    "mask_name,window_size",
    [("causal", None), ("sliding_window", 64), ("sliding_window", 256)],
)
def test_varlen_parameterized_masks(
    seqlens_q, seqlens_k, tile_m, tile_n, nheads, mask_name, window_size
):
    """Test varlen with parameterized masks."""
    # Skip sliding window when any seqlen_q > seqlen_k
    if mask_name == "sliding_window" and any(
        sq > sk for sq, sk in zip(seqlens_q, seqlens_k)
    ):
        pytest.skip("Sliding window not supported for seqlen_q > seqlen_k")

    (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        cu_total_m_blocks,
        cu_total_n_blocks,
    ) = _call_compute_block_sparsity_varlen(
        seqlens_q, seqlens_k, nheads, tile_m, tile_n, mask_name, window_size=window_size
    )

    all_match, error_msg = _compare_block_sparsity_varlen(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        cu_total_m_blocks,
        cu_total_n_blocks,
        seqlens_q,
        seqlens_k,
        nheads,
        tile_m,
        tile_n,
        mask_name,
        window_size=window_size,
    )
    assert all_match, f"Mismatch: {error_msg}"


@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("tile_m,tile_n", [(64, 64), (128, 128)])
def test_varlen_matches_fixed_length(nheads, tile_m, tile_n):
    """Verify that varlen with uniform sequence lengths produces identical
    results to the fixed-length path."""
    seqlen_q, seqlen_k = 256, 256
    batch_size = 3
    mask_name = "causal"

    # Fixed-length result
    fixed_mask_cnt, fixed_mask_idx, fixed_full_cnt, fixed_full_idx = (
        _call_compute_block_sparsity(
            batch_size, nheads, seqlen_q, seqlen_k, tile_m, tile_n, mask_name
        )
    )

    # Varlen with uniform lengths
    seqlens_q = [seqlen_q] * batch_size
    seqlens_k = [seqlen_k] * batch_size
    (
        vl_mask_cnt,
        vl_mask_idx,
        vl_full_cnt,
        vl_full_idx,
        cu_total_m_blocks,
        cu_total_n_blocks,
    ) = _call_compute_block_sparsity_varlen(
        seqlens_q, seqlens_k, nheads, tile_m, tile_n, mask_name
    )

    num_m = (seqlen_q + tile_m - 1) // tile_m
    num_n = (seqlen_k + tile_n - 1) // tile_n
    cu_m = cu_total_m_blocks.cpu().tolist()
    cu_n = cu_total_n_blocks.cpu().tolist()

    for b in range(batch_size):
        for h in range(nheads):
            for m in range(num_m):
                global_m = cu_m[b] + m
                n_base = cu_n[b] + m * num_n

                # Counts should match
                assert (
                    vl_mask_cnt[h, global_m].item() == fixed_mask_cnt[b, h, m].item()
                ), f"Mask count mismatch at b={b}, h={h}, m={m}"
                assert (
                    vl_full_cnt[h, global_m].item() == fixed_full_cnt[b, h, m].item()
                ), f"Full count mismatch at b={b}, h={h}, m={m}"

                mc = vl_mask_cnt[h, global_m].item()
                fc = vl_full_cnt[h, global_m].item()
                vl_mask_set = set(vl_mask_idx[h, n_base : n_base + mc].tolist())
                vl_full_set = set(vl_full_idx[h, n_base : n_base + fc].tolist())
                fixed_mask_set = set(fixed_mask_idx[b, h, m, :mc].tolist())
                fixed_full_set = set(fixed_full_idx[b, h, m, :fc].tolist())

                assert vl_mask_set == fixed_mask_set, (
                    f"Mask idx mismatch at b={b}, h={h}, m={m}: "
                    f"varlen={sorted(vl_mask_set)}, fixed={sorted(fixed_mask_set)}"
                )
                assert vl_full_set == fixed_full_set, (
                    f"Full idx mismatch at b={b}, h={h}, m={m}: "
                    f"varlen={sorted(vl_full_set)}, fixed={sorted(fixed_full_set)}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
