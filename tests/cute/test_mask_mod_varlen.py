# mask_mod varlen test script
# Forward-only, no block sparsity (block sparsity will be added later)
#
# Since flex_attention doesn't support varlen natively, we compare
# results sequence-by-sequence: run the kernel with cu_seqlens (packed),
# then run flex_attention per-sequence and compare.
#
# Usage:
#   pytest test_mask_mod_varlen.py -v -s

import math
import random

import pytest
import torch
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute import utils
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
from mask_mod_definitions import (
    get_mask_pair,
    random_doc_id_tensor,
    STATIC_MASKS,
    PARAMETERIZED_MASK_FACTORIES,
    cute_global_packed_doc_mask,
    cute_global_ima_mask,
    cute_global_causal_window_mask,
    global_packed_doc_flex_factory,
    global_ima_flex_factory,
    global_causal_window_flex_factory,
    make_packed_doc_ids,
    make_global_thresholds,
    make_global_windows,
)

COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]


@pytest.fixture(autouse=True)
def reset_torch_state():
    """Reset torch dynamo/compile state between tests to avoid state pollution."""
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    yield
    torch._dynamo.reset()
    torch.cuda.empty_cache()


# =============================================================================
# Seqlen configs for varlen (list of per-sequence lengths)
# =============================================================================

SEQLEN_CONFIGS = [
    # Simple cases
    ([1], [1]),
    ([64], [64]),
    ([128], [128]),
    # Multiple sequences, same length
    ([128, 128], [128, 128]),
    ([64, 64, 64], [64, 64, 64]),
    # Multiple sequences, varying lengths
    ([64, 128], [64, 128]),
    ([32, 64, 128], [32, 64, 128]),
    ([113, 203], [113, 203]),
    ([256, 512], [256, 512]),
    # Asymmetric Q/K lengths
    ([64, 128], [32, 64]),
    ([100, 100], [50, 50]),
    # Edge cases
    ([1, 1], [1, 1]),
    ([1, 256], [1, 256]),
    ([256, 1], [256, 1]),
    ([17, 33, 65], [17, 33, 65]),
    # Larger sequences
    ([1024, 1024], [1024, 1024]),
    ([256, 512, 256], [128, 256, 128]),
]

SEQLEN_CONFIGS_SMOKE = [
    ([128, 128], [128, 128]),
    ([64, 128], [64, 128]),
    ([113, 203], [113, 203]),
    ([256, 512], [256, 512]),
    ([64, 128], [32, 64]),
]


# =============================================================================
# Helper functions
# =============================================================================


def setup_varlen_tensors(
    seqlens_q, seqlens_k, num_heads, num_kv_heads, head_dim, dtype
):
    """Create packed Q, K, V tensors and cu_seqlens for varlen."""
    device = "cuda"
    batch_size = len(seqlens_q)
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, num_kv_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device=device,
        dtype=torch.int32,
    )

    return q, k, v, cu_seqlens_q, cu_seqlens_k


def run_flex_per_sequence(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    mask_mod_flex_factory,
    seqlens_q,
    seqlens_k,
    num_heads,
    num_kv_heads,
    head_dim,
    dtype=None,
):
    """Run flex_attention per-sequence as reference for varlen.

    mask_mod_flex_factory(seq_idx, seqlen_q_i, seqlen_k_i) -> mask_mod function
    that takes (b, h, q_idx, kv_idx) for that sequence.
    """
    batch_size = len(seqlens_q)
    results = []

    for i in range(batch_size):
        sq = seqlens_q[i]
        sk = seqlens_k[i]

        # Extract packed slices
        q_slice = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].unsqueeze(0)  # (1, sq, H, D)
        k_slice = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(
            0
        )  # (1, sk, Hkv, D)
        v_slice = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)

        if dtype is not None:
            q_slice = q_slice.to(dtype)
            k_slice = k_slice.to(dtype)
            v_slice = v_slice.to(dtype)

        # Transpose to (B, H, S, D) for flex_attention
        q_t = q_slice.transpose(1, 2)
        k_t = k_slice.transpose(1, 2)
        v_t = v_slice.transpose(1, 2)

        # Expand KV heads for GQA
        if num_heads != num_kv_heads:
            repeat_factor = num_heads // num_kv_heads
            k_t = k_t.repeat_interleave(repeat_factor, dim=1)
            v_t = v_t.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(head_dim)

        mask_mod = mask_mod_flex_factory(i, sq, sk)

        if mask_mod is None:
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        else:
            block_mask = create_block_mask(
                mask_mod,
                B=1,
                H=num_heads,
                Q_LEN=sq,
                KV_LEN=sk,
                device=q.device,
            )
            out = flex_attention(
                q_t, k_t, v_t, block_mask=block_mask, scale=scale, enable_gqa=True
            )

        results.append(out.transpose(1, 2).squeeze(0))  # back to (sq, H, D)

    return torch.cat(results, dim=0)


def check_varlen_results(
    out_cute,
    out_ref_fp32,
    out_pt,
    seqlens_q,
    cu_seqlens_q,
    test_name,
    rtol=2,
    extra_atol=2e-3,
):
    """Compare CuTE output against per-sequence flex references."""
    assert not torch.isnan(out_cute).any(), f"{test_name}: NaN in output"
    assert torch.isfinite(out_cute).all(), f"{test_name}: Inf in output"
    assert out_cute.shape == out_ref_fp32.shape, (
        f"{test_name}: Shape mismatch: {out_cute.shape} vs {out_ref_fp32.shape}"
    )

    num_seqs = len(seqlens_q)
    max_cute_error = 0.0
    max_pt_error = 0.0

    for i in range(num_seqs):
        start = cu_seqlens_q[i]
        end = cu_seqlens_q[i + 1]
        cute_seq = out_cute[start:end]
        ref_seq = out_ref_fp32[start:end]
        pt_seq = out_pt[start:end]

        max_cute_error = max(max_cute_error, (cute_seq - ref_seq).abs().max().item())
        max_pt_error = max(max_pt_error, (pt_seq - ref_seq).abs().max().item())

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()

    print(f"\n{test_name}:")
    print(f"  PyTorch vs FP32 ref: {max_pt_error:.2e}")
    print(f"  CuTE vs FP32 ref: {max_cute_error:.2e}")

    tol = rtol * max_pt_error + fwd_atol + extra_atol
    assert max_cute_error <= tol, (
        f"{test_name}: CuTE error {max_cute_error:.2e} exceeds tolerance {tol:.2e} "
        f"(rtol={rtol} * pt_err={max_pt_error:.2e} + fwd_atol={fwd_atol:.2e} + extra={extra_atol:.2e})"
    )


# =============================================================================
# Core test runner
# =============================================================================


def _run_varlen_mask_test(
    seqlens_q,
    seqlens_k,
    num_heads,
    num_kv_heads,
    head_dim,
    dtype,
    mask_name,
    window_size=None,
):
    """Run a varlen mask_mod test: kernel with cu_seqlens vs per-sequence flex_attention."""
    torch.manual_seed(42)
    random.seed(42)

    batch_size = len(seqlens_q)
    pack_gqa = num_heads != num_kv_heads

    if mask_name == "sliding_window":
        # Skip configs where any seqlen_q > seqlen_k
        for sq, sk in zip(seqlens_q, seqlens_k):
            if sq > sk:
                pytest.skip(
                    "sliding_window requires seqlen_q <= seqlen_k for each sequence"
                )

    q, k, v, cu_seqlens_q, cu_seqlens_k = setup_varlen_tensors(
        seqlens_q, seqlens_k, num_heads, num_kv_heads, head_dim, dtype
    )

    if mask_name == "block_causal":
        offsets = [sk - sq for sq, sk in zip(seqlens_q, seqlens_k)]
        if len(set(offsets)) > 1:
            pytest.skip(
                "block_causal captures offset as compile-time constant; "
                "varlen with different per-sequence offsets not supported"
            )

    aux_tensors_arg = None

    if mask_name == "document":
        max_seqlen = max(max(seqlens_q), max(seqlens_k))
        max_doc_len = max(max(seqlens_q), max(seqlens_k))
        doc_ids = random_doc_id_tensor(
            num_heads, batch_size, max_doc_len, device="cuda"
        ).to(dtype=torch.int32, device="cuda")
        doc_ids.__leading_dim__ = 2
        aux_tensors_arg = [doc_ids]

        from mask_mod_definitions import flex_document_mask

        cute_mask_mod = get_mask_pair("document")[0]

        def flex_factory(seq_idx, sq, sk, doc_ids=doc_ids):
            # Pre-slice to 1D using Python ints *outside* the vmapped closure.
            # create_block_mask vmaps over all four args (b, h, q_idx, kv_idx);
            # multi-dim indexing like doc_id[b, h, q_idx] with 0-dim vmap tensors
            # triggers .item() internally. 1D tensor[0d_tensor] is a safe gather.
            doc_row = doc_ids[seq_idx, 0]  # (max_doc_len,)

            def _mask(b, h, q_idx, kv_idx):
                return doc_row[q_idx] == doc_row[kv_idx]

            return _mask

    elif mask_name == "ima":
        total_k = sum(seqlens_k)
        pytest.skip(
            "IMA mask requires global index handling for varlen - not yet implemented"
        )

    else:
        if mask_name in STATIC_MASKS:
            cute_mask_mod = get_mask_pair(mask_name)[0]

            def flex_factory(seq_idx, sq, sk):
                return get_mask_pair(mask_name)[1]

        elif mask_name in PARAMETERIZED_MASK_FACTORIES:
            cute_mask_mod = get_mask_pair(
                mask_name,
                seqlen_q=seqlens_q[0],
                seqlen_k=seqlens_k[0],
                window_size=window_size,
            )[0]

            def flex_factory(seq_idx, sq, sk):
                _, flex_mask = get_mask_pair(
                    mask_name,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    window_size=window_size,
                )
                return flex_mask

        else:
            raise ValueError(f"Unknown mask: {mask_name}")

    # Run the kernel with varlen (packed format)
    out = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_tuple = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=None,
        seqused_k=None,
        page_table=None,
        softmax_scale=softmax_scale,
        causal=False,
        softcap=None,
        window_size_left=-1,
        window_size_right=-1,
        learnable_sink=None,
        tile_mn=(128, 128),
        pack_gqa=pack_gqa,
        _arch=None,
        score_mod=None,
        mask_mod=cute_mask_mod,
        block_sparse_tensors=None,
        return_lse=True,
        aux_tensors=aux_tensors_arg,
    )
    out_cute = out_tuple[0]

    # Run per-sequence flex_attention references
    out_ref_fp32 = run_flex_per_sequence(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        flex_factory,
        seqlens_q,
        seqlens_k,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype=torch.float32,
    )
    out_pt = run_flex_per_sequence(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        flex_factory,
        seqlens_q,
        seqlens_k,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype=dtype,
    )

    # Check results
    mask_desc = f"mask_mod={mask_name}"
    if window_size is not None:
        mask_desc += f"(w={window_size})"
    test_name = (
        f"{mask_desc} varlen seqs_q={seqlens_q}, seqs_k={seqlens_k}, "
        f"H={num_heads}/{num_kv_heads}, D={head_dim}"
    )
    check_varlen_results(
        out_cute, out_ref_fp32, out_pt, seqlens_q, cu_seqlens_q, test_name
    )


# =============================================================================
# Test cases
# =============================================================================

# Masks that don't need recompilation per seqlen (fast)
STATIC_MASK_NAMES = ["block_diagonal", "mini_causal"]

# Masks that need per-seqlen compilation (slower)
PARAMETERIZED_MASK_CONFIGS = [
    ("causal", None),
    ("block_causal", None),
    ("sliding_window", 128),
    ("sliding_window", 256),
    ("document", None),
]


@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
@pytest.mark.parametrize("mask_name", STATIC_MASK_NAMES)
def test_varlen_static_masks(seqlens_q, seqlens_k, dtype, kv_mode, mask_name):
    """Test static mask_mods with varlen (packed) attention."""
    num_heads = 8
    if kv_mode == "gqa":
        if COMPUTE_CAPABILITY < 9:
            pytest.xfail("pack_gqa requires SM90+")
        num_kv_heads = 2
    else:
        num_kv_heads = num_heads

    _run_varlen_mask_test(
        seqlens_q=seqlens_q,
        seqlens_k=seqlens_k,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        dtype=dtype,
        mask_name=mask_name,
    )


@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
@pytest.mark.parametrize("mask_name,window_size", PARAMETERIZED_MASK_CONFIGS)
def test_varlen_parameterized_masks(
    seqlens_q, seqlens_k, dtype, kv_mode, mask_name, window_size
):
    """Test parameterized mask_mods with varlen (packed) attention.

    Uses fewer seqlen configs since these require recompilation per seqlen.
    """
    num_heads = 8
    if kv_mode == "gqa":
        if COMPUTE_CAPABILITY < 9:
            pytest.xfail("pack_gqa requires SM90+")
        num_kv_heads = 2
    else:
        num_kv_heads = num_heads

    _run_varlen_mask_test(
        seqlens_q=seqlens_q,
        seqlens_k=seqlens_k,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        dtype=dtype,
        mask_name=mask_name,
        window_size=window_size,
    )


# =============================================================================
# Global-index mask test runner
# =============================================================================


def _run_varlen_global_mask_test(
    seqlens_q,
    seqlens_k,
    num_heads,
    num_kv_heads,
    head_dim,
    dtype,
    mask_name,
):
    """Run a varlen global-index mask_mod test: kernel vs per-sequence flex_attention."""
    torch.manual_seed(42)
    random.seed(42)

    pack_gqa = num_heads != num_kv_heads

    q, k, v, cu_seqlens_q, cu_seqlens_k = setup_varlen_tensors(
        seqlens_q, seqlens_k, num_heads, num_kv_heads, head_dim, dtype
    )

    if mask_name == "global_packed_doc":
        doc_ids_q, doc_ids_k = make_packed_doc_ids(seqlens_q, seqlens_k, device="cuda")
        cute_mask = cute_global_packed_doc_mask
        flex_fac = global_packed_doc_flex_factory(
            doc_ids_q, doc_ids_k, cu_seqlens_q, cu_seqlens_k
        )
        aux_tensors = [doc_ids_q, doc_ids_k]
    elif mask_name == "global_ima":
        thresholds = make_global_thresholds(seqlens_k, device="cuda")
        cute_mask = cute_global_ima_mask
        flex_fac = global_ima_flex_factory(thresholds, cu_seqlens_k)
        aux_tensors = [thresholds]
    elif mask_name == "global_causal_window":
        windows = make_global_windows(seqlens_q, device="cuda")
        cute_mask = cute_global_causal_window_mask
        flex_fac = global_causal_window_flex_factory(windows, cu_seqlens_q)
        aux_tensors = [windows]
    else:
        raise ValueError(f"Unknown global mask: {mask_name}")

    out = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_tuple = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=None,
        seqused_k=None,
        page_table=None,
        softmax_scale=softmax_scale,
        causal=False,
        softcap=None,
        window_size_left=-1,
        window_size_right=-1,
        learnable_sink=None,
        tile_mn=(128, 128),
        pack_gqa=pack_gqa,
        _arch=None,
        score_mod=None,
        mask_mod=cute_mask,
        block_sparse_tensors=None,
        return_lse=True,
        aux_tensors=aux_tensors,
    )
    out_cute = out_tuple[0]

    out_ref_fp32 = run_flex_per_sequence(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        flex_fac,
        seqlens_q,
        seqlens_k,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype=torch.float32,
    )
    out_pt = run_flex_per_sequence(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        flex_fac,
        seqlens_q,
        seqlens_k,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype=dtype,
    )

    test_name = (
        f"global_mask={mask_name} varlen seqs_q={seqlens_q}, seqs_k={seqlens_k}, "
        f"H={num_heads}/{num_kv_heads}, D={head_dim}"
    )
    check_varlen_results(
        out_cute, out_ref_fp32, out_pt, seqlens_q, cu_seqlens_q, test_name
    )


GLOBAL_MASK_NAMES = ["global_packed_doc", "global_ima", "global_causal_window"]


@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("mask_name", GLOBAL_MASK_NAMES)
def test_varlen_global_masks(seqlens_q, seqlens_k, mask_name):
    """Test global-index mask_mods (aux-tensor-driven) with varlen packed attention."""
    _run_varlen_global_mask_test(
        seqlens_q, seqlens_k, 8, 8, 128, torch.bfloat16, mask_name
    )


# =============================================================================
# Block sparsity end-to-end tests
# =============================================================================


def _make_block_sparse_tensors(
    mask_mod,
    seqlens_q,
    seqlens_k,
    num_heads,
    tile_m,
    tile_n,
    device,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_k=None,
    aux_tensors=None,
):
    """Compute block sparse tensors, returning (tensors, cu_total_m_blocks, cu_total_n_blocks)."""
    batch_size = len(seqlens_q)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    if cu_seqlens_q is not None:
        cu_total_m_blocks_list = [0]
        for batch_idx in range(batch_size):
            num_m_blocks = (seqlens_q[batch_idx] + tile_m - 1) // tile_m
            cu_total_m_blocks_list.append(cu_total_m_blocks_list[-1] + num_m_blocks)
        cu_total_m_blocks = torch.tensor(
            cu_total_m_blocks_list, dtype=torch.int32, device=device
        )

        cu_total_n_blocks = None
        if cu_seqlens_k is not None or seqused_k is not None:
            cu_total_n_blocks_list = [0]
            for batch_idx in range(batch_size):
                num_m_blocks = (seqlens_q[batch_idx] + tile_m - 1) // tile_m
                num_n_blocks = (seqlens_k[batch_idx] + tile_n - 1) // tile_n
                cu_total_n_blocks_list.append(
                    cu_total_n_blocks_list[-1] + num_m_blocks * num_n_blocks
                )
            cu_total_n_blocks = torch.tensor(
                cu_total_n_blocks_list, dtype=torch.int32, device=device
            )
    else:
        cu_total_m_blocks = None
        cu_total_n_blocks = None

    block_sparse_tensors = compute_block_sparsity(
        tile_m=tile_m,
        tile_n=tile_n,
        batch_size=batch_size,
        num_heads=num_heads,
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        device=device,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_total_m_blocks=cu_total_m_blocks,
        cu_total_n_blocks=cu_total_n_blocks,
        seqused_k=seqused_k,
    )
    return block_sparse_tensors, cu_total_m_blocks, cu_total_n_blocks


def _run_fwd(
    q,
    k,
    v,
    mask_mod,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_k=None,
    block_sparse_tensors=None,
    cu_total_m_blocks=None,
    cu_total_n_blocks=None,
    aux_tensors=None,
):
    out = torch.empty_like(q)
    return _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=None,
        seqused_k=seqused_k,
        page_table=None,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
        softcap=None,
        window_size_left=-1,
        window_size_right=-1,
        learnable_sink=None,
        tile_mn=(128, 128),
        pack_gqa=False,
        _arch=None,
        score_mod=None,
        mask_mod=mask_mod,
        block_sparse_tensors=block_sparse_tensors,
        cu_total_m_blocks=cu_total_m_blocks,
        cu_total_n_blocks=cu_total_n_blocks,
        return_lse=False,
        aux_tensors=aux_tensors,
    )[0]


BLOCK_SPARSE_MASK_NAMES = [
    "causal",
    "block_diagonal",
    "mini_causal",
    "prefix_lm",
    "sliding_window",
    "dilated_sliding_window",
    "document",
    "ima",
]

BLOCK_SPARSE_SEQLEN_CONFIGS = [
    ([128, 192, 256], [128, 192, 256]),
    ([64, 128], [64, 128]),
    ([256, 512, 256], [256, 512, 256]),
    ([128, 192, 256], [64, 128, 192]),
]


# @pytest.mark.parametrize("varlen_q", [False, True])
@pytest.mark.parametrize("seqlens", BLOCK_SPARSE_SEQLEN_CONFIGS)
@pytest.mark.parametrize("mask_name", BLOCK_SPARSE_MASK_NAMES)
@pytest.mark.parametrize("varlen_q", [False, True])
@pytest.mark.parametrize("varlen_k", [False, True])
@pytest.mark.parametrize("use_seqused_k", [False, True])
@pytest.mark.parametrize("head_broadcast", [False, True])
def test_varlen_block_sparse(
    varlen_q, varlen_k, use_seqused_k, head_broadcast, mask_name, seqlens
):
    """Block sparsity + mask_mod should produce identical output to mask_mod alone."""
    if varlen_k and use_seqused_k:
        pytest.skip("packed K (cu_seqlens_k) and seqused_k are mutually exclusive")
    if not varlen_q and varlen_k:
        pytest.skip(
            "block sparsity with padded Q + packed K requires per-batch n-block offsets; not yet supported"
        )

    torch.manual_seed(42)
    random.seed(42)
    device = "cuda"
    num_heads = 4
    head_dim = 128
    dtype = torch.bfloat16
    # On Blackwell (SM100) q_stage=2 → effective tile is 256; elsewhere 128.
    tile_m = 256 if COMPUTE_CAPABILITY >= 10 else 128
    tile_n = 128

    base_seqlens_q, base_seqlens_k = seqlens
    batch_size = len(base_seqlens_q)
    max_seqlen_q = max(base_seqlens_q)
    max_seqlen_k = max(base_seqlens_k)

    seqlens_q = base_seqlens_q if varlen_q else [max_seqlen_q] * batch_size
    seqlens_k = (
        base_seqlens_k if (varlen_k or use_seqused_k) else [max_seqlen_k] * batch_size
    )

    def make_cu_seqlens(seqlens):
        return torch.tensor(
            [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
            device=device,
            dtype=torch.int32,
        )

    if varlen_q:
        q = torch.randn(sum(seqlens_q), num_heads, head_dim, device=device, dtype=dtype)
        cu_seqlens_q = make_cu_seqlens(seqlens_q)
    else:
        q = torch.randn(
            batch_size, max_seqlen_q, num_heads, head_dim, device=device, dtype=dtype
        )
        cu_seqlens_q = None

    if varlen_k:
        k = torch.randn(sum(seqlens_k), num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(sum(seqlens_k), num_heads, head_dim, device=device, dtype=dtype)
        cu_seqlens_k = make_cu_seqlens(seqlens_k)
        seqused_k = None
    else:
        k = torch.randn(
            batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, max_seqlen_k, num_heads, head_dim, device=device, dtype=dtype
        )
        cu_seqlens_k = None
        seqused_k = (
            torch.tensor(seqlens_k, dtype=torch.int32, device=device)
            if use_seqused_k
            else None
        )

    # Build mask_mod and aux_tensors for the requested mask
    aux_tensors = None
    if mask_name == "causal":
        mask_mod, _ = get_mask_pair(
            "causal", seqlen_q=max_seqlen_q, seqlen_k=max_seqlen_k
        )
    elif mask_name == "sliding_window":
        mask_mod, _ = get_mask_pair(
            "sliding_window",
            seqlen_q=max_seqlen_q,
            seqlen_k=max_seqlen_k,
            window_size=128,
        )
    elif mask_name == "document":
        max_doc_len = max(max_seqlen_q, max_seqlen_k)
        doc_ids = random_doc_id_tensor(
            num_heads, batch_size, max_doc_len, device=device
        )
        doc_ids.__leading_dim__ = 2
        aux_tensors = [doc_ids]
        mask_mod = get_mask_pair("document")[0]
    elif mask_name == "ima":
        bias = torch.randint(
            0,
            max(1, max_seqlen_k // 2),
            (max_seqlen_k,),
            dtype=torch.int32,
            device=device,
        )
        aux_tensors = [bias]
        mask_mod = get_mask_pair("ima")[0]
    else:
        mask_mod = get_mask_pair(mask_name)[0]

    num_heads_sparse = 1 if head_broadcast else num_heads
    block_sparse_tensors, cu_total_m_blocks, cu_total_n_blocks = (
        _make_block_sparse_tensors(
            mask_mod=mask_mod,
            seqlens_q=seqlens_q,
            seqlens_k=seqlens_k,
            num_heads=num_heads_sparse,
            tile_m=tile_m,
            tile_n=tile_n,
            device=device,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            aux_tensors=aux_tensors,
        )
    )

    out_with_block_sparsity = _run_fwd(
        q,
        k,
        v,
        mask_mod,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        block_sparse_tensors=block_sparse_tensors,
        cu_total_m_blocks=cu_total_m_blocks,
        cu_total_n_blocks=cu_total_n_blocks,
        aux_tensors=aux_tensors,
    )
    out_no_block_sparsity = _run_fwd(
        q,
        k,
        v,
        mask_mod,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        aux_tensors=aux_tensors,
    )

    assert not torch.isnan(out_with_block_sparsity).any(), "NaN in block-sparse output"
    max_err = (out_with_block_sparsity - out_no_block_sparsity).abs().max().item()
    assert max_err <= 0.01, (
        f"block-sparse output differs from mask-mod-only by {max_err}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
