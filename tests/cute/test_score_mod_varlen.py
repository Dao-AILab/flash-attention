import operator

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass._mlir.dialects import math as mlir_math
from flash_attn.cute.interface import _flash_attn_fwd
from score_mod_definitions import (
    # TensorSSA-based score mods
    score_mod_alibi,
    score_mod_batch_bias,
    score_mod_block_diagonal,
    score_mod_causal,
    score_mod_causal_v2,
    score_mod_debug_global_idx,
    score_mod_dual_buffer,
    score_mod_global_kv_bias,
    score_mod_global_logical_rel_plus_kv_bias,
    score_mod_global_q_and_kv_bias,
    score_mod_global_q_bias,
    score_mod_global_rel_plus_kv_bias,
    score_mod_identity,
    score_mod_rel_bias,
    score_mod_rel_bias_x2,
    score_mod_sliding_window,
    score_mod_stress_complex_arithmetic,
    score_mod_stress_conditional_mask,
    score_mod_stress_global_offset,
    score_mod_stress_multi_buffer,
    score_mod_stress_xor_pattern,
    score_mod_times_two,
)

# isort: split
from score_mod_definitions import (
    # Eager (torch) reference score mods
    identity_eager,
    causal_eager,
    rel_bias_eager,
    rel_bias_x2_eager,
    times_two_eager,
    alibi_eager,
    sliding_window_eager,
    block_diagonal_eager,
    causal_v2_eager,
    batch_bias_factory,
    dual_buffer_factory,
    packed_kv_bias_factory,
    packed_q_bias_factory,
    packed_rel_plus_kv_bias_factory,
    packed_q_and_kv_bias_factory,
    packed_logical_rel_plus_kv_bias_factory,
    stress_complex_arithmetic_factory,
    stress_conditional_mask_factory,
    stress_multi_buffer_factory,
    stress_global_offset_factory,
    stress_xor_pattern_factory,
    debug_global_idx_factory,
)

# =============================================================================
# Test pairs
# =============================================================================

# (cute_score_mod, eager_factory_or_fn, aux_type)
# aux_type: None, "batch", "dual_buffer"
TEST_PAIRS_6ARG = [
    (score_mod_identity, identity_eager, None),
    (score_mod_causal, causal_eager, None),
    (score_mod_rel_bias, rel_bias_eager, None),
    (score_mod_rel_bias_x2, rel_bias_x2_eager, None),
    (score_mod_times_two, times_two_eager, None),
    (score_mod_alibi, alibi_eager, None),
    (score_mod_sliding_window, sliding_window_eager, None),
    (score_mod_block_diagonal, block_diagonal_eager, None),
    (score_mod_causal_v2, causal_v2_eager, None),
    (score_mod_batch_bias, batch_bias_factory, "batch"),
    (score_mod_dual_buffer, dual_buffer_factory, "dual_buffer"),
]

# (cute_score_mod, eager_factory, aux_type, requires_global)
# aux_type: "kv", "q", "q_and_kv", "q_concat", "kv_with_cu", "multi_buffer"
# requires_global: "q" (needs varlen_q), "kv" (needs varlen_k), "both" (needs both)
TEST_PAIRS_8ARG = [
    (score_mod_global_kv_bias, packed_kv_bias_factory, "kv", "kv"),
    (score_mod_global_q_bias, packed_q_bias_factory, "q", "q"),
    (score_mod_global_rel_plus_kv_bias, packed_rel_plus_kv_bias_factory, "kv", "kv"),
    (score_mod_global_q_and_kv_bias, packed_q_and_kv_bias_factory, "q_and_kv", "both"),
    (
        score_mod_global_logical_rel_plus_kv_bias,
        packed_logical_rel_plus_kv_bias_factory,
        "kv",
        "kv",
    ),
    (
        score_mod_stress_complex_arithmetic,
        stress_complex_arithmetic_factory,
        "q_concat",
        "q",
    ),
    (
        score_mod_stress_conditional_mask,
        stress_conditional_mask_factory,
        "kv_with_cu",
        "both",
    ),
    (
        score_mod_stress_multi_buffer,
        stress_multi_buffer_factory,
        "multi_buffer",
        "both",
    ),
    (score_mod_stress_global_offset, stress_global_offset_factory, "kv", "kv"),
    (score_mod_stress_xor_pattern, stress_xor_pattern_factory, "kv_with_cu", "kv"),
    (score_mod_debug_global_idx, debug_global_idx_factory, "kv", "kv"),
]

SEQLEN_CONFIGS = [
    ([1], [1]),
    ([1, 1], [1, 1]),
    ([2, 3], [2, 3]),
    ([8, 16], [8, 16]),
    ([32, 32], [32, 32]),
    ([64, 128], [64, 128]),
    ([64, 56, 128], [64, 56, 128]),
    ([256, 512], [256, 512]),
    ([113, 203], [113, 203]),
    ([239, 1], [239, 1]),
    ([64], [64]),
    ([128, 128], [128, 128]),
    ([32, 32, 32, 32], [32, 32, 32, 32]),
    ([16, 32, 64, 128, 256], [16, 32, 64, 128, 256]),
    ([1, 1024], [1, 1024]),
    ([1024, 1], [1024, 1]),
    ([1, 256, 1], [1, 256, 1]),
    ([256, 1, 256], [256, 1, 256]),
    ([17, 33, 65], [17, 33, 65]),
    ([64, 128], [32, 64]),
    ([100, 100], [50, 50]),
    ([256, 512, 256], [128, 256, 128]),
    ([2, 1], [16384, 32 * 1024]),
    ([1, 1], [128 * 1024] * 2),
    ([2, 1], [8192, 8192]),
    ([1, 3], [8192, 8192]),
    ([3, 3], [8192, 8192]),
    ([128, 128], [8192, 8192]),
    ([2, 2, 2], [8 * 1024] * 3),
    ([2, 1], [1024 * 32, 16384]),
    ([1, 2], [1024 * 32, 16384]),
    ([1, 1, 1], [128 * 1024] * 3),
    ([1, 1, 1], [256 * 1024] * 3),
]

# =============================================================================
# Helper functions
# =============================================================================


def run_cute_flash(
    q,
    k,
    v,
    score_mod,
    aux_tensors=None,
    pack_gqa=False,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    """Run CuTE flash attention."""
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        out = torch.empty_like(q)
        _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            return_lse=True,
            score_mod=score_mod,
            out=out,
            lse=None,
            aux_tensors=aux_tensors,
            pack_gqa=pack_gqa,
        )
        return out

    out = torch.empty_like(q)
    _flash_attn_fwd(
        q,
        k,
        v,
        return_lse=True,
        score_mod=score_mod,
        out=out,
        lse=None,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
    )
    return out


def run_flex_varlen_ref(q, k, v, cu_seqlens_q, cu_seqlens_k, score_mod, dtype=None):
    """Run flex_attention per-sequence for varlen reference."""
    if cu_seqlens_q is not None:
        num_batches = len(cu_seqlens_q) - 1
    else:
        num_batches = len(cu_seqlens_k) - 1

    results = []
    for i in range(num_batches):
        # Get Q slice
        if cu_seqlens_q is not None:
            q_slice = (
                q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].unsqueeze(0).transpose(1, 2)
            )
        else:
            q_slice = q[i : i + 1].transpose(1, 2)

        # Get K/V slices
        if cu_seqlens_k is not None:
            k_slice = (
                k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0).transpose(1, 2)
            )
            v_slice = (
                v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0).transpose(1, 2)
            )
        else:
            k_slice = k[i : i + 1].transpose(1, 2)
            v_slice = v[i : i + 1].transpose(1, 2)

        if dtype is not None:
            q_slice, k_slice, v_slice = (
                q_slice.to(dtype),
                k_slice.to(dtype),
                v_slice.to(dtype),
            )

        def wrapped_mod(score, b, h, q_idx, kv_idx):
            return score_mod(score, i, h, q_idx, kv_idx)

        out = flex_attention(
            q_slice,
            k_slice,
            v_slice,
            score_mod=wrapped_mod,
            enable_gqa=q_slice.shape[1] != k_slice.shape[1],
        )
        results.append(out.transpose(1, 2).squeeze(0))

    return torch.cat(results, dim=0)


def setup_tensors(seqlens_q, seqlens_k, varlen_q, varlen_k, num_heads, head_dim, dtype):
    """Create Q, K, V tensors and cu_seqlens based on varlen flags."""
    batch_size = len(seqlens_q)

    if varlen_q:
        total_q = sum(seqlens_q)
        q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
        cu_seqlens_q = torch.tensor(
            [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
    else:
        seqlen_q = seqlens_q[0]  # All sequences have the same length for non-varlen
        q = torch.randn(
            batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=dtype
        )
        cu_seqlens_q = None

    if varlen_k:
        total_k = sum(seqlens_k)
        k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
        cu_seqlens_k = torch.tensor(
            [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
    else:
        seqlen_k = seqlens_k[0]  # All sequences have the same length for non-varlen
        k = torch.randn(
            batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype
        )
        cu_seqlens_k = None

    return q, k, v, cu_seqlens_q, cu_seqlens_k


def prepare_ref_tensors(
    q, k, v, cu_seqlens_q, cu_seqlens_k, varlen_q, varlen_k, batch_size, seqlens_q
):
    """Prepare tensors for flex_attention reference (handle mixed varlen formats)."""
    num_heads = q.shape[1] if varlen_q else q.shape[2]

    if not varlen_q and varlen_k:
        # Q is batched (batch_size, seqlen_q, num_heads, head_dim)
        # Need to convert to packed format (total_q, num_heads, head_dim) for reference
        seqlen_q = q.shape[1]
        q_packed = q.reshape(-1, num_heads, q.shape[-1])
        ref_cu_seqlens_q = torch.tensor(
            [seqlen_q * i for i in range(batch_size + 1)],
            device="cuda",
            dtype=torch.int32,
        )
        return q_packed, k, v, ref_cu_seqlens_q, cu_seqlens_k

    if varlen_q and not varlen_k:
        # K is batched (batch_size, seqlen_k, num_heads, head_dim)
        # Need to transpose to (batch_size, num_heads, seqlen_k, head_dim) for flex_attention
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        return q, k, v, cu_seqlens_q, None

    return q, k, v, cu_seqlens_q, cu_seqlens_k


def check_results(
    out_cute,
    out_ref_fp32,
    out_pt,
    test_name,
    rtol=2,
    extra_atol=1e-4,
    seqlens_q=None,
    cu_seqlens_q=None,
):
    """Compare CuTE output against references."""
    assert not torch.isnan(out_cute).any(), f"{test_name}: NaN in output"
    assert torch.isfinite(out_cute).all(), f"{test_name}: Inf in output"

    varlen_q = cu_seqlens_q is not None

    if varlen_q:
        # Unpack and compare per-sequence
        assert seqlens_q is not None, "varlen_q requires use of seqlens_q"
        num_seqs = len(seqlens_q)
        max_cute_error = 0.0
        max_pt_error = 0.0

        for i in range(num_seqs):
            # Extract sequences using cu_seqlens (all outputs are in packed format)
            start_q = cu_seqlens_q[i]
            end_q = cu_seqlens_q[i + 1]
            cute_seq = out_cute[start_q:end_q]
            ref_seq = out_ref_fp32[start_q:end_q]
            pt_seq = out_pt[start_q:end_q]

            max_cute_error = max(
                max_cute_error, (cute_seq - ref_seq).abs().max().item()
            )
            max_pt_error = max(max_pt_error, (pt_seq - ref_seq).abs().max().item())

        cute_error = max_cute_error
        pt_error = max_pt_error
    else:
        # Direct comparison
        pt_error = (out_pt - out_ref_fp32).abs().max().item()
        cute_error = (out_cute - out_ref_fp32).abs().max().item()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()

    print(f"\n{test_name}:")
    print(f"  PyTorch vs FP32 ref: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref: {cute_error:.2e}")

    tol = rtol * pt_error + fwd_atol + extra_atol
    assert cute_error <= tol, (
        f"{test_name}: CuTE error {cute_error:.2e} exceeds tolerance {tol:.2e}"
    )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("varlen_q", [True, False])
@pytest.mark.parametrize("varlen_k", [True, False])
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 2), (4, 2)])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_6ARG)
def test_varlen_with_score_mod(
    seqlens_q,
    seqlens_k,
    varlen_q,
    varlen_k,
    qhead_per_kvhead,
    num_kv_heads,
    dtype,
    score_mod_tuple,
):
    """Test varlen attention with 6-arg score_mod functions.

    Covers: both varlen, varlen Q only, varlen K only.
    Skips: neither varlen
    """
    if not varlen_q and not varlen_k:
        pytest.skip(
            "At least one of varlen_q or varlen_k must be True for varlen tests"
        )

    # For non-varlen dimension, all sequences must have same length
    if not varlen_q:
        seqlens_q = [seqlens_q[0]] * len(seqlens_q)
    if not varlen_k:
        seqlens_k = [seqlens_k[0]] * len(seqlens_k)

    torch.random.manual_seed(42)
    cute_score_mod, eager_factory, aux_type = score_mod_tuple

    num_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    head_dim = 128
    batch_size = len(seqlens_q)

    q, k, v, cu_seqlens_q, cu_seqlens_k = setup_tensors(
        seqlens_q, seqlens_k, varlen_q, varlen_k, num_heads, head_dim, dtype
    )

    # For pack_gqa, reduce K and V to num_kv_heads
    if pack_gqa:
        if varlen_k:
            # K and V are (total_k, num_heads, head_dim) - slice head dimension
            k = k[:, :num_kv_heads, :].clone()
            v = v[:, :num_kv_heads, :].clone()
        else:
            # K and V are (batch, seqlen_k, num_heads, head_dim) - slice head dimension
            k = k[:, :, :num_kv_heads, :].clone()
            v = v[:, :, :num_kv_heads, :].clone()

    # Setup aux tensors and eager score_mod
    aux_tensors = None
    if aux_type == "batch":
        bias = torch.zeros(batch_size, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias)
    elif aux_type == "dual_buffer":
        seqlen_q = seqlens_q[0] if not varlen_q else max(seqlens_q)
        head_bias = torch.randn(num_heads, device="cuda", dtype=dtype) * 0.2
        pos_bias = torch.arange(seqlen_q, device="cuda", dtype=dtype) * 0.01
        aux_tensors = [head_bias, pos_bias]
        eager_score_mod = eager_factory(head_bias, pos_bias)
    else:
        eager_score_mod = eager_factory

    # Prepare reference tensors
    q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k = prepare_ref_tensors(
        q, k, v, cu_seqlens_q, cu_seqlens_k, varlen_q, varlen_k, batch_size, seqlens_q
    )

    out_ref_fp32 = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=torch.float32
    )
    out_pt = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=dtype
    )
    out_cute = run_cute_flash(
        q,
        k,
        v,
        cute_score_mod,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )

    # Handle output shape differences for varlen_k only case
    if not varlen_q and varlen_k:
        seqlen_q = q.shape[1]
        out_ref_fp32 = out_ref_fp32.reshape(batch_size, seqlen_q, num_heads, head_dim)
        out_pt = out_pt.reshape(batch_size, seqlen_q, num_heads, head_dim)

    assert out_cute.shape == out_ref_fp32.shape, (
        f"Shape mismatch: {out_cute.shape} vs {out_ref_fp32.shape}"
    )

    test_name = f"{cute_score_mod.__name__} (varlen_q={varlen_q}, varlen_k={varlen_k})"
    extra_atol = 2e-3
    check_results(
        out_cute,
        out_ref_fp32,
        out_pt,
        test_name,
        extra_atol=extra_atol,
        seqlens_q=seqlens_q if varlen_q else None,
        cu_seqlens_q=cu_seqlens_q if varlen_q else None,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("varlen_q", [True, False])
@pytest.mark.parametrize("varlen_k", [True, False])
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 1), (4, 2)])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_8ARG)
def test_varlen_with_global_idx_score_mod(
    seqlens_q,
    seqlens_k,
    varlen_q,
    varlen_k,
    qhead_per_kvhead,
    num_kv_heads,
    dtype,
    score_mod_tuple,
):
    """Test varlen attention with 8-arg score_mod functions (global indices).

    These score_mods use q_idx_global and/or kv_idx_global for packed tensor indexing.
    Skips tests where required global indices aren't available.
    """
    if not varlen_q and not varlen_k:
        pytest.skip(
            "At least one of varlen_q or varlen_k must be True for varlen tests"
        )

    cute_score_mod, eager_factory, aux_type, requires_global = score_mod_tuple

    # cute_score_mod = score_mod_global_kv_bias_fresh

    # Skip if score_mod requires global indices we can't provide
    if requires_global == "q" and not varlen_q:
        pytest.skip(f"{cute_score_mod.__name__} requires varlen_q for q_idx_global")
    if requires_global == "kv" and not varlen_k:
        pytest.skip(f"{cute_score_mod.__name__} requires varlen_k for kv_idx_global")
    if requires_global == "both" and (not varlen_q or not varlen_k):
        pytest.skip(f"{cute_score_mod.__name__} requires both varlen_q and varlen_k")

    # For non-varlen dimension, all sequences must have same length
    if not varlen_q:
        seqlens_q = [seqlens_q[0]] * len(seqlens_q)
    if not varlen_k:
        seqlens_k = [seqlens_k[0]] * len(seqlens_k)

    torch.random.manual_seed(42)

    num_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    head_dim = 128
    batch_size = len(seqlens_q)
    max_rel_pos = 512

    # Compute total sizes for aux tensors
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    # Always create cu_seqlens for global index computation (needed by eager)
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )

    # Create tensors - layout depends on varlen flag
    if varlen_q:
        q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    else:
        seqlen_q = seqlens_q[0]
        q = torch.randn(
            batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=dtype
        )

    if varlen_k:
        k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    else:
        seqlen_k = seqlens_k[0]
        k = torch.randn(
            batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype
        )

    # For pack_gqa, reduce K and V to num_kv_heads
    if pack_gqa:
        if varlen_k:
            k = k[:, :num_kv_heads, :].clone()
            v = v[:, :num_kv_heads, :].clone()
        else:
            k = k[:, :, :num_kv_heads, :].clone()
            v = v[:, :, :num_kv_heads, :].clone()

    # Setup aux tensors based on indexing type
    if aux_type == "kv":
        bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias, cu_seqlens_k)
    elif aux_type == "q":
        bias = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias, cu_seqlens_q)
    elif aux_type == "q_and_kv":
        q_bias = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        kv_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [q_bias, kv_bias]
        eager_score_mod = eager_factory(q_bias, kv_bias, cu_seqlens_q, cu_seqlens_k)
    elif aux_type == "q_concat":
        bias = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias, cu_seqlens_q)
    elif aux_type == "kv_with_cu":
        kv_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [kv_bias]
        eager_score_mod = eager_factory(kv_bias, cu_seqlens_q, cu_seqlens_k)
    elif aux_type == "multi_buffer":
        batch_bias = torch.randn(batch_size, device="cuda", dtype=dtype) * 0.1
        head_scale = torch.randn(num_heads, device="cuda", dtype=dtype) * 0.1 + 1.0
        q_pos_bias = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        kv_pos_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        rel_pos_scale = (
            torch.randn(max_rel_pos * 2 + 1, device="cuda", dtype=dtype) * 0.1
        )
        aux_tensors = [batch_bias, head_scale, q_pos_bias, kv_pos_bias, rel_pos_scale]
        eager_score_mod = eager_factory(
            batch_bias,
            head_scale,
            q_pos_bias,
            kv_pos_bias,
            rel_pos_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_rel_pos,
        )
    else:
        raise ValueError(f"Unknown aux_type: {aux_type}")

    # Prepare reference tensors for flex_attention
    q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k = prepare_ref_tensors(
        q, k, v, cu_seqlens_q, cu_seqlens_k, varlen_q, varlen_k, batch_size, seqlens_q
    )

    out_ref_fp32 = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=torch.float32
    )
    out_pt = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=dtype
    )

    # For kernel: pass cu_seqlens only when actually varlen
    kernel_cu_seqlens_q = cu_seqlens_q if varlen_q else None
    kernel_cu_seqlens_k = cu_seqlens_k if varlen_k else None
    out_cute = run_cute_flash(
        q,
        k,
        v,
        cute_score_mod,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
        cu_seqlens_q=kernel_cu_seqlens_q,
        cu_seqlens_k=kernel_cu_seqlens_k,
    )

    # Reshape outputs to common format for comparison
    # Target shape: (batch_size, seqlen_q, num_heads, head_dim) for non-varlen_q
    #           or: (total_q, num_heads, head_dim) for varlen_q

    if varlen_q:
        # Both ref and cute should be (total_q, num_heads, head_dim)
        # ref comes from concatenating per-sequence outputs
        out_ref_final = out_ref_fp32
        out_pt_final = out_pt
        out_cute_final = out_cute
    else:
        # cute is (batch_size, seqlen_q, num_heads, head_dim)
        # ref is (total_q, num_heads, head_dim) from run_flex_varlen_ref concatenation
        # Need to reshape ref to match cute
        seqlen_q = seqlens_q[0]
        out_ref_final = out_ref_fp32.reshape(batch_size, seqlen_q, num_heads, head_dim)
        out_pt_final = out_pt.reshape(batch_size, seqlen_q, num_heads, head_dim)
        out_cute_final = out_cute

    assert out_cute_final.shape == out_ref_final.shape, (
        f"Shape mismatch: {out_cute_final.shape} vs {out_ref_final.shape}"
    )

    test_name = f"{cute_score_mod.__name__} (varlen_q={varlen_q}, varlen_k={varlen_k}, {aux_type})"

    # Debug: print first few values for stress_complex
    if "stress_complex" in cute_score_mod.__name__:
        print(f"\nDEBUG {test_name}:")
        print(f"seqlens_q: {seqlens_q}")
        print(f"cu_seqlens_q: {cu_seqlens_q}")
        print(f"Bias tensor: {aux_tensors[0][:6]}")

        # Print expected values for first few positions
        print("\nExpected reference values for first positions:")
        for b in range(min(3, len(seqlens_q))):
            for h in range(min(2, num_heads)):
                for q in range(min(2, seqlens_q[b])):
                    q_global = cu_seqlens_q[b] + q
                    bias_q = aux_tensors[0][q_global].item()
                    scale = (b + 1) * (h + 1) * 0.001
                    print(
                        f"  REF: b={b} h={h} q_local={q} q_global={q_global} bias_q={bias_q:.6f} scale={scale:.6f}"
                    )

        print(f"\nout_cute_final[0,0,0]: {out_cute_final[0, 0, 0]}")
        print(f"out_ref_final[0,0,0]: {out_ref_final[0, 0, 0]}")
        print(f"Difference: {(out_cute_final[0, 0, 0] - out_ref_final[0, 0, 0]).abs()}")

    check_results(
        out_cute_final,
        out_ref_final,
        out_pt_final,
        test_name,
        extra_atol=1e-3,
        seqlens_q=seqlens_q if varlen_q else None,
        cu_seqlens_q=cu_seqlens_q if varlen_q else None,
        seqlens_q=seqlens_q if varlen_q else None,
        cu_seqlens_q=cu_seqlens_q if varlen_q else None
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
