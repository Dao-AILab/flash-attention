import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention
from flash_attn.cute.interface import _flash_attn_fwd
from test_score_mod import _generate_block_kvcache
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
)  # isort: split
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

IS_SM90 = torch.cuda.get_device_capability()[0] == 9

# =============================================================================
# Test pairs
# =============================================================================

# (cute_score_mod, eager_factory_or_fn, aux_type)
# aux_type: None, "batch", "dual_buffer"
# All score_mods use 7-arg signature: (tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors)
TEST_PAIRS_NO_GLOBAL = [
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
# All score_mods use 7-arg signature and compute global indices from seqlen_info
TEST_PAIRS_WITH_GLOBAL = [
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
    page_table=None,
    seqused_k=None,
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
            seqused_k=seqused_k,
            page_table=page_table,
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
        seqused_k=seqused_k,
        page_table=page_table,
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
        seqlen_q = q.shape[1]
        q_packed = q.reshape(-1, num_heads, q.shape[-1])
        ref_cu_seqlens_q = torch.tensor(
            [seqlen_q * i for i in range(batch_size + 1)],
            device="cuda",
            dtype=torch.int32,
        )
        return q_packed, k, v, ref_cu_seqlens_q, cu_seqlens_k

    if varlen_q and not varlen_k:
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
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(4, 2)])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_NO_GLOBAL)
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
    """Test varlen attention with score_mod functions that don't use global indices.

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

    if pack_gqa:
        if varlen_k:
            k = k[:, :num_kv_heads, :].clone()
            v = v[:, :num_kv_heads, :].clone()
        else:
            k = k[:, :, :num_kv_heads, :].clone()
            v = v[:, :, :num_kv_heads, :].clone()

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("varlen_q", [True, False])
@pytest.mark.parametrize("varlen_k", [True, False])
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 1), (4, 2)])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_WITH_GLOBAL)
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
    """Test varlen attention with score_mod functions that use global indices.

    These score_mods compute q_idx_global and/or kv_idx_global from seqlen_info for packed tensor indexing.
    Skips tests where required global indices aren't available.
    """
    if not varlen_q and not varlen_k:
        pytest.skip(
            "At least one of varlen_q or varlen_k must be True for varlen tests"
        )

    cute_score_mod, eager_factory, aux_type, requires_global = score_mod_tuple

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

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

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

    if varlen_q:
        out_ref_final = out_ref_fp32
        out_pt_final = out_pt
        out_cute_final = out_cute
    else:
        seqlen_q = seqlens_q[0]
        out_ref_final = out_ref_fp32.reshape(batch_size, seqlen_q, num_heads, head_dim)
        out_pt_final = out_pt.reshape(batch_size, seqlen_q, num_heads, head_dim)
        out_cute_final = out_cute

    assert out_cute_final.shape == out_ref_final.shape, (
        f"Shape mismatch: {out_cute_final.shape} vs {out_ref_final.shape}"
    )

    test_name = f"{cute_score_mod.__name__} (varlen_q={varlen_q}, varlen_k={varlen_k}, {aux_type})"

    check_results(
        out_cute_final,
        out_ref_final,
        out_pt_final,
        test_name,
        extra_atol=1e-3,
        seqlens_q=seqlens_q if varlen_q else None,
        cu_seqlens_q=cu_seqlens_q if varlen_q else None,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [None, 128])
@pytest.mark.parametrize("varlen_q", [True, False])
@pytest.mark.parametrize("varlen_k", [True, False])
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(4, 2)])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_NO_GLOBAL)
def test_varlen_score_mod_kvcache(
    seqlens_q,
    seqlens_k,
    varlen_q,
    varlen_k,
    qhead_per_kvhead,
    num_kv_heads,
    page_size,
    dtype,
    score_mod_tuple,
):
    """Test varlen attention with score_mod and paged KV cache."""
    if IS_SM90 and page_size is not None:
        pytest.xfail("paged KV not supported on SM90")

    if not varlen_q and not varlen_k:
        pytest.skip(
            "At least one of varlen_q or varlen_k must be True for varlen tests"
        )

    if page_size is not None and varlen_k:
        pytest.skip("Paged KV requires batched (non-varlen) K")

    if not varlen_q:
        seqlens_q = [seqlens_q[0]] * len(seqlens_q)
    if not varlen_k:
        seqlens_k = [seqlens_k[0]] * len(seqlens_k)

    # Skip if page_size doesn't divide seqlens evenly (for simplicity)
    if page_size is not None and not varlen_k:
        if seqlens_k[0] % page_size != 0:
            pytest.skip("page_size must divide seqlen_k")

    torch.random.manual_seed(42)
    cute_score_mod, eager_factory, aux_type = score_mod_tuple

    num_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    head_dim = 128
    batch_size = len(seqlens_q)
    device = "cuda"

    # Setup tensors
    q, k, v, cu_seqlens_q, cu_seqlens_k = setup_tensors(
        seqlens_q, seqlens_k, varlen_q, varlen_k, num_heads, head_dim, dtype
    )

    if pack_gqa:
        if varlen_k:
            k = k[:, :num_kv_heads, :].clone()
            v = v[:, :num_kv_heads, :].clone()
        else:
            k = k[:, :, :num_kv_heads, :].clone()
            v = v[:, :, :num_kv_heads, :].clone()

    page_table = None
    k_cache_paged = None
    v_cache_paged = None
    k_cache = k
    v_cache = v

    if page_size is not None:
        seqlen_k = seqlens_k[0]
        (
            k_cache_bhsd,
            v_cache_bhsd,
            page_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, page_size, batch_size, num_kv_heads, head_dim, device, dtype
        )
        k_cache = k_cache_bhsd.transpose(1, 2)  # BHSD -> BSHD
        v_cache = v_cache_bhsd.transpose(1, 2)
        seqused_k = torch.tensor(seqlens_k, dtype=torch.int32, device=device)
    else:
        seqused_k = None

    # Setup aux tensors and eager score_mod
    aux_tensors = None
    if aux_type == "batch":
        bias = torch.zeros(batch_size, device=device, dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias)
    elif aux_type == "dual_buffer":
        seqlen_q = seqlens_q[0] if not varlen_q else max(seqlens_q)
        head_bias = torch.randn(num_heads, device=device, dtype=dtype) * 0.2
        pos_bias = torch.arange(seqlen_q, device=device, dtype=dtype) * 0.01
        aux_tensors = [head_bias, pos_bias]
        eager_score_mod = eager_factory(head_bias, pos_bias)
    else:
        eager_score_mod = eager_factory

    # Prepare reference tensors
    q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k = prepare_ref_tensors(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        varlen_q,
        varlen_k,
        batch_size,
        seqlens_q,
    )

    out_ref_fp32 = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=torch.float32
    )
    out_pt = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=dtype
    )

    k_input = k_cache_paged if page_size is not None else k_cache
    v_input = v_cache_paged if page_size is not None else v_cache

    out_cute = run_cute_flash(
        q,
        k_input,
        v_input,
        cute_score_mod,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
        cu_seqlens_q=cu_seqlens_q if varlen_q else None,
        cu_seqlens_k=cu_seqlens_k if (varlen_k and page_size is None) else None,
        page_table=page_table if page_size is not None else None,
        seqused_k=seqused_k if page_size is not None else None,
    )

    if not varlen_q and varlen_k:
        seqlen_q = q.shape[1]
        out_ref_fp32 = out_ref_fp32.reshape(batch_size, seqlen_q, num_heads, head_dim)
        out_pt = out_pt.reshape(batch_size, seqlen_q, num_heads, head_dim)

    assert out_cute.shape == out_ref_fp32.shape, (
        f"Shape mismatch: {out_cute.shape} vs {out_ref_fp32.shape}"
    )

    test_name = f"{cute_score_mod.__name__} (varlen_q={varlen_q}, varlen_k={varlen_k}, paged={page_size is not None})"
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [None, 128])
@pytest.mark.parametrize("varlen_q", [True, False])
@pytest.mark.parametrize("varlen_k", [True, False])
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 1), (4, 2)])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_WITH_GLOBAL)
def test_varlen_score_mod_with_paged_kvcache_global(
    seqlens_q,
    seqlens_k,
    varlen_q,
    varlen_k,
    qhead_per_kvhead,
    num_kv_heads,
    page_size,
    dtype,
    score_mod_tuple,
):
    """Test varlen attention with global idx score_mod and paged KV cache."""
    if IS_SM90 and page_size is not None:
        pytest.xfail("paged KV not supported on SM90")

    if page_size is not None and varlen_k:
        pytest.skip("Paged KV cache requires batched (non-varlen) K")

    if not varlen_q and not varlen_k:
        pytest.skip(
            "At least one of varlen_q or varlen_k must be True for varlen tests"
        )

    if not varlen_q:
        seqlens_q = [seqlens_q[0]] * len(seqlens_q)
    if not varlen_k:
        seqlens_k = [seqlens_k[0]] * len(seqlens_k)

    if page_size is not None and not varlen_k:
        if seqlens_k[0] % page_size != 0:
            pytest.skip("page_size must divide seqlen_k")

    cute_score_mod, eager_factory, aux_type, requires_global = score_mod_tuple

    if requires_global == "q" and not varlen_q:
        pytest.skip(f"{cute_score_mod.__name__} requires varlen_q for q_idx_global")
    if requires_global == "kv" and not varlen_k:
        pytest.skip(f"{cute_score_mod.__name__} requires varlen_k for kv_idx_global")
    if requires_global == "both" and (not varlen_q or not varlen_k):
        pytest.skip(f"{cute_score_mod.__name__} requires both varlen_q and varlen_k")

    torch.random.manual_seed(42)

    num_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    head_dim = 128
    batch_size = len(seqlens_q)
    max_rel_pos = 512
    device = "cuda"

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

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
    cu_seqlens_k_for_kernel = cu_seqlens_k if varlen_k else None

    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    if varlen_k:
        k = torch.randn(total_k, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_k, num_heads, head_dim, device=device, dtype=dtype)
    else:
        seqlen_k = seqlens_k[0]
        k = torch.randn(
            batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seqlen_k, num_heads, head_dim, device=device, dtype=dtype
        )

    if pack_gqa:
        if varlen_k:
            k = k[:, :num_kv_heads, :].clone()
            v = v[:, :num_kv_heads, :].clone()
        else:
            k = k[:, :, :num_kv_heads, :].clone()
            v = v[:, :, :num_kv_heads, :].clone()

    page_table = None
    k_cache_paged = None
    v_cache_paged = None
    k_cache = k
    v_cache = v

    if page_size is not None:
        seqlen_k = seqlens_k[0]
        (
            k_cache_bhsd,
            v_cache_bhsd,
            page_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, page_size, batch_size, num_kv_heads, head_dim, device, dtype
        )
        k_cache = k_cache_bhsd.transpose(1, 2)  # BHSD -> BSHD
        v_cache = v_cache_bhsd.transpose(1, 2)
        seqused_k = torch.tensor(seqlens_k, dtype=torch.int32, device=device)
    else:
        seqused_k = None

    if aux_type == "kv":
        bias = torch.randn(total_k, device=device, dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias, cu_seqlens_k)
    elif aux_type == "q":
        bias = torch.randn(total_q, device=device, dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias, cu_seqlens_q)
    elif aux_type == "q_and_kv":
        q_bias = torch.randn(total_q, device=device, dtype=dtype) * 0.1
        kv_bias = torch.randn(total_k, device=device, dtype=dtype) * 0.1
        aux_tensors = [q_bias, kv_bias]
        eager_score_mod = eager_factory(q_bias, kv_bias, cu_seqlens_q, cu_seqlens_k)
    elif aux_type == "q_concat":
        bias = torch.randn(total_q, device=device, dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias, cu_seqlens_q)
    elif aux_type == "kv_with_cu":
        kv_bias = torch.randn(total_k, device=device, dtype=dtype) * 0.1
        aux_tensors = [kv_bias]
        eager_score_mod = eager_factory(kv_bias, cu_seqlens_q, cu_seqlens_k)
    elif aux_type == "multi_buffer":
        batch_bias = torch.randn(batch_size, device=device, dtype=dtype) * 0.1
        head_scale = torch.randn(num_heads, device=device, dtype=dtype) * 0.1 + 1.0
        q_pos_bias = torch.randn(total_q, device=device, dtype=dtype) * 0.1
        kv_pos_bias = torch.randn(total_k, device=device, dtype=dtype) * 0.1
        rel_pos_scale = (
            torch.randn(max_rel_pos * 2 + 1, device=device, dtype=dtype) * 0.1
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

    q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k = prepare_ref_tensors(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        True,
        varlen_k,
        batch_size,
        seqlens_q,
    )

    out_ref_fp32 = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=torch.float32
    )
    out_pt = run_flex_varlen_ref(
        q_ref, k_ref, v_ref, ref_cu_q, ref_cu_k, eager_score_mod, dtype=dtype
    )

    # Run CuTE
    k_input = k_cache_paged if page_size is not None else k_cache
    v_input = v_cache_paged if page_size is not None else v_cache

    out_cute = torch.empty_like(q)
    _flash_attn_fwd(
        q,
        k_input,
        v_input,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k_for_kernel if page_size is None else None,
        seqused_k=seqused_k if page_size is not None else None,
        page_table=page_table,
        return_lse=True,
        score_mod=cute_score_mod,
        out=out_cute,
        lse=None,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
    )

    assert out_cute.shape == out_ref_fp32.shape, (
        f"Shape mismatch: {out_cute.shape} vs {out_ref_fp32.shape}"
    )

    test_name = f"{cute_score_mod.__name__} (paged={page_size is not None}, {aux_type})"
    check_results(
        out_cute,
        out_ref_fp32,
        out_pt,
        test_name,
        extra_atol=1e-3,
        seqlens_q=seqlens_q,
        cu_seqlens_q=cu_seqlens_q,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
