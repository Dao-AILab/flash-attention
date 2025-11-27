import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as mlir_math
import operator
from torch.nn.attention.flex_attention import flex_attention
from flash_attn.cute.interface import _flash_attn_fwd


# =============================================================================
# 6-argument score_mod functions
# =============================================================================


@cute.jit
def score_mod_identity(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    return tSrS_ssa


@cute.jit
def score_mod_causal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    mask = operator.ge(q_idx, kv_idx)
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_rel_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    diff = q_idx - kv_idx
    abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
    return tSrS_ssa + abs_diff.to(cutlass.Float32)


@cute.jit
def score_mod_rel_bias_x2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    diff = q_idx - kv_idx
    abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
    scaled = abs_diff * cute.full_like(abs_diff, 2)
    return tSrS_ssa + scaled.to(cutlass.Float32)


@cute.jit
def score_mod_times_two(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    return tSrS_ssa * cute.full_like(tSrS_ssa, 2)


@cute.jit
def score_mod_alibi(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    score = tSrS_ssa.to(cutlass.Float32)
    slope_exp = (h_idx + cute.full_like(h_idx, 1)) * cute.full_like(h_idx, -8)
    slope = cute.math.exp2(
        slope_exp.to(cutlass.Float32)
        * cute.full_like(score, 0.125 * 0.6931471805599453 * 1.4426950408889634)
    )
    diff = q_idx - kv_idx
    abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype).to(cutlass.Float32)
    return score - slope * abs_diff


@cute.jit
def score_mod_sliding_window(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    diff = q_idx - kv_idx
    abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
    mask = operator.le(abs_diff, cute.full_like(abs_diff, 256))
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_block_diagonal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    q_block = q_idx // 64
    kv_block = kv_idx // 64
    mask = operator.eq(q_block, kv_block)
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_causal_v2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    diff = q_idx - kv_idx
    mask = operator.ge(diff, cute.full_like(diff, 0))
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_batch_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    batch_bias = aux_tensors[0]
    dtype = batch_bias.element_type
    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = batch_bias[b_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)
    return tSrS_ssa + bias_val


@cute.jit
def score_mod_dual_buffer(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    head_bias = aux_tensors[0]
    pos_bias = aux_tensors[1]
    dtype = head_bias.element_type

    h_frag = cute.make_fragment(1, cutlass.Int32)
    h_frag.store(h_idx)
    head_val_frag = cute.make_fragment(1, dtype)
    head_val_frag[0] = head_bias[h_frag[0]]
    head_val = (head_val_frag.load()).to(cutlass.Float32)

    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx)
    pos_val_frag = cute.make_fragment(1, dtype)
    pos_val_frag[0] = pos_bias[q_frag[0]]
    pos_val = (pos_val_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + head_val + pos_val


# =============================================================================
# 8-argument score_mod functions (with global indices)
# =============================================================================


@cute.jit
def score_mod_global_kv_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Per-token bias using global kv index."""
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type
    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    return tSrS_ssa + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_global_q_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Per-token bias using global q index."""
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type
    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[q_frag[0]]
    return tSrS_ssa + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_global_rel_plus_kv_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Relative position (logical) + per-token bias (global kv)."""
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type

    rel_pos = q_idx - kv_idx
    rel_pos_abs = cute.TensorSSA(mlir_math.absi(rel_pos), rel_pos.shape, rel_pos.dtype)
    rel_bias = rel_pos_abs.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.1)

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]

    return tSrS_ssa + rel_bias + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_global_q_and_kv_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Both q and kv global indices."""
    q_bias = aux_tensors[0]
    kv_bias = aux_tensors[1]
    dtype = q_bias.element_type

    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx_global)
    q_bias_frag = cute.make_fragment(1, dtype)
    q_bias_frag[0] = q_bias[q_frag[0]]

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    kv_bias_frag = cute.make_fragment(1, dtype)
    kv_bias_frag[0] = kv_bias[kv_frag[0]]

    return (
        tSrS_ssa
        + (q_bias_frag.load()).to(cutlass.Float32)
        + (kv_bias_frag.load()).to(cutlass.Float32)
    )


@cute.jit
def score_mod_global_logical_rel_plus_kv_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Logical relative + global-indexed per-token bias."""
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type

    rel_pos = q_idx - kv_idx
    rel_pos_abs = cute.TensorSSA(mlir_math.absi(rel_pos), rel_pos.shape, rel_pos.dtype)
    rel_bias = rel_pos_abs.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.01)

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]

    return tSrS_ssa + rel_bias + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_stress_complex_arithmetic(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """All indices in complex arithmetic."""
    bias = aux_tensors[0]
    dtype = bias.element_type

    rel_pos = q_idx - kv_idx
    rel_pos_sq = rel_pos * rel_pos

    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx_global)
    bias_q_frag = cute.make_fragment(1, dtype)
    bias_q_frag[0] = bias[q_frag[0]]
    bias_q = (bias_q_frag.load()).to(cutlass.Float32)

    scale = (b_idx + cute.full_like(b_idx, 1)) * (h_idx + cute.full_like(h_idx, 1))
    scale_f32 = scale.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.001)

    rel_bias = rel_pos_sq.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.0001)

    return tSrS_ssa + rel_bias + bias_q * scale_f32


@cute.jit
def score_mod_stress_conditional_mask(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Conditional masking with global vs logical."""
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)

    is_causal = operator.ge(q_idx, kv_idx)

    global_diff = q_idx_global - kv_idx_global
    is_nearby = operator.le(
        cute.TensorSSA(mlir_math.absi(global_diff), global_diff.shape, global_diff.dtype),
        cute.full_like(global_diff, 512),
    )

    both_conditions = is_causal & is_nearby
    return cute.where(both_conditions, tSrS_ssa + bias_val, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_stress_multi_buffer(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Multiple aux tensors with different indexing."""
    batch_bias = aux_tensors[0]
    head_scale = aux_tensors[1]
    q_pos_bias = aux_tensors[2]
    kv_pos_bias = aux_tensors[3]
    rel_pos_scale = aux_tensors[4]

    dtype = batch_bias.element_type

    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    bb_frag = cute.make_fragment(1, dtype)
    bb_frag[0] = batch_bias[b_frag[0]]
    bb_val = (bb_frag.load()).to(cutlass.Float32)

    h_frag = cute.make_fragment(1, cutlass.Int32)
    h_frag.store(h_idx)
    hs_frag = cute.make_fragment(1, dtype)
    hs_frag[0] = head_scale[h_frag[0]]
    hs_val = (hs_frag.load()).to(cutlass.Float32)

    qg_frag = cute.make_fragment(1, cutlass.Int32)
    qg_frag.store(q_idx_global)
    qpb_frag = cute.make_fragment(1, dtype)
    qpb_frag[0] = q_pos_bias[qg_frag[0]]
    qpb_val = (qpb_frag.load()).to(cutlass.Float32)

    kvg_frag = cute.make_fragment(1, cutlass.Int32)
    kvg_frag.store(kv_idx_global)
    kvpb_frag = cute.make_fragment(1, dtype)
    kvpb_frag[0] = kv_pos_bias[kvg_frag[0]]
    kvpb_val = (kvpb_frag.load()).to(cutlass.Float32)

    rel_idx = q_idx - kv_idx + cute.full_like(q_idx, 512)
    rel_idx_clamped = cute.where(
        operator.lt(rel_idx, cute.full_like(rel_idx, 0)), cute.full_like(rel_idx, 0), rel_idx
    )
    rel_idx_clamped = cute.where(
        operator.gt(rel_idx_clamped, cute.full_like(rel_idx_clamped, 1024)),
        cute.full_like(rel_idx_clamped, 1024),
        rel_idx_clamped,
    )
    ri_frag = cute.make_fragment(1, cutlass.Int32)
    ri_frag.store(rel_idx_clamped)
    rps_frag = cute.make_fragment(1, dtype)
    rps_frag[0] = rel_pos_scale[ri_frag[0]]
    rps_val = (rps_frag.load()).to(cutlass.Float32)

    return tSrS_ssa * hs_val + bb_val + qpb_val + kvpb_val + rps_val * cute.full_like(tSrS_ssa, 0.1)


@cute.jit
def score_mod_stress_global_offset(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """Verify global - logical = offset."""
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]

    return tSrS_ssa + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_stress_xor_pattern(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global
):
    """XOR-based pattern using index bits."""
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type

    xor_logical = q_idx ^ kv_idx
    pattern_logical = xor_logical & cute.full_like(xor_logical, 0xFF)
    pattern_bias = pattern_logical.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.001)

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]

    return (
        tSrS_ssa
        + pattern_bias
        + (bias_frag.load()).to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.1)
    )


# =============================================================================
# Eager reference functions
# =============================================================================


def identity_eager(score, b, h, q_idx, kv_idx):
    return score


def causal_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def rel_bias_eager(score, b, h, q_idx, kv_idx):
    return score + torch.abs(q_idx - kv_idx)


def rel_bias_x2_eager(score, b, h, q_idx, kv_idx):
    return score + 2 * torch.abs(q_idx - kv_idx)


def times_two_eager(score, b, h, q_idx, kv_idx):
    return score * 2


def alibi_eager(score, b, h, q_idx, kv_idx):
    slope = 2 ** (-8 * (h + 1) / 8)
    return score - slope * torch.abs(q_idx - kv_idx)


def sliding_window_eager(score, b, h, q_idx, kv_idx):
    return torch.where(torch.abs(q_idx - kv_idx) <= 256, score, float("-inf"))


def block_diagonal_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx // 64 == kv_idx // 64, score, float("-inf"))


def causal_v2_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx - kv_idx >= 0, score, float("-inf"))


def batch_bias_factory(bias_tensor):
    def mod(score, b, h, q_idx, kv_idx):
        return score + bias_tensor[b]

    return mod


def dual_buffer_factory(head_bias, pos_bias):
    def mod(score, b, h, q_idx, kv_idx):
        return score + head_bias[h] + pos_bias[q_idx]

    return mod


def packed_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        return score + bias_tensor[cu_seqlens_k[b] + kv_idx]

    return mod


def packed_q_bias_factory(bias_tensor, cu_seqlens_q):
    def mod(score, b, h, q_idx, kv_idx):
        return score + bias_tensor[cu_seqlens_q[b] + q_idx]

    return mod


def packed_rel_plus_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.1
        return score + rel_bias + bias_tensor[cu_seqlens_k[b] + kv_idx]

    return mod


def packed_q_and_kv_bias_factory(q_bias, kv_bias, cu_seqlens_q, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        return score + q_bias[cu_seqlens_q[b] + q_idx] + kv_bias[cu_seqlens_k[b] + kv_idx]

    return mod


def packed_logical_rel_plus_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.01
        return score + rel_bias + bias_tensor[cu_seqlens_k[b] + kv_idx]

    return mod


def stress_complex_arithmetic_factory(bias, cu_seqlens_q):
    def mod(score, b, h, q_idx, kv_idx):
        rel_pos_sq = (q_idx - kv_idx) ** 2
        q_global = cu_seqlens_q[b] + q_idx
        bias_q = bias[q_global]
        scale = (b + 1) * (h + 1) * 0.001
        rel_bias = rel_pos_sq.float() * 0.0001
        return score + rel_bias + bias_q * scale

    return mod


def stress_conditional_mask_factory(token_bias, cu_seqlens_q, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        kv_global = cu_seqlens_k[b] + kv_idx
        bias_val = token_bias[kv_global]
        is_causal = q_idx >= kv_idx
        q_global = cu_seqlens_q[b] + q_idx
        global_diff = q_global - kv_global
        is_nearby = torch.abs(global_diff) <= 512
        both_conditions = is_causal & is_nearby
        return torch.where(both_conditions, score + bias_val, float("-inf"))

    return mod


def stress_multi_buffer_factory(
    batch_bias,
    head_scale,
    q_pos_bias,
    kv_pos_bias,
    rel_pos_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    max_rel_pos=512,
):
    def mod(score, b, h, q_idx, kv_idx):
        bb_val = batch_bias[b]
        hs_val = head_scale[h]
        qpb_val = q_pos_bias[cu_seqlens_q[b] + q_idx]
        kvpb_val = kv_pos_bias[cu_seqlens_k[b] + kv_idx]
        rel_idx = (q_idx - kv_idx + max_rel_pos).clamp(0, max_rel_pos * 2)
        rps_val = rel_pos_scale[rel_idx]
        return score * hs_val + bb_val + qpb_val + kvpb_val + rps_val * 0.1

    return mod


def stress_global_offset_factory(token_bias, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        return score + token_bias[cu_seqlens_k[b] + kv_idx]

    return mod


def stress_xor_pattern_factory(token_bias, cu_seqlens_q, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        xor_logical = q_idx ^ kv_idx
        pattern_bias = (xor_logical & 0xFF).float() * 0.001
        kv_global = cu_seqlens_k[b] + kv_idx
        return score + pattern_bias + token_bias[kv_global] * 0.1

    return mod


# =============================================================================
# Test pairs
# =============================================================================

# (cute_score_mod, eager_factory_or_fn, aux_type)
# aux_type: None, "batch", "dual_buffer"
TEST_PAIRS_6ARG = [
    (score_mod_identity, identity_eager, None),
    # (score_mod_causal, causal_eager, None),
    # (score_mod_rel_bias, rel_bias_eager, None),
    # (score_mod_rel_bias_x2, rel_bias_x2_eager, None),
    # (score_mod_times_two, times_two_eager, None),
    # (score_mod_alibi, alibi_eager, None),
    # (score_mod_sliding_window, sliding_window_eager, None),
    # (score_mod_block_diagonal, block_diagonal_eager, None),
    # (score_mod_causal_v2, causal_v2_eager, None),
    # (score_mod_batch_bias, batch_bias_factory, "batch"),
    # (score_mod_dual_buffer, dual_buffer_factory, "dual_buffer"),
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
    (score_mod_stress_complex_arithmetic, stress_complex_arithmetic_factory, "q_concat", "q"),
    (score_mod_stress_conditional_mask, stress_conditional_mask_factory, "kv_with_cu", "both"),
    (score_mod_stress_multi_buffer, stress_multi_buffer_factory, "multi_buffer", "both"),
    (score_mod_stress_global_offset, stress_global_offset_factory, "kv", "kv"),
    (score_mod_stress_xor_pattern, stress_xor_pattern_factory, "kv_with_cu", "kv"),
]

SEQLEN_CONFIGS = [
    ([1], [1]),
    ([1, 1], [1, 1]),
    ([2, 3], [2, 3]),
    ([8, 16], [8, 16]),
    ([32, 64], [32, 64]),
    ([64, 128], [64, 128]),
    ([64, 56, 128], [64, 56, 128]),
    ([256, 512], [256, 512]),
    ([113, 203], [113, 203]),
    ([239, 1], [239, 1]),
    ([64], [64]),
    ([128], [128]),
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
]


# =============================================================================
# Helper functions
# =============================================================================


def run_cute_flash(
    q, k, v, score_mod, aux_tensors=None, pack_gqa=False, cu_seqlens_q=None, cu_seqlens_k=None
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
            q_slice = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].unsqueeze(0).transpose(1, 2)
        else:
            q_slice = q[i : i + 1].transpose(1, 2)

        # Get K/V slices
        if cu_seqlens_k is not None:
            k_slice = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0).transpose(1, 2)
            v_slice = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0).transpose(1, 2)
        else:
            k_slice = k[i : i + 1].transpose(1, 2)
            v_slice = v[i : i + 1].transpose(1, 2)

        if dtype is not None:
            q_slice, k_slice, v_slice = q_slice.to(dtype), k_slice.to(dtype), v_slice.to(dtype)

        def wrapped_mod(score, b, h, q_idx, kv_idx, seq_idx=i):
            return score_mod(score, seq_idx, h, q_idx, kv_idx)

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
            [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()), device="cuda", dtype=torch.int32
        )
    else:
        seqlen_q = seqlens_q[0]
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=dtype)
        cu_seqlens_q = None

    if varlen_k:
        total_k = sum(seqlens_k)
        k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
        cu_seqlens_k = torch.tensor(
            [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()), device="cuda", dtype=torch.int32
        )
    else:
        seqlen_k = seqlens_k[0]
        k = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype)
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
            [seqlen_q * i for i in range(batch_size + 1)], device="cuda", dtype=torch.int32
        )
        return q_packed, k, v, ref_cu_seqlens_q, cu_seqlens_k

    if varlen_q and not varlen_k:
        # K is batched (batch_size, seqlen_k, num_heads, head_dim)
        # Need to transpose to (batch_size, num_heads, seqlen_k, head_dim) for flex_attention
        k_ref = k.transpose(1, 2)
        v_ref = v.transpose(1, 2)
        return q, k, v, cu_seqlens_q, None

    return q, k, v, cu_seqlens_q, cu_seqlens_k


def check_results(out_cute, out_ref_fp32, out_pt, test_name, rtol=2, extra_atol=1e-4):
    """Compare CuTE output against references."""
    assert not torch.isnan(out_cute).any(), f"{test_name}: NaN in output"
    assert torch.isfinite(out_cute).all(), f"{test_name}: Inf in output"

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\n{test_name}:")
    print(f"  PyTorch vs FP32 ref: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + extra_atol, (
        f"{test_name}: CuTE error {cute_error:.2e} exceeds tolerance"
    )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("varlen_q", [True, False])
@pytest.mark.parametrize("varlen_k", [True, False])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_6ARG)
def test_varlen_with_score_mod(seqlens_q, seqlens_k, varlen_q, varlen_k, dtype, score_mod_tuple):
    """Test varlen attention with 6-arg score_mod functions.

    Covers: both varlen, varlen Q only, varlen K only.
    Skips: neither varlen
    """
    if not varlen_q and not varlen_k:
        pytest.skip("At least one of varlen_q or varlen_k must be True for varlen tests")

    # For non-varlen dimension, all sequences must have same length
    if not varlen_q:
        seqlens_q = [seqlens_q[0]] * len(seqlens_q)
    if not varlen_k:
        seqlens_k = [seqlens_k[0]] * len(seqlens_k)

    torch.random.manual_seed(42)
    cute_score_mod, eager_factory, aux_type = score_mod_tuple

    num_heads = 4
    head_dim = 128
    batch_size = len(seqlens_q)

    q, k, v, cu_seqlens_q, cu_seqlens_k = setup_tensors(
        seqlens_q, seqlens_k, varlen_q, varlen_k, num_heads, head_dim, dtype
    )

    # Setup aux tensors and eager score_mod
    aux_tensors = None
    if aux_type == "batch":
        bias = torch.randn(batch_size, device="cuda", dtype=dtype) * 0.1
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
    check_results(out_cute, out_ref_fp32, out_pt, test_name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("varlen_q", [True, False])
@pytest.mark.parametrize("varlen_k", [True, False])
@pytest.mark.parametrize("seqlens_q,seqlens_k", SEQLEN_CONFIGS)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_8ARG)
def test_varlen_with_global_idx_score_mod(
    seqlens_q, seqlens_k, varlen_q, varlen_k, dtype, score_mod_tuple
):
    """Test varlen attention with 8-arg score_mod functions (global indices).

    These score_mods use q_idx_global and/or kv_idx_global for packed tensor indexing.
    Skips tests where required global indices aren't available.
    """
    if not varlen_q and not varlen_k:
        pytest.skip("At least one of varlen_q or varlen_k must be True for varlen tests")

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

    num_heads = 4
    head_dim = 128
    batch_size = len(seqlens_q)
    max_rel_pos = 512

    # Compute total sizes for aux tensors
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    # Always create cu_seqlens for global index computation (needed by eager)
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()), device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()), device="cuda", dtype=torch.int32
    )

    # Create tensors - layout depends on varlen flag
    if varlen_q:
        q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    else:
        seqlen_q = seqlens_q[0]
        q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=dtype)

    if varlen_k:
        k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    else:
        seqlen_k = seqlens_k[0]
        k = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype)

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
        rel_pos_scale = torch.randn(max_rel_pos * 2 + 1, device="cuda", dtype=dtype) * 0.1
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
        cu_seqlens_q=kernel_cu_seqlens_q,
        cu_seqlens_k=kernel_cu_seqlens_k,
    )

    # Handle output shape differences for varlen_k only case
    if not varlen_q and varlen_k:
        seqlen_q = q.shape[1]
        out_ref_fp32 = out_ref_fp32.reshape(batch_size, seqlen_q, num_heads, head_dim)
        out_pt = out_pt.reshape(batch_size, seqlen_q, num_heads, head_dim)

    assert out_cute.shape == out_ref_fp32.shape, (
        f"Shape mismatch: {out_cute.shape} vs {out_ref_fp32.shape}"
    )

    test_name = f"{cute_score_mod.__name__} (varlen_q={varlen_q}, varlen_k={varlen_k}, {aux_type})"
    check_results(out_cute, out_ref_fp32, out_pt, test_name, extra_atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
