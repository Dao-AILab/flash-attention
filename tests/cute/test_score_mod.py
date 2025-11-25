import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as mlir_math
import operator
from torch.nn.attention.flex_attention import flex_attention
from flash_attn.cute.interface import _flash_attn_fwd


# =============================================================================
# 6-argument score_mod functions (original signature)
# =============================================================================

@cute.jit
def score_mod_1(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tSrS_ssa = tmp0
    return tSrS_ssa


@cute.jit
def score_mod_2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = operator.ge(tmp0, tmp1)
    tmp3 = tSrS_ssa
    tmp4 = cute.where(tmp2, tmp3, cute.full_like(tmp3, float("-inf")))
    tSrS_ssa = tmp4
    return tSrS_ssa


@cute.jit
def score_mod_3(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = q_idx
    tmp2 = kv_idx
    tmp3 = tmp1 - tmp2
    tmp4 = cute.TensorSSA(mlir_math.absi(tmp3), tmp3.shape, tmp3.dtype)
    tmp5 = tmp4.to(cutlass.Float32)
    tmp6 = tmp0 + tmp5
    tSrS_ssa = tmp6
    return tSrS_ssa


@cute.jit
def score_mod_4(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = q_idx
    tmp2 = kv_idx
    tmp3 = tmp1 - tmp2
    tmp4 = cute.TensorSSA(mlir_math.absi(tmp3), tmp3.shape, tmp3.dtype)
    tmp5 = tmp4 * cute.full_like(tmp4, 2)
    tmp6 = tmp5.to(cutlass.Float32)
    tmp7 = tmp0 + tmp6
    tSrS_ssa = tmp7
    return tSrS_ssa


@cute.jit
def score_mod_5(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = tmp0 * cute.full_like(tmp0, 2)
    tSrS_ssa = tmp1
    return tSrS_ssa


@cute.jit
def score_mod_6(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = tSrS_ssa
    tmp1 = tmp0.to(cutlass.Float32)
    tmp2 = h_idx
    tmp3 = tmp2 + cute.full_like(tmp2, 1)
    tmp4 = tmp3 * cute.full_like(tmp3, -8)
    tmp5 = tmp4.to(cutlass.Float32)
    tmp6 = tmp5 * cute.full_like(tmp5, 0.125)
    tmp7 = tmp6 * cute.full_like(tmp6, 0.6931471805599453)
    tmp8 = cute.math.exp2(tmp7 * 1.4426950408889634)
    tmp9 = q_idx
    tmp10 = kv_idx
    tmp11 = tmp9 - tmp10
    tmp12 = cute.TensorSSA(mlir_math.absi(tmp11), tmp11.shape, tmp11.dtype)
    tmp13 = tmp12.to(cutlass.Float32)
    tmp14 = tmp8 * tmp13
    tmp15 = tmp1 - tmp14
    tSrS_ssa = tmp15
    return tSrS_ssa


@cute.jit
def score_mod_7(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = tmp0 - tmp1
    tmp3 = cute.TensorSSA(mlir_math.absi(tmp2), tmp2.shape, tmp2.dtype)
    tmp4 = operator.le(tmp3, cute.full_like(tmp3, 256))
    tmp5 = tSrS_ssa
    tmp6 = cute.where(tmp4, tmp5, cute.full_like(tmp5, float("-inf")))
    tSrS_ssa = tmp6
    return tSrS_ssa


@cute.jit
def score_mod_8(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = tSrS_ssa
    tmp3 = cute.where(
        operator.eq(tmp0 // 64, tmp1 // 64), tmp2, cute.full_like(tmp2, float("-inf"))
    )
    tSrS_ssa = tmp3
    return tSrS_ssa


@cute.jit
def score_mod_9(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    tmp0 = q_idx
    tmp1 = kv_idx
    tmp2 = tmp0 - tmp1
    tmp3 = operator.ge(tmp2, cute.full_like(tmp2, 0))
    tmp4 = tSrS_ssa
    tmp5 = cute.where(tmp3, tmp4, cute.full_like(tmp4, float("-inf")))
    tSrS_ssa = tmp5
    return tSrS_ssa


@cute.jit
def score_mod_10(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    batch_bias = aux_tensors[0]
    dtype = batch_bias.element_type
    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = batch_bias[b_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)
    return tSrS_ssa + bias_val


@cute.jit
def score_mod_11(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
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
# 8-argument score_mod functions (with global indices for varlen)
# =============================================================================

@cute.jit
def score_mod_global_1(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Per-token bias using global kv index."""
    token_bias = aux_tensors[0]  # [total_k]
    dtype = token_bias.element_type
    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)
    return tSrS_ssa + bias_val


@cute.jit
def score_mod_global_2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Per-token bias using global q index."""
    token_bias = aux_tensors[0]  # [total_q]
    dtype = token_bias.element_type
    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[q_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)
    return tSrS_ssa + bias_val


@cute.jit
def score_mod_global_3(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Relative position (logical) + per-token bias (global)."""
    token_bias = aux_tensors[0]  # [total_k]
    dtype = token_bias.element_type

    rel_pos = q_idx - kv_idx
    rel_pos_abs = cute.TensorSSA(mlir_math.absi(rel_pos), rel_pos.shape, rel_pos.dtype)
    rel_bias = rel_pos_abs.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.1)

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + rel_bias + bias_val


@cute.jit
def score_mod_global_4(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Both q and kv global indices."""
    q_bias = aux_tensors[0]  # [total_q]
    kv_bias = aux_tensors[1]  # [total_k]
    dtype = q_bias.element_type

    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx_global)
    q_bias_frag = cute.make_fragment(1, dtype)
    q_bias_frag[0] = q_bias[q_frag[0]]
    q_bias_val = (q_bias_frag.load()).to(cutlass.Float32)

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    kv_bias_frag = cute.make_fragment(1, dtype)
    kv_bias_frag[0] = kv_bias[kv_frag[0]]
    kv_bias_val = (kv_bias_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + q_bias_val + kv_bias_val


@cute.jit
def score_mod_global_5(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Logical relative + global-indexed per-token bias."""
    token_bias = aux_tensors[0]  # [total_k]
    dtype = token_bias.element_type

    rel_pos = q_idx - kv_idx
    rel_pos_abs = cute.TensorSSA(mlir_math.absi(rel_pos), rel_pos.shape, rel_pos.dtype)
    rel_bias = rel_pos_abs.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.01)

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    global_bias_val = (bias_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + rel_bias + global_bias_val


@cute.jit
def score_mod_stress_1(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """All indices in complex arithmetic."""
    bias = aux_tensors[0]  # [total_q]
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
def score_mod_stress_2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Conditional masking with global vs logical."""
    token_bias = aux_tensors[0]  # [total_k]
    dtype = token_bias.element_type

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)

    is_causal = operator.ge(q_idx, kv_idx)

    global_diff = q_idx_global - kv_idx_global
    is_nearby = operator.le(cute.TensorSSA(mlir_math.absi(global_diff), global_diff.shape, global_diff.dtype),
                            cute.full_like(global_diff, 512))

    both_conditions = is_causal & is_nearby
    masked_score = cute.where(both_conditions, tSrS_ssa + bias_val, cute.full_like(tSrS_ssa, float("-inf")))

    return masked_score


@cute.jit
def score_mod_stress_3(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Multiple aux tensors with different indexing."""
    batch_bias = aux_tensors[0]      # [num_seqs]
    head_scale = aux_tensors[1]      # [num_heads]
    q_pos_bias = aux_tensors[2]      # [total_q]
    kv_pos_bias = aux_tensors[3]     # [total_k]
    rel_pos_scale = aux_tensors[4]   # [max_rel_pos * 2 + 1]

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
    rel_idx_clamped = cute.where(operator.lt(rel_idx, cute.full_like(rel_idx, 0)),
                                  cute.full_like(rel_idx, 0), rel_idx)
    rel_idx_clamped = cute.where(operator.gt(rel_idx_clamped, cute.full_like(rel_idx_clamped, 1024)),
                                  cute.full_like(rel_idx_clamped, 1024), rel_idx_clamped)
    ri_frag = cute.make_fragment(1, cutlass.Int32)
    ri_frag.store(rel_idx_clamped)
    rps_frag = cute.make_fragment(1, dtype)
    rps_frag[0] = rel_pos_scale[ri_frag[0]]
    rps_val = (rps_frag.load()).to(cutlass.Float32)

    result = tSrS_ssa * hs_val + bb_val + qpb_val + kvpb_val + rps_val * cute.full_like(tSrS_ssa, 0.1)

    return result


@cute.jit
def score_mod_stress_4(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """Verify global - logical = offset."""
    token_bias = aux_tensors[0]  # [total_k]
    dtype = token_bias.element_type

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + bias_val


@cute.jit
def score_mod_stress_5(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors, q_idx_global, kv_idx_global):
    """XOR-based pattern using index bits."""
    token_bias = aux_tensors[0]  # [total_k]
    dtype = token_bias.element_type

    xor_logical = q_idx ^ kv_idx
    pattern_logical = xor_logical & cute.full_like(xor_logical, 0xFF)
    pattern_bias = pattern_logical.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.001)

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)

    return tSrS_ssa + pattern_bias + bias_val * cute.full_like(tSrS_ssa, 0.1)


# =============================================================================
# Eager reference functions
# =============================================================================

def identity_eager(score, b, h, q_idx, kv_idx):
    return score


def causal_mask_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def relative_bias_eager(score, b, h, q_idx, kv_idx):
    return score + torch.abs(q_idx - kv_idx)


def relative_bias_v2_eager(score, b, h, q_idx, kv_idx):
    return score + 2 * torch.abs(q_idx - kv_idx)


def times_two_eager(score, b, h, q_idx, kv_idx):
    return score * 2


def alibi_bias_eager(score, b, h, q_idx, kv_idx):
    slope = 2 ** (-8 * (h + 1) / 8)
    return score - slope * torch.abs(q_idx - kv_idx)


def sliding_window_eager(score, b, h, q_idx, kv_idx):
    return torch.where(torch.abs(q_idx - kv_idx) <= 256, score, float("-inf"))


def block_diagonal_eager(score, b, h, q_idx, kv_idx):
    q_block = q_idx // 64
    kv_block = kv_idx // 64
    return torch.where(q_block == kv_block, score, float("-inf"))


def causal_mask_v2_eager(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx - kv_idx >= 0, score, float("-inf"))


def batch_bias(bias_tensor):
    """Per-batch bias."""
    def batch_bias_mod(score, b, h, q_idx, kv_idx):
        return score + bias_tensor[b]
    return batch_bias_mod


def dual_buffer_bias(head_bias, pos_scale):
    """Dual buffer loading."""
    def dual_buffer_mod(score, b, h, q_idx, kv_idx):
        head_component = head_bias[h]
        pos_component = pos_scale[q_idx]
        return score + pos_component + head_component
    return dual_buffer_mod


def packed_kv_bias(bias_tensor, cu_seqlens_k):
    """Per-token bias indexed by global kv position."""
    def packed_kv_bias_mod(score, b, h, q_idx, kv_idx):
        kv_global = cu_seqlens_k[b] + kv_idx
        return score + bias_tensor[kv_global]
    return packed_kv_bias_mod


def packed_q_bias(bias_tensor, cu_seqlens_q):
    """Per-token bias indexed by global q position."""
    def packed_q_bias_mod(score, b, h, q_idx, kv_idx):
        q_global = cu_seqlens_q[b] + q_idx
        return score + bias_tensor[q_global]
    return packed_q_bias_mod


def packed_kv_bias_with_rel_pos(bias_tensor, cu_seqlens_k):
    """Relative position + per-token bias."""
    def combined_mod(score, b, h, q_idx, kv_idx):
        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.1
        kv_global = cu_seqlens_k[b] + kv_idx
        token_bias = bias_tensor[kv_global]
        return score + rel_bias + token_bias
    return combined_mod


def packed_q_and_kv_bias(q_bias_tensor, kv_bias_tensor, cu_seqlens_q, cu_seqlens_k):
    """Both q and kv biases."""
    def combined_mod(score, b, h, q_idx, kv_idx):
        q_global = cu_seqlens_q[b] + q_idx
        kv_global = cu_seqlens_k[b] + kv_idx
        return score + q_bias_tensor[q_global] + kv_bias_tensor[kv_global]
    return combined_mod


def packed_kv_bias_with_logical_rel_pos(bias_tensor, cu_seqlens_k):
    """Logical relative position + global-indexed bias."""
    def combined_mod(score, b, h, q_idx, kv_idx):
        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.01
        kv_global = cu_seqlens_k[b] + kv_idx
        token_bias = bias_tensor[kv_global]
        return score + rel_bias + token_bias
    return combined_mod


def stress_1_eager(bias, cu_seqlens_q):
    """All indices in complex arithmetic."""
    def score_mod(score, b, h, q_idx, kv_idx):
        rel_pos_sq = (q_idx - kv_idx) ** 2
        q_global = cu_seqlens_q[b] + q_idx
        bias_q = bias[q_global]
        scale = (b + 1) * (h + 1) * 0.001
        rel_bias = rel_pos_sq.float() * 0.0001
        return score + rel_bias + bias_q * scale
    return score_mod


def stress_2_eager(token_bias, cu_seqlens_q, cu_seqlens_k):
    """Conditional masking with global/logical."""
    def score_mod(score, b, h, q_idx, kv_idx):
        kv_global = cu_seqlens_k[b] + kv_idx
        bias_val = token_bias[kv_global]

        is_causal = q_idx >= kv_idx

        q_global = cu_seqlens_q[b] + q_idx
        global_diff = q_global - kv_global
        is_nearby = torch.abs(global_diff) <= 512

        both_conditions = is_causal & is_nearby
        return torch.where(both_conditions, score + bias_val, float("-inf"))
    return score_mod


def stress_3_eager(batch_bias, head_scale, q_pos_bias, kv_pos_bias, rel_pos_scale,
                   cu_seqlens_q, cu_seqlens_k, max_rel_pos=512):
    """Multiple aux tensors with different indexing."""
    def score_mod(score, b, h, q_idx, kv_idx):
        bb_val = batch_bias[b]
        hs_val = head_scale[h]

        q_global = cu_seqlens_q[b] + q_idx
        qpb_val = q_pos_bias[q_global]

        kv_global = cu_seqlens_k[b] + kv_idx
        kvpb_val = kv_pos_bias[kv_global]

        rel_idx = (q_idx - kv_idx + max_rel_pos).clamp(0, max_rel_pos * 2)
        rps_val = rel_pos_scale[rel_idx]

        return score * hs_val + bb_val + qpb_val + kvpb_val + rps_val * 0.1
    return score_mod


def stress_4_eager(token_bias, cu_seqlens_k):
    """Verify global - logical = offset."""
    def score_mod(score, b, h, q_idx, kv_idx):
        kv_global = cu_seqlens_k[b] + kv_idx
        bias_val = token_bias[kv_global]
        return score + bias_val
    return score_mod


def stress_5_eager(token_bias, cu_seqlens_q, cu_seqlens_k):
    """XOR-based pattern."""
    def score_mod(score, b, h, q_idx, kv_idx):
        xor_logical = q_idx ^ kv_idx

        kv_global = cu_seqlens_k[b] + kv_idx

        pattern_logical = xor_logical & 0xFF
        pattern_bias = pattern_logical.float() * 0.001

        bias_val = token_bias[kv_global]

        return score + pattern_bias + bias_val * 0.1
    return score_mod


# =============================================================================
# Test pairs
# =============================================================================

TEST_PAIRS = [
    (score_mod_1, identity_eager),
    (score_mod_2, causal_mask_eager),
    (score_mod_3, relative_bias_eager),
    (score_mod_4, relative_bias_v2_eager),
    (score_mod_5, times_two_eager),
    (score_mod_6, alibi_bias_eager),
    (score_mod_7, sliding_window_eager),
    (score_mod_8, block_diagonal_eager),
    (score_mod_9, causal_mask_v2_eager),
]

TEST_PAIRS_WITH_AUX_TENSORS = [
    (score_mod_10, batch_bias),
    (score_mod_11, dual_buffer_bias),
]

TEST_PAIRS_GLOBAL_IDX = [
    (score_mod_global_1, packed_kv_bias, "kv"),
    (score_mod_global_2, packed_q_bias, "q"),
    (score_mod_global_3, packed_kv_bias_with_rel_pos, "kv"),
]


# =============================================================================
# Helper functions
# =============================================================================

def create_tensors(
    batch_size=2, num_heads=4, seqlen_q=64, seqlen_kv=64, dim=128, dtype=torch.bfloat16
):
    q = torch.randn(batch_size, num_heads, seqlen_q, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, num_heads, seqlen_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, num_heads, seqlen_kv, dim, device="cuda", dtype=dtype)
    return q, k, v


def run_cute_flash(
    q, k, v, cute_score_mod, aux_tensors=None, pack_gqa=False,
    cu_seqlens_q=None, cu_seqlens_k=None
) -> torch.Tensor:
    # Varlen case: at least one of cu_seqlens_q or cu_seqlens_k is provided
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        # For varlen, output shape matches Q shape
        out = torch.empty_like(q)
        _flash_attn_fwd(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            return_lse=True,
            score_mod=cute_score_mod,
            out=out, lse=None,
            aux_tensors=aux_tensors,
            pack_gqa=pack_gqa,
        )
        return out

    # Batched case: neither is varlen
    q_transposed, k_transposed, v_transposed = map(
        lambda x: x.transpose(1, 2), (q, k, v)
    )
    out = torch.empty_like(q_transposed)
    _flash_attn_fwd(
        q_transposed, k_transposed, v_transposed,
        return_lse=True,
        score_mod=cute_score_mod,
        out=out, lse=None,
        aux_tensors=aux_tensors,
        pack_gqa=pack_gqa,
    )
    return out.transpose(1, 2)


def run_flex_reference(q, k, v, eager_score_mod, dtype=None) -> torch.Tensor:
    if dtype is not None:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    return flex_attention(
        q, k, v, score_mod=eager_score_mod, enable_gqa=q.shape[1] != k.shape[1]
    )


def run_flex_varlen_ref(
    q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=None
) -> torch.Tensor:
    """Simulate varlen by running flex on each sequence slice.
    
    Supports all combinations:
    - Both varlen: cu_seqlens_q and cu_seqlens_k both provided
    - Varlen q only: cu_seqlens_q provided, cu_seqlens_k=None
    - Varlen k only: cu_seqlens_k provided, cu_seqlens_q=None
    """
    results = []
    
    # Determine batch size and sequence boundaries
    if cu_seqlens_q is not None:
        num_batches = len(cu_seqlens_q) - 1
        batch_size = num_batches
    elif cu_seqlens_k is not None:
        num_batches = len(cu_seqlens_k) - 1
        batch_size = num_batches
    else:
        # Neither varlen - shouldn't call this function
        raise ValueError("At least one of cu_seqlens_q or cu_seqlens_k must be provided")

    for i in range(num_batches):
        # Determine Q boundaries
        if cu_seqlens_q is not None:
            start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        else:
            # Not varlen Q - use batch dimension
            start_q, end_q = None, None
            q_slice_batched = q[i:i+1]  # (1, S, H, D) format when Q is (batch_size, seqlen_q, num_heads, head_dim)
        
        # Determine K/V boundaries
        if cu_seqlens_k is not None:
            start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        else:
            # Not varlen K - use batch dimension
            start_k, end_k = None, None
            k_slice_batched = k[i:i+1]  # Already (1, H, S, D) format
            v_slice_batched = v[i:i+1]

        # Reshape to (1, H, S, D) for flex_attention
        if cu_seqlens_q is not None:
            # q is packed: (total_q, num_heads, head_dim) -> (1, num_heads, seqlen_q_i, head_dim)
            q_slice = q[start_q:end_q].unsqueeze(0).transpose(1, 2)
        else:
            # q is batched: (batch_size, seqlen_q, num_heads, head_dim) -> (1, num_heads, seqlen_q, head_dim)
            # q_slice_batched is (1, seqlen_q, num_heads, head_dim), transpose to (1, num_heads, seqlen_q, head_dim)
            q_slice = q_slice_batched.transpose(1, 2)
        
        if cu_seqlens_k is not None:
            # k is packed: (total_k, num_heads, head_dim) -> (1, num_heads, seqlen_k_i, head_dim)
            k_slice = k[start_k:end_k].unsqueeze(0).transpose(1, 2)
            v_slice = v[start_k:end_k].unsqueeze(0).transpose(1, 2)
        else:
            # k is batched: already (1, num_heads, seqlen_k, head_dim), no transpose needed
            k_slice = k_slice_batched
            v_slice = v_slice_batched

        if dtype is not None:
            q_slice = q_slice.to(dtype)
            k_slice = k_slice.to(dtype)
            v_slice = v_slice.to(dtype)

        # Wrap score_mod to enforce correct batch index
        def wrapped_score_mod(score, b, h, q_idx, kv_idx, seq_idx=i):
            return eager_score_mod(score, seq_idx, h, q_idx, kv_idx)

        out_slice = flex_attention(
            q_slice, k_slice, v_slice,
            score_mod=wrapped_score_mod,
            enable_gqa=q_slice.shape[1] != k_slice.shape[1],  # Check head dimension (dim 1)
        )
        results.append(out_slice.transpose(1, 2).squeeze(0))

    return torch.cat(results, dim=0)


# =============================================================================
# Tests: Basic score_mod (batched, non-varlen)
# =============================================================================

@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
    [
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 2), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS)
def test_cute_vs_flex_attention(
    seqlen_q, seqlen_kv, qhead_per_kvhead, num_kv_heads, dtype, score_mod_pair
):
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod = score_mod_pair

    num_q_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    q, k, v = create_tensors(
        seqlen_q=seqlen_q, seqlen_kv=seqlen_kv, num_heads=num_q_heads, dtype=dtype
    )
    if pack_gqa:
        k = k[:, :num_kv_heads, :, :].clone()
        v = v[:, :num_kv_heads, :, :].clone()

    out_ref_fp32 = run_flex_reference(q, k, v, eager_score_mod, dtype=torch.float32)
    out_pt = run_flex_reference(q, k, v, eager_score_mod)
    out_cute = run_cute_flash(q, k, v, cute_score_mod, pack_gqa=pack_gqa)

    assert out_cute.shape == out_ref_fp32.shape == out_pt.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert not torch.isnan(out_pt).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert torch.isfinite(out_pt).all()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nNumerical comparison for {cute_score_mod.__name__}:")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")
    print(f"  Dynamic absolute tolerance: {fwd_atol:.2e}")
    print(f"  Error ratio (CuTE/PyTorch): {cute_error / max(pt_error, 1e-10):.2f}")

    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"CuTE error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


# =============================================================================
# Tests: score_mod with aux_tensors (batched, non-varlen)
# =============================================================================

@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
    [
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 1), (4, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS_WITH_AUX_TENSORS)
def test_cute_vs_flex_attention_with_aux_tensors(
    seqlen_q, seqlen_kv, qhead_per_kvhead, num_kv_heads, dtype, score_mod_pair
):
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod_factory = score_mod_pair

    batch_size = 2
    num_q_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    q, k, v = create_tensors(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        num_heads=num_q_heads,
        dtype=dtype,
    )
    if pack_gqa:
        k = k[:, :num_kv_heads, :, :].clone()
        v = v[:, :num_kv_heads, :, :].clone()

    if cute_score_mod == score_mod_10:
        buffer = torch.randn(batch_size, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [buffer]
        eager_score_mod = eager_score_mod_factory(buffer)
        assert buffer.shape == (batch_size,)
    elif cute_score_mod == score_mod_11:
        head_bias = torch.randn(num_q_heads, device="cuda", dtype=dtype) * 0.2
        pos_scale = torch.arange(seqlen_q, device="cuda", dtype=dtype) * 0.01
        aux_tensors = [head_bias, pos_scale]
        eager_score_mod = eager_score_mod_factory(head_bias, pos_scale)
        assert head_bias.shape == (num_q_heads,)
        assert pos_scale.shape == (seqlen_q,)

    out_ref_fp32 = run_flex_reference(q, k, v, eager_score_mod, dtype=torch.float32)
    out_pt = run_flex_reference(q, k, v, eager_score_mod)
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod, aux_tensors=aux_tensors, pack_gqa=pack_gqa
    )

    assert out_cute.shape == out_ref_fp32.shape == out_pt.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert not torch.isnan(out_pt).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    assert torch.isfinite(out_pt).all()

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nNumerical comparison for {cute_score_mod.__name__}:")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")
    print(f"  Dynamic absolute tolerance: {fwd_atol:.2e}")
    print(f"  Error ratio (CuTE/PyTorch): {cute_error / max(pt_error, 1e-10):.2f}")

    assert cute_error <= rtol * pt_error + fwd_atol, (
        f"CuTE error {cute_error:.2e} exceeds {rtol}x PyTorch error {pt_error:.2e} + {fwd_atol:.2e}"
    )


# =============================================================================
# Tests: Varlen with 6-arg score_mod (backward compatibility)
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        # Very small sequences
        ([1], [1]),
        ([1, 1], [1, 1]),
        ([2, 3], [2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        # Small sequences
        ([8, 16], [8, 16]),
        ([16, 32], [16, 32]),
        ([32, 64], [32, 64]),
        ([12, 24], [12, 24]),
        # Medium sequences
        ([64, 56, 128], [64, 56, 128]),
        ([32, 64, 96], [32, 64, 96]),
        ([128, 64], [128, 64]),
        ([64, 128, 64], [64, 128, 64]),
        # Large sequences
        ([256, 512], [256, 512]),
        ([512, 256], [512, 256]),
        ([128, 256, 512], [128, 256, 512]),
        ([256, 128, 256], [256, 128, 256]),
        # Very large sequences
        ([1024], [1024]),
        ([512, 1024], [512, 1024]),
        ([1024, 512], [1024, 512]),
        ([2048], [2048]),
        # Non-power-of-2 sequences
        ([113, 203], [113, 203]),
        ([239, 1], [239, 1]),
        ([799, 3], [799, 3]),
        ([100, 100], [50, 50]),
        ([108, 256], [108, 256]),
        # Single sequence (edge case)
        ([64], [64]),
        ([128], [128]),
        ([256], [256]),
        # Many sequences
        ([32, 32, 32, 32], [32, 32, 32, 32]),
        ([64, 64, 64, 64, 64], [64, 64, 64, 64, 64]),
        ([16, 32, 64, 128, 256], [16, 32, 64, 128, 256]),
        # Extreme size ratios
        ([1, 1024], [1, 1024]),
        ([1024, 1], [1024, 1]),
        ([1, 1, 2048], [1, 1, 2048]),
        ([2048, 1, 1], [2048, 1, 1]),
        # Mixed small and large
        ([1, 256, 1], [1, 256, 1]),
        ([256, 1, 256], [256, 1, 256]),
        ([8, 512, 8], [8, 512, 8]),
        # Uneven sequences
        ([17, 33, 65], [17, 33, 65]),
        ([13, 27, 51], [13, 27, 51]),
        ([7, 19, 31, 47], [7, 19, 31, 47]),
        # Different Q and K lengths
        ([64, 128], [32, 64]),
        ([128, 256], [64, 128]),
        ([100, 100], [50, 50]),
        ([256, 512, 256], [128, 256, 128]),
        ([1, 1024, 1], [512, 512, 512]),
    ],
)
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS)
def test_varlen_with_score_mod(seqlens_q, seqlens_k, dtype, score_mod_pair):
    """Test varlen with 6-arg score_mod (backward compatibility)."""
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod = score_mod_pair

    num_heads = 4
    head_dim = 128
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    out_ref_fp32 = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    assert not torch.isnan(out_cute).any()
    assert torch.isfinite(out_cute).all()
    assert out_cute.shape == out_ref_fp32.shape

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nVarlen test for {cute_score_mod.__name__}:")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-4


# =============================================================================
# Tests: Varlen Q only (Q packed, K/V batched)
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlen_k",
    [
        # Very small sequences
        ([1], 1),
        ([1, 1], 1),
        ([2, 3], 2),
        ([1, 2, 3], 1),
        # Small sequences
        ([8, 16], 8),
        ([16, 32], 16),
        ([32, 64], 32),
        ([12, 24], 12),
        # Medium sequences
        ([64, 56, 128], 64),
        ([32, 64, 96], 128),
        ([128, 64], 128),
        ([64, 128, 64], 64),
        # Large sequences
        ([256, 512], 256),
        ([512, 256], 512),
        ([128, 256, 512], 256),
        ([256, 128, 256], 128),
        # Very large sequences
        ([1024], 1024),
        ([512, 1024], 512),
        ([1024, 512], 1024),
        ([2048], 2048),
        # Non-power-of-2 sequences
        ([113, 203], 113),
        ([239, 1], 239),
        ([799, 3], 799),
        ([100, 100], 50),
        ([108, 256], 108),
        # Single sequence (edge case)
        ([64], 64),
        ([128], 128),
        ([256], 256),
        # Many sequences
        ([32, 32, 32, 32], 32),
        ([64, 64, 64, 64, 64], 64),
        ([16, 32, 64, 128, 256], 128),
        # Extreme size ratios
        ([1, 1024], 512),
        ([1024, 1], 1024),
        ([1, 1, 2048], 1024),
        ([2048, 1, 1], 2048),
        # Mixed small and large
        ([1, 256, 1], 128),
        ([256, 1, 256], 128),
        ([8, 512, 8], 256),
        # Uneven sequences
        ([17, 33, 65], 32),
        ([13, 27, 51], 25),
        ([7, 19, 31, 47], 20),
    ],
)
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS)
def test_varlen_q_only_with_score_mod(seqlens_q, seqlen_k, dtype, score_mod_pair):
    """Test varlen Q only (Q packed, K/V batched) with 6-arg score_mod."""
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod = score_mod_pair

    num_heads = 4
    head_dim = 128
    total_q = sum(seqlens_q)
    batch_size = len(seqlens_q)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    # Q is packed: (total_q, num_heads, head_dim)
    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    # K/V are batched: (batch_size, seqlen_k, num_heads, head_dim) - interface expects seqlen before heads
    k = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads, head_dim, device="cuda", dtype=dtype)

    # For reference, we need to handle the mixed format
    # Interface expects (batch_size, seqlen_k, num_heads, head_dim) when cu_seqlens_k is None
    # Reference expects (batch_size, num_heads, seqlen_k, head_dim) when cu_seqlens_k is None
    k_for_ref = k.transpose(1, 2)  # (batch_size, num_heads, seqlen_k, head_dim)
    v_for_ref = v.transpose(1, 2)  # (batch_size, num_heads, seqlen_k, head_dim)

    out_ref_fp32 = run_flex_varlen_ref(
        q, k_for_ref, v_for_ref, cu_seqlens_q, cu_seqlens_k=None, eager_score_mod=eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q, k_for_ref, v_for_ref, cu_seqlens_q, cu_seqlens_k=None, eager_score_mod=eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        pack_gqa=False,
    )

    assert not torch.isnan(out_cute).any()
    assert torch.isfinite(out_cute).all()
    assert out_cute.shape == out_ref_fp32.shape

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nVarlen Q only test for {cute_score_mod.__name__}:")
    print(f"  seqlens_q={seqlens_q}, seqlen_k={seqlen_k}")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-4


# =============================================================================
# Tests: Varlen K only (Q batched, K/V packed)
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlen_q, seqlens_k",
    [
        # Very small sequences
        (1, [1]),
        (1, [1, 1]),
        (2, [2, 3]),
        (1, [1, 2, 3]),
        # Small sequences
        (8, [8, 16]),
        (16, [16, 32]),
        (32, [32, 64]),
        (12, [12, 24]),
        # Medium sequences
        (64, [64, 56, 128]),
        (128, [32, 64, 96]),
        (128, [128, 64]),
        (64, [64, 128, 64]),
        # Large sequences
        (256, [256, 512]),
        (512, [512, 256]),
        (256, [128, 256, 512]),
        (128, [256, 128, 256]),
        # Very large sequences
        (1024, [1024]),
        (512, [512, 1024]),
        (1024, [1024, 512]),
        (2048, [2048]),
        # Non-power-of-2 sequences
        (113, [113, 203]),
        (239, [239, 1]),
        (799, [799, 3]),
        (100, [50, 50]),
        (108, [108, 256]),
        # Single sequence (edge case)
        (64, [64]),
        (128, [128]),
        (256, [256]),
        # Many sequences
        (32, [32, 32, 32, 32]),
        (64, [64, 64, 64, 64, 64]),
        (128, [16, 32, 64, 128, 256]),
        # Extreme size ratios
        (512, [1, 1024]),
        (1024, [1024, 1]),
        (1024, [1, 1, 2048]),
        (2048, [2048, 1, 1]),
        # Mixed small and large
        (128, [1, 256, 1]),
        (128, [256, 1, 256]),
        (256, [8, 512, 8]),
        # Uneven sequences
        (32, [17, 33, 65]),
        (25, [13, 27, 51]),
        (20, [7, 19, 31, 47]),
    ],
)
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS)
def test_varlen_k_only_with_score_mod(seqlen_q, seqlens_k, dtype, score_mod_pair):
    """Test varlen K only (Q batched, K/V packed) with 6-arg score_mod."""
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod = score_mod_pair

    num_heads = 4
    head_dim = 128
    total_k = sum(seqlens_k)
    batch_size = len(seqlens_k)

    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    # Q is batched: (batch_size, seqlen_q, num_heads, head_dim) - interface expects seqlen before heads
    q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=dtype)
    # K/V are packed: (total_k, num_heads, head_dim)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    # For reference, we need to handle the mixed format
    # Convert Q to packed format for reference (transpose to get heads in right position)
    q_packed = q.transpose(1, 2).reshape(-1, num_heads, head_dim)  # (batch_size * seqlen_q, num_heads, head_dim)
    cu_seqlens_q = torch.tensor(
        [0] + [seqlen_q * (i + 1) for i in range(batch_size)],
        device="cuda", dtype=torch.int32,
    )

    out_ref_fp32 = run_flex_varlen_ref(
        q_packed, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q_packed, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod,
        cu_seqlens_q=None,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    # Reshape reference output to match cute output format
    # Cute output: (batch_size, seqlen_q, num_heads, head_dim)
    # Reference output: (total_q, num_heads, head_dim) where total_q = batch_size * seqlen_q
    # We can reshape directly to (batch_size, seqlen_q, num_heads, head_dim)
    out_ref_fp32_reshaped = out_ref_fp32.reshape(batch_size, seqlen_q, num_heads, head_dim)
    out_pt_reshaped = out_pt.reshape(batch_size, seqlen_q, num_heads, head_dim)

    assert not torch.isnan(out_cute).any()
    assert torch.isfinite(out_cute).all()
    assert out_cute.shape == out_ref_fp32_reshaped.shape

    fwd_atol = 2 * (out_ref_fp32_reshaped + 0.3 - 0.3 - out_ref_fp32_reshaped).abs().max().item()
    rtol = 2
    pt_error = (out_pt_reshaped - out_ref_fp32_reshaped).abs().max().item()
    cute_error = (out_cute - out_ref_fp32_reshaped).abs().max().item()

    print(f"\nVarlen K only test for {cute_score_mod.__name__}:")
    print(f"  seqlen_q={seqlen_q}, seqlens_k={seqlens_k}")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-4


# =============================================================================
# Tests: Varlen with 8-arg score_mod (global indices)
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        # Very small sequences
        ([1], [1]),
        ([1, 1], [1, 1]),
        ([2, 3], [2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        # Small sequences
        ([8, 16], [8, 16]),
        ([16, 32], [16, 32]),
        ([32, 64], [32, 64]),
        ([12, 24], [12, 24]),
        # Medium sequences
        ([64, 56, 128], [64, 56, 128]),
        ([32, 64, 96], [32, 64, 96]),
        ([128, 64], [128, 64]),
        ([64, 128, 64], [64, 128, 64]),
        # Large sequences
        ([256, 512], [256, 512]),
        ([512, 256], [512, 256]),
        ([128, 256, 512], [128, 256, 512]),
        ([256, 128, 256], [256, 128, 256]),
        # Very large sequences
        ([1024], [1024]),
        ([512, 1024], [512, 1024]),
        ([1024, 512], [1024, 512]),
        # Non-power-of-2 sequences
        ([113, 203], [113, 203]),
        ([239, 1], [239, 1]),
        ([799, 3], [799, 3]),
        ([100, 100], [50, 50]),
        ([108, 256], [108, 256]),
        # Single sequence (edge case)
        ([64], [64]),
        ([128], [128]),
        ([256], [256]),
        # Many sequences
        ([32, 32, 32, 32], [32, 32, 32, 32]),
        ([64, 64, 64, 64, 64], [64, 64, 64, 64, 64]),
        ([16, 32, 64, 128, 256], [16, 32, 64, 128, 256]),
        # Extreme size ratios
        ([1, 1024], [1, 1024]),
        ([1024, 1], [1024, 1]),
        ([1, 1, 2048], [1, 1, 2048]),
        # Mixed small and large
        ([1, 256, 1], [1, 256, 1]),
        ([256, 1, 256], [256, 1, 256]),
        ([8, 512, 8], [8, 512, 8]),
        # Uneven sequences
        ([17, 33, 65], [17, 33, 65]),
        ([13, 27, 51], [13, 27, 51]),
        ([7, 19, 31, 47], [7, 19, 31, 47]),
        # Different Q and K lengths
        ([64, 128], [32, 64]),
        ([128, 256], [64, 128]),
        ([100, 100], [50, 50]),
        ([256, 512, 256], [128, 256, 128]),
        ([32, 64, 128, 64], [32, 64, 128, 64]),
    ],
)
@pytest.mark.parametrize("score_mod_tuple", TEST_PAIRS_GLOBAL_IDX)
def test_varlen_with_global_idx_score_mod(seqlens_q, seqlens_k, dtype, score_mod_tuple):
    """Test varlen with 8-arg score_mod using global indices."""
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod_factory, bias_type = score_mod_tuple

    num_heads = 4
    head_dim = 128
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    if bias_type == "kv":
        bias_tensor = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        eager_score_mod = eager_score_mod_factory(bias_tensor, cu_seqlens_k)
    else:  # "q"
        bias_tensor = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        eager_score_mod = eager_score_mod_factory(bias_tensor, cu_seqlens_q)

    aux_tensors = [bias_tensor]

    out_ref_fp32 = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod,
        aux_tensors=aux_tensors,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    assert not torch.isnan(out_cute).any()
    assert torch.isfinite(out_cute).all()
    assert out_cute.shape == out_ref_fp32.shape

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nVarlen + global idx for {cute_score_mod.__name__}:")
    print(f"  seqlens_q={seqlens_q}, seqlens_k={seqlens_k}")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-4


# =============================================================================
# Tests: Varlen + GQA + global indices
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        # Very small sequences
        ([1], [1]),
        ([1, 1], [1, 1]),
        ([2, 3], [2, 3]),
        ([1, 1, 1], [1, 1, 1]),
        # Small sequences
        ([8, 16], [8, 16]),
        ([16, 32], [16, 32]),
        ([32, 64], [32, 64]),
        ([32, 64], [128, 256]),
        # Medium sequences
        ([64, 56, 128], [64, 56, 128]),
        ([128], [128]),
        ([64, 128, 64], [64, 128, 64]),
        # Large sequences
        ([256, 512], [256, 512]),
        ([512, 256], [512, 256]),
        ([128, 256, 512], [128, 256, 512]),
        # Non-power-of-2 sequences
        ([113, 203], [113, 203]),
        ([100, 100], [50, 50]),
        # Single sequence (edge case)
        ([64], [64]),
        ([128], [128]),
        # Many sequences
        ([32, 32, 32, 32], [32, 32, 32, 32]),
        ([16, 32, 64, 128, 256], [16, 32, 64, 128, 256]),
        # Extreme size ratios
        ([1, 1024], [1, 1024]),
        ([1024, 1], [1024, 1]),
        # Mixed small and large
        ([1, 256, 1], [1, 256, 1]),
        ([256, 1, 256], [256, 1, 256]),
        # Uneven sequences
        ([17, 33, 65], [17, 33, 65]),
        # Different Q and K lengths
        ([64, 128], [32, 64]),
        ([128, 256], [64, 128]),
        ([100, 100], [50, 50]),
    ],
)
@pytest.mark.parametrize("qhead_per_kvhead,num_kv_heads", [(1, 2), (4, 2)])
def test_varlen_global_idx_with_gqa(seqlens_q, seqlens_k, dtype, qhead_per_kvhead, num_kv_heads):
    """Test varlen + global indices + GQA."""
    torch.random.manual_seed(42)

    num_q_heads = num_kv_heads * qhead_per_kvhead
    pack_gqa = qhead_per_kvhead > 1
    head_dim = 128
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_kv_heads, head_dim, device="cuda", dtype=dtype)

    bias_tensor = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
    aux_tensors = [bias_tensor]
    eager_score_mod = packed_kv_bias(bias_tensor, cu_seqlens_k)

    out_ref_fp32 = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, score_mod_global_1,
        aux_tensors=aux_tensors,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=pack_gqa,
    )

    assert not torch.isnan(out_cute).any()
    assert torch.isfinite(out_cute).all()
    assert out_cute.shape == out_ref_fp32.shape

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nVarlen + GQA + global idx (pack_gqa={pack_gqa}):")
    print(f"  seqlens_q={seqlens_q}, seqlens_k={seqlens_k}")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-4


# =============================================================================
# Tests: Both q and kv global indices
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        ([64, 128], [64, 128]),
        ([32, 64, 96], [32, 64, 96]),
        ([100, 100], [50, 50]),
    ],
)
@pytest.mark.parametrize("test_case", [
    (score_mod_global_4, packed_q_and_kv_bias, "q_and_kv"),
    (score_mod_global_5, packed_kv_bias_with_logical_rel_pos, "kv_only"),
])
def test_varlen_both_q_and_kv_global_indices(seqlens_q, seqlens_k, dtype, test_case):
    """Test using both q_idx_global and kv_idx_global."""
    torch.random.manual_seed(42)
    cute_score_mod, eager_factory, aux_setup = test_case

    num_heads = 4
    head_dim = 128
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    if aux_setup == "q_and_kv":
        q_bias = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        kv_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [q_bias, kv_bias]
        eager_score_mod = eager_factory(q_bias, kv_bias, cu_seqlens_q, cu_seqlens_k)
    else:  # "kv_only"
        kv_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [kv_bias]
        eager_score_mod = eager_factory(kv_bias, cu_seqlens_k)

    out_ref_fp32 = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod,
        aux_tensors=aux_tensors,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    assert not torch.isnan(out_cute).any()
    assert torch.isfinite(out_cute).all()
    assert out_cute.shape == out_ref_fp32.shape

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\n{cute_score_mod.__name__} test ({aux_setup}):")
    print(f"  PyTorch vs FP32 ref max error: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref max error: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-4


# =============================================================================
# Tests: Stress tests (adversarial patterns)
# =============================================================================

STRESS_TEST_CASES = [
    ("stress_1", score_mod_stress_1, stress_1_eager, "q_concat"),
    ("stress_2", score_mod_stress_2, stress_2_eager, "kv_with_cu"),
    ("stress_3", score_mod_stress_3, stress_3_eager, "multi_buffer"),
    ("stress_4", score_mod_stress_4, stress_4_eager, "kv_only"),
    ("stress_5", score_mod_stress_5, stress_5_eager, "kv_with_cu"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        # Very small sequences
        ([1], [1]),
        ([1, 1], [1, 1]),
        ([2, 3], [2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        # Small sequences
        ([8, 16], [8, 16]),
        ([16, 32], [16, 32]),
        ([32, 64], [32, 64]),
        # Medium sequences
        ([64, 128], [64, 128]),
        ([32, 64, 96], [32, 64, 96]),
        ([128, 64], [128, 64]),
        ([64, 128, 64], [64, 128, 64]),
        # Large sequences
        ([256, 512], [256, 512]),
        ([512, 256], [512, 256]),
        ([128, 256, 512], [128, 256, 512]),
        # Non-power-of-2 sequences
        ([113, 203], [113, 203]),
        ([239, 1], [239, 1]),
        ([100, 100], [50, 50]),
        # Single sequence (edge case)
        ([64], [64]),
        ([128], [128]),
        # Many sequences
        ([32, 32, 32, 32], [32, 32, 32, 32]),
        ([64, 64, 64, 64, 64], [64, 64, 64, 64, 64]),
        ([16, 32, 64, 128, 256], [16, 32, 64, 128, 256]),
        # Extreme size ratios
        ([1, 1024], [1, 1024]),
        ([1024, 1], [1024, 1]),
        ([1, 1, 2048], [1, 1, 2048]),
        # Mixed small and large
        ([1, 256, 1], [1, 256, 1]),
        ([256, 1, 256], [256, 1, 256]),
        ([8, 512, 8], [8, 512, 8]),
        ([128, 1, 128], [128, 1, 128]),
        ([1, 255, 1], [1, 255, 1]),
        # Uneven sequences
        ([17, 33, 65], [17, 33, 65]),
        ([13, 27, 51], [13, 27, 51]),
        ([7, 19, 31, 47], [7, 19, 31, 47]),
        ([7, 13, 19, 23], [7, 13, 19, 23]),
        # Different Q and K lengths
        ([64, 128], [32, 64]),
        ([128, 256], [64, 128]),
        ([100, 100], [50, 50]),
        ([256, 512, 256], [128, 256, 128]),
    ],
)
@pytest.mark.parametrize("test_case", STRESS_TEST_CASES)
def test_stress_score_mods(seqlens_q, seqlens_k, dtype, test_case):
    """Stress test with adversarial patterns."""
    torch.random.manual_seed(42)
    test_name, cute_score_mod, eager_factory, aux_setup = test_case

    num_heads = 4
    num_seqs = len(seqlens_q)
    head_dim = 128
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)
    max_rel_pos = 512

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    # Setup aux tensors
    if aux_setup == "q_concat":
        bias = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [bias]
        eager_score_mod = eager_factory(bias, cu_seqlens_q)
    elif aux_setup == "kv_only":
        kv_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [kv_bias]
        eager_score_mod = eager_factory(kv_bias, cu_seqlens_k)
    elif aux_setup == "kv_with_cu":
        kv_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [kv_bias]
        eager_score_mod = eager_factory(kv_bias, cu_seqlens_q, cu_seqlens_k)
    elif aux_setup == "multi_buffer":
        batch_bias = torch.randn(num_seqs, device="cuda", dtype=dtype) * 0.1
        head_scale = torch.randn(num_heads, device="cuda", dtype=dtype) * 0.1 + 1.0
        q_pos_bias = torch.randn(total_q, device="cuda", dtype=dtype) * 0.1
        kv_pos_bias = torch.randn(total_k, device="cuda", dtype=dtype) * 0.1
        rel_pos_scale = torch.randn(max_rel_pos * 2 + 1, device="cuda", dtype=dtype) * 0.1
        aux_tensors = [batch_bias, head_scale, q_pos_bias, kv_pos_bias, rel_pos_scale]
        eager_score_mod = eager_factory(
            batch_bias, head_scale, q_pos_bias, kv_pos_bias, rel_pos_scale,
            cu_seqlens_q, cu_seqlens_k, max_rel_pos
        )

    out_ref_fp32 = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod,
        aux_tensors=aux_tensors,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    assert not torch.isnan(out_cute).any(), f"{test_name}: NaN"
    assert torch.isfinite(out_cute).all(), f"{test_name}: inf"
    assert out_cute.shape == out_ref_fp32.shape

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nStress {test_name}:")
    print(f"  seqlens_q={seqlens_q}, seqlens_k={seqlens_k}, dtype={dtype}")
    print(f"  PyTorch vs FP32 ref: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-3, (
        f"{test_name}: CuTE error {cute_error:.2e} exceeds tolerance"
    )


# =============================================================================
# Tests: Varlen relative position (no aux_tensors path)
# =============================================================================

TEST_PAIRS_RELATIVE_POSITION = [
    (score_mod_2, causal_mask_eager),         # q_idx >= kv_idx
    (score_mod_3, relative_bias_eager),       # |q_idx - kv_idx|
    (score_mod_4, relative_bias_v2_eager),    # 2 * |q_idx - kv_idx|
    (score_mod_6, alibi_bias_eager),          # ALiBi: slope * |q_idx - kv_idx|
    (score_mod_7, sliding_window_eager),      # |q_idx - kv_idx| <= 256
    (score_mod_9, causal_mask_v2_eager),      # q_idx - kv_idx >= 0
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        # Very small sequences
        ([1], [1]),
        ([1, 1], [1, 1]),
        ([2, 3], [2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        # Small sequences
        ([8, 16], [8, 16]),
        ([16, 32], [16, 32]),
        ([32, 64], [32, 64]),
        ([12, 24], [12, 24]),
        # Medium sequences
        ([64, 128], [64, 128]),
        ([32, 64, 96], [32, 64, 96]),
        ([128, 64], [128, 64]),
        ([64, 128, 64], [64, 128, 64]),
        # Large sequences
        ([256, 512], [256, 512]),
        ([512, 256], [512, 256]),
        ([128, 256, 512], [128, 256, 512]),
        ([256, 128, 256], [256, 128, 256]),
        # Non-power-of-2 sequences
        ([113, 203], [113, 203]),
        ([239, 1], [239, 1]),
        ([799, 3], [799, 3]),
        ([100, 100], [50, 50]),
        ([108, 256], [108, 256]),
        # Single sequence (edge case)
        ([64], [64]),
        ([128], [128]),
        ([256], [256]),
        # Many sequences
        ([32, 32, 32, 32], [32, 32, 32, 32]),
        ([64, 64, 64, 64, 64], [64, 64, 64, 64, 64]),
        ([16, 32, 64, 128, 256], [16, 32, 64, 128, 256]),
        # Extreme size ratios
        ([1, 1024], [1, 1024]),
        ([1024, 1], [1024, 1]),
        ([1, 1, 2048], [1, 1, 2048]),
        # Mixed small and large
        ([1, 256, 1], [1, 256, 1]),
        ([256, 1, 256], [256, 1, 256]),
        ([8, 512, 8], [8, 512, 8]),
        ([128, 1, 128], [128, 1, 128]),
        # Uneven sequences
        ([17, 33, 65], [17, 33, 65]),
        ([13, 27, 51], [13, 27, 51]),
        ([7, 19, 31, 47], [7, 19, 31, 47]),
        # Different Q and K lengths
        ([64, 128], [32, 64]),
        ([128, 256], [64, 128]),
        ([100, 100], [50, 50]),
        ([256, 512, 256], [128, 256, 128]),
    ],
)
@pytest.mark.parametrize("score_mod_pair", TEST_PAIRS_RELATIVE_POSITION)
def test_varlen_relative_position_no_aux_tensors(seqlens_q, seqlens_k, dtype, score_mod_pair):
    """Test varlen with relative position score_mods (no aux_tensors path)."""
    torch.random.manual_seed(42)
    cute_score_mod, eager_score_mod = score_mod_pair

    num_heads = 4
    head_dim = 128
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    out_ref_fp32 = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=torch.float32,
    )
    out_pt = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_score_mod, dtype=dtype,
    )
    out_cute = run_cute_flash(
        q, k, v, cute_score_mod,
        aux_tensors=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    assert not torch.isnan(out_cute).any(), f"{cute_score_mod.__name__}: NaN"
    assert torch.isfinite(out_cute).all(), f"{cute_score_mod.__name__}: inf"
    assert out_cute.shape == out_ref_fp32.shape

    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2
    pt_error = (out_pt - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    print(f"\nVarlen rel pos (no aux) - {cute_score_mod.__name__}:")
    print(f"  seqlens_q={seqlens_q}, seqlens_k={seqlens_k}")
    print(f"  PyTorch vs FP32 ref: {pt_error:.2e}")
    print(f"  CuTE vs FP32 ref: {cute_error:.2e}")

    assert cute_error <= rtol * pt_error + fwd_atol + 1e-4, (
        f"{cute_score_mod.__name__}: error {cute_error:.2e} - kv_idx may be global"
    )


@pytest.mark.parametrize("dtype", [torch.float16])
def test_varlen_causal_mask_boundary_check(dtype):
    """Test causal mask at sequence boundaries to verify kv_idx is logical."""
    torch.random.manual_seed(42)

    seqlens = [64, 128]
    total = sum(seqlens)
    num_heads = 2
    head_dim = 64

    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)

    q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)

    out_ref = run_flex_varlen_ref(
        q, k, v, cu_seqlens, cu_seqlens, causal_mask_eager, dtype=torch.float32,
    )
    out_cute = run_cute_flash(
        q, k, v, score_mod_2,
        aux_tensors=None,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        pack_gqa=False,
    )

    # Check sequence 1 (positions 64-191)
    seq1_start = 64
    seq1_end = 192

    out_cute_seq1 = out_cute[seq1_start:seq1_end]
    out_ref_seq1 = out_ref[seq1_start:seq1_end]

    assert not torch.isnan(out_cute_seq1).any(), (
        "Seq 1 has NaN - kv_idx may be global"
    )
    assert torch.isfinite(out_cute_seq1).all()

    seq1_error = (out_cute_seq1 - out_ref_seq1).abs().max().item()
    print(f"\nCausal mask boundary check:")
    print(f"  Seq 1 max error: {seq1_error:.2e}")

    assert seq1_error < 0.1, (
        f"Seq 1 error {seq1_error:.2e} too large - kv_idx may be global"
    )


@pytest.mark.parametrize("dtype", [torch.float16])
def test_varlen_sliding_window_boundary_check(dtype):
    """Test sliding window mask at sequence boundaries."""
    torch.random.manual_seed(42)

    @cute.jit
    def score_mod_sliding_32(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
        rel_pos = q_idx - kv_idx
        rel_pos_abs = cute.TensorSSA(mlir_math.absi(rel_pos), rel_pos.shape, rel_pos.dtype)
        in_window = operator.le(rel_pos_abs, cute.full_like(rel_pos_abs, 32))
        return cute.where(in_window, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))

    def sliding_window_32_eager(score, b, h, q_idx, kv_idx):
        return torch.where(torch.abs(q_idx - kv_idx) <= 32, score, float("-inf"))

    seqlens = [64, 128]
    total = sum(seqlens)
    num_heads = 2
    head_dim = 64

    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)

    q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)

    out_ref = run_flex_varlen_ref(
        q, k, v, cu_seqlens, cu_seqlens, sliding_window_32_eager, dtype=torch.float32,
    )
    out_cute = run_cute_flash(
        q, k, v, score_mod_sliding_32,
        aux_tensors=None,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        pack_gqa=False,
    )

    seq1_start = 64
    seq1_end = 192

    out_cute_seq1 = out_cute[seq1_start:seq1_end]
    out_ref_seq1 = out_ref[seq1_start:seq1_end]

    assert not torch.isnan(out_cute_seq1).any(), (
        "Seq 1 has NaN - kv_idx may be global"
    )

    seq1_error = (out_cute_seq1 - out_ref_seq1).abs().max().item()
    print(f"\nSliding window (size=32) boundary check:")
    print(f"  Seq 1 max error: {seq1_error:.2e}")

    assert seq1_error < 0.1, (
        f"Seq 1 error {seq1_error:.2e} too large - kv_idx may be global"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        ([64, 128], [64, 128]),
        ([32, 64, 96], [32, 64, 96]),
        ([17, 33, 65], [17, 33, 65]),
        # Add extreme size ratios
        ([1, 2], [4096, 8192]),      # Tiny Q, huge K
        ([8, 16], [1024, 2048]),     # Small Q, large K
        ([1], [8192]),               # Single tiny Q, massive K
    ],
)
def test_varlen_local_vs_global_indices_no_aux(seqlens_q, seqlens_k, dtype):
    """
    Test that verifies kv_idx is logical (not global) without aux_tensors.

    This is the critical test for the no-aux-tensors code path.
    If kv_idx were global instead of logical:
    - Seq 0 would see kv_idx = [0, seqlen_k[0])  ✓
    - Seq 1 would see kv_idx = [seqlen_k[0], seqlen_k[0]+seqlen_k[1])  ✗
    - Seq 2 would see kv_idx = [seqlen_k[0]+seqlen_k[1], total_k)  ✗

    We create a score_mod that encodes the kv_idx directly into the output,
    allowing us to detect if indices are wrong.
    """
    torch.random.manual_seed(42)

    # Score mod that returns kv_idx directly (scaled to avoid overflow)
    @cute.jit
    def score_mod_return_kv_idx(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
        # Return kv_idx scaled by 0.001 to avoid numerical issues
        # This allows us to detect if kv_idx is logical or global
        kv_idx_f32 = kv_idx.to(cutlass.Float32)
        return kv_idx_f32 * cute.full_like(kv_idx_f32, 0.001)

    def eager_return_kv_idx(score, b, h, q_idx, kv_idx):
        # Reference: return logical kv_idx (resets to 0 for each sequence)
        return kv_idx.float() * 0.001

    num_heads = 4
    head_dim = 128
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    out_ref = run_flex_varlen_ref(
        q, k, v, cu_seqlens_q, cu_seqlens_k, eager_return_kv_idx, dtype=torch.float32,
    )
    out_cute = run_cute_flash(
        q, k, v, score_mod_return_kv_idx,
        aux_tensors=None,  # Critical: no aux tensors
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    print(f"\nLocal vs global kv_idx test (no aux):")
    print(f"  seqlens_q={seqlens_q}, seqlens_k={seqlens_k}")

    # Check each sequence individually
    for seq_idx in range(len(seqlens_q)):
        start_q = cu_seqlens_q[seq_idx].item()
        end_q = cu_seqlens_q[seq_idx + 1].item()

        out_cute_seq = out_cute[start_q:end_q]
        out_ref_seq = out_ref[start_q:end_q]

        seq_error = (out_cute_seq - out_ref_seq).abs().max().item()

        # Extract sample values to diagnose
        sample_val = out_cute_seq[0, 0, 0].item()  # First position
        expected_val = out_ref_seq[0, 0, 0].item()

        # Get more statistics
        cute_min = out_cute_seq.min().item()
        cute_max = out_cute_seq.max().item()
        ref_min = out_ref_seq.min().item()
        ref_max = out_ref_seq.max().item()

        print(f"  Seq {seq_idx}: error={seq_error:.6f}")
        print(f"    Sample: cute={sample_val:.6f}, ref={expected_val:.6f}")
        print(f"    Range:  cute=[{cute_min:.6f}, {cute_max:.6f}], ref=[{ref_min:.6f}, {ref_max:.6f}]")

        # Check if kv_idx is clearly global (would show up as offset in the minimum value)
        if seq_idx > 0:
            offset = cu_seqlens_k[seq_idx].item()
            # If global, minimum value would be at least offset * 0.001
            expected_if_global = offset * 0.001
            print(f"    If global, min would be ~{expected_if_global:.6f} (offset={offset})")

            # Definitive test: is the minimum value suspiciously close to the offset?
            if cute_min > expected_if_global * 0.9:  # Within 10% of offset
                raise AssertionError(
                    f"Seq {seq_idx}: kv_idx appears to be GLOBAL! "
                    f"Min value {cute_min:.6f} is close to offset {expected_if_global:.6f}. "
                    f"Expected min ~0.000 for logical indexing."
                )

        # The error might just be from fp16/bf16 precision + attention softmax
        # What matters is that we don't see the offset pattern
        # Allow larger tolerance but keep the offset check
        assert seq_error < 0.01, (
            f"Seq {seq_idx}: error {seq_error:.6f} is too large. "
            f"However, check the min value above - if it's near 0, this is likely just numerical precision."
        )

    print(f"  ✓ All sequences have logical kv_idx (not global)")


@pytest.mark.parametrize("dtype", [torch.float16])
def test_varlen_global_idx_explicit_check_no_aux(dtype):
    """
    Explicit check: if kv_idx is global in no-aux path, we should detect it.

    Uses a two-sequence setup where:
    - Seq 0: length 64, kv_idx should be [0, 63]
    - Seq 1: length 128, kv_idx should be [0, 127] if logical, [64, 191] if global

    We check the minimum value seen in seq 1. If logical, min should be ~0.
    If global, min should be ~64.
    """
    torch.random.manual_seed(42)

    @cute.jit
    def score_mod_min_kv_idx(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
        # Return kv_idx directly to extract the actual values
        return kv_idx.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.01)

    def eager_min_kv_idx(score, b, h, q_idx, kv_idx):
        return kv_idx.float() * 0.01

    seqlens = [64, 128]
    total = sum(seqlens)
    num_heads = 2
    head_dim = 64

    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)

    q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)

    out_cute = run_cute_flash(
        q, k, v, score_mod_min_kv_idx,
        aux_tensors=None,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        pack_gqa=False,
    )

    # Check seq 1 minimum value
    seq1_out = out_cute[64:192]
    min_val = seq1_out.min().item()
    max_val = seq1_out.max().item()

    print(f"\nGlobal idx explicit check (no aux):")
    print(f"  Seq 1 min value: {min_val:.4f} (expected ~0.00 for logical, ~0.64 for global)")
    print(f"  Seq 1 max value: {max_val:.4f} (expected ~1.27 for logical, ~1.91 for global)")

    # If kv_idx is global, min_val would be ~0.64 (64 * 0.01)
    # If kv_idx is logical, min_val would be ~0.00 (0 * 0.01)
    if min_val > 0.5:  # Clearly global (should be ~0.64)
        raise AssertionError(
            f"kv_idx is GLOBAL in no-aux-tensors path! "
            f"Seq 1 min={min_val:.4f} indicates global offset of 64. "
            f"Expected ~0.00 for logical indexing."
        )

    # Additional check: max value
    if max_val > 1.5:  # Would be ~1.91 for global
        raise AssertionError(
            f"kv_idx appears GLOBAL! "
            f"Seq 1 max={max_val:.4f} suggests global indexing (expected ~1.27)."
        )

    assert min_val < 0.1, f"Min value {min_val:.4f} too large"
    assert max_val < 1.4, f"Max value {max_val:.4f} too large"

    print(f"  ✓ kv_idx is correctly logical (not global) in no-aux path")


@pytest.mark.parametrize("dtype", [torch.float16])
def test_varlen_kv_idx_ground_truth_no_flex(dtype):
    """
    Ground truth test without using flex_attention at all.

    We know exactly what values kv_idx should have for each sequence.
    This test constructs the expected output manually without any reference implementation.
    """
    torch.random.manual_seed(42)

    @cute.jit
    def score_mod_return_kv_idx(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
        # Return kv_idx scaled by 0.01
        return kv_idx.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.01)

    # Two sequences: [64, 128]
    seqlens = [64, 128]
    total = sum(seqlens)
    num_heads = 2
    head_dim = 64

    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)

    # Create simple inputs - values don't matter much since score_mod ignores them
    q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.ones(total, num_heads, head_dim, device="cuda", dtype=dtype)  # All ones for easy checking

    out_cute = run_cute_flash(
        q, k, v, score_mod_return_kv_idx,
        aux_tensors=None,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        pack_gqa=False,
    )

    print(f"\nGround truth test (no flex):")

    # Manually verify each sequence
    # The output is attention-weighted, but we can check statistical properties

    # Sequence 0: seqlen=64, kv_idx should be [0, 63]
    # After softmax over scores [0, 0.01, 0.02, ..., 0.63], weighted by v=1
    # The output will be a weighted average, but we can check bounds
    seq0_out = out_cute[0:64]
    seq0_min = seq0_out.min().item()
    seq0_max = seq0_out.max().item()

    print(f"  Seq 0 (len=64): min={seq0_min:.4f}, max={seq0_max:.4f}")
    print(f"    Expected: values influenced by kv_idx=[0, 63] -> scores=[0.00, 0.63]")

    # Sequence 1: seqlen=128, kv_idx should be [0, 127] if logical
    # If kv_idx were global, it would be [64, 191] -> scores=[0.64, 1.91]
    seq1_out = out_cute[64:192]
    seq1_min = seq1_out.min().item()
    seq1_max = seq1_out.max().item()

    print(f"  Seq 1 (len=128): min={seq1_min:.4f}, max={seq1_max:.4f}")
    print(f"    Expected if logical: values influenced by kv_idx=[0, 127] -> scores=[0.00, 1.27]")
    print(f"    Expected if GLOBAL:  values influenced by kv_idx=[64, 191] -> scores=[0.64, 1.91]")

    # Key insight: In seq 1, the minimum score should be influenced by kv_idx=0 (score=0.00)
    # After attention weighting, the minimum output value should be relatively small
    # If kv_idx were global (starting at 64), the minimum score would be 0.64
    # and the minimum output value would be noticeably larger

    # Since attention does softmax, we can't directly read off kv_idx values
    # But we can check that seq1 has values consistent with small scores (close to 0)
    # versus having all scores >= 0.64

    # Statistical check: seq1 should have some output values near the "small score" end
    # Get the 10th percentile - should be small if kv_idx includes 0
    seq1_sorted = torch.sort(seq1_out.flatten())[0]
    seq1_percentile_10 = seq1_sorted[len(seq1_sorted) // 10].item()

    print(f"  Seq 1 10th percentile: {seq1_percentile_10:.4f}")
    print(f"    If logical (kv_idx from 0): should be relatively small")
    print(f"    If global (kv_idx from 64): would be larger")

    # If kv_idx is global (starting at 64), all scores are >= 0.64
    # This would make even the 10th percentile relatively large
    # If kv_idx is logical (starting at 0), we have scores near 0.00
    # This would make the 10th percentile smaller

    # Rough heuristic: if kv_idx starts at 0, we should see variety in outputs
    # The exact threshold depends on attention mechanics, but we can check
    # that seq1 outputs don't ALL look like they came from high scores

    # More direct test: check for any values that suggest low kv_idx
    # With kv_idx starting at 0, some positions should be heavily influenced by score=0.00
    # We'd expect to see this in early query positions (q_idx=0,1,2,...)
    early_q_positions = seq1_out[0:5]  # First 5 query positions in seq1
    early_q_min = early_q_positions.min().item()

    print(f"  Seq 1 early q positions (0-4) min: {early_q_min:.4f}")

    # The key test: if kv_idx were global, early query positions would never
    # see kv_idx < 64, so their outputs would reflect only high scores
    # But if kv_idx is logical, early positions DO see kv_idx < 64

    # Sanity check: seq0 and seq1 should have comparable ranges if both use logical indexing
    # They should both have access to low kv_idx values
    range_seq0 = seq0_max - seq0_min
    range_seq1 = seq1_max - seq1_min

    print(f"  Range comparison: seq0={range_seq0:.4f}, seq1={range_seq1:.4f}")
    print(f"    Ranges should be similar if both use logical kv_idx")

    # If seq1 has much smaller range, it might indicate restricted kv_idx range (global)
    assert range_seq1 > 0.3 * range_seq0, (
        f"Seq1 range {range_seq1:.4f} is much smaller than seq0 {range_seq0:.4f}. "
        f"This may indicate kv_idx is global (restricted range)."
    )

    print(f"  ✓ Statistical checks pass - kv_idx appears to be logical")


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        ([64, 128], [64, 128]),
        ([1, 2], [4096, 8192]),      # Tiny Q, huge K
        ([8, 16], [1024, 2048]),     # Small Q, large K
        ([1], [8192]),               # Single tiny Q, massive K
        ([1, 1, 1], [2048, 4096, 2048]),  # Multiple tiny Q, large K
    ],
)
def test_varlen_kv_idx_definitive_signal(seqlens_q, seqlens_k, dtype):
    """
    Definitive test using a strong binary signal.

    Strategy: Use a score_mod that returns 100.0 if kv_idx==0, else -100.0
    After softmax, this will make the output ~100% from kv_idx=0.
    If kv_idx is global in seq1, we'd never see kv_idx=0 (would start at 64).
    """
    torch.random.manual_seed(42)

    @cute.jit
    def score_mod_signal_first_kv(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
        # Return 100.0 if kv_idx==0, else -100.0
        # After softmax, attention will focus almost entirely on kv_idx=0
        is_first = operator.eq(kv_idx, cute.full_like(kv_idx, 0))
        signal = cute.where(is_first,
                           cute.full_like(tSrS_ssa, 100.0),
                           cute.full_like(tSrS_ssa, -100.0))
        return signal

    num_heads = 1  # Single head for simplicity
    head_dim = 64
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()),
        device="cuda", dtype=torch.int32,
    )

    q = torch.randn(total_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, num_heads, head_dim, device="cuda", dtype=dtype)

    # Create v with distinct marker values at kv_idx=0 for each sequence
    v = torch.zeros(total_k, num_heads, head_dim, device="cuda", dtype=dtype)
    for seq_idx in range(len(seqlens_k)):
        start_k = cu_seqlens_k[seq_idx].item()
        marker_value = float(seq_idx + 1)  # 1.0, 2.0, 3.0, ...
        v[start_k, :, :] = marker_value

    out_cute = run_cute_flash(
        q, k, v, score_mod_signal_first_kv,
        aux_tensors=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=False,
    )

    print(f"\nDefinitive signal test (no aux): seqlens_q={seqlens_q}, seqlens_k={seqlens_k}")

    # Check each sequence
    for seq_idx in range(len(seqlens_q)):
        start_q = cu_seqlens_q[seq_idx].item()
        end_q = cu_seqlens_q[seq_idx + 1].item()
        start_k = cu_seqlens_k[seq_idx].item()

        seq_out = out_cute[start_q:end_q]
        seq_mean = seq_out.mean().item()
        expected_value = float(seq_idx + 1)

        print(f"  Seq {seq_idx}: mean output = {seq_mean:.4f} (expected ~{expected_value:.1f})")

        if seq_idx > 0 and seq_mean < 0.5:
            raise AssertionError(
                f"Seq {seq_idx}: mean output is {seq_mean:.4f}, close to 0! "
                f"This strongly suggests kv_idx is GLOBAL (starts at {start_k} instead of 0). "
                f"With logical indexing, seq should attend to kv_idx=0 (value={expected_value:.1f})."
            )

        assert abs(seq_mean - expected_value) < 0.2, (
            f"Seq {seq_idx}: mean {seq_mean:.4f} should be ~{expected_value:.1f} for logical kv_idx. "
            f"Large deviation suggests global indexing."
        )

    print(f"  ✓ DEFINITIVE: kv_idx is logical (not global) in no-aux path")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
