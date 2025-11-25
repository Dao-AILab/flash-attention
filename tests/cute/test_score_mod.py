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
    # Varlen case: inputs already packed
    if cu_seqlens_q is not None:
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

    # Batched case
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
    """Simulate varlen by running flex on each sequence slice."""
    results = []
    num_batches = len(cu_seqlens_q) - 1

    for i in range(num_batches):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        # Reshape to (1, H, S, D)
        q_slice = q[start_q:end_q].unsqueeze(0).transpose(1, 2)
        k_slice = k[start_k:end_k].unsqueeze(0).transpose(1, 2)
        v_slice = v[start_k:end_k].unsqueeze(0).transpose(1, 2)

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
            enable_gqa=q_slice.shape[1] != k_slice.shape[1],
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
        ([64, 56, 128], [64, 56, 128]),
        ([12, 256], [12, 256]),
        ([100, 100], [50, 50]),
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
# Tests: Varlen with 8-arg score_mod (global indices)
# =============================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        ([64, 56, 128], [64, 56, 128]),
        ([12, 256], [12, 256]),
        ([100, 100], [50, 50]),
        ([32, 64, 128, 64], [32, 64, 128, 64]),
        ([17, 33, 65], [17, 33, 65]),
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
        ([128], [128]),
        ([32, 64], [128, 256]),
        ([1, 1, 1], [1, 1, 1]),
        ([64, 56, 128], [64, 56, 128]),
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
        ([64, 128], [64, 128]),
        ([17, 33, 65], [17, 33, 65]),
        ([100, 100], [50, 50]),
        ([1, 255, 1], [1, 255, 1]),
        ([128, 1, 128], [128, 1, 128]),
        ([7, 13, 19, 23], [7, 13, 19, 23]),
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
        ([64, 128], [64, 128]),
        ([32, 64, 96], [32, 64, 96]),
        ([17, 33, 65], [17, 33, 65]),
        ([128, 1, 128], [128, 1, 128]),
        ([100, 100], [50, 50]),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
