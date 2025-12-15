import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as mlir_math
import operator

# =============================================================================
# Score_mod functions that don't use global indices
# All use signature: (tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors)
# =============================================================================


@cute.jit
def score_mod_identity(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    return tSrS_ssa


@cute.jit
def score_mod_causal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    mask = operator.ge(q_idx, kv_idx)
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_rel_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    diff = q_idx - kv_idx
    abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
    return tSrS_ssa + abs_diff.to(cutlass.Float32)


@cute.jit
def score_mod_rel_bias_x2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    diff = q_idx - kv_idx
    abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
    scaled = abs_diff * cute.full_like(abs_diff, 2)
    return tSrS_ssa + scaled.to(cutlass.Float32)


@cute.jit
def score_mod_times_two(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    return tSrS_ssa * cute.full_like(tSrS_ssa, 2)


@cute.jit
def score_mod_alibi(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
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
def score_mod_sliding_window(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    diff = q_idx - kv_idx
    abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
    mask = operator.le(abs_diff, cute.full_like(abs_diff, 256))
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_block_diagonal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    q_block = q_idx // 64
    kv_block = kv_idx // 64
    mask = operator.eq(q_block, kv_block)
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_causal_v2(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    diff = q_idx - kv_idx
    mask = operator.ge(diff, cute.full_like(diff, 0))
    return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))


@cute.jit
def score_mod_batch_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    batch_bias = aux_tensors[0]
    dtype = batch_bias.element_type
    b_frag = cute.make_fragment(1, cutlass.Int32)
    b_frag.store(b_idx)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = batch_bias[b_frag[0]]
    bias_val = (bias_frag.load()).to(cutlass.Float32)
    return tSrS_ssa + bias_val


@cute.jit
def score_mod_dual_buffer(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
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
# Score_mod functions that use global indices
# All use signature: (tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors)
# Global indices computed as: q_idx_global = q_idx + seqlen_info.offset_q (and similarly for kv)
# =============================================================================


@cute.jit
def score_mod_global_kv_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Per-token bias using global kv index."""
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type
    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]

    return tSrS_ssa + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_global_q_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Per-token bias using global q index."""
    offset_q = seqlen_info.offset_q
    q_idx_global = q_idx + offset_q
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type
    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[q_frag[0]]
    return tSrS_ssa + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_global_rel_plus_kv_bias(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Relative position (logical) + per-token bias (global kv)."""
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
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
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Both q and kv global indices."""
    offset_q = seqlen_info.offset_q
    q_idx_global = q_idx + offset_q
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
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
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Logical relative + global-indexed per-token bias."""
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
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


# "Stress tests" - score_mods with complex global index usage

@cute.jit
def score_mod_stress_complex_arithmetic(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """All indices in complex arithmetic."""
    offset_q = seqlen_info.offset_q
    q_idx_global = q_idx + offset_q
    bias = aux_tensors[0]
    dtype = bias.element_type

    # Use absolute value instead of squaring to avoid overflow with large sequences
    rel_pos = q_idx - kv_idx
    rel_pos_abs = cute.TensorSSA(mlir_math.absi(rel_pos), rel_pos.shape, rel_pos.dtype)
    rel_bias = rel_pos_abs.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.001)

    q_frag = cute.make_fragment(1, cutlass.Int32)
    q_frag.store(q_idx_global)
    bias_q_frag = cute.make_fragment(1, dtype)
    bias_q_frag[0] = bias[q_frag[0]]
    bias_q = (bias_q_frag.load()).to(cutlass.Float32)

    scale = (b_idx + cute.full_like(b_idx, 1)) * (h_idx + cute.full_like(h_idx, 1))
    scale_f32 = scale.to(cutlass.Float32) * 0.001

    result = tSrS_ssa + rel_bias + bias_q * scale_f32
    return result


@cute.jit
def score_mod_stress_conditional_mask(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Conditional masking with global vs logical."""
    offset_q = seqlen_info.offset_q
    q_idx_global = q_idx + offset_q
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
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
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Multiple aux tensors with different indexing."""
    offset_q = seqlen_info.offset_q
    q_idx_global = q_idx + offset_q
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
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
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Verify global - logical = offset."""
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
    token_bias = aux_tensors[0]
    dtype = token_bias.element_type

    kv_frag = cute.make_fragment(1, cutlass.Int32)
    kv_frag.store(kv_idx_global)
    bias_frag = cute.make_fragment(1, dtype)
    bias_frag[0] = token_bias[kv_frag[0]]

    return tSrS_ssa + (bias_frag.load()).to(cutlass.Float32)


@cute.jit
def score_mod_stress_xor_pattern(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """XOR-based pattern using index bits."""
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
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


@cute.jit
def score_mod_debug_global_idx(
    tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    # Don't read from aux_tensors at all - just add the global index as bias
    offset_k = seqlen_info.offset_k
    kv_idx_global = kv_idx + offset_k
    bias = kv_idx_global.to(cutlass.Float32) * cute.full_like(tSrS_ssa, 0.001)
    return tSrS_ssa + bias


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
        # Calculate valid length for this sequence
        start = cu_seqlens_k[b]
        seq_len = cu_seqlens_k[b+1] - start

        # Clamp kv_idx.
        safe_kv_idx = torch.clamp(kv_idx, max=seq_len - 1)

        return score + bias_tensor[start + safe_kv_idx]
    return mod


def packed_q_bias_factory(bias_tensor, cu_seqlens_q):
    def mod(score, b, h, q_idx, kv_idx):
        start = cu_seqlens_q[b]
        seq_len = cu_seqlens_q[b+1] - start

        # Clamp q_idx
        safe_q_idx = torch.clamp(q_idx, max=seq_len - 1)

        return score + bias_tensor[start + safe_q_idx]
    return mod


def packed_rel_plus_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        start = cu_seqlens_k[b]
        seq_len = cu_seqlens_k[b+1] - start

        # Clamp kv_idx
        safe_kv_idx = torch.clamp(kv_idx, max=seq_len - 1)

        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.1
        return score + rel_bias + bias_tensor[start + safe_kv_idx]

    return mod


def packed_q_and_kv_bias_factory(q_bias, kv_bias, cu_seqlens_q, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        # Handle Q bounds
        q_start = cu_seqlens_q[b]
        q_len = cu_seqlens_q[b+1] - q_start
        safe_q_idx = torch.clamp(q_idx, max=q_len - 1)

        # Handle KV bounds
        kv_start = cu_seqlens_k[b]
        kv_len = cu_seqlens_k[b+1] - kv_start
        safe_kv_idx = torch.clamp(kv_idx, max=kv_len - 1)

        return score + q_bias[q_start + safe_q_idx] + kv_bias[kv_start + safe_kv_idx]

    return mod


def packed_logical_rel_plus_kv_bias_factory(bias_tensor, cu_seqlens_k):
    def mod(score, b, h, q_idx, kv_idx):
        rel_bias = torch.abs(q_idx - kv_idx).float() * 0.01
        return score + rel_bias + bias_tensor[cu_seqlens_k[b] + kv_idx]

    return mod


def stress_complex_arithmetic_factory(bias, cu_seqlens_q):
    def mod(score, b, h, q_idx, kv_idx):
        # Use absolute value instead of squaring to avoid overflow with large sequences
        rel_pos_abs = torch.abs(q_idx - kv_idx)
        q_global = cu_seqlens_q[b] + q_idx
        bias_q = bias[q_global]
        scale = (b + 1) * (h + 1) * 0.001
        rel_bias = rel_pos_abs * 0.001
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

def debug_global_idx_factory(bias, cu_seqlens_k):
    offsets = cu_seqlens_k.tolist()
    def mod(score, b, h, q_idx, kv_idx):
        global_kv = offsets[b] + kv_idx
        return score + global_kv.float() * 0.001
    return mod
