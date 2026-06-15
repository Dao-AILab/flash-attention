# Copyright (c) 2025, Tri Dao.
# Philox4x32 counter-based RNG + dropout helpers for FA4 CuTe-DSL kernels.
#
# Bit-exact with csrc/flash_attn/src/philox.cuh (6 loop rounds + 1 final round)
# and csrc/flash_attn/src/dropout.h indexing. Used to apply dropout to the
# attention probabilities P after softmax (forward) and to regenerate the
# identical keep-mask during the backward pass.

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr

PHILOX_A = 0xD2511F53
PHILOX_B = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85


@cute.jit
def _mulhi_u32(a: Int32, b: Int32) -> Int32:
    """High 32 bits of the UNSIGNED 32x32 product (int32 is signed in the DSL)."""
    a64 = a.to(Int64) & Int64(0xFFFFFFFF)
    b64 = b.to(Int64) & Int64(0xFFFFFFFF)
    return ((a64 * b64) >> 32).to(Int32)


@cute.jit
def philox_4x32(seed: Int64, subsequence: Int64, offset: Int64):
    """Return a 4-tuple of uint32 (as Int32) random words: philox(seed, subsequence, offset).

    Matches the 7-round (6 loop + 1 final) Philox in philox.cuh exactly.
    """
    k0 = (seed & Int64(0xFFFFFFFF)).to(Int32)
    k1 = ((seed >> 32) & Int64(0xFFFFFFFF)).to(Int32)
    c0 = (offset & Int64(0xFFFFFFFF)).to(Int32)
    c1 = ((offset >> 32) & Int64(0xFFFFFFFF)).to(Int32)
    c2 = (subsequence & Int64(0xFFFFFFFF)).to(Int32)
    c3 = ((subsequence >> 32) & Int64(0xFFFFFFFF)).to(Int32)
    for _ in cutlass.range_constexpr(6):
        hi0 = _mulhi_u32(Int32(PHILOX_A), c0)
        lo0 = Int32(PHILOX_A) * c0
        hi1 = _mulhi_u32(Int32(PHILOX_B), c2)
        lo1 = Int32(PHILOX_B) * c2
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = k0 + Int32(PHILOX_W0)
        k1 = k1 + Int32(PHILOX_W1)
    hi0 = _mulhi_u32(Int32(PHILOX_A), c0)
    lo0 = Int32(PHILOX_A) * c0
    hi1 = _mulhi_u32(Int32(PHILOX_B), c2)
    lo1 = Int32(PHILOX_B) * c2
    return hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0


@cute.jit
def uniform_from_uint32(x: Int32) -> Float32:
    """Map a uint32 to a Float32 in [0, 1) (24-bit mantissa, like curand_uniform)."""
    # take top 24 bits -> [0, 2^24) -> scale by 2^-24
    u = (x.to(Int64) & Int64(0xFFFFFFFF))
    top = (u >> 8).to(Int32)  # 24 bits
    return top.to(Float32) * Float32(1.0 / 16777216.0)


@cute.jit
def keep_threshold_u32(p_keep: Float32) -> Int32:
    """uint32 threshold: keep a draw if (uint32 >> 8) < round(p_keep * 2^24).

    Using the same 24-bit space as uniform_from_uint32 so the test is
    `uniform < p_keep`.
    """
    thr = cute.math.floor(p_keep * Float32(16777216.0))
    return thr.to(Int32)


@cute.jit
def apply_dropout(
    acc_S: cute.Tensor,        # post-softmax probabilities P (will be modified in-place)
    tScS: cute.Tensor,         # identity tensor: tScS[i] = (q_idx, kv_idx) global coords
    seed: Int64,
    offset: Int64,
    p_keep: Float32,           # 1 - dropout_p
    scale: Float32,            # 1 / (1 - dropout_p)
    batch_idx: Int32,
    head_idx: Int32,
    num_heads: cutlass.Constexpr[int],
    seqlen_k: Int32,
    transpose_indices: cutlass.Constexpr[bool] = False,
):
    """Apply inverted dropout to P in-place, keyed by global (batch, head, q_idx, kv_idx).

    Layout-agnostic: each element's RNG draw is determined solely by its global
    coordinates, so the forward pass and the backward recompute generate the
    IDENTICAL keep-mask regardless of fragment layout. When ``transpose_indices``
    is set (backward SdP_swapAB), tScS[i] holds (kv_idx, q_idx) instead of
    (q_idx, kv_idx), so we swap accordingly.
    """
    n_vals = cutlass.const_expr(cute.size(acc_S.shape))
    bh = batch_idx * Int32(num_heads) + head_idx
    if cutlass.const_expr(transpose_indices):
        q_pos = cutlass.const_expr(1)
        kv_pos = cutlass.const_expr(0)
    else:
        q_pos = cutlass.const_expr(0)
        kv_pos = cutlass.const_expr(1)
    for i in cutlass.range(n_vals, unroll_full=True):
        q_idx = tScS[i][q_pos]
        kv_idx = tScS[i][kv_pos]
        # Global linear element id within this (batch, head) plane.
        lin = q_idx.to(Int64) * seqlen_k.to(Int64) + kv_idx.to(Int64)
        subseq = bh.to(Int64)
        r0, r1, r2, r3 = philox_4x32(seed, subseq, offset + lin)
        u = uniform_from_uint32(r0)  # [0, 1)
        keep = u < p_keep
        acc_S[i] = acc_S[i] * scale if keep else Float32(0.0)

