"""
Dropout for Flash Attention CuTe DSL kernels.

FA2-style dropout using Philox 4x32 PRNG keyed on MMA layout positions.
Each Philox call produces 128 random bits used as 8 × 16-bit masks
covering 8 elements via logical_divide layout conversion.

Forward/backward tiles are matched when dropout is enabled, ensuring
identical MMA layouts and thus identical mask assignments. This is the
same approach FA2 C++ uses.

Reference: FA2 C++ csrc/flash_attn/src/{dropout.h, philox.cuh, utils.h}

Note: return_softmax (returning the dropout mask to the caller) is not yet
supported. The V=Identity extraction trick in the test suite provides
equivalent mask inspection capability for debugging.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32

from quack.rounding import (
    mul_wide_u32,
    PHILOX_ROUND_A,
    PHILOX_ROUND_B,
    PHILOX_KEY_A,
    PHILOX_KEY_B,
    PHILOX_N_ROUNDS_DEFAULT,
)


# ---------------------------------------------------------------------------
# Philox 4x32 PRNG
# ---------------------------------------------------------------------------


@cute.jit
def philox_4x32(
    c0_in: Uint32, c1_in: Uint32,
    c2_in: Uint32, c3_in: Uint32,
    k0_in: Uint32, k1_in: Uint32,
) -> tuple:
    """Philox 4x32 PRNG. Counter=(c0..c3), Key=(k0,k1)."""
    c0 = Uint32(c0_in)
    c1 = Uint32(c1_in)
    c2 = Uint32(c2_in)
    c3 = Uint32(c3_in)
    k0 = Uint32(k0_in)
    k1 = Uint32(k1_in)
    for _ in range(PHILOX_N_ROUNDS_DEFAULT):
        hi_b, lo_b = mul_wide_u32(c2, Uint32(PHILOX_ROUND_B))
        hi_a, lo_a = mul_wide_u32(c0, Uint32(PHILOX_ROUND_A))
        c0 = hi_b ^ c1 ^ k0
        c1 = lo_b
        c2 = hi_a ^ c3 ^ k1
        c3 = lo_a
        k0 = k0 + Uint32(PHILOX_KEY_A)
        k1 = k1 + Uint32(PHILOX_KEY_B)
    return c0, c1, c2, c3


# ---------------------------------------------------------------------------
# FA2-style dropout: MMA-layout keying via logical_divide
# ---------------------------------------------------------------------------


@cute.jit
def _extract_u16(r0: Uint32, r1: Uint32, r2: Uint32, r3: Uint32,
                 idx: cutlass.Constexpr[int]) -> Uint32:
    """Extract 16-bit value at index idx (0-7) from 4 Philox words.

    idx is constexpr so word/half selection is resolved at compile time.
    """
    if cutlass.const_expr(idx < 2):
        return (r0 >> Uint32((idx % 2) * 16)) & Uint32(65535)
    if cutlass.const_expr(idx < 4):
        return (r1 >> Uint32((idx % 2) * 16)) & Uint32(65535)
    if cutlass.const_expr(idx < 6):
        return (r2 >> Uint32((idx % 2) * 16)) & Uint32(65535)
    return (r3 >> Uint32((idx % 2) * 16)) & Uint32(65535)


@cute.jit
def apply_dropout_mask(
    acc_S: cute.Tensor,
    batch_idx: Int32,
    head_idx: Int32,
    nheads: Int32,
    m_block: Int32,
    n_block: Int32,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    num_warps_m: cutlass.Constexpr[int],
    p_keep_uint8: cutlass.Constexpr[int],
    rp_dropout: Float32,
    seed_lo: Uint32,
    seed_hi: Uint32,
):
    """Apply dropout mask via MMA-layout Philox keying (FA2-style).

    1. logical_divide the accumulator's N mode by 2:
       (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N/2))

    2. Iterate over (m, n_half) calling Philox once per 8 elements.
       Each call produces 128 bits = 8 × 16-bit random values.

    3. Apply masks using compile-time byte indexing via constexpr loops.

    Keying (matching FA2 dropout.h):
      offset      = (batch * nheads + head) * 32 + lane_id
      subsequence = (block_row, block_col) as uint2
      seed        = (seed_lo, seed_hi)

    Forward and backward use matched tile sizes when dropout is enabled,
    so the MMA layout and block_row/block_col values are identical.
    """
    tidx = cute.arch.thread_idx()[0]
    warp_id = tidx / Int32(32)
    lane_id = tidx % Int32(32)

    # 16-bit threshold from 8-bit parameter
    p_threshold = Uint32(p_keep_uint8) * Uint32(257)

    # FA2 offset: unique per (head, lane)
    philox_offset = Uint32((batch_idx * nheads + head_idx) * Int32(32) + lane_id)

    # Convert accumulator layout:
    # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N/2))
    divided = cute.logical_divide(acc_S, (None, None, 2))

    mma_m = cute.size(divided.shape[1])
    mma_n_half = cute.size(divided.shape[2][1])

    # FA2 block_row/block_col computation
    block_row_base = m_block * Int32(tile_m // 16) + warp_id
    block_col_base = n_block * Int32(tile_n // 32)

    for m in cutlass.range_constexpr(mma_m):
        block_row = block_row_base + Int32(m) * Int32(num_warps_m)

        for n_half in cutlass.range_constexpr(mma_n_half):
            block_col = block_col_base + Int32(n_half)

            # One Philox call per 8 elements (2 pair-halves × 4 registers)
            r0, r1, r2, r3 = philox_4x32(
                philox_offset, Uint32(0),
                Uint32(block_row), Uint32(block_col),
                seed_lo, seed_hi,
            )

            # Apply 8 × 16-bit masks to 8 elements.
            # j selects pair-half (inner dim of logical_divide),
            # reg selects register (mode 0). All indices constexpr.
            for j in cutlass.range_constexpr(2):
                for reg in cutlass.range_constexpr(4):
                    u16_idx = j * 4 + reg
                    rand_u16 = _extract_u16(r0, r1, r2, r3, u16_idx)

                    keep_f = Float32(Uint32(rand_u16 <= p_threshold))
                    divided[reg, m, (j, n_half)] = (
                        divided[reg, m, (j, n_half)] * (rp_dropout * keep_f)
                    )


@cute.jit
def apply_dropout_mask_sm100(
    acc_S: cute.Tensor,
    tScS_t2r: cute.Tensor,
    batch_idx: Int32,
    head_idx: Int32,
    nheads: Int32,
    m_block: Int32,
    n_block: Int32,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    p_keep_uint8: cutlass.Constexpr[int],
    rp_dropout: Float32,
    seed_lo: Uint32,
    seed_hi: Uint32,
    transpose: cutlass.Constexpr[bool] = False,
):
    """Apply dropout mask to SM100 TMEM layout (per-element, position-keyed).

    SM100 uses UMMA with TMEM — different register layout from SM80/SM90.
    Uses per-element Philox keyed on global (row, col) since TMEM elements
    don't group into the 8-element MMA register blocks used by SM80/SM90/SM120.

    NOTE: This keying scheme differs from apply_dropout_mask. Both are
    internally consistent (fwd/bwd masks match), but produce different
    dropout patterns for the same seed. This is acceptable since SM100
    kernels are a separate compilation target.

    TODO: Optimize to batch Philox calls once TMEM element grouping is
    understood (currently uses 1 byte of 128 random bits per call).
    """
    rng_key_lo = Uint32(batch_idx * nheads + head_idx)
    p_threshold = Uint32(p_keep_uint8)

    nelem = cute.size(tScS_t2r.shape)
    for i in cutlass.range_constexpr(nelem):
        if cutlass.const_expr(not transpose):
            local_row = tScS_t2r[i][0]
            local_col = tScS_t2r[i][1]
        else:
            local_row = tScS_t2r[i][1]
            local_col = tScS_t2r[i][0]
        global_row = local_row + m_block * tile_m
        global_col = local_col + n_block * tile_n

        pr0, _pr1, _pr2, _pr3 = philox_4x32(
            rng_key_lo, Uint32(0),
            Uint32(global_row), Uint32(global_col),
            seed_lo, seed_hi,
        )
        rand_byte = pr0 & Uint32(255)

        keep_f = Float32(Uint32(rand_byte <= p_threshold))
        acc_S[i] = acc_S[i] * (rp_dropout * keep_f)
