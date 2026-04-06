"""
Dropout for Flash Attention CuTe DSL kernels.

Philox 4x32 counter-based PRNG with 2x4 group batching for efficient
dropout mask generation. Each Philox call produces 128 random bits
used as 8 × 16-bit masks covering 8 elements — matching TriDao's
recommended approach from FA2.

Keying on global (row, col) position ensures identical masks in forward
and backward regardless of MMA layout, tile size, or architecture.

Reference: FA2 C++ in csrc/flash_attn/src/{dropout.h, philox.cuh}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32

from quack import layout_utils
from quack.rounding import (
    mul_wide_u32,
    PHILOX_ROUND_A,
    PHILOX_ROUND_B,
    PHILOX_KEY_A,
    PHILOX_KEY_B,
    PHILOX_N_ROUNDS_DEFAULT,
)

# Group dimensions: 2 rows × 4 cols = 8 elements per Philox call.
# Each element gets 16 random bits (128 / 8 = 16).
DROPOUT_GROUP_ROWS: int = 2
DROPOUT_GROUP_COLS: int = 4


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
# Dropout parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DropoutParams:
    """Dropout configuration."""

    p_dropout: float

    @property
    def p_keep_uint16(self) -> int:
        """16-bit threshold: random u16 <= this -> keep."""
        return int(65535 * (1.0 - self.p_dropout))

    @property
    def p_keep_uint8(self) -> int:
        """8-bit threshold (for backward compat)."""
        return int(255 * (1.0 - self.p_dropout))

    @property
    def rp_dropout(self) -> float:
        if self.p_dropout >= 1.0:
            return 0.0
        return 1.0 / (1.0 - self.p_dropout)

    @property
    def enabled(self) -> bool:
        return self.p_dropout > 0.0


# ---------------------------------------------------------------------------
# Dropout mask application — 8 elements per Philox call
# ---------------------------------------------------------------------------


@cute.jit
def apply_dropout_mask(
    acc_S: cute.Tensor,
    thr_mma: cute.TiledMma,
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
    """Apply dropout mask with 2x4 group batching.

    Each Philox call produces 128 bits = 8 × 16-bit random values,
    covering a 2-row × 4-col group of 8 elements. Elements within the
    group are assigned by (row % 2) * 4 + (col % 4), giving each element
    a unique 16-bit random value for the keep/drop decision.

    Uses global (row, col) position for the group key, ensuring identical
    masks in forward and backward regardless of MMA layout or tile size.

    Keying:
      counter = (batch*nheads + head, 0, row_group, col_group)
      key     = (seed_lo, seed_hi)
    """
    acc_shape = (tile_m, tile_n) if not cutlass.const_expr(transpose) else (tile_n, tile_m)
    cS = cute.make_identity_tensor(acc_shape)
    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=transpose)
    t0ScS_mn = layout_utils.reshape_acc_to_mn(
        thr_mma.get_slice(0).partition_C(cS), transpose=transpose
    )
    thr_col_offset = tScS_mn[0][1]

    rng_key_lo = Uint32(batch_idx * nheads + head_idx)
    # 16-bit threshold from the 8-bit parameter (scale up)
    p_threshold_16 = Uint32(p_keep_uint8) * Uint32(257)  # 0-255 -> 0-65535

    acc_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=transpose)
    nrow = cute.size(acc_mn.shape[0])
    ncol = cute.size(acc_mn.shape[1])

    # Cache: Philox called once per 2x4 group (8 elements)
    cache_rg = Uint32(0xFFFFFFFF)
    cache_cg = Uint32(0xFFFFFFFF)
    cache_r0 = Uint32(0)
    cache_r1 = Uint32(0)
    cache_r2 = Uint32(0)
    cache_r3 = Uint32(0)

    for r in cutlass.range_constexpr(nrow):
        local_row = tScS_mn[r, 0][0]
        global_row = local_row + m_block * tile_m
        row_group = Uint32(global_row) >> Uint32(1)  # // 2

        for c in cutlass.range_constexpr(ncol):
            col_local = t0ScS_mn[0, c][1]
            global_col = thr_col_offset + col_local + n_block * tile_n
            col_group = Uint32(global_col) >> Uint32(2)  # // 4

            # Philox on group change
            if (row_group != cache_rg) | (col_group != cache_cg):
                cache_rg = row_group
                cache_cg = col_group
                cache_r0, cache_r1, cache_r2, cache_r3 = philox_4x32(
                    rng_key_lo, Uint32(0), row_group, col_group,
                    seed_lo, seed_hi,
                )

            # Element index within the 2x4 group: (row%2)*4 + (col%4) -> 0..7
            elem_in_group = (Int32(global_row) & Int32(1)) * Int32(4) + (Int32(global_col) & Int32(3))

            # Extract 16-bit value: 8 values packed in 4 words (2 per word)
            # word_idx = elem_in_group // 2, half_idx = elem_in_group % 2
            # rand_u16 = (word >> (half_idx * 16)) & 0xFFFF
            word_idx = elem_in_group >> Int32(1)
            half_shift = Uint32(elem_in_group & Int32(1)) << Uint32(4)  # 0 or 16

            # Select word (constexpr-friendly since only 4 options)
            word = cache_r0
            if word_idx == Int32(1):
                word = cache_r1
            if word_idx == Int32(2):
                word = cache_r2
            if word_idx == Int32(3):
                word = cache_r3

            rand_u16 = (word >> half_shift) & Uint32(65535)

            keep_f = Float32(Uint32(rand_u16 <= p_threshold_16))
            acc_mn[r, c] = acc_mn[r, c] * (rp_dropout * keep_f)


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
    """Apply dropout mask to SM100 attention scores (TMEM layout).

    Same 2x4 group batching as apply_dropout_mask but uses TMEM
    coordinate tensor.
    """
    rng_key_lo = Uint32(batch_idx * nheads + head_idx)
    p_threshold_16 = Uint32(p_keep_uint8) * Uint32(257)

    cache_rg = Uint32(0xFFFFFFFF)
    cache_cg = Uint32(0xFFFFFFFF)
    cache_r0 = Uint32(0)
    cache_r1 = Uint32(0)
    cache_r2 = Uint32(0)
    cache_r3 = Uint32(0)

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
        row_group = Uint32(global_row) >> Uint32(1)
        col_group = Uint32(global_col) >> Uint32(2)

        if (row_group != cache_rg) | (col_group != cache_cg):
            cache_rg = row_group
            cache_cg = col_group
            cache_r0, cache_r1, cache_r2, cache_r3 = philox_4x32(
                rng_key_lo, Uint32(0), row_group, col_group,
                seed_lo, seed_hi,
            )

        elem_in_group = (Int32(global_row) & Int32(1)) * Int32(4) + (Int32(global_col) & Int32(3))
        word_idx = elem_in_group >> Int32(1)
        half_shift = Uint32(elem_in_group & Int32(1)) << Uint32(4)

        word = cache_r0
        if word_idx == Int32(1):
            word = cache_r1
        if word_idx == Int32(2):
            word = cache_r2
        if word_idx == Int32(3):
            word = cache_r3

        rand_u16 = (word >> half_shift) & Uint32(65535)

        keep_f = Float32(Uint32(rand_u16 <= p_threshold_16))
        acc_S[i] = acc_S[i] * (rp_dropout * keep_f)
