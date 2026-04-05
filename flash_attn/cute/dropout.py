"""
Dropout for Flash Attention CuTe DSL kernels.

Philox 4x32 counter-based PRNG for deterministic dropout mask generation.
Position-based keying on global (row, col) ensures identical masks in
forward and backward regardless of thread partitioning.

Uses quack's mul_wide_u32 and Philox constants.

Reference: FA2 C++ in csrc/flash_attn/src/{dropout.h, philox.cuh}
"""

from __future__ import annotations

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Philox 4x32 PRNG
# ---------------------------------------------------------------------------


@cute.jit
def philox_4x32(
    c0_in: Uint32,
    c1_in: Uint32,
    c2_in: Uint32,
    c3_in: Uint32,
    k0_in: Uint32,
    k1_in: Uint32,
) -> tuple:
    """Philox 4x32 PRNG with full 4-element counter + 2-element key.

    Extends quack.rounding.philox to accept FA2's full 6-input layout.
    Uses quack's mul_wide_u32 and default round count.
    """
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

    p_dropout: float  # Dropout probability (0 = disabled)

    @property
    def p_keep_uint8(self) -> int:
        """Threshold: random byte <= this -> keep. E.g. p=0.1 -> 229."""
        return int(255 * (1.0 - self.p_dropout))

    @property
    def rp_dropout(self) -> float:
        """Scale factor: 1 / (1 - p_dropout)."""
        if self.p_dropout >= 1.0:
            return 0.0
        return 1.0 / (1.0 - self.p_dropout)

    @property
    def enabled(self) -> bool:
        return self.p_dropout > 0.0


# ---------------------------------------------------------------------------
# Dropout mask application
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
    """Apply dropout mask to attention accumulator in-place.

    Per-element Philox keyed on global (row, col) position with full
    64-bit seed. Each element gets one Philox call; byte 0 of word 0
    determines keep/drop.

    Keying layout (all 6 Philox inputs used):
      counter[0]   = batch*nheads + head
      counter[1]   = 0
      counter[2:3] = (row, col) position
      key[0:1]     = (seed_lo, seed_hi) full 64-bit seed
    """
    acc_shape = (tile_m, tile_n) if not cutlass.const_expr(transpose) else (tile_n, tile_m)
    cS = cute.make_identity_tensor(acc_shape)
    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=transpose)
    t0ScS_mn = layout_utils.reshape_acc_to_mn(
        thr_mma.get_slice(0).partition_C(cS), transpose=transpose
    )
    thr_col_offset = tScS_mn[0][1]

    rng_key_lo = Uint32(batch_idx * nheads + head_idx)
    p_threshold = Uint32(p_keep_uint8)

    acc_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=transpose)
    nrow = cute.size(acc_mn.shape[0])
    ncol = cute.size(acc_mn.shape[1])

    for r in cutlass.range_constexpr(nrow):
        local_row = tScS_mn[r, 0][0]
        global_row = local_row + m_block * tile_m

        for c in cutlass.range_constexpr(ncol):
            col_local = t0ScS_mn[0, c][1]
            global_col = thr_col_offset + col_local + n_block * tile_n

            pr0, _pr1, _pr2, _pr3 = philox_4x32(
                rng_key_lo, Uint32(0),
                Uint32(global_row), Uint32(global_col),
                seed_lo, seed_hi,
            )
            rand_byte = pr0 & Uint32(255)

            keep_f = Float32(Uint32(rand_byte <= p_threshold))
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

    Same per-element Philox as apply_dropout_mask but uses a pre-computed
    coordinate tensor (tScS_t2r) instead of reshape_acc_to_mn, since SM100
    softmax operates on TMEM-loaded data with a different layout.

    When transpose=True (backward), coord indices are swapped because
    S^T = K @ Q^T uses an (N, M) identity tensor.
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
