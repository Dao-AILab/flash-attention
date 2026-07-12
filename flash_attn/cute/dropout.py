# Copyright (c) 2025, Tri Dao.

"""Standalone dropout primitives for the SM100 FA4 kernels.

These are architecture-agnostic ``@dsl_user_op`` helpers (Philox byte
extraction, threshold packing, mask packing/scaling, smem mask staging,
and the fused apply-dropout paths) shared by the forward
(``flash_fwd_sm100``) and backward (``flash_bwd_sm100``) kernels. They
were factored out of ``softmax.py`` to keep that module focused on the
online-softmax reduction itself.
"""

import cutlass.cute as cute

from cutlass import Float32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import Int32 as Int32Type, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm


# =============================================================================
# Forward-dropout HW/SW exp2 schedule constants.
#
# Both values are sweep winners on the canonical
# B=4,S=8K,Hq=8,Hk=1,D=128 config; every other setting regressed, so
# there is no runtime tuning lever.
# =============================================================================

# 8-bit bitmask selecting which g-slots in ``apply_exp2_convert_row_sum_pair``
# use a software polynomial exp2 (FMA pipe via ``utils.ex2_emulation_2``)
# vs the hardware ``MUFU.EX2`` (XU pipe). ``4`` = g=2 SW fills one XU-pipe
# scheduling bubble between ``MUFU.EX2`` and the dependent ``FSEL`` / ``F2FP``
# work, winning ~3 us.
FA4_DROP_E2E_MASK: int = 4

# Polynomial degree for the software exp2 emulator used at the SW slot(s)
# above. ``1`` (= 2 terms) is the winner of a 0..5 sweep; higher degrees
# re-open the scheduling bubble and regress.
FA4_DROP_E2E_POLY_DEGREE: int = 1


def fa2_philox_row_index(seq_row):
    """Decompose a global query-row index into its FA2 Philox key parts.

    FA2 lays a 16-row MMA output tile across a warp as 8 two-row lane
    groups, so the Philox counter for a row is keyed by:

      * ``row_arg``          = ``seq_row // 16`` (which 16-row MMA tile),
      * ``base_lane``        = ``(seq_row % 8) * 4`` (ctr_x base for the
        row's lane group; the 4 sub-lanes add 0..3),
      * ``byte_pair_offset`` = ``((seq_row // 8) % 2) * 2`` (top vs bottom
        8-row half selects which byte pair of each rng word to read).

    Returns ``(row_arg, base_lane, byte_pair_offset)`` as ``Uint32``.
    Shared by the forward 1-row path and the backward mask staging so
    both stay bit-identical to FA2. ``seq_row`` is a traced Int32.
    """
    row_arg = Uint32(seq_row // 16)
    base_lane = Uint32((seq_row % 8) * 4)
    byte_pair_offset = Uint32(((seq_row // 8) % 2) * 2)
    return row_arg, base_lane, byte_pair_offset


@dsl_user_op
def f32_to_dropout_threshold(p_dropout: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float dropout probability to packed 8-bit threshold for f16x2 comparison.

    Returns (uint8(p * 255) << 16) | uint8(p * 255).
    """
    val = Float32(p_dropout).ir_value(loc=loc, ip=ip)
    i32_ty = ir.IntegerType.get_signless(32)
    f32_ty = ir.F32Type.get()

    scale = arith.constant(f32_ty, 255.0, loc=loc, ip=ip)
    scaled = arith.mulf(val, scale, loc=loc, ip=ip)
    uint8_val = arith.fptoui(i32_ty, scaled, loc=loc, ip=ip)

    sixteen = arith.constant(i32_ty, 16, loc=loc, ip=ip)
    hi = arith.shli(uint8_val, sixteen, loc=loc, ip=ip)
    packed = arith.ori(hi, uint8_val, loc=loc, ip=ip)

    return Uint32(packed)


@dsl_user_op
def dropout_mask_to_bit(mask_val: Float32, bit_pos: Uint32, accum: Uint32, *, loc=None, ip=None) -> Uint32:
    """Pack a Float32 dropout mask (0.0 or 1.0) into a bit at bit_pos in accum."""
    i32_ty = ir.IntegerType.get_signless(32)
    f32_ty = ir.F32Type.get()
    val = Float32(mask_val).ir_value(loc=loc, ip=ip)
    half = arith.constant(f32_ty, 0.5, loc=loc, ip=ip)
    keep = arith.cmpf(arith.CmpFPredicate.OGT, val, half, loc=loc, ip=ip)
    one = arith.constant(i32_ty, 1, loc=loc, ip=ip)
    zero = arith.constant(i32_ty, 0, loc=loc, ip=ip)
    bit = arith.select(keep, one, zero, loc=loc, ip=ip)
    pos = Uint32(bit_pos).ir_value(loc=loc, ip=ip)
    shifted = arith.shli(bit, pos, loc=loc, ip=ip)
    acc = Uint32(accum).ir_value(loc=loc, ip=ip)
    result = arith.ori(acc, shifted, loc=loc, ip=ip)
    return Uint32(result)


@dsl_user_op
def dropout_bit_to_scale(bits: Uint32, bit_pos: Uint32, scale: Float32, *, loc=None, ip=None) -> Float32:
    """Extract bit at bit_pos from bits. Return scale if 1, else 0.0."""
    i32_ty = ir.IntegerType.get_signless(32)
    f32_ty = ir.F32Type.get()
    bits_val = Uint32(bits).ir_value(loc=loc, ip=ip)
    pos = Uint32(bit_pos).ir_value(loc=loc, ip=ip)
    shifted = arith.shrui(bits_val, pos, loc=loc, ip=ip)
    one = arith.constant(i32_ty, 1, loc=loc, ip=ip)
    bit = arith.andi(shifted, one, loc=loc, ip=ip)
    zero_i = arith.constant(i32_ty, 0, loc=loc, ip=ip)
    is_keep = arith.cmpi(arith.CmpIPredicate.ne, bit, zero_i, loc=loc, ip=ip)
    scale_val = Float32(scale).ir_value(loc=loc, ip=ip)
    zero_f = arith.constant(f32_ty, 0.0, loc=loc, ip=ip)
    return Float32(arith.select(is_keep, scale_val, zero_f, loc=loc, ip=ip))


@dsl_user_op
def f32_to_dropout_threshold_byte(p_dropout: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float dropout probability to single threshold byte as Uint32."""
    val = Float32(p_dropout).ir_value(loc=loc, ip=ip)
    i32_ty = ir.IntegerType.get_signless(32)
    f32_ty = ir.F32Type.get()
    scale = arith.constant(f32_ty, 255.0, loc=loc, ip=ip)
    scaled = arith.mulf(val, scale, loc=loc, ip=ip)
    return Uint32(arith.fptoui(i32_ty, scaled, loc=loc, ip=ip))


@dsl_user_op
def philox_extract_mask_bits_fa2(
    rng0: Uint32, rng1: Uint32, rng2: Uint32, rng3: Uint32,
    threshold_byte: Uint32,
    byte_pair_offset: Uint32,
    sub_lane: Uint32,
    *, loc=None, ip=None
) -> Uint32:
    """Extract 8 mask bits from one FA2-convention Philox call and place them
    at the correct positions within a 32-bit mask word.

    For each byte_group bg (0-3), extracts 2 bytes at byte_pair_offset and
    byte_pair_offset+1 from rng_words[bg], compares against threshold, and
    places the 2 result bits at positions bg*8 + sub_lane*2 and bg*8 + sub_lane*2 + 1
    in the output word.
    """
    i32_ty = ir.IntegerType.get_signless(32)
    r = [Uint32(x).ir_value(loc=loc, ip=ip) for x in [rng0, rng1, rng2, rng3]]
    th = Uint32(threshold_byte).ir_value(loc=loc, ip=ip)
    bpo = Uint32(byte_pair_offset).ir_value(loc=loc, ip=ip)
    sl = Uint32(sub_lane).ir_value(loc=loc, ip=ip)

    asm_lines = []
    asm_lines.append(
        "{ .reg .u32 b, sh0, sh1, base, res; .reg .pred p;"
        " mov.u32 res, 0;"
        " shl.b32 sh0, $6, 3;"
        " add.u32 sh1, sh0, 8;"
        " shl.b32 base, $7, 1;"
    )
    regs = ["$1", "$2", "$3", "$4"]
    for bg in range(4):
        bit_even = bg * 8
        bit_odd = bg * 8 + 1
        asm_lines.append(
            f" shr.b32 b, {regs[bg]}, sh0; and.b32 b, b, 255;"
            f" setp.le.u32 p, b, $5; @p or.b32 res, res, {1 << bit_even};"
            f" shr.b32 b, {regs[bg]}, sh1; and.b32 b, b, 255;"
            f" setp.le.u32 p, b, $5; @p or.b32 res, res, {1 << bit_odd};"
        )
    asm_lines.append(" shl.b32 res, res, base; mov.u32 $0, res; }")
    asm_str = "".join(asm_lines)

    result = llvm.inline_asm(
        i32_ty,
        [*r, th, bpo, sl],
        asm_str,
        "=r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def smem_mask_store_u32(
    smem_ptr: cute.Pointer, offset: Int32Type, mask_word: Uint32,
    *, loc=None, ip=None
) -> None:
    """Store a 32-bit mask word to shared memory at smem_ptr + offset * 4."""
    ptr_val = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    off_val = Int32Type(offset).ir_value(loc=loc, ip=ip)
    w_val = Uint32(mask_word).ir_value(loc=loc, ip=ip)
    i32_ty = ir.IntegerType.get_signless(32)

    llvm.inline_asm(
        i32_ty,
        [ptr_val, off_val, w_val],
        "{ .reg .u64 off64, addr;"
        " cvt.u64.s32 off64, $2;"
        " shl.b64 off64, off64, 2;"
        " add.u64 addr, $1, off64;"
        " st.shared.u32 [addr], $3; }",
        "=r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def smem_mask_load_u32(
    smem_ptr: cute.Pointer, offset: Int32Type,
    *, loc=None, ip=None
) -> Uint32:
    """Load a uint32 from shared memory at smem_ptr + offset * 4."""
    ptr_val = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    off_val = Int32Type(offset).ir_value(loc=loc, ip=ip)
    i32_ty = ir.IntegerType.get_signless(32)

    # has_side_effects=True is REQUIRED here. Without it, the compiler treats
    # this load as a pure function and may CSE/hoist it across barriers,
    # causing the consumer to read stale producer data (the very first
    # producer iteration's bits) on every iteration.
    result = llvm.inline_asm(
        i32_ty,
        [ptr_val, off_val],
        "{ .reg .u64 off64, addr;"
        " cvt.u64.s32 off64, $2;"
        " shl.b64 off64, off64, 2;"
        " add.u64 addr, $1, off64;"
        " ld.shared.u32 $0, [addr]; }",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def store_dropout_mask_u8(
    mask_ptr: cute.Pointer, col_offset: Int32Type, mask_val: Float32,
    *, loc=None, ip=None
) -> None:
    """Store a Float32 dropout mask (0.0 or 1.0) as uint8 to global memory."""
    from cutlass.cute.typing import Int32 as _Int32
    ptr_val = mask_ptr.toint(loc=loc, ip=ip).ir_value()
    off_val = _Int32(col_offset).ir_value(loc=loc, ip=ip)
    mask_ir = Float32(mask_val).ir_value(loc=loc, ip=ip)

    i32_ty = ir.IntegerType.get_signless(32)

    llvm.inline_asm(
        i32_ty,
        [ptr_val, off_val, mask_ir],
        "{ .reg .u64 off64, addr; .reg .u32 v;"
        " cvt.u64.s32 off64, $2;"
        " add.u64 addr, $1, off64;"
        " cvt.rzi.u32.f32 v, $3;"
        " st.global.u8 [addr], v; }",
        "=r,l,r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prmt_b32(rng: Uint32, b: Uint32, sel: Uint32, *, loc=None, ip=None) -> Uint32:
    """PTX ``prmt.b32 dst, rng, b, sel``: byte-permute (rng||b) per ``sel``.

    ``sel`` is a 32-bit value with 4 nibbles (low to high = byte 0..3 of dst).
    Each nibble's lower 3 bits select one of the 8 source bytes from
    ``(rng[0..3], b[0..3])``; bit 3 enables sign-extension from the MSB of
    the selected byte. To extract byte ``i`` of ``rng`` into the LSB of the
    result with the other bytes zeroed, pass ``b=0`` and ``sel=0x4440 | i``.
    """
    i32_ty = ir.IntegerType.get_signless(32)
    rng_ir = Uint32(rng).ir_value(loc=loc, ip=ip)
    b_ir = Uint32(b).ir_value(loc=loc, ip=ip)
    sel_ir = Uint32(sel).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        i32_ty,
        [rng_ir, b_ir, sel_ir],
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def apply_dropout_pair(
    val_lo: Float32, val_hi: Float32,
    rng_word: Uint32,
    sel_lo: Uint32, sel_hi: Uint32,
    thresh_byte: Uint32,
    rp_dropout: Float32,
    *, loc=None, ip=None
):
    """Apply dropout + rp_dropout scaling to a pair of Float32 values.

    Returns ``(select(keep_lo, val_lo * rp_dropout, 0),
               select(keep_hi, val_hi * rp_dropout, 0))`` where
    ``keep_x = byte_x <= thresh_byte`` and
    ``byte_x = prmt.b32(rng_word, 0, sel_x) & 0xff``.

    Bit-identical to FA2: ``prmt.b32(rng, 0, 0x4440 | i)`` produces exactly
    the same byte value as ``(rng >> (i*8)) & 0xff``, and the
    ``<= threshold`` comparison is unchanged.
    """
    val_lo_ir = Float32(val_lo).ir_value(loc=loc, ip=ip)
    val_hi_ir = Float32(val_hi).ir_value(loc=loc, ip=ip)
    thresh_byte_ir = Uint32(thresh_byte).ir_value(loc=loc, ip=ip)
    rp_ir = Float32(rp_dropout).ir_value(loc=loc, ip=ip)

    f32_ty = ir.F32Type.get()
    zero_f32 = arith.constant(f32_ty, 0.0, loc=loc, ip=ip)
    zero_u32 = Uint32(0)

    rnd_lo_u = prmt_b32(rng_word, zero_u32, sel_lo, loc=loc, ip=ip)
    rnd_hi_u = prmt_b32(rng_word, zero_u32, sel_hi, loc=loc, ip=ip)
    rnd_lo = Uint32(rnd_lo_u).ir_value(loc=loc, ip=ip)
    rnd_hi = Uint32(rnd_hi_u).ir_value(loc=loc, ip=ip)

    keep_lo = arith.cmpi(arith.CmpIPredicate.ule, rnd_lo, thresh_byte_ir, loc=loc, ip=ip)
    keep_hi = arith.cmpi(arith.CmpIPredicate.ule, rnd_hi, thresh_byte_ir, loc=loc, ip=ip)

    scaled_lo = arith.mulf(val_lo_ir, rp_ir, loc=loc, ip=ip)
    scaled_hi = arith.mulf(val_hi_ir, rp_ir, loc=loc, ip=ip)
    result_lo = arith.select(keep_lo, scaled_lo, zero_f32, loc=loc, ip=ip)
    result_hi = arith.select(keep_hi, scaled_hi, zero_f32, loc=loc, ip=ip)

    return (Float32(result_lo), Float32(result_hi))


@dsl_user_op
def apply_dropout_quad_select_only(
    v0: Float32, v1: Float32, v2: Float32, v3: Float32,
    rng_word: Uint32,
    sel_0: Uint32, sel_1: Uint32, sel_2: Uint32, sel_3: Uint32,
    thresh_byte: Uint32,
    *, loc=None, ip=None
):
    """Dropout select on 4 FP32 values that share one Philox word.

    Computes ``select(byte_i <= thresh_byte, v_i, 0)`` for i in 0..3,
    where ``byte_i = prmt(rng_word, 0, sel_i)``. Used on the
    ``.16x256b`` path where one rng_word covers a (top, bot) row-pair
    times 2 cols == 4 cells, and the per-element ``val * rp`` factor
    has already been baked into ``acc_S`` by exp2 (see ``log2_rp`` in
    ``scale_subtract_rowmax_pair`` and ``inv_rp`` in
    ``apply_exp2_convert_row_sum_pair``), so the select is pure SELP
    with no FMUL.

    Bit-identical to FA2: byte extraction matches PRMT exactly and
    the ``<= threshold`` comparison is unchanged.
    """
    val_irs = [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)]
    thresh_byte_ir = Uint32(thresh_byte).ir_value(loc=loc, ip=ip)
    sel_irs = [Uint32(s) for s in (sel_0, sel_1, sel_2, sel_3)]

    f32_ty = ir.F32Type.get()
    zero_f32 = arith.constant(f32_ty, 0.0, loc=loc, ip=ip)
    zero_u32 = Uint32(0)

    results = []
    for i in range(4):
        rnd_byte_u = prmt_b32(rng_word, zero_u32, sel_irs[i], loc=loc, ip=ip)
        rnd_byte = Uint32(rnd_byte_u).ir_value(loc=loc, ip=ip)
        keep = arith.cmpi(
            arith.CmpIPredicate.ule, rnd_byte, thresh_byte_ir, loc=loc, ip=ip
        )
        results.append(
            Float32(arith.select(keep, val_irs[i], zero_f32, loc=loc, ip=ip))
        )

    return (results[0], results[1], results[2], results[3])


@dsl_user_op
def gen_dropout_mask_f32(
    rng_word: Uint32, byte_shift: Uint32, threshold_packed: Uint32,
    *, loc=None, ip=None
):
    """Generate a pair of Float32 dropout masks using integer byte comparison."""
    rng_val = Uint32(rng_word).ir_value(loc=loc, ip=ip)
    shift_val = Uint32(byte_shift).ir_value(loc=loc, ip=ip)
    thresh_val = Uint32(threshold_packed).ir_value(loc=loc, ip=ip)

    i32_ty = ir.IntegerType.get_signless(32)
    f32_ty = ir.F32Type.get()

    mask_0xff = arith.constant(i32_ty, 0xFF, loc=loc, ip=ip)
    eight = arith.constant(i32_ty, 8, loc=loc, ip=ip)
    one_f32 = arith.constant(f32_ty, 1.0, loc=loc, ip=ip)
    zero_f32 = arith.constant(f32_ty, 0.0, loc=loc, ip=ip)

    thresh_byte = arith.andi(thresh_val, mask_0xff, loc=loc, ip=ip)

    shifted_lo = arith.shrui(rng_val, shift_val, loc=loc, ip=ip)
    rnd_lo = arith.andi(shifted_lo, mask_0xff, loc=loc, ip=ip)

    shift_hi = arith.addi(shift_val, eight, loc=loc, ip=ip)
    shifted_hi = arith.shrui(rng_val, shift_hi, loc=loc, ip=ip)
    rnd_hi = arith.andi(shifted_hi, mask_0xff, loc=loc, ip=ip)

    keep_lo = arith.cmpi(arith.CmpIPredicate.ule, rnd_lo, thresh_byte, loc=loc, ip=ip)
    keep_hi = arith.cmpi(arith.CmpIPredicate.ule, rnd_hi, thresh_byte, loc=loc, ip=ip)

    mask_lo_f32 = arith.select(keep_lo, one_f32, zero_f32, loc=loc, ip=ip)
    mask_hi_f32 = arith.select(keep_hi, one_f32, zero_f32, loc=loc, ip=ip)

    return (Float32(mask_lo_f32), Float32(mask_hi_f32))
