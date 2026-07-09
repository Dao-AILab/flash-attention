# Copyright (c) 2025, Tri Dao.

import math
import operator
from typing import Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

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


from cutlass import Float32, Int32, Boolean

from quack import layout_utils
import flash_attn.cute.utils as utils
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.philox import (
    philox_unpack_seed_offset,
    philox_rounds,
    philox_precompute_round_keys,
    philox_rounds_with_keys,
)
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import Int32 as Int32Type, Int64, Uint32, Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm


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
def philox_extract_mask_16bits(
    rng0: Uint32, rng1: Uint32, rng2: Uint32, rng3: Uint32,
    threshold_byte: Uint32,
    bit_offset: Uint32,
    *, loc=None, ip=None
) -> Uint32:
    """Extract 16 mask bits from one Philox call using ALL 4 bytes per word.

    Uses the optimized convention: each rng word's 4 bytes produce 4 consecutive
    mask bits. rng0 -> bits 0-3, rng1 -> bits 4-7, rng2 -> bits 8-11, rng3 -> bits 12-15.
    The 16 result bits are shifted left by bit_offset before returning.
    """
    i32_ty = ir.IntegerType.get_signless(32)
    r = [Uint32(x).ir_value(loc=loc, ip=ip) for x in [rng0, rng1, rng2, rng3]]
    th = Uint32(threshold_byte).ir_value(loc=loc, ip=ip)
    boff = Uint32(bit_offset).ir_value(loc=loc, ip=ip)

    asm_lines = []
    asm_lines.append(
        "{ .reg .u32 b, res; .reg .pred p;"
        " mov.u32 res, 0;"
    )
    regs = ["$1", "$2", "$3", "$4"]
    for w in range(4):
        for byte_idx in range(4):
            bit_pos = w * 4 + byte_idx
            shift = byte_idx * 8
            asm_lines.append(
                f" shr.b32 b, {regs[w]}, {shift}; and.b32 b, b, 255;"
                f" setp.le.u32 p, b, $5; @p or.b32 res, res, {1 << bit_pos};"
            )
    asm_lines.append(" shl.b32 res, res, $6; mov.u32 $0, res; }")
    asm_str = "".join(asm_lines)

    result = llvm.inline_asm(
        i32_ty,
        [*r, th, boff],
        asm_str,
        "=r,r,r,r,r,r,r",
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
def smem_mask_store_combined(
    smem_ptr: cute.Pointer, offset: Int32Type, bits_lo: Uint32, bits_hi: Uint32,
    *, loc=None, ip=None
) -> None:
    """Store (bits_hi << 16) | bits_lo to shared memory at smem_ptr + offset * 4."""
    ptr_val = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    off_val = Int32Type(offset).ir_value(loc=loc, ip=ip)
    lo_val = Uint32(bits_lo).ir_value(loc=loc, ip=ip)
    hi_val = Uint32(bits_hi).ir_value(loc=loc, ip=ip)
    i32_ty = ir.IntegerType.get_signless(32)

    llvm.inline_asm(
        i32_ty,
        [ptr_val, off_val, lo_val, hi_val],
        "{ .reg .u64 off64, addr; .reg .u32 combined;"
        " shl.b32 combined, $4, 16;"
        " or.b32 combined, combined, $3;"
        " cvt.u64.s32 off64, $2;"
        " shl.b64 off64, off64, 2;"
        " add.u64 addr, $1, off64;"
        " st.shared.u32 [addr], combined; }",
        "=r,l,r,r,r",
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


from flash_attn.cute.utils import AuxData


@cute.jit
def call_score_mod(
    score_mod: cutlass.Constexpr,
    score,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_data: AuxData,
):
    aux_tensors = aux_data.tensors if aux_data.tensors is not None else ()
    # Compatibility shim for pre-aux_scalars score_mod callables.
    if cutlass.const_expr(aux_data.scalars is not None):
        return score_mod(
            score,
            batch_idx,
            head_idx,
            q_idx=q_idx,
            kv_idx=kv_idx,
            seqlen_info=seqlen_info,
            aux_tensors=aux_tensors,
            aux_scalars=aux_data.scalars,
        )
    return score_mod(
        score,
        batch_idx,
        head_idx,
        q_idx=q_idx,
        kv_idx=kv_idx,
        seqlen_info=seqlen_info,
        aux_tensors=aux_tensors,
    )


@cute.jit
def call_score_mod_bwd(
    score_mod_bwd: cutlass.Constexpr,
    grad,
    score,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_data: AuxData,
):
    aux_tensors = aux_data.tensors if aux_data.tensors is not None else ()
    # Compatibility shim for pre-aux_scalars score_mod_bwd callables.
    if cutlass.const_expr(aux_data.scalars is not None):
        return score_mod_bwd(
            grad,
            score,
            batch_idx,
            head_idx,
            q_idx=q_idx,
            kv_idx=kv_idx,
            seqlen_info=seqlen_info,
            aux_tensors=aux_tensors,
            aux_scalars=aux_data.scalars,
        )
    return score_mod_bwd(
        grad,
        score,
        batch_idx,
        head_idx,
        q_idx=q_idx,
        kv_idx=kv_idx,
        seqlen_info=seqlen_info,
        aux_tensors=aux_tensors,
    )


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        # Each iteration processes one row of acc_S
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            # Update row_max before changing row_max_cur to safe value for -inf
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                # row_scale[r] = cute.math.exp2(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * scale_log2, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        if cutlass.const_expr(sink_val is not None and isinstance(sink_val, cute.Tensor)):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                LOG2_E = math.log2(math.e)
                row_sum[r] += cute.math.exp2(
                    sink_val_cur * LOG2_E - row_max[r] * scale_log2, fastmath=True
                )

            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = 0.0
    is_dropout: cutlass.Constexpr[bool] = False
    m_block_size: cutlass.Constexpr[int] = 128
    n_block_size: cutlass.Constexpr[int] = 128
    max_offset: cutlass.Constexpr[int] = 0

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
        is_dropout: cutlass.Constexpr[bool] = False,
        m_block_size: cutlass.Constexpr[int] = 128,
        n_block_size: cutlass.Constexpr[int] = 128,
        num_rows: cutlass.Constexpr[int] = 1,
        max_offset: cutlass.Constexpr[int] = 0,
    ):
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
            is_dropout=is_dropout,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            max_offset=max_offset,
        )

    @cute.jit
    def compute_row_max_local(self, acc_S_row: cute.TensorSSA, is_first: Boolean) -> Float32:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
        return row_max_new

    @cute.jit
    def update_row_max_from_local(
        self,
        row_max_new: Float32,
        is_first: Boolean,
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def update_row_max(self, acc_S_row: cute.TensorSSA, is_first: int) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        # self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=self.row_sum[0] * row_scale)
        self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)
        # tmp = self._compute_row_sum(acc_S_row_exp)
        # self.row_sum[0] = self.row_sum[0] * row_scale + tmp

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        row_max_scaled = row_max * self.scale_log2
        max_offset = Float32(self.max_offset)
        bias = max_offset - row_max_scaled
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (bias, bias),
            )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                # acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                if cutlass.const_expr(ex2_emu_freq == 0):
                    acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                    acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                else:
                    if cutlass.const_expr(
                        k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                        or j >= frg_cnt - 1
                        or j < ex2_emu_start_frg
                    ):
                        acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                        acc_S_row_frg[k + 1, j] = cute.math.exp2(
                            acc_S_row_frg[k + 1, j], fastmath=True
                        )
                    else:
                        # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.e2e_asm2(acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.ex2_emulation_2(
                            acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                        )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def apply_exp2_convert_row_sum(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        e2e: cutlass.Constexpr[bool] = False,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
        row_scale: Float32 = 1.0,
        is_first: int = False,
    ):
        """exp2(acc_S_row) in-place + bf16 convert + per-row row_sum accumulation.

        ``row_sum`` is accumulated only when ``self.is_dropout`` is True.
        Mirrors the non-dropout path API but with the row_sum update folded
        into the exp2 loop so the kernel does not need to call
        ``update_row_sum`` again after dropout. With dropout enabled the
        post-dropout values would not match raw softmax, so we must
        accumulate the row_sum BEFORE applying the dropout mask.
        """
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        frg_tile = 16
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )

        init_row_sum = self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else 0.0
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, frg_tile, 2):
                if cutlass.const_expr(not e2e):
                    acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                    acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                else:
                    if cutlass.const_expr(
                        k % e2e_freq < e2e_freq - e2e_res or j >= frg_cnt - e2e_frg_limit
                    ):
                        acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                        acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                    else:
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.ex2_emulation_2(
                            acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                        )
                if cutlass.const_expr(self.is_dropout):
                    init_row_sum += (acc_S_row_frg[k, j] + acc_S_row_frg[k + 1, j])
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

        if cutlass.const_expr(self.is_dropout):
            self.row_sum[0] = init_row_sum

    @cute.jit
    def apply_dropout_rP(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        m_block: Int32 = 0,
        n_block: Int32 = 0,
        rng_seed: Uint64 | None = None,
        rng_offset: Uint64 | None = None,
        p_dropout_8bit_packed: Uint32 | None = None,
        rp_dropout: Float32 | None = None,
        mask_row_ptr: cute.Pointer | None = None,
        mask_row_stride: Int32 | None = None,
        mask_row_valid: bool = True,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    ):
        """Apply dropout mask and rp_dropout scaling on the converted P tensor.

        Uses FA2-compatible Philox convention so that given the same RNG state,
        FA4 and FA2 produce bit-identical dropout masks.
        """
        assert cute.size(acc_S_row.shape) % 2 == 0
        total_cols = cute.size(acc_S_row)
        acc_S_row_2 = cute.logical_divide(acc_S_row, cute.make_layout(2))
        acc_S_row_converted_2 = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(2)
        )

        lane_idx_i32 = Int32(cute.arch.lane_idx())
        packed_row = m_block * self.m_block_size + Int32(
            cute.arch.warp_idx() % (self.m_block_size // 32)
        ) * 32 + lane_idx_i32

        if cutlass.const_expr(qhead_per_kvhead > 1):
            actual_seq_row = packed_row // qhead_per_kvhead
            q_head_offset = packed_row - actual_seq_row * qhead_per_kvhead
            rng_offset_adjusted = rng_offset + Uint64(q_head_offset * 32)
        else:
            actual_seq_row = packed_row
            rng_offset_adjusted = rng_offset

        key_x, key_y, base_ctr_x, base_ctr_y = philox_unpack_seed_offset(rng_seed, rng_offset_adjusted)

        fa2_row_arg = Uint32(actual_seq_row // 16)
        fa2_base_lane = Uint32((actual_seq_row % 8) * 4)
        fa2_byte_pair_offset = Uint32(((actual_seq_row // 8) % 2) * 2)

        n_blk_size = total_cols
        base_col = n_block * n_blk_size

        # Precompute prmt.b32 byte-select masks once per row entry. Selectors
        # are lane-uniform within an 8-lane group and constant across the
        # sub_lane / col_arg_local / byte_group inner loops.
        prmt_sel_lo = Uint32(0x4440) | fa2_byte_pair_offset
        prmt_sel_hi = Uint32(0x4441) | fa2_byte_pair_offset

        # Hoist the dropout-threshold byte extraction (low 8 bits only).
        thresh_byte = Uint32(p_dropout_8bit_packed) & Uint32(0xFF)

        for sub_lane in cutlass.range_constexpr(4):
            ctr_x_lane = base_ctr_x + fa2_base_lane + Uint32(sub_lane)

            for col_arg_local in cutlass.range_constexpr(total_cols // 32):
                fa2_col_arg = Uint32(base_col // 32 + col_arg_local)
                rng0, rng1, rng2, rng3 = philox_rounds(
                    key_x, key_y, ctr_x_lane, base_ctr_y,
                    fa2_row_arg,
                    fa2_col_arg,
                    use_lop3=True,
                )
                rng_words = (rng0, rng1, rng2, rng3)
                for byte_group in cutlass.range_constexpr(4):
                    rng_word = rng_words[byte_group]
                    phys_col_0 = col_arg_local * 32 + byte_group * 8 + sub_lane * 2
                    elem_idx = phys_col_0 // 2
                    acc_S_row_2[0, elem_idx], acc_S_row_2[1, elem_idx] = apply_dropout_pair(
                        acc_S_row_2[0, elem_idx], acc_S_row_2[1, elem_idx],
                        rng_word, prmt_sel_lo, prmt_sel_hi,
                        thresh_byte, rp_dropout,
                    )
                    if cutlass.const_expr(mask_row_ptr is not None):
                        if mask_row_valid:
                            byte_shift = fa2_byte_pair_offset * Uint32(8)
                            mask_lo, mask_hi = gen_dropout_mask_f32(
                                rng_word, byte_shift, p_dropout_8bit_packed
                            )
                            col = base_col + phys_col_0
                            if cutlass.const_expr(mask_row_stride is not None):
                                if col < mask_row_stride:
                                    store_dropout_mask_u8(mask_row_ptr, col, mask_lo)
                                if col + 1 < mask_row_stride:
                                    store_dropout_mask_u8(mask_row_ptr, col + 1, mask_hi)
                            else:
                                store_dropout_mask_u8(mask_row_ptr, col, mask_lo)
                                store_dropout_mask_u8(mask_row_ptr, col + 1, mask_hi)

        for i in cutlass.range_constexpr(total_cols // 2):
            acc_S_row_converted_2[None, i].store(
                acc_S_row_2[None, i].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def update_row_max_pair(
        self,
        acc_S: cute.Tensor,
        is_first: int,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        """Update self.row_max[0..3] over the 4-row lane fragment (.16x256b path).

        Mirrors the 1-row ``update_row_max`` API but per-row; returns
        (row_max_safe[0..3], acc_scale[0..3]) tensors.

        Fragment layout (Rep=4 split-P ST aligned, am-first stride):
        outer (am=2, an=4) with strides (16, 32); inner (2, 2, 4) strides
        (1, 2, 4). Flat cell index for (am=h, an, rep, i0, i1) is
        ``h*16 + an*32 + rep*4 + i0 + 2*i1``. The cells at ``base + 0..3``
        for ``base = h*16 + an*32 + rep*4`` are (top col c, top col c+1,
        bot col c, bot col c+1).
        """
        total = cute.size(acc_S)
        R_load: cutlass.Constexpr[int] = total // 8

        row_max_local = cute.make_rmem_tensor(4, Float32)
        for r in cutlass.range_constexpr(4):
            if cutlass.const_expr(is_first):
                row_max_local[r] = -Float32.inf
            else:
                row_max_local[r] = self.row_max[r]

        for h in cutlass.range_constexpr(2):
            top_slot: cutlass.Constexpr[int] = 2 * h
            bot_slot: cutlass.Constexpr[int] = 2 * h + 1
            for g in cutlass.range_constexpr(R_load):
                an: cutlass.Constexpr[int] = g // 4
                rep: cutlass.Constexpr[int] = g % 4
                base = h * 16 + an * 32 + rep * 4
                row_max_local[top_slot] = utils.fmax(
                    row_max_local[top_slot],
                    acc_S[base + 0],
                    acc_S[base + 1],
                )
                row_max_local[bot_slot] = utils.fmax(
                    row_max_local[bot_slot],
                    acc_S[base + 2],
                    acc_S[base + 3],
                )

        for r in cutlass.range_constexpr(4):
            row_max_local[r] = cute.arch.warp_reduction_max(
                row_max_local[r], threads_in_group=4
            )

        row_max_safe = cute.make_rmem_tensor(4, Float32)
        acc_scale = cute.make_rmem_tensor(4, Float32)
        for r in cutlass.range_constexpr(4):
            if cutlass.const_expr(is_first):
                row_max_safe[r] = (
                    row_max_local[r] if row_max_local[r] != -Float32.inf else Float32(0.0)
                )
                acc_scale[r] = Float32(0.0)
                self.row_max[r] = row_max_local[r]
            else:
                row_max_old = self.row_max[r]
                row_max_safe[r] = (
                    row_max_local[r] if row_max_local[r] != -Float32.inf else Float32(0.0)
                )
                scale_arg = (row_max_old - row_max_safe[r]) * self.scale_log2
                acc_scale[r] = cute.math.exp2(scale_arg, fastmath=True)
                if cutlass.const_expr(self.rescale_threshold > 0.0):
                    if scale_arg >= -self.rescale_threshold:
                        row_max_local[r] = row_max_old
                        row_max_safe[r] = row_max_old
                        acc_scale[r] = Float32(1.0)
                self.row_max[r] = row_max_local[r]

        return row_max_safe, acc_scale

    @cute.jit
    def scale_subtract_rowmax_pair(
        self,
        acc_S: cute.Tensor,
        row_max_safe: cute.Tensor,
        log2_rp: Float32 | None = None,
    ):
        """In-place fma: acc_S[i] = acc_S[i] * scale_log2 - row_max_safe[row(i)] * scale_log2.

        Uses ``fma_packed_f32x2`` to fuse (col c, c+1) of the same row.

        When ``log2_rp`` is given (dropout fastpath), it is folded into
        the FMA bias as ``-row_max_safe*scale_log2 + log2_rp`` so the
        subsequent ``exp2`` emits ``softmax_j * rp`` directly. This lets
        ``apply_dropout_rP_pair`` skip the per-element ``* rp`` FMUL
        (saves 128 FMUL per softmax_step per lane on the .16x256b path).
        """
        total = cute.size(acc_S)
        R_load: cutlass.Constexpr[int] = total // 8

        scale_log2 = self.scale_log2

        neg_rms_x_scale = cute.make_rmem_tensor(4, Float32)
        for r in cutlass.range_constexpr(4):
            if cutlass.const_expr(log2_rp is not None):
                neg_rms_x_scale[r] = -row_max_safe[r] * scale_log2 + log2_rp
            else:
                neg_rms_x_scale[r] = -row_max_safe[r] * scale_log2

        for h in cutlass.range_constexpr(2):
            top_slot: cutlass.Constexpr[int] = 2 * h
            bot_slot: cutlass.Constexpr[int] = 2 * h + 1
            top_nrm = neg_rms_x_scale[top_slot]
            bot_nrm = neg_rms_x_scale[bot_slot]
            for g in cutlass.range_constexpr(R_load):
                an: cutlass.Constexpr[int] = g // 4
                rep: cutlass.Constexpr[int] = g % 4
                base = h * 16 + an * 32 + rep * 4
                acc_S[base + 0], acc_S[base + 1] = cute.arch.fma_packed_f32x2(
                    (acc_S[base + 0], acc_S[base + 1]),
                    (scale_log2, scale_log2),
                    (top_nrm, top_nrm),
                )
                acc_S[base + 2], acc_S[base + 3] = cute.arch.fma_packed_f32x2(
                    (acc_S[base + 2], acc_S[base + 3]),
                    (scale_log2, scale_log2),
                    (bot_nrm, bot_nrm),
                )

    @cute.jit
    def apply_exp2_convert_row_sum_pair(
        self,
        acc_S: cute.Tensor,
        acc_S_converted: cute.Tensor,
        row_scale: cute.Tensor,
        is_first: int,
        skip_convert: cutlass.Constexpr[bool] = False,
        inv_rp: Float32 | None = None,
        ex2_emu_mask: cutlass.Constexpr[int] = 0,
        ex2_emu_poly_degree: cutlass.Constexpr[int] = 3,
    ):
        """exp2(acc_S) in-place + bf16 convert + per-row row_sum accumulation.

        row_sum update only when ``self.is_dropout`` (matches the 1-row API
        where non-dropout paths use a separate ``update_row_sum`` call).

        When ``skip_convert`` is True the FP32->BF16 conversion is elided;
        callers that immediately follow with ``apply_dropout_rP_pair``
        (which overwrites acc_S then performs its own conversion) should
        set this to avoid a redundant conversion pass.

        When ``inv_rp`` is given (paired with ``log2_rp`` baked into the
        preceding ``scale_subtract_rowmax_pair``), exp2 emits
        ``softmax_j * rp`` instead of raw ``softmax_j``. The row_sum
        accumulation then folds the ``inv_rp = 1 - p`` factor back in via
        two ``fma_packed_f32x2`` ops so ``self.row_sum`` still tracks the
        raw softmax sum used for the final O / row_sum normalization.

        ``ex2_emu_mask`` (constexpr, 0..0xFF) is the per-g selector for
        selective hardware/software exp2: bit ``g`` set => the ``g``-th
        iteration of the pair loop uses the FMA-pipe software polynomial
        path; bit cleared => the XU-pipe hardware ``MUFU.EX2``.
        """
        total = cute.size(acc_S)
        R_load: cutlass.Constexpr[int] = total // 8

        row_sum_local = cute.make_rmem_tensor(4, Float32)
        if cutlass.const_expr(self.is_dropout):
            for r in cutlass.range_constexpr(4):
                if cutlass.const_expr(is_first):
                    row_sum_local[r] = Float32(0.0)
                else:
                    row_sum_local[r] = self.row_sum[r] * row_scale[r]

        for h in cutlass.range_constexpr(2):
            top_slot: cutlass.Constexpr[int] = 2 * h
            bot_slot: cutlass.Constexpr[int] = 2 * h + 1
            for g in cutlass.range_constexpr(R_load):
                an: cutlass.Constexpr[int] = g // 4
                rep: cutlass.Constexpr[int] = g % 4
                base = h * 16 + an * 32 + rep * 4
                _g_sw: cutlass.Constexpr[bool] = (
                    (ex2_emu_mask >> g) & 1
                ) == 1
                if cutlass.const_expr(_g_sw):
                    e0, e1 = utils.ex2_emulation_2(
                        acc_S[base + 0], acc_S[base + 1],
                        poly_degree=ex2_emu_poly_degree,
                    )
                    e2, e3 = utils.ex2_emulation_2(
                        acc_S[base + 2], acc_S[base + 3],
                        poly_degree=ex2_emu_poly_degree,
                    )
                else:
                    e0 = cute.math.exp2(acc_S[base + 0], fastmath=True)
                    e1 = cute.math.exp2(acc_S[base + 1], fastmath=True)
                    e2 = cute.math.exp2(acc_S[base + 2], fastmath=True)
                    e3 = cute.math.exp2(acc_S[base + 3], fastmath=True)
                acc_S[base + 0] = e0
                acc_S[base + 1] = e1
                acc_S[base + 2] = e2
                acc_S[base + 3] = e3
                if cutlass.const_expr(self.is_dropout):
                    if cutlass.const_expr(inv_rp is not None):
                        (
                            row_sum_local[top_slot],
                            row_sum_local[bot_slot],
                        ) = cute.arch.fma_packed_f32x2(
                            (e0, e2),
                            (inv_rp, inv_rp),
                            (row_sum_local[top_slot], row_sum_local[bot_slot]),
                        )
                        (
                            row_sum_local[top_slot],
                            row_sum_local[bot_slot],
                        ) = cute.arch.fma_packed_f32x2(
                            (e1, e3),
                            (inv_rp, inv_rp),
                            (row_sum_local[top_slot], row_sum_local[bot_slot]),
                        )
                    else:
                        sum_top, sum_bot = cute.arch.add_packed_f32x2(
                            (e0, e2), (e1, e3)
                        )
                        (
                            row_sum_local[top_slot],
                            row_sum_local[bot_slot],
                        ) = cute.arch.add_packed_f32x2(
                            (row_sum_local[top_slot], row_sum_local[bot_slot]),
                            (sum_top, sum_bot),
                        )

        if cutlass.const_expr(not skip_convert):
            assert cute.size(acc_S) % 4 == 0
            acc_S_frg = cute.logical_divide(acc_S, cute.make_layout(4))
            acc_S_converted_frg = cute.logical_divide(
                acc_S_converted, cute.make_layout(4)
            )
            n_frg: cutlass.Constexpr[int] = cute.size(acc_S) // 4
            for j in cutlass.range_constexpr(n_frg):
                acc_S_converted_frg[None, j].store(
                    acc_S_frg[None, j].load().to(acc_S_converted.element_type)
                )

        if cutlass.const_expr(self.is_dropout):
            for r in cutlass.range_constexpr(4):
                self.row_sum[r] = row_sum_local[r]

    @cute.jit
    def apply_dropout_rP_pair(
        self,
        acc_S: cute.Tensor,
        acc_S_converted: cute.Tensor,
        m_block: Int32 = 0,
        n_block: Int32 = 0,
        rng_seed: Uint64 | None = None,
        rng_offset: Uint64 | None = None,
        p_dropout_8bit_packed: Uint32 | None = None,
        rp_dropout: Float32 | None = None,
        mDropoutMask=None,
        mask_row_stride: Int32 | None = None,
        batch_idx: Int32 = 0,
        head_idx: Int32 = 0,
        seqlen_q: Int32 = 0,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    ):
        """100%-util FA2 bit-identical dropout for the ``.16x256b`` 4-row lane fragment.

        Per (row_pair h, col_arg_local) -> 1 Philox call; all 4 bytes of
        each of the 4 rng words are used:
          bytes 0, 1 -> top row R cols (c, c+1)
          bytes 2, 3 -> bot row R+8 cols (c, c+1)
        Bit-identical to FA2 because the (row_arg, col_arg, ctr_x_lane)
        triple matches the FA2 convention.

        Compared with ``.32x32b`` 1-row dropout, philox calls per lane
        drop by 2x (no separate sub_lane loop) since the 4 lanes in a
        row-pair group already have distinct ctr_x_lane via
        lane_idx % 4 -> sub_lane.
        """
        total = cute.size(acc_S)
        R_load: cutlass.Constexpr[int] = total // 8
        n_blk_size: cutlass.Constexpr[int] = self.n_block_size
        base_col = n_block * n_blk_size

        lane_idx_i32 = Int32(cute.arch.lane_idx())
        sub_lane = Uint32(lane_idx_i32) & Uint32(0x3)
        group_idx = lane_idx_i32 >> 2  # 0..7
        warp_idx_in_wg = Int32(cute.arch.warp_idx() % (self.m_block_size // 32))

        fa2_base_lane = Uint32(group_idx) * Uint32(4)

        sel_top_lo = Uint32(0x4440)
        sel_top_hi = Uint32(0x4441)
        sel_bot_lo = Uint32(0x4442)
        sel_bot_hi = Uint32(0x4443)

        thresh_byte = Uint32(p_dropout_8bit_packed) & Uint32(0xFF)

        # pack_gqa > 1 path is intentionally not supported here: a lane's
        # 4 active rows in the .16x256b layout can straddle q-head buckets
        # when qhead_per_kvhead < 32, which would break the 1-philox-per-
        # row-pair sharing. The caller gates pack_gqa back to the .32x32b
        # path.
        n_col_arg_local: cutlass.Constexpr[int] = R_load // 4

        key_x, key_y, base_ctr_x, base_ctr_y = philox_unpack_seed_offset(
            rng_seed, rng_offset
        )
        ctr_x_lane = base_ctr_x + fa2_base_lane + sub_lane

        for h in cutlass.range_constexpr(2):
            top_row = (
                m_block * self.m_block_size
                + warp_idx_in_wg * 32
                + h * 16
                + group_idx
            )
            bot_row = top_row + 8

            fa2_row_arg = Uint32(top_row // 16)

            for col_arg_local in cutlass.range_constexpr(n_col_arg_local):
                fa2_col_arg = Uint32(base_col // 32 + col_arg_local)

                rng0, rng1, rng2, rng3 = philox_rounds(
                    key_x,
                    key_y,
                    ctr_x_lane,
                    base_ctr_y,
                    fa2_row_arg,
                    fa2_col_arg,
                    use_lop3=True,
                )
                rng_words = (rng0, rng1, rng2, rng3)

                for byte_group in cutlass.range_constexpr(4):
                    rng_word = rng_words[byte_group]
                    base = h * 16 + col_arg_local * 32 + byte_group * 4

                    (
                        acc_S[base + 0], acc_S[base + 1],
                        acc_S[base + 2], acc_S[base + 3],
                    ) = apply_dropout_quad_select_only(
                        acc_S[base + 0], acc_S[base + 1],
                        acc_S[base + 2], acc_S[base + 3],
                        rng_word,
                        sel_top_lo, sel_top_hi, sel_bot_lo, sel_bot_hi,
                        thresh_byte,
                    )

                    if cutlass.const_expr(mDropoutMask is not None):
                        phys_col_0 = (
                            col_arg_local * 32 + byte_group * 8 + Int32(sub_lane) * 2
                        )
                        col = base_col + phys_col_0
                        for which in cutlass.range_constexpr(2):
                            row_for_mask = (
                                top_row if cutlass.const_expr(which == 0) else bot_row
                            )
                            byte_shift = Uint32(which * 2 * 8)
                            mask_lo, mask_hi = gen_dropout_mask_f32(
                                rng_word, byte_shift, p_dropout_8bit_packed
                            )
                            mask_row_valid = row_for_mask < seqlen_q
                            bh_off = Int64(
                                (batch_idx * mDropoutMask.shape[1] + head_idx)
                                * mDropoutMask.shape[2]
                            )
                            mask_row_ptr = mDropoutMask.iterator + (
                                bh_off + Int64(row_for_mask)
                            ) * Int64(mask_row_stride)
                            if mask_row_valid:
                                if cutlass.const_expr(mask_row_stride is not None):
                                    if col < mask_row_stride:
                                        store_dropout_mask_u8(mask_row_ptr, col, mask_lo)
                                    if col + 1 < mask_row_stride:
                                        store_dropout_mask_u8(
                                            mask_row_ptr, col + 1, mask_hi
                                        )
                                else:
                                    store_dropout_mask_u8(mask_row_ptr, col, mask_lo)
                                    store_dropout_mask_u8(
                                        mask_row_ptr, col + 1, mask_hi
                                    )

        assert cute.size(acc_S) % 4 == 0
        acc_S_frg = cute.logical_divide(acc_S, cute.make_layout(4))
        acc_S_converted_frg = cute.logical_divide(
            acc_S_converted, cute.make_layout(4)
        )
        n_frg: cutlass.Constexpr[int] = cute.size(acc_S) // 4
        for j in cutlass.range_constexpr(n_frg):
            acc_S_converted_frg[None, j].store(
                acc_S_frg[None, j].load().to(acc_S_converted.element_type)
            )

    @cute.jit
    def scale_apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
        minus_row_max_scaled = -row_max * self.scale_log2
        for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (minus_row_max_scaled, minus_row_max_scaled),
            )

        # for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
        #     acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
        #         (acc_S_row[i], acc_S_row[i + 1]),
        #         (self.scale_log2, self.scale_log2),
        #         (minus_row_max_scaled, minus_row_max_scaled),
        #     )
        #     acc_S_row[i] = cute.math.exp2(acc_S_row[i], fastmath=True)
        #     acc_S_row[i + 1] = cute.math.exp2(acc_S_row[i + 1], fastmath=True)

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                #     cute.arch.fma_packed_f32x2(
                #         (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                #         (self.scale_log2, self.scale_log2),
                #         (minus_row_max_scaled, minus_row_max_scaled),
                #     )
                # )
                # acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                # acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
                acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )


@cute.jit
def floor_if_packed(
    q_idx,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Convert q_idx to packed format for Pack-GQA."""
    if cutlass.const_expr(qhead_per_kvhead == 1):
        return q_idx
    return q_idx // qhead_per_kvhead


@cute.jit
def apply_score_mod_inner(
    score_tensor,
    index_tensor,
    score_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_data: AuxData,
    fastdiv_mods,
    seqlen_info: SeqlenInfoQK,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
):
    """Shared implementation for applying score modification.

    Args:
        score_tensor: The scores to modify (acc_S for flash_fwd, tSrS_t2r for sm100)
        index_tensor: Index positions (tScS for flash_fwd, tScS_t2r for sm100)
        score_mod: The score modification function to apply
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        aux_scalars: Optional runtime scalar captures for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        seqlen_info: Sequence length info
        constant_q_idx: If provided, use this constant for all q_idx values
                        If None, compute q_idx per-element
        qhead_per_kvhead_packgqa: Pack-GQA replication factor. Divide q_idx by this
                                  when greater than 1 so score mods see logical heads.
        transpose_indices: If True, swap q_idx/kv_idx in index_tensor (for bwd kernel where S is transposed)
    """
    # Index positions in the index_tensor tuple
    # Forward: index_tensor[...][0] = q_idx, index_tensor[...][1] = kv_idx
    # Backward (transposed): index_tensor[...][0] = kv_idx, index_tensor[...][1] = q_idx
    if cutlass.const_expr(transpose_indices):
        q_idx_pos = cutlass.const_expr(1)
        kv_idx_pos = cutlass.const_expr(0)
    else:
        q_idx_pos = cutlass.const_expr(0)
        kv_idx_pos = cutlass.const_expr(1)

    n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # SSA values for batch (constant across all elements)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to((vec_size,))

    # Handle q_idx based on whether it's constant
    q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # For Pack-GQA with non-constant q_idx, we need per-element head indices
    # since a thread may process multiple query head indices
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            score_vec[j] = score_tensor[i + j] * softmax_scale

            # Extract head offset from packed q_idx for Pack-GQA
            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][q_idx_pos]
                # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            # If we will do loads we mod, in order to not read OOB
            if cutlass.const_expr(aux_data.tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][q_idx_pos], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = divmod(index_tensor[i + j][kv_idx_pos], seqlen_k_divmod)
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                # No bounds checking - direct indexing
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(index_tensor[i + j][q_idx_pos], qhead_per_kvhead)
                kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

        # Convert to SSA for score_mod call
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()
        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            # NB we do not apply Pack-GQA division here, as constant_q_idx is assumed to already be logical
            q_idx_const = constant_q_idx
            q_idx_ssa = utils.scalar_to_ssa(q_idx_const, cutlass.Int32).broadcast_to((vec_size,))

        # Compute head_idx_ssa: per-element for Pack-GQA with non-constant q_idx, constant otherwise
        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to((vec_size,))

        post_mod_scores = call_score_mod(
            score_mod,
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx_ssa,
            kv_idx_ssa,
            seqlen_info,
            aux_data,
        )

        # Write back modified scores
        score_vec.store(post_mod_scores)
        for j in cutlass.range(vec_size, unroll_full=True):
            score_tensor[i + j] = score_vec[j]


@cute.jit
def apply_score_mod_bwd_inner(
    grad_tensor,
    score_tensor,
    index_tensor,
    score_mod_bwd: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_data: AuxData,
    fastdiv_mods,
    seqlen_info,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    transpose_indices: cutlass.Constexpr[bool] = False,
):
    """Apply backward score modification (joint graph).

    Args:
        grad_tensor: in/out: dlogits rewritten in-place with d(scaled_scores)
        score_tensor: pre-mod scores (unscaled QK tile), scaled by softmax_scale internally
        index_tensor: Index positions (same as forward)
        score_mod_bwd: The backward score modification function (joint graph)
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply to score_tensor
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        aux_scalars: Optional runtime scalar captures for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        seqlen_info: Sequence length info
        constant_q_idx: If provided, use this constant for all q_idx values
        qhead_per_kvhead: Pack-GQA replication factor
        transpose_indices: If True, swap q_idx/kv_idx in index_tensor
    """
    # Index positions in the index_tensor tuple
    # Forward: index_tensor[...][0] = q_idx, index_tensor[...][1] = kv_idx
    # Backward (transposed): index_tensor[...][0] = kv_idx, index_tensor[...][1] = q_idx
    if cutlass.const_expr(transpose_indices):
        q_idx_pos = cutlass.const_expr(1)
        kv_idx_pos = cutlass.const_expr(0)
    else:
        q_idx_pos = cutlass.const_expr(0)
        kv_idx_pos = cutlass.const_expr(1)
    n_vals = cutlass.const_expr(cute.size(grad_tensor.shape))
    grad_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to((vec_size,))
    q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # For Pack-GQA with non-constant q_idx, we need per-element head indices
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            grad_vec[j] = grad_tensor[i + j]
            # Scale score so joint graph sees same value as forward score_mod
            score_vec[j] = score_tensor[i + j] * softmax_scale

            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][q_idx_pos]
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            if cutlass.const_expr(aux_data.tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][q_idx_pos], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = divmod(q_idx_floored, seqlen_q_divmod)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = divmod(index_tensor[i + j][kv_idx_pos], seqlen_k_divmod)
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                # No bounds checking - direct indexing
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(index_tensor[i + j][q_idx_pos], qhead_per_kvhead)
                kv_idx_vec[j] = index_tensor[i + j][kv_idx_pos]

        grad_ssa = grad_vec.load()
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()

        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            q_idx_ssa = utils.scalar_to_ssa(constant_q_idx, cutlass.Int32).broadcast_to((vec_size,))

        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to((vec_size,))

        grad_out_ssa = call_score_mod_bwd(
            score_mod_bwd,
            grad_ssa,
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx_ssa,
            kv_idx_ssa,
            seqlen_info,
            aux_data,
        )

        grad_vec.store(grad_out_ssa)
        for j in cutlass.range(vec_size, unroll_full=True):
            grad_tensor[i + j] = grad_vec[j]
