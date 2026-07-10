from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import Int32, Uint32, Int64, Uint64

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm

T = ir.Type

PHILOX_NUM_ROUNDS = 7

@dsl_user_op
def mulhilo32(a: Uint32, b: Uint32, *, loc=None, ip=None):
    """32x32 -> 64 bit unsigned multiply. Returns (lo32, hi32) as a tuple of Uint32.

    Uses a single ``mul.wide.u32`` PTX instruction (ptxas lowers it to
    one IMAD.WIDE.U32 SASS inst on SM10x, the same SASS that two
    separate ``mul.lo`` + ``mul.hi`` would produce). The i64 round-trip
    in MLIR is free at the SASS level because the trunc / shr / trunc
    collapse into the IMAD.WIDE's output register pair.
    """
    a_val = Uint32(a).ir_value(loc=loc, ip=ip)
    b_val = Uint32(b).ir_value(loc=loc, ip=ip)

    # Use PTX mul.wide.u32 to get full 64-bit product
    tmp_i64 = llvm.inline_asm(
        ir.IntegerType.get_signless(64),
        [a_val, b_val],
        "mul.wide.u32 $0, $1, $2;",
        "=l,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    # Extract lo and hi 32-bit halves (folded into IMAD.WIDE's output
    # register pair by ptxas; no extra SASS ops emitted).
    i32_ty = ir.IntegerType.get_signless(32)
    lo = arith.trunci(i32_ty, tmp_i64, loc=loc, ip=ip)
    shift = arith.constant(ir.IntegerType.get_signless(64), 32, loc=loc, ip=ip)
    hi_64 = arith.shrui(tmp_i64, shift, loc=loc, ip=ip)
    hi = arith.trunci(i32_ty, hi_64, loc=loc, ip=ip)

    return (Uint32(lo), Uint32(hi))

@dsl_user_op
def xor3_b32(a: Uint32, b: Uint32, c: Uint32, *, loc=None, ip=None):
    """3-input bitwise XOR via PTX ``lop3.b32`` (one ALU op).

    Computes ``a ^ b ^ c`` in a single LOP3.LUT instruction using
    immLut = 0x96 (truth table for the 3-input XOR boolean function:
    ``0xF0 ^ 0xCC ^ 0xAA``). Replaces the canonical ``(a ^ b) ^ c``
    sequence which compiles to two XOR instructions; cuts Philox's
    bit-pipe work by ~30% in the fwd path.

    IMPORTANT: only enable on the fwd path (see ``philox_single_round``
    docstring for the rationale). The bwd path must keep the 2-XOR form
    or it regresses by ~390 μs.
    """
    a_val = Uint32(a).ir_value(loc=loc, ip=ip)
    b_val = Uint32(b).ir_value(loc=loc, ip=ip)
    c_val = Uint32(c).ir_value(loc=loc, ip=ip)
    i32_ty = ir.IntegerType.get_signless(32)
    res = llvm.inline_asm(
        i32_ty,
        [a_val, b_val, c_val],
        "lop3.b32 $0, $1, $2, $3, 0x96;",
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(res)


@dsl_user_op
def philox_single_round(
    ctr_x: Uint32, ctr_y: Uint32, ctr_z: Uint32, ctr_w: Uint32,
    key_x: Uint32, key_y: Uint32,
    *, use_lop3: bool = False, loc=None, ip=None
):
    """One Philox round.

    ``use_lop3=True`` folds each 3-input XOR (``ret_x`` and ``ret_z``)
    into a single LOP3.LUT op. This saves ~15 μs on the fwd kernel but
    REGRESSES the bwd kernel by ~390 μs (training_test config
    B=4,S=8K,Hq=8,Hk=1,D=128,causal=True): bwd issues many more Philox
    chains per thread and the LOP3 path appears to increase register
    pressure / contend on the LOP3 functional unit there. Default is
    therefore False; only fwd softmax call sites pass True.
    """
    kPhiloxSA = Uint32(0xD2511F53)
    kPhiloxSB = Uint32(0xCD9E8D57)

    res0_lo, res0_hi = mulhilo32(kPhiloxSA, ctr_x, loc=loc, ip=ip)
    res1_lo, res1_hi = mulhilo32(kPhiloxSB, ctr_z, loc=loc, ip=ip)

    if use_lop3:
        # ret.x = res1_hi ^ ctr_y ^ key_x (1 LOP3 vs 2 scalar XOR)
        ret_x = xor3_b32(res1_hi, ctr_y, key_x, loc=loc, ip=ip)
        # ret.z = res0_hi ^ ctr_w ^ key_y (1 LOP3 vs 2 scalar XOR)
        ret_z = xor3_b32(res0_hi, ctr_w, key_y, loc=loc, ip=ip)
    else:
        ret_x = res1_hi ^ ctr_y ^ key_x
        ret_z = res0_hi ^ ctr_w ^ key_y
    # ret.y = res1_lo
    ret_y = res1_lo
    # ret.w = res0_lo
    ret_w = res0_lo

    return (ret_x, ret_y, ret_z, ret_w)


@dsl_user_op
def philox_unpack_seed_offset(seed: Uint64, offset: Uint64, *, loc=None, ip=None):
    """Unpack seed and offset into (key_x, key_y, ctr_x, ctr_y) for reuse in loops.

    Call once per block; use philox_rounds in the inner loop to avoid repeated unpack.
    """
    i32_ty = ir.IntegerType.get_signless(32)
    i64_ty = ir.IntegerType.get_signless(64)
    shift32 = arith.constant(i64_ty, 32, loc=loc, ip=ip)

    seed_val = Uint64(seed).ir_value(loc=loc, ip=ip)
    key_x = Uint32(arith.trunci(i32_ty, seed_val, loc=loc, ip=ip))
    key_y = Uint32(arith.trunci(i32_ty, arith.shrui(seed_val, shift32, loc=loc, ip=ip), loc=loc, ip=ip))

    offset_val = Uint64(offset).ir_value(loc=loc, ip=ip)
    ctr_x = Uint32(arith.trunci(i32_ty, offset_val, loc=loc, ip=ip))
    ctr_y = Uint32(arith.trunci(i32_ty, arith.shrui(offset_val, shift32, loc=loc, ip=ip), loc=loc, ip=ip))

    return (key_x, key_y, ctr_x, ctr_y)


@dsl_user_op
def philox_rounds(
    key_x: Uint32, key_y: Uint32, ctr_x: Uint32, ctr_y: Uint32,
    row_id: Uint32, col_id: Uint32,
    *, use_lop3: bool = False, loc=None, ip=None
):
    """N-round Philox core (N = PHILOX_NUM_ROUNDS).

    Use with philox_unpack_seed_offset for loop-invariant seed/offset.
    ``use_lop3`` is forwarded to ``philox_single_round`` — only enable
    on fwd softmax call sites; bwd must use the default (see
    ``philox_single_round`` docstring).
    """
    kPhilox10A = Uint32(0x9E3779B9)
    kPhilox10B = Uint32(0xBB67AE85)

    ctr_z = row_id
    ctr_w = col_id

    for _ in range(PHILOX_NUM_ROUNDS - 1):
        ctr_x, ctr_y, ctr_z, ctr_w = philox_single_round(
            ctr_x, ctr_y, ctr_z, ctr_w, key_x, key_y,
            use_lop3=use_lop3, loc=loc, ip=ip,
        )
        key_x = key_x + kPhilox10A
        key_y = key_y + kPhilox10B

    out_x, out_y, out_z, out_w = philox_single_round(
        ctr_x, ctr_y, ctr_z, ctr_w, key_x, key_y,
        use_lop3=use_lop3, loc=loc, ip=ip,
    )
    return (out_x, out_y, out_z, out_w)


@dsl_user_op
def philox(seed: Uint64, subsequence: Uint64, offset: Uint64, *, loc=None, ip=None):
    kPhilox10A = Uint32(0x9E3779B9)
    kPhilox10B = Uint32(0xBB67AE85)
    i32_ty = ir.IntegerType.get_signless(32)
    i64_ty = ir.IntegerType.get_signless(64)

    seed_val = Uint64(seed).ir_value(loc=loc, ip=ip)
    key_x = Uint32(arith.trunci(i32_ty, seed_val, loc=loc, ip=ip))
    shift32 = arith.constant(i64_ty, 32, loc=loc, ip=ip)
    key_y = Uint32(arith.trunci(i32_ty, arith.shrui(seed_val, shift32, loc=loc, ip=ip), loc=loc, ip=ip))

    offset_val = Uint64(offset).ir_value(loc=loc, ip=ip)
    ctr_x = Uint32(arith.trunci(i32_ty, offset_val, loc=loc, ip=ip))
    ctr_y = Uint32(arith.trunci(i32_ty, arith.shrui(offset_val, shift32, loc=loc, ip=ip), loc=loc, ip=ip))

    subseq_val = Uint64(subsequence).ir_value(loc=loc, ip=ip)
    ctr_z = Uint32(arith.trunci(i32_ty, subseq_val, loc=loc, ip=ip))
    ctr_w = Uint32(arith.trunci(i32_ty, arith.shrui(subseq_val, shift32, loc=loc, ip=ip), loc=loc, ip=ip))

    for _ in range(PHILOX_NUM_ROUNDS - 1):
        ctr_x, ctr_y, ctr_z, ctr_w = philox_single_round(
            ctr_x, ctr_y, ctr_z, ctr_w, key_x, key_y, loc=loc, ip=ip
        )
        key_x = key_x + kPhilox10A
        key_y = key_y + kPhilox10B

    out_x, out_y, out_z, out_w = philox_single_round(
        ctr_x, ctr_y, ctr_z, ctr_w, key_x, key_y, loc=loc, ip=ip
    )

    return (out_x, out_y, out_z, out_w)


@dsl_user_op
def philox_with_row_col(
    seed: Uint64, row_id: Uint32, col_id: Uint32, offset: Uint64,
    *, loc=None, ip=None
):
    """Philox RNG with row_id/col_id as separate Uint32 (avoids Uint64 pack/unpack).

    Semantically equivalent to philox(seed, (col_id << 32) | row_id, offset).
    Always uses 7 rounds. For loops with invariant seed/offset, use philox_unpack_seed_offset + philox_rounds.
    """
    key_x, key_y, ctr_x, ctr_y = philox_unpack_seed_offset(seed, offset, loc=loc, ip=ip)
    return philox_rounds(key_x, key_y, ctr_x, ctr_y, row_id, col_id, loc=loc, ip=ip)
