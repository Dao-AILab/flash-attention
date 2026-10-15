# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@cute.jit
def clz(x: Int32) -> Int32:
    # for i in cutlass.range_constexpr(32):
    #     if (1 << (31 - i)) & x:
    #         return Int32(i)
    # return Int32(32)
    # Early exit is not supported yet
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res


@dsl_user_op
def exp2_bf16x2(x0, x1, scale_log2, gauge_log2, masked=False, *, loc=None, ip=None):
    slope = Float32(scale_log2) * Float32(128.0)
    bias = Float32(12582912.0) + Float32(128.0) * (Float32(125.859375) - Float32(gauge_log2))
    mask_asm = (
        "prmt.b32 mask, $3, $4, 0x7632;\n"
        "mov.b32 neg_inf, 0xff80ff80;\n"
        "set.neu.u32.bf16x2 mask, mask, neg_inf;\n"
        "and.b32 pk, pk, mask;\n"
    ) if masked else ""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1, slope, bias)],
        "{\n"
        ".reg .b64 a, b, c, d;\n"
        ".reg .b32 v0, v1, pk, uu, mk, coefficient, mask, neg_inf;\n"
        "mov.b64 a, {$3, $4};\n"
        "mov.b64 b, {$5, $5};\n"
        "mov.b64 c, {$6, $6};\n"
        "fma.rn.f32x2 d, a, b, c;\n"
        "mov.b64 {v0, v1}, d;\n"
        "prmt.b32 pk, v0, v1, 0x5410;\n"
        "mov.b32 mk, 0x00400040;\n"
        "mov.b32 coefficient, 0x3f4b3f4b;\n"
        "add.s16x2 uu, pk, mk;\n"
        "fma.rn.relu.bf16x2 pk, uu, coefficient, pk;\n"
        + mask_asm
        + "mov.b32 $0, pk;\n"
        "shl.b32 v0, pk, 16;\n"
        "and.b32 v1, pk, 0xffff0000;\n"
        "mov.b32 $1, v0;\n"
        "mov.b32 $2, v1;\n"
        "}\n",
        "=r,=f,=f,f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
    )
