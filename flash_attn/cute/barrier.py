import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def red_relaxed(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_eq(lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        read_val = Int32(0)
        while read_val != val:
            read_val = ld_acquire(flag_ptr)


@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: cutlass.Constexpr[Int32]
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        red_release(flag_ptr, val)
        # red_relaxed(flag_ptr, val)


@dsl_user_op
def atomic_add_acq_rel(
    lock_ptr: cute.Pointer, val: Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    """GPU-scope acq_rel fetch-add; returns the old value (release of this CTA's writes, acquire of others')."""
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    old = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "atom.add.acq_rel.gpu.global.s32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(old)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.acq_rel.gpu;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
