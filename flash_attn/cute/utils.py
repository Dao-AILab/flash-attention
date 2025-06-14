# Copyright (c) 2025, Tri Dao.

import math
from typing import Type, Callable, Optional

import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack


def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if swapAB:
        return make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy(
            copy_atom,
            layout_tv=tiled_mma.tv_layout_A_tiled,
            tiler_mn=(tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(2)),
        )


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if swapAB:
        return make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy(
            copy_atom,
            layout_tv=tiled_mma.tv_layout_B_tiled,
            tiler_mn=(tiled_mma.get_tile_size(1), tiled_mma.get_tile_size(2)),
        )


def make_tiled_copy_C(copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
    return cute.make_tiled_copy(
        copy_atom,
        layout_tv=tiled_mma.tv_layout_C_tiled,
        tiler_mn=(tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(1)),
    )


def mma_make_fragment_A(
    smem: cute.Tensor, thr_mma: cute.core.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if swapAB:
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor, thr_mma: cute.core.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if swapAB:
        return mma_make_fragment_A(smem, thr_mma)
    else:
        return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


def get_smem_store_atom(arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric]) -> cute.CopyAtom:
    if arch < 90:
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), element_type, num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), element_type,
        )



def max_constexpr(
    a: cutlass.Constexpr[cute.Numeric], b: cutlass.Constexpr[cute.Numeric]
) -> cutlass.Constexpr[cute.Numeric]:
    return a if a > b else b


def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE
) -> cute.TensorSSA | cute.Numeric:
    if isinstance(val, cute.TensorSSA):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in range(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in range(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
            (acc_layout_col_major.shape[0][0], *acc_layout_col_major.shape[0][2:], acc_layout_col_major.shape[2]),  # MMA_N
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
            (acc_layout_col_major.stride[0][0], *acc_layout_col_major.stride[0][2:], acc_layout_col_major.stride[2]),  # MMA_N
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # Due to the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
    acc_layout_divided = cute.logical_divide(acc_layout, (None, None, 2))
    rA_mma_view = cute.make_layout(
        (
            (acc_layout_divided.shape[0], acc_layout_divided.shape[2][0]),
            acc_layout_divided.shape[1],
            acc_layout_divided.shape[2][1],
        ),
        stride=(
            (acc_layout_divided.stride[0], acc_layout_divided.stride[2][0]),
            acc_layout_divided.stride[1],
            acc_layout_divided.stride[2][1],
        ),
    )
    return rA_mma_view


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem.
    """
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def exp2f(x: cute.TensorSSA | cutlass.Float32) -> cute.TensorSSA | cutlass.Float32:
    """exp2f calculation for both vector and scalar.

    :param x: input value
    :type x: cute.TensorSSA or cutlass.Float32
    :return: exp2 value
    :rtype: cute.TensorSSA or cutlass.Float32
    """
    if isinstance(x, cute.TensorSSA):
        res = cute.make_fragment(x.shape, cutlass.Float32)
        res.store(x)
        for i in range(cute.size(x.shape)):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)


@dsl_user_op
def log2f(a: float | cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def atomic_add_fp32(
    a: float | cutlass.Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    # gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    # # cache_hint = cutlass.Int64(0x12F0000000000000)
    # llvm.inline_asm(
    #     None,
    #     [gmem_ptr_i64, cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
    #     # [gmem_ptr_i64, cutlass.Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
    #     "red.global.add.f32 [$0], $1;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, 0x12F0000000000000;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, $2;",
    #     "l,f",
    #     # "l,f,l",
    #     has_side_effects=True,
    #     is_align_stack=False,
    #     asm_dialect=llvm.AsmDialect.AD_ATT,
    # )
    nvvm.atomicrmw(
        res=T.f32(),
        op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=cutlass.Float32(a).ir_value()
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(
        cute.make_layout(
            (tAcA.shape[0][1], cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in range(tApA.shape[0]):
        for rest_k in range(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@dsl_user_op
def barrier_sync(barrier_id: int | cutlass.Int32, number_of_threads: int | cutlass.Int32,
                 *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [cutlass.Int32(barrier_id).ir_value(loc=loc, ip=ip), cutlass.Int32(number_of_threads).ir_value(loc=loc, ip=ip)],
        "bar.sync $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def barrier_arrive(barrier_id: int | cutlass.Int32, number_of_threads: int | cutlass.Int32, *, loc=None, ip=None) -> None:
    """
    Arrive at a named barrier.
    """
    barrier_id = cutlass.Int32(barrier_id).ir_value(loc=loc, ip=ip)
    number_of_threads = cutlass.Int32(number_of_threads).ir_value(loc=loc, ip=ip)
    nvvm.barrier_arrive(
        barrier_id=barrier_id, number_of_threads=number_of_threads, loc=loc, ip=ip
    )
    # llvm.inline_asm(
    #     None,
    #     [barrier_id, number_of_threads],
    #     "bar.arrive $0, $1;",
    #     "r,r",
    #     has_side_effects=True,
    #     is_align_stack=False,
    #     asm_dialect=llvm.AsmDialect.AD_ATT,
    # )


@dsl_user_op
def cp_async_mbarrier_arrive_shared(
    mbar_ptr: cute.Pointer, noinc: bool = False, *, loc=None, ip=None
) -> None:
    nvvm.cp_async_mbarrier_arrive_shared(
        mbar_ptr.llvm_ptr,
        noinc=noinc,
        loc=loc,
        ip=ip,
    )


def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if cutlass.const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx


# @dsl_user_op
# def warp_vote_any_lt(a: float | cutlass.Float32, b: float | cutlass.Float32, *, loc=None, ip=None) -> cutlass.Boolean:
#     mask = cutlass.Int32(-1)
#     return cutlass.Boolean(
#         llvm.inline_asm(
#             T.i32(),
#             [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip), mask.ir_value(loc=loc, ip=ip)],
#             ".pred p1, p2;\n"
#             "setp.lt.f32 p1, $1, $2;\n"
#             "vote.sync.any.pred p2, p1, $3;\n"
#             "selp.u32 $0, 1, 0, p2;",
#             # "selp.u32 $0, 1, 0, p1;",
#             "=r,f,f,r",
#             has_side_effects=False,
#             is_align_stack=False,
#             asm_dialect=llvm.AsmDialect.AD_ATT,
#         )
#     )


@dsl_user_op
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
    *,
    loc=None,
    ip=None
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_fragment(1, type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in range(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]
