# Copyright (c) 2025, Tri Dao.
from typing import Type, Union, Optional
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_og


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    # A_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if const_expr(swap_AB):
        gemm(tiled_mma, acc, tCrB, tCrA, zero_init=zero_init, wg_wait=wg_wait, swap_AB=False)
    else:
        warpgroup.fence()
        # We make a new mma_atom since we'll be modifying its attribute (accumulate).
        # Otherwise the compiler complains "operand #0 does not dominate this use"
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    shape: cute.Shape,
    stage: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(layout, dtype, major_mode_size),
        dtype,
    )
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged


