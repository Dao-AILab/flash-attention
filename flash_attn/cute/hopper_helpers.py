# Copyright (c) 2025, Tri Dao.
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup


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
    if swap_AB:
        gemm(tiled_mma, acc, tCrB, tCrA, zero_init=zero_init, wg_wait=wg_wait, swap_AB=False)
    else:
        warpgroup.fence()
        tiled_mma.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if cutlass.const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)
