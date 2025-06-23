# Copyright (c) 2025, Tri Dao.
from typing import Type, Callable, Optional

import cutlass
import cutlass.cute as cute


def get_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = dtype.width // 8
    bytes_per_row = k_dim * dtype_byte
    smem_k_block_size = (128 if bytes_per_row % 128 == 0 else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))) // dtype_byte
    swizzle_bits = 4 if smem_k_block_size == 128 else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout((8 if k_dim % 32 == 0 else 16, smem_k_block_size), order=(1, 0)),
    )


def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
    A_in_regs: cutlass.Constexpr[bool] = False,
    B_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if swap_AB:
        gemm(
            tiled_mma, acc, tCrB, tCrA, tCsB, tCsA, smem_thr_copy_B, smem_thr_copy_A, hook_fn,
            A_in_regs=B_in_regs, B_in_regs=A_in_regs, swap_AB=False
        )
    else:
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
        if not A_in_regs:
            cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
        if not B_in_regs:
            cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
        for k in range(cute.size(tCsA.shape[2])):
            if k < cute.size(tCsA.shape[2]) - 1:
                if not A_in_regs:
                    cute.copy(smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy_view[None, None, k + 1])
                if not B_in_regs:
                    cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
            cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            if cutlass.const_expr(k == 0 and hook_fn is not None):
                hook_fn()


def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in range(cute.size(tCrA.shape[2])):
        if k < cute.size(tCrA.shape[2]) - 1:
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        if cutlass.const_expr(k == 0 and hook_fn is not None):
            hook_fn()
