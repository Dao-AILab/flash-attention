# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32


def _as_bshkrd_tensor(
    tensor: cute.Tensor,
    h_k: Int32,
    h_r: Int32,
    varlen: bool,
) -> cute.Tensor:
    """Normalize (B,S,H,D)/(S,H,D) tensors to (B,S,H_k,H_r,D) preserving real strides."""
    if cutlass.const_expr(cute.rank(tensor.layout) == 5):
        return tensor
    if cutlass.const_expr(cute.rank(tensor.layout) == 4):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[0], tensor.shape[1], h_k, h_r, tensor.shape[3]),
                stride=(
                    tensor.stride[0],
                    tensor.stride[1],
                    tensor.stride[2] * h_r,
                    tensor.stride[2],
                    tensor.stride[3],
                ),
            ),
        )
    assert cutlass.const_expr(cute.rank(tensor.layout) == 3), "Expected rank-3 varlen tensor"
    assert cutlass.const_expr(varlen), "Rank-3 input is only valid for varlen"
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, tensor.shape[0], h_k, h_r, tensor.shape[2]),
            stride=(
                0,
                tensor.stride[0],
                tensor.stride[1] * h_r,
                tensor.stride[1],
                tensor.stride[2],
            ),
        ),
    )


def _as_shhb_tensor(
    tensor: cute.Tensor,
    h_k: Int32,
    h_r: Int32,
    b: Int32,
    varlen: bool,
) -> cute.Tensor:
    """Normalize (B,H,S)/(H,S) tensors to (S, ((H_r, H_k), B)) view."""
    if cutlass.const_expr(cute.rank(tensor.layout) == 3):
        return cute.make_tensor(
            tensor.iterator,
            cute.make_layout(
                (tensor.shape[2], ((h_r, h_k), tensor.shape[0])),
                stride=(
                    tensor.stride[2],
                    ((tensor.stride[1], tensor.stride[1] * h_r), tensor.stride[0]),
                ),
            ),
        )
    assert cutlass.const_expr(cute.rank(tensor.layout) == 2), "Expected rank-2 varlen tensor"
    assert cutlass.const_expr(varlen), "Rank-2 input is only valid for varlen"
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (tensor.shape[1], ((h_r, h_k), b)),
            stride=(
                tensor.stride[1],
                ((tensor.stride[0], tensor.stride[0] * h_r), 0),
            ),
        ),
    )
