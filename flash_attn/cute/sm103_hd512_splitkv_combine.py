# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Specialized combine for SM103 head-dimension-512 SplitKV decode."""

import math
from functools import partial

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import BFloat16, Float32, Int32


log2_e = math.log2(math.e)
exp2 = partial(cute.math.exp2, fastmath=True)


class BlackwellHd512SplitKVCombine:
    """Warp-cooperative combine for normalized FP32 SplitKV partials.

    This specialization consumes a fixed capacity of 32 splits for Q32/D512.
    The per-request ``mNumSplitsDynamic`` value may be any value in [1, 32].
    """

    def __init__(self, head_dim: int = 512, num_heads: int = 32):
        if head_dim != 512 or num_heads != 32:
            raise ValueError("SM103 hd512 combine requires Q32/D512")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dimensions_per_cta = 128
        self.num_threads = 128
        self.max_splits = 32

    @cute.jit
    def __call__(
        self,
        mOPartial: cute.Tensor,
        mLSEPartial: cute.Tensor,
        mO: cute.Tensor,
        mNumSplitsDynamic: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(len(mOPartial.shape) != 4):
            raise ValueError("O partial must have shape (splits, batch, 32, 512)")
        if cutlass.const_expr(len(mLSEPartial.shape) != 3):
            raise ValueError("LSE partial must have shape (splits, batch, 32)")
        if cutlass.const_expr(len(mO.shape) != 3):
            raise ValueError("output must have shape (batch, 32, 512)")
        if cutlass.const_expr(len(mNumSplitsDynamic.shape) != 1):
            raise ValueError("dynamic split metadata must have shape (batch,)")

        splits, batch, heads, head_dim = mOPartial.shape
        if cutlass.const_expr(
            splits != self.max_splits or heads != self.num_heads or head_dim != self.head_dim
        ):
            raise ValueError("SM103 hd512 combine requires S32/Q32/D512")
        if cutlass.const_expr(mLSEPartial.shape != (splits, batch, heads)):
            raise ValueError("LSE partial shape does not match O partial")
        if cutlass.const_expr(mO.shape != (batch, heads, head_dim)):
            raise ValueError("output shape does not match O partial")
        if cutlass.const_expr(mNumSplitsDynamic.shape[0] != batch):
            raise ValueError("dynamic split metadata does not match batch")

        if cutlass.const_expr(
            mOPartial.element_type is not cutlass.Float32
            or mLSEPartial.element_type is not cutlass.Float32
        ):
            raise TypeError("SM103 hd512 combine requires FP32 O/LSE partials")
        if cutlass.const_expr(mO.element_type is not cutlass.BFloat16):
            raise TypeError("SM103 hd512 combine requires BF16 output")
        if cutlass.const_expr(mNumSplitsDynamic.element_type is not cutlass.Int32):
            raise TypeError("SM103 hd512 combine requires int32 split metadata")

        if cutlass.const_expr(
            mOPartial.stride[-1] != 1 or mO.stride[-1] != 1 or mNumSplitsDynamic.stride[-1] != 1
        ):
            raise ValueError("SM103 hd512 combine requires contiguous inner dimensions")

        self.kernel(
            mOPartial,
            mLSEPartial,
            mO,
            mNumSplitsDynamic,
        ).launch(
            grid=(self.head_dim // self.dimensions_per_cta, self.num_heads, batch),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @staticmethod
    @cute.kernel
    def kernel(
        mOPartial: cute.Tensor,
        mLSEPartial: cute.Tensor,
        mO: cute.Tensor,
        mNumSplitsDynamic: cute.Tensor,
    ):
        d_block, head, batch = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        dim = d_block * 128 + tidx

        valid_splits = mNumSplitsDynamic[batch]
        if valid_splits > 0:
            lane_is_valid = lane < valid_splits
            lse_lane = -Float32.inf
            if lane_is_valid:
                lse_lane = mLSEPartial[lane, batch, head]
            lse_max = cute.arch.warp_reduction_max(lse_lane)

            correction_lane = Float32(0.0)
            if lane_is_valid:
                correction_lane = exp2(log2_e * (lse_lane - lse_max))
            correction_sum = cute.arch.warp_reduction_sum(correction_lane)

            out_acc = Float32(0.0)
            for split in cutlass.range(valid_splits):
                correction = cute.arch.shuffle_sync(correction_lane, split)
                out_acc += correction * mOPartial[split, batch, head, dim]

            # valid_splits == 1 intentionally stores the only split.
            mO[batch, head, dim] = mO.element_type(out_acc / correction_sum)


_COMPILE_OPTIONS = "--opt-level 2"
_ADAPTER_ABI_VERSION = 1
_compiled_combine_cache: dict[tuple[object, ...], object] = {}


def _tensor_spec(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
    )


def _as_cute_tensor(
    tensor: torch.Tensor,
    element_type,
    assumed_align: int,
) -> cute.Tensor:
    result = from_dlpack(tensor, assumed_align=assumed_align)
    result.element_type = element_type
    return result


def run_sm103_hd512_splitkv_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    valid_splits: torch.Tensor,
) -> None:
    """Compile-cache and launch the normalized-partial specialization."""
    if not out_partial.is_cuda:
        raise ValueError("SM103 hd512 combine requires CUDA tensors")
    tensors = (lse_partial, out, valid_splits)
    if any(tensor.device != out_partial.device for tensor in tensors):
        raise ValueError("SM103 hd512 combine tensors must share one device")
    if torch.cuda.get_device_capability(out_partial.device) != (10, 3):
        raise ValueError("SM103 hd512 combine requires compute capability 10.3")
    if out_partial.dtype != torch.float32 or lse_partial.dtype != torch.float32:
        raise TypeError("SM103 hd512 combine requires FP32 partials")
    if out.dtype != torch.bfloat16 or valid_splits.dtype != torch.int32:
        raise TypeError("SM103 hd512 combine requires BF16 output and int32 splits")
    if out_partial.ndim != 4 or tuple(out_partial.shape[2:]) != (32, 512):
        raise ValueError("expected O partial shape (32, batch, 32, 512)")
    splits, batch = (int(out_partial.shape[0]), int(out_partial.shape[1]))
    if splits != 32:
        raise ValueError("SM103 hd512 combine requires split capacity 32")
    if tuple(lse_partial.shape) != (splits, batch, 32):
        raise ValueError("LSE partial shape does not match O partial")
    if tuple(out.shape) != (batch, 32, 512):
        raise ValueError("output shape does not match O partial")
    if tuple(valid_splits.shape) != (batch,):
        raise ValueError("valid-split shape does not match batch")
    if not out_partial.is_contiguous() or not out.is_contiguous():
        raise ValueError("O partial and output must be contiguous")
    expected_lse_stride = (batch * 32, 1, batch)
    if tuple(lse_partial.stride()) != expected_lse_stride:
        raise ValueError(
            f"expected transposed LSE strides {expected_lse_stride}, got {lse_partial.stride()}"
        )
    if not valid_splits.is_contiguous():
        raise ValueError("valid splits must be contiguous")

    device_index = out_partial.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (
        _ADAPTER_ABI_VERSION,
        device_index,
        _tensor_spec(out_partial),
        _tensor_spec(lse_partial),
        _tensor_spec(out),
        _tensor_spec(valid_splits),
        _COMPILE_OPTIONS,
    )
    compiled = _compiled_combine_cache.get(key)
    out_partial_cute = _as_cute_tensor(out_partial, Float32, 16)
    lse_partial_cute = _as_cute_tensor(lse_partial, Float32, 4)
    out_cute = _as_cute_tensor(out, BFloat16, 16)
    valid_splits_cute = _as_cute_tensor(valid_splits, Int32, 4)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SM103 hd512 combine was not warmed before CUDA graph capture")
        compiled = cute.compile(
            BlackwellHd512SplitKVCombine(),
            out_partial_cute,
            lse_partial_cute,
            out_cute,
            valid_splits_cute,
            cutlass_torch.default_stream(),
            options=_COMPILE_OPTIONS,
        )
        _compiled_combine_cache[key] = compiled
    compiled(
        out_partial_cute,
        lse_partial_cute,
        out_cute,
        valid_splits_cute,
        cutlass_torch.current_stream(),
    )


__all__ = [
    "BlackwellHd512SplitKVCombine",
    "run_sm103_hd512_splitkv_combine",
]
