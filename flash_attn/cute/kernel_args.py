"""General call arguments for the main forward/backward kernels.

Each kernel class owns a local Args NamedTuple declaring exactly the arguments it
accepts. The interface builds one superset namedtuple (FwdKernelArgs / BwdKernelArgs)
and hands the same object to every architecture; each kernel narrows it through
normalize_kernel_args, which rejects any arguments the kernel does not explicitly
accept.

The same namedtuple carries CuTe tensors when compiling and torch tensors when invoking the
compiled kernel, hence the TensorArg union.
"""

from typing import NamedTuple, Optional, Union

import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.utils import AuxData, DescaleTensors

TensorArg = Union[cute.Tensor, torch.Tensor]


class FwdKernelArgs(NamedTuple):
    # mQ/mK are absent for the weight-absorbed MLA kernel, which reads mQv/mV instead.
    mQ: Optional[TensorArg]
    mK: Optional[TensorArg]
    mV: TensorArg
    mO: TensorArg
    softmax_scale: Float32
    mQv: Optional[TensorArg] = None
    mLSE: Optional[TensorArg] = None
    mCuSeqlensQ: Optional[TensorArg] = None
    mCuSeqlensK: Optional[TensorArg] = None
    mSeqUsedQ: Optional[TensorArg] = None
    mSeqUsedK: Optional[TensorArg] = None
    mPageTable: Optional[TensorArg] = None
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    learnable_sink: Optional[TensorArg] = None
    descale_tensors: Optional[DescaleTensors] = None
    blocksparse_tensors: Optional[BlockSparseTensors] = None
    aux_data: Optional[AuxData] = None
    num_splits_dynamic_ptr: Optional[TensorArg] = None
    tile_count_semaphore: Optional[TensorArg] = None
    virtual_batch_idx_ptr: Optional[TensorArg] = None
    num_nheads_in_l2_ptr: Optional[TensorArg] = None
    mCuTotalMBlocks: Optional[TensorArg] = None
    mCuTotalSplitsMBlocks: Optional[TensorArg] = None
    mBlocksToBatchIdx: Optional[TensorArg] = None
    max_seqlen_q: Optional[Int32] = None
    mP: Optional[TensorArg] = None
    mRowMax: Optional[TensorArg] = None
    mIndexTopk: Optional[TensorArg] = None


class BwdKernelArgs(NamedTuple):
    mQ: TensorArg
    mK: TensorArg
    mV: TensorArg
    mdO: TensorArg
    mLSE: TensorArg
    mdPsum: TensorArg
    mdQaccum: TensorArg
    mdK: TensorArg
    mdV: TensorArg
    softmax_scale: Float32
    mCuSeqlensQ: Optional[TensorArg] = None
    mCuSeqlensK: Optional[TensorArg] = None
    mSeqUsedQ: Optional[TensorArg] = None
    mSeqUsedK: Optional[TensorArg] = None
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    mdQ_semaphore: Optional[TensorArg] = None
    mdK_semaphore: Optional[TensorArg] = None
    mdV_semaphore: Optional[TensorArg] = None
    aux_data: Optional[AuxData] = None
    blocksparse_tensors: Optional[BlockSparseTensors] = None
    mCuTotalMBlocks: Optional[TensorArg] = None


def normalize_kernel_args(args: tuple, args_cls: type, kernel_name: str):
    """Narrow a *KernelArgs namedtuple to a kernel's local args_cls,
    checking validity of input args

    Raises a TypeError if
        - args is not a NamedTuple
        - any given non-None argument is not supported by kernel_name
        - any raw Python scalar arguments are passed in, as they get compiled away
            and disappear (should be passed to __init__ instead)
    """
    if not (isinstance(args, tuple) and hasattr(args, "_fields")):
        raise TypeError(
            f"{kernel_name}.__call__ expects a NamedTuple of kernel arguments, "
            f"got {type(args).__name__}"
        )
    if type(args) is not args_cls:
        unsupported = [
            f for f, v in zip(args._fields, args) if f not in args_cls._fields and v is not None
        ]
        if unsupported:
            raise TypeError(
                f"{kernel_name} does not support argument(s): {', '.join(unsupported)}. "
                f"It accepts: {', '.join(args_cls._fields)}"
            )
        # None means "not provided", so the kernel's own default applies. Any field a kernel
        # may legitimately receive as None must therefore declare a default.
        args = args_cls(
            **{f: v for f, v in zip(args._fields, args) if f in args_cls._fields and v is not None}
        )
    # A raw Python scalar inside a namedtuple is baked into the kernel as a compile-time
    # constant rather than becoming a dynamic argument, so a later call silently reuses it.
    baked = [f for f, v in zip(args._fields, args) if isinstance(v, (bool, int, float))]
    if baked:
        raise TypeError(
            f"{kernel_name} received raw Python scalar argument(s): {', '.join(baked)}. "
            f"Wrap them in cutlass.Int32/Float32 to keep them dynamic kernel arguments"
        )
    return args
