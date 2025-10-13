import csv
import math
import torch
import os
import random
import functools
import triton
import triton.language as tl
import numpy as np
from typing import Literal, Optional, Union, Tuple

# -------------------------------
# Gloabl Variables
# -------------------------------
AUTOTUNE = os.environ.get("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "0").lower() in (
    "1",
    "true",
    "yes",
)
DEBUG = os.environ.get("FLASH_ATTENTION_TRITON_AMD_DEBUG", "0").lower() in (
    "1",
    "true",
    "yes",
)
if AUTOTUNE or DEBUG:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
USE_TRITON_INTERPRET = os.environ.get("TRITON_INTERPRET", "0").lower() in (
    "1",
    "true",
    "yes",
)
DEBUG_TRITON = (
    os.environ.get("DEBUG_TRITON", "0").lower() in ("1", "true", "yes")
    and USE_TRITON_INTERPRET
)
DEBUG_TRITON_DETAIL = (
    os.environ.get("DEBUG_TRITON_DETAIL", "0").lower() in ("1", "true", "yes")
    and USE_TRITON_INTERPRET
)
if USE_TRITON_ROCM:  # TODO remove this
    random.seed(42)
BWD_MODE: Literal["fused", "fused_atomic", "split"] = "fused"
USE_EXP2 = True
PHILOX_SEED = 0x1BF58
PHILOX_OFFSET = 0x1D4B49
SHAPE_EXPECTATIONS: Literal["exact", "rounded"] = "exact"


# -------------------------------
# Input Helper
# -------------------------------
def random_seqlens_composition(SEQ_LEN, BATCH):
    # generate a random composition of N into Z positive parts.
    idx = torch.randperm(SEQ_LEN - 1)[: BATCH - 1] + 1
    idx, _ = torch.sort(idx)
    breakpoints = torch.cat(
        [
            torch.tensor([0], dtype=torch.long),
            idx,
            torch.tensor([SEQ_LEN], dtype=torch.long),
        ]
    )
    seqlens = (breakpoints[1:] - breakpoints[:-1]).to(torch.int32)
    return seqlens


def generate_varlen_tensor(
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    batch_size: Optional[int] = None,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    mode: Literal["random", "ones", "incremental", "identity"] = "random",
):
    if DEBUG:
        print("total_seqlen", total_seqlen)
        print("num_heads", num_heads)
        print("head_size", head_size)

    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # get valid batch_size
    if batch_size is None:
        valid_batch_sizes = [
            bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= total_seqlen
        ]
        batch_size = random.choice(valid_batch_sizes)

    # get seqlens
    if equal_seqlens:
        seqlens = torch.full(
            (batch_size,), total_seqlen // batch_size, dtype=torch.int32, device=device
        )
        seqlens[-1] += total_seqlen % batch_size
    else:
        seqlens = random_seqlens_composition(total_seqlen, batch_size).to(device=device)

    # create cumulative sequence lengths
    cu_seqlens = (
        torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(dim=0)]
        )
        .to(torch.int32)
        .to(device=device)
    )
    max_seqlen = torch.max(seqlens).to(torch.int32).item()

    # create varlen tensor based on mode
    if mode == "incremental":
        x = torch.zeros(total_seqlen, num_heads, head_size, dtype=dtype, device=device)
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            x[start:end, :, :] = (
                torch.arange(length, dtype=dtype, device=device)
                .view(length, 1, 1)
                .expand(length, num_heads, head_size)
            )
    elif mode == "identity":
        x = torch.zeros(total_seqlen, num_heads, head_size, dtype=dtype, device=device)
        # for each batch, create identity pattern within that batch's sequence
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            # create identity pattern for positions within this batch
            for pos in range(min(length, head_size)):
                x[start + pos, :, pos] = 1.0
    elif mode == "random":
        x = torch.randn(
            (total_seqlen, num_heads, head_size), dtype=dtype, device=device
        )
    elif mode == "ones":
        x = torch.ones((total_seqlen, num_heads, head_size), dtype=dtype, device=device)
    else:
        raise ValueError(f"Unkown mode {mode}")

    if is_fp8_dtype:
        # cast to fp8
        x, descale_x = cast_to_fp8(
            x, og_fp8_dtype, "thd", cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen, descale_x
    else:
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen


def generate_bshd_tensor(
    BATCH,
    SEQ_LEN,
    NUM_HEADS,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    mode: Literal["random", "ones", "incremental", "identity"] = "random",
):
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor based on mode
    tensor_shape = (BATCH, SEQ_LEN, NUM_HEADS, D_HEAD)
    if mode == "incremental":
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, SEQ_LEN, 1, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    elif mode == "identity":
        x = torch.zeros(tensor_shape, dtype=dtype, device=device)
        # create identity pattern: position i has value 1 at dimension i
        for i in range(min(SEQ_LEN, D_HEAD)):
            x[:, i, :, i] = 1.0
    elif mode == "random":
        x = torch.randn(tensor_shape, dtype=dtype, device=device)
    elif mode == "ones":
        x = torch.ones(tensor_shape, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unkown mode {mode}")

    if is_fp8_dtype:
        # cast to fp8
        x, descale_x = cast_to_fp8(x, og_fp8_dtype, "bshd")
        x.requires_grad_()
        return x, descale_x
    else:
        x.requires_grad_()
        return x


def generate_bhsd_tensor(
    BATCH,
    NUM_HEADS,
    SEQ_LEN,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    mode: Literal["random", "ones", "incremental", "identity"] = "random",
):
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor based on mode
    tensor_shape = (BATCH, NUM_HEADS, SEQ_LEN, D_HEAD)
    if mode == "incremental":
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, 1, SEQ_LEN, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    elif mode == "identity":
        x = torch.zeros(tensor_shape, dtype=dtype, device=device)
        # create identity pattern: position i has value 1 at dimension i
        for i in range(min(SEQ_LEN, D_HEAD)):
            x[:, :, i, i] = 1.0
    elif mode == "random":
        x = torch.randn(tensor_shape, dtype=dtype, device=device)
    elif mode == "ones":
        x = torch.ones(tensor_shape, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unkown mode {mode}")

    if is_fp8_dtype:
        raise ValueError("fp8 not supported for bhsd yet")
    else:
        x.requires_grad_()
        return x


def generate_bshd_qkv_packed(
    BATCH,
    SEQ_LEN,
    NUM_HEADS,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate QKV packed tensor with shape (BATCH, SEQ_LEN, 3, NUM_HEADS, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, SEQ_LEN, 3, NUM_HEADS, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, SEQ_LEN, 1, 1, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for QKV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_bshd_kv_packed(
    BATCH,
    SEQ_LEN,
    NUM_HEADS,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate KV packed tensor with shape (BATCH, SEQ_LEN, 2, NUM_HEADS, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, SEQ_LEN, 2, NUM_HEADS, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, SEQ_LEN, 1, 1, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for KV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_bhsd_qkv_packed(
    BATCH,
    NUM_HEADS,
    SEQ_LEN,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate QKV packed tensor with shape (BATCH, 3, NUM_HEADS, SEQ_LEN, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, 3, NUM_HEADS, SEQ_LEN, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, 1, 1, SEQ_LEN, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for QKV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_bhsd_kv_packed(
    BATCH,
    NUM_HEADS,
    SEQ_LEN,
    D_HEAD,
    dtype: torch.dtype = torch.float16,
    device="cuda",
    DEBUG_INPUT=False,
):
    """Generate KV packed tensor with shape (BATCH, 2, NUM_HEADS, SEQ_LEN, D_HEAD)"""
    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # gen tensor
    tensor_shape = (BATCH, 2, NUM_HEADS, SEQ_LEN, D_HEAD)
    if DEBUG_INPUT:
        x = (
            torch.arange(SEQ_LEN, dtype=dtype, device=device)
            .view(1, 1, 1, SEQ_LEN, 1)
            .expand(*tensor_shape)
            .contiguous()
        )
    else:
        x = torch.randn(tensor_shape, dtype=dtype, device=device)

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for KV packing yet")
    else:
        x.requires_grad_()
        return x


def generate_varlen_qkv_packed(
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    batch_size: Optional[int] = None,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    DEBUG_INPUT: bool = False,
):
    """Generate varlen QKV packed tensor with shape (total_seqlen, 3, num_heads, head_size)"""
    if DEBUG:
        print("generate_varlen_qkv_packed")
        print("total_seqlen", total_seqlen)
        print("num_heads", num_heads)
        print("head_size", head_size)

    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # get valid batch_size
    if batch_size is None:
        valid_batch_sizes = [
            bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= total_seqlen
        ]
        batch_size = random.choice(valid_batch_sizes)

    # get seqlens
    if equal_seqlens:
        seqlens = torch.full(
            (batch_size,), total_seqlen // batch_size, dtype=torch.int32, device=device
        )
        seqlens[-1] += total_seqlen % batch_size
    else:
        seqlens = random_seqlens_composition(total_seqlen, batch_size).to(device=device)

    # create cumulative sequence lengths
    cu_seqlens = (
        torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(dim=0)]
        )
        .to(torch.int32)
        .to(device=device)
    )
    max_seqlen = torch.max(seqlens).to(torch.int32).item()

    # create varlen qkv packed tensor
    if DEBUG_INPUT:
        x = torch.zeros(
            total_seqlen, 3, num_heads, head_size, dtype=dtype, device=device
        )
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            x[start:end, :, :, :] = (
                torch.arange(length, dtype=dtype, device=device)
                .view(length, 1, 1, 1)
                .expand(length, 3, num_heads, head_size)
            )
    else:
        x = torch.randn(
            (total_seqlen, 3, num_heads, head_size), dtype=dtype, device=device
        )

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for QKV packing yet")
    else:
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen


def generate_varlen_kv_packed(
    total_seqlen: int,
    num_heads: int,
    head_size: int,
    batch_size: Optional[int] = None,
    equal_seqlens: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    DEBUG_INPUT: bool = False,
):
    """Generate varlen KV packed tensor with shape (total_seqlen, 2, num_heads, head_size)"""
    if DEBUG:
        print("generate_varlen_kv_packed")
        print("total_seqlen", total_seqlen)
        print("num_heads", num_heads)
        print("head_size", head_size)

    # save fp8 type
    is_fp8_dtype = is_dtype_fp8(dtype)
    if is_fp8_dtype:
        og_fp8_dtype = dtype
        dtype = torch.float32

    # get valid batch_size
    if batch_size is None:
        valid_batch_sizes = [
            bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= total_seqlen
        ]
        batch_size = random.choice(valid_batch_sizes)

    # get seqlens
    if equal_seqlens:
        seqlens = torch.full(
            (batch_size,), total_seqlen // batch_size, dtype=torch.int32, device=device
        )
        seqlens[-1] += total_seqlen % batch_size
    else:
        seqlens = random_seqlens_composition(total_seqlen, batch_size).to(device=device)

    # create cumulative sequence lengths
    cu_seqlens = (
        torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(dim=0)]
        )
        .to(torch.int32)
        .to(device=device)
    )
    max_seqlen = torch.max(seqlens).to(torch.int32).item()

    # create varlen kv packed tensor
    if DEBUG_INPUT:
        x = torch.zeros(
            total_seqlen, 2, num_heads, head_size, dtype=dtype, device=device
        )
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            length = end - start

            x[start:end, :, :, :] = (
                torch.arange(length, dtype=dtype, device=device)
                .view(length, 1, 1, 1)
                .expand(length, 2, num_heads, head_size)
            )
    else:
        x = torch.randn(
            (total_seqlen, 2, num_heads, head_size), dtype=dtype, device=device
        )

    if is_fp8_dtype:
        # cast to fp8 - need to handle the packed dimension
        raise NotImplementedError("FP8 not supported for KV packing yet")
    else:
        x.requires_grad_()
        return x, cu_seqlens, max_seqlen


# -------------------------------
# Alibi
# -------------------------------
@triton.jit
def compute_alibi_block(
    alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False
):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


# -------------------------------
# FP8
# -------------------------------
def is_dtype_fp8(dtype) -> bool:
    supported = {
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
    if dtype not in supported:
        return False
    return True


_RECOMMENDED_FP8_REPLACEMENTS = {
    "gfx942": {
        torch.float8_e4m3fn: torch.float8_e4m3fnuz,
        torch.float8_e5m2: torch.float8_e5m2fnuz,
    },
}


def get_recommended_fp8_dtype(x):
    dtype = x.dtype if isinstance(x, torch.Tensor) else x
    if not is_dtype_fp8(dtype):
        return dtype
    arch = get_arch()
    return _RECOMMENDED_FP8_REPLACEMENTS.get(arch, {}).get(dtype, dtype)


def is_fp8(x) -> bool:
    """Return whether tensor(s) use FP8.

    Accepts either a single tensor or a list/tuple of tensors.

    Rules:
      * Single tensor: return True if FP8 (after arch validation), else False.
      * Multiple tensors:
          - If all tensors are FP8 -> return True.
          - If none are FP8 -> return False.
          - If a mix of FP8 and non-FP8 -> raise ValueError.

    Empty list/tuple returns False.
    """

    def _is_fp8_single(t: torch.Tensor) -> bool:
        if is_dtype_fp8(t.dtype):
            arch = get_arch()
            if arch not in ("gfx942", "gfx950"):
                raise RuntimeError(
                    f"{arch} is not in the list of supported architectures for FP8"
                )
            return True
        return False

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        flags = [_is_fp8_single(t) for t in x]
        if all(flags):
            return True
        if not any(flags):
            return False
        raise ValueError(
            "Mixed FP8 and non-FP8 tensors provided; either all or none must be FP8."
        )
    else:
        return _is_fp8_single(x)


@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


@triton.jit
def _cast_varlen_to_fp8_kernel_2d(
    X,
    X_fp8,
    Descale,
    cu_seqlens,
    H,
    MAX_SEQLEN,
    stride_batch,
    stride_seq,
    stride_head,
    stride_dim,
    stride_out_batch,
    stride_out_seq,
    stride_out_head,
    stride_out_dim,
    stride_desc_batch,
    stride_desc_head,
    FP8_CLAMP_VAL,
    FP8_MAX,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Process one (batch, head) pair per kernel
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)

    # Get sequence bounds for this batch
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + b_id)
        seq_end = tl.load(cu_seqlens + b_id + 1)
        seqlen = seq_end - seq_start
    else:
        seq_start = 0
        seqlen = MAX_SEQLEN

    # initialize max value tracker
    x_max_val = 0.0

    # STEP 1: Find max absolute value across the entire sequence
    num_of_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    for blk_idx in range(0, num_of_blocks):
        # print("blk_idx:", blk_idx)
        # offsets
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)

        # Create mask for valid elements
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim

        # Load block
        adj_x = (
            b_id * stride_batch
            + h_id * stride_head
            + seq_start * stride_seq
            + offs_seq[:, None] * stride_seq
            + offs_dim[None, :] * stride_dim
        )
        x_block = tl.load(X + adj_x, mask=mask_seq, other=0.0)
        # print("x_block:", x_block)

        # Find max absolute value in this block
        block_max = tl.max(tl.abs(x_block))
        # print("block_max:", block_max)

        # Update overall max
        x_max_val = tl.maximum(x_max_val, block_max)
        # print("x_max_val:", x_max_val)

    # clamp to avoid division by zero issues
    x_max_val = tl.maximum(x_max_val, FP8_CLAMP_VAL)

    # compute scale and descale factors for the entire sequence
    scale = FP8_MAX / x_max_val
    descale = x_max_val / FP8_MAX

    # store descale factor for this (batch, head) pair
    desc_ptr = Descale + b_id * stride_desc_batch + h_id  # * stride_desc_head
    tl.store(desc_ptr, descale)

    # STEP 2: Apply scaling to the entire sequence and convert to FP8
    for blk_idx in range(0, num_of_blocks):
        # offsets
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)

        # Create mask for valid elements
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim

        # Load block - Using the fixed addressing
        addr = (
            b_id * stride_batch
            + h_id * stride_head
            + seq_start * stride_seq
            + offs_seq[:, None] * stride_seq
            + offs_dim[None, :] * stride_dim
        )
        x_block = tl.load(X + addr, mask=mask_seq, other=0.0)

        # Apply scale and convert to FP8
        x_fp8_block = (x_block * scale).to(X_fp8.type.element_ty)

        # Store results
        addr_out = (
            b_id * stride_out_batch
            + h_id * stride_out_head
            + seq_start * stride_out_seq
            + offs_seq[:, None] * stride_out_seq
            + offs_dim[None, :] * stride_out_dim
        )
        tl.store(X_fp8 + addr_out, x_fp8_block, mask=mask_seq)


def cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    layout: Literal["bshd", "thd"],
    clamp_val: float = 1e-9,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if False:
        print()
        print("cast_to_fp8")
        print("x:", x, x.shape)
        print("fp8_dtype:", fp8_dtype)
        print("cu_seqlens:", cu_seqlens)
        print("max_seqlen:", max_seqlen)
        print("clamp_val:", clamp_val)

    # check types are valid
    assert x.dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    } and is_dtype_fp8(fp8_dtype), f"Cannot cast {x.dtype} to {fp8_dtype}"

    # extract dimensions
    batch, max_seqlen_final, num_heads, head_dim = get_shape_from_layout(
        x, layout, cu_seqlens, max_seqlen
    )
    is_varlen = layout == "thd"
    fp8_max = torch.finfo(fp8_dtype).max
    if False:
        print("batch:", batch)
        print("max_seqlen_final:", max_seqlen_final)
        print("num_heads:", num_heads)
        print("head_dim:", head_dim)

    # get closest power of 2 for head_dim
    padded_head_dim = 1 << (head_dim - 1).bit_length()
    padded_head_dim = max(padded_head_dim, 32)

    # kernel params
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros(
        (batch, num_heads), device=x.device, dtype=torch.float32
    )
    BLOCK_SIZE = 128

    # calculate strides
    stride_batch, stride_head, stride_seq, stride_dim = get_stride_from_layout(
        x, layout
    )
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim = (
        get_stride_from_layout(x_fp8, layout)
    )
    stride_desc_batch, stride_desc_head = descale_factors.stride()

    if False:
        print("stride_batch", stride_batch)
        print("stride_head", stride_head)
        print("stride_seq", stride_seq)
        print("stride_dim", stride_dim)
        print("stride_out_batch", stride_out_batch)
        print("stride_out_head", stride_out_head)
        print("stride_out_seq", stride_out_seq)
        print("stride_out_dim", stride_out_dim)
        print("stride_desc_batch", stride_desc_batch)
        print("stride_desc_head", stride_desc_head)

    grid = (batch, num_heads)
    _cast_varlen_to_fp8_kernel_2d[grid](
        x,
        x_fp8,
        descale_factors,
        cu_seqlens,
        num_heads,
        max_seqlen_final,
        stride_batch,
        stride_seq,
        stride_head,
        stride_dim,
        stride_out_batch,
        stride_out_seq,
        stride_out_head,
        stride_out_dim,
        stride_desc_batch,
        stride_desc_head,
        clamp_val,
        fp8_max,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=padded_head_dim,
        ACTUAL_HEAD_DIM=head_dim,
        IS_VARLEN=is_varlen,
    )

    if False:
        print("x_fp8:", x_fp8, x_fp8.shape)
        print("descale_factors:", descale_factors, descale_factors.shape)
    return x_fp8, descale_factors


# -------------------------------
# Misc
# -------------------------------
def get_shape_from_layout(
    x: torch.Tensor,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> tuple[int, int, int, int]:
    if layout == "bhsd":
        batch, num_heads, max_seqlen_final, head_dim = x.shape
    elif layout == "bshd":
        batch, max_seqlen_final, num_heads, head_dim = x.shape
    elif layout == "thd":
        total_seqlen, num_heads, head_dim = x.shape
        if cu_seqlens is None:
            raise ValueError("cu_seqlens must be provided for varlen (thd) layout")
        if max_seqlen is None:
            raise ValueError("max_seqlen must be provided for varlen (thd) layout")

        batch, max_seqlen_final, num_heads, head_dim = (
            len(cu_seqlens) - 1,
            max_seqlen,
            num_heads,
            head_dim,
        )
    else:
        assert False, "Got unsupported layout."

    return batch, max_seqlen_final, num_heads, head_dim


def get_shapes_from_layout(
    q,
    k,
    layout,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    batch_q, seqlen_q, nheads_q, head_size_q = get_shape_from_layout(
        q, layout, cu_seqlens_q, max_seqlen_q
    )
    batch_k, seqlen_k, nheads_k, head_size_k = get_shape_from_layout(
        k, layout, cu_seqlens_k, max_seqlen_k
    )

    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, seqlen_q, seqlen_k


def get_stride_from_layout(x: torch.Tensor, layout: Literal["bshd", "bhsd", "thd"]):
    if layout == "thd":
        strides = (0, x.stride(1), x.stride(0), x.stride(2))
    elif layout == "bhsd":
        strides = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
    elif layout == "bshd":
        strides = (x.stride(0), x.stride(2), x.stride(1), x.stride(3))
    else:
        assert False, "Got unsupported layout."
    return strides


def get_shape_and_strides_from_layout(
    x: torch.Tensor,
    layout: Literal["bshd", "bhsd", "thd"],
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    return get_shape_from_layout(
        x, layout, cu_seqlens, max_seqlen
    ), get_stride_from_layout(x, layout)


def get_strides_from_layout(q, k, v, o, layout):
    q_strides = get_stride_from_layout(q, layout)
    k_strides = get_stride_from_layout(k, layout)
    v_strides = get_stride_from_layout(v, layout)
    o_strides = get_stride_from_layout(o, layout)
    return q_strides, k_strides, v_strides, o_strides


def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(
        -1
    )  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(
        0
    )  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return (
        -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos
    )  # (Z, H, N_CTX_Q, N_CTX_K)


def round_multiple(x, m):
    return (x + m - 1) // m * m


def save_tensor_to_csv(tensor, filename, decimal_places=2):
    """
    save a 2d tensor to csv file

    args:
        tensor: torch tensor of shape [rows, cols]
        filename: output csv filename
        decimal_places: number of decimal places (default: 2)
    """
    # ensure tensor is 2d
    if tensor.ndim != 2:
        raise ValueError(f"tensor must be 2d, got shape {tensor.shape}")

    # ensure filename ends with .csv
    if not filename.endswith(".csv"):
        filename = filename + ".csv"

    # save to csv using numpy
    np.savetxt(
        filename,
        tensor.detach().cpu().numpy(),
        delimiter=",",
        fmt=f"%.{decimal_places}f",
    )


# -------------------------------
# Dropouts
# -------------------------------
def create_dropout_mask(dropout_p, shape, seed):
    device = "cuda"
    rand_vals = torch.rand(
        shape,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
        dtype=torch.float32,
    )
    return rand_vals > dropout_p


def create_dropout_mask_varlen(
    dropout_p, batch, nheads_q, cu_seqlens_q, cu_seqlens_k, philox_seed
):
    device = "cuda"
    qlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    klens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_qlen = qlens.max()
    max_klen = klens.max()
    dropout_mask = torch.zeros((batch, nheads_q, max_qlen, max_klen), device=device)
    for b in range(batch):
        qlen = qlens[b]
        klen = klens[b]
        rand_vals = torch.rand(
            (nheads_q, qlen, klen),
            generator=torch.Generator(device=device).manual_seed(philox_seed),
            device=device,
            dtype=torch.float32,
        )
        submask = rand_vals > dropout_p
        dropout_mask[b, :, :qlen, :klen] = submask

    return dropout_mask


def write_dropout_mask(x, tensor_name="tensor"):
    batch, head, seqlen_m, seqlen_n = x.shape
    x = x.tolist()

    with open(f"{tensor_name}.csv", "w") as f:
        writer = csv.writer(f)
        for b in range(batch):
            for h in range(head):
                dropout_mask = x[b][h]
                if True:
                    BLOCK_M = 64
                    BLOCK_N = 64

                    # Calculate number of blocks in each dimension
                    m_blocks = math.ceil(seqlen_m / BLOCK_M)
                    n_blocks = math.ceil(seqlen_n / BLOCK_N)

                    # Process each block
                    for m_block in range(m_blocks):
                        # Calculate row range for current block
                        row_start = m_block * BLOCK_M
                        row_end = min(row_start + BLOCK_M, seqlen_m)

                        for n_block in range(n_blocks):
                            # Calculate column range for current block
                            col_start = n_block * BLOCK_N
                            col_end = min(col_start + BLOCK_N, seqlen_n)

                            # Extract and write the current block
                            for row_idx in range(row_start, row_end):
                                row_data = dropout_mask[row_idx][col_start:col_end]
                                writer.writerow(row_data)
                else:
                    writer.writerows(dropout_mask)


# -------------------------------
# Rotary
# -------------------------------
@triton.jit
def _rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
    seqlen,
    nheads,
    seqlen_ro,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch
        OUT = OUT + pid_batch * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk_half[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk_half[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        )
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0).to(
            tl.float32
        )
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk[None, None, :] < ROTARY_DIM)
        )
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


def _apply_rotary_kernel(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed, max_seqlen must also be provided"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    assert sin.shape == cos.shape
    rotary_dim = 2 * rotary_dim_half
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in (torch.int32, torch.int64)
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    out = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        out[..., rotary_dim:].copy_(x[..., rotary_dim:])

    # Block heuristics
    BLOCK_M = 8 if rotary_dim <= 128 else 4
    grid = (
        triton.cdiv(nheads, 2),
        triton.cdiv(seqlen, BLOCK_M),
        batch,
    )

    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(_rotary_kernel)[grid](
            out,
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,
            nheads,
            seqlen_ro,
            out.stride(0) if not is_varlen else 0,
            out.stride(-3),
            out.stride(-2),
            out.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3),
            x.stride(-2),
            x.stride(-1),
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=BLOCK_M,
            BLOCK_H=2,
        )
    return out


class _ApplyRotary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool,
        inplace: bool,
        seqlen_offsets: Union[int, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
    ):
        out = _apply_rotary_kernel(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=False,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dx = _apply_rotary_kernel(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Public API: apply rotary embeddings to tensor x.

    Args:
        x: (B, S, H, D) if `cu_seqlens` is None else (total_S, H, D).
        cos, sin: (S_rotary, rotary_dim/2)
        interleaved: GPT-J style if True.
        inplace: modify x in place (saves memory if rotary_dim == D).
        seqlen_offsets: int or (B,) tensor of starting offsets per sequence (KV cache decode).
        cu_seqlens: (B+1,) tensor enabling varlen mode.
        max_seqlen: required when `cu_seqlens` is provided.
    """
    # FP8 path: upcast to bfloat16 (preferred) or float16 for rotary math to avoid excessive error
    original_dtype = x.dtype
    is_fp8_input = original_dtype == getattr(torch, "float8_e4m3fn", None)
    if is_fp8_input:
        # Choose bf16 if available in cos.dtype path; otherwise fallback to float16
        target_dtype = (
            torch.bfloat16
            if cos.dtype == torch.bfloat16 or torch.cuda.is_bf16_supported()
            else torch.float16
        )
        # Upcast x, cos, sin for computation (without modifying originals in-place)
        x_up = x.to(target_dtype)
        cos_up = cos.to(target_dtype) if cos.dtype != target_dtype else cos
        sin_up = sin.to(target_dtype) if sin.dtype != target_dtype else sin
        out_up = _ApplyRotary.apply(
            x_up,
            cos_up,
            sin_up,
            interleaved,
            False,
            seqlen_offsets,
            cu_seqlens,
            max_seqlen,
        )
        # Cast result back to original fp8 dtype
        if inplace:
            x.copy_(out_up.to(original_dtype))
            return x
        return out_up.to(original_dtype)
    else:
        return _ApplyRotary.apply(
            x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
        )


def apply_rotary(
    q: torch.Tensor,
    k_new: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    causal: bool,
    local: bool,
    interleaved: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """High-level rotary application used by AMD prefill & decode paths.

    Policy (matches test reference & legacy semantics):
      - If causal OR local attention ⇒ apply rotary directly on (B, S, H, D).
      - Else (non-causal global) ⇒ flatten heads into sequence: (B, 1, S*H, D),
        apply rotary once, then unflatten back.
      - k_new (incremental KV slice) is always rotated directly when provided.

    Args:
        q: (B, S, H, D)
        k_new: Optional (B, S_k, H_k, D)
        cos, sin: rotary caches (S_rotary, rotary_dim/2)
        causal: causal attention flag
        local: sliding-window / local attention flag (pre-computed outside)
        interleaved: GPT-J style rotary layout
        seqlen_offsets: int or (B,) tensor of per-sequence start offsets
    Returns:
        (q_rot, k_new_rot)
    """
    assert q.ndim == 4, f"Expected q shape (B,S,H,D), got {q.shape}"
    B, S, H, D = q.shape
    use_flatten = (not causal) and (not local)

    if use_flatten:
        # Flatten (S,H) -> (S*H) with an added singleton dim to preserve expected 4D shape.
        q_flat = q.reshape(B, S * H, D).unsqueeze(1)  # (B, 1, S*H, D)
        q_flat = apply_rotary_emb(
            q_flat,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )
        # Restore shape back to (B, S, H, D)
        q = q_flat.view(B, 1, S * H, D).reshape(B, S, H, D)
    else:
        q = apply_rotary_emb(
            q,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )

    if k_new is not None:
        k_new = apply_rotary_emb(
            k_new,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )
    return q, k_new


# -------------------------------
# Runtime info
# -------------------------------
@functools.cache
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@functools.cache
def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


@functools.cache
def get_cu_count():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


@functools.cache
def is_cdna():
    return is_hip() and get_arch() in (
        "gfx908",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx950",
    )


@functools.cache
def is_rdna():
    return is_hip() and get_arch() in (
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
    )
