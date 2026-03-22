"""
Triton kernel helper functions shared across flash attention modules.

This module contains Triton JIT-compiled helper functions that are used within
the main attention kernels (fwd_prefill, fwd_decode, bwd). These are kept 
separate from utils.py to allow stricter type checking on pure Python utilities.
"""
from typing import Literal, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from .utils import DEBUG, get_shape_from_layout, get_stride_from_layout, is_fp8


@triton.jit
def compute_alibi_block(
    alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False
):
    """
    Compute ALiBi (Attention with Linear Biases) block.
    
    When seqlen_k and seqlen_q are different, the diagonal sticks to the 
    bottom right of the attention matrix.
    """
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5
    # offs_m = [0, 1], offs_n = [0, 1, 2, 3, 4]
    # Result: [[-3, -2, -1, 0, -1], [-4, -3, -2, -1, 0]]
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    """Compute FP8 scaling and descaling factors for a block."""
    x_amax = tl.max(tl.abs(x))
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
    """Cast tensor to FP8 with per-(batch, head) scaling."""
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
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)

        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim

        adj_x = (
            b_id * stride_batch
            + h_id * stride_head
            + seq_start * stride_seq
            + offs_seq[:, None] * stride_seq
            + offs_dim[None, :] * stride_dim
        )
        x_block = tl.load(X + adj_x, mask=mask_seq, other=0.0)
        block_max = tl.max(tl.abs(x_block))
        x_max_val = tl.maximum(x_max_val, block_max)

    # clamp to avoid division by zero
    x_max_val = tl.maximum(x_max_val, FP8_CLAMP_VAL)

    # compute scale and descale factors
    scale = FP8_MAX / x_max_val
    descale = x_max_val / FP8_MAX

    # store descale factor
    desc_ptr = Descale + b_id * stride_desc_batch + h_id
    tl.store(desc_ptr, descale)

    # STEP 2: Apply scaling and convert to FP8
    for blk_idx in range(0, num_of_blocks):
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)

        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim

        addr = (
            b_id * stride_batch
            + h_id * stride_head
            + seq_start * stride_seq
            + offs_seq[:, None] * stride_seq
            + offs_dim[None, :] * stride_dim
        )
        x_block = tl.load(X + addr, mask=mask_seq, other=0.0)
        x_fp8_block = (x_block * scale).to(X_fp8.type.element_ty)

        addr_out = (
            b_id * stride_out_batch
            + h_id * stride_out_head
            + seq_start * stride_out_seq
            + offs_seq[:, None] * stride_out_seq
            + offs_dim[None, :] * stride_out_dim
        )
        tl.store(X_fp8 + addr_out, x_fp8_block, mask=mask_seq)


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
    """Apply rotary positional embeddings."""
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


# -------------------------------
# Python wrappers for Triton kernels
# -------------------------------


def cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    layout: Literal["bshd", "thd"],
    clamp_val: float = 1e-9,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast tensor to FP8 with per-(batch, head) scaling factors."""
    if DEBUG > 0:
        print()
        print("cast_to_fp8")
        print("x:", x, x.shape)
        print("fp8_dtype:", fp8_dtype)
        print("cu_seqlens:", cu_seqlens)
        print("max_seqlen:", max_seqlen)
        print("clamp_val:", clamp_val)

    assert x.dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    } and is_fp8(fp8_dtype), f"Cannot cast {x.dtype} to {fp8_dtype}"

    batch, max_seqlen_final, num_heads, head_dim = get_shape_from_layout(
        x, layout, cu_seqlens, max_seqlen
    )
    is_varlen = layout == "thd"
    fp8_max = torch.finfo(fp8_dtype).max

    padded_head_dim = 1 << (head_dim - 1).bit_length()
    padded_head_dim = max(padded_head_dim, 32)

    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros(
        (batch, num_heads), device=x.device, dtype=torch.float32
    )
    BLOCK_SIZE = 128

    stride_batch, stride_head, stride_seq, stride_dim = get_stride_from_layout(x, layout)
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim = get_stride_from_layout(x_fp8, layout)
    stride_desc_batch, stride_desc_head = descale_factors.stride()

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

    return x_fp8, descale_factors


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
    """Apply rotary positional embeddings using Triton kernel."""
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed, max_seqlen must also be provided"
        total_seqlen, nheads, headdim = x.shape
        assert cu_seqlens is not None
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
    ) -> torch.Tensor:
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
    def backward(ctx, do: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
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
    """Apply rotary embeddings to tensor x.

    Args:
        x: (B, S, H, D) if `cu_seqlens` is None else (total_S, H, D).
        cos, sin: (S_rotary, rotary_dim/2)
        interleaved: GPT-J style if True.
        inplace: modify x in place.
        seqlen_offsets: int or (B,) tensor of starting offsets per sequence.
        cu_seqlens: (B+1,) tensor enabling varlen mode.
        max_seqlen: required when `cu_seqlens` is provided.
    """
    original_dtype = x.dtype
    is_fp8_input = original_dtype == getattr(torch, "float8_e4m3fn", None)
    if is_fp8_input:
        target_dtype = (
            torch.bfloat16
            if cos.dtype == torch.bfloat16 or torch.cuda.is_bf16_supported()
            else torch.float16
        )
        x_up = x.to(target_dtype)
        cos_up = cos.to(target_dtype) if cos.dtype != target_dtype else cos
        sin_up = sin.to(target_dtype) if sin.dtype != target_dtype else sin
        out_up = _ApplyRotary.apply(
            x_up, cos_up, sin_up, interleaved, False, seqlen_offsets, cu_seqlens, max_seqlen
        )
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
    """Apply rotary embeddings to q and optionally k_new.

    Policy:
      - If causal OR local attention: apply rotary directly on (B, S, H, D).
      - Else (non-causal global): flatten heads into sequence, apply, unflatten.
      - k_new is always rotated directly when provided.
    """
    assert q.ndim == 4, f"Expected q shape (B,S,H,D), got {q.shape}"
    B, S, H, D = q.shape
    use_flatten = (not causal) and (not local)

    if use_flatten:
        q_flat = q.reshape(B, S * H, D).unsqueeze(1)
        q_flat = apply_rotary_emb(q_flat, cos, sin, interleaved=interleaved, seqlen_offsets=seqlen_offsets)
        q = q_flat.view(B, 1, S * H, D).reshape(B, S, H, D)
    else:
        q = apply_rotary_emb(q, cos, sin, interleaved=interleaved, seqlen_offsets=seqlen_offsets)

    if k_new is not None:
        k_new = apply_rotary_emb(k_new, cos, sin, interleaved=interleaved, seqlen_offsets=seqlen_offsets)
    return q, k_new
