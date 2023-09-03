from typing import Union

import torch

import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 2}),
#         triton.Config({"BLOCK_M": 4}),
#         triton.Config({"BLOCK_M": 8}),
#         triton.Config({"BLOCK_M": 16}),
#     ],
#     key=["CACHE_KEY_SEQLEN", "BLOCK_K", "INTERLEAVED"]
# )
@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K // 2)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    X = X + (
        pid_batch * stride_x_batch
        + rm[:, None] * stride_x_seqlen
        + pid_head * stride_x_nheads
        + rk[None, :] * stride_x_headdim * (2 if INTERLEAVED else 1)
    )
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk[None, :])

    cos = tl.load(
        COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk[None, :] < rotary_dim_half), other=1.0
    ).to(tl.float32)
    sin = tl.load(
        SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk[None, :] < rotary_dim_half), other=0.0
    ).to(tl.float32)
    x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim_half), other=0.0).to(
        tl.float32
    )
    x1 = tl.load(
        X + stride_x_headdim * (1 if INTERLEAVED else rotary_dim_half),
        mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim_half),
        other=0.0,
    ).to(tl.float32)
    if not CONJUGATE:
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
    else:
        o0 = x0 * cos + x1 * sin
        o1 = -x0 * sin + x1 * cos

    # write back result
    OUT = OUT + (
        pid_batch * stride_out_batch
        + rm[:, None] * stride_out_seqlen
        + pid_head * stride_out_nheads
        + rk[None, :] * stride_out_headdim * (2 if INTERLEAVED else 1)
    )
    tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim_half))
    tl.store(
        OUT + stride_out_headdim * (1 if INTERLEAVED else rotary_dim_half),
        o1,
        mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim_half),
    )


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim)
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)  # noqa
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    rotary_kernel[grid](
        output,  # data ptrs
        x,
        cos,
        sin,
        seqlen_offsets,
        seqlen,  # shapes
        nheads,
        rotary_dim,
        seqlen_ro,
        seqlen // 128,  # key for triton cache (limit number of compilations)
        output.stride(0),  # strides
        output.stride(1),
        output.stride(2),
        output.stride(3),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_K,
        isinstance(seqlen_offsets, torch.Tensor),
        interleaved,
        conjugate,
        BLOCK_M,
    )
    return output
