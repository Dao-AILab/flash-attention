"""Fused Triton MXFP8 quantization for the trainable MXFP8 forwards.

One pass per operand writes what both directions consume: the E4M3 payload and its interleave_sf
scale buffer for the forward (BlockScales), and the payload dequantized to BF16 for the mirrored
BF16 backward (NOTE [MXFP8 mirrored BF16 backward] in flash_bwd_sm100.py), exact for E4M3 x 2^k
above BF16's subnormal floor. The eager blockscaled_utils quantizers make several FP32 passes per
operand plus a separate dequantization; for finite inputs these kernels write byte-identical
payloads and scales (the same 2^ceil(log2(amax / 448)) scale rule, exact power-of-two scaling,
round-to-nearest-even E4M3 with satfinite).

- `quantize_rows(x)`: Q / K, one scale per 32 elements along head_dim (MN = seqlen).

The `mx_exponent`, `exp2_int`, `to_e4m3`, and `interleaved_sf_offset` helpers are the shared
MXFP8 primitives of the Triton preps.
"""

import math

import torch
import triton
import triton.language as tl

SF_VEC = 32  # elements per UE8M0 scale (blockscaled_utils.SF_VEC_SIZE)


@triton.jit
def mx_exponent(amax):
    """MX exponent e, 2^e = 2^ceil(log2(amax / 448)) clamped to [-127, 127], 0 for amax 0
    (blockscaled_utils._quantize_groups: frexp exponent - 9 + (mantissa > 0.875))."""
    bits = amax.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    e = (bits >> 23) - 135 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    e = tl.minimum(tl.maximum(e, -127), 127)
    return tl.where(amax == 0.0, 0, e)


@triton.jit
def exp2_int(e):
    """2^e exactly for integer e in [-127, 127] (2^-127 is the FP32 subnormal 0x400000)."""
    return tl.where(e == -127, 0x400000, (e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def to_e4m3(x, e):
    """E4M3 of x 2^-e for the mx_exponent exponent e of x's group. There is no clamp to [-448, 448]:
    for finite x the exponent rule gives |x| 2^-e <= 448, and the conversion is cvt.rn.satfinite
    (+-inf -> +-448, NaN -> NaN)."""
    return (x * exp2_int(-e)).to(tl.float8e4nv)


@triton.jit
def interleaved_sf_offset(bh, m, g, rest_m, rest_k):
    """Byte of (MN index m, K group g) in the interleave_sf buffer (batch * heads, rest_m, rest_k,
    32, 4, 4)."""
    return (
        ((bh * rest_m + m // 128) * rest_k + g // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + g % 4
    )


@triton.jit
def _quantize_rows_kernel(
    x_ptr, x8_ptr, sf_ptr, hat_ptr, rows, S, H, DIM: tl.constexpr, ROWS: tl.constexpr
):
    """ROWS rows of the (batch, S, H) x DIM input: E4M3 payload, UE8M0 scales per 32 columns into
    the interleave_sf buffer (MN = S, K = DIM / 32), and the dequantized payload in BF16."""
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    valid = row < rows
    offs = row[:, None] * DIM + tl.arange(0, DIM)[None, :]
    x = tl.load(x_ptr + offs, mask=valid[:, None], other=0.0).to(tl.float32)
    groups = tl.reshape(x, (ROWS, DIM // 32, 32))
    e = mx_exponent(tl.max(tl.abs(groups), axis=2))
    x8 = to_e4m3(groups, e[:, :, None])
    hat = x8.to(tl.float32) * exp2_int(e)[:, :, None]
    tl.store(x8_ptr + offs, tl.reshape(x8, (ROWS, DIM)), mask=valid[:, None])
    tl.store(hat_ptr + offs, tl.reshape(hat, (ROWS, DIM)).to(tl.bfloat16), mask=valid[:, None])
    token = (row // H) % S
    bh = (row // (H * S)) * H + row % H
    g = tl.arange(0, DIM // 32)[None, :]
    sf_offs = interleaved_sf_offset(
        bh[:, None], token[:, None], g, tl.cdiv(S, 128), tl.cdiv(DIM // 32, 4)
    )
    tl.store(sf_ptr + sf_offs, (e + 127).to(tl.uint8), mask=valid[:, None])


def _interleaved_buffer(batch: int, nheads: int, mn: int, sf_k: int, device) -> torch.Tensor:
    """An interleave_sf byte buffer; zeroed (the padding interleave_sf writes) only when MN or K
    leaves a partial tile."""
    rest_m, rest_k = math.ceil(mn / 128), math.ceil(sf_k / 4)
    alloc = torch.zeros if mn % 128 or sf_k % 4 else torch.empty
    return alloc(batch * nheads * rest_m * rest_k * 512, dtype=torch.uint8, device=device)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP8 of a contiguous (batch, seqlen, heads, dim) Q or K along dim (a multiple of 32):
    (E4M3 payload, flat interleave_sf UE8M0 scales, BF16 dequantized payload)."""
    batch, seqlen, nheads, dim = x.shape
    assert x.is_contiguous() and dim % SF_VEC == 0
    x8 = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    hat = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    sf = _interleaved_buffer(batch, nheads, seqlen, dim // SF_VEC, x.device)
    rows, rows_per_program = batch * seqlen * nheads, 32
    _quantize_rows_kernel[(triton.cdiv(rows, rows_per_program),)](
        x, x8, sf, hat, rows, seqlen, nheads, dim, rows_per_program, num_warps=4
    )
    return x8, sf.view(torch.float8_e8m0fnu), hat
