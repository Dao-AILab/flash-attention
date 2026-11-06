"""Host-side MXFP8 quantization of attention operands for block-scaled MMA.

    q8, sfq, _ = quantize_to_fp8_with_sf(q)        # scales along head_dim, raw (B, S, H, D/32)
    k8, sfk, _ = quantize_to_fp8_with_sf(k)
    v8, sfv, _ = quantize_v_to_fp8_with_sf(v)      # scales along keys, already interleaved
    sfq, sfk = interleave_sf(sfq), interleave_sf(sfk)  # the layout the kernel consumes

Q/K return raw scales so callers can also dequantize or inspect them; V returns the interleaved
buffer directly because its (MN=head_dim, K=keys) orientation differs from Q/K's.
"""

import math

import torch
import torch.nn.functional as F

SF_VEC_SIZE = 32  # elements per E8M0 scale; the only block size the MXFP8 kernels support


def _quantize_groups(
    x: torch.Tensor, dim: int, sf_vec_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize x to E4M3 with one power-of-two scale per sf_vec_size elements along dim.

    Groups start at index zero and the last partial group is zero-padded. The scale is
    2**ceil(log2(amax / 448)) clamped to the E8M0 exponent range, so group maxima never
    saturate; all-zero groups use scale one. The exponent comes from frexp, so a group
    maximum just above a scale boundary is not rounded down onto the boundary by log2.

    Returns E4M3 x (same shape), E8M0 scales and FP32 scales; the scales have the group
    axis at dim.
    """
    dim = dim % x.ndim
    size = x.shape[dim]
    num_groups = math.ceil(size / sf_vec_size)
    pad = [0, 0] * (x.ndim - 1 - dim) + [0, num_groups * sf_vec_size - size]
    padded = F.pad(x.float(), pad)
    groups = padded.unflatten(dim, (num_groups, sf_vec_size))
    absmax = groups.abs().amax(dim=dim + 1, keepdim=True)
    # 448 = 0.875 * 2**9, the largest finite E4M3 value.
    mantissa, exponent = torch.frexp(absmax)
    exponent = (exponent - 9 + (mantissa > 0.875)).clamp(-127, 127)
    exponent = torch.where(absmax == 0, 0, exponent)
    scales = torch.exp2(exponent.float())
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    x_fp8 = (
        (groups / scales)
        .clamp(-fp8_max, fp8_max)
        .flatten(dim, dim + 1)
        .narrow(dim, 0, size)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    sf = (exponent + 127).to(torch.uint8).view(torch.float8_e8m0fnu).squeeze(dim + 1)
    return x_fp8, sf, scales.squeeze(dim + 1)


def quantize_to_fp8_with_sf(
    x: torch.Tensor, sf_vec_size: int = SF_VEC_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize Q or K (batch, seqlen, nheads, hdim) to E4M3 with E8M0 scales along hdim.

    Returns:
        x_fp8: (batch, seqlen, nheads, hdim) in FP8 E4M3FN
        sf_e8m0: (batch, seqlen, nheads, sf_k) in E8M0FNU; pass it through interleave_sf
        scale_float: (batch, seqlen, nheads, sf_k) FP32 scales for dequantization
    """
    return _quantize_groups(x, -1, sf_vec_size)


def dequantize_fp8_with_sf(
    x_fp8: torch.Tensor, scale_float: torch.Tensor, sf_vec_size: int = SF_VEC_SIZE
) -> torch.Tensor:
    """Dequantize quantize_to_fp8_with_sf output back to FP32."""
    scales = scale_float.float().repeat_interleave(sf_vec_size, dim=-1)
    return x_fp8.float() * scales[..., : x_fp8.shape[-1]]


def quantize_v_to_fp8_with_sf(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize finite V (B, S, H, D) to E4M3, sharing a scale across 32 keys.

    Each batch/head/channel has independent groups starting at key zero, with the scale
    rule of quantize_to_fp8_with_sf.

    Returns V in contiguous (B, S, H, D) layout, interleaved UE8M0 SFV, and
    FP32 scales in (B, ceil(S / 32), H, D) layout for reference dequantization.
    SFV uses BlockScaledBasicChunk with MN=D and K=S, not Q/K's MN=S, K=D:
    physical shape (B, H, ceil(D / 128), ceil(S / 128), 32, 4, 4). Its byte
    at [b, h, d//128, g//4, d%32, (d%128)//32, g%4] scales V[b, 32*g:32*g+32, h, d].

    Along-key scales depend on future keys in each group, so this representation alone is
    for noncausal attention. Causal attention also needs the original BF16 V
    (`BlockScales.v_bf16`) so the diagonal blocks never read a scale shared with future keys;
    local attention and append-only KV caches are not supported.
    """
    if v.ndim != 4 or v.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("V must be a float16, bfloat16, or float32 (B, S, H, D) tensor")
    v_fp8, sf, scales = _quantize_groups(v, 1, SF_VEC_SIZE)
    return v_fp8, interleave_sf(sf.permute(0, 3, 2, 1)), scales


def dequantize_v_fp8_with_sf(v_fp8: torch.Tensor, scale_float: torch.Tensor) -> torch.Tensor:
    """Dequantize V using logical (B, ceil(S / 32), H, D) along-key scales."""
    batch, seqlen, nheads, hdim = v_fp8.shape
    if scale_float.shape != (batch, math.ceil(seqlen / SF_VEC_SIZE), nheads, hdim):
        raise ValueError("V scales must have shape (B, ceil(S / 32), H, D)")
    scales = scale_float.float().repeat_interleave(SF_VEC_SIZE, dim=1)[:, :seqlen]
    return v_fp8.float() * scales


def interleave_sf(sf: torch.Tensor) -> torch.Tensor:
    """Interleave a K-major scale factor tensor into the BlockScaledBasicChunk atom layout.

    Input:  UE8M0 sf with shape (batch, mn, nheads, sf_k): for Q/K (batch, seqlen, nheads,
            ceil(hdim / 32)); for PV's V operand (batch, hdim_v, nheads, ceil(seqlen_k / 32)),
            since its MN axis is channels and its K axis is key positions.
    Output: contiguous (batch, nheads, REST_M, REST_K, 32, 4, 4) storage, zero-padded to
            whole 128 x 128 tiles. The attention interface also accepts this storage flattened
            as a raw buffer; the kernel reconstructs its logical layout from the operand
            dimensions.
    """
    batch, mn, nheads, sf_k = sf.shape
    rest_m, rest_k = math.ceil(mn / 128), math.ceil(sf_k / 4)
    # F.pad cannot write 0.0 into E8M0 (not representable), so pad the bytes.
    sf_bytes = F.pad(
        sf.permute(0, 2, 1, 3).view(torch.uint8), (0, 4 * rest_k - sf_k, 0, 128 * rest_m - mn)
    )
    # MN -> (REST_M, 4, 32) and SF_K -> (REST_K, 4), ordered (REST_M, REST_K, 32, 4, 4).
    chunks = sf_bytes.reshape(batch, nheads, rest_m, 4, 32, rest_k, 4).permute(0, 1, 2, 5, 4, 3, 6)
    return chunks.contiguous().view(sf.dtype)
