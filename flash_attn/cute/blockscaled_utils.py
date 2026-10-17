import math

import torch
import torch.nn.functional as F


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


def quantize_to_fp8_with_sf(x_bf16, sf_vec_size=32):
    """Quantize Q or K (batch, seqlen, nheads, hdim) to E4M3 with E8M0 scales along hdim.

    Returns:
        x_fp8: (batch, seqlen, nheads, hdim) in FP8 E4M3FN
        sf_e8m0: (batch, seqlen, nheads, sf_k) in E8M0FNU; pass it through interleave_sf
        scale_float: (batch, seqlen, nheads, sf_k) FP32 scales for dequantization
    """
    return _quantize_groups(x_bf16, -1, sf_vec_size)


def dequantize_fp8_with_sf(x_fp8, scale_float, sf_vec_size=32):
    """Dequantize quantize_to_fp8_with_sf output back to FP32."""
    scales = scale_float.float().repeat_interleave(sf_vec_size, dim=-1)
    return x_fp8.float() * scales[..., : x_fp8.shape[-1]]


def quantize_v_to_fp8_with_sf(
    v: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize finite V (B, S, H, D) to E4M3, sharing a scale across 32 keys.

    Each batch/head/channel has independent groups starting at key zero, with the scale
    rule of quantize_to_fp8_with_sf.

    Returns V in contiguous (B, S, H, D) layout, interleaved UE8M0 SFV, and
    FP32 scales in (B, ceil(S / 32), H, D) layout for reference dequantization.
    SFV uses BlockScaledBasicChunk with MN=D and K=S, not Q/K's MN=S, K=D:
    physical shape (B, H, ceil(D / 128), ceil(S / 128), 32, 4, 4). Its byte
    at [b, h, d//128, g//4, d%32, (d%128)//32, g%4] scales V[b, 32*g:32*g+32, h, d].

    Along-key scales depend on future keys in each group: this representation
    is for noncausal attention, not causal/local attention or an append-only KV cache.
    """
    if v.ndim != 4 or v.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("V must be a float16, bfloat16, or float32 (B, S, H, D) tensor")
    if sf_vec_size != 32:
        raise ValueError("Native MXFP8 PV requires sf_vec_size=32")
    v_fp8, sf, scales = _quantize_groups(v, 1, sf_vec_size)
    sfv = interleave_sf(sf.permute(0, 3, 2, 1), sf_vec_size)
    return v_fp8, sfv, scales


def dequantize_v_fp8_with_sf(
    v_fp8: torch.Tensor, scale_float: torch.Tensor, sf_vec_size: int = 32
) -> torch.Tensor:
    """Dequantize V using logical (B, ceil(S / 32), H, D) along-key scales."""
    if sf_vec_size != 32:
        raise ValueError("Native MXFP8 PV requires sf_vec_size=32")
    batch, seqlen, nheads, hdim = v_fp8.shape
    if scale_float.shape != (batch, math.ceil(seqlen / sf_vec_size), nheads, hdim):
        raise ValueError("V scales must have shape (B, ceil(S / 32), H, D)")
    scales = scale_float.float().repeat_interleave(sf_vec_size, dim=1)[:, :seqlen]
    return v_fp8.float() * scales


def interleave_sf(sf, sf_vec_size):
    """Interleave a K-major scale factor tensor into the BlockScaledBasicChunk atom layout.

    Input:  sf with shape (batch, seqlen, nheads, sf_k) where sf_k = ceil(hdim / sf_vec_size)
    Output: contiguous (batch, nheads, REST_M, REST_K, 32, 4, 4) storage.
            The attention interface also accepts this storage flattened as a raw buffer;
            the kernel reconstructs its logical layout from the operand dimensions.
            For PV's V operand, pass (batch, hdim_v, nheads, ceil(seqlen_k / 32)):
            its MN axis is channels and its K axis is key positions.
    """
    batch, seqlen, nheads, sf_k = sf.shape

    seqlen_padded = ((seqlen + 127) // 128) * 128
    sf_k_padded = ((sf_k + 3) // 4) * 4
    rest_m = seqlen_padded // 128
    rest_k = sf_k_padded // 4

    sf_work = sf.permute(0, 2, 1, 3)  # (batch, nheads, seqlen, sf_k)

    if seqlen_padded != seqlen or sf_k_padded != sf_k:
        sf_padded = torch.zeros(
            batch,
            nheads,
            seqlen_padded,
            sf_k_padded,
            dtype=sf.dtype,
            device=sf.device,
        )
        sf_padded[:, :, :seqlen, :sf_k] = sf_work
    else:
        sf_padded = sf_work.contiguous()

    # Decompose M -> (REST_M, 4, 32), SF_K -> (REST_K, 4)
    sf_decomp = sf_padded.reshape(batch, nheads, rest_m, 4, 32, rest_k, 4)
    # Permute to (batch, nheads, REST_M, REST_K, 32, 4, 4) and make contiguous
    return sf_decomp.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
