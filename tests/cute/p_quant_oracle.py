"""CPU reference codecs for quant-dequant of attention probabilities.

Each function mirrors the kernel's FP32 arithmetic operation for operation, with
explicit round-to-nearest-even and saturation (PyTorch's ``float8_e4m3fn`` cast
does not saturate, so it is only a cross-check for finite in-range inputs).

``rounding`` selects the tie rule. Only "rne" is the contract; "half_away" and
"trunc" exist so tests can show that a wrong rule is detected.
"""

import torch

from flash_attn.cute.p_quant import (
    E2M1_MAX,
    E4M3_MAX,
    NVFP4_BLOCK,
    NVFP4_GLOBAL_SCALE,
    NVFP4_SCALE_FACTOR,
    resolve_p_quant,
)

# E4M3: 4 exponent bits (bias 7), 3 mantissa bits. Normals span 2**-6 .. 448,
# subnormals are multiples of 2**-9.
_E4M3_MIN_EXP = -6
_E4M3_MAX_EXP = 8
_E4M3_MANT_BITS = 3
# E2M1: 2 exponent bits (bias 1), 1 mantissa bit. Normals span 1 .. 6,
# the only subnormal is 0.5.
_E2M1_MIN_EXP = 0
_E2M1_MAX_EXP = 2
_E2M1_MANT_BITS = 1

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_int(x: torch.Tensor, rounding: str) -> torch.Tensor:
    if rounding == "rne":
        return torch.round(x)  # ties to even
    if rounding == "half_away":
        return torch.sign(x) * torch.floor(x.abs() + 0.5)
    if rounding == "trunc":
        return torch.trunc(x)
    raise ValueError(f"unknown rounding {rounding!r}")


def _minifloat_rne_sat(
    x: torch.Tensor,
    min_exp: int,
    max_exp: int,
    mant_bits: int,
    max_value: float,
    rounding: str,
) -> torch.Tensor:
    """Round FP32 ``x`` to a sign/exponent/mantissa minifloat grid, saturating at ``max_value``."""
    assert x.dtype == torch.float32
    # frexp: |x| = m * 2**e with m in [0.5, 1), so the binade exponent is e - 1.
    _, e = torch.frexp(x.abs())
    binade = (e - 1).clamp(min=min_exp, max=max_exp)
    # Spacing of representable values inside the binade (subnormals share min_exp's).
    step = torch.ldexp(torch.ones_like(x), binade - mant_bits)
    q = _round_int(x / step, rounding) * step  # exact: step is a power of two
    q = q.clamp(min=-max_value, max=max_value)
    q = torch.where(x == 0, x, q)  # keep signed zero
    return torch.where(torch.isnan(x), x, q)


def e4m3_round(x: torch.Tensor, rounding: str = "rne") -> torch.Tensor:
    """FP32 -> E4M3 -> FP32 with saturation at 448."""
    return _minifloat_rne_sat(
        x, _E4M3_MIN_EXP, _E4M3_MAX_EXP, _E4M3_MANT_BITS, E4M3_MAX, rounding
    )


def e2m1_round(x: torch.Tensor, rounding: str = "rne") -> torch.Tensor:
    """FP32 -> E2M1 -> FP32 with saturation at 6."""
    return _minifloat_rne_sat(
        x, _E2M1_MIN_EXP, _E2M1_MAX_EXP, _E2M1_MANT_BITS, E2M1_MAX, rounding
    )


def qdq_e4m3(x: torch.Tensor, rounding: str = "rne") -> torch.Tensor:
    """Elementwise FP8 quant-dequant of the scaled probability."""
    return e4m3_round(x, rounding)


def nvfp4_block_scale(x: torch.Tensor, rounding: str = "rne") -> torch.Tensor:
    """E4M3 scale code of each 16-key block: E4M3(448 * block_max). Shape (..., n_blocks)."""
    assert x.dtype == torch.float32 and x.shape[-1] % NVFP4_BLOCK == 0
    blocks = x.reshape(*x.shape[:-1], -1, NVFP4_BLOCK)
    block_max = blocks.amax(dim=-1)
    return e4m3_round(
        block_max * torch.tensor(NVFP4_SCALE_FACTOR, dtype=torch.float32), rounding
    )


def qdq_nvfp4(x: torch.Tensor, rounding: str = "rne") -> torch.Tensor:
    """NVFP4 quant-dequant along the last (key) axis, in blocks of 16.

    Per block: sigma = E4M3(448 * max); e = sigma / 2688; code = E2M1(x / e);
    result = code * e. A block whose sigma is 0 becomes 0.
    """
    assert x.dtype == torch.float32 and x.shape[-1] % NVFP4_BLOCK == 0
    blocks = x.reshape(*x.shape[:-1], -1, NVFP4_BLOCK)
    sigma = nvfp4_block_scale(x, rounding).unsqueeze(-1)
    eff = sigma / torch.tensor(NVFP4_GLOBAL_SCALE, dtype=torch.float32)
    safe_eff = torch.where(sigma == 0, torch.ones_like(eff), eff)
    code = e2m1_round(blocks / safe_eff, rounding)
    out = torch.where(sigma == 0, torch.zeros_like(blocks), code * safe_eff)
    return out.reshape(x.shape)


# Reference codec of each codec name: (FP32 codec inputs with keys on the last axis, rounding)
# -> quant-dequant. A new codec adds an entry here next to its kernel helper.
ORACLE_CODECS = {"e4m3": qdq_e4m3, "nvfp4": qdq_nvfp4}


def qdq(x: torch.Tensor, p_quant, rounding: str = "rne") -> torch.Tensor:
    """Apply the codec of ``p_quant`` (a name or a ``PQuantFormat``) to codec inputs ``x``.

    ``x`` is the scaled running-max-relative probability (already multiplied by
    the format's ``2**max_offset``), FP32, keys on the last axis.
    """
    fmt = resolve_p_quant(p_quant)
    if fmt is None or fmt.codec == "identity":
        return x
    if fmt.codec not in ORACLE_CODECS:
        raise ValueError(f"no oracle for codec {fmt.codec!r}")
    return ORACLE_CODECS[fmt.codec](x, rounding)
