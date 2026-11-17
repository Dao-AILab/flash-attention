"""Formats for fake quantization (quant-dequant) of attention probabilities.

Host-only: no CuTe DSL imports, so API validation, kernel construction and the
test oracles can all read the same table.

A format fixes two things:

* the codec applied to the P@V operand (the row sum and LSE never see it), and
* the softmax scaling policy the codec needs: ``max_offset`` (log2 of the
  factor applied to the running-max-relative probability) and
  ``rescale_threshold`` (0 means the row max is adopted whenever it moves, so
  the scaled probability never exceeds ``2**max_offset``).
"""

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = [
    "E2M1_MAX",
    "E4M3_MAX",
    "NVFP4_BLOCK",
    "NVFP4_GLOBAL_SCALE",
    "NVFP4_SCALE_FACTOR",
    "PQuantFormat",
    "p_quant_format_names",
    "register_p_quant_codec",
    "register_p_quant_format",
    "resolve_p_quant",
    "unregister_p_quant_format",
]

E4M3_MAX = 448.0
E2M1_MAX = 6.0
NVFP4_BLOCK = 16
# Two-level NVFP4 recipe with a global amax of 1: the block scale code is
# E4M3(448 * block_max) and the effective scale is code / (448 * 6).
NVFP4_SCALE_FACTOR = E4M3_MAX
NVFP4_GLOBAL_SCALE = E4M3_MAX * E2M1_MAX

# Codec names a format may use. "identity" needs no kernel helper; every other name needs one
# in flash_attn.cute.softmax._P_QUANT_CODECS. register_p_quant_codec adds a name.
_CODECS = {"identity", "e4m3", "nvfp4"}


@dataclass(frozen=True)
class PQuantFormat:
    """One quant-dequant format for the attention probabilities.

    name: public name, also the compile-key entry.
    codec: "e4m3" (elementwise FP8), "nvfp4" (E2M1 codes with one E4M3 scale per
        ``block`` keys) or "identity" (no rounding; a control that keeps the policy).
    block: keys that share one scale; 1 for elementwise codecs.
    max_offset: integer log2 of the probability scale fed to the codec.
    rescale_threshold: row-max lag in log2 units; quantized formats need 0.
    input_max: largest codec input the format represents without saturating.
    diagnostic: True for formats that exist only as test controls.
    """

    name: str
    codec: str
    block: int
    max_offset: int
    rescale_threshold: float
    input_max: float
    diagnostic: bool = False

    def __post_init__(self):
        if self.codec not in _CODECS:
            raise ValueError(
                f"unknown P quant codec {self.codec!r}; expected one of {sorted(_CODECS)}"
            )
        if self.block < 1 or self.block & (self.block - 1):
            raise ValueError(f"P quant block must be a power of two, got {self.block}")
        if not isinstance(self.max_offset, int):
            raise ValueError(
                f"P quant max_offset must be an integer, got {self.max_offset!r}; "
                "fractional offsets are not supported"
            )
        if self.rescale_threshold < 0:
            raise ValueError("P quant rescale_threshold must be non-negative")
        # The scaled probability can reach 2**(max_offset + rescale_threshold):
        # a lagging row max lets it exceed 2**max_offset by the threshold.
        scaled_max = 2.0 ** (self.max_offset + self.rescale_threshold)
        if scaled_max > self.input_max:
            raise ValueError(
                f"P quant format {self.name!r}: scaled probability can reach {scaled_max:g}, "
                f"above the format's input range {self.input_max:g}"
            )


_BUILTIN_FORMATS = (
    PQuantFormat("fp8_e4m3", "e4m3", 1, 8, 0.0, E4M3_MAX),
    PQuantFormat("nvfp4", "nvfp4", NVFP4_BLOCK, 0, 0.0, 1.0),
    # Controls: same scaling policy as the format they mirror, no rounding.
    PQuantFormat("identity_fp8_e4m3", "identity", 1, 8, 0.0, E4M3_MAX, diagnostic=True),
    PQuantFormat("identity_nvfp4", "identity", NVFP4_BLOCK, 0, 0.0, 1.0, diagnostic=True),
)

_FORMATS: Dict[str, PQuantFormat] = {fmt.name: fmt for fmt in _BUILTIN_FORMATS}
_NONE_NAMES = (None, "none")


def p_quant_format_names(include_diagnostic: bool = False):
    """Names accepted by ``resolve_p_quant`` (besides None / "none")."""
    return sorted(
        name for name, fmt in _FORMATS.items() if include_diagnostic or not fmt.diagnostic
    )


def resolve_p_quant(p_quant) -> Optional[PQuantFormat]:
    """Map a ``p_quant`` argument to its registered format; None and "none" mean no
    quantization. A ``PQuantFormat`` object is accepted only if it is the registered entry
    of its name, so an unregistered or altered recipe cannot reach a kernel."""
    if p_quant in _NONE_NAMES:
        return None
    name = p_quant.name if isinstance(p_quant, PQuantFormat) else p_quant
    if not isinstance(name, str) or name not in _FORMATS:
        raise ValueError(
            f"unknown p_quant {name!r}; supported formats: {p_quant_format_names()} "
            "(or None / 'none')"
        )
    fmt = _FORMATS[name]
    if isinstance(p_quant, PQuantFormat) and p_quant != fmt:
        raise ValueError(f"p_quant format {name!r} does not match its registered recipe")
    return fmt


def register_p_quant_codec(name: str) -> None:
    """Allow formats to use codec ``name``. The kernel helper for it goes into
    ``flash_attn.cute.softmax._P_QUANT_CODECS[name]`` (a function from one block of FP32
    probabilities to its quant-dequant); no API or dispatch code changes."""
    _CODECS.add(name)


def register_p_quant_format(fmt: PQuantFormat) -> PQuantFormat:
    """Add a format to the table. Its codec must be one the kernel implements."""
    if fmt.name in _NONE_NAMES or fmt.name in _FORMATS:
        raise ValueError(f"P quant format name {fmt.name!r} is already taken")
    _FORMATS[fmt.name] = fmt
    return fmt


def unregister_p_quant_format(name: str) -> None:
    """Remove a format added with ``register_p_quant_format`` (built-ins stay)."""
    if any(fmt.name == name for fmt in _BUILTIN_FORMATS):
        raise ValueError(f"cannot remove built-in P quant format {name!r}")
    _FORMATS.pop(name, None)
