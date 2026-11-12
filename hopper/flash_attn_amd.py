"""Select the AMD attention backend, for the build and runtime imports.

On ROCm, FlashAttention has two backends: its own CK extension and aiter. aiter has two
implementations: its Triton kernels and its general MHA API (FMHA v3 or CK, per call).
flash_attn/flash_attn_amd.py (FA2) and hopper/flash_attn_amd.py (FA3) are the same file.
"""

import importlib
import warnings
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Union


@dataclass(frozen=True)
class Ck:
    """FlashAttention's compiled extension: CK on ROCm, CUDA otherwise."""


@dataclass(frozen=True)
class Aiter:
    impl: Literal["triton", "general"]


AmdBackend = Union[Ck, Aiter]

# FlashAttention's compiled extension for each version: CK kernels on ROCm, CUDA otherwise.
# FA3 has no CK extension, so on ROCm it always uses aiter.
CK_MODULES = {2: "flash_attn_2_cuda", 3: None}


def select_amd_backend(
    env: Mapping[str, str], torch_version=None, fa_version: Optional[Literal[2, 3]] = None
) -> AmdBackend:
    """FLASH_ATTENTION_TRITON_AMD_ENABLE selects aiter's Triton implementation and
    FLASH_ATTENTION_AITER_AMD_ENABLE selects aiter's general implementation.

    The runtime passes torch.version and its FlashAttention version. aiter then requires ROCm. On
    ROCm, FA3 uses aiter's Triton implementation, and FA2 without its extension falls back to it.
    FA2's setup.py passes only env, since its extension does not exist yet during the build; FA3
    has no CK extension to look for, so hopper/setup.py selects as its runtime does.
    """
    if (torch_version is None) != (fa_version is None):
        raise TypeError("pass torch_version and fa_version together")
    triton = env.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
    general = env.get("FLASH_ATTENTION_AITER_AMD_ENABLE", "FALSE") == "TRUE"
    if triton and general:
        raise ValueError(
            "Select only one of FLASH_ATTENTION_TRITON_AMD_ENABLE "
            "and FLASH_ATTENTION_AITER_AMD_ENABLE"
        )
    backend = Aiter("triton") if triton else Aiter("general") if general else Ck()
    if torch_version is None:
        return backend
    is_rocm = getattr(torch_version, "hip", None) is not None
    if isinstance(backend, Aiter) and not is_rocm:
        raise RuntimeError("the aiter backend requires ROCm")
    if isinstance(backend, Ck) and is_rocm and CK_MODULES[fa_version] is None:
        return Aiter("triton")
    if isinstance(backend, Ck) and is_rocm:
        try:
            importlib.import_module(CK_MODULES[fa_version])
        except ImportError:
            warnings.warn(
                f"{CK_MODULES[fa_version]} not found, falling back to aiter's Triton implementation"
            )
            return Aiter("triton")
    return backend
