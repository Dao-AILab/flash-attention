"""Select the AMD attention backend for build and runtime imports.

On ROCm, FlashAttention has two backends: its own CK extension and aiter. aiter has two
implementations: its Triton kernels and its general MHA API (FMHA v3 or CK, per call).
"""

import importlib
import os
import warnings
from typing import Literal, NamedTuple, Optional

AmdBackend = Literal["ck", "aiter"]
AiterImpl = Literal["triton", "general"]


class AmdSelection(NamedTuple):
    backend: AmdBackend
    aiter_impl: Optional[AiterImpl]  # None when the backend is ck


def select_amd_backend(is_rocm: Optional[bool] = None) -> AmdSelection:
    """Select the requested backend, applying runtime checks when ROCm is known.

    FLASH_ATTENTION_TRITON_AMD_ENABLE selects aiter's Triton implementation and
    FLASH_ATTENTION_AITER_AMD_ENABLE selects aiter's general implementation.
    """
    triton = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
    aiter = os.getenv("FLASH_ATTENTION_AITER_AMD_ENABLE", "FALSE") == "TRUE"
    if triton and aiter:
        raise ValueError(
            "Select only one of FLASH_ATTENTION_TRITON_AMD_ENABLE "
            "and FLASH_ATTENTION_AITER_AMD_ENABLE"
        )
    if aiter:
        if is_rocm is False:
            raise RuntimeError("FLASH_ATTENTION_AITER_AMD_ENABLE requires ROCm")
        return AmdSelection("aiter", "general")
    if triton:
        return AmdSelection("aiter", "triton")
    if is_rocm:
        try:
            importlib.import_module("flash_attn_2_cuda")
        except ImportError:
            warnings.warn(
                "flash_attn_2_cuda (which has ROCm/HIP kernels) not found, "
                "falling back to Triton implementation"
            )
            return AmdSelection("aiter", "triton")
    return AmdSelection("ck", None)
