"""Select the AMD attention backend for build and runtime imports."""

import importlib
import os
import warnings
from typing import Literal, Optional

AmdBackend = Literal["ck", "triton", "aiter"]


def select_amd_backend(is_rocm: Optional[bool] = None) -> AmdBackend:
    """Select the requested mode, applying runtime checks when ROCm is known."""
    triton = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
    aiter = os.getenv("FLASH_ATTENTION_AITER_AMD_ENABLE", "FALSE") == "TRUE"
    if triton and aiter:
        raise ValueError(
            "Select only one of FLASH_ATTENTION_TRITON_AMD_ENABLE "
            "and FLASH_ATTENTION_AITER_AMD_ENABLE"
        )
    backend: AmdBackend = "aiter" if aiter else "triton" if triton else "ck"
    if backend == "aiter" and is_rocm is False:
        raise RuntimeError("FLASH_ATTENTION_AITER_AMD_ENABLE requires ROCm")
    if backend == "ck" and is_rocm:
        try:
            importlib.import_module("flash_attn_2_cuda")
        except ImportError:
            warnings.warn(
                "flash_attn_2_cuda (which has ROCm/HIP kernels) not found, "
                "falling back to Triton implementation"
            )
            return "triton"
    return backend
