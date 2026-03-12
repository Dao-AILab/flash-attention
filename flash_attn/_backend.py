"""Backend dispatch for flash attention v2.

ROCm: routes through aiter (CK by default, triton if FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE).
CUDA: uses compiled flash_attn_2_cuda C++ extension.
"""

import os
import torch

IS_ROCM = torch.version.hip is not None

if IS_ROCM:
    if os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE":
        from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2
    else:
        from aiter.ops.ck.flash_attn_ck_amd import flash_attn_2
else:
    import flash_attn_2_cuda as flash_attn_2
