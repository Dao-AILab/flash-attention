"""Backend dispatch for flash attention v3.

ROCm: routes through aiter (triton backend).
CUDA: uses compiled flash_attn_3._C extension.
"""

import torch

IS_ROCM = torch.version.hip is not None

if IS_ROCM:
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
else:
    import flash_attn_3._C  # Registers operators with PyTorch

    flash_attn_3 = torch.ops.flash_attn_3
