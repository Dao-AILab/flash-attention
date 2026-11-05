"""Import-time compatibility across nvidia-cutlass-dsl releases.

Select implementations here, on the host at import time, so kernels use stable names.
"""

import re
from importlib.metadata import version

CUTEDSL_VERSION = tuple(
    int(part) for part in re.match(r"(\d+)\.(\d+)", version("nvidia-cutlass-dsl")).groups()
)

if CUTEDSL_VERSION >= (4, 8):
    from cutlass.memory import SmemAllocator, TmemAllocator, get_smem_capacity_in_bytes
else:
    from cutlass.utils import SmemAllocator, TmemAllocator, get_smem_capacity_in_bytes

__all__ = ["CUTEDSL_VERSION", "SmemAllocator", "TmemAllocator", "get_smem_capacity_in_bytes"]
