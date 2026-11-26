"""Import-time compatibility across nvidia-cutlass-dsl releases.

Select implementations here, on the host at import time, so kernels use stable names.
"""

import re
from importlib.metadata import version

CUTEDSL_VERSION = tuple(
    int(part) for part in re.match(r"(\d+)\.(\d+)", version("nvidia-cutlass-dsl")).groups()
)

# 4.8 moved these out of cutlass.utils; the old names emit DSLDeprecationWarning from 4.8 on.
if CUTEDSL_VERSION >= (4, 8):
    from cutlass.memory import SmemAllocator, TmemAllocator, get_smem_capacity_in_bytes
    from cutlass.tensor_utils import LayoutEnum, TensorMapManager, TensorMapUpdateMode
else:
    from cutlass.utils import SmemAllocator, TmemAllocator, get_smem_capacity_in_bytes
    from cutlass.utils import LayoutEnum, TensorMapManager, TensorMapUpdateMode

__all__ = [
    "CUTEDSL_VERSION",
    "LayoutEnum",
    "SmemAllocator",
    "TensorMapManager",
    "TensorMapUpdateMode",
    "TmemAllocator",
    "get_smem_capacity_in_bytes",
]
