"""
Infinity Cache (LLC) Aware Head Grouping for Flash Attention

This module provides functionality to optimize flash attention by processing
heads in groups that fit in the Last Level Cache (LLC / Infinity Cache).

AMD RDNA3 cache hierarchy:
- L2 Cache: 6 MB (per-die, fast)
- Infinity Cache (L3/LLC): 96 MB (acts as memory-side cache)

For large sequence lengths, we want K,V to fit in the 96 MB Infinity Cache.
By processing heads in groups that fit, we achieve up to 2x speedup.

Example: gfx1100 with 96MB Infinity Cache, 40 heads, seqlen=17160, head_dim=128
- K,V for all 40 heads = 352 MB (exceeds 96 MB LLC)
- K,V for 10 heads = 88 MB (fits in 96 MB LLC)
- Processing 10 heads at a time gives 1.95x speedup
"""

import os
from typing import Tuple, Dict
import torch

from .utils import get_arch

# Infinity Cache (LLC) sizes for AMD GPUs in bytes
# Note: This is the L3/Infinity Cache, NOT the L2 cache
# RDNA3: L2=6MB, Infinity Cache (LLC)=96MB
AMD_LLC_CACHE_SIZES: Dict[str, int] = {
    # RDNA2
    "gfx1030": 128 * 1024 * 1024,  # RX 6900 XT - 128 MB Infinity Cache
    # RDNA3 consumer
    "gfx1100": 96 * 1024 * 1024,   # RX 7900 XTX - 96 MB Infinity Cache
    "gfx1101": 64 * 1024 * 1024,   # RX 7800 XT - 64 MB Infinity Cache
    "gfx1102": 32 * 1024 * 1024,   # RX 7600 - 32 MB Infinity Cache
    # RDNA4
    "gfx1200": 32 * 1024 * 1024,   # RX 9060/XT - 32 MB Infinity Cache
    "gfx1201": 64 * 1024 * 1024,   # RX 9070/XT - 64 MB Infinity Cache
}

# Legacy alias for backwards compatibility
AMD_L2_CACHE_SIZES = AMD_LLC_CACHE_SIZES

# Environment variable to override LLC cache size (in MB)
LLC_CACHE_OVERRIDE_ENV = "FLASH_ATTN_LLC_CACHE_MB"
L2_CACHE_OVERRIDE_ENV = "FLASH_ATTN_L2_CACHE_MB"  # Legacy alias

# Environment variable to disable head grouping
DISABLE_HEAD_GROUPING_ENV = "FLASH_ATTN_DISABLE_HEAD_GROUPING"

# Cached LLC size per device
_llc_cache_size_cache: Dict[int, int] = {}


def get_llc_cache_size(device_index: int = 0) -> int:
    """
    Get Infinity Cache (LLC) size for the specified GPU device.
    
    For RDNA3, this is the 96 MB Infinity Cache, not the 6 MB L2.
    
    Returns:
        LLC cache size in bytes
    """
    global _llc_cache_size_cache
    
    if device_index in _llc_cache_size_cache:
        return _llc_cache_size_cache[device_index]
    
    # Check for environment override (new name first, then legacy)
    for env_var in [LLC_CACHE_OVERRIDE_ENV, L2_CACHE_OVERRIDE_ENV]:
        if env_var in os.environ:
            try:
                size_mb = int(os.environ[env_var])
                size_bytes = size_mb * 1024 * 1024
                _llc_cache_size_cache[device_index] = size_bytes
                return size_bytes
            except ValueError:
                pass
    
    # Get architecture using utils.get_arch()
    arch = get_arch().name
    
    # Check exact match first
    if arch in AMD_LLC_CACHE_SIZES:
        size = AMD_LLC_CACHE_SIZES[arch]
        _llc_cache_size_cache[device_index] = size
        return size
    
    # Check prefix match (e.g., gfx1100 matches gfx1100)
    for known_arch, size in AMD_LLC_CACHE_SIZES.items():
        if arch.startswith(known_arch):
            _llc_cache_size_cache[device_index] = size
            return size
    
    # Default: assume 96 MB (conservative for RDNA3)
    default_size = 96 * 1024 * 1024
    _llc_cache_size_cache[device_index] = default_size
    return default_size


# Legacy alias
get_l2_cache_size = get_llc_cache_size


def calculate_optimal_head_group_size(
    seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    device_index: int = 0,
    llc_utilization: float = 1.0  # Use 100% of LLC - optimal for long sequences
) -> int:
    """
    Calculate the optimal number of heads to process together to fit K,V in LLC.
    """
    llc_size = get_llc_cache_size(device_index)
    
    # Get element size in bytes
    if dtype in (torch.float16, torch.bfloat16):
        elem_size = 2
    elif dtype == torch.float32:
        elem_size = 4
    elif 'float8' in str(dtype).lower():
        elem_size = 1
    else:
        elem_size = 2  # Default to fp16
    
    # Memory for K and V per head
    kv_per_head = seqlen_k * head_dim * elem_size * 2  # *2 for K and V
    
    # Target LLC usage
    target_llc = int(llc_size * llc_utilization)
    
    # Calculate number of heads that fit
    if kv_per_head == 0:
        return 1
    
    head_group_size = max(1, target_llc // kv_per_head)
    
    return head_group_size


def is_head_grouping_beneficial(
    nheads: int,
    seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    device_index: int = 0,
    threshold_ratio: float = 1.5
) -> Tuple[bool, int]:
    """
    Determine if head grouping would be beneficial and return optimal group size.
    
    Head grouping is only beneficial for RDNA GPUs with Infinity Cache (LLC).
    CDNA GPUs (MI250, MI300, etc.) have different cache architectures.
    """
    # Check if disabled via environment
    if os.environ.get(DISABLE_HEAD_GROUPING_ENV, "0") == "1":
        return False, nheads
    
    # Only apply head grouping to RDNA GPUs (which have Infinity Cache)
    arch = get_arch()
    if not arch.is_rdna:
        return False, nheads
    
    llc_size = get_llc_cache_size(device_index)
    
    # Get element size
    if dtype in (torch.float16, torch.bfloat16):
        elem_size = 2
    elif dtype == torch.float32:
        elem_size = 4
    elif 'float8' in str(dtype).lower():
        elem_size = 1
    else:
        elem_size = 2
    
    # Total K,V memory for all heads
    total_kv = nheads * seqlen_k * head_dim * elem_size * 2
    
    # Only group if K,V significantly exceeds LLC
    if total_kv < llc_size * threshold_ratio:
        return False, nheads
    
    # Calculate optimal group size
    group_size = calculate_optimal_head_group_size(
        seqlen_k, head_dim, dtype, device_index
    )
    
    # Only group if we'd have at least 2 groups
    if group_size >= nheads:
        return False, nheads
    
    # Minimum group size to avoid excessive kernel launches
    min_group_size = max(1, nheads // 16)  # At most 16 groups
    group_size = max(group_size, min_group_size)
    
    return True, min(group_size, nheads)


def print_head_grouping_info(
    nheads: int,
    seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    device_index: int = 0
):
    """Print diagnostic information about head grouping."""
    llc_size = get_llc_cache_size(device_index)
    arch = get_arch()
    
    if dtype in (torch.float16, torch.bfloat16):
        elem_size = 2
    elif dtype == torch.float32:
        elem_size = 4
    elif 'float8' in str(dtype).lower():
        elem_size = 1
    else:
        elem_size = 2
    
    total_kv = nheads * seqlen_k * head_dim * elem_size * 2
    should_group, group_size = is_head_grouping_beneficial(
        nheads, seqlen_k, head_dim, dtype, device_index
    )
    
    print(f"\n=== Infinity Cache (LLC) Aware Head Grouping ===")
    print(f"GPU: {arch.name}")
    print(f"Infinity Cache (LLC): {llc_size / (1024*1024):.1f} MB")
    print(f"Heads: {nheads}, SeqLen: {seqlen_k}, HeadDim: {head_dim}")
    print(f"Total K,V Memory: {total_kv / (1024*1024):.1f} MB")
    print(f"LLC Ratio: {total_kv / llc_size:.2f}x")
    print(f"Should Group: {should_group}")
    if should_group:
        kv_per_group = group_size * seqlen_k * head_dim * elem_size * 2
        num_groups = (nheads + group_size - 1) // group_size
        print(f"Group Size: {group_size} heads ({num_groups} groups)")
        print(f"K,V per Group: {kv_per_group / (1024*1024):.1f} MB")
    print("=" * 48 + "\n")
