"""
L2 Cache-Aware Head Grouping for Flash Attention

This module provides functionality to optimize flash attention by processing
heads in groups that fit in the L2 cache. This is particularly important for
consumer AMD GPUs like gfx1100 (RX 7900 XTX) where the L2 cache is smaller
than datacenter GPUs.

The key insight is that for large sequence lengths, the K and V tensors for
all heads may exceed L2 cache capacity, causing cache thrashing. By processing
heads in groups that fit in L2, we can achieve up to 2x speedup.

Example: gfx1100 with 96MB L2, 40 heads, seqlen=17160, head_dim=128
- K,V for all 40 heads = 352 MB (exceeds 96 MB L2)
- K,V for 10 heads = 88 MB (fits in 96 MB L2)
- Processing 10 heads at a time gives 1.95x speedup
"""

import os
import functools
from typing import Optional, Tuple, Dict
import torch

# L2 cache sizes for AMD GPUs in bytes
# Source: AMD documentation and hardware specs
AMD_L2_CACHE_SIZES: Dict[str, int] = {
    # RDNA3 workstaion
    "gfx1100": 96 * 1024 * 1024,   # RX 7900 XTX/XT - 96 MB
}

# Environment variable to override L2 cache size (in MB)
L2_CACHE_OVERRIDE_ENV = "FLASH_ATTN_L2_CACHE_MB"
# Environment variable to disable head grouping
DISABLE_HEAD_GROUPING_ENV = "FLASH_ATTN_DISABLE_HEAD_GROUPING"

# Cached L2 size per device
_l2_cache_size_cache: Dict[int, int] = {}


@functools.lru_cache(maxsize=None)
def get_gcn_arch_name(device_index: int = 0) -> str:
    """Get the GCN architecture name for an AMD GPU."""
    try:
        props = torch.cuda.get_device_properties(device_index)
        if hasattr(props, 'gcnArchName'):
            return props.gcnArchName
        # Fallback: try to get from name
        name = props.name.lower()
        if 'gfx' in name:
            # Extract gfxXXXX from name
            import re
            match = re.search(r'gfx\d+', name)
            if match:
                return match.group()
    except Exception:
        pass
    return "unknown"


def get_l2_cache_size(device_index: int = 0) -> int:
    """
    Get L2 cache size for the specified GPU device.
    
    Returns:
        L2 cache size in bytes
    """
    global _l2_cache_size_cache
    
    if device_index in _l2_cache_size_cache:
        return _l2_cache_size_cache[device_index]
    
    # Check for environment override
    if L2_CACHE_OVERRIDE_ENV in os.environ:
        try:
            size_mb = int(os.environ[L2_CACHE_OVERRIDE_ENV])
            size_bytes = size_mb * 1024 * 1024
            _l2_cache_size_cache[device_index] = size_bytes
            return size_bytes
        except ValueError:
            pass
    
    # Get architecture and look up cache size
    arch = get_gcn_arch_name(device_index)
    
    # Check exact match first
    if arch in AMD_L2_CACHE_SIZES:
        size = AMD_L2_CACHE_SIZES[arch]
        _l2_cache_size_cache[device_index] = size
        return size
    
    # Check prefix match (e.g., gfx1100 matches gfx1100)
    for known_arch, size in AMD_L2_CACHE_SIZES.items():
        if arch.startswith(known_arch):
            _l2_cache_size_cache[device_index] = size
            return size
    
    # Default: assume 96 MB (conservative for RDNA3)
    default_size = 96 * 1024 * 1024
    _l2_cache_size_cache[device_index] = default_size
    return default_size


def calculate_optimal_head_group_size(
    seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    device_index: int = 0,
    l2_utilization: float = 1.0  #use higher utilization by default to improve 1280x720 performance
) -> int:
    """
    Calculate the optimal number of heads to process together to fit K,V in L2.
    
    The calculation is:
        K,V memory for N heads = N * seqlen_k * head_dim * dtype_size * 2 (for K and V)
        
    We want: K,V memory <= L2_cache * utilization
    So: N <= (L2_cache * utilization) / (seqlen_k * head_dim * dtype_size * 2)
    
    Args:
        seqlen_k: Sequence length of K/V
        head_dim: Head dimension
        dtype: Data type of tensors
        device_index: GPU device index
        l2_utilization: Fraction of L2 to target (default 0.9 to leave room for Q)
    
    Returns:
        Optimal number of heads to process together (minimum 1)
    """
    l2_size = get_l2_cache_size(device_index)
    
    # Get element size in bytes
    if dtype in (torch.float16, torch.bfloat16):
        elem_size = 2
    elif dtype == torch.float32:
        elem_size = 4
    elif hasattr(torch, 'float8_e4m3fnuz') and dtype == torch.float8_e4m3fnuz:
        elem_size = 1
    elif hasattr(torch, 'float8_e5m2fnuz') and dtype == torch.float8_e5m2fnuz:
        elem_size = 1
    elif hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
        elem_size = 1
    elif hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
        elem_size = 1
    elif 'float8' in str(dtype).lower():
        elem_size = 1
    else:
        elem_size = 2  # Default to fp16
    
    # Memory for K and V per head
    kv_per_head = seqlen_k * head_dim * elem_size * 2  # *2 for K and V
    
    # Target L2 usage (leave some room for Q and other data)
    target_l2 = int(l2_size * l2_utilization)
    
    # Calculate number of heads that fit
    if kv_per_head == 0:
        return 1
    
    head_group_size = max(1, target_l2 // kv_per_head)
    
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
    
    Head grouping is beneficial when:
    1. Total K,V memory exceeds L2 cache size by a significant margin
    2. Processing in groups allows K,V to fit in L2
    3. The overhead of multiple kernel launches is worth the cache benefit
    
    Args:
        nheads: Number of attention heads
        seqlen_k: Sequence length of K/V
        head_dim: Head dimension
        dtype: Data type
        device_index: GPU device index
        threshold_ratio: K,V must exceed L2 by this ratio to enable grouping
        
    Returns:
        (should_group, group_size): Whether to group and the optimal group size
    """
    # Check if disabled via environment
    if os.environ.get(DISABLE_HEAD_GROUPING_ENV, "0") == "1":
        return False, nheads
    
    l2_size = get_l2_cache_size(device_index)
    
    # Get element size
    if dtype in (torch.float16, torch.bfloat16):
        elem_size = 2
    elif dtype == torch.float32:
        elem_size = 4
    elif hasattr(torch, 'float8_e4m3fnuz') and dtype == torch.float8_e4m3fnuz:
        elem_size = 1
    elif hasattr(torch, 'float8_e5m2fnuz') and dtype == torch.float8_e5m2fnuz:
        elem_size = 1
    elif hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
        elem_size = 1
    elif hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
        elem_size = 1
    elif 'float8' in str(dtype).lower():
        elem_size = 1
    else:
        elem_size = 2
    
    # Total K,V memory for all heads
    total_kv = nheads * seqlen_k * head_dim * elem_size * 2
    
    # Only group if K,V significantly exceeds L2
    if total_kv < l2_size * threshold_ratio:
        return False, nheads
    
    # Calculate optimal group size
    group_size = calculate_optimal_head_group_size(
        seqlen_k, head_dim, dtype, device_index
    )
    
    # Only group if we'd have at least 2 groups
    # (otherwise grouping adds overhead with no benefit)
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
    l2_size = get_l2_cache_size(device_index)
    arch = get_gcn_arch_name(device_index)
    
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
    
    print(f"\n=== L2 Cache-Aware Head Grouping ===")
    print(f"GPU: {arch}")
    print(f"L2 Cache: {l2_size / (1024*1024):.1f} MB")
    print(f"Heads: {nheads}, SeqLen: {seqlen_k}, HeadDim: {head_dim}")
    print(f"Total K,V Memory: {total_kv / (1024*1024):.1f} MB")
    print(f"L2 Ratio: {total_kv / l2_size:.2f}x")
    print(f"Should Group: {should_group}")
    if should_group:
        kv_per_group = group_size * seqlen_k * head_dim * elem_size * 2
        num_groups = (nheads + group_size - 1) // group_size
        print(f"Group Size: {group_size} heads ({num_groups} groups)")
        print(f"K,V per Group: {kv_per_group / (1024*1024):.1f} MB")
    print("=" * 40 + "\n")
