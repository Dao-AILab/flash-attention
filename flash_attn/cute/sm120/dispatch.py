# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dispatch bridge to the SM12x forward and backward hosts.

The FA4 interface dispatches through this module, so the SM12x implementation and its launch
state stay in ``flash_attn/cute/sm120``.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def get_forward_host(arch: int):
    """Return the optional architecture-owned forward host."""
    if arch // 10 == 12:
        from flash_attn.cute.sm120.runtime import (
            sm120_forward_host,
        )

        return sm120_forward_host
    return None


@lru_cache(maxsize=None)
def get_backward_host(arch: int):
    """Return the optional architecture-owned backward, a module exposing bwd_config(inputs)
    and flash_attn_bwd(...)."""
    if arch // 10 == 12:
        from flash_attn.cute.sm120 import bwd_runtime

        return bwd_runtime
    return None
