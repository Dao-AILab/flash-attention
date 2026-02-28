"""Shim that wraps our SM70-compatible flash_attn for use by vLLM.

vLLM v0.6.5's bundled vllm_flash_attn adds an `out` parameter to
flash_attn_varlen_func and flash_attn_with_kvcache. Our flash_attn
(built with SM70/V100 support) doesn't have this parameter, so we
wrap the functions to handle it.
"""

__version__ = "2.8.3"

import functools
from flash_attn.flash_attn_interface import (
    flash_attn_func as _flash_attn_func,
    flash_attn_varlen_func as _flash_attn_varlen_func,
    flash_attn_with_kvcache as _flash_attn_with_kvcache,
)


def flash_attn_varlen_func(*args, out=None, **kwargs):
    result = _flash_attn_varlen_func(*args, **kwargs)
    if out is not None:
        out.copy_(result)
        return out
    return result


def flash_attn_with_kvcache(*args, out=None, **kwargs):
    result = _flash_attn_with_kvcache(*args, **kwargs)
    if out is not None:
        out.copy_(result)
        return out
    return result


def flash_attn_func(*args, out=None, **kwargs):
    result = _flash_attn_func(*args, **kwargs)
    if out is not None:
        out.copy_(result)
        return out
    return result
