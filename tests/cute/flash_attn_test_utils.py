"""Shared utilities for FA4 CuTe tests."""

import gc
from functools import wraps

import torch

from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


def retry_on_oom(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            if "out of memory" in str(e).lower():
                if hasattr(_flash_attn_fwd, "compile_cache"):
                    _flash_attn_fwd.compile_cache.clear()
                if hasattr(_flash_attn_bwd, "compile_cache"):
                    _flash_attn_bwd.compile_cache.clear()
                gc.collect()
                torch.cuda.empty_cache()
                return func(*args, **kwargs)
            else:
                raise
    return wrapper

