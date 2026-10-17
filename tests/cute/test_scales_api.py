"""scales= argument guards (CPU tensors, no kernel launch)."""

import pytest
import torch

from flash_attn.cute import BlockScales, TensorwiseScales, flash_attn_func, flash_attn_varlen_func


def fp8_inputs(dtype=torch.float8_e4m3fn):
    return [torch.zeros(2, 128, heads, 64, dtype=dtype) for heads in (4, 2, 2)]


@pytest.mark.parametrize("entry", [flash_attn_func, flash_attn_varlen_func])
def test_public_entries_forward_scales(entry):
    q, k, v = fp8_inputs(torch.bfloat16)
    with pytest.raises(AssertionError, match="only supported for FP8"):
        entry(q, k, v, scales=TensorwiseScales(q=torch.ones(2, 2)))


def test_tensorwise_scales_are_per_batch_and_kv_head():
    q, k, v = fp8_inputs()
    with pytest.raises(AssertionError, match=r"scales.k shape .* != expected \(2, 2\)"):
        flash_attn_func(q, k, v, scales=TensorwiseScales(k=torch.ones(2, 4)))


def test_scales_format_is_the_type():
    q, k, v = fp8_inputs()
    with pytest.raises(TypeError, match="scales must be"):
        flash_attn_func(q, k, v, scales=(torch.ones(2, 2),) * 3)


def test_block_scales_are_dense_only():
    q, k, v = fp8_inputs()
    sf = torch.ones(512).to(torch.float8_e8m0fnu)
    with pytest.raises(NotImplementedError, match="use flash_attn_func"):
        flash_attn_varlen_func(q, k, v.to(torch.bfloat16), scales=BlockScales(sf, sf))
