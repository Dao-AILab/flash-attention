"""Native MXFP8 PV API guards (no kernel launch)."""

import pytest
import torch

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute import interface
from flash_attn.cute.blockscaled_utils import quantize_v_to_fp8_with_sf
from flash_attn.cute.interface import BlockScales


@pytest.fixture
def pv_inputs():
    q = torch.zeros(1, 128, 1, 64, dtype=torch.float8_e4m3fn)
    k = torch.zeros(1, 129, 1, 64, dtype=torch.float8_e4m3fn)
    v, sfv, _ = quantize_v_to_fp8_with_sf(torch.ones(1, 129, 1, 64))
    sfq = torch.full((512,), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sfk = torch.full((1024,), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return q, k, v, BlockScales(sfq, sfk, sfv)


@pytest.mark.parametrize("missing", [("q",), ("k",), ("q", "k")])
def test_native_pv_requires_qk_scales(pv_inputs, missing):
    q, k, v, scales = pv_inputs
    scales = scales._replace(**dict.fromkeys(missing))
    with pytest.raises(ValueError, match="requires both scales.q and scales.k"):
        interface.flash_attn_func(q, k, v, scales=scales)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"causal": True}, "noncausal"),
        ({"window_size": (None, 0)}, "noncausal"),
        ({"window_size": (32, 32)}, "nonlocal"),
        ({"num_splits": 2}, "unsplit"),
        ({"learnable_sink": torch.ones(1)}, "sinks"),
        ({"score_mod": lambda x: x}, "modifiers"),
        ({"mask_mod": lambda x: x}, "modifiers"),
        ({"softcap": 10.0}, "modifiers"),
        ({"aux_tensors": [torch.ones(1)]}, "auxiliary"),
        ({"gather_kv_indices": torch.zeros(1, dtype=torch.int32)}, "dense"),
        ({"block_sparse_tensors": object()}, "dense"),
    ],
)
def test_native_pv_rejects_unsupported_public_modes(
    pv_inputs, kwargs, match
):
    q, k, v, scales = pv_inputs
    with pytest.raises(NotImplementedError, match=match):
        interface.flash_attn_func(q, k, v, scales=scales, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cu_seqlens_q": torch.tensor([0, 128], dtype=torch.int32)},
        {"cu_seqlens_k": torch.tensor([0, 129], dtype=torch.int32)},
        {"seqused_q": torch.tensor([128], dtype=torch.int32)},
        {"seqused_k": torch.tensor([129], dtype=torch.int32)},
        {"page_table": torch.zeros(1, 1, dtype=torch.int32)},
    ],
)
def test_native_pv_rejects_unsupported_forward_modes(pv_inputs, kwargs):
    q, k, v, scales = pv_inputs
    with pytest.raises(NotImplementedError, match="Native MXFP8 PV"):
        interface._flash_attn_fwd(q, k, v, scales=scales, **kwargs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_native_pv_requires_e4m3_v(pv_inputs, dtype):
    q, k, v, scales = pv_inputs
    with pytest.raises(ValueError, match="requires e4m3 Q/K/V"):
        interface.flash_attn_func(q, k, v.to(dtype), scales=scales)


@pytest.mark.parametrize("invalid", ["dtype", "size", "contiguity", "device"])
def test_native_pv_validates_sfv_buffer(pv_inputs, invalid):
    q, k, v, scales = pv_inputs
    sfv = scales.v
    if invalid == "dtype":
        sfv = sfv.view(torch.uint8)
    elif invalid == "size":
        sfv = sfv.flatten()[:-1]
    elif invalid == "contiguity":
        sfv = sfv.transpose(-1, -2)
    else:
        sfv = torch.empty(sfv.shape, dtype=sfv.dtype, device="meta")
    with pytest.raises(ValueError, match="scales.v must be"):
        interface.flash_attn_func(q, k, v, scales=scales._replace(v=sfv))


def test_native_pv_rejects_autograd(pv_inputs):
    q, k, v, scales = pv_inputs
    q.requires_grad_()
    with pytest.raises(NotImplementedError, match="forward-only"):
        interface.flash_attn_func(q, k, v, scales=scales)


def test_qk_only_still_requires_bf16_v(pv_inputs):
    q, k, v, scales = pv_inputs
    with pytest.raises(ValueError, match="QK MXFP8 requires e4m3 Q/K and bf16 V"):
        interface.flash_attn_func(q, k, v, scales=scales._replace(v=None))
