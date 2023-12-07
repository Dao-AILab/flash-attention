# Copyright (c) 2023, Tri Dao.

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from flash_attn.ops.triton.layernorm import (
    layer_norm_fn,
    layer_norm_ref,
    rms_norm_ref,
    layer_norm_linear_fn,
)


is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize("prenorm", [True, False])
# @pytest.mark.parametrize("prenorm", [True])
@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize("is_rms_norm", [True])
@pytest.mark.parametrize("has_residual", [True, False])
# @pytest.mark.parametrize("has_residual", [False])
@pytest.mark.parametrize(
    "weight_dtype", [torch.float32, torch.float16] + ([torch.bfloat16] if is_sm8x else [])
)
# @pytest.mark.parametrize("weight_dtype", [torch.float32])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize("input_dtype,residual_dtype", [(torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("hidden_size", [192, 2048, 2560, 3000, 8192])
# @pytest.mark.parametrize("hidden_size", [256])
def test_layer_norm(
    hidden_size, input_dtype, residual_dtype, weight_dtype, has_residual, is_rms_norm, prenorm
):
    device = "cuda"
    if any(x == torch.bfloat16 for x in [input_dtype, residual_dtype, weight_dtype]):
        atol = 5e-2
    elif any(x == torch.float16 for x in [input_dtype, residual_dtype, weight_dtype]):
        atol = 1e-2
    else:
        atol = 1e-4
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    # batch_size = 1
    # seqlen = 1
    layer_norm_ref_fn = layer_norm_ref if not is_rms_norm else rms_norm_ref
    allclose = (
        lambda x, x_pt, x_ref, atol=atol: (x - x_ref).abs().max()
        <= 2 * (x_pt - x_ref).abs().max() + atol
    )
    x0 = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0_pt = x0.detach().clone().requires_grad_()
    x0_ref = x0.detach().clone().requires_grad_()
    if has_residual:
        res = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        res_pt = res.detach().clone().requires_grad_()
        res_ref = res.detach().clone().requires_grad_()
    else:
        res, res_pt, res_ref = None, None, None
    weight = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
    if not is_rms_norm:
        bias = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
    else:
        bias = None
    weight_pt = weight.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_pt = bias.detach().clone().requires_grad_() if bias is not None else None
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None

    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    out, *rest = layer_norm_fn(
        x0,
        weight,
        bias,
        residual=res,
        eps=1e-6,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=is_rms_norm,
    )
    out_pt, *rest_pt = layer_norm_ref_fn(
        x0_pt, weight_pt, bias_pt, residual=res_pt, eps=1e-6, prenorm=prenorm
    )
    out_ref, *rest_ref = layer_norm_ref_fn(
        x0_ref, weight_ref, bias_ref, residual=res_ref, eps=1e-6, prenorm=prenorm, upcast=True
    )
    if prenorm:
        residual = rest[0]
        residual_pt = rest_pt[0]
        residual_ref = rest_ref[0]
    assert out.dtype == input_dtype
    if prenorm:
        assert residual.dtype == residual_dtype
        assert allclose(residual, residual_pt, residual_ref)
    assert allclose(out, out_pt, out_ref)

    g = torch.randn_like(out) / batch_size
    if not prenorm:
        out.backward(g)
        out_pt.backward(g)
        out_ref.backward(g)
    else:
        (out * F.sigmoid(residual)).backward(g)
        (out_pt * F.sigmoid(residual_pt)).backward(g)
        (out_ref * F.sigmoid(residual_ref.to(dtype=residual_dtype))).backward(g)
    assert allclose(x0.grad, x0_pt.grad, x0_ref.grad)
    if has_residual:
        assert allclose(res.grad, res_pt.grad, res_ref.grad)
    assert allclose(weight.grad, weight_pt.grad, weight_ref.grad)
    if bias is not None:
        assert allclose(bias.grad, bias_pt.grad, bias_ref.grad)


@pytest.mark.parametrize("prenorm", [True, False])
# @pytest.mark.parametrize("prenorm", [True])
@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize("is_rms_norm", [True])
@pytest.mark.parametrize("has_residual", [True, False])
# @pytest.mark.parametrize("has_residual", [False])
@pytest.mark.parametrize("weight_dtype", [torch.float32])
@pytest.mark.parametrize(
"input_dtype,residual_dtype",
[(torch.float16, torch.float16), (torch.float16, torch.float32)]
+ ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize("input_dtype,residual_dtype", [(torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("hidden_size", [192, 2048, 2560, 3000])
# @pytest.mark.parametrize("hidden_size", [256])
def test_layer_norm_linear(
    hidden_size, input_dtype, residual_dtype, weight_dtype, has_residual, is_rms_norm, prenorm
):
    device = "cuda"
    if any(x == torch.bfloat16 for x in [input_dtype, residual_dtype, weight_dtype]):
        atol = 5e-2
    elif any(x == torch.float16 for x in [input_dtype, residual_dtype, weight_dtype]):
        atol = 1e-2
    else:
        atol = 1e-4
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    seqlen = 512
    # batch_size = 1
    # seqlen = 1
    layer_norm_ref_fn = layer_norm_ref if not is_rms_norm else rms_norm_ref
    allclose = (
        lambda x, x_pt, x_ref, atol=atol: (x - x_ref).abs().max()
        <= 2 * (x_pt - x_ref).abs().max() + atol
    )
    x0 = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0_pt = x0.detach().clone().requires_grad_()
    x0_ref = x0.detach().clone().requires_grad_()
    if has_residual:
        res = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        res_pt = res.detach().clone().requires_grad_()
        res_ref = res.detach().clone().requires_grad_()
    else:
        res, res_pt, res_ref = None, None, None
    norm_weight = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
    if not is_rms_norm:
        norm_bias = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
    else:
        norm_bias = None
    norm_weight_pt = norm_weight.detach().clone().requires_grad_()
    norm_weight_ref = norm_weight.detach().clone().requires_grad_()
    norm_bias_pt = norm_bias.detach().clone().requires_grad_() if norm_bias is not None else None
    norm_bias_ref = norm_bias.detach().clone().requires_grad_() if norm_bias is not None else None
    linear_weight = torch.empty(
        2 * hidden_size, hidden_size, device=device, dtype=weight_dtype, requires_grad=True
    )
    torch.nn.init.xavier_uniform_(linear_weight)
    if not is_rms_norm:
        linear_bias = torch.randn(
            2 * hidden_size, device=device, dtype=weight_dtype, requires_grad=True
        )
    else:
        linear_bias = None
    linear_weight_pt = linear_weight.detach().clone().requires_grad_()
    linear_weight_ref = linear_weight.detach().clone().requires_grad_()
    linear_bias_pt = (
        linear_bias.detach().clone().requires_grad_() if linear_bias is not None else None
    )
    linear_bias_ref = (
        linear_bias.detach().clone().requires_grad_() if linear_bias is not None else None
    )

    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    with torch.autocast(device_type="cuda", dtype=input_dtype):
        out, *rest = layer_norm_linear_fn(
            x0,
            norm_weight,
            norm_bias,
            linear_weight,
            linear_bias,
            residual=res,
            eps=1e-6,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=is_rms_norm,
        )
    out_pt, *rest_pt = layer_norm_ref_fn(
        x0_pt, norm_weight_pt, norm_bias_pt, residual=res_pt, eps=1e-6, prenorm=prenorm
    )
    with torch.autocast(device_type="cuda", dtype=input_dtype):
        out_pt = F.linear(out_pt, linear_weight_pt, linear_bias_pt)
    out_ref, *rest_ref = layer_norm_ref_fn(
        x0_ref,
        norm_weight_ref,
        norm_bias_ref,
        residual=res_ref,
        eps=1e-6,
        prenorm=prenorm,
        upcast=True,
    )
    out_ref = F.linear(out_ref.to(linear_weight_ref.dtype), linear_weight_ref, linear_bias_ref)
    if prenorm:
        residual = rest[0]
        residual_pt = rest_pt[0]
        residual_ref = rest_ref[0]
    assert out.dtype == input_dtype
    if prenorm:
        assert residual.dtype == residual_dtype
        assert allclose(residual, residual_pt, residual_ref)
    assert allclose(out, out_pt, out_ref)

    g = torch.randn_like(out) / batch_size
    out.backward(g)
    out_pt.backward(g)
    out_ref.backward(g)
    assert allclose(x0.grad, x0_pt.grad, x0_ref.grad)
    if has_residual:
        assert allclose(res.grad, res_pt.grad, res_ref.grad)
    assert allclose(norm_weight.grad, norm_weight_pt.grad, norm_weight_ref.grad)
    if norm_bias is not None:
        assert allclose(norm_bias.grad, norm_bias_pt.grad, norm_bias_ref.grad)
    assert allclose(linear_weight.grad, linear_weight_pt.grad, linear_weight_ref.grad)
    if linear_bias is not None:
        assert allclose(linear_bias.grad, linear_bias_pt.grad, linear_bias_ref.grad)
