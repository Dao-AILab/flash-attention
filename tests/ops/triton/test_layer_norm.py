import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn.ops.triton.layernorm import layer_norm_fn, layer_norm_ref, rms_norm_ref


is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize("is_rms_norm", [True])
@pytest.mark.parametrize("has_residual", [True, False])
# @pytest.mark.parametrize("has_residual", [True])
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
    hidden_size, input_dtype, residual_dtype, weight_dtype, has_residual, is_rms_norm
):
    device = "cuda"
    if any(x == torch.bfloat16 for x in [input_dtype, residual_dtype, weight_dtype]):
        atol = 5e-2
    elif any(x == torch.float16 for x in [input_dtype, residual_dtype, weight_dtype]):
        atol = 5e-3
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

    out, *rest = layer_norm_fn(x0, weight, bias, residual=res, eps=1e-6, is_rms_norm=is_rms_norm)
    out_pt, *rest_pt = layer_norm_ref_fn(x0_pt, weight_pt, bias_pt, residual=res_pt, eps=1e-6)
    out_ref, *rest_ref = layer_norm_ref_fn(
        x0_ref, weight_ref, bias_ref, residual=res_ref, eps=1e-6, upcast=True
    )
    if has_residual:
        residual = rest[0]
        residual_pt = rest_pt[0]
        residual_ref = rest_ref[0]
        residual_ref = x0_ref + res_ref
    assert out.dtype == input_dtype
    if has_residual:
        assert residual.dtype == residual_dtype
        assert allclose(residual, residual_pt, residual_ref)
    assert allclose(out, out_pt, out_ref)

    g = torch.randn_like(out) / batch_size
    if not has_residual:
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
