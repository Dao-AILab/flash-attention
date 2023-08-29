import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.ops.fused_dense import FusedDense, FusedMLP


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("return_residual", [False, True])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("out_features", [1024, 4096])
@pytest.mark.parametrize("in_features", [1024, 4096])
def test_fused_linear_bias(in_features, out_features, has_bias, return_residual, dtype):
    device = "cuda"
    rtol, atol = (3e-3, 1e-2) if dtype == torch.bfloat16 else (3e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x_pt = torch.randn(
        batch_size, seqlen, in_features, device=device, dtype=dtype, requires_grad=True
    )
    x = x_pt.detach().clone().requires_grad_()
    model_pt = torch.nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype)
    model = FusedDense(
        in_features,
        out_features,
        bias=has_bias,
        return_residual=return_residual,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        if has_bias:
            model.bias.copy_(model_pt.bias)
    out_pt = model_pt(x_pt)
    if not return_residual:
        out = model(x)
    else:
        out, x_copy = model(x)
        x_copy = (
            x_copy[..., :out_features]
            if out_features < in_features
            else F.pad(x_copy, (0, out_features - in_features))
        )
        x_pt_copy = (
            x_pt[..., :out_features]
            if out_features < in_features
            else F.pad(x_pt, (0, out_features - in_features))
        )
        # Just add some random function of the residual
        out_pt = out_pt + F.gelu(x_pt_copy)
        out = out + F.gelu(x_copy)

    # with torch.no_grad():
    #     out_fl = F.linear(x_pt.float(), model.weight.float(), model.bias.float()).half()
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)

    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(out) / 32
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=atol)
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(model.weight.grad, model_pt.weight.grad, rtol=rtol, atol=atol * 10)
    if has_bias:
        assert torch.allclose(model.bias.grad, model_pt.bias.grad, rtol=rtol, atol=atol * 5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("heuristic", ["auto", -1])
# @pytest.mark.parametrize('heuristic', ['auto'])
@pytest.mark.parametrize("checkpoint_lvl", [0, 1, 2])
# @pytest.mark.parametrize('checkpoint_lvl', [1])
@pytest.mark.parametrize("return_residual", [False, True])
# @pytest.mark.parametrize('return_residual', [False])
@pytest.mark.parametrize("has_bias2", [True, False])
@pytest.mark.parametrize("has_bias1", [True, False])
# @pytest.mark.parametrize('has_bias2', [True])
# @pytest.mark.parametrize('has_bias1', [True])
@pytest.mark.parametrize("activation", ["gelu_approx", "relu"])
# @pytest.mark.parametrize('activation', ['relu'])
@pytest.mark.parametrize("out_features", [1024, 4096])
@pytest.mark.parametrize("in_features", [1024, 4096])
# @pytest.mark.parametrize('out_features', [4096])
# @pytest.mark.parametrize('in_features', [1024])
def test_fused_mlp(
    in_features,
    out_features,
    activation,
    has_bias1,
    has_bias2,
    return_residual,
    checkpoint_lvl,
    heuristic,
    dtype,
):
    device = "cuda"
    rtol, atol = (3e-3, 3e-2) if dtype == torch.bfloat16 else (3e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x_pt = torch.randn(
        batch_size, seqlen, in_features, device=device, dtype=dtype, requires_grad=True
    )
    x = x_pt.detach().clone().requires_grad_()
    model_pt_fc1 = torch.nn.Linear(
        in_features, out_features, bias=has_bias1, device=device, dtype=dtype
    )
    model_pt_fc2 = torch.nn.Linear(
        out_features, in_features, bias=has_bias2, device=device, dtype=dtype
    )
    model = FusedMLP(
        in_features,
        out_features,
        in_features,
        activation=activation,
        bias1=has_bias1,
        bias2=has_bias2,
        return_residual=return_residual,
        checkpoint_lvl=checkpoint_lvl,
        heuristic=heuristic,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        model.fc1.weight.copy_(model_pt_fc1.weight)
        if has_bias1:
            model.fc1.bias.copy_(model_pt_fc1.bias)
        model.fc2.weight.copy_(model_pt_fc2.weight)
        if has_bias2:
            model.fc2.bias.copy_(model_pt_fc2.bias)
    activation_fn = (
        partial(F.gelu, approximate="tanh")
        if activation == "gelu_approx"
        else partial(F.relu, inplace=True)
    )
    out_pt = model_pt_fc2(activation_fn(model_pt_fc1(x_pt)))
    if not return_residual:
        out = model(x)
    else:
        out, x_copy = model(x)
        # Just add some random function of the residual
        out_pt = out_pt + F.gelu(x_pt)
        out = out + F.gelu(x_copy)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)

    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(out) / 32
    out_pt.backward(g)
    out.backward(g)
    # The error for relu is higher still
    if activation == "relu":
        atol = 1e-1 if dtype == torch.bfloat16 else 5e-2
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=atol)
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(
        model.fc1.weight.grad, model_pt_fc1.weight.grad, rtol=rtol, atol=atol * 10
    )
    if has_bias1:
        assert torch.allclose(model.fc1.bias.grad, model_pt_fc1.bias.grad, rtol=rtol, atol=atol * 5)
    assert torch.allclose(
        model.fc2.weight.grad, model_pt_fc2.weight.grad, rtol=rtol, atol=atol * 10
    )
    if has_bias2:
        assert torch.allclose(model.fc2.bias.grad, model_pt_fc2.bias.grad, rtol=rtol, atol=atol * 5)
