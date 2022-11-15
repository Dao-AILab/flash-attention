import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from flash_attn.ops.fused_dense import FusedDenseTD, FusedDenseGeluDenseTD
from flash_attn.ops.fused_dense import FusedDenseResidual, FusedDenseResGeluDense


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('out_features', [1024, 4096])
@pytest.mark.parametrize('in_features', [1024, 4096])
def test_fused_linear_bias(in_features, out_features, dtype):
    device = 'cuda'
    rtol, atol = (3e-3, 1e-2) if dtype == torch.bfloat16 else (3e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x_pt = torch.randn(batch_size, seqlen, in_features, device=device, dtype=dtype, requires_grad=True)
    x = x_pt.detach().clone().requires_grad_()
    model_pt = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)
    model = FusedDenseTD(in_features, out_features, device=device, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
    out_pt = model_pt(x_pt)
    out = model(x)
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
    assert torch.allclose(model.bias.grad, model_pt.bias.grad, rtol=rtol, atol=atol * 5)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('out_features,in_features', [(1024, 1024), (4096, 4096)])
def test_fused_linear_bias_residual(in_features, out_features, dtype):
    device = 'cuda'
    rtol, atol = (3e-3, 1e-2) if dtype == torch.bfloat16 else (3e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x_pt = torch.randn(batch_size, seqlen, in_features, device=device, dtype=dtype, requires_grad=True)
    x = x_pt.detach().clone().requires_grad_()
    model_pt = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)
    model = FusedDenseResidual(in_features, out_features, device=device, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
    out_pt = model_pt(x_pt) + F.gelu(x_pt)  # Just add some random function of the residual x_pt
    out, x_copy = model(x)
    out = out + F.gelu(x_copy)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol * 2)

    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(out) / 32
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=atol)
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(model.weight.grad, model_pt.weight.grad, rtol=rtol, atol=atol * 10)
    assert torch.allclose(model.bias.grad, model_pt.bias.grad, rtol=rtol, atol=atol * 5)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('heuristic', [1, -1])
@pytest.mark.parametrize('checkpoint_lvl', [0, 1, 2])
@pytest.mark.parametrize('out_features', [1024, 4096])
@pytest.mark.parametrize('in_features', [1024, 4096])
def test_fused_dense_gelu_dense(in_features, out_features, checkpoint_lvl, heuristic, dtype):
    device = 'cuda'
    rtol, atol = (3e-3, 1e-2) if dtype == torch.bfloat16 else (3e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x_pt = torch.randn(batch_size, seqlen, in_features, device=device, dtype=dtype, requires_grad=True)
    x = x_pt.detach().clone().requires_grad_()
    model_pt_fc1 = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)
    model_pt_fc2 = torch.nn.Linear(out_features, in_features, device=device, dtype=dtype)
    model = FusedDenseGeluDenseTD(in_features, out_features, in_features,
                                  checkpoint_lvl=checkpoint_lvl, heuristic=heuristic,
                                  device=device, dtype=dtype)
    with torch.no_grad():
        model.fc1.weight.copy_(model_pt_fc1.weight)
        model.fc1.bias.copy_(model_pt_fc1.bias)
        model.fc2.weight.copy_(model_pt_fc2.weight)
        model.fc2.bias.copy_(model_pt_fc2.bias)
    out_pt = model_pt_fc2(F.gelu(model_pt_fc1(x_pt), approximate='tanh'))
    out = model(x)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)

    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(out) / 32
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=atol)
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(model.fc1.weight.grad, model_pt_fc1.weight.grad, rtol=rtol, atol=atol * 10)
    assert torch.allclose(model.fc1.bias.grad, model_pt_fc1.bias.grad, rtol=rtol, atol=atol * 5)
    assert torch.allclose(model.fc2.weight.grad, model_pt_fc2.weight.grad, rtol=rtol, atol=atol * 10)
    assert torch.allclose(model.fc2.bias.grad, model_pt_fc2.bias.grad, rtol=rtol, atol=atol * 5)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('checkpoint_lvl', [0, 1, 2])
@pytest.mark.parametrize('out_features', [1024, 4096])
@pytest.mark.parametrize('in_features', [1024, 4096])
def test_fused_dense_residual_gelu_dense(in_features, out_features, checkpoint_lvl, dtype):
    device = 'cuda'
    rtol, atol = (3e-3, 1e-2) if dtype == torch.bfloat16 else (3e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x_pt = torch.randn(batch_size, seqlen, in_features, device=device, dtype=dtype, requires_grad=True)
    x = x_pt.detach().clone().requires_grad_()
    model_pt_fc1 = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)
    model_pt_fc2 = torch.nn.Linear(out_features, in_features, device=device, dtype=dtype)
    model = FusedDenseResGeluDense(in_features, out_features, in_features,
                                   checkpoint_lvl=checkpoint_lvl,
                                   device=device, dtype=dtype)
    with torch.no_grad():
        model.fc1.weight.copy_(model_pt_fc1.weight)
        model.fc1.bias.copy_(model_pt_fc1.bias)
        model.fc2.weight.copy_(model_pt_fc2.weight)
        model.fc2.bias.copy_(model_pt_fc2.bias)
    out_pt = model_pt_fc2(F.gelu(model_pt_fc1(x_pt), approximate='tanh')) + F.gelu(x_pt)
    out, x_copy = model(x)
    out = out + F.gelu(x_copy)
    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol * 2)

    # If we don't divide by batch_size, the gradient gets a bit too large.
    g = torch.randn_like(out) / 32
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=atol)
    # The error for d_weight and d_bias is quite a bit higher
    assert torch.allclose(model.fc1.weight.grad, model_pt_fc1.weight.grad, rtol=rtol, atol=atol * 10)
    assert torch.allclose(model.fc1.bias.grad, model_pt_fc1.bias.grad, rtol=rtol, atol=atol * 5)
    assert torch.allclose(model.fc2.weight.grad, model_pt_fc2.weight.grad, rtol=rtol, atol=atol * 10)
    assert torch.allclose(model.fc2.bias.grad, model_pt_fc2.bias.grad, rtol=rtol, atol=atol * 5)
