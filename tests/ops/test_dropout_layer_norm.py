import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from flash_attn.ops.layer_norm import DropoutAddLayerNorm, dropout_add_layer_norm


is_sm8x = torch.cuda.get_device_capability('cuda')[0] >= 8

@pytest.mark.parametrize('has_rowscale', [True, False])
# @pytest.mark.parametrize('has_rowscale', [True])
@pytest.mark.parametrize('has_residual', [True, False])
# @pytest.mark.parametrize('has_residual', [False])
@pytest.mark.parametrize('dropout_p', [0.37, 0.0])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('weight_dtype', [torch.float32, torch.float16])
# @pytest.mark.parametrize('weight_dtype', [torch.float32])
@pytest.mark.parametrize('input_dtype,residual_dtype',
                         [(torch.float16, torch.float16), (torch.float16, torch.float32),
                          (torch.float32, torch.float32)]
                         + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []))
# @pytest.mark.parametrize('input_dtype,residual_dtype', [(torch.float16, torch.float32)])
@pytest.mark.parametrize('hidden_size', [768, 1024, 1280, 1536, 1600, 2048, 2560, 3072, 4096, 5120])
# @pytest.mark.parametrize('hidden_size', [768])
def test_dropout_layer_norm_training(hidden_size, input_dtype, residual_dtype, weight_dtype,
                                     dropout_p, has_residual, has_rowscale):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    # Backward numerical error is high, and this case isn't used
    if has_rowscale and not has_residual:
        pytest.skip()
    device = 'cuda'
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x0_pt = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=input_dtype,
                        requires_grad=True)
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_residual:
        x1_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        x1 = x1_pt.detach().clone().requires_grad_()
        x1_ref = x1_pt.detach().clone().float().requires_grad_()
    else:
        x1 = None
    if has_rowscale:
        rowscale = torch.empty(batch_size, seqlen, device=device, dtype=input_dtype)
        survival_rate = 0.87
        rowscale = rowscale.bernoulli_(survival_rate) / survival_rate
        x0_scaled_pt = x0_pt * rearrange(rowscale, '... -> ... 1')
        x0_scaled_ref = x0_ref * rearrange(rowscale, '... -> ... 1')
    else:
        rowscale = None
        x0_scaled_pt = x0_pt
        x0_scaled_ref = x0_ref
    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    torch.nn.init.normal_(model_pt.weight)
    torch.nn.init.normal_(model_pt.bias)
    model_ref = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float32)
    model = DropoutAddLayerNorm(hidden_size, p=dropout_p, device=device, dtype=weight_dtype)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
        model_ref.weight.copy_(model_pt.weight)
        model_ref.bias.copy_(model_pt.bias)
    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    out, dmask = dropout_add_layer_norm(x0, x1, model.weight, model.bias, model.p,
                                        model.epsilon, rowscale=rowscale,
                                        residual_in_fp32=residual_in_fp32, return_dropout_mask=True)
    assert out.dtype == input_dtype
    print(f'Actual dropout fraction: {1 - dmask.float().mean().item()}')
    if has_residual:
        residual_pt = ((x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p) + x1_pt.float()).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask.float()) / (1 - dropout_p) + x1_ref
    else:
        residual_pt = ((x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p)).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask.float()) / (1 - dropout_p)
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(dtype=input_dtype)
    out_ref = model_ref(residual_ref)
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4

    g = torch.randn_like(out) / batch_size
    out_pt.backward(g)
    out.backward(g)
    out_ref.backward(g)
    assert (x0.grad - x0_ref.grad).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad).abs().max() + 1e-4
    if has_residual:
        assert (x1.grad - x1_ref.grad).abs().max() <= 4 * (x1_pt.grad - x1_ref.grad).abs().max() + 1e-4
    assert (model.weight.grad - model_ref.weight.grad).abs().max() <= 2 * (model_pt.weight.grad - model_ref.weight.grad).abs().max() + 3e-5
    assert (model.bias.grad - model_ref.bias.grad).abs().max() <= 2 * (model_pt.bias.grad - model_ref.bias.grad).abs().max() + 3e-5


@pytest.mark.parametrize('weight_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('input_dtype,residual_dtype',
                         [(torch.float16, torch.float16), (torch.float16, torch.float32),
                          (torch.float32, torch.float32)]
                         + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []))
@pytest.mark.parametrize('hidden_size', [768, 1024, 1280, 1536, 1600, 2048, 2560, 3072, 4096, 5120])
def test_dropout_layer_norm_eval(hidden_size, input_dtype, residual_dtype, weight_dtype):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    device = 'cuda'
    # rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    dropout_p = 0.37
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    seqlen = 512
    x0_pt = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=input_dtype,
                        requires_grad=True)
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    x1_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
    x1 = x1_pt.detach().clone().requires_grad_()
    x1_ref = x1_pt.detach().clone().float().requires_grad_()
    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    model = DropoutAddLayerNorm(hidden_size, p=dropout_p, device=device, dtype=weight_dtype)
    model_ref = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
        model_ref.weight.copy_(model_pt.weight)
        model_ref.bias.copy_(model_pt.bias)
    model_pt.eval()
    model.eval()
    model_ref.eval()
    out = model(x0, x1)
    residual_pt = (x0_pt.float() + x1_pt.float()).to(dtype=residual_dtype)
    residual_ref = x0_ref + x1_ref
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(input_dtype)
    out_ref = model_ref(residual_ref)
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize('has_rowscale', [True, False])
@pytest.mark.parametrize('has_residual', [True, False])
@pytest.mark.parametrize('dropout_p', [0.37, 0.0])
@pytest.mark.parametrize('weight_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('input_dtype,residual_dtype',
                         [(torch.float16, torch.float16), (torch.float16, torch.float32),
                          (torch.float32, torch.float32)]
                         + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []))
@pytest.mark.parametrize('hidden_size', [768, 1024, 1280, 1536, 1600, 2048, 2560, 3072, 4096, 5120])
def test_dropout_layer_norm_prenorm_training(hidden_size, input_dtype, residual_dtype, weight_dtype,
                                             dropout_p, has_residual, has_rowscale):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    # Backward numerical error is high, and this case isn't used
    if has_rowscale and not has_residual:
        pytest.skip()
    device = 'cuda'
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 2e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x0_pt = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=input_dtype,
                        requires_grad=True)
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_residual:
        x1_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        x1 = x1_pt.detach().clone().requires_grad_()
        x1_ref = x1_pt.detach().clone().float().requires_grad_()
    else:
        x1 = None
    if has_rowscale:
        rowscale = torch.empty(batch_size, seqlen, device=device, dtype=input_dtype)
        survival_rate = 0.87
        rowscale = rowscale.bernoulli_(survival_rate) / survival_rate
        x0_scaled_pt = x0_pt * rearrange(rowscale, '... -> ... 1')
        x0_scaled_ref = x0_ref * rearrange(rowscale, '... -> ... 1')
    else:
        rowscale = None
        x0_scaled_pt = x0_pt
        x0_scaled_ref = x0_ref
    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    model_ref = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float32)
    model = DropoutAddLayerNorm(hidden_size, prenorm=True, p=dropout_p, device=device,
                                dtype=weight_dtype)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
        model_ref.weight.copy_(model_pt.weight)
        model_ref.bias.copy_(model_pt.bias)
    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    out, residual, dmask = dropout_add_layer_norm(x0, x1, model.weight, model.bias, model.p,
                                                  model.epsilon, rowscale=rowscale, prenorm=True,
                                                  residual_in_fp32=residual_in_fp32,
                                                  return_dropout_mask=True)
    print(f'Actual dropout fraction: {1 - dmask.float().mean().item()}')
    if has_residual:
        residual_pt = ((x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p) + x1_pt.float()).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask.float()) / (1 - dropout_p) + x1_ref
    else:
        residual_pt = ((x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p)).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask.float()) / (1 - dropout_p)
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(dtype=input_dtype)
    out_ref = model_ref(residual_ref)
    assert out.dtype == input_dtype
    assert residual.dtype == residual_dtype
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4
    assert (residual - residual_ref).abs().max() <= 4 * (residual_pt - residual_ref).abs().max() + 1e-4

    g = torch.randn_like(out) / batch_size
    (out_pt * F.sigmoid(residual_pt)).backward(g)
    (out * F.sigmoid(residual)).backward(g)
    (out_ref * F.sigmoid(residual_ref.to(dtype=residual_dtype))).backward(g)
    assert (x0.grad - x0_ref.grad).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad).abs().max() + 1e-4
    if has_residual:
        assert (x1.grad - x1_ref.grad).abs().max() <= 4 * (x1_pt.grad - x1_ref.grad).abs().max() + 1e-4
    assert (model.weight.grad - model_ref.weight.grad).abs().max() <= 2 * (model_pt.weight.grad - model_ref.weight.grad).abs().max() + 2e-4
    assert (model.bias.grad - model_ref.bias.grad).abs().max() <= 2 * (model_pt.bias.grad - model_ref.bias.grad).abs().max() + 2e-4


@pytest.mark.parametrize('weight_dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('input_dtype,residual_dtype',
                         [(torch.float16, torch.float16), (torch.float16, torch.float32),
                          (torch.float32, torch.float32)]
                         + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []))
@pytest.mark.parametrize('hidden_size', [768, 1024, 1280, 1536, 1600, 2048, 2560, 3072, 4096, 5120])
def test_dropout_layer_norm_prenorm_eval(hidden_size, input_dtype, residual_dtype, weight_dtype):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    device = 'cuda'
    # rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    dropout_p = 0.37
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    seqlen = 512
    x0_pt = torch.randn(batch_size, seqlen, hidden_size, device=device, dtype=input_dtype,
                        requires_grad=True)
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    x1_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
    x1 = x1_pt.detach().clone().requires_grad_()
    x1_ref = x1_pt.detach().clone().float().requires_grad_()
    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    model = DropoutAddLayerNorm(hidden_size, prenorm=True, p=dropout_p, device=device,
                                dtype=weight_dtype)
    model_ref = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
        model_ref.weight.copy_(model_pt.weight)
        model_ref.bias.copy_(model_pt.bias)
    model_pt.eval()
    model.eval()
    model_ref.eval()
    out, residual = model(x0, x1)
    residual_pt = (x0_pt.float() + x1_pt.float()).to(dtype=residual_dtype)
    residual_ref = x0_ref + x1_ref
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(input_dtype)
    out_ref = model_ref(residual_ref)
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4
    assert (residual - residual_ref).abs().max() <= 4 * (residual_pt - residual_ref).abs().max() + 1e-4
