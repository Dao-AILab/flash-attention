import torch
import torch.nn.functional as F

from flash_attn.ops.activations import bias_gelu_impl


def test_bias_gelu_backward():
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    bias = torch.randn(8, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()

    out = bias_gelu_impl(x, bias)
    out_ref = F.gelu(x_ref + bias_ref, approximate="tanh")
    assert torch.allclose(out, out_ref, atol=1e-6)

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    assert x.grad.shape == x.shape
    assert bias.grad.shape == bias.shape
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6)
    assert torch.allclose(bias.grad, bias_ref.grad, atol=1e-6)
