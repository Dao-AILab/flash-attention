import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from flash_attn.layers.rotary import apply_rotary_emb_func, apply_rotary_emb_torch


is_sm8x = torch.cuda.get_device_capability('cuda') >= (8, 0)

@pytest.mark.parametrize('dtype', ([torch.float16] if not is_sm8x else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', ([torch.float16]))
@pytest.mark.parametrize('rotary_fraction', [1.0, 0.5])
# @pytest.mark.parametrize('rotary_fraction', [0.5])
@pytest.mark.parametrize('inplace', [False, True])
# @pytest.mark.parametrize('inplace', [False])
def test_rotary_single_tensor(inplace, rotary_fraction, dtype):
    rtol = 1e-3
    batch_size = 32
    nheads = 4
    seqlen = 217
    headdim = 128
    x = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device='cuda',
                    requires_grad=True)
    x_pt = x.detach().clone().requires_grad_()
    rotary_dim = int(rotary_fraction * headdim)
    assert rotary_dim % 2 == 0
    angle = torch.randn(seqlen, rotary_dim // 2, device='cuda')
    cos = torch.cos(angle).to(dtype=dtype)
    sin = torch.sin(angle).to(dtype=dtype)
    out = apply_rotary_emb_func(x, cos, sin, inplace)
    out_pt = apply_rotary_emb_torch(x_pt, cos, sin)
    # Numerical error if we just do any arithmetic
    atol = ((out + 0.3 - 0.3) - out).abs().max().item()
    assert torch.allclose(out, out_pt, rtol=rtol, atol=2 * atol)
    g = torch.randn_like(out)
    g_pt = g.clone()  # If inplace=True, we might modify the gradient inplace
    out.backward(g)
    out_pt.backward(g_pt)
    atol = ((x_pt.grad + 0.3 - 0.3) - x_pt.grad).abs().max().item()
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=2 * atol)
