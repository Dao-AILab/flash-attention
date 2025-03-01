# Copyright (c) 2024, Tri Dao.

import pytest
import torch
import torch.nn.functional as F
from flash_attn.losses.cross_entropy import CrossEntropyLoss

is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32] + ([torch.bfloat16] if is_sm8x else [])
)
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("precompute_lse", [False, True])
# @pytest.mark.parametrize("precompute_lse", [False])
@pytest.mark.parametrize("inplace_backward", [False, True])
# @pytest.mark.parametrize("inplace_backward", [False])
@pytest.mark.parametrize("lse_square_scale", [0.0, 1e-2])
@pytest.mark.parametrize("return_z_loss", [False, True])
# @pytest.mark.parametrize("lse_square_scale", [1e-2])
@pytest.mark.parametrize("logit_scale", [1.0, 0.7])
# @pytest.mark.parametrize("logit_scale", [1.0])
@pytest.mark.parametrize("smoothing", [0.0, 0.9])
# @pytest.mark.parametrize("smoothing", [0.0])
@pytest.mark.parametrize("vocab_size", [50257, 128256])  # test vocab larger than 64k for split
# @pytest.mark.parametrize("vocab_size", [12])
def test_cross_entropy_loss(
    vocab_size,
    smoothing,
    logit_scale,
    lse_square_scale,
    return_z_loss,
    inplace_backward,
    precompute_lse,
    dtype,
):
    if precompute_lse and (logit_scale != 1.0 or smoothing != 0.0):
        pytest.skip("precompute_lse only works with logit_scale=1.0 and smoothing=0.0")
    device = "cuda"
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1 if dtype == torch.float32 else 4  # Otherwise OOM
    seqlen = 4096 if lse_square_scale == 0.0 and logit_scale == 1.0 else 1024  # Otherwise OOM
    x_pt = torch.randn(
        batch_size * seqlen, vocab_size, device=device, dtype=dtype, requires_grad=True
    )
    x = x_pt.detach().clone().requires_grad_()
    y = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    if batch_size * seqlen > 10:
        y[torch.randperm(batch_size * seqlen)[:10]] = -100
    model_pt = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    model = CrossEntropyLoss(
        label_smoothing=smoothing,
        logit_scale=logit_scale,
        lse_square_scale=lse_square_scale,
        return_z_loss=return_z_loss,
        inplace_backward=inplace_backward,
    )
    if precompute_lse:
        with torch.no_grad():
            lse = torch.logsumexp(x.float(), dim=-1)
    else:
        lse = None
    if return_z_loss:
        out, out_z_loss = model(x, y, precomputed_lse=lse)
    else:
        out = model(x, y, precomputed_lse=lse)
    x_pt_scaled = (x_pt.float() * logit_scale) if logit_scale != 1.0 else x_pt.float()
    out_pt = model_pt(x_pt_scaled, y)
    if lse_square_scale > 0.0:
        lse_pt = torch.logsumexp(x_pt_scaled, dim=-1)
        z_loss_pt = lse_square_scale * (lse_pt[y != -100] ** 2).mean()
        if return_z_loss:
            assert torch.allclose(out_z_loss, z_loss_pt, rtol=rtol, atol=atol)
        out_pt += z_loss_pt
    assert torch.allclose(out, out_pt, rtol=1e-5, atol=1e-6)

    g = torch.randn_like(out)
    out_pt.backward(g)
    out.backward(g)
    assert torch.allclose(x.grad, x_pt.grad, rtol=rtol, atol=atol)
