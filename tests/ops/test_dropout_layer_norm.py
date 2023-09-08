import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn.ops.layer_norm import (
    DropoutAddLayerNorm,
    dropout_add_layer_norm,
    dropout_add_layer_norm_parallel_residual,
    dropout_add_layer_norm_subset,
)
from flash_attn.ops.rms_norm import (
    DropoutAddRMSNorm,
    dropout_add_rms_norm,
    dropout_add_rms_norm_parallel_residual,
    dropout_add_rms_norm_subset,
)

try:
    from apex.normalization import FusedRMSNorm
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
except:
    FusedRMSNorm, fused_rms_norm_affine = None, None


is_sm8x = torch.cuda.get_device_capability("cuda")[0] >= 8


@pytest.mark.parametrize("is_rms_norm", [False, True])
@pytest.mark.parametrize("has_colscale", [True, False])
# @pytest.mark.parametrize('has_colscale', [False])
@pytest.mark.parametrize("has_rowscale", [True, False])
# @pytest.mark.parametrize('has_rowscale', [True])
@pytest.mark.parametrize("has_residual", [True, False])
# @pytest.mark.parametrize('has_residual', [False])
@pytest.mark.parametrize("dropout_p", [0.37, 0.0])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
# @pytest.mark.parametrize('weight_dtype', [torch.float32])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize('input_dtype,residual_dtype', [(torch.float16, torch.float32)])
@pytest.mark.parametrize(
    "hidden_size",
    [192, 256, 384, 768, 1024, 1280, 1536, 1600, 2048, 2560, 3000, 3072, 4096, 5120, 6144],
)
# @pytest.mark.parametrize('hidden_size', [256])
def test_dropout_layer_norm_training(
    hidden_size,
    input_dtype,
    residual_dtype,
    weight_dtype,
    dropout_p,
    has_residual,
    has_rowscale,
    has_colscale,
    is_rms_norm,
):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    if is_rms_norm and FusedRMSNorm is None:
        pytest.skip()  # We need Apex's FusedRMSNorm to test
    layer_norm_cls = torch.nn.LayerNorm if not is_rms_norm else FusedRMSNorm
    our_layer_norm_cls = DropoutAddLayerNorm if not is_rms_norm else DropoutAddRMSNorm
    our_layer_norm_func = dropout_add_layer_norm if not is_rms_norm else dropout_add_rms_norm
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_colscale:
        colscale = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        colscale_pt = colscale.detach().clone().requires_grad_()
        colscale_ref = colscale.detach().clone().float().requires_grad_()
    else:
        colscale = None
    if has_residual:
        res_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        res = res_pt.detach().clone().requires_grad_()
        res_ref = res_pt.detach().clone().float().requires_grad_()
    else:
        res = None
    if has_rowscale:
        rowscale = torch.empty(batch_size, seqlen, device=device, dtype=input_dtype)
        survival_rate = 0.87
        rowscale = rowscale.bernoulli_(survival_rate) / survival_rate
        x0_scaled_pt = x0_pt * rearrange(rowscale, "... -> ... 1")
        x0_scaled_ref = x0_ref * rearrange(rowscale, "... -> ... 1")
    else:
        rowscale = None
        x0_scaled_pt = x0_pt
        x0_scaled_ref = x0_ref
    if has_colscale:
        x0_scaled_pt = x0_scaled_pt * colscale_pt
        x0_scaled_ref = x0_scaled_ref * colscale_ref
    model_pt = layer_norm_cls(hidden_size).to(device=device, dtype=weight_dtype)
    torch.nn.init.normal_(model_pt.weight)
    if not is_rms_norm:
        torch.nn.init.normal_(model_pt.bias)
    model_ref = layer_norm_cls(hidden_size).to(device=device, dtype=torch.float32)
    model = our_layer_norm_cls(hidden_size, p=dropout_p, device=device, dtype=weight_dtype)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model_ref.weight.copy_(model_pt.weight)
        if not is_rms_norm:
            model.bias.copy_(model_pt.bias)
            model_ref.bias.copy_(model_pt.bias)
    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    out, dmask = our_layer_norm_func(
        x0,
        res,
        model.weight,
        model.bias,
        model.p,
        model.eps,
        rowscale=rowscale,
        layerscale=colscale,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=True,
    )
    assert out.dtype == input_dtype
    print(f"Actual dropout fraction: {1 - dmask.float().mean().item()}")
    if has_residual:
        residual_pt = (
            (x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p) + res_pt.float()
        ).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask.float()) / (1 - dropout_p) + res_ref
    else:
        residual_pt = ((x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p)).to(
            dtype=residual_dtype
        )
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
        assert (res.grad - res_ref.grad).abs().max() <= 4 * (
            res_pt.grad - res_ref.grad
        ).abs().max() + 1e-4
    assert (model.weight.grad - model_ref.weight.grad).abs().max() <= 3 * (
        model_pt.weight.grad - model_ref.weight.grad
    ).abs().max() + 3e-5
    if not is_rms_norm:
        assert (model.bias.grad - model_ref.bias.grad).abs().max() <= 2 * (
            model_pt.bias.grad - model_ref.bias.grad
        ).abs().max() + 3e-5
    if has_colscale:
        assert (colscale.grad - colscale_ref.grad).abs().max() <= 2 * (
            colscale_pt.grad - colscale_ref.grad
        ).abs().max() + 2e-4


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
@pytest.mark.parametrize("hidden_size", [768, 1024, 1280, 1536, 1600, 2048, 2560, 3072, 4096, 5120])
def test_dropout_layer_norm_eval(hidden_size, input_dtype, residual_dtype, weight_dtype):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    dropout_p = 0.37
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    seqlen = 512
    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    res_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
    res = res_pt.detach().clone().requires_grad_()
    res_ref = res_pt.detach().clone().float().requires_grad_()
    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    torch.nn.init.normal_(model_pt.weight)
    torch.nn.init.normal_(model_pt.bias)
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
    out = model(x0, res)
    residual_pt = (x0_pt.float() + res_pt.float()).to(dtype=residual_dtype)
    residual_ref = x0_ref + res_ref
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(input_dtype)
    out_ref = model_ref(residual_ref)
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("is_rms_norm", [False, True])
@pytest.mark.parametrize("has_colscale", [True, False])
@pytest.mark.parametrize("has_rowscale", [True, False])
@pytest.mark.parametrize("has_residual", [True, False])
@pytest.mark.parametrize("dropout_p", [0.37, 0.0])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize('has_colscale', [True])
# @pytest.mark.parametrize('has_rowscale', [False])
# @pytest.mark.parametrize('has_residual', [True])
# @pytest.mark.parametrize('dropout_p', [0.0])
# @pytest.mark.parametrize('weight_dtype', [torch.float32])
# @pytest.mark.parametrize('input_dtype,residual_dtype', [(torch.float32, torch.float32)])
@pytest.mark.parametrize(
    "hidden_size",
    [192, 256, 384, 768, 1024, 1280, 1536, 1600, 2048, 2560, 3000, 3072, 4096, 5120, 6144],
)
# @pytest.mark.parametrize('hidden_size', [256])
def test_dropout_layer_norm_prenorm_training(
    hidden_size,
    input_dtype,
    residual_dtype,
    weight_dtype,
    dropout_p,
    has_residual,
    has_rowscale,
    has_colscale,
    is_rms_norm,
):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    if is_rms_norm and FusedRMSNorm is None:
        pytest.skip()  # We need Apex's FusedRMSNorm to test
    layer_norm_cls = torch.nn.LayerNorm if not is_rms_norm else FusedRMSNorm
    our_layer_norm_cls = DropoutAddLayerNorm if not is_rms_norm else DropoutAddRMSNorm
    our_layer_norm_func = dropout_add_layer_norm if not is_rms_norm else dropout_add_rms_norm
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 2e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_colscale:
        colscale = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        colscale_pt = colscale.detach().clone().requires_grad_()
        colscale_ref = colscale.detach().clone().float().requires_grad_()
    else:
        colscale = None
    if has_residual:
        res_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        res = res_pt.detach().clone().requires_grad_()
        res_ref = res_pt.detach().clone().float().requires_grad_()
    else:
        res = None
    if has_rowscale:
        rowscale = torch.empty(batch_size, seqlen, device=device, dtype=input_dtype)
        survival_rate = 0.87
        rowscale = rowscale.bernoulli_(survival_rate) / survival_rate
        x0_scaled_pt = x0_pt * rearrange(rowscale, "... -> ... 1")
        x0_scaled_ref = x0_ref * rearrange(rowscale, "... -> ... 1")
    else:
        rowscale = None
        x0_scaled_pt = x0_pt
        x0_scaled_ref = x0_ref
    if has_colscale:
        x0_scaled_pt = x0_scaled_pt * colscale_pt
        x0_scaled_ref = x0_scaled_ref * colscale_ref
    model_pt = layer_norm_cls(hidden_size).to(device=device, dtype=weight_dtype)
    torch.nn.init.normal_(model_pt.weight)
    if not is_rms_norm:
        torch.nn.init.normal_(model_pt.bias)
    model_ref = layer_norm_cls(hidden_size).to(device=device, dtype=torch.float32)
    model = our_layer_norm_cls(
        hidden_size, prenorm=True, p=dropout_p, device=device, dtype=weight_dtype
    )
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model_ref.weight.copy_(model_pt.weight)
        if not is_rms_norm:
            model.bias.copy_(model_pt.bias)
            model_ref.bias.copy_(model_pt.bias)
    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    out, residual, dmask = our_layer_norm_func(
        x0,
        res,
        model.weight,
        model.bias,
        model.p,
        model.eps,
        rowscale=rowscale,
        layerscale=colscale,
        prenorm=True,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=True,
    )
    print(f"Actual dropout fraction: {1 - dmask.float().mean().item()}")
    if has_residual:
        residual_pt = (
            (x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p) + res_pt.float()
        ).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask.float()) / (1 - dropout_p) + res_ref
    else:
        residual_pt = ((x0_scaled_pt.float() * dmask.float()) / (1 - dropout_p)).to(
            dtype=residual_dtype
        )
        residual_ref = (x0_scaled_ref * dmask.float()) / (1 - dropout_p)
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(dtype=input_dtype)
    out_ref = model_ref(residual_ref)
    assert out.dtype == input_dtype
    assert residual.dtype == residual_dtype
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4
    assert (residual - residual_ref).abs().max() <= 4 * (
        residual_pt - residual_ref
    ).abs().max() + 1e-4

    g = torch.randn_like(out) / batch_size
    (out_pt * F.sigmoid(residual_pt)).backward(g)
    (out * F.sigmoid(residual)).backward(g)
    (out_ref * F.sigmoid(residual_ref.to(dtype=residual_dtype))).backward(g)
    assert (x0.grad - x0_ref.grad).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad).abs().max() + 1e-4
    if has_residual:
        assert (res.grad - res_ref.grad).abs().max() <= 4 * (
            res_pt.grad - res_ref.grad
        ).abs().max() + 1e-4
    assert (model.weight.grad - model_ref.weight.grad).abs().max() <= 2 * (
        model_pt.weight.grad - model_ref.weight.grad
    ).abs().max() + 2e-4
    if not is_rms_norm:
        assert (model.bias.grad - model_ref.bias.grad).abs().max() <= 2 * (
            model_pt.bias.grad - model_ref.bias.grad
        ).abs().max() + 2e-4
    if has_colscale:
        assert (colscale.grad - colscale_ref.grad).abs().max() <= 2 * (
            colscale_pt.grad - colscale_ref.grad
        ).abs().max() + 2e-4


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
@pytest.mark.parametrize("hidden_size", [768, 1024, 1280, 1536, 1600, 2048, 2560, 3072, 4096, 5120])
def test_dropout_layer_norm_prenorm_eval(hidden_size, input_dtype, residual_dtype, weight_dtype):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    dropout_p = 0.37
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    seqlen = 512
    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    res_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
    res = res_pt.detach().clone().requires_grad_()
    res_ref = res_pt.detach().clone().float().requires_grad_()
    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    torch.nn.init.normal_(model_pt.weight)
    torch.nn.init.normal_(model_pt.bias)
    model = DropoutAddLayerNorm(
        hidden_size, prenorm=True, p=dropout_p, device=device, dtype=weight_dtype
    )
    model_ref = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
        model_ref.weight.copy_(model_pt.weight)
        model_ref.bias.copy_(model_pt.bias)
    model_pt.eval()
    model.eval()
    model_ref.eval()
    out, residual = model(x0, res)
    residual_pt = (x0_pt.float() + res_pt.float()).to(dtype=residual_dtype)
    residual_ref = x0_ref + res_ref
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(input_dtype)
    out_ref = model_ref(residual_ref)
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4
    assert (residual - residual_ref).abs().max() <= 4 * (
        residual_pt - residual_ref
    ).abs().max() + 1e-4


@pytest.mark.parametrize("has_colscale", [True, False])
@pytest.mark.parametrize("has_residual", [True, False])
@pytest.mark.parametrize("dropout_p", [0.37, 0.0])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize('has_colscale', [True])
# @pytest.mark.parametrize('has_residual', [True])
# @pytest.mark.parametrize('dropout_p', [0.0])
# @pytest.mark.parametrize('weight_dtype', [torch.float32])
# @pytest.mark.parametrize('input_dtype,residual_dtype', [(torch.float32, torch.float32)])
@pytest.mark.parametrize(
    "hidden_size",
    [192, 256, 384, 768, 1024, 1280, 1536, 1600, 2048, 2560, 3000, 3072, 4096, 5120, 6144],
)
# @pytest.mark.parametrize('hidden_size', [256])
def test_dropout_layer_norm_subset_training(
    hidden_size, input_dtype, residual_dtype, weight_dtype, dropout_p, has_residual, has_colscale
):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 2e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    drop_path_rate = 0.4
    drop_path_scale = 1 / (1 - drop_path_rate)

    def generate_droppath_masks(batch_size, seqlen, drop_path_rate, device):
        # Do it on CPU so we can get the numrows (with .item()) without GPU-CPU sync
        mask_batch = torch.rand(batch_size) < 1 - drop_path_rate
        numrows = (mask_batch).sum().item() * seqlen
        mask_batch = mask_batch.to(device=device, non_blocking=True)
        mask_batch_seqlen = repeat(mask_batch, "b -> (b s)", s=seqlen)
        subset = torch.cumsum(mask_batch_seqlen, dim=0, dtype=torch.int32).masked_fill_(
            ~mask_batch_seqlen, 0
        )
        return mask_batch, numrows, rearrange(subset, "(b s) -> b s", b=batch_size)

    x0_mask_batch, x0_numrows, x0_subset = generate_droppath_masks(
        batch_size, seqlen, drop_path_rate, device
    )
    out_mask_batch, out_numrows, out_subset = generate_droppath_masks(
        batch_size, seqlen, drop_path_rate, device
    )

    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone()[x0_mask_batch].requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_colscale:
        colscale = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        colscale_pt = colscale.detach().clone().requires_grad_()
        colscale_ref = colscale.detach().clone().float().requires_grad_()
    else:
        colscale = None
    if has_residual:
        res_pt = torch.randn_like(x0_pt, dtype=residual_dtype, requires_grad=True)
        res = res_pt.detach().clone().requires_grad_()
        res_ref = res_pt.detach().clone().float().requires_grad_()
    else:
        res = None

    if has_colscale:
        x0_scaled_pt = x0_pt * colscale_pt
        x0_scaled_ref = x0_ref * colscale_ref
    else:
        x0_scaled_pt = x0_pt
        x0_scaled_ref = x0_ref

    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    torch.nn.init.normal_(model_pt.weight)
    torch.nn.init.normal_(model_pt.bias)
    model_ref = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float32)
    model = DropoutAddLayerNorm(
        hidden_size, prenorm=False, p=dropout_p, device=device, dtype=weight_dtype
    )
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
        model_ref.weight.copy_(model_pt.weight)
        model_ref.bias.copy_(model_pt.bias)

    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    out, dmask = dropout_add_layer_norm_subset(
        x0,
        res,
        model.weight,
        model.bias,
        model.p,
        model.eps,
        layerscale=colscale,
        x0_subset=x0_subset,
        out_subset=out_subset,
        rowscale_const=drop_path_scale,
        out_numrows=out_numrows,
        prenorm=False,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=True,
    )
    print(f"Actual dropout fraction: {1 - dmask.float().mean().item()}")

    x0_scaled_pt = (
        x0_scaled_pt.masked_fill(repeat(~x0_mask_batch, "b -> b s d", s=seqlen, d=hidden_size), 0)
        * drop_path_scale
    )
    x0_scaled_ref = (
        x0_scaled_ref.masked_fill(repeat(~x0_mask_batch, "b -> b s d", s=seqlen, d=hidden_size), 0)
        * drop_path_scale
    )
    dmask_expanded = torch.zeros_like(x0_pt, dtype=torch.uint8)
    dmask_expanded[x0_mask_batch] = dmask
    if has_residual:
        residual_pt = (
            (x0_scaled_pt.float() * dmask_expanded.float()) / (1 - dropout_p) + res_pt.float()
        ).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask_expanded.float()) / (1 - dropout_p) + res_ref
    else:
        residual_pt = ((x0_scaled_pt.float() * dmask_expanded.float()) / (1 - dropout_p)).to(
            dtype=residual_dtype
        )
        residual_ref = (x0_scaled_ref * dmask_expanded.float()) / (1 - dropout_p)
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(dtype=input_dtype)[out_mask_batch]
    out_ref = model_ref(residual_ref)[out_mask_batch]
    assert out.dtype == input_dtype
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4

    g = torch.randn_like(out) / batch_size
    out_pt.backward(g)
    out.backward(g)
    out_ref.backward(g)
    assert (x0.grad - x0_ref.grad[x0_mask_batch]).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad)[
        x0_mask_batch
    ].abs().max() + 1e-4
    if has_residual:
        assert (res.grad - res_ref.grad).abs().max() <= 4 * (
            res_pt.grad - res_ref.grad
        ).abs().max() + 1e-4
    assert (model.weight.grad - model_ref.weight.grad).abs().max() <= 2 * (
        model_pt.weight.grad - model_ref.weight.grad
    ).abs().max() + 2e-4
    assert (model.bias.grad - model_ref.bias.grad).abs().max() <= 2 * (
        model_pt.bias.grad - model_ref.bias.grad
    ).abs().max() + 2e-4
    if has_colscale:
        assert (colscale.grad - colscale_ref.grad).abs().max() <= 2 * (
            colscale_pt.grad - colscale_ref.grad
        ).abs().max() + 2e-4


@pytest.mark.parametrize("has_colscale", [True, False])
@pytest.mark.parametrize("has_residual", [True, False])
@pytest.mark.parametrize("dropout_p", [0.37, 0.0])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize('has_colscale', [True])
# @pytest.mark.parametrize('has_residual', [True])
# @pytest.mark.parametrize('dropout_p', [0.0])
# @pytest.mark.parametrize('weight_dtype', [torch.float32])
# @pytest.mark.parametrize('input_dtype,residual_dtype', [(torch.float32, torch.float32)])
@pytest.mark.parametrize(
    "hidden_size",
    [192, 256, 384, 768, 1024, 1280, 1536, 1600, 2048, 2560, 3000, 3072, 4096, 5120, 6144],
)
# @pytest.mark.parametrize('hidden_size', [256])
def test_dropout_layer_norm_subset_prenorm_training(
    hidden_size, input_dtype, residual_dtype, weight_dtype, dropout_p, has_residual, has_colscale
):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 2e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    drop_path_rate = 0.4
    drop_path_scale = 1 / (1 - drop_path_rate)

    def generate_droppath_masks(batch_size, seqlen, drop_path_rate, device):
        # Do it on CPU so we can get the numrows (with .item()) without GPU-CPU sync
        mask_batch = torch.rand(batch_size) < 1 - drop_path_rate
        numrows = (mask_batch).sum().item() * seqlen
        mask_batch = mask_batch.to(device=device, non_blocking=True)
        mask_batch_seqlen = repeat(mask_batch, "b -> (b s)", s=seqlen)
        subset = torch.cumsum(mask_batch_seqlen, dim=0, dtype=torch.int32).masked_fill_(
            ~mask_batch_seqlen, 0
        )
        return mask_batch, numrows, rearrange(subset, "(b s) -> b s", b=batch_size)

    x0_mask_batch, x0_numrows, x0_subset = generate_droppath_masks(
        batch_size, seqlen, drop_path_rate, device
    )
    out_mask_batch, out_numrows, out_subset = generate_droppath_masks(
        batch_size, seqlen, drop_path_rate, device
    )

    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone()[x0_mask_batch].requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_colscale:
        colscale = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        colscale_pt = colscale.detach().clone().requires_grad_()
        colscale_ref = colscale.detach().clone().float().requires_grad_()
    else:
        colscale = None
    if has_residual:
        res_pt = torch.randn_like(x0_pt, dtype=residual_dtype, requires_grad=True)
        res = res_pt.detach().clone().requires_grad_()
        res_ref = res_pt.detach().clone().float().requires_grad_()
    else:
        res = None

    if has_colscale:
        x0_scaled_pt = x0_pt * colscale_pt
        x0_scaled_ref = x0_ref * colscale_ref
    else:
        x0_scaled_pt = x0_pt
        x0_scaled_ref = x0_ref

    model_pt = torch.nn.LayerNorm(hidden_size, device=device, dtype=weight_dtype)
    torch.nn.init.normal_(model_pt.weight)
    torch.nn.init.normal_(model_pt.bias)
    model_ref = torch.nn.LayerNorm(hidden_size, device=device, dtype=torch.float32)
    model = DropoutAddLayerNorm(
        hidden_size, prenorm=True, p=dropout_p, device=device, dtype=weight_dtype
    )
    with torch.no_grad():
        model.weight.copy_(model_pt.weight)
        model.bias.copy_(model_pt.bias)
        model_ref.weight.copy_(model_pt.weight)
        model_ref.bias.copy_(model_pt.bias)

    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32
    out, residual, dmask = dropout_add_layer_norm_subset(
        x0,
        res,
        model.weight,
        model.bias,
        model.p,
        model.eps,
        layerscale=colscale,
        x0_subset=x0_subset,
        out_subset=out_subset,
        rowscale_const=drop_path_scale,
        out_numrows=out_numrows,
        prenorm=True,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=True,
    )
    print(f"Actual dropout fraction: {1 - dmask.float().mean().item()}")

    x0_scaled_pt = (
        x0_scaled_pt.masked_fill(repeat(~x0_mask_batch, "b -> b s d", s=seqlen, d=hidden_size), 0)
        * drop_path_scale
    )
    x0_scaled_ref = (
        x0_scaled_ref.masked_fill(repeat(~x0_mask_batch, "b -> b s d", s=seqlen, d=hidden_size), 0)
        * drop_path_scale
    )
    dmask_expanded = torch.zeros_like(x0_pt, dtype=torch.uint8)
    dmask_expanded[x0_mask_batch] = dmask
    if has_residual:
        residual_pt = (
            (x0_scaled_pt.float() * dmask_expanded.float()) / (1 - dropout_p) + res_pt.float()
        ).to(dtype=residual_dtype)
        residual_ref = (x0_scaled_ref * dmask_expanded.float()) / (1 - dropout_p) + res_ref
    else:
        residual_pt = ((x0_scaled_pt.float() * dmask_expanded.float()) / (1 - dropout_p)).to(
            dtype=residual_dtype
        )
        residual_ref = (x0_scaled_ref * dmask_expanded.float()) / (1 - dropout_p)
    out_pt = model_pt(residual_pt.to(dtype=weight_dtype)).to(dtype=input_dtype)[out_mask_batch]
    out_ref = model_ref(residual_ref)[out_mask_batch]
    assert out.dtype == input_dtype
    assert residual.dtype == residual_dtype
    assert (out - out_ref).abs().max() <= 4 * (out_pt - out_ref).abs().max() + 1e-4
    assert (residual - residual_ref).abs().max() <= 4 * (
        residual_pt - residual_ref
    ).abs().max() + 1e-4

    g = torch.randn_like(out) / batch_size
    (out_pt * F.sigmoid(residual_pt[out_mask_batch]) + residual_pt.mean(0, keepdim=True)).backward(
        g
    )
    (out * F.sigmoid(residual[out_mask_batch]) + residual.mean(0, keepdim=True)).backward(g)
    (
        out_ref * F.sigmoid(residual_ref[out_mask_batch].to(dtype=residual_dtype))
        + residual_ref.mean(0, keepdim=True)
    ).backward(g)
    assert (x0.grad - x0_ref.grad[x0_mask_batch]).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad)[
        x0_mask_batch
    ].abs().max() + 1e-4
    if has_residual:
        assert (res.grad - res_ref.grad).abs().max() <= 4 * (
            res_pt.grad - res_ref.grad
        ).abs().max() + 1e-4
    assert (model.weight.grad - model_ref.weight.grad).abs().max() <= 2 * (
        model_pt.weight.grad - model_ref.weight.grad
    ).abs().max() + 2e-4
    assert (model.bias.grad - model_ref.bias.grad).abs().max() <= 2 * (
        model_pt.bias.grad - model_ref.bias.grad
    ).abs().max() + 2e-4
    if has_colscale:
        assert (colscale.grad - colscale_ref.grad).abs().max() <= 2 * (
            colscale_pt.grad - colscale_ref.grad
        ).abs().max() + 2e-4


@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize('is_rms_norm', [False])
@pytest.mark.parametrize("tied_norm", [False, True])
# @pytest.mark.parametrize('tied_norm', [False])
@pytest.mark.parametrize("has_residual", [True, False])
# @pytest.mark.parametrize('has_residual', [False])
@pytest.mark.parametrize("has_x1", [True, False])
# @pytest.mark.parametrize('has_x1', [True])
@pytest.mark.parametrize("dropout_p", [0.37, 0.0])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
# @pytest.mark.parametrize('weight_dtype', [torch.float16])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize('input_dtype,residual_dtype', [(torch.float16, torch.float32)])
@pytest.mark.parametrize(
    "hidden_size",
    [192, 256, 384, 768, 1024, 1280, 1536, 1600, 2048, 2560, 3000, 3072, 4096, 5120, 6144],
)
# @pytest.mark.parametrize('hidden_size', [256])
def test_dropout_layer_norm_parallel_residual_training(
    hidden_size,
    input_dtype,
    residual_dtype,
    weight_dtype,
    dropout_p,
    has_x1,
    has_residual,
    tied_norm,
    is_rms_norm,
):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    if is_rms_norm and fused_rms_norm_affine is None:
        pytest.skip()  # We need Apex's FusedRMSNorm to test
    our_layer_norm_func = (
        dropout_add_layer_norm_parallel_residual
        if not is_rms_norm
        else dropout_add_rms_norm_parallel_residual
    )
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_x1:
        x1_pt = torch.randn(
            batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
        )
        x1 = x1_pt.detach().clone().requires_grad_()
        x1_ref = x1_pt.detach().clone().float().requires_grad_()
    else:
        x1 = None
    if has_residual:
        res_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        res = res_pt.detach().clone().requires_grad_()
        res_ref = res_pt.detach().clone().float().requires_grad_()
    else:
        res = None
    weight0 = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
    bias0 = (
        torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        if not is_rms_norm
        else None
    )
    weight0_pt = weight0.detach().clone().requires_grad_()
    weight0_ref = weight0.detach().clone().float().requires_grad_()
    bias0_pt = bias0.detach().clone().requires_grad_() if bias0 is not None else None
    bias0_ref = bias0.detach().clone().float().requires_grad_() if bias0 is not None else None
    if not tied_norm:
        weight1 = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        bias1 = (
            torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
            if not is_rms_norm
            else None
        )
        weight1_pt = weight1.detach().clone().requires_grad_()
        weight1_ref = weight1.detach().clone().float().requires_grad_()
        bias1_pt = bias1.detach().clone().requires_grad_() if bias1 is not None else None
        bias1_ref = bias1.detach().clone().float().requires_grad_() if bias1 is not None else None
    else:
        weight1, bias1 = None, None
    epsilon = 1e-5
    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32

    out0, out1, dmask0, dmask1 = our_layer_norm_func(
        x0,
        x1,
        res,
        weight0,
        bias0,
        weight1,
        bias1,
        dropout_p,
        epsilon,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=True,
    )
    assert out0.dtype == input_dtype
    if not tied_norm:
        assert out1.dtype == input_dtype
    print(f"Actual dropout fraction: {1 - dmask0.float().mean().item()}")
    if has_residual:
        if has_x1:
            residual_pt = (
                (x0_pt.float() * dmask0.float()) / (1 - dropout_p)
                + (x1_pt.float() * dmask1.float()) / (1 - dropout_p)
                + res_pt.float()
            ).to(dtype=residual_dtype)
            residual_ref = (
                (x0_ref * dmask0.float()) / (1 - dropout_p)
                + (x1_ref * dmask1.float()) / (1 - dropout_p)
            ) + res_ref
        else:
            residual_pt = ((x0_pt.float() * dmask0.float()) / (1 - dropout_p) + res_pt.float()).to(
                dtype=residual_dtype
            )
            residual_ref = (x0_ref * dmask0.float()) / (1 - dropout_p) + res_ref
    else:
        if has_x1:
            residual_pt = (
                (x0_pt.float() * dmask0.float()) / (1 - dropout_p)
                + (x1_pt.float() * dmask1.float()) / (1 - dropout_p)
            ).to(dtype=residual_dtype)
            residual_ref = (x0_ref * dmask0.float()) / (1 - dropout_p) + (
                x1_ref * dmask1.float()
            ) / (1 - dropout_p)
        else:
            residual_pt = ((x0_pt.float() * dmask0.float()) / (1 - dropout_p)).to(
                dtype=residual_dtype
            )
            residual_ref = (x0_ref * dmask0.float()) / (1 - dropout_p)
    if not is_rms_norm:
        out0_pt = F.layer_norm(
            residual_pt.to(dtype=weight_dtype), (hidden_size,), weight0_pt, bias0_pt, eps=epsilon
        ).to(dtype=input_dtype)
        out0_ref = F.layer_norm(residual_ref, (hidden_size,), weight0_ref, bias0_ref, eps=epsilon)
        if not tied_norm:
            out1_pt = F.layer_norm(
                residual_pt.to(dtype=weight_dtype),
                (hidden_size,),
                weight1_pt,
                bias1_pt,
                eps=epsilon,
            ).to(dtype=input_dtype)
            out1_ref = F.layer_norm(
                residual_ref, (hidden_size,), weight1_ref, bias1_ref, eps=epsilon
            )
    else:
        out0_pt = fused_rms_norm_affine(
            residual_pt.to(dtype=weight_dtype), weight0_pt, (hidden_size,), eps=epsilon
        ).to(dtype=input_dtype)
        out0_ref = fused_rms_norm_affine(residual_ref, weight0_ref, (hidden_size,), eps=epsilon)
        if not tied_norm:
            out1_pt = fused_rms_norm_affine(
                residual_pt.to(dtype=weight_dtype), weight1_pt, (hidden_size,), eps=epsilon
            ).to(dtype=input_dtype)
            out1_ref = fused_rms_norm_affine(residual_ref, weight1_ref, (hidden_size,), eps=epsilon)

    assert (out0 - out0_ref).abs().max() <= 4 * (out0_pt - out0_ref).abs().max() + 1e-4
    if not tied_norm:
        assert (out1 - out1_ref).abs().max() <= 4 * (out1_pt - out1_ref).abs().max() + 1e-4

    g0 = torch.randn_like(out0) / batch_size
    if tied_norm:
        out0.backward(g0)
        out0_pt.backward(g0)
        out0_ref.backward(g0)
    else:
        g1 = torch.randn_like(out1) / batch_size
        (out0 * g0 + out1 * g1).sum().backward()
        (out0_pt * g0 + out1_pt * g1).sum().backward()
        (out0_ref * g0 + out1_ref * g1).sum().backward()
    assert (x0.grad - x0_ref.grad).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad).abs().max() + 1e-4
    if has_x1:
        assert (x1.grad - x1_ref.grad).abs().max() <= 4 * (
            x1_pt.grad - x1_ref.grad
        ).abs().max() + 1e-4
    if has_residual:
        assert (res.grad - res_ref.grad).abs().max() <= 4 * (
            res_pt.grad - res_ref.grad
        ).abs().max() + 1e-4
    assert (weight0.grad - weight0_ref.grad).abs().max() <= 3 * (
        weight0_pt.grad - weight0_ref.grad
    ).abs().max() + 3e-5
    if not is_rms_norm:
        assert (bias0.grad - bias0_ref.grad).abs().max() <= 2 * (
            bias0_pt.grad - bias0_ref.grad
        ).abs().max() + 3e-5
    if not tied_norm:
        assert (weight1.grad - weight1_ref.grad).abs().max() <= 3 * (
            weight1_pt.grad - weight1_ref.grad
        ).abs().max() + 3e-5
        if not is_rms_norm:
            assert (bias1.grad - bias1_ref.grad).abs().max() <= 2 * (
                bias1_pt.grad - bias1_ref.grad
            ).abs().max() + 3e-5


@pytest.mark.parametrize("is_rms_norm", [False, True])
# @pytest.mark.parametrize('is_rms_norm', [False])
@pytest.mark.parametrize("tied_norm", [False, True])
# @pytest.mark.parametrize('tied_norm', [False])
@pytest.mark.parametrize("has_residual", [True, False])
# @pytest.mark.parametrize('has_residual', [False])
@pytest.mark.parametrize("has_x1", [True, False])
# @pytest.mark.parametrize('has_x1', [True])
@pytest.mark.parametrize("dropout_p", [0.37, 0.0])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
# @pytest.mark.parametrize('weight_dtype', [torch.float16])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)]
    + ([(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)] if is_sm8x else []),
)
# @pytest.mark.parametrize('input_dtype,residual_dtype', [(torch.float16, torch.float32)])
@pytest.mark.parametrize(
    "hidden_size",
    [192, 256, 384, 768, 1024, 1280, 1536, 1600, 2048, 2560, 3000, 3072, 4096, 5120, 6144],
)
# @pytest.mark.parametrize('hidden_size', [256])
def test_dropout_layer_norm_parallel_residual_prenorm_training(
    hidden_size,
    input_dtype,
    residual_dtype,
    weight_dtype,
    dropout_p,
    has_x1,
    has_residual,
    tied_norm,
    is_rms_norm,
):
    if weight_dtype == torch.float16 and input_dtype == torch.bfloat16:
        pytest.skip()  # Not supported
    if is_rms_norm and fused_rms_norm_affine is None:
        pytest.skip()  # We need Apex's FusedRMSNorm to test
    our_layer_norm_func = (
        dropout_add_layer_norm_parallel_residual
        if not is_rms_norm
        else dropout_add_rms_norm_parallel_residual
    )
    device = "cuda"
    # rtol, atol = (1e-5, 1e-6) if input_dtype == torch.float32 else (1e-3, 1e-4)
    rtol, atol = (1e-3, 1e-4)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x0_pt = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
    )
    x0 = x0_pt.detach().clone().requires_grad_()
    x0_ref = x0_pt.detach().clone().float().requires_grad_()
    if has_x1:
        x1_pt = torch.randn(
            batch_size, seqlen, hidden_size, device=device, dtype=input_dtype, requires_grad=True
        )
        x1 = x1_pt.detach().clone().requires_grad_()
        x1_ref = x1_pt.detach().clone().float().requires_grad_()
    else:
        x1 = None
    if has_residual:
        res_pt = torch.randn_like(x0, dtype=residual_dtype, requires_grad=True)
        res = res_pt.detach().clone().requires_grad_()
        res_ref = res_pt.detach().clone().float().requires_grad_()
    else:
        res = None
    weight0 = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
    bias0 = (
        torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        if not is_rms_norm
        else None
    )
    weight0_pt = weight0.detach().clone().requires_grad_()
    weight0_ref = weight0.detach().clone().float().requires_grad_()
    bias0_pt = bias0.detach().clone().requires_grad_() if bias0 is not None else None
    bias0_ref = bias0.detach().clone().float().requires_grad_() if bias0 is not None else None
    if not tied_norm:
        weight1 = torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
        bias1 = (
            torch.randn(hidden_size, device=device, dtype=weight_dtype, requires_grad=True)
            if not is_rms_norm
            else None
        )
        weight1_pt = weight1.detach().clone().requires_grad_()
        weight1_ref = weight1.detach().clone().float().requires_grad_()
        bias1_pt = bias1.detach().clone().requires_grad_() if bias1 is not None else None
        bias1_ref = bias1.detach().clone().float().requires_grad_() if bias1 is not None else None
    else:
        weight1, bias1 = None, None
    epsilon = 1e-5
    residual_in_fp32 = (not has_residual) and residual_dtype == torch.float32

    out0, out1, residual, dmask0, dmask1 = our_layer_norm_func(
        x0,
        x1,
        res,
        weight0,
        bias0,
        weight1,
        bias1,
        dropout_p,
        epsilon,
        prenorm=True,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=True,
    )
    assert out0.dtype == input_dtype
    if not tied_norm:
        assert out1.dtype == input_dtype
    print(f"Actual dropout fraction: {1 - dmask0.float().mean().item()}")
    if has_residual:
        if has_x1:
            residual_pt = (
                (x0_pt.float() * dmask0.float()) / (1 - dropout_p)
                + (x1_pt.float() * dmask1.float()) / (1 - dropout_p)
                + res_pt.float()
            ).to(dtype=residual_dtype)
            residual_ref = (
                (x0_ref * dmask0.float()) / (1 - dropout_p)
                + (x1_ref * dmask1.float()) / (1 - dropout_p)
            ) + res_ref
        else:
            residual_pt = ((x0_pt.float() * dmask0.float()) / (1 - dropout_p) + res_pt.float()).to(
                dtype=residual_dtype
            )
            residual_ref = (x0_ref * dmask0.float()) / (1 - dropout_p) + res_ref
    else:
        if has_x1:
            residual_pt = (
                (x0_pt.float() * dmask0.float()) / (1 - dropout_p)
                + (x1_pt.float() * dmask1.float()) / (1 - dropout_p)
            ).to(dtype=residual_dtype)
            residual_ref = (x0_ref * dmask0.float()) / (1 - dropout_p) + (
                x1_ref * dmask1.float()
            ) / (1 - dropout_p)
        else:
            residual_pt = ((x0_pt.float() * dmask0.float()) / (1 - dropout_p)).to(
                dtype=residual_dtype
            )
            residual_ref = (x0_ref * dmask0.float()) / (1 - dropout_p)
    if not is_rms_norm:
        out0_pt = F.layer_norm(
            residual_pt.to(dtype=weight_dtype), (hidden_size,), weight0_pt, bias0_pt, eps=epsilon
        ).to(dtype=input_dtype)
        out0_ref = F.layer_norm(residual_ref, (hidden_size,), weight0_ref, bias0_ref, eps=epsilon)
        if not tied_norm:
            out1_pt = F.layer_norm(
                residual_pt.to(dtype=weight_dtype),
                (hidden_size,),
                weight1_pt,
                bias1_pt,
                eps=epsilon,
            ).to(dtype=input_dtype)
            out1_ref = F.layer_norm(
                residual_ref, (hidden_size,), weight1_ref, bias1_ref, eps=epsilon
            )
    else:
        out0_pt = fused_rms_norm_affine(
            residual_pt.to(dtype=weight_dtype), weight0_pt, (hidden_size,), eps=epsilon
        ).to(dtype=input_dtype)
        out0_ref = fused_rms_norm_affine(residual_ref, weight0_ref, (hidden_size,), eps=epsilon)
        if not tied_norm:
            out1_pt = fused_rms_norm_affine(
                residual_pt.to(dtype=weight_dtype), weight1_pt, (hidden_size,), eps=epsilon
            ).to(dtype=input_dtype)
            out1_ref = fused_rms_norm_affine(residual_ref, weight1_ref, (hidden_size,), eps=epsilon)

    assert (out0 - out0_ref).abs().max() <= 4 * (out0_pt - out0_ref).abs().max() + 1e-4
    if not tied_norm:
        assert (out1 - out1_ref).abs().max() <= 4 * (out1_pt - out1_ref).abs().max() + 1e-4
    assert (residual - residual_ref).abs().max() <= 4 * (
        residual_pt - residual_ref
    ).abs().max() + 1e-4

    g0 = torch.randn_like(out0) / batch_size
    if tied_norm:
        (out0 * F.sigmoid(residual)).backward(g0)
        (out0_pt * F.sigmoid(residual_pt)).backward(g0)
        (out0_ref * F.sigmoid(residual_ref)).backward(g0)
    else:
        g1 = torch.randn_like(out1) / batch_size
        (out0 * F.sigmoid(residual) * g0 + out1 * g1).sum().backward()
        (out0_pt * F.sigmoid(residual_pt) * g0 + out1_pt * g1).sum().backward()
        (out0_ref * F.sigmoid(residual_ref) * g0 + out1_ref * g1).sum().backward()
    assert (x0.grad - x0_ref.grad).abs().max() <= 4 * (x0_pt.grad - x0_ref.grad).abs().max() + 1e-4
    if has_x1:
        assert (x1.grad - x1_ref.grad).abs().max() <= 4 * (
            x1_pt.grad - x1_ref.grad
        ).abs().max() + 1e-4
    if has_residual:
        assert (res.grad - res_ref.grad).abs().max() <= 4 * (
            res_pt.grad - res_ref.grad
        ).abs().max() + 1e-4
    assert (weight0.grad - weight0_ref.grad).abs().max() <= 3 * (
        weight0_pt.grad - weight0_ref.grad
    ).abs().max() + 3e-5
    if not is_rms_norm:
        assert (bias0.grad - bias0_ref.grad).abs().max() <= 2 * (
            bias0_pt.grad - bias0_ref.grad
        ).abs().max() + 3e-5
    if not tied_norm:
        assert (weight1.grad - weight1_ref.grad).abs().max() <= 3 * (
            weight1_pt.grad - weight1_ref.grad
        ).abs().max() + 3e-5
        if not is_rms_norm:
            assert (bias1.grad - bias1_ref.grad).abs().max() <= 2 * (
                bias1_pt.grad - bias1_ref.grad
            ).abs().max() + 3e-5


def test_dropout_layer_norm_randomness():
    hidden_size = 256
    dtype = torch.float32
    dropout_p = 0.1
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    seqlen = 512
    x0 = torch.randn(
        batch_size, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True
    )
    res = torch.randn_like(x0, dtype=dtype, requires_grad=True)
    model = DropoutAddLayerNorm(hidden_size, p=dropout_p, device=device, dtype=dtype)
    torch.random.manual_seed(42)
    _, dmask0 = dropout_add_layer_norm(
        x0, res, model.weight, model.bias, model.p, model.eps, return_dropout_mask=True
    )
    # Subsequent call should have a different dropout mask
    _, dmask1 = dropout_add_layer_norm(
        x0, res, model.weight, model.bias, model.p, model.eps, return_dropout_mask=True
    )
    torch.random.manual_seed(42)
    # Resetting the seed, should get the same dropout mask
    _, dmask2 = dropout_add_layer_norm(
        x0, res, model.weight, model.bias, model.p, model.eps, return_dropout_mask=True
    )
    assert not torch.equal(dmask0, dmask1)
    assert torch.equal(dmask0, dmask2)
