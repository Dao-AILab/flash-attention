import re

import pytest
import torch
from flash_attn.models.vit import vit_base_patch16_224 as flash_vit_base_patch16_224
from timm.models.vision_transformer import vit_base_patch16_224


@pytest.mark.parametrize("fused_mlp", [False, True])
# @pytest.mark.parametrize('fused_mlp', [False])
@pytest.mark.parametrize("optimized", [False, True])
# @pytest.mark.parametrize('optimized', [True])
def test_vit(optimized, fused_mlp):
    """Check that our implementation of ViT matches the timm's implementation:
    the output of our forward pass in fp16 should be around the same as
    timm' forward pass in fp16, when compared to timm's forward pass in fp32.
    """
    dtype = torch.float16
    device = "cuda"

    kwargs = {}
    if optimized:
        kwargs = dict(use_flash_attn=True, fused_bias_fc=True, fused_dropout_add_ln=True)
    kwargs["fused_mlp"] = fused_mlp
    model = flash_vit_base_patch16_224(**kwargs).to(device=device, dtype=dtype)

    model_ref = vit_base_patch16_224(pretrained=True).to(device=device)
    model_timm = vit_base_patch16_224(pretrained=True).to(device=device, dtype=dtype)

    model.load_state_dict(model_ref.state_dict())

    model.eval()
    model_ref.eval()
    model_timm.eval()

    torch.manual_seed(0)
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
    out = model(x)
    out_timm = model_timm(x)
    out_ref = model_ref(x.float())

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"timm fp16 max diff: {(out_timm - out_ref).abs().max().item()}")
    print(f"timm fp16 mean diff: {(out_timm - out_ref).abs().mean().item()}")
    rtol = 2 if not fused_mlp else 8
    assert (out - out_ref).abs().max().item() < rtol * (out_timm - out_ref).abs().max().item()
