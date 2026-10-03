import pytest
import torch

from flash_attn.modules.mha import MHA


@pytest.mark.parametrize(
    ("cross_attn", "num_heads_kv"),
    [
        pytest.param(False, 2, id="gqa-self-attention"),
        pytest.param(True, 8, id="equal-head-cross-attention"),
        pytest.param(True, 2, id="gqa-cross-attention"),
    ],
)
def test_mha_dwconv(cross_attn, num_heads_kv):
    batch_size, query_seqlen, kv_seqlen, embed_dim = 2, 5, 7, 64
    model = MHA(
        embed_dim=embed_dim,
        num_heads=8,
        num_heads_kv=num_heads_kv,
        cross_attn=cross_attn,
        dwconv=True,
    )
    x = torch.randn(batch_size, query_seqlen, embed_dim, requires_grad=True)
    x_kv = torch.randn(batch_size, kv_seqlen, embed_dim, requires_grad=True) if cross_attn else None

    output = model(x, x_kv=x_kv)
    assert output.shape == x.shape

    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    if x_kv is not None:
        assert torch.isfinite(x_kv.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
