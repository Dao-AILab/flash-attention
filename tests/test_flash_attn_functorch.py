"""Tests for functorch / torch.func compatibility (see Dao-AILab/flash-attention#2071)."""

import pytest
import torch

try:
    import flash_attn_2_cuda  # noqa: F401
except ImportError:
    pytest.skip("flash_attn CUDA extension not built", allow_module_level=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attn_func_torch_func_grad():
    from flash_attn.flash_attn_interface import flash_attn_func

    batch, seqlen, nheads, headdim = 1, 32, 2, 32
    q = torch.randn(
        batch, seqlen, nheads, headdim, device="cuda", dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        batch, seqlen, nheads, headdim, device="cuda", dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen, nheads, headdim, device="cuda", dtype=torch.float16, requires_grad=True
    )

    def loss_fn(qx):
        o = flash_attn_func(qx, k, v, causal=False)
        return o.float().sum()

    g = torch.func.grad(loss_fn)(q)
    assert g.shape == q.shape
