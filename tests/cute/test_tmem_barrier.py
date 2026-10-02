"""Run with compute-sanitizer --tool synccheck to check cross-role TMEM barriers."""

import math

import pytest
import torch

from flash_attn.cute import flash_attn_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110 TMEM allocation protocol",
)


@pytest.mark.parametrize("seqlen_q", [1, 65], ids=["q_stage1", "q_stage2"])
def test_tmem_cross_role_handoff(seqlen_q):
    torch.manual_seed(0)
    head_dim, num_heads, seqlen_k = 128, 4, 257
    q = torch.randn(
        1, seqlen_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(1, seqlen_k, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    # The larger packed Q crosses the Q-stage boundary.
    out, lse = flash_attn_func(q, k, v, pack_gqa=True, num_splits=3, return_lse=True)
    q, k, v = (tensor.cpu().transpose(1, 2) for tensor in (q, k, v))
    scores = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(head_dim)
    expected = (scores.softmax(-1) @ v.double()).transpose(1, 2)
    eager = ((q @ k.transpose(-1, -2) / math.sqrt(head_dim)).softmax(-1) @ v).transpose(
        1, 2
    )
    error = (out.cpu().double() - expected).abs()
    eager_error = (eager.double() - expected).abs()
    # Allow BF16 output rounding in addition to the low-precision eager error.
    allowance = torch.finfo(out.dtype).eps * expected.abs()
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    assert error.mean() <= eager_error.mean() + allowance.mean()
    assert error.max() <= eager_error.max() + allowance.max()
    lse_error = (lse.cpu().double() - scores.logsumexp(-1)).abs()
    assert lse_error.max() <= head_dim * torch.finfo(torch.float32).eps * (
        1 + scores.abs().max()
    )
