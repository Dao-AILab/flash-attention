"""Native Thor short-prefill output and gradient coverage at the scheduling boundaries."""

import pytest
import torch

from flash_attn.cute import flash_attn_func


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sink_value", [None, 8.0])
@pytest.mark.parametrize("sq,sk,hkv,batch", [
    (65, 129, 8, 1), (128, 128, 8, 1), (128, 257, 8, 1),
    (64, 128, 4, 1), (32, 64, 2, 1), (129, 129, 8, 1), (128, 128, 8, 2),
])
def test_short_prefill(dtype, sink_value, sq, sk, hkv, batch):
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("Native SM110 validation")
    torch.manual_seed(20261001)
    q = torch.randn(batch, sq, 16, 128, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch, sk, hkv, 128, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    sink = None if sink_value is None else torch.full((16,), sink_value, device="cuda", requires_grad=True)
    out, lse = flash_attn_func(q, k, v, causal=True, pack_gqa=True, num_splits=1,
                               learnable_sink=sink, return_lse=True)
    scores = q.float().transpose(1, 2) @ k.float().repeat_interleave(16 // hkv, dim=2).transpose(1, 2).transpose(2, 3)
    scores *= 128 ** -0.5
    mask = torch.arange(sk, device="cuda")[None, :] <= torch.arange(sq, device="cuda")[:, None] + sk - sq
    scores = scores.masked_fill(~mask, -torch.inf)
    if sink is not None:
        scores_with_sink = torch.cat((scores, sink[None, :, None, None].expand(batch, -1, sq, 1)), dim=-1)
    else:
        scores_with_sink = scores
    probabilities = scores_with_sink.softmax(-1)[..., :sk]
    ref = (probabilities @ v.float().repeat_interleave(16 // hkv, dim=2).transpose(1, 2)).transpose(1, 2)
    ref_lse = scores_with_sink.logsumexp(-1)
    torch.testing.assert_close(out.float(), ref, atol=0.006, rtol=0.02)
    torch.testing.assert_close(lse, ref_lse, atol=0.003, rtol=0.003)
    upstream = torch.randn_like(out)
    upstream_lse = torch.randn_like(lse)
    inputs = (q, k, v) if sink is None else (q, k, v, sink)
    actual_grads = torch.autograd.grad((out, lse), inputs, (upstream, upstream_lse), retain_graph=True)
    reference_grads = torch.autograd.grad((ref, ref_lse), inputs, (upstream.float(), upstream_lse))
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.025, rtol=0.03)
