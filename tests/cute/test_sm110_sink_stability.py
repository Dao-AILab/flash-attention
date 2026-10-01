"""Native SM110 sink output, LSE and gradient regressions against CPU FP64."""

import pytest
import torch

from flash_attn.cute import flash_attn_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="Requires a native SM110 GPU",
)


def reference(q, k, v, sink, causal, scale):
    qh = q.transpose(1, 2)
    repeat = q.shape[2] // k.shape[2]
    kh, vh = [x.repeat_interleave(repeat, dim=2).transpose(1, 2) for x in (k, v)]
    scores = (qh @ kh.transpose(-2, -1)) * scale
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        valid = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
        scores = scores.masked_fill(~valid, -torch.inf)
    sink_col = sink.view(1, -1, 1, 1).expand(q.shape[0], q.shape[2], q.shape[1], 1)
    lse = torch.logsumexp(torch.cat((scores, sink_col), dim=-1), dim=-1)
    out = (torch.exp(scores - lse[..., None]) @ vh).transpose(1, 2)
    return out, lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sink_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("sq,sk,causal", [(7, 128, False), (7, 3, True), (131, 257, True)])
@pytest.mark.parametrize("logit,sink_value", [(0.0, 100.0), (0.0, 8.0), (-100.0, 0.0), (100.0, 0.0)])
def test_sink_extreme_logits(dtype, sink_dtype, pack_gqa, sq, sk, causal, logit, sink_value):
    q = torch.ones(1, sq, 4, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.full((1, sk, 2, 64), logit / 64, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.ones_like(k, requires_grad=True)
    sink = torch.full((4,), sink_value, device="cuda", dtype=sink_dtype, requires_grad=True)
    inputs = (q, k, v, sink)
    cpu_inputs = tuple(x.detach().to(device="cpu", dtype=torch.float64).requires_grad_() for x in inputs)
    expected_out, expected_lse = reference(*cpu_inputs, causal, 1.0)
    out, lse = flash_attn_func(q, k, v, learnable_sink=sink, causal=causal,
                              softmax_scale=1.0, pack_gqa=pack_gqa, num_splits=1, return_lse=True)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(out.double().cpu(), expected_out, rtol=0.01, atol=2e-4)
    torch.testing.assert_close(lse.double().cpu(), expected_lse, rtol=1e-5, atol=2e-3)
    grads = torch.autograd.grad(lse.sum() + 0 * out.sum(), inputs)
    expected = torch.autograd.grad(expected_lse.sum() + 0 * expected_out.sum(), cpu_inputs)
    assert grads[-1].dtype == sink_dtype
    for actual, wanted in zip(grads, expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double().cpu(), wanted, rtol=0.02, atol=2e-3)
    if causal and sq > sk:
        assert torch.count_nonzero(out[:, :sq-sk]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("sq,sk,causal", [(19, 37, False), (7, 3, True), (131, 257, True)])
def test_sink_random_gradients(dtype, pack_gqa, sq, sk, causal):
    torch.manual_seed(123)
    q = (0.3 * torch.randn(1, sq, 4, 128, device="cuda", dtype=dtype)).requires_grad_()
    k = (0.3 * torch.randn(1, sk, 2, 128, device="cuda", dtype=dtype)).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    sink = torch.tensor([-4.0, 0.0, 3.0, 12.0], device="cuda", requires_grad=True)
    inputs = (q, k, v, sink)
    cpu_inputs = tuple(x.detach().to(device="cpu", dtype=torch.float64).requires_grad_() for x in inputs)
    scale = 0.7 / 128**0.5
    wanted_out, wanted_lse = reference(*cpu_inputs, causal, scale)
    out, lse = flash_attn_func(q, k, v, learnable_sink=sink, causal=causal, softmax_scale=scale,
                              pack_gqa=pack_gqa, num_splits=1, return_lse=True)
    torch.testing.assert_close(out.double().cpu(), wanted_out, rtol=0.03, atol=3e-3)
    torch.testing.assert_close(lse.double().cpu(), wanted_lse, rtol=1e-4, atol=1e-3)
    dout, dlse = torch.randn_like(out), torch.randn_like(lse)
    grads = torch.autograd.grad((out, lse), inputs, (dout, dlse))
    expected = torch.autograd.grad((wanted_out, wanted_lse), cpu_inputs,
                                   (dout.double().cpu(), dlse.double().cpu()))
    for actual, wanted in zip(grads, expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double().cpu(), wanted, rtol=0.03, atol=3e-3)
