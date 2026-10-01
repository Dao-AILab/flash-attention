"""SM90 learnable-sink normalization and gradient regressions."""

import math

import pytest
import torch

from flash_attn.cute import flash_attn_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="This regression suite targets a real SM90 GPU",
)


def _reference(q, k, v, sink, *, causal=False, window=(None, None), scale=None):
    """Small FP64 reference; ordinary logits are scaled, the sink logit is not."""
    batch, sq, hq, dim = q.shape
    sk, hkv = k.shape[1:3]
    assert hq % hkv == 0
    scale = dim**-0.5 if scale is None else scale
    qh = q.double().transpose(1, 2)
    kh = k.double().repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
    vh = v.double().repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
    scores = (qh @ kh.transpose(-2, -1)) * scale
    anchor = torch.arange(sq, device=q.device)[:, None] + sk - sq
    col = torch.arange(sk, device=q.device)[None, :]
    valid = torch.ones((sq, sk), device=q.device, dtype=torch.bool)
    if causal:
        valid &= col <= anchor
    left, right = window
    if left is not None:
        valid &= col >= anchor - left
    if right is not None:
        valid &= col <= anchor + right
    scores = scores.masked_fill(~valid[None, None], -torch.inf)
    sink_col = sink.double().view(1, hq, 1, 1).expand(batch, hq, sq, 1)
    lse = torch.logsumexp(torch.cat((scores, sink_col), dim=-1), dim=-1)
    # This reference's backward tests always use finite sinks, so LSE is finite.
    prob = torch.exp(scores - lse[..., None])
    return (prob @ vh).transpose(1, 2), lse


def _analytic_expected(q, k, sink, causal):
    batch, sq, heads, _ = q.shape
    sk = k.shape[1]
    if causal:
        count = (torch.arange(sq, device=q.device) + sk - sq + 1).clamp(0, sk)
    else:
        count = torch.full((sq,), sk, device=q.device)
    log_count = count.double().log().view(1, 1, sq)
    s = sink.double().view(1, heads, 1)
    lse = torch.logaddexp(log_count, s).expand(batch, heads, sq)
    mass = torch.exp(log_count - lse)
    return mass.transpose(1, 2)[..., None].expand_as(q), lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("sq,sk,causal", [(4, 64, False), (7, 3, True)])
def test_sink_analytic_normalization(dtype, pack_gqa, sq, sk, causal):
    # Per-head values exercise a negligible sink, equal logits, rebasing, and overflow.
    q = torch.zeros(1, sq, 4, 64, device="cuda", dtype=dtype)
    k = torch.zeros(1, sk, 2, 64, device="cuda", dtype=dtype)
    v = torch.ones_like(k)
    sink = torch.tensor([-100.0, 0.0, 8.0, 100.0], device="cuda")
    out, lse = flash_attn_func(
        q,
        k,
        v,
        learnable_sink=sink,
        causal=causal,
        pack_gqa=pack_gqa,
        num_splits=1,
        return_lse=True,
    )
    expected_out, expected_lse = _analytic_expected(q, k, sink, causal)
    assert lse is not None and lse.shape == expected_lse.shape
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(lse.double(), expected_lse, rtol=0.0, atol=1e-4)
    rtol = 3e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(out.double(), expected_out, rtol=rtol, atol=1e-4)
    if causal and sq > sk:
        assert torch.count_nonzero(out[:, : sq - sk]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sink_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sink_dominant_lse_gradient(dtype, sink_dtype):
    # Separate from the LSE assertion above: wrong LSE must not hide wrong dsink.
    batch, sq, sk, heads, dim = 2, 4, 64, 2, 64
    q = torch.zeros(batch, sq, heads, dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.zeros(batch, sk, heads, dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.ones_like(k, requires_grad=True)
    sink = torch.full((heads,), 100.0, device="cuda", dtype=sink_dtype, requires_grad=True)
    out, lse = flash_attn_func(
        q,
        k,
        v,
        learnable_sink=sink,
        pack_gqa=False,
        num_splits=1,
        return_lse=True,
    )
    assert lse is not None
    dq, dk, dv, dsink = torch.autograd.grad(
        (out, lse),
        (q, k, v, sink),
        (torch.zeros_like(out), torch.ones_like(lse)),
    )
    assert dsink.dtype == sink_dtype
    # d sum(LSE) / d sink ~= number of query rows in the batch.
    torch.testing.assert_close(
        dsink.float(),
        torch.full_like(dsink.float(), batch * sq),
        rtol=0.0,
        atol=1e-3,
    )
    for grad in (dq, dk, dv):
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize(
    "sq,sk,dim", [(19, 37, 64), (7, 3, 128), (131, 517, 96), (9, 257, 192), (9, 257, 256)]
)
@pytest.mark.parametrize(
    "causal,window",
    [
        (False, (None, None)),
        (True, (None, None)),
        (False, (2, 0)),
    ],
)
def test_sink_random_forward_backward(dtype, pack_gqa, sq, sk, dim, causal, window):
    torch.manual_seed(123)
    q = (0.5 * torch.randn(1, sq, 4, dim, device="cuda", dtype=dtype)).requires_grad_()
    k = (0.5 * torch.randn(1, sk, 2, dim, device="cuda", dtype=dtype)).requires_grad_()
    v = torch.randn(1, sk, 2, dim, device="cuda", dtype=dtype, requires_grad=True)
    sink = torch.tensor([-4.0, 0.0, 3.0, 12.0], device="cuda", requires_grad=True)
    inputs = (q, k, v, sink)
    ref_inputs = tuple(x.detach().double().requires_grad_() for x in inputs)
    # A non-default scale catches accidentally scaling the sink logit too.
    scale = 0.7 / math.sqrt(dim)
    out, lse = flash_attn_func(
        q,
        k,
        v,
        learnable_sink=sink,
        causal=causal,
        window_size=window,
        softmax_scale=scale,
        pack_gqa=pack_gqa,
        num_splits=1,
        return_lse=True,
    )
    ref_out, ref_lse = _reference(
        *ref_inputs,
        causal=causal,
        window=window,
        scale=scale,
    )
    assert lse is not None and lse.shape == ref_lse.shape
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    rtol, atol = (5e-3, 5e-4) if dtype == torch.float16 else (3e-2, 5e-3)
    torch.testing.assert_close(out.double(), ref_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(lse.double(), ref_lse, rtol=2e-4, atol=2e-4)
    dout, dlse = torch.randn_like(out), torch.randn_like(lse)
    grads = torch.autograd.grad((out, lse), inputs, (dout, dlse))
    ref_grads = torch.autograd.grad(
        (ref_out, ref_lse),
        ref_inputs,
        (dout.double(), dlse.double()),
    )
    for actual, expected in zip(grads, ref_grads):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, rtol=rtol, atol=atol)


def test_minus_inf_sink_matches_no_sink_including_masked_rows():
    q = torch.zeros(1, 7, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 3, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    sink = torch.full((4,), -torch.inf, device="cuda")
    kwargs = {"causal": True, "pack_gqa": False, "num_splits": 1, "return_lse": True}
    out0, lse0 = flash_attn_func(q, k, v, **kwargs)
    out1, lse1 = flash_attn_func(q, k, v, learnable_sink=sink, **kwargs)
    torch.testing.assert_close(out1, out0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lse1, lse0, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(out1[:, :4]) == 0
    assert torch.isneginf(lse1[..., :4]).all()


def test_zero_k_keeps_existing_sink_contract():
    q = torch.zeros(1, 5, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, 0, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.empty_like(k)
    sink = torch.tensor([0.0, 100.0], device="cuda")
    out, lse = flash_attn_func(q, k, v, learnable_sink=sink, return_lse=True)
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(lse, sink.view(1, 2, 1).expand(1, 2, 5))
