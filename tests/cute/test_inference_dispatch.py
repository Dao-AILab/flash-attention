"""Dense inference retains output/LSE semantics and the differentiable route."""

import math

import pytest
import torch

from flash_attn.cute import flash_attn_func, interface


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sink_value", [None, 8.0])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_inference_dispatch(dtype, causal, sink_value, requires_grad):
    torch.manual_seed(20261001)
    q = torch.randn(2, 17, 8, 64, device="cuda", dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(2, 33, 4, 64, device="cuda", dtype=dtype, requires_grad=requires_grad)
    v = torch.randn_like(k, requires_grad=requires_grad)
    sink = None if sink_value is None else torch.full(
        (8,), sink_value, device="cuda", dtype=torch.float32, requires_grad=requires_grad
    )
    inputs = [q, k, v] + ([] if sink is None else [sink])
    refs = [t.detach().double().cpu().requires_grad_(requires_grad) for t in inputs]
    qr, kr, vr = refs[:3]
    scores = torch.einsum("bqhd,bkhd->bhqk", qr, kr.repeat_interleave(2, dim=2)) / math.sqrt(64)
    if causal:
        mask = torch.arange(33)[None, :] > torch.arange(17)[:, None] + 16
        scores = scores.masked_fill(mask, -math.inf)
    if sink is not None:
        scores = torch.cat([scores, refs[3][None, :, None, None].expand(2, 8, 17, 1)], dim=-1)
    lse_ref = torch.logsumexp(scores, dim=-1)
    out_ref = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1)[..., :33], vr.repeat_interleave(2, dim=2))
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            out, lse = flash_attn_func(q, k, v, causal=causal, learnable_sink=sink, return_lse=True)
        assert not out.requires_grad and not lse.requires_grad
        torch.testing.assert_close(out.double().cpu(), out_ref.detach(), rtol=0.02, atol=0.008)
        torch.testing.assert_close(lse.double().cpu(), lse_ref.detach(), rtol=0.002, atol=0.003)
    if requires_grad:
        out, lse = flash_attn_func(q, k, v, causal=causal, learnable_sink=sink, return_lse=True)
        dout = torch.randn_like(out)
        (out.mul(dout).sum() + 0.1 * lse.sum()).backward()
        (out_ref.mul(dout.double().cpu()).sum() + 0.1 * lse_ref.sum()).backward()
        for actual, ref in zip(inputs, refs):
            torch.testing.assert_close(actual.grad.double().cpu(), ref.grad, rtol=0.03, atol=0.008)


@pytest.mark.parametrize("explicit_qv", [False, True])
def test_inference_shared_kv_dispatch(monkeypatch, explicit_qv):
    q = torch.zeros(1, 2, 4, 512)
    kv = torch.zeros(1, 3, 1, 512)
    qv = q.clone() if explicit_qv else None
    calls = []

    def observed_fwd(q_arg, k_arg, v_arg, **kwargs):
        calls.append((q_arg, k_arg, v_arg, kwargs))
        return torch.zeros_like(q), None, None, None, None

    monkeypatch.setattr(interface, "_flash_attn_fwd", observed_fwd)
    with torch.no_grad():
        flash_attn_func(q, kv, kv, qv=qv)
    q_arg, k_arg, v_arg, kwargs = calls[0]
    assert q_arg is None and k_arg is None and v_arg is kv
    assert kwargs["qv"] is (qv if explicit_qv else q)
