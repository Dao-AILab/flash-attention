"""Sink-gradient precision oracles independent of randomized-test tolerances."""

import os

import pytest
import torch

import flash_attn.cute.interface as interface
from flash_attn.cute.testing import attention_ref, is_fake_mode, maybe_fake_tensor_mode

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11],
    reason="Dense O residual requires Blackwell",
)


def call_attention(q, k, v, sink, num_splits, varlen, causal=False):
    kwargs = dict(
        learnable_sink=sink, num_splits=num_splits, causal=causal, return_lse=True
    )
    if not varlen:
        return interface.flash_attn_func(q, k, v, **kwargs)
    batch, sq = q.shape[:2]
    sk = k.shape[1]
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sq
    cu_k = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sk
    return interface.flash_attn_varlen_func(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=sq,
        max_seqlen_k=sk,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("varlen", [False, True])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sink_backward_consumes_residual(dtype, num_splits, varlen, monkeypatch):
    torch.manual_seed(0)
    q = torch.zeros(2, 113, 6, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 257, 2, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    # FP32 sink keeps its final gradient cast from hiding the improvement.
    sink = torch.zeros(6, device="cuda", dtype=torch.float32, requires_grad=True)
    output = call_attention(q, k, v, sink, num_splits, varlen, causal=True)[0]
    grad_output = dout.flatten(0, 1) if varlen else dout
    enabled = torch.autograd.grad(output, sink, grad_output)[0]
    original = interface._bwd_preprocess

    def disable(*args, **kwargs):
        kwargs["o_lo"] = None
        return original(*args, **kwargs)

    disable.compile_cache = original.compile_cache
    monkeypatch.setattr(interface, "_bwd_preprocess", disable)
    output_off = call_attention(q, k, v, sink, num_splits, varlen, causal=True)[0]
    disabled = torch.autograd.grad(output_off, sink, grad_output)[0]
    if is_fake_mode():
        return
    # q=0, sink=0: visible keys and sink have equal, exactly representable
    # unnormalized weights. This removes the PV probability-rounding confound.
    visible = (
        torch.arange(257, device="cuda")[None, :]
        <= torch.arange(113, device="cuda")[:, None] + 257 - 113
    ).double()
    denom = visible.sum(-1) + 1
    out_ref = torch.einsum("qk,bkhd->bqhd", visible, v.double().repeat_interleave(3, 2))
    out_ref /= denom[None, :, None, None]
    exact = -((dout.double() * out_ref).sum(-1) / denom[None, :, None]).sum((0, 1))
    enabled_error = (enabled.double() - exact).abs().max()
    disabled_error = (disabled.double() - exact).abs().max()
    assert torch.equal(output, output_off)
    assert enabled_error <= 1e-6
    assert enabled_error * 32 < disabled_error


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("num_splits", [1, 3])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sink_inference_skips_residual(context, varlen, num_splits, monkeypatch):
    torch.manual_seed(5)
    q = torch.randn(
        1, 128, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    sink = torch.randn(2, device="cuda", dtype=torch.float32, requires_grad=True)
    observed = []
    original = interface._flash_attn_fwd

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result[4] is not None)
        return result

    observe.compile_cache = original.compile_cache
    monkeypatch.setattr(interface, "_flash_attn_fwd", observe)
    with context():
        plain = call_attention(q, k, v, sink, num_splits, varlen)
        detached = call_attention(q, k, v, sink.detach(), num_splits, varlen)
    with torch.enable_grad():
        training = call_attention(q, k, v, sink, num_splits, varlen)[0]
        gradient = torch.autograd.grad(training, sink, torch.ones_like(training))[0]
    if is_fake_mode():
        return
    assert observed == [False, False, True]
    assert torch.equal(plain[0], detached[0])
    assert torch.equal(plain[1], detached[1])
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_reference_keeps_small_sink_probability(dtype):
    q = torch.zeros(1, 1, 1, 64, device="cuda", dtype=dtype)
    k = torch.zeros(1, 1024, 1, 64, device="cuda", dtype=dtype)
    sink = torch.full((1,), -5.0, device="cuda", dtype=torch.float32)
    _, attention, _, probability = attention_ref(
        q,
        k,
        k,
        learnable_sink=sink,
        return_lse=True,
        return_sink_prob=True,
    )
    if is_fake_mode():
        return
    expected = torch.exp(sink.double()) / (1024 + torch.exp(sink.double()))
    assert probability.dtype == torch.float32
    torch.testing.assert_close(
        probability.flatten().double(), expected, atol=0, rtol=1e-6
    )
    assert (1 - attention.float().sum(-1)).item() == 0
