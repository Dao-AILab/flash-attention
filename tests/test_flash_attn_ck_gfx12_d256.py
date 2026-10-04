"""Focused D256 backward validation for a CK extension built with gfx1200/1201.

Run with FLASH_ATTENTION_TRITON_AMD_ENABLE unset (the native CK backend).
No benchmark thresholds are part of these correctness tests.
"""

import os

import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires ROCm", allow_module_level=True)
arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(
    ":"
)[0]
if arch not in ("gfx1200", "gfx1201"):
    pytest.skip("requires gfx1200/gfx1201", allow_module_level=True)
if os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "").lower() == "true":
    pytest.skip("requires the native CK backend", allow_module_level=True)

from flash_attn import flash_attn_func  # noqa: E402 - skip before native import


def reference(q, k, v, causal, scale):
    repeats = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(repeats, dim=2)
    v = v.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        visible = torch.arange(sk, device=q.device)[None, :] <= (
            torch.arange(sq, device=q.device)[:, None] + sk - sq
        )
        scores = scores.masked_fill(~visible, -torch.inf)
        # Avoid softmax(-inf, ..., -inf) while keeping empty-row derivatives zero.
        valid = visible.any(dim=-1)[None, None, :, None]
        scores = torch.where(valid, scores, 0.0)
        p = torch.where(valid, scores.softmax(dim=-1), 0.0)
    else:
        p = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", p, v)


def make_tensor(shape, dtype, layout):
    if layout == "strided":
        b, s, h, d = shape
        return torch.randn(b, 2 * s, h, d, device="cuda", dtype=dtype)[
            :, ::2
        ].requires_grad_()
    if layout == "unaligned":
        # Last dimension contiguous, but base and row/head strides are not 16B aligned.
        x = torch.randn(*shape[:-1], shape[-1] + 1, device="cuda", dtype=dtype)
        return x[..., 1:].requires_grad_()
    return torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)


def check_case(
    sq, sk, causal, hk=4, layout="contiguous", dtype=torch.bfloat16, deterministic=False
):
    torch.manual_seed(731)
    q = make_tensor((2, sq, 4, 256), dtype, layout)
    k = make_tensor((2, sk, hk, 256), dtype, layout)
    v = make_tensor((2, sk, hk, 256), dtype, layout)
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    scale = 0.075  # Exercise non-default scaling as well as rectangular causal masking.
    expected = reference(qr, kr, vr, causal, scale)
    actual = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=causal,
        deterministic=deterministic,
    )
    dout = torch.randn_like(actual)
    got_grad = torch.autograd.grad(actual, (q, k, v), dout)
    ref_grad = torch.autograd.grad(expected, (qr, kr, vr), dout.double())
    for got, ref in zip((actual, *got_grad), (expected, *ref_grad)):
        assert torch.isfinite(got).all()
        error = (got.double() - ref).norm() / ref.norm().clamp_min(1e-12)
        assert error.item() < 0.01
        torch.testing.assert_close(got.double(), ref, atol=0.04, rtol=0.04)
    if causal and sq > sk:
        assert torch.count_nonzero(actual[:, : sq - sk]).item() == 0
        assert torch.count_nonzero(got_grad[0][:, : sq - sk]).item() == 0
    if deterministic:
        repeated = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
            deterministic=True,
        )
        repeated_grad = torch.autograd.grad(repeated, (q, k, v), dout)
        for a, b in zip(got_grad, repeated_grad):
            assert torch.equal(a, b)


@pytest.mark.parametrize("sq,sk", [(1, 1), (31, 65), (65, 31), (65, 65), (128, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_d256_backward(sq, sk, causal):
    check_case(sq, sk, causal)


@pytest.mark.parametrize("hk", [1, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_d256_gqa_strided(hk, causal):
    check_case(65, 97, causal, hk=hk, layout="strided")


@pytest.mark.parametrize("mode", ["unaligned", "fp16", "deterministic", "disabled"])
def test_d256_ck_fallback(mode, monkeypatch):
    if mode == "disabled":
        monkeypatch.setenv("FA_D256_BWD", "0")
    check_case(
        65,
        65,
        True,
        layout="unaligned" if mode == "unaligned" else "contiguous",
        dtype=torch.float16 if mode == "fp16" else torch.bfloat16,
        deterministic=mode == "deterministic",
    )


def test_d256_nondefault_stream():
    with torch.cuda.stream(torch.cuda.Stream()):
        check_case(65, 97, True)


def test_d256_query_device_not_current():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires a second GPU")
    selected = torch.cuda.current_device()
    other = (selected + 1) % torch.cuda.device_count()
    q = torch.randn(
        1, 32, 4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flash_attn_func(q, k, v)
    dout = torch.randn_like(out)
    expected = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    torch.cuda.synchronize(selected)
    with torch.cuda.device(other):
        actual = torch.autograd.grad(out, (q, k, v), dout)
        assert torch.cuda.current_device() == other
    torch.cuda.synchronize(selected)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
