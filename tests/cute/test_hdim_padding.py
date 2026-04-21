# Copyright (c) 2026

"""Tests for SM90 bwd tile_hdim alignment and stride consistency.

Covers head_dim in ``(128, 192]`` (the dKV_swapAB=True bwd config).
``head_dim in {136, 144, 152, 160, 168, 176}`` requires a multiple-of-64
tile_hdim for the SM90 WGMMA M=64 atom to partition the dK/dV
accumulator, and the preprocess / postprocess / dq_accum allocation
must all agree on that stride."""

import gc
import os
import random
from functools import wraps

import pytest
import torch

from flash_attn.cute.interface import (
    _flash_attn_bwd,
    _flash_attn_fwd,
    flash_attn_func,
)
from flash_attn.cute.testing import (
    attention_ref,
    is_fake_mode,
    maybe_fake_tensor_mode,
)


USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_SM90 = torch.cuda.get_device_capability()[0] == 9


def retry_on_oom(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError:
            if hasattr(_flash_attn_fwd, "compile_cache"):
                _flash_attn_fwd.compile_cache.clear()
            if hasattr(_flash_attn_bwd, "compile_cache"):
                _flash_attn_bwd.compile_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
            return func(*args, **kwargs)

    return wrapper


SWAPAB_HEAD_DIMS = [136, 144, 152, 160, 168, 176, 184, 192]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("has_window", [False, True])
@pytest.mark.parametrize("d", SWAPAB_HEAD_DIMS)
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (512, 512), (113, 211)])
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hdim_padding_fwd_bwd(seqlen_q, seqlen_k, d, has_window, causal, mha_type, dtype):
    if not IS_SM90:
        pytest.skip("dKV_swapAB=True config is SM90-specific")
    if has_window and causal:
        pytest.skip("redundant with causal=True")
    if mha_type != "mha":
        pytest.xfail("SM90 GQA bwd currently requires head_dim == head_dim_v")

    device = "cuda"
    random.seed(0)
    torch.manual_seed(0)
    batch = 2
    nheads = 4
    nheads_kv = nheads if mha_type == "mha" else 2
    dtype_ref = dtype
    d_v = 128

    q_ref = (
        torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
        .requires_grad_()
    )
    k_ref = (
        torch.randn(batch, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
        .requires_grad_()
    )
    v_ref = (
        torch.randn(batch, seqlen_k, nheads_kv, d_v, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
        .requires_grad_()
    )

    window_size = (None, None)
    if has_window:
        w_left = random.randrange(1, max(seqlen_k, 2))
        window_size = (w_left, 0 if causal else random.randrange(0, seqlen_k))

    q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, causal=causal, window_size=window_size, upcast=True
    )
    out_pt, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )

    out, _lse = flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
    )

    if is_fake_mode():
        return

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2.0
    fwd_err = (out - out_ref).abs().max().item()
    pt_err = (out_pt - out_ref).abs().max().item()
    assert fwd_err <= rtol * pt_err + fwd_atol, (
        f"fwd d={d}: kernel err {fwd_err:.3e} > {rtol} * pt err {pt_err:.3e} + {fwd_atol:.3e}"
    )

    g = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        g,
        retain_graph=True,
    )
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)

    for name, kernel_g, ref_g, pt_g in [
        ("dq", dq, dq_ref, dq_pt),
        ("dk", dk, dk_ref, dk_pt),
        ("dv", dv, dv_ref, dv_pt),
    ]:
        atol = 2 * (ref_g + 0.3 - 0.3 - ref_g).abs().max().item()
        kernel_err = (kernel_g - ref_g).abs().max().item()
        pt_err = (pt_g - ref_g).abs().max().item()
        assert kernel_err <= rtol * pt_err + atol, (
            f"bwd d={d} {name}: kernel err {kernel_err:.3e} > "
            f"{rtol} * pt err {pt_err:.3e} + {atol:.3e}"
        )


@pytest.mark.parametrize(
    "d_first,d_second",
    [
        (144, 160),
        (160, 144),
        (144, 176),
        (176, 144),
        (128, 144),
        (144, 192),
    ],
)
@retry_on_oom
def test_cross_hdim_no_stale_accum(d_first, d_second):
    """Two head_dims that share a padded tile_hdim (e.g. 144 and 160 both
    pad to 192) must be independently correct when run back-to-back in
    the same process — if preprocess zero-fills less than the bwd kernel
    writes, the second call reads stale accumulator data left by the
    first."""
    if not IS_SM90:
        pytest.skip("dKV_swapAB=True config is SM90-specific")

    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen, heads, d_v = 1, 128, 2, 128
    rtol = 2.0

    def one_trial(d):
        random.seed(0)
        torch.manual_seed(0)
        q_ref = torch.randn(batch, seqlen, heads, d, device=device, dtype=dtype).requires_grad_()
        k_ref = torch.randn(batch, seqlen, heads, d, device=device, dtype=dtype).requires_grad_()
        v_ref = torch.randn(batch, seqlen, heads, d_v, device=device, dtype=dtype).requires_grad_()
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
        out_ref, _ = attention_ref(q_ref, k_ref, v_ref, causal=True, upcast=True)
        out_pt, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            causal=True,
            upcast=False,
            reorder_ops=True,
        )
        out, _ = flash_attn_func(q, k, v, causal=True)
        g = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(
            out_ref,
            (q_ref, k_ref, v_ref),
            g,
            retain_graph=True,
        )
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
        for name, kernel_g, ref_g, pt_g in [
            ("dq", dq, dq_ref, dq_pt),
            ("dk", dk, dk_ref, dk_pt),
            ("dv", dv, dv_ref, dv_pt),
        ]:
            atol = 2 * (ref_g + 0.3 - 0.3 - ref_g).abs().max().item()
            kernel_err = (kernel_g - ref_g).abs().max().item()
            pt_err = (pt_g - ref_g).abs().max().item()
            assert kernel_err <= rtol * pt_err + atol, (
                f"d={d} {name}: kernel err {kernel_err:.3e} > "
                f"{rtol} * pt err {pt_err:.3e} + {atol:.3e}"
            )

    one_trial(d_first)
    one_trial(d_second)
