"""Forward O/LSE regressions for SingleTileLPTScheduler CLC L2 swizzle.

Requires real SM100/SM110 execution (skipped under FakeTensorMode).
"""

import os
from contextlib import contextmanager
from unittest import mock

import pytest
import torch

from flash_attn.cute import interface
from flash_attn.cute import utils as cute_utils
from flash_attn.cute.cache_utils import JITCache
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.prepare_scheduler import SchedulerMetadataTensorsTorch
from flash_attn.cute.testing import attention_ref, is_fake_mode
from flash_attn.cute.tile_scheduler import SchedulingMode, SingleTileLPTScheduler

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1"
    or is_fake_mode()
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="CLC swizzle O/LSE tests require real SM100/SM110 execution",
)

_captured_schedulers: list[tuple] = []
_orig_fwd_init = FlashAttentionForwardSm100.__init__


def _spy_init(self_inner, *args, **kwargs):
    _orig_fwd_init(self_inner, *args, **kwargs)
    _captured_schedulers.append(
        (
            self_inner.TileScheduler,
            self_inner.scheduling_mode,
            self_inner.use_2cta_instrs,
            self_inner.is_split_kv,
            self_inner.pack_gqa,
            tuple(self_inner.cluster_shape_mn),
        )
    )


@contextmanager
def _track_scheduler(monkeypatch, clc_enabled: bool):
    monkeypatch.setenv("FA_CLC", str(int(clc_enabled)))
    monkeypatch.setattr(cute_utils, "_fa_clc_enabled", clc_enabled)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    with mock.patch.object(FlashAttentionForwardSm100, "__init__", _spy_init):
        yield


def _predicted_swizzle(batch, scheduled_heads, sk, headdim, element_size):
    head_bytes = sk * (headdim + headdim) * element_size
    heads_fit = (50 * 1024 * 1024) // head_bytes
    return 1 << (max(heads_fit, 1).bit_length() - 1)


def _assert_scheduler(num_splits, pack_gqa, mode, use_2cta):
    assert _captured_schedulers, "Expected a forward kernel construction"
    sched_cls, sched_mode, use_2cta_actual, is_split_kv, pack_gqa_actual, cluster = (
        _captured_schedulers[-1]
    )
    assert sched_cls is SingleTileLPTScheduler, sched_cls
    assert sched_mode == mode, sched_mode
    assert use_2cta_actual == use_2cta, use_2cta_actual
    assert cluster == ((2, 1) if use_2cta else (1, 1)), cluster
    assert is_split_kv == (num_splits > 1), is_split_kv
    assert pack_gqa_actual == pack_gqa, pack_gqa_actual


@torch.no_grad()
def check_case(
    monkeypatch,
    *,
    batch,
    sq,
    sk,
    q_heads,
    kv_heads,
    dtype,
    causal=True,
    window_size=(None, None),
    num_splits=1,
    pack_gqa=False,
    headdim=128,
    mode=SchedulingMode.CLC,
    use_2cta=False,
    clc_enabled=True,
    scheduler_metadata=None,
):
    _captured_schedulers.clear()
    # Fresh in-memory cache so each case constructs the kernel and hits the scheduler spy.
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", JITCache())
    torch.manual_seed(42)
    q = torch.randn(batch, sq, q_heads, headdim, device="cuda", dtype=dtype)
    k = torch.randn(batch, sk, kv_heads, headdim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    kwargs = {
        "causal": causal,
        "window_size": window_size,
        "num_splits": num_splits,
        "pack_gqa": pack_gqa,
        "return_lse": True,
    }
    with _track_scheduler(monkeypatch, clc_enabled):
        if scheduler_metadata is None:
            out, lse = interface.flash_attn_func(q, k, v, **kwargs)
        else:
            out, lse = interface.flash_attn_varlen_func(
                q, k, v, scheduler_metadata=scheduler_metadata, **kwargs
            )
    torch.cuda.synchronize()
    _assert_scheduler(num_splits, pack_gqa, mode, use_2cta)
    assert lse is not None and out.shape == q.shape
    assert lse.shape == (batch, q_heads, sq) and lse.dtype == torch.float32
    assert torch.isfinite(out).all().item()

    out_ref_fp32, _, lse_ref = attention_ref(
        q.float(),
        k.float(),
        v.float(),
        causal=causal,
        window_size=window_size,
        return_lse=True,
    )
    out_ref = out_ref_fp32.to(dtype)
    out_pt, _ = attention_ref(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
    )
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    out_error = (out - out_ref).abs().max().item()
    reference_error = (out_pt - out_ref).abs().max().item()
    assert out_error <= 2 * reference_error + fwd_atol, (
        f"O error={out_error}, reference_error={reference_error}, atol={fwd_atol}"
    )

    assert not torch.isnan(lse).any().item()
    assert torch.equal(torch.isneginf(lse), torch.isneginf(lse_ref))
    assert torch.equal(torch.isposinf(lse), torch.isposinf(lse_ref))
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-4, rtol=1e-5)
    masked_rows = torch.isneginf(lse_ref).transpose(1, 2)
    if sq > sk and causal:
        assert masked_rows.any().item()
    if masked_rows.any().item():
        assert torch.count_nonzero(out[masked_rows]).item() == 0

    scheduled_heads = kv_heads if pack_gqa else q_heads
    swizzle = _predicted_swizzle(batch, scheduled_heads, sk, headdim, q.element_size())
    return {
        "predicted_swizzle": swizzle,
        "predicted_full_sections": batch * scheduled_heads // swizzle,
        "predicted_residual_heads": batch * scheduled_heads % swizzle,
    }


@pytest.fixture
def clc_monkeypatch(monkeypatch):
    monkeypatch.setenv("FA_CLC", "1")
    monkeypatch.setattr(cute_utils, "_fa_clc_enabled", True)
    return monkeypatch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("mask", ["causal", "local"])
@pytest.mark.parametrize(
    "batch,heads,full,residual",
    [(2, 4, 1, 0), (3, 5, 1, 7), (1, 3, 0, 3)],
    ids=["full", "full-and-residual", "residual-only"],
)
@pytest.mark.parametrize("num_splits", [1, 3], ids=["split1", "split3"])
def test_swizzle_sections(
    clc_monkeypatch, dtype, mask, batch, heads, full, residual, num_splits
):
    report = check_case(
        clc_monkeypatch,
        batch=batch,
        sq=513,
        sk=8192,
        q_heads=heads,
        kv_heads=heads,
        dtype=dtype,
        causal=mask == "causal",
        window_size=(None, None) if mask == "causal" else (1023, 0),
        num_splits=num_splits,
    )
    assert report["predicted_swizzle"] == 8
    assert report["predicted_full_sections"] == full
    assert report["predicted_residual_heads"] == residual


@pytest.mark.parametrize("sq", [1, 2])
@pytest.mark.parametrize("pack_gqa", [False, True], ids=["unpacked", "packed"])
def test_autoregressive_gqa(clc_monkeypatch, sq, pack_gqa):
    check_case(
        clc_monkeypatch,
        batch=3,
        sq=sq,
        sk=8193,
        q_heads=10,
        kv_heads=5,
        dtype=torch.bfloat16,
        pack_gqa=pack_gqa,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("num_splits", [1, 3], ids=["split1", "split3"])
def test_fully_masked_rows(clc_monkeypatch, dtype, num_splits):
    check_case(
        clc_monkeypatch,
        batch=1,
        sq=513,
        sk=129,
        q_heads=3,
        kv_heads=3,
        dtype=dtype,
        num_splits=num_splits,
    )


@pytest.mark.parametrize("clc_enabled", [False, True], ids=["static", "clc-requested"])
@pytest.mark.parametrize("headdim", [128, 256], ids=["hd128", "hd256"])
@pytest.mark.parametrize("mask", ["causal", "local"])
@pytest.mark.parametrize("pack_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_static_and_hd256_fallback(
    monkeypatch,
    clc_enabled,
    headdim,
    mask,
    pack_gqa,
    dtype,
):
    monkeypatch.setenv("FA_CLC", str(int(clc_enabled)))
    monkeypatch.setattr(cute_utils, "_fa_clc_enabled", clc_enabled)
    monkeypatch.setattr(cute_utils, "_fa_disable_2cta_enabled", False)
    mode = (
        SchedulingMode.CLC if clc_enabled and headdim == 128 else SchedulingMode.STATIC
    )
    check_case(
        monkeypatch,
        batch=3,
        sq=513,
        sk=1025,
        q_heads=10 if pack_gqa else 5,
        kv_heads=5,
        dtype=dtype,
        causal=mask == "causal",
        window_size=(None, None) if mask == "causal" else (255, 0),
        pack_gqa=pack_gqa,
        headdim=headdim,
        mode=mode,
        use_2cta=headdim == 256,
        clc_enabled=clc_enabled,
    )


@pytest.mark.parametrize("clc_enabled", [False, True], ids=["static", "clc"])
def test_repeat_call_stable(monkeypatch, clc_enabled):
    kwargs = {
        "batch": 3,
        "sq": 129,
        "sk": 1025,
        "q_heads": 5,
        "kv_heads": 5,
        "dtype": torch.bfloat16,
        "mode": SchedulingMode.CLC if clc_enabled else SchedulingMode.STATIC,
        "clc_enabled": clc_enabled,
    }
    check_case(monkeypatch, **kwargs)
    check_case(monkeypatch, **kwargs)


@pytest.mark.parametrize("clc_enabled", [False, True], ids=["static", "clc"])
def test_dynamic_split_metadata(monkeypatch, clc_enabled):
    counts = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    metadata = SchedulerMetadataTensorsTorch(None, counts, None, None, None)
    check_case(
        monkeypatch,
        batch=3,
        sq=513,
        sk=8192,
        q_heads=5,
        kv_heads=5,
        dtype=torch.bfloat16,
        num_splits=3,
        scheduler_metadata=metadata,
        mode=SchedulingMode.CLC if clc_enabled else SchedulingMode.STATIC,
        clc_enabled=clc_enabled,
    )
