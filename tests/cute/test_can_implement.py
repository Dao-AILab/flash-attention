# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Tests for the `can_implement` pre-compile guards wired into interface.py.
#
# These tests do NOT require a GPU. They cover two layers:
#   1. The `can_implement` static methods themselves (the exact predicate each guard
#      evaluates), with deliberately invalid configs that must be rejected and valid
#      configs that must be accepted.
#   2. The interface guards in `_flash_attn_fwd` / `_flash_attn_bwd`, asserting that an
#      invalid config raises a clear `RuntimeError` *before* `cute.compile` is reached,
#      and that a valid config passes the guard (reaching compilation).
#
# Only architectures whose selected kernel class exposes `can_implement` are guarded:
# forward = SM80 / SM120, backward = SM80 / SM90 / SM120. The SM90 and SM100 forward
# kernels and the SM100 backward kernel do not define `can_implement`, so they are not
# exercised here.

import cutlass
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn.cute import interface
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

FP16 = cutlass.Float16
FP32 = cutlass.Float32


# ---------------------------------------------------------------------------
# Layer 1: `can_implement` static methods (pure, no GPU, no compile).
# ---------------------------------------------------------------------------


def test_fwd_sm80_can_implement_rejects_invalid_and_accepts_valid():
    # Unsupported dtype.
    assert not FlashAttentionForwardSm80.can_implement(FP32, 64, 64, 128, 128, 1, 128, False)
    # head_dim not a multiple of 8.
    assert not FlashAttentionForwardSm80.can_implement(FP16, 100, 64, 128, 128, 1, 128, False)
    # tile_n not a multiple of 16.
    assert not FlashAttentionForwardSm80.can_implement(FP16, 64, 64, 128, 100, 1, 128, False)
    # Valid config.
    assert FlashAttentionForwardSm80.can_implement(FP16, 64, 64, 128, 128, 1, 128, False)


def test_fwd_sm120_can_implement_enforces_99kb_smem_budget():
    # head_dim=256 with tile 128x64 needs ~131 KB of SMEM, exceeding SM120's 99 KB
    # budget, so it must be rejected (this is the config interface.py selects for SM120
    # head_dim>64).
    assert not FlashAttentionForwardSm120.can_implement(FP16, 256, 256, 128, 64, 1, 128, False)
    # The very same config fits within SM80's larger 163 KB budget: confirms the
    # rejection above is specifically the tighter SM120 capacity, not generic logic.
    assert FlashAttentionForwardSm80.can_implement(FP16, 256, 256, 128, 64, 1, 128, False)
    # Valid SM120 config (head_dim=64, tile 128x128 -> ~48 KB).
    assert FlashAttentionForwardSm120.can_implement(FP16, 64, 64, 128, 128, 1, 128, False)


def test_bwd_sm80_can_implement_rejects_invalid_and_accepts_valid():
    # Unsupported dtype.
    assert not FlashAttentionBackwardSm80.can_implement(FP32, 64, 64, 64, 64, 1, 1, 128, False)
    # n_block_size not a multiple of 16.
    assert not FlashAttentionBackwardSm80.can_implement(FP16, 64, 64, 64, 60, 1, 1, 128, False)
    # Valid config.
    assert FlashAttentionBackwardSm80.can_implement(FP16, 64, 64, 64, 64, 1, 1, 128, False)


def test_bwd_sm90_can_implement_rejects_invalid_and_accepts_valid():
    # Unsupported dtype.
    assert not FlashAttentionBackwardSm90.can_implement(FP32, 64, 64, 128, 64, 2, 256, False)
    # tile_n not a multiple of 16.
    assert not FlashAttentionBackwardSm90.can_implement(FP16, 64, 64, 128, 60, 2, 256, False)
    # num_threads not a multiple of 32.
    assert not FlashAttentionBackwardSm90.can_implement(FP16, 64, 64, 128, 64, 2, 100, False)
    # (tile_m * 2) must be divisible by num_threads: 64*2=128 is not divisible by 256.
    assert not FlashAttentionBackwardSm90.can_implement(FP16, 64, 64, 64, 64, 2, 256, False)
    # Valid config: 128*2=256 is divisible by num_threads=256.
    assert FlashAttentionBackwardSm90.can_implement(FP16, 64, 64, 128, 64, 2, 256, False)


def test_bwd_sm120_can_implement_enforces_99kb_smem_budget():
    # head_dim=256 with tile 64x64 needs ~128 KB of SMEM, exceeding SM120's 99 KB budget.
    assert not FlashAttentionBackwardSm120.can_implement(FP16, 256, 256, 64, 64, 1, 1, 128, False)
    # Same config fits within SM80's 163 KB budget.
    assert FlashAttentionBackwardSm80.can_implement(FP16, 256, 256, 64, 64, 1, 1, 128, False)
    # Valid SM120 config (head_dim=64).
    assert FlashAttentionBackwardSm120.can_implement(FP16, 64, 64, 64, 64, 2, 2, 128, False)


# ---------------------------------------------------------------------------
# Layer 2: interface guards raise RuntimeError before cute.compile.
#
# We force the SM120 path (tight 99 KB SMEM budget) via the `_arch` override (forward)
# or by patching `_get_device_arch` (backward). FakeTensorMode keeps everything off the
# GPU; the guards run before `cute.compile`, so no actual kernel is built.
# ---------------------------------------------------------------------------


class _CompileReached(Exception):
    """Sentinel raised in place of cute.compile to prove the guard let the flow pass."""


def _raise_compile_reached(*args, **kwargs):
    raise _CompileReached()


def _fake_qkv(head_dim, *, dtype=torch.float16, seqlen=128):
    q = torch.randn(1, seqlen, 1, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(1, seqlen, 1, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(1, seqlen, 1, head_dim, dtype=dtype, device="cuda")
    return q, k, v


def test_fwd_guard_raises_runtime_error_before_compile_sm120():
    with FakeTensorMode():
        q, k, v = _fake_qkv(256)
        with pytest.raises(RuntimeError, match="forward kernel"):
            _flash_attn_fwd(q, k, v, causal=False, _arch=120)


def test_fwd_guard_passes_for_valid_config_sm120(monkeypatch):
    monkeypatch.setattr(interface.cute, "compile", _raise_compile_reached)
    with FakeTensorMode():
        q, k, v = _fake_qkv(64)
        # Valid config: the guard must not raise, so the flow reaches our patched compile.
        with pytest.raises(_CompileReached):
            _flash_attn_fwd(q, k, v, causal=False, _arch=120)


def test_bwd_guard_raises_runtime_error_before_compile_sm120(monkeypatch):
    monkeypatch.setattr(interface, "_get_device_arch", lambda: 120)
    # The preprocess kernel compiles before the main guard; stub it out so the test
    # stays fast and isolates the can_implement guard.
    monkeypatch.setattr(interface, "_bwd_preprocess", lambda *a, **k: None)
    with FakeTensorMode():
        q, k, v = _fake_qkv(256)
        out = torch.randn(1, 128, 1, 256, dtype=torch.float16, device="cuda")
        dout = torch.randn(1, 128, 1, 256, dtype=torch.float16, device="cuda")
        lse = torch.randn(1, 1, 128, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="backward kernel"):
            _flash_attn_bwd(q, k, v, out, dout, lse, causal=False)


def test_bwd_guard_passes_for_valid_config_sm120(monkeypatch):
    monkeypatch.setattr(interface, "_get_device_arch", lambda: 120)
    monkeypatch.setattr(interface, "_bwd_preprocess", lambda *a, **k: None)
    monkeypatch.setattr(interface.cute, "compile", _raise_compile_reached)
    with FakeTensorMode():
        q, k, v = _fake_qkv(64)
        out = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
        dout = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
        lse = torch.randn(1, 1, 128, dtype=torch.float32, device="cuda")
        with pytest.raises(_CompileReached):
            _flash_attn_bwd(q, k, v, out, dout, lse, causal=False)
