import sys
import types

import pytest
import torch

sys.modules.setdefault("flash_attn_3", types.SimpleNamespace(_C=object()))
sys.modules.setdefault("flash_attn_3._C", object())

from hopper import flash_attn_interface as hopper_interface


class _FakeTensor:
    def __init__(self, device="cuda"):
        self.device = device


def _make_qkv():
    q = _FakeTensor()
    k = _FakeTensor()
    v = _FakeTensor()
    return q, k, v


def test_hopper_guard_rejects_sm120(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))

    q, k, v = _make_qkv()
    with pytest.raises(RuntimeError, match=r"only supports Hopper GPUs \(SM90\)\. Got SM120\."):
        hopper_interface.flash_attn_func(q, k, v)


def test_hopper_qkvpacked_guard_rejects_sm120(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))

    qkv = _FakeTensor()
    with pytest.raises(RuntimeError, match=r"only supports Hopper GPUs \(SM90\)\. Got SM120\."):
        hopper_interface.flash_attn_qkvpacked_func(qkv)


def test_hopper_varlen_guard_rejects_sm120(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))

    q, k, v = _make_qkv()
    cu = _FakeTensor()
    with pytest.raises(RuntimeError, match=r"only supports Hopper GPUs \(SM90\)\. Got SM120\."):
        hopper_interface.flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q=1,
            max_seqlen_k=1,
        )


def test_hopper_kvcache_guard_rejects_sm120(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))

    q, k_cache, v_cache = _make_qkv()
    with pytest.raises(RuntimeError, match=r"only supports Hopper GPUs \(SM90\)\. Got SM120\."):
        hopper_interface.flash_attn_with_kvcache(q, k_cache, v_cache)
