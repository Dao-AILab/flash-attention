import json
import subprocess
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

if "flash_attn" not in sys.modules:
    package = types.ModuleType("flash_attn")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "flash_attn")]
    sys.modules["flash_attn"] = package

from flash_attn.cute.autotune import (
    _compile_barrier,
    _decode_config,
    _measure_configs,
    _normalize_kwargs,
    compile_bucket,
    tune_fwd_config,
)
from flash_attn.cute.cache_utils import CUTE_DSL_CACHE_ENABLED
from flash_attn.cute import interface
from flash_attn.cute.config import (
    FwdConfig,
    Sm90Registers,
    Sm100Registers,
    derive_fwd_config,
    select_fwd_config,
)
from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_fwd_combine

_BASE_CONFIG = FwdConfig(
    device_capacity=10,
    tile_m=128,
    tile_n=128,
    mma_pv_is_rs=False,
    intra_wg_overlap=False,
    q_stage=1,
    pack_gqa=False,
    use_clc_scheduler=False,
    is_static_persistent=False,
    use_tma_o=False,
    use_s_ping_pong=False,
    num_splits=1,
    use_2cta_instrs=False,
    registers=Sm100Registers(softmax=200, correction=64, other=48),
    overlap_sO_sQ=False,
    kv_stage=12,
)


@pytest.mark.parametrize("pack_gqa", [False, True])
def test_compile_payload_round_trips_registers_pack_gqa_and_layout(pack_gqa):
    sm90 = replace(
        _BASE_CONFIG,
        device_capacity=9,
        mma_pv_is_rs=True,
        intra_wg_overlap=True,
        registers=Sm90Registers(mma=240, producer=24),
    )
    configs = (
        replace(_BASE_CONFIG, q_stage=2, kv_stage=10, overlap_sO_sQ=True),
        sm90,
        replace(_BASE_CONFIG, device_capacity=8, registers=None),
    )
    for config in configs:
        config = replace(config, pack_gqa=pack_gqa)
        assert _decode_config(json.loads(json.dumps(asdict(config)))) == config


@pytest.mark.parametrize(
    "name",
    [
        "lse",
        "softcap",
        "gather_kv_indices",
        "score_mod",
        "learnable_sink",
        "seqused_q",
        "qv",
        "return_lse",
    ],
)
def test_initial_scope_rejects_lse_modifier_and_sparse_paths(name):
    with pytest.raises(NotImplementedError, match=name):
        _normalize_kwargs({name: True})


def test_dynamic_shape_churn_has_bounded_compile_specializations(monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_NUM_SMS", "152")

    class SpecRecorder:
        """Record unique keys and report cache hits without compiling."""

        def __init__(self):
            self.keys = set()

        def __contains__(self, key):
            self.keys.add(key)
            return True

    main, combine = SpecRecorder(), SpecRecorder()
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", main)
    monkeypatch.setattr(_flash_attn_fwd_combine, "compile_cache", combine)
    with FakeTensorMode():
        # More distinct shapes than allowed keys detects shape-dependent compilation.
        for index in range(32):
            batch = 1 + index
            seqlen_q = 1 + (index * 131) % 8191
            seqlen_k = 1 + (index * 8191) % 65521
            q = torch.empty(batch, seqlen_q, 32, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.empty(batch, seqlen_k, 8, 128, device="cuda", dtype=torch.bfloat16)
            _flash_attn_fwd(q, k, k, num_splits=0, _arch=103)

    assert 0 < len(main.keys) <= 8
    assert 0 < len(combine.keys) <= 2


def test_compile_barrier_identifies_the_failing_config(monkeypatch):
    bad = replace(_BASE_CONFIG, tile_m=64)

    def compile_one(config, *_):
        if config == bad:
            raise subprocess.TimeoutExpired("compile", 1)

    monkeypatch.setattr("flash_attn.cute.autotune._compile_one", compile_one)
    kwargs = {"v": torch.empty(1, 1, 1, 64), "_arch": 100}
    with pytest.raises(RuntimeError, match=r"parallel compile failed.*tile_m=64"):
        _compile_barrier((_BASE_CONFIG, bad), kwargs, workers=2)


def test_compile_bucket_deduplicates_exact_split_counts():
    split_two = replace(_BASE_CONFIG, num_splits=2)
    assert compile_bucket(split_two, 128) == compile_bucket(
        replace(_BASE_CONFIG, num_splits=4), 128
    )
    for distinct in (
        _BASE_CONFIG,
        replace(_BASE_CONFIG, num_splits=33),
        replace(split_two, use_tma_o=True),
        replace(split_two, pack_gqa=True),
    ):
        assert compile_bucket(split_two, 128) != compile_bucket(distinct, 128)


def test_measure_rejects_a_config_with_wrong_output(monkeypatch):
    wrong = replace(_BASE_CONFIG, q_stage=2)

    def corrupt_candidate(*, config=None, out=None):
        if config is None:
            return (torch.zeros(1, 2, 1, 64),)
        return (out.fill_(1.0),)

    monkeypatch.setattr("flash_attn.cute.autotune._flash_attn_fwd", corrupt_candidate)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with pytest.raises(RuntimeError, match="q_stage=2.*disagrees with the default output"):
        _measure_configs((wrong,), {})


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110 runtime autotuning test",
)
@pytest.mark.skipif(
    not CUTE_DSL_CACHE_ENABLED,
    reason="runtime autotuning requires FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1",
)
def test_tune_fwd_config_runtime():
    torch.manual_seed(0)
    q = torch.randn(1, 65, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 257, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    base = replace(_BASE_CONFIG, device_capacity=torch.cuda.get_device_capability()[0])
    for varlen in (False, True):
        kwargs = {"q": q, "k": k, "v": v}
        if varlen:
            kwargs = {
                "q": q.flatten(0, 1),
                "k": k.flatten(0, 1),
                "v": v.flatten(0, 1),
                "cu_seqlens_q": torch.tensor([0, 17, 65], device="cuda", dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0, 129, 257], device="cuda", dtype=torch.int32),
                "max_seqlen_q": 48,
                "max_seqlen_k": 129,
            }
        captured = []

        def record(inputs):
            captured.append(inputs)
            return select_fwd_config(inputs)

        with mock.patch.object(interface, "select_fwd_config", record):
            _flash_attn_fwd(**kwargs)
        # Fewer KV stages fit with q_stage=2 or fp32 SplitKV partials.
        configs = tuple(
            derive_fwd_config(base, captured[0], **changes)
            for changes in (
                {"use_tma_o": not varlen},
                {"use_tma_o": not varlen, "q_stage": 2},
                {"num_splits": 2},
            )
        )
        assert [config.kv_stage for config in configs] == [12, 10, 11]
        assert tune_fwd_config(configs, kwargs, compile_workers=2) in configs
