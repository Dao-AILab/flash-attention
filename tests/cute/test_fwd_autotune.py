import os
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

if "flash_attn" not in sys.modules:
    package = types.ModuleType("flash_attn")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "flash_attn")]
    sys.modules["flash_attn"] = package

from flash_attn.cute.autotune import _compile_barrier, _decode_kwargs, _normalize_kwargs
from flash_attn.cute.config import (
    FwdCombineKernelSpec,
    FwdConfig,
    FwdMainKernelSpec,
    FwdSm100RegisterAllocation,
    fwd_config_compile_bucket,
    select_fwd_config,
)
from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_fwd_combine

_BASE_CONFIG = FwdConfig(
    device_capacity=10,
    tile_m=128,
    tile_n=128,
    num_stages=0,
    num_threads=512,
    mma_pv_is_rs=False,
    intra_wg_overlap=False,
    q_stage=1,
    use_clc_scheduler=False,
    is_static_persistent=False,
    use_tma_o=False,
    num_splits=1,
    use_2cta_instrs=False,
    registers=FwdSm100RegisterAllocation(192, 80, 48),
)


def test_fake_compile_payload_preserves_tensor_metadata():
    encoded = {
        "q": {
            "tensor": True,
            "shape": [3, 5, 7],
            "stride": [35, 7, 1],
            "dtype": "bfloat16",
        },
        "causal": False,
    }

    with FakeTensorMode():
        decoded = _decode_kwargs(encoded)

    assert decoded["q"].shape == (3, 5, 7)
    assert decoded["q"].stride() == (35, 7, 1)
    assert decoded["q"].dtype == torch.bfloat16
    assert decoded["q"].device.type == "cuda"
    assert decoded["causal"] is False


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
        """Pretend every key is compiled while recording cache cardinality."""

        def __init__(self, expected_type):
            self.expected_type = expected_type
            self.keys = set()

        def __contains__(self, key):
            assert isinstance(key, self.expected_type)
            self.keys.add(key)
            return True

    main = SpecRecorder(FwdMainKernelSpec)
    combine = SpecRecorder(FwdCombineKernelSpec)
    original_main = _flash_attn_fwd.compile_cache
    original_combine = _flash_attn_fwd_combine.compile_cache
    _flash_attn_fwd.compile_cache = main
    _flash_attn_fwd_combine.compile_cache = combine
    select_fwd_config.cache_clear()
    try:
        with FakeTensorMode():
            # One more unique shape than the selector cache can retain.
            for index in range(1025):
                batch = 1 + index % 64
                seqlen_q = 1 + (index * 131) % 8191
                seqlen_k = 1 + (index * 8191) % 65521
                q = torch.empty(
                    batch, seqlen_q, 32, 128, device="cuda", dtype=torch.bfloat16
                )
                k = torch.empty(
                    batch, seqlen_k, 8, 128, device="cuda", dtype=torch.bfloat16
                )
                v = torch.empty_like(k)
                _flash_attn_fwd(q, k, v, num_splits=0, _arch=103)
        selector_info = select_fwd_config.cache_info()
    finally:
        _flash_attn_fwd.compile_cache = original_main
        _flash_attn_fwd_combine.compile_cache = original_combine
        select_fwd_config.cache_clear()

    assert len(main.keys) == 4
    assert len(combine.keys) == 1
    assert selector_info.currsize == 1024


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110 runtime autotuning test",
)
def test_tune_fwd_config_runtime(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = """
import os
import sys
import types
package = types.ModuleType("flash_attn")
package.__path__ = [os.path.join(os.getcwd(), "flash_attn")]
sys.modules["flash_attn"] = package
import torch
from dataclasses import replace
from flash_attn.cute.autotune import tune_fwd_config
from flash_attn.cute.config import FwdConfig, FwdSm100RegisterAllocation
q = torch.randn(1, 65, 8, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 257, 8, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
capacity = torch.cuda.get_device_capability()[0]
base = FwdConfig(
    device_capacity=capacity,
    tile_m=128,
    tile_n=128,
    num_stages=0,
    num_threads=512,
    mma_pv_is_rs=False,
    intra_wg_overlap=False,
    q_stage=1,
    use_clc_scheduler=False,
    is_static_persistent=False,
    use_tma_o=True,
    num_splits=1,
    use_2cta_instrs=False,
    registers=FwdSm100RegisterAllocation(200, 64, 48),
)
split = replace(base, use_tma_o=False, num_splits=2)
configs = (base, replace(base, q_stage=2), split)
winner = tune_fwd_config(configs, {"q": q, "k": k, "v": v}, compile_workers=2)
assert winner in configs

q = q.reshape(-1, 8, 64)
k = k.reshape(-1, 8, 64)
v = v.reshape(-1, 8, 64)
cu_q = torch.tensor([0, 17, 65], device="cuda", dtype=torch.int32)
cu_k = torch.tensor([0, 129, 257], device="cuda", dtype=torch.int32)
varlen = replace(base, use_tma_o=False, is_static_persistent=False)
configs = (varlen, replace(varlen, q_stage=2), replace(varlen, num_splits=2))
winner = tune_fwd_config(
    configs,
    {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": 48,
        "max_seqlen_k": 129,
    },
    compile_workers=2,
)
assert winner in configs
"""
    env = os.environ.copy()
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(tmp_path / "cache")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        check=True,
        timeout=300,
    )


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
    split_four = replace(_BASE_CONFIG, num_splits=4)
    alternate_registers = replace(
        _BASE_CONFIG, registers=FwdSm100RegisterAllocation(184, 80, 64)
    )

    assert fwd_config_compile_bucket(split_two, 128) == fwd_config_compile_bucket(
        split_four, 128
    )
    assert fwd_config_compile_bucket(_BASE_CONFIG, 128) != fwd_config_compile_bucket(
        split_two, 128
    )
    assert fwd_config_compile_bucket(_BASE_CONFIG, 128) != fwd_config_compile_bucket(
        alternate_registers, 128
    )
