"""Real-GPU regressions for SingleTileLPTScheduler's CLC L2 swizzle.

Run with the FA4 editable install from this checkout:

    FLASH_ATTENTION_FAKE_TENSOR=0 python -m pytest -q -s \
        tests/cute/test_lpt_clc_swizzle.py

The fixtures enable CLC independently of the ambient FA_CLC setting. Without
a filter, a process runs 32 cases: 24 MHA swizzle cases and 8 edge cases.
To run this same test file against a separate baseline editable install, set
FA_EXPECTED_ROOT to that checkout's root; it defaults to this file's checkout.
Unavailable/unsupported GPUs and FakeTensorMode are skipped: these tests need
real SM100/SM110 execution. An unexpected import or missing constructor
evidence fails. Skipped cases do not count as GPU correctness validation.

These are forward O/LSE checks with verified scheduler selection. They do not
prove that a hardware CLC steal occurred, and do not exercise 2CTA or backward.
The swizzle printed in evidence is a prediction from the current 50 MiB source
heuristic and input shape, not a measurement of device scheduler parameters.

Run only the six bf16, split1 section cases for a smaller first check:

    FLASH_ATTENTION_FAKE_TENSOR=0 python -m pytest -q -s -x \
        tests/cute/test_lpt_clc_swizzle.py::test_swizzle_sections \
        -k 'bf16 and split1'

The suite shares a private in-memory forward cache. Every distinct compile key
must first capture its constructor; later cases reuse that kernel and its
recorded scheduler evidence. Disk hits cannot bypass this verification. Keep
cases in one pytest process to benefit from reuse.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from flash_attn.cute import interface, utils as cute_utils
from flash_attn.cute.cache_utils import JITCache
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.testing import attention_ref, is_fake_mode
from flash_attn.cute.tile_scheduler import SchedulingMode, SingleTileLPTScheduler


class SchedulerEvidenceCache(JITCache):
    """Bind each compiled kernel to its constructor evidence, using its real key."""

    def __init__(self):
        super().__init__()
        self.constructors = {}
        self.active_evidence = None

    def __setitem__(self, key, kernel):
        evidence = self.active_evidence
        assert evidence is not None, "A compile occurred outside a tracked test case."
        assert len(evidence["ctors"]) == 1, (
            f"A new compile key must capture one constructor: {evidence['ctors']}"
        )
        self.constructors[key] = dict(evidence["ctors"][0])
        super().__setitem__(key, kernel)

    def __getitem__(self, key):
        kernel = super().__getitem__(key)
        evidence = self.active_evidence
        assert evidence is not None, "A kernel lookup occurred outside a tracked test case."
        assert key in self.constructors, "Cached kernel has no constructor evidence."
        evidence["kernel_lookups"].append(key)
        evidence["selected_ctor"] = self.constructors[key]
        evidence["cache_hit"] = len(evidence["ctors"]) == 0
        return kernel

    def clear(self):
        super().clear()
        self.constructors.clear()
        self.active_evidence = None


@pytest.fixture(scope="module")
def compiled_forward_cache():
    cache = SchedulerEvidenceCache()
    yield cache
    cache.clear()


@pytest.fixture(scope="module", autouse=True)
def require_real_gpu():
    if os.environ.get("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1" or is_fake_mode():
        pytest.skip("CLC O/LSE checks require real GPU execution, not FakeTensorMode.")
    if not torch.cuda.is_available():
        pytest.skip("CLC swizzle tests require a real SM100/SM110 CUDA GPU.")
    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 11):
        pytest.skip(f"CLC swizzle tests require SM10x/SM11x, got {major}.{minor}.")
    expected_root = Path(
        os.environ.get("FA_EXPECTED_ROOT", str(Path(__file__).resolve().parents[2]))
    ).expanduser().resolve()
    expected = expected_root / "flash_attn/cute/interface.py"
    assert Path(interface.__file__).resolve() == expected, (
        f"Imported {interface.__file__}; expected {expected}. "
        "Install the selected checkout's flash_attn/cute editable."
    )


@pytest.fixture
def scheduler_evidence(monkeypatch, request, compiled_forward_cache):
    """Verify new compile keys and reuse their evidence on subsequent cases."""
    monkeypatch.setenv("FA_CLC", "1")
    monkeypatch.setattr(cute_utils, "_fa_clc_enabled", True)
    # Use a private in-memory cache, never unverified disk kernels. Keep this
    # cache across cases so changing only runtime shapes can reuse compilation.
    cache = compiled_forward_cache
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", cache)
    evidence = {
        "case": request.node.name, "FA_CLC": 1, "ctors": [], "configs": [],
        "kernel_lookups": [], "selected_ctor": None, "cache_hit": None,
    }
    cache.active_evidence = evidence
    original_init = FlashAttentionForwardSm100.__init__
    original_config = interface._get_fwd_config

    def capture_init(kernel, *args, **kwargs):
        original_init(kernel, *args, **kwargs)
        evidence["ctors"].append({
            "class": kernel.TileScheduler,
            "mode": kernel.scheduling_mode,
            "cluster": tuple(kernel.cluster_shape_mn),
            "use_2cta": kernel.use_2cta_instrs,
            "is_split_kv": kernel.is_split_kv,
            "pack_gqa": kernel.pack_gqa,
            "q_stage": kernel.q_stage,
            "m_block_size": kernel.m_block_size,
            "n_block_size": kernel.n_block_size,
        })

    def capture_config(*args, **kwargs):
        config = original_config(*args, **kwargs)
        evidence["configs"].append({"num_splits": config.num_splits})
        return config

    monkeypatch.setattr(FlashAttentionForwardSm100, "__init__", capture_init)
    monkeypatch.setattr(interface, "_get_fwd_config", capture_config)
    # FP32 reference GEMMs must not silently use TF32.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(42)
    try:
        yield evidence
    finally:
        cache.active_evidence = None


def check_scheduler(evidence, num_splits, pack_gqa):
    assert len(evidence["kernel_lookups"]) == 1, (
        f"Expected one actual forward kernel lookup; got {len(evidence['kernel_lookups'])}"
    )
    assert evidence["cache_hit"] is not None and evidence["selected_ctor"] is not None, (
        "The executed kernel must have constructor evidence for its compile key."
    )
    expected_constructors = 0 if evidence["cache_hit"] else 1
    assert len(evidence["ctors"]) == expected_constructors, (
        f"Unexpected constructor count for cache_hit={evidence['cache_hit']}: "
        f"{evidence['ctors']}"
    )
    assert len(evidence["configs"]) == 1, (
        f"Expected an actual forward config; captured {evidence['configs']}"
    )
    ctor = evidence["selected_ctor"]
    assert ctor["class"] is SingleTileLPTScheduler, ctor
    assert ctor["mode"] == SchedulingMode.CLC, ctor
    assert ctor["cluster"] == (1, 1), ctor
    assert not ctor["use_2cta"], ctor
    assert ctor["is_split_kv"] == (num_splits > 1), ctor
    assert ctor["pack_gqa"] == pack_gqa, ctor
    assert evidence["configs"][0]["num_splits"] == num_splits, evidence


@torch.no_grad()
def check_case(
    evidence, *, batch, sq, sk, q_heads, kv_heads, dtype,
    causal=True, window_size=(None, None), num_splits=1, pack_gqa=False,
):
    assert not is_fake_mode(), "Actual CUDA execution is required for every case."
    q = torch.randn(batch, sq, q_heads, 128, device="cuda", dtype=dtype)
    k = torch.randn(batch, sk, kv_heads, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    out, lse = interface.flash_attn_func(
        q, k, v, causal=causal, window_size=window_size,
        num_splits=num_splits, pack_gqa=pack_gqa, return_lse=True,
    )
    torch.cuda.synchronize()
    check_scheduler(evidence, num_splits, pack_gqa)
    assert lse is not None
    assert out.shape == q.shape
    assert lse.shape == (batch, q_heads, sq)
    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all().item(), "O contains NaN/Inf."

    # attention_ref casts its LSE back to the input dtype. Supplying FP32
    # inputs preserves a meaningful FP32 LSE reference for the GPU's FP32 LSE.
    out_ref_fp32, attn_ref, lse_ref = attention_ref(
        q.float(), k.float(), v.float(), causal=causal,
        window_size=window_size, return_lse=True,
    )
    del attn_ref
    out_ref = out_ref_fp32.to(dtype)
    out_pt, attn_pt = attention_ref(
        q, k, v, causal=causal, window_size=window_size,
        upcast=False, reorder_ops=True,
    )
    del attn_pt
    # Same output error bound as tests/cute/test_clc_fuzz.py: twice the
    # low-precision PyTorch reference error plus its rounding allowance.
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    out_error = (out - out_ref).abs().max().item()
    reference_error = (out_pt - out_ref).abs().max().item()
    assert out_error <= 2 * reference_error + fwd_atol, (
        f"O error={out_error}, reference_error={reference_error}, atol={fwd_atol}"
    )

    assert not torch.isnan(lse).any().item(), "LSE contains NaN."
    assert torch.equal(torch.isneginf(lse), torch.isneginf(lse_ref)), "LSE -inf rows differ."
    assert torch.equal(torch.isposinf(lse), torch.isposinf(lse_ref)), "LSE +inf rows differ."
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-4, rtol=1e-5)
    masked_rows = torch.isneginf(lse_ref).transpose(1, 2)
    if sq > sk and causal:
        assert masked_rows.any().item(), "The fully masked row case must contain empty rows."
    if masked_rows.any().item():
        assert torch.count_nonzero(out[masked_rows]).item() == 0, "Fully masked O rows must be zero."

    ctor = evidence["selected_ctor"]
    # GQA packing changes the number of heads seen by the scheduler.
    scheduled_heads = kv_heads if pack_gqa else q_heads
    head_bytes = sk * (128 + 128) * q.element_size()
    heads_fit = (50 * 1024 * 1024) // head_bytes
    predicted_swizzle = 1 << (max(heads_fit, 1).bit_length() - 1)
    report = {
        "case": evidence["case"], "FA_CLC": evidence["FA_CLC"],
        "forward_cache_hit": evidence["cache_hit"],
        "dtype": str(dtype), "q_shape": list(q.shape), "k_shape": list(k.shape),
        "scheduler": ctor["class"].__name__, "mode": ctor["mode"].name,
        "cluster": ctor["cluster"], "use_2cta": ctor["use_2cta"],
        "num_splits": evidence["configs"][0]["num_splits"],
        "pack_gqa": ctor["pack_gqa"], "q_stage": ctor["q_stage"],
        "m_block_size": ctor["m_block_size"], "n_block_size": ctor["n_block_size"],
        "predicted_swizzle": predicted_swizzle,
        "predicted_full_sections": batch * scheduled_heads // predicted_swizzle,
        "predicted_residual_heads": batch * scheduled_heads % predicted_swizzle,
        "out_max_error": out_error, "out_reference_error": reference_error,
        "out_atol": fwd_atol,
        "lse_max_error": (lse[finite] - lse_ref[finite]).abs().max().item(),
        "masked_rows": masked_rows.sum().item(),
    }
    print("LPT_EVIDENCE " + json.dumps(report, sort_keys=True))
    return report


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("mask", ["causal", "local"])
@pytest.mark.parametrize(
    "batch,heads,full,residual",
    [(2, 4, 1, 0), (3, 5, 1, 7), (1, 3, 0, 3)],
    ids=["full", "full-and-residual", "residual-only"],
)
@pytest.mark.parametrize("num_splits", [1, 3], ids=["split1", "split3"])
def test_swizzle_sections(scheduler_evidence, dtype, mask, batch, heads, full, residual, num_splits):
    report = check_case(
        scheduler_evidence, batch=batch, sq=513, sk=8192, q_heads=heads,
        kv_heads=heads, dtype=dtype, causal=mask == "causal",
        window_size=(None, None) if mask == "causal" else (1023, 0),
        num_splits=num_splits,
    )
    assert report["predicted_swizzle"] == 8
    assert report["predicted_full_sections"] == full
    assert report["predicted_residual_heads"] == residual


@pytest.mark.parametrize("sq", [1, 2])
@pytest.mark.parametrize("pack_gqa", [False, True], ids=["unpacked", "packed"])
def test_autoregressive_gqa(scheduler_evidence, sq, pack_gqa):
    check_case(
        scheduler_evidence, batch=3, sq=sq, sk=8193, q_heads=10, kv_heads=5,
        dtype=torch.bfloat16, num_splits=1, pack_gqa=pack_gqa,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("num_splits", [1, 3], ids=["split1", "split3"])
def test_fully_masked_rows(scheduler_evidence, dtype, num_splits):
    check_case(
        scheduler_evidence, batch=1, sq=513, sk=129, q_heads=3, kv_heads=3,
        dtype=dtype, num_splits=num_splits,
    )
