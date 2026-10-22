"""Resume gates: CPU oracle/CLI checks and opt-in-on-SM100 real GPU checks.

CPU only (does not initialize CUDA)::
    CUDA_VISIBLE_DEVICES='' python -m pytest --noconftest tests/cute/test_swapab_resume.py -k cpu -q

GPU (run sequentially, after smoke gates; not a fake-tensor test)::
    python -m pytest --noconftest tests/cute/test_swapab_resume.py -k 'not cpu' -x -v

Limitations: these bounded cases do not establish performance or race freedom.
Persistent reuse covers dense BF16 Dqk96/Dv64, splits1/3, not paging or callbacks.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).with_name("swapab_cache_probe.py")


def reference(q, k, v, used):
    """Independent FP32 dense GQA oracle; empty rows have zero O/-inf LSE."""
    import torch

    ratio = q.shape[2] // k.shape[2]
    scores = torch.einsum(
        "bqhd,bkhd->bhqk", q.float(), k.repeat_interleave(ratio, 2).float()
    ) * (q.shape[-1] ** -0.5)
    positions = torch.arange(k.shape[1], device=q.device)
    scores.masked_fill_(
        positions[None, None, None, :] >= used[:, None, None, None], -torch.inf
    )
    lse = scores.logsumexp(-1)
    out = torch.einsum(
        "bhqk,bkhd->bqhd",
        scores.softmax(-1).nan_to_num(),
        v.repeat_interleave(ratio, 2).float(),
    )
    return out, lse


def test_cpu_reference_empty_and_truncated_rows():
    import torch

    q = torch.zeros(2, 1, 6, 64)
    k = torch.zeros(2, 3, 2, 64)
    v = torch.full((2, 3, 2, 96), 7.0)
    # An excluded token must not leak into the average.
    v[1, 2] = 1000
    out, lse = reference(q, k, v, torch.tensor([0, 2], dtype=torch.int32))
    assert torch.count_nonzero(out[0]) == 0
    assert torch.isneginf(lse[0]).all()
    torch.testing.assert_close(out[1], torch.full_like(out[1], 7.0))
    torch.testing.assert_close(lse[1], torch.full_like(lse[1], 2.0).log())


def test_cpu_probe_help():
    result = subprocess.run(
        [sys.executable, str(PROBE), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "populate" in result.stdout and "reuse" in result.stdout


def test_cpu_probe_requires_explicit_cache_directory():
    env = os.environ.copy()
    env.pop("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", None)
    # A CLI configuration error must fail before importing torch or initializing CUDA.
    result = subprocess.run(
        [sys.executable, str(PROBE), "--phase", "reuse"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR must be set" in result.stderr


@pytest.fixture
def sm100(monkeypatch):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("real SM100 required; not a fake-tensor compilation test")

    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "swapab")
    monkeypatch.setenv("FLASH_ATTENTION_FAKE_TENSOR", "0")
    return torch


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("dims", [(64, 96), (128, 64)])
@pytest.mark.parametrize("ns", [1, 3])
def test_dense_graph_changing_inputs_and_used_lengths(sm100, dtype_name, dims, ns):
    torch = sm100
    from flash_attn.cute.interface import _flash_attn_fwd

    torch.manual_seed(734)
    dq, dv = dims
    dtype = getattr(torch, dtype_name)
    q = torch.randn(2, 1, 6, dq, device="cuda", dtype=dtype)
    # Keep noncontiguous head strides, fixed shapes, and stable device pointers.
    k = torch.randn(2, 1025, 4, dq, device="cuda", dtype=dtype)[:, :, ::2]
    v = torch.randn(2, 1025, 4, dv, device="cuda", dtype=dtype)[:, :, ::2]
    used = torch.tensor([1025, 513], device="cuda", dtype=torch.int32)
    out = torch.empty(2, 1, 6, dv, device="cuda", dtype=dtype)
    lse = torch.empty(2, 6, 1, device="cuda", dtype=torch.float32)

    def run():
        return _flash_attn_fwd(
            q, k, v, seqused_k=used, num_splits=ns, out=out, lse=lse, return_lse=True
        )[:2]

    # Warm kernels/allocator on an explicit side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_out, captured_lse = run()
    assert captured_out.data_ptr() == out.data_ptr()
    assert captured_lse.data_ptr() == lse.data_ptr()
    assert any(
        key[0] == ("sm100_swapab", 1) for key in _flash_attn_fwd.compile_cache.cache
    )
    previous = None
    # Go nonempty -> empty -> nonempty to detect stale output/split scratch.
    for lengths in ([1025, 513], [0, 1], [257, 0], [1, 1025]):
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn_like(k))
        v.copy_(torch.randn_like(v))
        used.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        expected, expected_lse = reference(q, k, v, used)
        torch.testing.assert_close(out.float(), expected, atol=0.004, rtol=0.025)
        torch.testing.assert_close(lse, expected_lse, atol=0.004, rtol=0.003)
        assert not torch.isnan(out).any() and not torch.isnan(lse).any()
        if previous is not None:
            assert not torch.equal(out, previous), (
                "graph replay ignored changed device inputs"
            )
        previous = out.clone()
    torch.cuda.synchronize()


def test_fresh_process_persistent_cache(sm100, tmp_path):
    env = os.environ.copy()
    env.update(
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(tmp_path / "persistent"),
        FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="1",
        FLASH_ATTENTION_FAKE_TENSOR="0",
        FLASH_ATTENTION_SM100_BACKEND="swapab",
    )
    reports = []
    for phase in ("populate", "reuse"):
        result = subprocess.run(
            [sys.executable, str(PROBE), "--phase", phase],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, f"{phase}\n{result.stdout}\n{result.stderr}"
        records = [
            json.loads(line.removeprefix("CACHE_PROBE "))
            for line in result.stdout.splitlines()
            if line.startswith("CACHE_PROBE ")
        ]
        assert len(records) == 1, result.stdout
        reports.append(records[0])
    populated, reused = reports
    assert populated["pid"] != reused["pid"]
    assert populated["compiles"] > 0 and reused["compiles"] == 0
    assert populated["forward_keys"] == reused["forward_keys"]
    assert populated["object_files"] == reused["object_files"]
    assert populated["splits"] == reused["splits"] == [1, 3]
