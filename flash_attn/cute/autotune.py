# Copyright (c) 2026, Dao-AILab.

"""Explicit FA4 forward autotuning for eager, output-only BF16/FP16 calls.

Compile FwdConfig candidates in isolated parallel workers, then time CUDA graphs
on the caller's tensors sequentially. Supports output-only calls without Flex,
modifiers, sparsity, LSE, or autograd.
"""

from __future__ import annotations

import os
import pickle
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from .cache_utils import CUTE_DSL_CACHE_ENABLED
from .config import FwdConfig
from .cute_dsl_utils import maybe_contiguous
from .interface import _flash_attn_fwd, combine_kernel_geometry, fwd_codegen_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUNDS = 5
_REPLAYS = 31
_WARMUP = 5
_CLOCK_WARMUP = 20
_COMPILE_TIMEOUT_SECONDS = 15 * 60


def _normalize_kwargs(forward_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate output-only inputs and pin the device architecture."""
    kwargs = dict(forward_kwargs)
    if "config" in kwargs:
        raise ValueError("pass candidate configs separately, not in forward_kwargs")
    for name in ("out", "lse"):
        if kwargs.get(name) is not None:
            raise NotImplementedError(f"output-only autotuning does not accept {name}")
    unsupported = [
        name
        for name in (
            "score_mod",
            "mask_mod",
            "block_sparse_tensors",
            "aux_tensors",
            "aux_scalars",
            "gather_kv_indices",
            "learnable_sink",
            # seqused_q leaves unused rows undefined; full-output checks would fail.
            "seqused_q",
            # Absorbed MLA has a fixed forward config.
            "qv",
        )
        if kwargs.get(name) is not None
    ]
    if kwargs.get("softcap") not in (None, 0.0):
        unsupported.append("softcap")
    if kwargs.get("return_lse", False):
        unsupported.append("return_lse")
    if unsupported:
        raise NotImplementedError(
            "output-only autotuning does not support " + ", ".join(unsupported)
        )

    tensors = [value for value in kwargs.values() if isinstance(value, torch.Tensor)]
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("output-only autotuning does not support autograd")
    if any(isinstance(tensor, FakeTensor) for tensor in tensors):
        raise RuntimeError("autotuning requires real CUDA tensors")
    q, k, v = (kwargs.get(name) for name in ("q", "k", "v"))
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
        raise TypeError("forward_kwargs must contain Q, K, and V tensors")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all tensor arguments must be CUDA tensors")
    if any(tensor.device != v.device for tensor in tensors):
        raise ValueError("all tensor arguments must be on the same CUDA device")
    if v.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("output-only autotuning supports BF16 and FP16 only")

    major, minor = torch.cuda.get_device_capability(v.device)
    device_arch = major * 10 + minor
    if kwargs.get("_arch") not in (None, device_arch):
        raise ValueError(
            f"requested architecture {kwargs['_arch']} does not match "
            f"{v.device} architecture {device_arch}"
        )
    kwargs["_arch"] = device_arch
    # Worker compiles must match the canonical layouts used by the interface.
    for name, align_bytes in (("q", 16), ("k", 16), ("v", 16), ("page_table", 4)):
        if kwargs.get(name) is not None:
            kwargs[name] = maybe_contiguous(kwargs[name], align_bytes=align_bytes)
    return kwargs


def _decode_kwargs(encoded: Mapping[str, Any]) -> dict[str, Any]:
    """Recreate tensor metadata inside the worker's FakeTensorMode."""
    return {
        name: (
            torch.empty_strided(
                tuple(value["shape"]),
                tuple(value["stride"]),
                dtype=getattr(torch, value["dtype"]),
                device="cuda:0",
            )
            if isinstance(value, dict) and value.get("tensor")
            else value
        )
        for name, value in encoded.items()
    }


def _compile_one(
    config: FwdConfig,
    encoded_kwargs: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Compile a full forward call in an isolated fake-tensor process on device's GPU."""
    device_arch = encoded_kwargs["_arch"]
    env = os.environ.copy()
    # A UUID names the caller's GPU whatever form the caller's CUDA_VISIBLE_DEVICES takes.
    env["CUDA_VISIBLE_DEVICES"] = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    env["CUTE_DSL_ARCH"] = {
        90: "sm_90a",
        100: "sm_100a",
        103: "sm_103a",
        110: "sm_110a",
        120: "sm_120a",
    }.get(device_arch, f"sm_{device_arch}")
    env["FLASH_ATTENTION_ARCH"] = str(device_arch)
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    code = """
import os
import sys
import types
# Skip flash_attn/__init__.py, which imports the FA2 CUDA extension.
package = types.ModuleType("flash_attn")
package.__path__ = [os.path.join(os.getcwd(), "flash_attn")]
sys.modules["flash_attn"] = package
from flash_attn.cute.autotune import _compile_worker
_compile_worker()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        input=pickle.dumps({"config": config, "forward_kwargs": encoded_kwargs}),
        env=env,
        cwd=_REPO_ROOT,
        timeout=_COMPILE_TIMEOUT_SECONDS,
    )


def _compile_worker() -> None:
    """Create tensors and compile forward within one worker FakeTensorMode."""
    request = pickle.load(sys.stdin.buffer)
    with FakeTensorMode():
        _flash_attn_fwd(**_decode_kwargs(request["forward_kwargs"]), config=request["config"])


def _compile_barrier(
    configs: Sequence[FwdConfig],
    forward_kwargs: Mapping[str, Any],
    workers: int,
) -> None:
    """Finish parallel compilation before measurement."""
    encoded = {
        name: (
            {
                "tensor": True,
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in forward_kwargs.items()
    }
    v = forward_kwargs["v"]

    requests = {}
    for config in configs:
        requests.setdefault(compile_bucket(config, v.shape[-1]), config)
    with ThreadPoolExecutor(max_workers=min(workers, len(requests))) as pool:
        futures = {
            pool.submit(_compile_one, config, encoded, v.device): config
            for config in requests.values()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except (subprocess.SubprocessError, OSError) as error:
                pool.shutdown(cancel_futures=True)
                detail = getattr(error, "stderr", None) or getattr(error, "stdout", None)
                detail = detail.decode(errors="replace").strip() if detail else str(error)
                raise RuntimeError(
                    f"parallel compile failed for {futures[future]}:\n{detail}"
                ) from error


def compile_bucket(
    config: FwdConfig, head_dim_v: int
) -> tuple[FwdConfig, tuple[int, int, int] | None]:
    """Deduplicate exact split counts that share main and combine codegen.

    Main kernels key on fwd_codegen_config; the combine kernel keys on its geometry.
    """
    combine = (
        combine_kernel_geometry(head_dim_v, config.num_splits) if config.num_splits > 1 else None
    )
    return fwd_codegen_config(config), combine


def _measure_configs(configs: Sequence[FwdConfig], forward_kwargs: Mapping[str, Any]) -> FwdConfig:
    """Capture exact graph paths, time them cyclically, and return the winner."""
    graphs = []
    reference = _flash_attn_fwd(**forward_kwargs)[0]
    call_kwargs = {**forward_kwargs, "out": torch.empty_like(reference)}
    for config in configs:
        # Stale values from the previous config must not hide tiles this one skips.
        call_kwargs["out"].fill_(float("nan"))
        for _ in range(_WARMUP):
            _flash_attn_fwd(**call_kwargs, config=config)
        torch.cuda.synchronize()
        # A fast wrong answer must never win; SplitKV only reorders the reduction.
        try:
            torch.testing.assert_close(call_kwargs["out"], reference, atol=2e-2, rtol=2e-2)
        except AssertionError as error:
            raise RuntimeError(f"config {config} disagrees with the default output") from error

        # SplitKV partials are allocated from each graph's private memory pool.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _flash_attn_fwd(**call_kwargs, config=config)
        for _ in range(_WARMUP):
            graph.replay()
        torch.cuda.synchronize()
        graphs.append(graph)

    elapsed = [[] for _ in configs]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    device = forward_kwargs["v"].device
    generator = torch.Generator(device=device).manual_seed(0)
    clock_operand = torch.randn(
        4096, 4096, dtype=forward_kwargs["v"].dtype, device=device, generator=generator
    )
    clock_output = torch.empty_like(clock_operand)
    for round_index in range(_ROUNDS):
        for position in range(len(configs)):
            index = (round_index + position) % len(configs)
            # Settle GPU clocks before each arm.
            for _ in range(_CLOCK_WARMUP):
                torch.mm(clock_operand, clock_operand, out=clock_output)
            torch.cuda.synchronize(device)
            for _ in range(_REPLAYS):
                start.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                elapsed[index].append(start.elapsed_time(end))
    return configs[min(range(len(configs)), key=lambda index: statistics.median(elapsed[index]))]


def tune_fwd_config(
    configs: Sequence[FwdConfig],
    forward_kwargs: Mapping[str, Any],
    *,
    compile_workers: int = 8,
) -> FwdConfig:
    """Return the fastest exact config for one concrete eager forward call."""
    configs = tuple(dict.fromkeys(configs))
    if not configs:
        raise ValueError("at least one FwdConfig is required")
    if compile_workers < 1:
        raise ValueError("compile_workers must be positive")
    if torch.compiler.is_compiling():
        raise RuntimeError("autotuning must run outside torch.compile")
    if not CUTE_DSL_CACHE_ENABLED:
        raise RuntimeError("set FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 before importing FA4")

    kwargs = _normalize_kwargs(forward_kwargs)
    device = kwargs["v"].device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("autotuning cannot start inside CUDA graph capture")
        _compile_barrier(configs, kwargs, compile_workers)
        return _measure_configs(configs, kwargs)
