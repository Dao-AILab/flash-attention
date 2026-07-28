# Copyright (c) 2026, Dao-AILab.

"""Explicit runtime-input autotuning for FA4 forward.

Given exact ``FwdConfig`` values and real ``_flash_attn_fwd`` arguments, compile
in isolated parallel subprocesses, capture CUDA graphs on the caller's tensors,
measure candidates sequentially, and return the winner.

This first version is eager, BF16/FP16, and output-only. It does not alter
``config=None``, persist shape policy, or tune Flex, modifier, sparse, LSE, or
autograd calls.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from .cache_utils import CUTE_DSL_CACHE_ENABLED
from .config import FwdConfig, fwd_config_compile_bucket
from .interface import _flash_attn_fwd

_COMPILE_REQUEST = "FA4_FWD_AUTOTUNE_COMPILE_REQUEST"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUNDS = 5
_REPLAYS = 31
_WARMUP = 5
_CLOCK_WARMUP = 20
_COMPILE_TIMEOUT_SECONDS = 15 * 60


def _normalize_kwargs(forward_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the initial output-only scope and pin the real device arch."""
    kwargs = dict(forward_kwargs)
    if "config" in kwargs:
        raise ValueError("pass candidate configs separately, not in forward_kwargs")
    for name in ("out", "lse", "out_partial", "lse_partial"):
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
            # seqused_q leaves unused output rows undefined, so full-output checks are invalid.
            "seqused_q",
            # Absorbed MLA currently has one fixed forward config.
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
    return kwargs


def _decode_kwargs(encoded: Mapping[str, Any]) -> dict[str, Any]:
    """Recreate fake tensor metadata inside a worker's FakeTensorMode."""
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
    visible_device: str,
    device_arch: int,
) -> None:
    """Compile one full forward request in an isolated fake-tensor process."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_device
    env["CUTE_DSL_ARCH"] = {
        90: "sm_90a",
        100: "sm_100a",
        103: "sm_103a",
        110: "sm_110a",
        120: "sm_120a",
    }.get(device_arch, f"sm_{device_arch}")
    env["FLASH_ATTENTION_ARCH"] = str(device_arch)
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    env[_COMPILE_REQUEST] = json.dumps({"config": asdict(config), "forward_kwargs": encoded_kwargs})
    code = """
import os
import sys
import types
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
        text=True,
        env=env,
        cwd=_REPO_ROOT,
        timeout=_COMPILE_TIMEOUT_SECONDS,
    )


def _compile_worker() -> None:
    """Subprocess entry point: create and invoke tensors in one FakeTensorMode."""
    request = json.loads(os.environ[_COMPILE_REQUEST])
    with FakeTensorMode():
        _flash_attn_fwd(
            **_decode_kwargs(request["forward_kwargs"]),
            config=FwdConfig(**request["config"]),
        )


def _compile_barrier(
    configs: Sequence[FwdConfig],
    forward_kwargs: Mapping[str, Any],
    workers: int,
) -> None:
    """Compile deduplicated requests in parallel before any measurement."""
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
    try:
        json.dumps(encoded)
    except TypeError as error:
        raise TypeError("forward arguments must be tensors or JSON-serializable scalars") from error

    v = forward_kwargs["v"]
    device_index = 0 if v.device.index is None else v.device.index
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [item.strip() for item in visible.split(",") if item.strip()]
        if visible is not None
        else None
    )
    if devices is not None and device_index >= len(devices):
        raise ValueError(f"CUDA device {device_index} is outside CUDA_VISIBLE_DEVICES={visible!r}")
    visible_device = str(device_index) if devices is None else devices[device_index]

    requests = {}
    for config in configs:
        requests.setdefault(fwd_config_compile_bucket(config, v.shape[-1]), config)
    failure = None
    with ThreadPoolExecutor(max_workers=min(workers, len(requests))) as pool:
        futures = {
            pool.submit(
                _compile_one,
                config,
                encoded,
                visible_device,
                forward_kwargs["_arch"],
            ): config
            for config in requests.values()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except (subprocess.SubprocessError, OSError) as error:
                failure = futures[future], error
                for pending in futures:
                    pending.cancel()
                break
    if failure is not None:
        config, error = failure
        detail = getattr(error, "stderr", None) or getattr(error, "stdout", None)
        detail = str(detail).strip() if detail else str(error)
        raise RuntimeError(f"parallel compile failed for {config}:\n{detail}") from error


def _measure_configs(configs: Sequence[FwdConfig], forward_kwargs: Mapping[str, Any]) -> FwdConfig:
    """Capture exact graph paths, time them cyclically, and return the winner."""
    graphs = []
    q, v = (forward_kwargs[name] for name in ("q", "v"))
    shared_output = torch.empty(*q.shape[:-1], v.shape[-1], dtype=q.dtype, device=q.device)
    max_splits = max(config.num_splits for config in configs)
    out_partial = lse_partial = None
    if max_splits > 1:
        num_heads = q.shape[-2]
        lse_shape = (
            (num_heads, q.shape[0])
            if forward_kwargs.get("cu_seqlens_q") is not None
            else (q.shape[0], num_heads, q.shape[1])
        )
        out_partial = torch.empty(
            max_splits,
            *shared_output.shape,
            dtype=torch.float32,
            device=q.device,
        )
        lse_partial = torch.empty(
            max_splits,
            *lse_shape,
            dtype=torch.float32,
            device=q.device,
        )

    for config in configs:
        call_kwargs = {**forward_kwargs, "out": shared_output}
        if config.num_splits > 1:
            call_kwargs.update(
                out_partial=out_partial[: config.num_splits],
                lse_partial=lse_partial[: config.num_splits],
            )
        for _ in range(_WARMUP):
            _flash_attn_fwd(**call_kwargs, config=config)
        torch.cuda.synchronize()

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
            for _ in range(_CLOCK_WARMUP):
                torch.mm(clock_operand, clock_operand, out=clock_output)
            torch.cuda.synchronize(device)
            for _ in range(_REPLAYS):
                start.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                elapsed[index].append(start.elapsed_time(end) * 1e3)
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
