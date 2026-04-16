#!/usr/bin/env python3
"""CLC benchmark for dense, varlen, and block-sparse FA4 sweeps.

Run with benchmark against the yaml sweep:
  python benchmarks/clc_bench.py --config benchmarks/configs/clc.yaml

Useful overrides:
  --workers 64                         # compile parallelism
  --case_filter q16_kv4                # run matching cases only
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import types
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Literal

try:
    from jsonargparse import CLI
except ImportError as exc:
    raise SystemExit(
        "Missing jsonargparse. Install it with "
        "uv pip install jsonargparse pyyaml"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results" / "clc"
CSV_FLOAT_DIGITS = 3
BLOCK_SIZE_Q = 256
BLOCK_SIZE_K = 128
INTERNAL_REQUEST_ENV = "CLC_BENCH_INTERNAL_REQUEST"

DTypeName = Literal["bfloat16", "float16"]


@dataclass(frozen=True)
class DenseSweep:
    enabled: bool = True
    batches: list[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    seqlen_pairs: list[list[int]] = field(
        default_factory=lambda: [[32, 8192], [1024, 1024], [2048, 2048], [4096, 4096], [8192, 8192], [16384, 16384]]
    )
    head_dims: list[int | list[int]] = field(default_factory=lambda: [64, 96, 128, [192, 128]])
    head_pairs: list[list[int]] = field(default_factory=lambda: [[16, 16], [16, 8], [16, 4], [16, 2], [16, 1]])
    causal: bool | list[bool] = True


@dataclass(frozen=True)
class VarlenSweep:
    enabled: bool = True
    max_q_tokens: list[int] = field(default_factory=lambda: [2048, 4096, 8192, 16384, 32768])
    max_kv_tokens: list[int] = field(default_factory=lambda: [2048, 4096, 8192, 16384, 32768])
    batches: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    patterns: list[str] = field(default_factory=lambda: ["uniform", "longtail"])
    head_dims: list[int | list[int]] = field(default_factory=lambda: [64, 96, 128, [192, 128]])
    head_pairs: list[list[int]] = field(default_factory=lambda: [[16, 8], [16, 4], [16, 2], [16, 1]])
    causal: bool | list[bool] = False


@dataclass(frozen=True)
class BlockSparseSweep:
    enabled: bool = False
    batches: list[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    seqlen_pairs: list[list[int]] = field(
        default_factory=lambda: [[1024, 1024], [2048, 2048], [4096, 4096], [4096, 8192]]
    )
    head_dims: list[int | list[int]] = field(default_factory=lambda: [64, 128, [192, 128]])
    head_pairs: list[list[int]] = field(default_factory=lambda: [[16, 16], [16, 4], [16, 1]])
    mask_names: list[str] = field(default_factory=lambda: ["block_diagonal"])
    sliding_window_sizes: list[int] = field(default_factory=lambda: [2048])


@dataclass(frozen=True)
class Case:
    name: str
    mode: Literal["dense", "varlen", "block_sparse"]
    q_heads: int
    kv_heads: int
    d: int
    dv: int
    causal: bool
    batch: int | None = None
    seqlen_q: int | None = None
    seqlen_k: int | None = None
    seqlens_q: list[int] | None = None
    seqlens_k: list[int] | None = None
    pattern: str = ""
    mask_name: str = ""
    window_size: int | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_out_dir() -> Path:
    return RESULTS_ROOT / utc_timestamp()


def head_pair_label(q_heads: int, kv_heads: int) -> str:
    return f"q{q_heads}_kv{kv_heads}"


def token_label(value: int) -> str:
    return f"{value // 1024}k" if value >= 1024 and value % 1024 == 0 else str(value)


def head_dim_label(d: int, dv: int) -> str:
    return f"h{d}" if d == dv else f"h{d}_dv{dv}"


def head_dim_pairs(head_dims: list[int | list[int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    invalid_pairs: list[int | list[int]] = []
    for dims in head_dims:
        if isinstance(dims, int):
            pairs.append((dims, dims))
            continue
        if len(dims) == 1:
            pairs.append((dims[0], dims[0]))
            continue
        if len(dims) == 2:
            pairs.append((dims[0], dims[1]))
            continue
        invalid_pairs.append(dims)
    if invalid_pairs:
        raise ValueError(f"Expected d or [d] or [d, dv], got {invalid_pairs}")
    return pairs


def dense_case_name(q_heads: int, kv_heads: int, causal: bool, d: int, dv: int, batch: int, seqlen_q: int, seqlen_k: int) -> str:
    causal_name = "causal" if causal else "noncausal"
    pair = head_pair_label(q_heads, kv_heads)
    dims = head_dim_label(d, dv)
    if seqlen_q == seqlen_k:
        return f"{pair}_{causal_name}_{dims}_{token_label(seqlen_q)}_b{batch}"
    return f"{pair}_{causal_name}_q{seqlen_q}_k{seqlen_k}_{dims}_b{batch}"


def varlen_case_name(
    pattern: str,
    q_heads: int,
    kv_heads: int,
    causal: bool,
    d: int,
    dv: int,
    batch: int,
    max_q_tokens: int,
    max_kv_tokens: int,
) -> str:
    causal_name = "causal" if causal else "noncausal"
    pair = head_pair_label(q_heads, kv_heads)
    dims = head_dim_label(d, dv)
    return (
        f"varlen_{pattern}_{pair}_{causal_name}_{dims}_"
        f"b{batch}_q{token_label(max_q_tokens)}_kv{token_label(max_kv_tokens)}"
    )


def normalize_lengths(weights: list[float], total_tokens: int) -> list[int]:
    if total_tokens < len(weights):
        raise ValueError(f"total_tokens={total_tokens} is smaller than batch={len(weights)}")
    scaled = [weight / sum(weights) * total_tokens for weight in weights]
    lengths = [max(1, int(math.floor(value))) for value in scaled]
    delta = total_tokens - sum(lengths)
    order = sorted(
        range(len(weights)),
        key=lambda idx: scaled[idx] - math.floor(scaled[idx]),
        reverse=delta > 0,
    )
    cursor = 0
    while delta != 0:
        idx = order[cursor % len(order)]
        if delta > 0:
            lengths[idx] += 1
            delta -= 1
        elif lengths[idx] > 1:
            lengths[idx] -= 1
            delta += 1
        cursor += 1
    return lengths


def pattern_weights(pattern: str, batch: int) -> list[float]:
    match pattern:
        case "uniform":
            return [1.0] * batch
        case "spiky":
            return [32.0] + [1.0] * (batch - 1)
        case "longtail":
            return [float(batch - idx) for idx in range(batch)]
        case "bimodal":
            split = max(1, batch // 2)
            return [8.0] * split + [1.0] * (batch - split)
        case "staircase":
            return [1.0 + idx for idx in range(batch)]
        case "loss_shape":
            base = [130, 1, 1, 1, 1674, 68, 157, 1, 1, 1, 1, 1, 1, 9, 1, 5]
            if batch == len(base):
                return [float(value) for value in base]
            return [float(base[idx % len(base)]) for idx in range(batch)]
        case _:
            raise ValueError(f"Unsupported varlen pattern: {pattern}")


def bool_values(value: bool | list[bool]) -> list[bool]:
    return [value] if isinstance(value, bool) else value


def generate_cases(
    dense: DenseSweep,
    varlen: VarlenSweep,
    block_sparse: BlockSparseSweep,
    case_filter: str = "",
) -> list[Case]:
    cases: list[Case] = []
    if dense.enabled:
        for batch, seqlen_pair, (d, dv), (q_heads, kv_heads), causal in product(
            dense.batches,
            dense.seqlen_pairs,
            head_dim_pairs(dense.head_dims),
            dense.head_pairs,
            bool_values(dense.causal),
        ):
            seqlen_q, seqlen_k = seqlen_pair
            cases.append(
                Case(
                    name=dense_case_name(q_heads, kv_heads, causal, d, dv, batch, seqlen_q, seqlen_k),
                    mode="dense",
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    d=d,
                    dv=dv,
                    causal=causal,
                    batch=batch,
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                )
            )
    if varlen.enabled:
        for max_q_tokens, max_kv_tokens, batch, pattern, (d, dv), (q_heads, kv_heads), causal in product(
            varlen.max_q_tokens,
            varlen.max_kv_tokens,
            varlen.batches,
            varlen.patterns,
            head_dim_pairs(varlen.head_dims),
            varlen.head_pairs,
            bool_values(varlen.causal),
        ):
            weights = pattern_weights(pattern, batch)
            lengths_q = normalize_lengths(weights, max_q_tokens)
            lengths_k = normalize_lengths(weights, max(batch, max_kv_tokens))
            cases.append(
                Case(
                    name=varlen_case_name(pattern, q_heads, kv_heads, causal, d, dv, batch, max_q_tokens, max_kv_tokens),
                    mode="varlen",
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    d=d,
                    dv=dv,
                    causal=causal,
                    batch=batch,
                    seqlens_q=lengths_q,
                    seqlens_k=lengths_k,
                    pattern=pattern,
                )
            )
    if block_sparse.enabled:
        for batch, seqlen_pair, (d, dv), (q_heads, kv_heads), mask_name in product(
            block_sparse.batches,
            block_sparse.seqlen_pairs,
            head_dim_pairs(block_sparse.head_dims),
            block_sparse.head_pairs,
            block_sparse.mask_names,
        ):
            seqlen_q, seqlen_k = seqlen_pair
            if seqlen_q > seqlen_k:
                continue
            window_sizes = block_sparse.sliding_window_sizes if mask_name == "sliding_window" else [None]
            for window_size in window_sizes:
                window_label = f"_w{window_size}" if window_size is not None else ""
                pair = head_pair_label(q_heads, kv_heads)
                dims = head_dim_label(d, dv)
                cases.append(
                    Case(
                        name=(
                            f"block_sparse_{mask_name}{window_label}_{pair}_"
                            f"{dims}_q{seqlen_q}_k{seqlen_k}_b{batch}"
                        ),
                        mode="block_sparse",
                        q_heads=q_heads,
                        kv_heads=kv_heads,
                        d=d,
                        dv=dv,
                        causal=False,
                        batch=batch,
                        seqlen_q=seqlen_q,
                        seqlen_k=seqlen_k,
                        mask_name=mask_name,
                        window_size=window_size,
                    )
                )
    if case_filter:
        needle = case_filter.lower()
        cases = [case for case in cases if needle in case.name.lower()]
    return cases


def compile_q_stage(case: Case) -> int:
    max_seqlen_q = max(case.seqlens_q) if case.seqlens_q is not None else case.seqlen_q
    qhead_per_kvhead = case.q_heads // case.kv_heads
    return 2 if max_seqlen_q is not None and max_seqlen_q * qhead_per_kvhead > 128 else 1



def compile_signature(case: Case) -> tuple:
    q_stage = compile_q_stage(case)
    if case.mode == "block_sparse":
        return (
            case.mode,
            case.q_heads,
            case.kv_heads,
            case.d,
            case.dv,
            case.mask_name,
            case.window_size,
            q_stage,
        )
    return case.mode, case.q_heads, case.kv_heads, case.d, case.dv, case.causal, q_stage


def select_compile_cases(cases: list[Case]) -> list[Case]:
    selected: dict[tuple, Case] = {}
    for case in cases:
        selected.setdefault(compile_signature(case), case)
    return list(selected.values())


def benchmark_cuda_samples_in_microseconds(func, *args, **kwargs) -> list[float]:
    num_iters = kwargs.pop("NUM_ITERS", 100)
    warmup_iters = kwargs.pop("MEMORY_WARMUP_ITERS", 25)
    is_vetted_benchmarking = kwargs.pop("IS_VETTED_BENCHMARKING", False)
    from torch._inductor.runtime.benchmarking import benchmarker

    return [
        float(sample_ms) * 1e3
        for sample_ms in benchmarker.benchmark_gpu(
            lambda: func(*args, **kwargs),
            benchmark_iters=num_iters,
            memory_warmup_iters=warmup_iters,
            return_mode="all",
            is_vetted_benchmarking=is_vetted_benchmarking,
        )
    ]


def flash_attn_imports():
    if "flash_attn" not in sys.modules:
        stub = types.ModuleType("flash_attn")
        stub.__path__ = [str(REPO_ROOT / "flash_attn")]
        sys.modules["flash_attn"] = stub
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from flash_attn.cute import utils as cute_utils
    from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

    return torch, FakeTensorMode, cute_utils, flash_attn_func, flash_attn_varlen_func


def block_sparse_imports():
    if "flash_attn" not in sys.modules:
        stub = types.ModuleType("flash_attn")
        stub.__path__ = [str(REPO_ROOT / "flash_attn")]
        sys.modules["flash_attn"] = stub
    if str(REPO_ROOT / "tests" / "cute") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "tests" / "cute"))
    from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
    from mask_mod_definitions import get_mask_pair

    return compute_block_sparsity, get_mask_pair


def build_cu_seqlens(torch_mod, lengths: list[int]) -> torch_mod.Tensor:
    cu_seqlens = torch_mod.zeros(len(lengths) + 1, device="cuda", dtype=torch_mod.int32)
    cu_seqlens[1:] = torch_mod.tensor(lengths, device="cuda", dtype=torch_mod.int32).cumsum(0)
    return cu_seqlens


def build_dense_inputs(torch_mod, flash_attn_func, case: Case, dtype, factory):
    q = factory(case.batch, case.seqlen_q, case.q_heads, case.d, device="cuda", dtype=dtype)
    k = factory(case.batch, case.seqlen_k, case.kv_heads, case.d, device="cuda", dtype=dtype)
    v = factory(case.batch, case.seqlen_k, case.kv_heads, case.dv, device="cuda", dtype=dtype)
    return flash_attn_func, dict(q=q, k=k, v=v, causal=case.causal)


def build_varlen_inputs(torch_mod, flash_attn_varlen_func, case: Case, dtype, factory):
    lengths_q = case.seqlens_q or []
    lengths_k = case.seqlens_k or lengths_q
    total_q = sum(lengths_q)
    total_k = sum(lengths_k)
    q = factory(total_q, case.q_heads, case.d, device="cuda", dtype=dtype)
    k = factory(total_k, case.kv_heads, case.d, device="cuda", dtype=dtype)
    v = factory(total_k, case.kv_heads, case.dv, device="cuda", dtype=dtype)
    return flash_attn_varlen_func, dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=build_cu_seqlens(torch_mod, lengths_q),
        cu_seqlens_k=build_cu_seqlens(torch_mod, lengths_k),
        max_seqlen_q=max(lengths_q),
        max_seqlen_k=max(lengths_k),
        causal=case.causal,
    )


def build_block_sparse_compile_tensors(torch_mod, case: Case):
    num_m_blocks = math.ceil((case.seqlen_q or 0) / BLOCK_SIZE_Q)
    count_shape = (case.batch, 1, num_m_blocks)
    index_shape = (*count_shape, 1)
    return dict(
        mask_block_cnt=torch_mod.zeros(count_shape, device="cuda", dtype=torch_mod.int32),
        mask_block_idx=torch_mod.zeros(index_shape, device="cuda", dtype=torch_mod.int32),
        full_block_cnt=torch_mod.zeros(count_shape, device="cuda", dtype=torch_mod.int32),
        full_block_idx=torch_mod.zeros(index_shape, device="cuda", dtype=torch_mod.int32),
    )



def build_block_sparse_inputs(torch_mod, flash_attn_func, case: Case, dtype, tensor_factory, fake_tensor: bool):
    compute_block_sparsity, get_mask_pair = block_sparse_imports()
    if case.mask_name in {"document", "ima"}:
        raise ValueError(f"Aux-backed block-sparse masks are not supported by clc_bench.py: {case.mask_name}")
    q = tensor_factory(case.batch, case.seqlen_q, case.q_heads, case.d, device="cuda", dtype=dtype)
    k = tensor_factory(case.batch, case.seqlen_k, case.kv_heads, case.d, device="cuda", dtype=dtype)
    v = tensor_factory(case.batch, case.seqlen_k, case.kv_heads, case.dv, device="cuda", dtype=dtype)
    cute_mask, _ = get_mask_pair(
        case.mask_name,
        seqlen_q=case.seqlen_q,
        seqlen_k=case.seqlen_k,
        window_size=case.window_size,
    )
    if fake_tensor:
        block_sparse_tensors = build_block_sparse_compile_tensors(torch_mod, case)
    else:
        _, sparse_tensors = compute_block_sparsity(
            tile_m=BLOCK_SIZE_Q,
            tile_n=BLOCK_SIZE_K,
            batch_size=case.batch,
            num_heads=1,
            seqlen_q=case.seqlen_q,
            seqlen_k=case.seqlen_k,
            mask_mod=cute_mask,
            aux_tensors=None,
            device="cuda",
            compute_full_blocks=True,
            use_fast_sampling=False,
        )
        block_sparse_tensors = dict(
            mask_block_cnt=sparse_tensors.mask_block_cnt,
            mask_block_idx=sparse_tensors.mask_block_idx,
            full_block_cnt=sparse_tensors.full_block_cnt,
            full_block_idx=sparse_tensors.full_block_idx,
        )
    return flash_attn_func, dict(
        q=q,
        k=k,
        v=v,
        causal=False,
        mask_mod=cute_mask,
        **block_sparse_tensors,
        block_size=(BLOCK_SIZE_Q, BLOCK_SIZE_K),
    )


def build_inputs(case: Case, dtype_name: DTypeName, fake_tensor: bool):
    torch, FakeTensorMode, _, flash_attn_func, flash_attn_varlen_func = flash_attn_imports()
    dtype = getattr(torch, dtype_name)
    tensor_factory = torch.empty if fake_tensor else torch.randn
    context = FakeTensorMode() if fake_tensor else nullcontext()
    with context:
        match case.mode:
            case "block_sparse":
                return build_block_sparse_inputs(torch, flash_attn_func, case, dtype, tensor_factory, fake_tensor)
            case "dense":
                return build_dense_inputs(torch, flash_attn_func, case, dtype, tensor_factory)
            case "varlen":
                return build_varlen_inputs(torch, flash_attn_varlen_func, case, dtype, tensor_factory)


def attended_pairs(seqlen_q: int, seqlen_k: int, causal: bool) -> float:
    """Lower-right aligned causal: last query aligns with last key.
    When M > N, only the bottom N query rows attend (triangle of size N),
    so valid pairs = N*(N+1)/2, not the upper-left formula M*N - N*(N-1)/2.
    """
    if not causal:
        return float(seqlen_q * seqlen_k)
    if seqlen_q <= seqlen_k:
        return float(seqlen_q * (2 * seqlen_k - seqlen_q + 1) / 2)
    return float(seqlen_k * (seqlen_k + 1) / 2)


def block_sparse_pairs(case: Case) -> float:
    seqlen_q = case.seqlen_q or 0
    seqlen_k = case.seqlen_k or 0
    match case.mask_name:
        case "block_diagonal":
            total = 0
            for q_idx in range(seqlen_q):
                block_start = (q_idx // BLOCK_SIZE_K) * BLOCK_SIZE_K
                block_end = min(block_start + BLOCK_SIZE_K, seqlen_k)
                total += max(0, block_end - block_start)
            return float(total)
        case "sliding_window":
            window = case.window_size or 0
            offset = seqlen_k - seqlen_q
            total = 0
            for q_idx in range(seqlen_q):
                center = q_idx + offset
                lower = max(0, center - window)
                upper = min(seqlen_k - 1, center + window)
                total += max(0, upper - lower + 1)
            return float(total)
        case _:
            raise ValueError(f"Unsupported block-sparse FLOP mask: {case.mask_name}")


def fwd_flops(case: Case, kwargs: dict | None = None) -> float:
    if case.mode == "dense":
        return (case.batch or 0) * case.q_heads * 2 * attended_pairs(
            case.seqlen_q or 0,
            case.seqlen_k or 0,
            case.causal,
        ) * (case.d + case.dv)
    if case.mode == "block_sparse":
        num_pairs = (case.batch or 0) * block_sparse_pairs(case)
        return case.q_heads * 2 * num_pairs * (case.d + case.dv)
    lengths_q = case.seqlens_q or []
    lengths_k = case.seqlens_k or lengths_q
    total = 0.0
    for seqlen_q, seqlen_k in zip(lengths_q, lengths_k):
        total += case.q_heads * 2 * attended_pairs(seqlen_q, seqlen_k, case.causal) * (case.d + case.dv)
    return total


def tflops(flop_count: float, time_us: float) -> float:
    return 0.0 if time_us <= 0 else flop_count / time_us / 1e6


def case_shape(case: Case) -> str:
    match case.mode:
        case "dense" | "block_sparse":
            if case.seqlen_q == case.seqlen_k:
                return token_label(case.seqlen_q or 0)
            return f"q={token_label(case.seqlen_q or 0)} kv={token_label(case.seqlen_k or 0)}"
        case "varlen":
            lengths_q = case.seqlens_q or []
            lengths_k = case.seqlens_k or lengths_q
            total_q = sum(lengths_q)
            total_k = sum(lengths_k)
            max_q = max(lengths_q, default=0)
            max_k = max(lengths_k, default=0)
            if total_q == total_k and max_q == max_k:
                return f"total={token_label(total_q)} max={token_label(max_q)}"
            return (
                f"q_total={token_label(total_q)} kv_total={token_label(total_k)} "
                f"q_max={token_label(max_q)} kv_max={token_label(max_k)}"
            )


def case_metadata(case: Case) -> dict:
    return {
        "name": case.name,
        "mode": case.mode,
        "shape": case_shape(case),
        "batch": case.batch,
        "q_heads": case.q_heads,
        "kv_heads": case.kv_heads,
        "d": case.d,
        "dv": case.dv,
        "causal": case.causal,
        "pattern": case.pattern,
        "mask_name": case.mask_name,
        "window_size": case.window_size,
    }


def summarize_profile(case: Case, samples_off: list[float], samples_on: list[float], flop_count: float) -> dict:
    mean_off = statistics.mean(samples_off)
    mean_on = statistics.mean(samples_on)
    paired_log_ratios = [math.log(off / on) for off, on in zip(samples_off, samples_on)]
    mean_log_ratio = statistics.mean(paired_log_ratios)
    stderr_log_ratio = (
        statistics.stdev(paired_log_ratios) / math.sqrt(len(paired_log_ratios))
        if len(paired_log_ratios) > 1
        else 0.0
    )
    ci95_low = math.exp(mean_log_ratio - 1.96 * stderr_log_ratio)
    ci95_high = math.exp(mean_log_ratio + 1.96 * stderr_log_ratio)
    return {
        **case_metadata(case),
        "samples_off_us": samples_off,
        "samples_on_us": samples_on,
        "mean_off_us": mean_off,
        "mean_on_us": mean_on,
        "median_off_us": statistics.median(samples_off),
        "median_on_us": statistics.median(samples_on),
        "mean_off_tflops": tflops(flop_count, mean_off),
        "mean_on_tflops": tflops(flop_count, mean_on),
        "speedup_on_vs_off": mean_off / mean_on,
        "pct_change_on_vs_off": (mean_off / mean_on - 1.0) * 100.0,
        "ci95_low_speedup": ci95_low,
        "ci95_high_speedup": ci95_high,
        "ci95_excludes_1x": ci95_low > 1.0 or ci95_high < 1.0,
    }


def run_single_case(
    case: Case,
    clc: int,
    fake_tensor: bool,
    dtype_name: DTypeName,
    bench_iters: int,
    seed: int,
) -> dict:
    os.environ["FA_CLC"] = str(clc)
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    torch, _, cute_utils, _, _ = flash_attn_imports()
    if not fake_tensor and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runtime profiling in clc_bench.py")
    torch.manual_seed(seed)
    fn, kwargs = build_inputs(case, dtype_name, fake_tensor)
    cute_utils._fa_clc_enabled = bool(clc)
    fn(**kwargs)
    if fake_tensor:
        return {"case": case.name, "clc": clc, "compiled": True}
    torch.cuda.synchronize()
    warmup_iters = min(25, max(10, bench_iters // 10))
    return {
        "case": case.name,
        "clc": clc,
        "time_us": statistics.mean(
            benchmark_cuda_samples_in_microseconds(
                fn,
                **kwargs,
                NUM_ITERS=bench_iters,
                MEMORY_WARMUP_ITERS=warmup_iters,
            )
        ),
    }


def run_single_subprocess(case: Case, clc: int, dtype_name: DTypeName, bench_iters: int, seed: int, script_path: Path) -> dict:
    env = os.environ.copy()
    env[INTERNAL_REQUEST_ENV] = json.dumps(
        {
            "case": asdict(case),
            "clc": clc,
            "fake_tensor": True,
            "dtype_str": dtype_name,
            "bench_iters": bench_iters,
            "seed": seed,
        }
    )
    command = [sys.executable, str(script_path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise RuntimeError(f"Single-case compile failed for {case.name} clc={clc}:\n{detail}") from exc
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON result found for {case.name} clc={clc}")


def run_compile(cases: list[Case], dtype_name: DTypeName, workers: int, bench_iters: int, seed: int, script_path: Path) -> list[dict]:
    compile_cases = select_compile_cases(cases)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_single_subprocess, case, clc, dtype_name, bench_iters, seed, script_path): (case.name, clc)
            for case in compile_cases
            for clc in (0, 1)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            print(f"[{index}/{len(futures)}] compiled {row['case']} clc={row['clc']}")
            rows.append(row)
    return sorted(rows, key=lambda row: (row["case"], row["clc"]))


def run_profile(cases: list[Case], dtype_name: DTypeName, profile_repeats: int, bench_iters: int, seed: int) -> list[dict]:
    torch, _, cute_utils, _, _ = flash_attn_imports()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for clc_bench.py")
    torch.manual_seed(seed)
    total_iters = profile_repeats * bench_iters
    warmup_iters = min(25, max(10, total_iters // 10))
    rows: list[dict] = []
    for index, case in enumerate(cases, start=1):
        fn, kwargs = build_inputs(case, dtype_name, fake_tensor=False)
        samples: dict[int, list[float]] = {}
        for clc in (0, 1):
            cute_utils._fa_clc_enabled = bool(clc)
            fn(**kwargs)
            torch.cuda.synchronize()
            samples[clc] = benchmark_cuda_samples_in_microseconds(
                fn,
                **kwargs,
                NUM_ITERS=total_iters,
                MEMORY_WARMUP_ITERS=warmup_iters,
            )
        row = summarize_profile(case, samples[0], samples[1], fwd_flops(case, kwargs))
        print(
            f"[{index}/{len(cases)}] {case.name}: "
            f"off={row['mean_off_us']:.3f}us on={row['mean_on_us']:.3f}us "
            f"speedup={row['speedup_on_vs_off']:.3f}x "
            f"ci95=[{row['ci95_low_speedup']:.3f}, {row['ci95_high_speedup']:.3f}]"
        )
        rows.append(row)
    return rows


def round_scalar_row(row: dict) -> dict:
    return {
        key: round(value, CSV_FLOAT_DIGITS) if isinstance(value, float) else value
        for key, value in row.items()
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_rows = [
        round_scalar_row({key: value for key, value in row.items() if not isinstance(value, list)})
        for row in rows
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_rows)


def main(
    out_dir: Path | None = None,
    workers: int = 32,
    profile_repeats: int = 3,
    bench_iters: int = 64,
    dtype_str: DTypeName = "bfloat16",
    seed: int = 0,
    case_filter: str = "",
    dense: DenseSweep = DenseSweep(),
    varlen: VarlenSweep = VarlenSweep(),
    block_sparse: BlockSparseSweep = BlockSparseSweep(),
) -> None:
    if (request_json := os.environ.get(INTERNAL_REQUEST_ENV)) is not None:
        request = json.loads(request_json)
        print(
            json.dumps(
                run_single_case(
                    Case(**request["case"]),
                    request["clc"],
                    request["fake_tensor"],
                    request["dtype_str"],
                    request["bench_iters"],
                    request["seed"],
                )
            )
        )
        return

    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    cases = generate_cases(dense, varlen, block_sparse, case_filter)
    if not cases:
        raise ValueError("No cases selected. Adjust the YAML sweep or case_filter.")

    run_dir = out_dir or default_out_dir()
    print(f"cases={len(cases)}")
    print(f"compile_cases={len(select_compile_cases(cases))}")
    print(f"out_dir={run_dir}")
    print(f"python={sys.executable}")

    script_path = Path(__file__).resolve()
    run_compile(cases, dtype_str, workers, bench_iters, seed, script_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_rows = run_profile(cases, dtype_str, profile_repeats, bench_iters, seed)
    profile_csv = run_dir / "profile.csv"
    write_csv(profile_csv, profile_rows)
    print("Profile written to:")
    print(profile_csv)


if __name__ == "__main__":
    CLI(main, as_positional=False)
