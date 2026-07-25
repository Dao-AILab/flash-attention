#!/usr/bin/env python3
"""Benchmark SM100/SM110 forward selector and epilogue hypotheses.

Run the frozen sweep with:
  python benchmarks/fwd_heuristics_bench.py \
    --config benchmarks/configs/fwd_heuristics.yaml

The frozen shape families target q-stage, CLC, persistence, output-epilogue,
and CTA regions. Every case compiles, validates, and measures its complete
production bucket candidate set.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from functools import cache
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

if TYPE_CHECKING:
    from flash_attn.cute.config import FwdConfig, FwdHeuristicInputs

try:
    from jsonargparse import CLI
except ImportError as exc:
    raise SystemExit(
        "Missing jsonargparse. Install it with uv pip install jsonargparse pyyaml"
    ) from exc

from clc_bench import (
    Case,
    build_inputs,
    case_metadata,
    dense_case_name,
    normalize_lengths,
    pattern_weights,
    varlen_case_name,
)
from realistic_workloads import generate_realistic_workloads

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results" / "fwd_heuristics"
INTERNAL_REQUEST_ENV = "FWD_HEURISTICS_INTERNAL_REQUEST"
ExperimentName = Literal[
    "q_stage_hdim192",
    "clc_varlen_mha",
    "nonpersistent_hdim64",
    "direct_epilogue_hdim64",
    "one_cta_hdim128",
    "realistic",
]
DTypeName = Literal["bfloat16", "float16"]


@cache
def flash_attn_modules():
    """Load the installed FA4 modules after enabling its persistent compile cache."""
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    from flash_attn.cute import config as config_module
    from flash_attn.cute import interface as interface_module
    from flash_attn.cute import utils as utils_module

    return config_module, interface_module, utils_module


@dataclass(frozen=True)
class VarlenPolicySweep:
    enabled: bool = True
    discovery_token_pairs: list[list[int]] = field(
        default_factory=lambda: [
            [512, 512],
            [2048, 2048],
            [8192, 8192],
            [32768, 32768],
            [4096, 16384],
            [16384, 4096],
        ]
    )
    holdout_token_pairs: list[list[int]] = field(
        default_factory=lambda: [[3072, 6144], [12288, 6144], [24576, 24576]]
    )
    discovery_batches: list[int] = field(default_factory=lambda: [4, 16])
    holdout_batches: list[int] = field(default_factory=lambda: [8, 24])
    discovery_patterns: list[str] = field(
        default_factory=lambda: ["uniform", "longtail", "spiky"]
    )
    holdout_patterns: list[str] = field(
        default_factory=lambda: ["bimodal", "staircase"]
    )
    q_heads: list[int] = field(default_factory=lambda: [8, 16])
    head_dims: list[int | list[int]] = field(default_factory=lambda: [[192, 128]])


@dataclass(frozen=True)
class DensePolicySweep:
    enabled: bool = True
    discovery_seqlen_pairs: list[list[int]] = field(
        default_factory=lambda: [
            [128, 128],
            [256, 256],
            [1024, 1024],
            [4096, 4096],
            [16384, 16384],
            [32, 8192],
            [128, 8192],
            [2048, 8192],
            [8192, 2048],
        ]
    )
    holdout_seqlen_pairs: list[list[int]] = field(
        default_factory=lambda: [[384, 640], [1536, 3072], [6144, 12288], [12288, 6144]]
    )
    discovery_batches: list[int] = field(default_factory=lambda: [1, 4, 16, 32])
    holdout_batches: list[int] = field(default_factory=lambda: [2, 8, 24])
    head_pairs: list[list[int]] = field(
        default_factory=lambda: [[16, 16], [16, 4], [16, 1]]
    )


@dataclass(frozen=True)
class DenseArchitectureSweep:
    enabled: bool = True
    discovery_seqlen_pairs: list[list[int]] = field(
        default_factory=lambda: [
            [512, 512],
            [2048, 2048],
            [8192, 8192],
            [16384, 16384],
            [32768, 32768],
            [4096, 16384],
            [16384, 4096],
        ]
    )
    holdout_seqlen_pairs: list[list[int]] = field(
        default_factory=lambda: [
            [768, 1536],
            [6144, 12288],
            [12288, 6144],
            [24576, 24576],
        ]
    )
    discovery_batches: list[int] = field(default_factory=lambda: [1, 2, 4])
    holdout_batches: list[int] = field(default_factory=lambda: [1, 3])
    q_heads: list[int] = field(default_factory=lambda: [8, 16])


@dataclass(frozen=True)
class BenchCase:
    experiment: ExperimentName
    phase: Literal["discovery", "boundary", "holdout"]
    case: Case


@dataclass(frozen=True)
class Variant:
    name: str
    bucket_name: str
    config: FwdConfig


@dataclass(frozen=True)
class CudaBenchmarkStats:
    """CUDA timing samples and robust summary quantiles."""

    samples_us: tuple[float, ...]
    median_us: float
    p05_us: float
    p95_us: float

    @staticmethod
    def quantile(samples: Sequence[float], q: float) -> float:
        """Linearly interpolate one quantile from non-empty samples."""
        if not samples:
            raise ValueError("samples must not be empty")
        ordered = sorted(float(sample) for sample in samples)
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    @classmethod
    def from_samples(cls, samples_us: Sequence[float]) -> CudaBenchmarkStats:
        """Summarize raw per-replay timings without discarding samples."""
        samples = tuple(float(sample) for sample in samples_us)
        return cls(
            samples_us=samples,
            median_us=cls.quantile(samples, 0.5),
            p05_us=cls.quantile(samples, 0.05),
            p95_us=cls.quantile(samples, 0.95),
        )


def benchmark_cuda_function_stats(
    func: Callable[[], object],
    *,
    num_iters: int,
    warmup_iters: int,
    use_cuda_graphs: bool,
) -> CudaBenchmarkStats:
    """Collect per-call CUDA-event timings under a fixed-pointer warm-cache contract."""
    for _ in range(warmup_iters):
        func()
    torch.cuda.synchronize()

    timed_call = func
    if use_cuda_graphs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            func()
        torch.cuda.synchronize()
        timed_call = graph.replay
        for _ in range(warmup_iters):
            timed_call()
        torch.cuda.synchronize()

    samples_us = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_call()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3)
    return CudaBenchmarkStats.from_samples(samples_us)


DEFAULT_Q_STAGE_SWEEP = VarlenPolicySweep()
DEFAULT_CLC_SWEEP = VarlenPolicySweep(head_dims=[64, 96, 128])
DEFAULT_NONPERSISTENT_SWEEP = DensePolicySweep()
DEFAULT_DIRECT_EPILOGUE_SWEEP = DenseArchitectureSweep(
    discovery_seqlen_pairs=[
        [4096, 4096],
        [8192, 8192],
        [16384, 16384],
        [32768, 32768],
        [4096, 16384],
        [16384, 4096],
    ],
    holdout_seqlen_pairs=[[6144, 12288], [12288, 6144], [24576, 24576]],
)
DEFAULT_ONE_CTA_SWEEP = DenseArchitectureSweep()


def utc_timestamp() -> str:
    """Return a sortable UTC timestamp for result directories."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def head_dim_pairs(values: list[int | list[int]]) -> list[tuple[int, int]]:
    """Normalize scalar and pair head-dimension configuration values."""
    pairs = []
    for value in values:
        if isinstance(value, int):
            pairs.append((value, value))
        elif len(value) == 1:
            pairs.append((value[0], value[0]))
        elif len(value) == 2:
            pairs.append((value[0], value[1]))
        else:
            raise ValueError(f"Expected d or [d, dv], got {value}")
    return pairs


def make_varlen_case(
    experiment: ExperimentName,
    phase: Literal["discovery", "holdout"],
    total_q: int,
    total_k: int,
    batch: int,
    pattern: str,
    q_heads: int,
    d: int,
    dv: int,
) -> BenchCase:
    """Build a deterministic noncausal varlen MHA benchmark case."""
    weights = pattern_weights(pattern, batch)
    lengths_q = normalize_lengths(weights, total_q)
    lengths_k = normalize_lengths(weights, total_k)
    return BenchCase(
        experiment=experiment,
        phase=phase,
        case=Case(
            name=varlen_case_name(
                pattern, q_heads, q_heads, False, d, dv, batch, total_q, total_k
            ),
            mode="varlen",
            q_heads=q_heads,
            kv_heads=q_heads,
            d=d,
            dv=dv,
            causal=False,
            batch=batch,
            seqlens_q=lengths_q,
            seqlens_k=lengths_k,
            pattern=pattern,
        ),
    )


def make_dense_case(
    experiment: ExperimentName,
    phase: Literal["discovery", "holdout"],
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
) -> BenchCase:
    """Build a deterministic noncausal dense benchmark case."""
    return BenchCase(
        experiment=experiment,
        phase=phase,
        case=Case(
            name=dense_case_name(
                q_heads,
                kv_heads,
                False,
                head_dim,
                head_dim,
                batch,
                seqlen_q,
                seqlen_k,
            ),
            mode="dense",
            q_heads=q_heads,
            kv_heads=kv_heads,
            d=head_dim,
            dv=head_dim,
            causal=False,
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
        ),
    )


def generate_varlen_cases(
    experiment: ExperimentName,
    sweep: VarlenPolicySweep,
) -> list[BenchCase]:
    """Expand a frozen varlen discovery and holdout matrix."""
    if not sweep.enabled:
        return []
    cases = []
    phases = (
        (
            "discovery",
            sweep.discovery_token_pairs,
            sweep.discovery_batches,
            sweep.discovery_patterns,
        ),
        (
            "holdout",
            sweep.holdout_token_pairs,
            sweep.holdout_batches,
            sweep.holdout_patterns,
        ),
    )
    for phase, token_pairs, batches, patterns in phases:
        for token_pair, batch, pattern, q_heads, (d, dv) in product(
            token_pairs,
            batches,
            patterns,
            sweep.q_heads,
            head_dim_pairs(sweep.head_dims),
        ):
            cases.append(
                make_varlen_case(
                    experiment,
                    phase,
                    token_pair[0],
                    token_pair[1],
                    batch,
                    pattern,
                    q_heads,
                    d,
                    dv,
                )
            )
    return cases


def generate_dense_cases(sweep: DensePolicySweep) -> list[BenchCase]:
    """Expand the frozen dense hdim-64 discovery and holdout matrix."""
    if not sweep.enabled:
        return []
    cases = []
    phases = (
        ("discovery", sweep.discovery_seqlen_pairs, sweep.discovery_batches),
        ("holdout", sweep.holdout_seqlen_pairs, sweep.holdout_batches),
    )
    for phase, seqlen_pairs, batches in phases:
        for seqlen_pair, batch, head_pair in product(
            seqlen_pairs, batches, sweep.head_pairs
        ):
            q_heads, kv_heads = head_pair
            cases.append(
                make_dense_case(
                    "nonpersistent_hdim64",
                    phase,
                    q_heads,
                    kv_heads,
                    64,
                    batch,
                    seqlen_pair[0],
                    seqlen_pair[1],
                )
            )
    return cases


def generate_dense_architecture_cases(
    experiment: ExperimentName,
    sweep: DenseArchitectureSweep,
    head_dim: int,
) -> list[BenchCase]:
    """Expand dense MHA cases for architecture and epilogue comparisons."""
    if not sweep.enabled:
        return []
    cases = []
    phases = (
        ("discovery", sweep.discovery_seqlen_pairs, sweep.discovery_batches),
        ("holdout", sweep.holdout_seqlen_pairs, sweep.holdout_batches),
    )
    for phase, seqlen_pairs, batches in phases:
        for seqlen_pair, batch, q_heads in product(
            seqlen_pairs, batches, sweep.q_heads
        ):
            cases.append(
                make_dense_case(
                    experiment,
                    phase,
                    q_heads,
                    q_heads,
                    head_dim,
                    batch,
                    seqlen_pair[0],
                    seqlen_pair[1],
                )
            )
    return cases


def deduplicate_cases(cases: list[BenchCase]) -> list[BenchCase]:
    """Keep each phase and shape once when hypothesis sweeps overlap."""
    unique = {}
    for bench_case in cases:
        unique.setdefault((bench_case.phase, bench_case.case.name), bench_case)
    return list(unique.values())


def generate_cases(
    q_stage_hdim192: VarlenPolicySweep,
    clc_varlen_mha: VarlenPolicySweep,
    nonpersistent_hdim64: DensePolicySweep,
    direct_epilogue_hdim64: DenseArchitectureSweep,
    one_cta_hdim128: DenseArchitectureSweep,
    experiment_filter: str,
    case_filter: str,
    phase_filter: str,
    extra_cases: Sequence[BenchCase] = (),
) -> list[BenchCase]:
    """Generate and filter all configured experiments."""
    cases = [
        *generate_varlen_cases("q_stage_hdim192", q_stage_hdim192),
        *generate_varlen_cases("clc_varlen_mha", clc_varlen_mha),
        *generate_dense_cases(nonpersistent_hdim64),
        *generate_dense_architecture_cases(
            "direct_epilogue_hdim64", direct_epilogue_hdim64, 64
        ),
        *generate_dense_architecture_cases("one_cta_hdim128", one_cta_hdim128, 128),
        *extra_cases,
    ]
    if experiment_filter:
        cases = [
            bench_case
            for bench_case in cases
            if experiment_filter in bench_case.experiment
        ]
    if case_filter:
        cases = [
            bench_case
            for bench_case in cases
            if case_filter.lower() in bench_case.case.name.lower()
        ]
    if phase_filter:
        cases = [bench_case for bench_case in cases if bench_case.phase == phase_filter]
    return deduplicate_cases(cases)


def fwd_heuristic_inputs(case: Case, dtype_name: DTypeName) -> FwdHeuristicInputs:
    """Build production selector metadata from one benchmark case."""
    is_varlen = case.mode == "varlen"
    if is_varlen:
        lengths_q = case.seqlens_q or []
        lengths_k = case.seqlens_k or lengths_q
        batch_size = len(lengths_q)
        max_seqlen_q = max(lengths_q)
        max_seqlen_k = max(lengths_k)
    else:
        batch_size = case.batch or 0
        max_seqlen_q = case.seqlen_q or 0
        max_seqlen_k = case.seqlen_k or 0
    total_k = (
        batch_size * math.ceil(max_seqlen_k / case.page_size) * case.page_size
        if case.page_size is not None
        else sum(case.seqlens_k or case.seqlens_q or [])
        if is_varlen
        else batch_size * max_seqlen_k
    )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    config_module, interface_module, utils_module = flash_attn_modules()
    device_arch = interface_module._get_device_arch()
    causal, local, window_size_left, window_size_right = (
        interface_module._resolve_causal_local_window(
            case.causal,
            None,
            None,
            case.mask_mod_name or None,
        )
    )
    return config_module.FwdHeuristicInputs(
        device_capacity=device_arch // 10,
        device_arch=device_arch,
        num_sms=props.multi_processor_count,
        dtype=str(getattr(torch, dtype_name)),
        head_dim=case.d,
        head_dim_v=case.dv,
        num_heads=case.q_heads,
        num_heads_kv=case.kv_heads,
        batch_size=batch_size,
        total_q=(sum(case.seqlens_q or []) if is_varlen else batch_size * max_seqlen_q),
        total_k=total_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        local=local,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        is_varlen=is_varlen,
        is_varlen_q=is_varlen,
        has_cu_seqlens_q=is_varlen,
        has_cu_seqlens_k=is_varlen and case.page_size is None,
        has_seqused=case.page_size is not None,
        pack_gqa=case.q_heads != case.kv_heads,
        page_size=case.page_size,
        use_block_sparsity=False,
        sparse_q_block_size=None,
        has_qv=case.has_qv,
        has_gather_kv=case.gather_kv_length is not None,
        requested_tile_m=None,
        requested_tile_n=None,
        requested_num_threads=384,
        requested_mma_pv_is_rs=None,
        requested_intra_wg_overlap=None,
        requested_num_splits=None if case.num_splits < 1 else case.num_splits,
        requested_use_clc_scheduler=utils_module._get_use_clc_scheduler_default(),
        disable_2cta=utils_module._get_disable_2cta_default(is_fwd=True),
    )


def config_variant_name(default: FwdConfig, config: FwdConfig) -> str:
    """Name a bucket candidate by the fields it changes from the default."""
    if config == default:
        return "production_default"
    changes = [
        f"{item.name}_{getattr(config, item.name)}"
        for item in fields(config)
        if getattr(config, item.name) != getattr(default, item.name)
    ]
    return "__".join(changes)


def variants_for_case(
    bench_case: BenchCase, dtype_name: DTypeName
) -> tuple[Variant, ...]:
    """Resolve every candidate from a case's production bucket."""
    inputs = fwd_heuristic_inputs(bench_case.case, dtype_name)
    config_module, _, _ = flash_attn_modules()
    bucket = config_module.get_fwd_config_bucket(inputs)
    return tuple(
        Variant(
            name=config_variant_name(bucket.default, config),
            bucket_name=bucket.name,
            config=config,
        )
        for config in bucket.candidates
    )


def compile_signature(
    bench_case: BenchCase, variant: Variant, dtype_name: DTypeName
) -> tuple:
    """Project a candidate onto its main and combine codegen boundaries."""
    case = bench_case.case
    config = variant.config
    main_config = (
        *(
            getattr(config, item.name)
            for item in fields(config)
            if item.name != "num_splits"
        ),
        config.num_splits > 1,
    )
    if config.num_splits > 1:
        config_module, _, _ = flash_attn_modules()
        combine_tile_m, _ = config_module.fwd_combine_tile(case.dv)
        combine_bucket = config_module.combine_log_max_splits(
            config.num_splits, combine_tile_m
        )
    else:
        combine_bucket = None
    return (
        dtype_name,
        case.mode,
        case.q_heads // case.kv_heads,
        case.d,
        case.dv,
        case.causal,
        case.has_qv,
        case.page_size,
        case.score_mod_name,
        case.mask_mod_name,
        case.has_learnable_sink,
        case.gather_kv_length,
        main_config,
        combine_bucket,
    )


def select_compile_requests(
    cases: list[BenchCase], dtype_name: DTypeName
) -> list[tuple[BenchCase, Variant]]:
    """Choose one shape for every distinct candidate specialization."""
    selected = {}
    for bench_case in cases:
        for variant in variants_for_case(bench_case, dtype_name):
            selected.setdefault(
                compile_signature(bench_case, variant, dtype_name),
                (bench_case, variant),
            )
    return list(selected.values())


def compile_variant(
    bench_case: BenchCase, variant: Variant, dtype_name: DTypeName, seed: int
) -> dict:
    """Compile one specialization with fake tensors and the persistent disk cache."""
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    torch.manual_seed(seed)
    _, interface_module, _ = flash_attn_modules()
    with FakeTensorMode():
        _, kwargs = build_inputs(bench_case.case, dtype_name, fake_tensor=False)
        interface_module._flash_attn_fwd(**kwargs, config=variant.config)
    return {
        "experiment": bench_case.experiment,
        "case": bench_case.case.name,
        "bucket_name": variant.bucket_name,
        "variant": variant.name,
        "compiled": True,
    }


def compile_subprocess(
    bench_case: BenchCase,
    variant: Variant,
    dtype_name: DTypeName,
    seed: int,
    script_path: Path,
) -> dict:
    """Compile a specialization in an isolated subprocess."""
    env = os.environ.copy()
    env[INTERNAL_REQUEST_ENV] = json.dumps(
        {
            "bench_case": asdict(bench_case),
            "variant": asdict(variant),
            "dtype_name": dtype_name,
            "seed": seed,
        }
    )
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(
        f"No compile result for {bench_case.case.name} {variant.name}:\n{completed.stderr}"
    )


def run_compile(
    cases: list[BenchCase],
    dtype_name: DTypeName,
    workers: int,
    seed: int,
    script_path: Path,
) -> list[dict]:
    """Populate the CuTeDSL cache in parallel before GPU timing."""
    requests = select_compile_requests(cases, dtype_name)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                compile_subprocess, bench_case, variant, dtype_name, seed, script_path
            ): (
                bench_case,
                variant,
            )
            for bench_case, variant in requests
        }
        for index, future in enumerate(as_completed(futures), start=1):
            bench_case, variant = futures[future]
            try:
                row = future.result()
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or exc.stdout).strip()
                raise RuntimeError(
                    f"Compile failed for {bench_case.case.name} {variant.name}:\n{detail}"
                ) from exc
            print(
                f"[{index}/{len(futures)}] compiled {row['experiment']} {row['variant']}"
            )
            rows.append(row)
    return rows


def split_workspace_kwargs(
    case: Case, shared_kwargs: dict, config: FwdConfig
) -> dict[str, torch.Tensor]:
    """Preallocate reusable SplitKV partial output and LSE workspaces."""
    if config.num_splits == 1:
        return {}
    q = shared_kwargs["q"]
    output_shape = (*q.shape[:-1], case.dv)
    lse_shape = (
        (case.batch, case.q_heads, case.seqlen_q)
        if case.mode == "dense"
        else (case.q_heads, q.shape[0])
    )
    return {
        "out_partial": torch.empty(
            config.num_splits,
            *output_shape,
            dtype=torch.float32,
            device="cuda",
        ),
        "lse_partial": torch.empty(
            config.num_splits,
            *lse_shape,
            dtype=torch.float32,
            device="cuda",
        ),
    }


def make_variant_callables(
    bench_case: BenchCase,
    dtype_name: DTypeName,
    seed: int,
    variants: tuple[Variant, ...],
):
    """Create all variants over identical tensors with preallocated outputs."""
    torch.manual_seed(seed)
    _, interface_module, _ = flash_attn_modules()
    _, shared_kwargs = build_inputs(bench_case.case, dtype_name, fake_tensor=False)
    callables = {}
    outputs = {}
    for variant in variants:
        output = torch.empty(
            *shared_kwargs["q"].shape[:-1],
            bench_case.case.dv,
            device="cuda",
            dtype=getattr(torch, dtype_name),
        )
        kwargs = {
            **shared_kwargs,
            "out": output,
            **split_workspace_kwargs(bench_case.case, shared_kwargs, variant.config),
        }

        def invoke(kwargs=kwargs, config=variant.config):
            return interface_module._flash_attn_fwd(**kwargs, config=config)[0]

        callables[variant.name] = invoke
        outputs[variant.name] = output
    return shared_kwargs, callables, outputs


def make_clock_warmup(dtype_name: DTypeName, iterations: int):
    """Create an untimed GEMM warmup that raises idle GPU clocks before each arm."""
    if iterations == 0:
        return lambda: None
    operand = torch.randn(4096, 4096, device="cuda", dtype=getattr(torch, dtype_name))
    output = torch.empty_like(operand)

    def warmup():
        for _ in range(iterations):
            torch.mm(operand, operand, out=output)
        torch.cuda.synchronize()

    return warmup


def benchmark_case(
    bench_case: BenchCase,
    dtype_name: DTypeName,
    rounds: int,
    iters_per_round: int,
    warmup_iters: int,
    clock_warmup_iters: int,
    use_cuda_graphs: bool,
    seed: int,
) -> dict:
    """Time every bucket candidate with rotated sequential replay rounds."""
    if rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}")
    variants = variants_for_case(bench_case, dtype_name)
    _, callables, _ = make_variant_callables(bench_case, dtype_name, seed, variants)
    warm_clocks = make_clock_warmup(dtype_name, clock_warmup_iters)
    samples = {variant.name: [] for variant in variants}
    round_medians = {variant.name: [] for variant in variants}
    for round_idx in range(rounds):
        offset = round_idx % len(variants)
        order = variants[offset:] + variants[:offset]
        for variant in order:
            warm_clocks()
            stats = benchmark_cuda_function_stats(
                callables[variant.name],
                num_iters=iters_per_round,
                warmup_iters=warmup_iters,
                use_cuda_graphs=use_cuda_graphs,
            )
            samples[variant.name].extend(stats.samples_us)
            round_medians[variant.name].append(stats.median_us)
    variant_stats = {
        variant.name: CudaBenchmarkStats.from_samples(samples[variant.name])
        for variant in variants
    }
    production = next(
        variant for variant in variants if variant.name == "production_default"
    )
    winner = min(variants, key=lambda variant: variant_stats[variant.name].median_us)
    production_stats = variant_stats[production.name]
    winner_stats = variant_stats[winner.name]
    return {
        "experiment": bench_case.experiment,
        "phase": bench_case.phase,
        **case_metadata(bench_case.case),
        "dtype": dtype_name,
        "bucket_name": production.bucket_name,
        "production_config": asdict(production.config),
        "production_median_us": production_stats.median_us,
        "winner_config": asdict(winner.config),
        "winner_median_us": winner_stats.median_us,
        "production_to_winner_speedup": production_stats.median_us
        / winner_stats.median_us,
        "config_results": [
            {
                "name": variant.name,
                "config": asdict(variant.config),
                "samples_us": samples[variant.name],
                "round_medians_us": round_medians[variant.name],
                "median_us": variant_stats[variant.name].median_us,
                "p05_us": variant_stats[variant.name].p05_us,
                "p95_us": variant_stats[variant.name].p95_us,
            }
            for variant in variants
        ],
    }


def manual_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: torch.Tensor | None,
    *,
    causal: bool,
    scale: float,
    score_mod_name: str = "",
    mask_mod_name: str = "",
    learnable_sink: torch.Tensor | None = None,
    gather_kv_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention with campaign modifiers, QV scores, and sink logits."""
    if q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = q @ k.transpose(-1, -2)
    if qv is not None:
        scores += qv @ v.transpose(-1, -2)
    scores *= scale
    if score_mod_name == "times_two":
        scores *= 2
    elif score_mod_name:
        raise ValueError(f"Unsupported score modifier reference: {score_mod_name}")
    if causal:
        q_positions = torch.arange(q.shape[-2], device=q.device)[:, None]
        k_positions = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores.masked_fill_(
            k_positions > q_positions + k.shape[-2] - q.shape[-2], -torch.inf
        )
    if mask_mod_name == "causal_window_128":
        q_positions = torch.arange(q.shape[-2], device=q.device)[:, None]
        k_positions = torch.arange(k.shape[-2], device=q.device)[None, :]
        centers = q_positions + k.shape[-2] - q.shape[-2]
        scores.masked_fill_(
            (k_positions > centers) | (k_positions < centers - 128), -torch.inf
        )
    elif mask_mod_name:
        raise ValueError(f"Unsupported mask modifier reference: {mask_mod_name}")
    if gather_kv_indices is not None:
        if gather_kv_indices.dim() == 2:
            gather_kv_indices = gather_kv_indices.unsqueeze(0)
        gather_mask = torch.zeros(
            gather_kv_indices.shape[0],
            gather_kv_indices.shape[1],
            k.shape[-2],
            dtype=torch.bool,
            device=q.device,
        ).scatter_(-1, gather_kv_indices.long(), True)
        scores.masked_fill_(~gather_mask[:, None], -torch.inf)
    if learnable_sink is None:
        probabilities = torch.softmax(scores, dim=-1)
        return torch.nan_to_num(probabilities, nan=0.0) @ v
    sink = learnable_sink.float().view(1, -1, 1, 1)
    normalizer_max = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    weights = torch.exp(scores - normalizer_max)
    denominator = weights.sum(dim=-1, keepdim=True) + torch.exp(sink - normalizer_max)
    return (weights / denominator) @ v


def reference_output(case: Case, kwargs: dict) -> torch.Tensor:
    """Compute an independent float32 SDPA reference for dense or varlen inputs."""
    scale = 1.0 / math.sqrt(case.d + case.dv if case.has_qv else case.d)
    if case.mode == "dense":
        q = kwargs["q"].float().transpose(1, 2)
        k = kwargs["k"].float().transpose(1, 2)
        v = kwargs["v"].float().transpose(1, 2)
        if (
            case.has_qv
            or (case.causal and q.shape[-2] != k.shape[-2])
            or case.score_mod_name
            or case.mask_mod_name
            or case.has_learnable_sink
            or case.gather_kv_length is not None
        ):
            qv = kwargs["qv"].float().transpose(1, 2) if case.has_qv else None
            return manual_attention_reference(
                q,
                k,
                v,
                qv,
                causal=case.causal,
                scale=scale,
                score_mod_name=case.score_mod_name,
                mask_mod_name=case.mask_mod_name,
                learnable_sink=kwargs.get("learnable_sink"),
                gather_kv_indices=kwargs.get("gather_kv_indices"),
            ).transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=case.causal,
            scale=scale,
            enable_gqa=case.q_heads != case.kv_heads,
        ).transpose(1, 2)
    outputs = []
    q_start = 0
    k_start = 0
    for batch_index, (q_length, k_length) in enumerate(
        zip(case.seqlens_q or [], case.seqlens_k or [])
    ):
        q = (
            kwargs["q"][q_start : q_start + q_length]
            .float()
            .transpose(0, 1)
            .unsqueeze(0)
        )
        if case.page_size is None:
            k_slice = kwargs["k"][k_start : k_start + k_length]
            v_slice = kwargs["v"][k_start : k_start + k_length]
        else:
            max_pages = kwargs["page_table"].shape[1]
            page_start = batch_index * max_pages
            page_end = page_start + max_pages
            k_slice = kwargs["k"][page_start:page_end].flatten(0, 1)[:k_length]
            v_slice = kwargs["v"][page_start:page_end].flatten(0, 1)[:k_length]
        k = k_slice.float().transpose(0, 1).unsqueeze(0)
        v = v_slice.float().transpose(0, 1).unsqueeze(0)
        if (
            case.has_qv
            or (case.causal and q_length != k_length)
            or case.score_mod_name
            or case.mask_mod_name
            or case.has_learnable_sink
            or case.gather_kv_length is not None
        ):
            qv = (
                kwargs["qv"][q_start : q_start + q_length]
                .float()
                .transpose(0, 1)
                .unsqueeze(0)
                if case.has_qv
                else None
            )
            output = manual_attention_reference(
                q,
                k,
                v,
                qv,
                causal=case.causal,
                scale=scale,
                score_mod_name=case.score_mod_name,
                mask_mod_name=case.mask_mod_name,
                learnable_sink=kwargs.get("learnable_sink"),
                gather_kv_indices=(
                    kwargs["gather_kv_indices"][q_start : q_start + q_length]
                    if case.gather_kv_length is not None
                    else None
                ),
            )
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=case.causal,
                scale=scale,
                enable_gqa=case.q_heads != case.kv_heads,
            )
        outputs.append(output.squeeze(0).transpose(0, 1))
        q_start += q_length
        k_start += k_length
    return torch.cat(outputs)


def correctness_cases() -> list[BenchCase]:
    """Return small nonuniform cases that exercise every forced configuration."""
    return [
        BenchCase(
            "q_stage_hdim192",
            "holdout",
            Case(
                name="correctness_q_stage_hdim192",
                mode="varlen",
                q_heads=8,
                kv_heads=8,
                d=192,
                dv=128,
                causal=False,
                batch=3,
                seqlens_q=[127, 193, 61],
                seqlens_k=[131, 211, 73],
                pattern="manual",
            ),
        ),
        BenchCase(
            "clc_varlen_mha",
            "holdout",
            Case(
                name="correctness_clc_varlen_mha",
                mode="varlen",
                q_heads=8,
                kv_heads=8,
                d=128,
                dv=128,
                causal=False,
                batch=4,
                seqlens_q=[65, 129, 257, 33],
                seqlens_k=[71, 193, 263, 41],
                pattern="manual",
            ),
        ),
        BenchCase(
            "nonpersistent_hdim64",
            "holdout",
            Case(
                name="correctness_nonpersistent_hdim64",
                mode="dense",
                q_heads=16,
                kv_heads=4,
                d=64,
                dv=64,
                causal=False,
                batch=2,
                seqlen_q=384,
                seqlen_k=512,
            ),
        ),
        BenchCase(
            "direct_epilogue_hdim64",
            "holdout",
            Case(
                name="correctness_direct_epilogue_hdim64",
                mode="dense",
                q_heads=8,
                kv_heads=8,
                d=64,
                dv=64,
                causal=False,
                batch=2,
                seqlen_q=384,
                seqlen_k=512,
            ),
        ),
        BenchCase(
            "one_cta_hdim128",
            "holdout",
            Case(
                name="correctness_one_cta_hdim128",
                mode="dense",
                q_heads=8,
                kv_heads=8,
                d=128,
                dv=128,
                causal=False,
                batch=2,
                seqlen_q=384,
                seqlen_k=512,
            ),
        ),
        BenchCase(
            "realistic",
            "holdout",
            Case(
                name="correctness_score_mod",
                mode="dense",
                q_heads=8,
                kv_heads=2,
                d=64,
                dv=64,
                causal=True,
                batch=2,
                seqlen_q=65,
                seqlen_k=97,
                score_mod_name="times_two",
                scenario="correctness",
            ),
        ),
        BenchCase(
            "realistic",
            "holdout",
            Case(
                name="correctness_mask_mod",
                mode="dense",
                q_heads=8,
                kv_heads=2,
                d=64,
                dv=64,
                causal=False,
                batch=2,
                seqlen_q=65,
                seqlen_k=193,
                mask_mod_name="causal_window_128",
                scenario="correctness",
            ),
        ),
        BenchCase(
            "realistic",
            "holdout",
            Case(
                name="correctness_learnable_sink",
                mode="dense",
                q_heads=8,
                kv_heads=2,
                d=64,
                dv=64,
                causal=True,
                batch=2,
                seqlen_q=65,
                seqlen_k=97,
                has_learnable_sink=True,
                scenario="correctness",
            ),
        ),
        BenchCase(
            "realistic",
            "holdout",
            Case(
                name="correctness_paged_gqa",
                mode="varlen",
                q_heads=8,
                kv_heads=2,
                d=128,
                dv=128,
                causal=True,
                batch=2,
                seqlens_q=[3, 5],
                seqlens_k=[131, 97],
                pattern="manual",
                page_size=128,
                scenario="correctness",
            ),
        ),
        BenchCase(
            "realistic",
            "holdout",
            Case(
                name="correctness_mla_gather",
                mode="dense",
                q_heads=128,
                kv_heads=1,
                d=64,
                dv=512,
                causal=True,
                batch=2,
                seqlen_q=3,
                seqlen_k=256,
                has_qv=True,
                gather_kv_length=128,
                profile="deepseek_v3_absorbed_decode",
                scenario="correctness",
            ),
        ),
        BenchCase(
            "realistic",
            "holdout",
            Case(
                name="correctness_mla_paged",
                mode="varlen",
                q_heads=128,
                kv_heads=1,
                d=64,
                dv=512,
                causal=True,
                batch=2,
                seqlens_q=[1, 1],
                seqlens_k=[131, 97],
                pattern="manual",
                has_qv=True,
                page_size=64,
                profile="deepseek_v3_absorbed_decode",
                scenario="correctness",
            ),
        ),
        BenchCase(
            "realistic",
            "holdout",
            Case(
                name="correctness_mla_absorbed",
                mode="varlen",
                q_heads=128,
                kv_heads=1,
                d=64,
                dv=512,
                causal=True,
                batch=3,
                seqlens_q=[1, 1, 1],
                seqlens_k=[17, 31, 43],
                pattern="manual",
                has_qv=True,
                profile="deepseek_v3_absorbed_decode",
                scenario="correctness",
            ),
        ),
    ]


def run_correctness(
    dtype_name: DTypeName, seed: int, experiment_filter: str
) -> list[dict]:
    """Check both variants against float32 SDPA on representative cases."""
    atol, rtol = (0.04, 0.04) if dtype_name == "bfloat16" else (0.015, 0.015)
    rows = []
    for bench_case in correctness_cases():
        if experiment_filter and experiment_filter not in bench_case.experiment:
            continue
        variants = variants_for_case(bench_case, dtype_name)
        kwargs, callables, outputs = make_variant_callables(
            bench_case, dtype_name, seed, variants
        )
        reference = reference_output(bench_case.case, kwargs)
        for variant in variants:
            callables[variant.name]()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                outputs[variant.name].float(), reference, atol=atol, rtol=rtol
            )
        production_variant = next(
            variant for variant in variants if variant.name == "production_default"
        )
        production_output = torch.empty_like(outputs[production_variant.name])
        _, interface_module, _ = flash_attn_modules()
        interface_module._flash_attn_fwd(
            **kwargs,
            out=production_output,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            production_output.float(), reference, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            production_output.float(),
            outputs[production_variant.name].float(),
            atol=atol,
            rtol=rtol,
        )
        rows.append(
            {
                "experiment": bench_case.experiment,
                "case": bench_case.case.name,
                "bucket_name": production_variant.bucket_name,
                "production_variant": production_variant.name,
                "passed": True,
            }
        )
        print(f"correctness passed: {bench_case.experiment}")
    return rows


def git_output(*args: str) -> str:
    """Return one Git command's stdout without changing repository state."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def environment_metadata(
    dtype_name: DTypeName,
    seed: int,
    rounds: int,
    iters_per_round: int,
    warmup_iters: int,
    clock_warmup_iters: int,
    use_cuda_graphs: bool,
) -> dict:
    """Capture the source, software, hardware, and timing contract."""
    import cutlass

    from flash_attn.cute.cache_utils import _compute_source_fingerprint

    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git_output("rev-parse", "HEAD"),
        "repo_status": git_output("status", "--short"),
        "repo_diff_sha256": __import__("hashlib")
        .sha256(git_output("diff", "--no-ext-diff").encode())
        .hexdigest(),
        "fa_source_fingerprint": _compute_source_fingerprint(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass": getattr(cutlass, "__version__", "unknown"),
        "gpu_name": props.name,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "gpu_sms": props.multi_processor_count,
        "gpu_total_memory_bytes": props.total_memory,
        "clock_policy": "device default; no clock lock requested",
        "cache_contract": "fixed-pointer warm/reuse; no input-buffer rotation",
        "dtype": dtype_name,
        "seed": seed,
        "rounds": rounds,
        "iters_per_round": iters_per_round,
        "warmup_iters_per_capture": warmup_iters,
        "clock_warmup": f"{clock_warmup_iters} untimed 4096x4096 {dtype_name} GEMMs before each arm",
        "round_order": "cyclic candidate rotation",
        "timing": (
            "local CUDA-event timing with fixed-pointer CUDA graph replay"
            if use_cuda_graphs
            else "local per-call CUDA-event timing"
        ),
        "timed_region": "one internal FA4 forward call with output preallocated; excludes allocation, compile, warmup, and graph capture",
    }


def aggregate_bucket_results(rows: list[dict]) -> list[dict]:
    """Compare only configs measured for every case in each bucket and phase."""
    groups = {}
    for row in rows:
        groups.setdefault((row["bucket_name"], row["phase"]), []).append(row)

    summaries = []
    for (bucket_name, phase), group in groups.items():
        result_maps = [
            {
                json.dumps(result["config"], sort_keys=True): result
                for result in row["config_results"]
            }
            for row in group
        ]
        common_keys = set(result_maps[0]).intersection(
            *(set(row) for row in result_maps[1:])
        )
        all_keys = set().union(*(set(row) for row in result_maps))
        production_keys = {
            json.dumps(row["production_config"], sort_keys=True) for row in group
        }
        if len(production_keys) != 1:
            raise RuntimeError(f"Bucket {bucket_name} has multiple production defaults")
        production_key = next(iter(production_keys))
        if production_key not in common_keys:
            raise RuntimeError(
                f"Bucket {bucket_name} does not measure its default in every case"
            )

        totals = {key: 0.0 for key in common_keys}
        names = {key: set() for key in common_keys}
        wins = {key: 0 for key in common_keys}
        for result_map in result_maps:
            for key in common_keys:
                totals[key] += result_map[key]["median_us"]
                names[key].add(result_map[key]["name"])
            wins[min(common_keys, key=lambda key: result_map[key]["median_us"])] += 1
        winner_key = min(common_keys, key=totals.__getitem__)
        summaries.append(
            {
                "bucket_name": bucket_name,
                "phase": phase,
                "cases": len(group),
                "production_config": json.loads(production_key),
                "winner_config": json.loads(winner_key),
                "production_total_us": totals[production_key],
                "winner_total_us": totals[winner_key],
                "production_to_winner_speedup": totals[production_key]
                / totals[winner_key],
                "excluded_incomplete_configs": [
                    json.loads(key) for key in sorted(all_keys - common_keys)
                ],
                "config_results": [
                    {
                        "config": json.loads(config_key),
                        "names": sorted(names[config_key]),
                        "total_us": total,
                        "case_wins": wins[config_key],
                    }
                    for config_key, total in sorted(
                        totals.items(), key=lambda item: item[1]
                    )
                ],
            }
        )
    return summaries


def write_results_checkpoint(
    path: Path,
    metadata: dict,
    realistic_skipped: Sequence[str],
    correctness: list[dict],
    rows: list[dict],
    failures: list[dict],
    total_cases: int,
) -> list[dict]:
    """Atomically preserve raw samples and progress during long campaigns."""
    bucket_summaries = aggregate_bucket_results(rows)
    payload = {
        "metadata": metadata,
        "progress": {"completed": len(rows), "total": total_cases},
        "realistic_skipped": list(realistic_skipped),
        "correctness": correctness,
        "failures": failures,
        "bucket_summaries": bucket_summaries,
        "results": rows,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)
    return bucket_summaries


def main(
    out_dir: Path | None = None,
    workers: int = 16,
    rounds: int = 5,
    iters_per_round: int = 31,
    warmup_iters: int = 10,
    clock_warmup_iters: int = 20,
    use_cuda_graphs: bool = True,
    dtype_name: DTypeName = "bfloat16",
    seed: int = 0,
    experiment_filter: str = "",
    case_filter: str = "",
    phase_filter: str = "",
    bucket_filter: str = "",
    list_buckets: bool = False,
    realistic_workloads: Path | None = None,
    q_stage_hdim192: VarlenPolicySweep = DEFAULT_Q_STAGE_SWEEP,
    clc_varlen_mha: VarlenPolicySweep = DEFAULT_CLC_SWEEP,
    nonpersistent_hdim64: DensePolicySweep = DEFAULT_NONPERSISTENT_SWEEP,
    direct_epilogue_hdim64: DenseArchitectureSweep = DEFAULT_DIRECT_EPILOGUE_SWEEP,
    one_cta_hdim128: DenseArchitectureSweep = DEFAULT_ONE_CTA_SWEEP,
) -> None:
    """Compile, validate, and benchmark the frozen forward policy sweep."""
    if request_json := os.environ.get(INTERNAL_REQUEST_ENV):
        request = json.loads(request_json)
        raw_bench_case = request["bench_case"]
        bench_case = BenchCase(
            experiment=raw_bench_case["experiment"],
            phase=raw_bench_case["phase"],
            case=Case(**raw_bench_case["case"]),
        )
        raw_variant = request["variant"]
        config_module, _, _ = flash_attn_modules()
        variant = Variant(
            name=raw_variant["name"],
            bucket_name=raw_variant["bucket_name"],
            config=config_module.FwdConfig(**raw_variant["config"]),
        )
        print(
            json.dumps(
                compile_variant(
                    bench_case,
                    variant,
                    request["dtype_name"],
                    request["seed"],
                )
            )
        )
        return

    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    realistic = (
        generate_realistic_workloads(
            realistic_workloads,
            dtype_name,
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        )
        if realistic_workloads is not None
        else None
    )
    extra_cases = (
        [BenchCase("realistic", item.phase, item.case) for item in realistic.cases]
        if realistic is not None
        else []
    )
    cases = generate_cases(
        q_stage_hdim192,
        clc_varlen_mha,
        nonpersistent_hdim64,
        direct_epilogue_hdim64,
        one_cta_hdim128,
        experiment_filter,
        case_filter,
        phase_filter,
        extra_cases,
    )
    bucket_names = {
        variants_for_case(bench_case, dtype_name)[0].bucket_name for bench_case in cases
    }
    if list_buckets:
        print("\n".join(sorted(bucket_names)))
        return
    if bucket_filter:
        cases = [
            bench_case
            for bench_case in cases
            if bucket_filter in variants_for_case(bench_case, dtype_name)[0].bucket_name
        ]
    if not cases:
        raise ValueError("No cases selected")
    run_dir = out_dir or RESULTS_ROOT / utc_timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    compile_cases = [*cases, *correctness_cases()]
    print(
        f"cases={len(cases)} "
        f"compile_requests={len(select_compile_requests(compile_cases, dtype_name))}"
    )
    print(f"out_dir={run_dir}")

    run_compile(compile_cases, dtype_name, workers, seed, Path(__file__).resolve())
    correctness = run_correctness(dtype_name, seed, experiment_filter)
    metadata = environment_metadata(
        dtype_name,
        seed,
        rounds,
        iters_per_round,
        warmup_iters,
        clock_warmup_iters,
        use_cuda_graphs,
    )

    rows = []
    failures = []
    results_path = run_dir / "results.json"
    skipped = realistic.skipped if realistic is not None else ()
    write_results_checkpoint(
        results_path, metadata, skipped, correctness, rows, failures, len(cases)
    )
    for index, bench_case in enumerate(cases, start=1):
        try:
            row = benchmark_case(
                bench_case,
                dtype_name,
                rounds,
                iters_per_round,
                warmup_iters,
                clock_warmup_iters,
                use_cuda_graphs,
                seed,
            )
        except Exception as error:
            failures.append(
                {
                    "case": bench_case.case.name,
                    "phase": bench_case.phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            write_results_checkpoint(
                results_path,
                metadata,
                skipped,
                correctness,
                rows,
                failures,
                len(cases),
            )
            raise
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {row['experiment']} {row['phase']} {row['name']}: "
            f"production={row['production_median_us']:.3f}us "
            f"winner={row['winner_median_us']:.3f}us "
            f"({row['production_to_winner_speedup']:.3f}x)"
        )
        if index % 10 == 0 or index == len(cases):
            write_results_checkpoint(
                results_path,
                metadata,
                skipped,
                correctness,
                rows,
                failures,
                len(cases),
            )
        torch.cuda.empty_cache()

    bucket_summaries = aggregate_bucket_results(rows)
    for summary in bucket_summaries:
        print(
            f"bucket {summary['bucket_name']} {summary['phase']}: "
            f"production-to-winner={summary['production_to_winner_speedup']:.3f}x"
        )
    print(f"results={run_dir / 'results.json'}")


if __name__ == "__main__":
    CLI(main, as_positional=False)
