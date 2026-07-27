#!/usr/bin/env python3
"""Run data-driven A/B measurements of explicit FA4 forward configs.

A campaign is only data: named baseline/candidate config overrides and dense or
packed-varlen shape grids. The runner compiles every distinct specialization in
isolated fake-tensor subprocesses, checks both arms against a float32 reference,
then measures fixed-pointer CUDA-graph replays in cyclic order. Its summary puts
the per-policy geometric mean, range, and paired-round 95% interval directly in
results.json and stdout.

Example:
  gpu-run -- python benchmarks/fwd_config_bench.py \
    --campaign benchmarks/configs/fwd_config_sm103.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import types
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import accumulate, product
from pathlib import Path
from typing import Literal

import torch
import yaml
from torch._subclasses.fake_tensor import FakeTensorMode

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results" / "fwd_config"
COMPILE_REQUEST_ENV = "FWD_CONFIG_BENCH_COMPILE_REQUEST"
COMPILE_TIMEOUT_SECONDS = 15 * 60
DTypeName = Literal["bfloat16", "float16"]
Phase = Literal["discovery", "boundary", "holdout", "correctness"]
Mode = Literal["dense", "varlen"]


@dataclass(frozen=True)
class Case:
    experiment: str
    phase: Phase
    name: str
    mode: Mode
    q_heads: int
    kv_heads: int
    d: int
    dv: int
    causal: bool
    batch: int
    num_splits: int = 1
    seqlen_q: int | None = None
    seqlen_k: int | None = None
    seqlens_q: tuple[int, ...] = ()
    seqlens_k: tuple[int, ...] = ()

    @property
    def total_q(self) -> int:
        return (
            sum(self.seqlens_q)
            if self.mode == "varlen"
            else self.batch * (self.seqlen_q or 0)
        )

    @property
    def total_k(self) -> int:
        return (
            sum(self.seqlens_k)
            if self.mode == "varlen"
            else self.batch * (self.seqlen_k or 0)
        )

    @property
    def max_q(self) -> int:
        return max(self.seqlens_q) if self.mode == "varlen" else self.seqlen_q or 0

    @property
    def max_k(self) -> int:
        return max(self.seqlens_k) if self.mode == "varlen" else self.seqlen_k or 0


@dataclass(frozen=True)
class Arm:
    name: Literal["baseline", "candidate"]
    config: object


@dataclass(frozen=True)
class TimingStats:
    samples_us: tuple[float, ...]
    median_us: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> TimingStats:
        values = tuple(float(value) for value in samples)
        return cls(values, statistics.median(values))


def flash_modules():
    """Import the local FA4 package without importing the optional legacy extension."""
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    if "flash_attn" not in sys.modules:
        package = types.ModuleType("flash_attn")
        package.__path__ = [str(REPO_ROOT / "flash_attn")]
        sys.modules["flash_attn"] = package
    from flash_attn.cute import config as config_module
    from flash_attn.cute import interface as interface_module

    return config_module, interface_module


def normalize_lengths(pattern: str, batch: int, total: int) -> tuple[int, ...]:
    """Build deterministic packed lengths with an exact token total."""
    match pattern:
        case "uniform":
            weights = [1.0] * batch
        case "longtail":
            weights = [float(batch - index) for index in range(batch)]
        case "bimodal":
            weights = [8.0] * max(1, batch // 2) + [1.0] * (batch - max(1, batch // 2))
        case "staircase":
            weights = [float(index + 1) for index in range(batch)]
        case _:
            raise ValueError(f"unknown packed-length pattern {pattern!r}")
    if total < batch:
        raise ValueError(f"total tokens {total} must be at least batch {batch}")
    scaled = [weight * total / sum(weights) for weight in weights]
    lengths = [max(1, math.floor(value)) for value in scaled]
    delta = total - sum(lengths)
    order = sorted(
        range(batch),
        key=lambda index: scaled[index] - math.floor(scaled[index]),
        reverse=delta > 0,
    )
    cursor = 0
    while delta:
        index = order[cursor % batch]
        if delta > 0:
            lengths[index] += 1
            delta -= 1
        elif lengths[index] > 1:
            lengths[index] -= 1
            delta += 1
        cursor += 1
    return tuple(lengths)


def dim_pairs(values: Sequence[int | Sequence[int]]) -> tuple[tuple[int, int], ...]:
    pairs = []
    for value in values:
        if isinstance(value, int):
            pairs.append((value, value))
        elif len(value) == 2:
            pairs.append((int(value[0]), int(value[1])))
        else:
            raise ValueError(f"head_dims entries must be D or [D, DV], got {value}")
    return tuple(pairs)


def bool_values(value: bool | Sequence[bool]) -> tuple[bool, ...]:
    return (value,) if isinstance(value, bool) else tuple(value)


def expand_grid(experiment: str, grid: dict) -> list[Case]:
    """Expand one compact dense or varlen Cartesian grid."""
    mode = grid["mode"]
    common = product(
        grid["head_pairs"],
        dim_pairs(grid["head_dims"]),
        grid["batches"],
        bool_values(grid.get("causal", False)),
        grid.get("num_splits", [1]),
    )
    cases = []
    for (q_heads, kv_heads), (d, dv), batch, causal, num_splits in common:
        if q_heads % kv_heads:
            raise ValueError(
                f"Q heads {q_heads} must be divisible by KV heads {kv_heads}"
            )
        if mode == "dense":
            shapes = ((q, k, "") for q, k in grid["seqlen_pairs"])
        elif mode == "varlen":
            shapes = (
                (total_q, total_k, pattern)
                for (total_q, total_k), pattern in product(
                    grid["token_pairs"], grid.get("patterns", ["uniform"])
                )
            )
        else:
            raise ValueError(f"unsupported mode {mode!r}")
        for q_length, k_length, pattern in shapes:
            suffix = (
                f"q{q_length}_k{k_length}_b{batch}"
                if mode == "dense"
                else f"q{q_length}_k{k_length}_b{batch}_{pattern}"
            )
            suffix += f"_c{int(causal)}_s{num_splits}"
            cases.append(
                Case(
                    experiment=experiment,
                    phase=grid["phase"],
                    name=f"{experiment}__{suffix}__h{q_heads}kv{kv_heads}d{d}v{dv}",
                    mode=mode,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    d=d,
                    dv=dv,
                    causal=causal,
                    batch=batch,
                    num_splits=num_splits,
                    seqlen_q=q_length if mode == "dense" else None,
                    seqlen_k=k_length if mode == "dense" else None,
                    seqlens_q=(
                        normalize_lengths(pattern, batch, q_length)
                        if mode == "varlen"
                        else ()
                    ),
                    seqlens_k=(
                        normalize_lengths(pattern, batch, k_length)
                        if mode == "varlen"
                        else ()
                    ),
                )
            )
    return cases


def expand_case_table(experiment: str, table: dict) -> list[Case]:
    """Expand a compact table of intentionally selected timing cells."""
    columns = table["columns"]
    defaults = {
        key: value for key, value in table.items() if key not in ("columns", "rows")
    }
    cases = []
    for row in table["rows"]:
        if len(row) != len(columns):
            raise ValueError(f"{experiment} case row does not match {columns}")
        values = {**defaults, **dict(zip(columns, row))}
        grid = {
            "phase": values["phase"],
            "mode": values["mode"],
            "head_pairs": [[values["q_heads"], values["kv_heads"]]],
            "head_dims": [
                [values["head_dim"], values.get("head_dim_v", values["head_dim"])]
            ],
            "batches": [values["batch"]],
            "causal": values.get("causal", False),
            "num_splits": [values.get("num_splits", 1)],
        }
        if values["mode"] == "dense":
            grid["seqlen_pairs"] = [[values["seqlen_q"], values["seqlen_k"]]]
        else:
            grid["token_pairs"] = [[values["total_q"], values["total_k"]]]
            grid["patterns"] = [values.get("pattern", "uniform")]
        cases.extend(expand_grid(experiment, grid))
    return cases


def explicit_case(experiment: str, values: dict) -> Case:
    """Load one intentionally small independent-correctness case."""
    mode = values["mode"]
    d, dv = dim_pairs([values["head_dim"]])[0]
    lengths_q = tuple(values.get("seqlens_q", ()))
    lengths_k = tuple(values.get("seqlens_k", lengths_q))
    batch = len(lengths_q) if mode == "varlen" else values["batch"]
    return Case(
        experiment=experiment,
        phase="correctness",
        name=f"{experiment}__correctness",
        mode=mode,
        q_heads=values["q_heads"],
        kv_heads=values["kv_heads"],
        d=d,
        dv=dv,
        causal=values.get("causal", False),
        batch=batch,
        num_splits=values.get("num_splits", 1),
        seqlen_q=values.get("seqlen_q"),
        seqlen_k=values.get("seqlen_k"),
        seqlens_q=lengths_q,
        seqlens_k=lengths_k,
    )


def load_campaign(path: Path) -> tuple[dict, dict[str, dict], list[Case], list[Case]]:
    spec = yaml.safe_load(path.read_text())
    experiments = {item["name"]: item for item in spec["experiments"]}
    if len(experiments) != len(spec["experiments"]):
        raise ValueError("experiment names must be unique")
    cases = [
        case
        for experiment in experiments.values()
        for grid in experiment.get("grids", ())
        for case in expand_grid(experiment["name"], grid)
    ] + [
        case
        for experiment in experiments.values()
        if "case_table" in experiment
        for case in expand_case_table(experiment["name"], experiment["case_table"])
    ]
    correctness = [
        explicit_case(experiment["name"], experiment["correctness"])
        for experiment in experiments.values()
    ]
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("campaign generated duplicate case names")
    return spec.get("settings", {}), experiments, cases, correctness


def selector_inputs(case: Case, dtype_name: DTypeName):
    """Construct exactly the host metadata used by the production selector."""
    config_module, interface_module = flash_modules()
    arch = interface_module._get_device_arch()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    is_varlen = case.mode == "varlen"
    return config_module.FwdHeuristicInputs(
        device_arch=arch,
        num_sms=props.multi_processor_count,
        dtype=str(getattr(torch, dtype_name)),
        head_dim=case.d,
        head_dim_v=case.dv,
        num_heads=case.q_heads,
        num_heads_kv=case.kv_heads,
        batch_size=case.batch,
        total_q=case.total_q,
        total_k=case.total_k,
        max_seqlen_q=case.max_q,
        max_seqlen_k=case.max_k,
        seqlen_k_per_split=None,
        causal=case.causal,
        local=False,
        window_size_left=None,
        window_size_right=None,
        is_varlen_q=is_varlen,
        has_cu_seqlens_q=is_varlen,
        has_cu_seqlens_k=is_varlen,
        has_seqused=False,
        pack_gqa=case.q_heads != case.kv_heads,
        page_size=None,
        use_block_sparsity=False,
        sparse_q_block_size=None,
        has_qv=False,
        has_gather_kv=False,
        has_score_mod=False,
        has_mask_mod=False,
        has_learnable_sink=False,
        has_lse=False,
        requested_tile_m=None,
        requested_tile_n=None,
        requested_mma_pv_is_rs=None,
        requested_intra_wg_overlap=None,
        requested_num_splits=None if case.num_splits < 1 else case.num_splits,
        requested_use_clc_scheduler=False,
        disable_2cta=False,
    )


def resolve_arms(
    case: Case, experiment: dict, dtype_name: DTypeName
) -> tuple[Arm, Arm]:
    config_module, _ = flash_modules()
    inputs = selector_inputs(case, dtype_name)
    selected = config_module.select_fwd_config(inputs)
    arms = tuple(
        Arm(name, replace(selected, **experiment[name]))
        for name in ("baseline", "candidate")
    )
    if arms[0].config == arms[1].config:
        raise ValueError(
            f"{case.name} resolves to identical baseline and candidate configs"
        )
    for arm in arms:
        config_module.validate_fwd_config(arm.config, inputs)
    return arms


def build_inputs(case: Case, dtype, factory: Callable) -> dict:
    """Allocate standard dense or packed-varlen Q/K/V tensors."""
    if case.mode == "dense":
        return {
            "q": factory(
                case.batch,
                case.seqlen_q,
                case.q_heads,
                case.d,
                device="cuda",
                dtype=dtype,
            ),
            "k": factory(
                case.batch,
                case.seqlen_k,
                case.kv_heads,
                case.d,
                device="cuda",
                dtype=dtype,
            ),
            "v": factory(
                case.batch,
                case.seqlen_k,
                case.kv_heads,
                case.dv,
                device="cuda",
                dtype=dtype,
            ),
            "causal": case.causal,
            "num_splits": case.num_splits,
        }
    cu_q = torch.tensor(
        [0, *accumulate(case.seqlens_q)], device="cuda", dtype=torch.int32
    )
    cu_k = torch.tensor(
        [0, *accumulate(case.seqlens_k)], device="cuda", dtype=torch.int32
    )
    return {
        "q": factory(case.total_q, case.q_heads, case.d, device="cuda", dtype=dtype),
        "k": factory(case.total_k, case.kv_heads, case.d, device="cuda", dtype=dtype),
        "v": factory(case.total_k, case.kv_heads, case.dv, device="cuda", dtype=dtype),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": case.max_q,
        "max_seqlen_k": case.max_k,
        "causal": case.causal,
        "num_splits": case.num_splits,
    }


def compile_signature(case: Case, arm: Arm) -> tuple:
    """Deduplicate requests across exact main/combine code-generation boundaries."""
    config_module, _ = flash_modules()
    main, combine = config_module.fwd_config_compile_bucket(arm.config, case.dv)
    return (
        case.mode,
        case.q_heads // case.kv_heads,
        case.d,
        case.dv,
        case.causal,
        main,
        combine,
    )


def compile_one(case: Case, arm: Arm, dtype_name: DTypeName, seed: int) -> dict:
    """Compile while tensor creation and invocation share one FakeTensorMode."""
    config_module, interface_module = flash_modules()
    config = config_module.FwdConfig(**asdict(arm.config))
    torch.manual_seed(seed)
    with FakeTensorMode():
        kwargs = build_inputs(case, getattr(torch, dtype_name), torch.empty)
        interface_module._flash_attn_fwd(**kwargs, config=config)
    return {"case": case.name, "arm": arm.name, "compiled": True}


def compile_subprocess(case: Case, arm: Arm, dtype_name: DTypeName, seed: int) -> dict:
    env = os.environ.copy()
    env[COMPILE_REQUEST_ENV] = json.dumps(
        {
            "case": asdict(case),
            "arm": {"name": arm.name, "config": asdict(arm.config)},
            "dtype": dtype_name,
            "seed": seed,
        }
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=COMPILE_TIMEOUT_SECONDS,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"compile request produced no JSON:\n{completed.stderr}")


def compile_barrier(
    cases: Sequence[Case],
    experiments: dict[str, dict],
    dtype_name: DTypeName,
    workers: int,
    seed: int,
) -> list[dict]:
    requests = {}
    for case in cases:
        for arm in resolve_arms(case, experiments[case.experiment], dtype_name):
            requests.setdefault(compile_signature(case, arm), (case, arm))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(compile_subprocess, case, arm, dtype_name, seed): (case, arm)
            for case, arm in requests.values()
        }
        for index, future in enumerate(as_completed(futures), 1):
            case, arm = futures[future]
            try:
                row = future.result()
            except (subprocess.SubprocessError, OSError) as error:
                detail = getattr(error, "stderr", None) or getattr(
                    error, "stdout", None
                )
                detail = str(detail).strip() if detail else str(error)
                raise RuntimeError(
                    f"compile failed for {case.name}/{arm.name}:\n{detail}"
                ) from error
            rows.append(row)
            print(f"[{index}/{len(futures)}] compiled {case.experiment}/{arm.name}")
    return rows


def attention_reference(case: Case, kwargs: dict) -> torch.Tensor:
    """Independent float32 GQA/MQA reference with lower-right causal masking."""
    outputs = []
    q_offset = k_offset = 0
    lengths = (
        zip(case.seqlens_q, case.seqlens_k)
        if case.mode == "varlen"
        else [(case.seqlen_q or 0, case.seqlen_k or 0)] * case.batch
    )
    for batch_index, (q_length, k_length) in enumerate(lengths):
        if case.mode == "dense":
            q = kwargs["q"][batch_index].float().transpose(0, 1)
            k = kwargs["k"][batch_index].float().transpose(0, 1)
            v = kwargs["v"][batch_index].float().transpose(0, 1)
        else:
            q = kwargs["q"][q_offset : q_offset + q_length].float().transpose(0, 1)
            k = kwargs["k"][k_offset : k_offset + k_length].float().transpose(0, 1)
            v = kwargs["v"][k_offset : k_offset + k_length].float().transpose(0, 1)
        if case.q_heads != case.kv_heads:
            repeat_heads = case.q_heads // case.kv_heads
            k = k.repeat_interleave(repeat_heads, dim=0)
            v = v.repeat_interleave(repeat_heads, dim=0)
        scores = q @ k.transpose(-1, -2) / math.sqrt(case.d)
        if case.causal:
            q_positions = torch.arange(q_length, device="cuda")[:, None]
            k_positions = torch.arange(k_length, device="cuda")[None, :]
            scores.masked_fill_(
                k_positions > q_positions + k_length - q_length, -torch.inf
            )
        output = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0) @ v
        outputs.append(output.transpose(0, 1))
        q_offset += q_length
        k_offset += k_length
    return torch.stack(outputs) if case.mode == "dense" else torch.cat(outputs)


def run_correctness(
    cases: Sequence[Case],
    experiments: dict[str, dict],
    dtype_name: DTypeName,
    seed: int,
) -> list[dict]:
    _, interface_module = flash_modules()
    tolerance = 0.04 if dtype_name == "bfloat16" else 0.015
    rows = []
    for case in cases:
        torch.manual_seed(seed)
        kwargs = build_inputs(case, getattr(torch, dtype_name), torch.randn)
        reference = attention_reference(case, kwargs)
        for arm in resolve_arms(case, experiments[case.experiment], dtype_name):
            output = interface_module._flash_attn_fwd(**kwargs, config=arm.config)[0]
            torch.testing.assert_close(
                output.float(), reference, atol=tolerance, rtol=tolerance
            )
        automatic = interface_module._flash_attn_fwd(**kwargs)[0]
        torch.testing.assert_close(
            automatic.float(), reference, atol=tolerance, rtol=tolerance
        )
        rows.append({"experiment": case.experiment, "case": case.name, "passed": True})
        print(f"correctness passed: {case.experiment}")
    return rows


def split_workspaces(case: Case, kwargs: dict, config) -> dict:
    if config.num_splits == 1:
        return {}
    q = kwargs["q"]
    output_shape = (*q.shape[:-1], case.dv)
    lse_shape = (
        (case.batch, case.q_heads, case.seqlen_q)
        if case.mode == "dense"
        else (case.q_heads, case.total_q)
    )
    return {
        "out_partial": torch.empty(
            config.num_splits, *output_shape, dtype=torch.float32, device="cuda"
        ),
        "lse_partial": torch.empty(
            config.num_splits, *lse_shape, dtype=torch.float32, device="cuda"
        ),
    }


def make_callables(
    case: Case, arms: Sequence[Arm], dtype_name: DTypeName, seed: int
) -> dict[str, Callable]:
    _, interface_module = flash_modules()
    torch.manual_seed(seed)
    kwargs = build_inputs(case, getattr(torch, dtype_name), torch.randn)
    callables = {}
    for arm in arms:
        call_kwargs = {
            **kwargs,
            "out": torch.empty(
                *kwargs["q"].shape[:-1],
                case.dv,
                device="cuda",
                dtype=getattr(torch, dtype_name),
            ),
            **split_workspaces(case, kwargs, arm.config),
        }

        def invoke(call_kwargs=call_kwargs, config=arm.config):
            return interface_module._flash_attn_fwd(**call_kwargs, config=config)[0]

        callables[arm.name] = invoke
    return callables


def benchmark_stats(
    function: Callable, iterations: int, warmup_iterations: int
) -> TimingStats:
    for _ in range(warmup_iterations):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    for _ in range(warmup_iterations):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return TimingStats.from_samples(samples)


def clock_warmup(dtype_name: DTypeName, iterations: int) -> Callable:
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
    case: Case,
    experiment: dict,
    dtype_name: DTypeName,
    rounds: int,
    iterations: int,
    warmup_iterations: int,
    warm_clocks: Callable,
    seed: int,
) -> dict:
    arms = resolve_arms(case, experiment, dtype_name)
    callables = make_callables(case, arms, dtype_name, seed)
    samples = {arm.name: [] for arm in arms}
    round_medians = {arm.name: [] for arm in arms}
    for round_index in range(rounds):
        order = arms[round_index % 2 :] + arms[: round_index % 2]
        for arm in order:
            warm_clocks()
            stats = benchmark_stats(callables[arm.name], iterations, warmup_iterations)
            samples[arm.name].extend(stats.samples_us)
            round_medians[arm.name].append(stats.median_us)
    medians = {
        name: TimingStats.from_samples(values).median_us
        for name, values in samples.items()
    }
    return {
        **asdict(case),
        "baseline_config": asdict(arms[0].config),
        "candidate_config": asdict(arms[1].config),
        "baseline_median_us": medians["baseline"],
        "candidate_median_us": medians["candidate"],
        "speedup": medians["baseline"] / medians["candidate"],
        "baseline_samples_us": samples["baseline"],
        "candidate_samples_us": samples["candidate"],
        "baseline_round_medians_us": round_medians["baseline"],
        "candidate_round_medians_us": round_medians["candidate"],
    }


def summarize(rows: Sequence[dict], experiments: dict[str, dict]) -> list[dict]:
    """Emit one review-ready gain summary per policy."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["experiment"], []).append(row)
    summaries = []
    for experiment, group in sorted(groups.items()):
        ratios = [row["speedup"] for row in group]
        paired_round_logs = [
            [
                math.log(baseline / candidate)
                for baseline, candidate in zip(
                    row["baseline_round_medians_us"],
                    row["candidate_round_medians_us"],
                )
            ]
            for row in group
        ]
        mean_log = statistics.mean(map(statistics.mean, paired_round_logs))
        # The campaign cells are fixed. Estimate timing uncertainty from paired
        # rounds within each cell rather than pretending the grid is random.
        variance = (
            sum(
                statistics.variance(logs) / len(logs) if len(logs) > 1 else 0.0
                for logs in paired_round_logs
            )
            / len(group) ** 2
        )
        stderr = math.sqrt(variance)
        summaries.append(
            {
                "experiment": experiment,
                "cases": len(group),
                "phases": sorted({row["phase"] for row in group}),
                "baseline_overrides": experiments[experiment]["baseline"],
                "candidate_overrides": experiments[experiment]["candidate"],
                "geomean_speedup": math.exp(mean_log),
                "time_weighted_speedup": sum(row["baseline_median_us"] for row in group)
                / sum(row["candidate_median_us"] for row in group),
                "minimum_speedup": min(ratios),
                "maximum_speedup": max(ratios),
                "wins_neutral_losses": [
                    sum(value > 1.01 for value in ratios),
                    sum(0.99 <= value <= 1.01 for value in ratios),
                    sum(value < 0.99 for value in ratios),
                ],
                "geomean_ci95": [
                    math.exp(mean_log - 1.96 * stderr),
                    math.exp(mean_log + 1.96 * stderr),
                ],
            }
        )
    return summaries


def metadata(campaign: Path, clock_warmup_iters: int) -> dict:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    git = lambda *args: subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_status": git("status", "--short"),
        "campaign": str(campaign),
        "campaign_sha256": hashlib.sha256(campaign.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sms": props.multi_processor_count,
        "timing_contract": (
            "fixed-pointer CUDA-graph replay; preallocated outputs/workspaces; cyclic arms; "
            f"{clock_warmup_iters} untimed 4096x4096 GEMMs before each arm"
        ),
    }


def write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--experiment", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    if request_json := os.environ.get(COMPILE_REQUEST_ENV):
        request = json.loads(request_json)
        config_module, _ = flash_modules()
        case = Case(**request["case"])
        arm = Arm(
            request["arm"]["name"], config_module.FwdConfig(**request["arm"]["config"])
        )
        print(json.dumps(compile_one(case, arm, request["dtype"], request["seed"])))
        return

    args = parse_args()
    settings, experiments, cases, correctness_cases = load_campaign(args.campaign)
    if args.experiment:
        cases = [case for case in cases if case.experiment == args.experiment]
        correctness_cases = [
            case for case in correctness_cases if case.experiment == args.experiment
        ]
    if args.phase:
        cases = [case for case in cases if case.phase == args.phase]
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case.name]
    if not cases:
        raise ValueError("campaign filters selected no timing cases")
    if args.dry_run:
        counts = {}
        for case in cases:
            counts[(case.experiment, case.phase)] = (
                counts.get((case.experiment, case.phase), 0) + 1
            )
        print(
            json.dumps(
                {f"{key[0]}/{key[1]}": value for key, value in sorted(counts.items())},
                indent=2,
            )
        )
        return

    dtype_name = settings.get("dtype", "bfloat16")
    seed = settings.get("seed", 0)
    workers = args.workers or settings.get("workers", 16)
    rounds = settings.get("rounds", 7)
    iterations = settings.get("iters_per_round", 31)
    warmup_iterations = settings.get("warmup_iters", 10)
    clock_warmup_iterations = settings.get("clock_warmup_iters", 20)
    run_dir = args.out_dir or RESULTS_ROOT / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "results.json"
    print(f"cases={len(cases)} out_dir={run_dir}")

    compiled = compile_barrier(
        [*cases, *correctness_cases], experiments, dtype_name, workers, seed
    )
    correctness = run_correctness(correctness_cases, experiments, dtype_name, seed)
    payload = {
        "metadata": metadata(args.campaign, clock_warmup_iterations),
        "progress": {"completed": 0, "total": len(cases)},
        "compile": compiled,
        "correctness": correctness,
        "summaries": [],
        "results": [],
    }
    write_checkpoint(output, payload)
    warm_clocks = clock_warmup(dtype_name, clock_warmup_iterations)
    for index, case in enumerate(cases, 1):
        row = benchmark_case(
            case,
            experiments[case.experiment],
            dtype_name,
            rounds,
            iterations,
            warmup_iterations,
            warm_clocks,
            seed,
        )
        payload["results"].append(row)
        payload["progress"]["completed"] = index
        payload["summaries"] = summarize(payload["results"], experiments)
        write_checkpoint(output, payload)
        print(f"[{index}/{len(cases)}] {case.name}: {row['speedup']:.3f}x")
        torch.cuda.empty_cache()

    for summary in payload["summaries"]:
        wins, neutral, losses = summary["wins_neutral_losses"]
        low, high = summary["geomean_ci95"]
        print(
            f"{summary['experiment']}: n={summary['cases']} "
            f"geomean={summary['geomean_speedup']:.3f}x "
            f"weighted={summary['time_weighted_speedup']:.3f}x "
            f"range=[{summary['minimum_speedup']:.3f}, "
            f"{summary['maximum_speedup']:.3f}]x "
            f"W/N/L={wins}/{neutral}/{losses} "
            f"geomean95=[{low:.3f}, {high:.3f}]"
        )
    print(f"results={output}")


if __name__ == "__main__":
    main()
