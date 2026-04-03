#!/usr/bin/env python3
"""FA4 CI driver — runs inside an Apptainer SIF on a self-hosted GPU runner.

Requires FA4_SIF (path to the .sif image) to be set, either via env var or --sif.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TEST_FILTER = ""  # empty = run all; CI overrides via --test-filter
DEFAULT_TEST_TARGET = "tests/cute/test_flash_attn.py"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    extra_env: dict[str, str]


# ── GPU helpers ───────────────────────────────────────────────────────────────

def parse_free_gpu_indices(nvidia_smi_output: str, min_free_memory_mb: int) -> list[str]:
    indices: list[str] = []
    for raw_line in nvidia_smi_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            index, free_memory = [part.strip() for part in line.split(",", maxsplit=1)]
            if int(free_memory) >= min_free_memory_mb:
                indices.append(index)
        except ValueError as exc:
            raise ValueError(f"Unexpected nvidia-smi output line: {raw_line!r}") from exc
    return indices


def select_visible_devices(free_gpu_indices: list[str], use_all_free_gpus: bool) -> str:
    if not free_gpu_indices:
        raise ValueError("No GPUs satisfy the free-memory threshold")
    if use_all_free_gpus:
        return ",".join(free_gpu_indices)
    return free_gpu_indices[0]


def read_free_gpu_indices(min_free_memory_mb: int) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return parse_free_gpu_indices(result.stdout, min_free_memory_mb)


# ── Step plan ─────────────────────────────────────────────────────────────────

def build_step_plan(
    test_target: str,
    test_filter: str,
    compile_workers: int,
    run_workers: int,
    test_visible_devices: str,
    benchmark_visible_devices: str,
    skip_benchmark: bool,
) -> list[Step]:
    pytest_base = ["python3", "-m", "pytest", test_target, *(["-k", test_filter] if test_filter else [])]

    steps = [
        Step(
            name="Pass 1: compile kernels (no GPU memory)",
            command=[*pytest_base, "-n", str(compile_workers), "-x"],
            extra_env={"FLASH_ATTENTION_FAKE_TENSOR": "1"},
        ),
        Step(
            name="Pass 2: run tests on GPU",
            command=[*pytest_base, "-n", str(run_workers), "-x"],
            extra_env={
                "FLASH_ATTENTION_FAKE_TENSOR": "0",
                "CUDA_VISIBLE_DEVICES": test_visible_devices,
            },
        ),
    ]
    if not skip_benchmark:
        steps.append(Step(
            name="Benchmark (FA4 fwd, hdim=128, seqlen=8192)",
            command=[
                "python3", "benchmarks/benchmark_attn.py",
                "--backend", "fa4", "--fwd",
                "--headdim", "128", "--seqlen", "8192",
                "--causal", "both", "--warmup", "1", "--rep", "3",
            ],
            extra_env={"CUDA_VISIBLE_DEVICES": benchmark_visible_devices},
        ))
    return steps


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(step: Step, repo_root: Path, base_env: dict[str, str], sif: str, work_dir: str) -> None:
    print(f"=== {step.name} ===")

    # Install FA4 from the current repo inside this exec invocation.
    # Must be done per-step because --writable-tmpfs creates a fresh overlay each time.
    install_cmd = f"uv pip install --system --break-system-packages --no-deps -q -e {shlex.quote(str(repo_root / 'flash_attn/cute'))}"

    # Convert relative test/benchmark paths to absolute so we can run from /tmp.
    # Running from /tmp ensures Python does not insert repo_root into sys.path[0]
    # (which would cause flash_attn/__init__.py to trigger FA2 imports unavailable in the SIF).
    command = [
        str(repo_root / arg) if (arg.startswith("tests/") or arg.startswith("benchmarks/")) else arg
        for arg in step.command
    ]
    env_exports = " && ".join(f"export {k}={shlex.quote(v)}" for k, v in step.extra_env.items())
    inner_cmd = shlex.join(command)
    shell_parts = [install_cmd]
    if env_exports:
        shell_parts.append(env_exports)
    shell_parts.append(f"cd /tmp && {inner_cmd}")
    cmd = ["apptainer", "exec", "--nv", "--writable-tmpfs", "--bind", work_dir, sif, "bash", "-c", " && ".join(shell_parts)]
    subprocess.run(cmd, check=True, cwd=repo_root, env=base_env)


# ── CLI ───────────────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sif", default=os.environ.get("FA4_SIF", ""),
                        help="Apptainer .sif image path (or set FA4_SIF env var)")
    parser.add_argument("--test-target", default=DEFAULT_TEST_TARGET)
    parser.add_argument("--test-filter", default=DEFAULT_TEST_FILTER)
    parser.add_argument("--compile-workers", type=int, default=1)
    parser.add_argument("--run-workers", type=int, default=1)
    parser.add_argument("--min-free-memory-mb", type=int, default=40000)
    parser.add_argument("--use-all-free-gpus", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    repo_root = args.repo_root.resolve()

    if not args.sif:
        raise SystemExit("FA4_SIF is not set — provide --sif or set the FA4_SIF env var.")
    print(f"Using SIF: {args.sif}")

    free_gpu_indices = read_free_gpu_indices(args.min_free_memory_mb)
    test_visible_devices = select_visible_devices(free_gpu_indices, args.use_all_free_gpus)
    benchmark_visible_devices = free_gpu_indices[0]
    print(f"Running tests on GPUs: {test_visible_devices}")

    base_env = {**os.environ, "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1"}
    work_dir = os.environ.get("CI_WORK_DIR", f"/scratch/user/{os.environ.get('USER', 'user')}")

    for step in build_step_plan(
        test_target=args.test_target,
        test_filter=args.test_filter,
        compile_workers=args.compile_workers,
        run_workers=args.run_workers,
        test_visible_devices=test_visible_devices,
        benchmark_visible_devices=benchmark_visible_devices,
        skip_benchmark=args.skip_benchmark,
    ):
        run_step(step, repo_root=repo_root, base_env=base_env, sif=args.sif, work_dir=work_dir)

    print("=== All tests passed ===")


if __name__ == "__main__":
    main()
