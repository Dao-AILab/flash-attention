#!/usr/bin/env python3
"""FA4 CI driver — runs inside an Apptainer SIF on a self-hosted GPU runner.

Requires FA4_SIF (path to the .sif image) to be set, either via env var or --sif.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.request
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

def read_idle_gpu_indices(max_used_memory_mb: int = 1000) -> list[str]:
    """Return indices of GPUs that are truly idle: utilization==0 and only driver memory used."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    indices: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Unexpected nvidia-smi output: {raw_line!r}")
        idx, util, mem_used = parts[0], int(parts[1]), int(parts[2])
        if util == 0 and mem_used <= max_used_memory_mb:
            indices.append(idx)
    return indices


def select_visible_devices(idle_gpu_indices: list[str], use_all_free_gpus: bool) -> str:
    if not idle_gpu_indices:
        raise ValueError("No idle GPUs available")
    if use_all_free_gpus:
        return ",".join(idle_gpu_indices)
    return idle_gpu_indices[0]


def post_slack(webhook_url: str, text: str) -> None:
    if not webhook_url:
        return
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(webhook_url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"WARNING: Slack notification failed: {e}")


def wait_for_idle_gpus(
    max_used_memory_mb: int,
    poll_interval_s: int,
    timeout_s: int,
    webhook_url: str,
) -> list[str]:
    """Poll until at least one GPU is idle or timeout expires. Returns idle GPU indices."""
    deadline = time.monotonic() + timeout_s
    first_check = True
    while True:
        indices = read_idle_gpu_indices(max_used_memory_mb)
        if indices:
            if not first_check:
                print(f"Idle GPUs available: {indices}")
            return indices

        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            hostname = os.environ.get("RUNNER_NAME", os.uname().nodename)
            run_url = ""
            repo = os.environ.get("GITHUB_REPOSITORY", "")
            run_id = os.environ.get("GITHUB_RUN_ID", "")
            server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            if repo and run_id:
                run_url = f" | <{server}/{repo}/actions/runs/{run_id}|View run>"
            post_slack(
                webhook_url,
                f":warning: *FA4 Nightly skipped* — no idle GPUs on `{hostname}` "
                f"after {timeout_s // 60} min{run_url}",
            )
            raise SystemExit(f"No idle GPUs after {timeout_s // 60} min — aborting.")

        if first_check:
            print(f"No idle GPUs found. Waiting up to {timeout_s // 60} min "
                  f"(polling every {poll_interval_s // 60} min)...")
            first_check = False
        print(f"  still waiting... {remaining // 60} min remaining")
        time.sleep(poll_interval_s)


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
            name="Benchmark (FA4 fwd, hdim=128, causal=both, seqlen=1K-32K)",
            command=[
                "python3", "benchmarks/benchmark_attn.py",
                "--backend", "fa4", "--fwd", "--bwd",
                "--headdim", "128",
                "--seqlen", "1024,2048,4096,8192,16384,32768",
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
    parser.add_argument("--run-workers", type=int, default=0,
                        help="xdist workers for Pass 2 (default: 0 = one per free GPU)")
    parser.add_argument("--max-used-memory-mb", type=int, default=1000,
                        help="GPU is considered idle if memory.used <= this value (default: 1000 MB)")
    parser.add_argument("--gpu-wait-timeout-min", type=int, default=60,
                        help="Minutes to wait for an idle GPU before giving up (default: 60)")
    parser.add_argument("--gpu-poll-interval-min", type=int, default=5,
                        help="Minutes between idle-GPU polls (default: 5)")
    parser.add_argument("--use-all-free-gpus", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    repo_root = args.repo_root.resolve()

    if not args.sif:
        raise SystemExit("FA4_SIF is not set — provide --sif or set the FA4_SIF env var.")
    print(f"Using SIF: {args.sif}")

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    idle_gpu_indices = wait_for_idle_gpus(
        max_used_memory_mb=args.max_used_memory_mb,
        poll_interval_s=args.gpu_poll_interval_min * 60,
        timeout_s=args.gpu_wait_timeout_min * 60,
        webhook_url=webhook_url,
    )
    test_visible_devices = select_visible_devices(idle_gpu_indices, args.use_all_free_gpus)
    benchmark_visible_devices = idle_gpu_indices[0]
    run_workers = args.run_workers or len(idle_gpu_indices)
    print(f"Idle GPUs: {idle_gpu_indices}")
    print(f"Running tests on: {test_visible_devices} ({run_workers} workers)")

    base_env = {**os.environ, "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1"}
    work_dir = os.environ.get("CI_WORK_DIR", f"/scratch/user/{os.environ.get('USER', 'user')}")

    for step in build_step_plan(
        test_target=args.test_target,
        test_filter=args.test_filter,
        compile_workers=args.compile_workers,
        run_workers=run_workers,
        test_visible_devices=test_visible_devices,
        benchmark_visible_devices=benchmark_visible_devices,
        skip_benchmark=args.skip_benchmark,
    ):
        run_step(step, repo_root=repo_root, base_env=base_env, sif=args.sif, work_dir=work_dir)

    print("=== All tests passed ===")


if __name__ == "__main__":
    main()
