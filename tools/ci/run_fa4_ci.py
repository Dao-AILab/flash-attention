#!/usr/bin/env python3
"""FA4 CI driver — runs inside an Apptainer SIF on a self-hosted GPU runner.

Requires FA4_SIF (path to the .sif image) to be set, either via env var or --sif.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TEST_FILTER = ""  # empty = run all; CI overrides via --test-filter
DEFAULT_TEST_TARGETS = ["tests/cute/test_flash_attn.py", "tests/cute/test_flash_attn_dim512.py"]


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


# ── Runtime DSL pin (decouples cutlass-dsl from the baked image) ─────────────────

def read_dep_spec(pyproject_path: Path, name: str) -> str:
    """Read a dependency's version specifier (e.g. '==4.6.0.dev0', '>=0.5.0') from pyproject.

    Regex (not tomllib) on purpose: this runs on the HOST python (3.10 on the self-hosted runner),
    which may not have a TOML parser or `packaging`. Any `[extras]` between the name and the
    specifier are ignored — the caller re-adds the extra it needs.
    """
    text = pyproject_path.read_text()
    m = re.search(rf"{re.escape(name)}(?:\[[^\]]*\])?\s*([=<>!~][^\"'\],]*)", text)
    if not m:
        raise SystemExit(f"Could not find a version specifier for `{name}` in {pyproject_path}")
    return m.group(1).strip()


def read_cuda_major() -> int:
    """Driver's max CUDA major from nvidia-smi header (picks the cutlass-dsl libs variant)."""
    out = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\s*(\d+)\.", out)
    if not m:
        raise SystemExit("Could not parse 'CUDA Version: X.Y' from nvidia-smi output")
    return int(m.group(1))


# ── Step plan ─────────────────────────────────────────────────────────────────

def build_step_plan(
    test_target: str | list[str],
    test_filter: str,
    compile_workers: int,
    run_workers: int,
    test_visible_devices: str,
    benchmark_visible_devices: str,
    skip_benchmark: bool,
) -> list[Step]:
    test_targets = [test_target] if isinstance(test_target, str) else test_target
    pytest_base = ["python3", "-m", "pytest", *test_targets, *(["-k", test_filter] if test_filter else [])]

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

def prepare_overlay(
    repo_root: Path,
    base_env: dict[str, str],
    sif: str,
    work_dir: str,
    overlay: str,
    cutlass_spec: str,
    quack_spec: str,
    dsl_variant: str,
) -> None:
    """Provision the disk-backed overlay once: install the DSL stack + FA4, then floor-check.

    This is what decouples the fast-moving DSL stack from the baked image — CI installs the versions
    declared in flash_attn/cute/pyproject.toml at runtime, so a floor bump no longer needs an image
    rebake. nvidia-cutlass-dsl and quack-kernels are COUPLED (quack annotates cutlass internals like
    cute.core.ThrMma at import), so they must be installed together at compatible versions; uv's
    joint resolve picks the quack that matches the pinned cutlass (exactly what the image build does).

    Why this shape:
    - The baked cutlass-dsl ships a `.pth` that adds a vendored `nvidia_cutlass_dsl/python_packages`
      tree to sys.path. A plain `uv pip install` over it leaves stale files/.pyc in that tree and
      silently mixes versions (symptoms: cute.core missing `ThrMma`, an old libs `fmax` signature).
      So we delete the baked cutlass + quack trees outright, then install into a clean slate.
    - The install goes into the image's real site-packages (not a PYTHONPATH shim, which would leave
      the baked copy co-resident on the `nvidia.*` namespace and re-introduce the mix).
    - That needs a writable, roomy layer: a DISK-backed --overlay (created in main()), not the
      RAM-backed --writable-tmpfs, which is too small and ENOSPCs on a DSL reinstall.
    - --prerelease=allow: the cutlass pin is often a dev build (e.g. 4.6.0.dev0) with transitive
      pre-releases that uv otherwise refuses.
    """
    print(f"=== Provision overlay: cutlass-dsl[{dsl_variant}]{cutlass_spec} + quack-kernels{quack_spec} + FA4 ===")
    site_packages = "SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')"
    nuke_baked_dsl = (
        'rm -rf "$SP"/nvidia_cutlass_dsl* "$SP"/nvidia/cutlass_dsl* '
        '"$SP"/quack "$SP"/quack_kernels* 2>/dev/null || true'
    )
    uv_cache_export = f"export UV_CACHE_DIR={shlex.quote(os.path.join(work_dir, 'uv_cache'))}"
    dsl_install_cmd = (
        f"uv pip install --system --break-system-packages --prerelease=allow -q "
        f"'nvidia-cutlass-dsl[{dsl_variant}]{cutlass_spec}' 'quack-kernels{quack_spec}'"
    )
    # Install FA4 from the current repo. --no-deps keeps the SIF's baked torch/cudnn (and the
    # runtime DSL stack installed above).
    fa4_install_cmd = f"uv pip install --system --break-system-packages --no-deps -q -e {shlex.quote(str(repo_root / 'flash_attn/cute'))}"
    # Sanity-check the importable deps satisfy the pyproject floors (verifies the runtime install
    # took effect; for any image-backed dep it still catches a stale SIF below the floor).
    floor_check_cmd = (
        f"python3 {shlex.quote(str(repo_root / 'tools/ci/assert_dsl_floor.py'))} "
        f"{shlex.quote(str(repo_root / 'flash_attn/cute/pyproject.toml'))}"
    )
    # sync: flush the overlay image before this session tears down (see wait_for_overlay_release).
    parts = [uv_cache_export, site_packages, nuke_baked_dsl, dsl_install_cmd, fa4_install_cmd, floor_check_cmd, "sync"]
    cmd = ["apptainer", "exec", "--nv", "--overlay", overlay, "--bind", work_dir, sif, "bash", "-c", " && ".join(parts)]
    subprocess.run(cmd, check=True, cwd=repo_root, env=base_env)


# ── Overlay hand-off between apptainer sessions ────────────────────────────────────────────────
# `apptainer exec` returns before its fuse2fs child has finished flushing the ext3 overlay image, so a
# session opened right after may see the image without the previous session's whiteouts: the SIF-baked
# packages that prepare_overlay() deleted reappear next to the installed ones. Seen 2026-09-16: the
# baked cutlass-dsl 4.6.0 `lib/` runtime was linked instead of 4.8.0 `cu13/lib/` ("Symbols not found").
# Defence: wait for the image to be released, verify the stack from a fresh session (re-provision
# once), and pin the verified runtime via CUTE_DSL_LIBS.

def _pids_holding(path: str) -> list[int]:
    """PIDs with `path` open (the fuse2fs of a live session)."""
    holders = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target == path or target == f"{path} (deleted)":
                holders.append(int(pid))
                break
    return holders


def wait_for_overlay_release(overlay: str, timeout_s: float = 60.0) -> float:
    """Block until no process holds the overlay image open; return the seconds waited.

    Fails after `timeout_s`: opening another session on a still-held image is exactly the
    stale-view hazard this guards against, so continuing would defeat the check.
    """
    real = os.path.realpath(overlay)
    start = time.monotonic()
    while True:
        holders = _pids_holding(real)
        waited = time.monotonic() - start
        if not holders:
            if waited > 0.5:
                print(f"(overlay hand-off: waited {waited:.1f}s for the previous session to release {overlay})")
            return waited
        if waited > timeout_s:
            raise RuntimeError(f"{overlay} still held by pid(s) {holders} after {timeout_s:.0f}s")
        time.sleep(0.2)


# Runs in a fresh session after provisioning; prints "CUTE_DSL_LIBS=<paths>" on success.
_VERIFY_OVERLAY_PY = r"""
import importlib.metadata as md, os, sys
repo = os.path.realpath(sys.argv[1])
problems = []

def dists():
    for d in md.distributions():
        yield (d.metadata.get("Name") or "").lower().replace("_", "-"), d

# 1. One visible distribution per runtime package (two = the SIF-baked copy leaked through).
seen = {}
for name, d in dists():
    if name.startswith("nvidia-cutlass-dsl") or name in ("quack-kernels", "flash-attn-4"):
        seen.setdefault(name, []).append(f"{d.version} @ {d.locate_file('')}")
for name, versions in seen.items():
    if len(versions) > 1:
        problems.append(f"{name}: {len(versions)} distributions visible: {versions}")

# 2. FA4 imports from the checkout, not from a baked copy.
import flash_attn.cute.interface as iface
if not os.path.realpath(iface.__file__).startswith(repo):
    problems.append(f"flash_attn.cute imports from {iface.__file__}, expected under {repo}")

# 3. The runtime the JIT will link belongs to the installed cutlass-dsl distribution.
import cutlass.cutlass_dsl as cd
libs = cd.CuTeDSL._get_dsl().get_shared_libs()
runtimes = [l for l in libs if os.path.basename(l).startswith("libcute_dsl_runtime")]
owned = set()
for name, d in dists():
    if name.startswith("nvidia-cutlass-dsl"):
        for f in d.files or []:
            owned.add(os.path.realpath(str(d.locate_file(f))))
if not runtimes:
    problems.append(f"no libcute_dsl_runtime in get_shared_libs(): {libs}")
for l in runtimes:
    if os.path.realpath(l) not in owned:
        problems.append(f"runtime {l} is not part of the installed cutlass-dsl distribution (stale SIF-baked copy?)")

if problems:
    print("OVERLAY VERIFY FAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print(f"overlay verify OK: FA4 from {os.path.dirname(iface.__file__)}, runtime {runtimes[0]}")
print("CUTE_DSL_LIBS=" + os.pathsep.join(libs))
"""


def verify_overlay(repo_root: Path, base_env: dict[str, str], sif: str, work_dir: str, overlay: str) -> str | None:
    """Check the provisioned overlay from a fresh session; return CUTE_DSL_LIBS to pin, or None."""
    print("=== Verify overlay (fresh session) ===")
    # Re-run the version floor check here too: in the provisioning session it sees the in-memory
    # overlay state, which is always right; this session sees what the tests will see.
    floor_check = (
        f"python3 {shlex.quote(str(repo_root / 'tools/ci/assert_dsl_floor.py'))} "
        f"{shlex.quote(str(repo_root / 'flash_attn/cute/pyproject.toml'))}"
    )
    inner = f"cd /tmp && {floor_check} && python3 -c {shlex.quote(_VERIFY_OVERLAY_PY)} {shlex.quote(str(repo_root))}"
    cmd = ["apptainer", "exec", "--overlay", overlay, "--bind", work_dir, sif, "bash", "-c", inner]
    proc = subprocess.run(cmd, cwd=repo_root, env=base_env, text=True, capture_output=True)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr[-2000:], end="")
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("CUTE_DSL_LIBS="):
            return line[len("CUTE_DSL_LIBS="):]
    return None


def provision_verified_overlay(
    repo_root: Path, base_env: dict[str, str], sif: str, work_dir: str, overlay: str,
    cutlass_spec: str, quack_spec: str, dsl_variant: str,
) -> str:
    """Create, provision and verify the overlay; re-provision once if verification fails."""
    for attempt in (1, 2):
        if os.path.exists(overlay):
            os.remove(overlay)
        subprocess.run(["apptainer", "overlay", "create", "--size", "4096", overlay], check=True)
        prepare_overlay(
            repo_root=repo_root, base_env=base_env, sif=sif, work_dir=work_dir, overlay=overlay,
            cutlass_spec=cutlass_spec, quack_spec=quack_spec, dsl_variant=dsl_variant,
        )
        wait_for_overlay_release(overlay)
        dsl_libs = verify_overlay(repo_root=repo_root, base_env=base_env, sif=sif, work_dir=work_dir, overlay=overlay)
        wait_for_overlay_release(overlay)
        if dsl_libs:
            return dsl_libs
        if attempt == 1:
            print("Overlay verification failed — recreating the overlay and provisioning again.")
    raise SystemExit("Overlay verification failed twice; see the OVERLAY VERIFY FAILED lines above.")


def run_step(step: Step, repo_root: Path, base_env: dict[str, str], sif: str, work_dir: str, overlay: str, dsl_libs: str | None = None) -> None:
    print(f"=== {step.name} ===")
    # Convert relative test/benchmark paths to absolute so we can run from /tmp.
    # Running from /tmp ensures Python does not insert repo_root into sys.path[0]
    # (which would cause flash_attn/__init__.py to trigger FA2 imports unavailable in the SIF).
    command = [
        str(repo_root / arg) if (arg.startswith("tests/") or arg.startswith("benchmarks/")) else arg
        for arg in step.command
    ]
    extra_env = dict(step.extra_env)
    if dsl_libs:
        # Pin the runtime verified by verify_overlay() so JIT linking does not depend on discovery.
        extra_env.setdefault("CUTE_DSL_LIBS", dsl_libs)
    env_exports = " && ".join(f"export {k}={shlex.quote(v)}" for k, v in extra_env.items())
    inner_cmd = shlex.join(command)
    shell_parts = [env_exports] if env_exports else []
    shell_parts.append(f"cd /tmp && {inner_cmd}")
    cmd = ["apptainer", "exec", "--nv", "--overlay", overlay, "--bind", work_dir, sif, "bash", "-c", " && ".join(shell_parts)]
    subprocess.run(cmd, check=True, cwd=repo_root, env=base_env)
    wait_for_overlay_release(overlay)


# ── CLI ───────────────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sif", default=os.environ.get("FA4_SIF", ""),
                        help="Apptainer .sif image path (or set FA4_SIF env var)")
    parser.add_argument("--test-target", nargs="+", default=DEFAULT_TEST_TARGETS,
                        help="One or more pytest file paths or node IDs")
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

    # Runtime DSL versions, read straight from pyproject (single source of truth) and installed into
    # a fresh disk-backed overlay that replaces the SIF's baked versions. Recreate the overlay each
    # run so a changed pin can't inherit a stale install from a previous job.
    pyproject = repo_root / "flash_attn/cute/pyproject.toml"
    cutlass_spec = read_dep_spec(pyproject, "nvidia-cutlass-dsl")
    quack_spec = read_dep_spec(pyproject, "quack-kernels")
    dsl_variant = "cu13" if read_cuda_major() >= 13 else "cu12"
    overlay = os.path.join(work_dir, "fa4_ci_overlay.img")
    os.makedirs(work_dir, exist_ok=True)
    print(f"Runtime DSL: cutlass-dsl[{dsl_variant}]{cutlass_spec} + quack-kernels{quack_spec} (into {overlay})")

    try:
        dsl_libs = provision_verified_overlay(
            repo_root=repo_root, base_env=base_env, sif=args.sif, work_dir=work_dir,
            overlay=overlay, cutlass_spec=cutlass_spec, quack_spec=quack_spec, dsl_variant=dsl_variant,
        )
        for step in build_step_plan(
            test_target=args.test_target,
            test_filter=args.test_filter,
            compile_workers=args.compile_workers,
            run_workers=args.run_workers,
            test_visible_devices=test_visible_devices,
            benchmark_visible_devices=benchmark_visible_devices,
            skip_benchmark=args.skip_benchmark,
        ):
            run_step(step, repo_root=repo_root, base_env=base_env, sif=args.sif, work_dir=work_dir, overlay=overlay, dsl_libs=dsl_libs)
    finally:
        # The overlay can hold gigabytes; don't leave it behind on the runner between jobs.
        if os.path.exists(overlay):
            os.remove(overlay)

    print("=== All tests passed ===")


if __name__ == "__main__":
    main()
