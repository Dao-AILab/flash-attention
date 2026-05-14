#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_CODE_ROOT/.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"
LOG_ROOT="$HARNESS_CODE_ROOT/logs/benchmark"
CURRENT_DIR="$LOG_ROOT/current"
PREVIOUS_DIR="$LOG_ROOT/previous"
RUNS="${BENCHMARK_RUNS:-3}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

CMD=(
    python3
    "$REPO/harness/harness/benchmark/bench_sm100_hd256.py"
    --compare-baseline
    --nheads 16
    --nheads-kv 16
    --rep 50
    --warmup 10
)

mkdir -p "$LOG_ROOT"

echo "[benchmark] repo=$REPO"
echo "[benchmark] logs=$LOG_ROOT"
echo "[benchmark] runs=$RUNS"
echo "[benchmark] command=${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

rm -rf "$PREVIOUS_DIR"
if [[ -d "$CURRENT_DIR" ]]; then
    mv "$CURRENT_DIR" "$PREVIOUS_DIR"
fi
mkdir -p "$CURRENT_DIR"

(
    cd /tmp
    REPO="$REPO" python3 - <<'PY'
import hashlib
import importlib.metadata as md
import os
import subprocess
import sys
from pathlib import Path
import flash_attn.cute.interface as interface
import flash_attn.cute.pipeline as pipeline
import flash_attn.cute.flash_fwd as flash_fwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward as bwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel as dkdv
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel as dq

repo = Path(os.environ["REPO"]).resolve()
interface_path = Path(interface.__file__).resolve()
expected = repo / "flash_attn" / "cute" / "interface.py"
print(f"[benchmark] flash_attn.cute.interface={interface_path}")
if interface_path != expected:
    raise SystemExit(f"benchmark must use repo-local CuteDSL interface: expected {expected}")

print(f"[benchmark] repo={repo}")
print(f"[benchmark] cwd={Path.cwd()}")
print(f"[benchmark] sys.path[0]={sys.path[0]!r}")
for pkg in [
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "quack-kernels",
]:
    try:
        print(f"[benchmark] package {pkg}={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"[benchmark] package {pkg}=NOT_INSTALLED")

try:
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
except Exception as exc:
    git_head = f"UNKNOWN:{exc}"
print(f"[benchmark] git_head={git_head}")

def stamp(label: str, path: Path) -> None:
    path = path.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    stat = path.stat()
    under_repo = repo in path.parents or path == repo
    print(
        f"[source] {label} path={path} under_repo={under_repo} "
        f"sha256={digest} mtime_ns={stat.st_mtime_ns} size={stat.st_size}"
    )

for label, module in [
    ("interface", interface),
    ("pipeline", pipeline),
    ("flash_fwd", flash_fwd),
    ("sm100_hd256_bwd", bwd),
    ("sm100_hd256_dkdv", dkdv),
    ("sm100_hd256_dq", dq),
]:
    stamp(label, Path(module.__file__))
PY
    for i in $(seq 1 "$RUNS"); do
        log="$CURRENT_DIR/run_${i}.log"
        echo "[benchmark] START run $i -> $log"
        "${CMD[@]}" 2>&1 | tee "$log"
        echo "[benchmark] DONE run $i"
    done
)

python3 "$SCRIPT_DIR/compare_benchmark.py" \
    --current "$CURRENT_DIR" \
    --previous "$PREVIOUS_DIR" \
    --report "$CURRENT_DIR/benchmark_report.md"
