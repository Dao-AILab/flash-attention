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
    harness/harness/benchmark/bench_sm100_hd256.py
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
    cd "$REPO"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    python3 - <<'PY'
from pathlib import Path
import flash_attn.cute.interface as interface

repo = Path.cwd().resolve()
interface_path = Path(interface.__file__).resolve()
expected = repo / "flash_attn" / "cute" / "interface.py"
print(f"[benchmark] flash_attn.cute.interface={interface_path}")
if interface_path != expected:
    raise SystemExit(f"benchmark must use repo-local CuteDSL interface: expected {expected}")
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
