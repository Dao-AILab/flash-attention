#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash harness/harness/hang_detect_fix/cuda_gdb_snapshot.sh <pid> [log]" >&2
    exit 2
fi

PID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG="${2:-$HARNESS_ROOT/logs/hang_detect_fix/gdb_hang.log}"

mkdir -p "$(dirname "$LOG")"

cuda-gdb --pid "$PID" \
    -batch \
    -ex "set logging file $LOG" \
    -ex "set logging on" \
    -ex "info cuda kernels" \
    -ex "info cuda warps" \
    -ex "info cuda threads sm 0" \
    -ex "x/8i \$pc" \
    -ex "bt" \
    -ex "set logging off"

echo "[hang_detect_fix] wrote $LOG"
