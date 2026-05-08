#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash harness/harness/benchmark/export_sass.sh <before-cubin-or-so> <after-cubin-or-so>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$HARNESS_CODE_ROOT/logs/benchmark/sass"
BEFORE="$1"
AFTER="$2"

mkdir -p "$OUT_DIR"

dump_sass() {
    local input="$1"
    local output="$2"
    if command -v nvdisasm >/dev/null 2>&1; then
        nvdisasm "$input" > "$output"
    elif command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump --dump-sass "$input" > "$output"
    else
        echo "Neither nvdisasm nor cuobjdump is available" >&2
        return 1
    fi
}

dump_sass "$BEFORE" "$OUT_DIR/before.sass"
dump_sass "$AFTER" "$OUT_DIR/after.sass"
diff -u "$OUT_DIR/before.sass" "$OUT_DIR/after.sass" > "$OUT_DIR/sass.diff" || true
echo "[benchmark] SASS outputs written under $OUT_DIR"
