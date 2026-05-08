#!/usr/bin/env bash
# Run the three standard CuteDSL head_dim=256 UT groups sequentially.
# Usage:
#   bash harness/harness/test/run_hd256_ut.sh
#   bash harness/harness/test/run_hd256_ut.sh --preflight-only
#
# Logs:
#   harness/logs/test/preflight.log
#   harness/logs/test/ut_hd256_output.log
#   harness/logs/test/ut_hd256_varlen_output.log
#   harness/logs/test/ut_varlen.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"
LOGDIR="$HARNESS_ROOT/logs/test"
TEST_FILE="$REPO/tests/cute/test_flash_attn.py"
VARLEN_TEST_FILE="$REPO/tests/cute/test_flash_attn_varlen.py"
PREFLIGHT_ONLY=0

if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=1
fi

mkdir -p "$LOGDIR"

run_preflight() {
    (cd "$REPO" && python3 - <<'PY'
import re
from pathlib import Path

import flash_attn
import flash_attn.cute.flash_fwd as flash_fwd

print(f"[preflight] flash_attn={flash_attn.__file__}")
print(f"[preflight] flash_attn.cute.flash_fwd={flash_fwd.__file__}")

targets = [
    (Path("tests/cute/test_flash_attn.py"), "test_flash_attn_output", "d"),
    (Path("tests/cute/test_flash_attn.py"), "test_flash_attn_varlen_output", "d"),
    (Path("tests/cute/test_flash_attn_varlen.py"), "test_varlen", "D"),
]

def ensure_hd256_only(path: Path, func_name: str, param_name: str) -> bool:
    text = path.read_text()
    marker = f"def {func_name}("
    def_index = text.find(marker)
    if def_index < 0:
        raise RuntimeError(f"{path}:{func_name} not found")

    pattern = re.compile(
        rf'(?m)^@pytest\.mark\.parametrize\(\s*(["\']){re.escape(param_name)}\1\s*,\s*\[[^\]]*\]\s*\)'
    )
    matches = [m for m in pattern.finditer(text) if m.end() < def_index]
    if not matches:
        raise RuntimeError(f"{path}:{func_name} has no active parametrize for {param_name!r}")

    match = matches[-1]
    current = match.group(0)
    desired = f'@pytest.mark.parametrize("{param_name}", [256])'
    if re.sub(r"\s+", "", current) == re.sub(r"\s+", "", desired):
        print(f"[preflight] {path}:{func_name}:{param_name} already [256]")
        return False

    updated = text[:match.start()] + desired + text[match.end():]
    path.write_text(updated)
    print(f"[preflight] patched {path}:{func_name}:{param_name} to [256]")
    return True

changed = False
for item in targets:
    changed = ensure_hd256_only(*item) or changed

if changed:
    print("[preflight] head_dim parametrization was temporarily restricted to [256]")
else:
    print("[preflight] head_dim parametrization already restricted to [256]")
PY
    )
}

if ! run_preflight > "$LOGDIR/preflight.log" 2>&1; then
    cat "$LOGDIR/preflight.log"
    exit 1
fi
cat "$LOGDIR/preflight.log"

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    echo "[preflight] preflight-only mode; UT execution skipped"
    exit 0
fi

PYTEST_XDIST_ARGS=()
if cd "$REPO" && python3 -m pytest --help 2>/dev/null | grep -q -- "--numprocesses"; then
    PYTEST_XDIST_ARGS=(-n 0)
fi

pass=0
fail=0
results=()

run_test_group() {
    local name="$1"
    local logfile="$2"
    shift 2
    echo "[$(date '+%H:%M:%S')] START  $name -> $logfile"
    if cd "$REPO" && python3 -m pytest -v -s "${PYTEST_XDIST_ARGS[@]}" --tb=long "$@" > "$logfile" 2>&1; then
        results+=("PASS  $name")
        ((pass++))
    else
        results+=("FAIL  $name")
        ((fail++))
    fi
    echo "[$(date '+%H:%M:%S')] DONE   $name  ($(tail -1 "$logfile"))"
}

run_test_group "hd256 output" "$LOGDIR/ut_hd256_output.log" \
    "$TEST_FILE::test_flash_attn_output"

run_test_group "hd256 varlen output" "$LOGDIR/ut_hd256_varlen_output.log" \
    "$TEST_FILE::test_flash_attn_varlen_output"

run_test_group "varlen" "$LOGDIR/ut_varlen.log" \
    "$VARLEN_TEST_FILE::test_varlen"

echo ""
echo "=============================="
echo "  SUMMARY: $pass passed, $fail failed"
echo "=============================="
for r in "${results[@]}"; do
    echo "  $r"
done

exit "$fail"
