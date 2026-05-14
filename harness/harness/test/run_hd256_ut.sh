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
PYTEST_IMPORT_ARGS=(--import-mode=importlib --rootdir="$REPO")

if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=1
fi

mkdir -p "$LOGDIR"

run_preflight() {
    (cd /tmp && REPO="$REPO" python3 - <<'PY'
import hashlib
import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path

import flash_attn
import flash_attn.cute.interface as interface
import flash_attn.cute.pipeline as pipeline
import flash_attn.cute.flash_fwd as flash_fwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward as bwd
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel as dkdv
import flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel as dq

print(f"[preflight] flash_attn={flash_attn.__file__}")
print(f"[preflight] flash_attn.__path__={list(getattr(flash_attn, '__path__', []))}")
print(f"[preflight] flash_attn.cute.flash_fwd={flash_fwd.__file__}")

repo = Path(os.environ["REPO"])
print(f"[preflight] repo={repo}")
print(f"[preflight] cwd={Path.cwd()}")
print(f"[preflight] sys.path[0]={sys.path[0]!r}")
for pkg in [
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "quack-kernels",
]:
    try:
        print(f"[preflight] package {pkg}={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"[preflight] package {pkg}=NOT_INSTALLED")

try:
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
except Exception as exc:
    git_head = f"UNKNOWN:{exc}"
print(f"[preflight] git_head={git_head}")

try:
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"],
        text=True,
    ).splitlines()
except Exception as exc:
    dirty = [f"UNKNOWN:{exc}"]
print(f"[preflight] git_dirty_count={len(dirty)}")
for line in dirty[:40]:
    print(f"[preflight] git_dirty {line}")

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

targets = [
    (repo / "tests/cute/test_flash_attn.py", "test_flash_attn_output", "d"),
    (repo / "tests/cute/test_flash_attn.py", "test_flash_attn_varlen_output", "d"),
    (repo / "tests/cute/test_flash_attn_varlen.py", "test_varlen", "D"),
]
for label, path in [
    ("test_flash_attn", targets[0][0]),
    ("test_flash_attn_varlen", targets[2][0]),
]:
    stamp(label, path)

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
if cd /tmp && python3 -m pytest --help 2>/dev/null | grep -q -- "--numprocesses"; then
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
    if cd /tmp && python3 -m pytest -v -s "${PYTEST_IMPORT_ARGS[@]}" "${PYTEST_XDIST_ARGS[@]}" --tb=long "$@" > "$logfile" 2>&1; then
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
