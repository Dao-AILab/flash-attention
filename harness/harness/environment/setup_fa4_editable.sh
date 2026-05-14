#!/usr/bin/env bash
# Reset and install the repo-local CuteDSL FA4 editable runtime.
#
# Usage:
#   bash harness/harness/environment/setup_fa4_editable.sh
#   bash harness/harness/environment/setup_fa4_editable.sh --verify-only
#   bash harness/harness/environment/setup_fa4_editable.sh --skip-interface
#
# The default verification imports flash_attn.cute.interface. Use
# --skip-interface only when validating installation while current source has
# a known syntax/import error unrelated to package setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$HARNESS_ROOT/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VERIFY_INTERFACE=1
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-interface)
            VERIFY_INTERFACE=0
            ;;
        --verify-only)
            VERIFY_ONLY=1
            ;;
        *)
            echo "usage: $0 [--verify-only] [--skip-interface]" >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$VERIFY_ONLY" == "1" && "$VERIFY_INTERFACE" == "0" ]]; then
    echo "--verify-only cannot be combined with --skip-interface" >&2
    exit 2
fi

run() {
    echo "+ $*"
    "$@"
}

echo "[env] repo=$REPO"
echo "[env] python=$($PYTHON -c 'import sys; print(sys.executable)')"

PACKAGES=(
    flash-attn-4
    nvidia-cutlass-dsl
    nvidia-cutlass-dsl-libs-base
    quack-kernels
    apache-tvm-ffi
    torch-c-dlpack-ext
)

if [[ "$VERIFY_ONLY" == "0" ]]; then
    echo "[env] uninstalling stale CuteDSL/FA4 components"
    run "$PYTHON" -m pip uninstall -y "${PACKAGES[@]}"

    echo "[env] installing pinned CuteDSL runtime components without dependency drift"
    run "$PYTHON" -m pip install --no-deps \
        "nvidia-cutlass-dsl==4.4.2" \
        "nvidia-cutlass-dsl-libs-base==4.4.2" \
        "apache-tvm-ffi>=0.1.5,<0.2" \
        "torch-c-dlpack-ext==0.1.5" \
        "quack-kernels==0.4.1"

    echo "[env] installing repo-local flash_attn/cute editable package"
    run "$PYTHON" -m pip install --no-deps -e "$REPO/flash_attn/cute"
else
    echo "[env] verify-only: preserving existing installed packages"
fi

echo "[env] verifying installation"
(
    cd /tmp
    VERIFY_INTERFACE="$VERIFY_INTERFACE" REPO="$REPO" "$PYTHON" - <<'PY'
import importlib
import os
import site
from pathlib import Path

repo = Path(os.environ["REPO"]).resolve()
verify_interface = os.environ["VERIFY_INTERFACE"] == "1"

site_dirs = [Path(p) for p in site.getsitepackages()]
pth_files = [p / "nvidia_cutlass_dsl.pth" for p in site_dirs]
existing_pth = [p for p in pth_files if p.exists()]
if not existing_pth:
    raise SystemExit("nvidia_cutlass_dsl.pth was not installed")
print(f"[verify] cutlass pth={existing_pth[0]}")

flash_attn = importlib.import_module("flash_attn")
print(f"[verify] flash_attn_file={getattr(flash_attn, '__file__', None)}")
print(f"[verify] flash_attn_path={list(getattr(flash_attn, '__path__', []))}")

cutlass = importlib.import_module("cutlass")
from cutlass import Float32, Int32  # noqa: F401
print(f"[verify] cutlass={cutlass.__file__}")

quack = importlib.import_module("quack")
print(f"[verify] quack={quack.__file__}")

if verify_interface:
    interface = importlib.import_module("flash_attn.cute.interface")
    interface_path = Path(interface.__file__).resolve()
    print(f"[verify] interface={interface_path}")
    if repo not in interface_path.parents:
        raise SystemExit(f"interface is not repo-local: {interface_path}")
else:
    print("[verify] interface import skipped")
PY
)

echo "[env] setup complete"
