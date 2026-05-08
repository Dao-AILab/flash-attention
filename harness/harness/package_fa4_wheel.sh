#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${HARNESS_ROOT}/.." && pwd)"
CUTE_DIR="${REPO_ROOT}/flash_attn/cute"
DIST_DIR="${HARNESS_ROOT}/dist"

cd "${REPO_ROOT}"

if [[ ! -f "${CUTE_DIR}/pyproject.toml" ]]; then
  echo "[wheel] missing ${CUTE_DIR}/pyproject.toml" >&2
  exit 1
fi

cleanup_source_artifacts() {
  rm -rf "${CUTE_DIR}/build" "${CUTE_DIR}/flash_attn_4.egg-info"
  rmdir "${CUTE_DIR}/cute" >/dev/null 2>&1 || true
}

trap cleanup_source_artifacts EXIT

mkdir -p "${DIST_DIR}"
rm -f "${DIST_DIR}"/*.whl
cleanup_source_artifacts

echo "[wheel] repo=${REPO_ROOT}"
echo "[wheel] source=${CUTE_DIR}"
echo "[wheel] dist=${DIST_DIR}"

if python3 -c 'import build' >/dev/null 2>&1; then
  python3 -m build --wheel --outdir "${DIST_DIR}" "${CUTE_DIR}"
else
  python3 -m pip wheel --no-deps --wheel-dir "${DIST_DIR}" "${CUTE_DIR}"
fi

shopt -s nullglob
wheels=("${DIST_DIR}"/flash_attn_4-*.whl)
shopt -u nullglob

if (( ${#wheels[@]} != 1 )); then
  echo "[wheel] expected exactly one flash_attn_4 wheel, found ${#wheels[@]}" >&2
  ls -la "${DIST_DIR}" >&2
  exit 1
fi

echo "[wheel] built ${wheels[0]}"
