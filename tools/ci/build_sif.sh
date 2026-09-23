#!/usr/bin/env bash
# Build the FA4 Apptainer SIF image.
#
# Usage:
#   ./tools/ci/build_sif.sh [OUTPUT_PATH]
#
# Default output: /scratch/user/$USER/attention_fa4_<date>.sif
# All temp/cache dirs are redirected to /scratch to avoid filling the root volume.
#
# Example:
#   ./tools/ci/build_sif.sh
#   ./tools/ci/build_sif.sh ~/scratch/my_fa4.sif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="$SCRIPT_DIR/fa4.def"

DATE=$(date +%Y%m%d)
SCRATCH_BASE="${CI_WORK_DIR:-/scratch/user/${USER}}"
DEFAULT_OUT="${SCRATCH_BASE}/attention_fa4_${DATE}.sif"
OUTPUT="${1:-$DEFAULT_OUT}"
TMP_DIR="$SCRATCH_BASE/apptainer_tmp"
CACHE_DIR="$SCRATCH_BASE/apptainer_cache"

mkdir -p "$TMP_DIR" "$CACHE_DIR"

echo "=== FA4 SIF Build ==="
echo "  def file : $DEF_FILE"
echo "  output   : $OUTPUT"
echo "  tmp dir  : $TMP_DIR"
echo "  cache dir: $CACHE_DIR"
echo

sudo \
  APPTAINER_TMPDIR="$TMP_DIR" \
  APPTAINER_CACHEDIR="$CACHE_DIR" \
  apptainer build "$OUTPUT" "$DEF_FILE"

echo
echo "Build complete: $OUTPUT"
echo "File size: $(du -sh "$OUTPUT" | cut -f1)"
echo
echo "To use in CI, set on the runner:"
echo "  export FA4_SIF=$OUTPUT"
