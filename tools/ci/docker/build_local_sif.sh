#!/usr/bin/env bash
# Build the FA4 CI container on the runner and convert it to a SIF — no registry involved.
#
#   tools/ci/docker/build_local_sif.sh [OUTPUT_DIR]      # default OUTPUT_DIR = $CI_WORK_DIR
#
# Output: OUTPUT_DIR/local-sif/flash-attn-4-cu13.0-<date>-<sha>.sif — publish that path as the repo
# variable FA4_LOCAL_SIF. Keep it in local-sif/: registry mode prunes *.sif directly under CI_WORK_DIR.
# Rebuild only when the Dockerfile changes; the DSL stack is installed at job time by run_fa4_ci.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
OUT_DIR="${1:-${CI_WORK_DIR:-/scratch/user/$USER}}"
DATE=$(date +%y.%m.%d)
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
IMAGE="flash-attn-4:flash-attn-cu13.0-${DATE}"
SIF_DIR="${OUT_DIR}/local-sif"
SIF="${SIF_DIR}/flash-attn-4-cu13.0-${DATE}-${GIT_SHA}.sif"
TMP_DIR="${APPTAINER_TMPDIR:-/scratch/apptainer_tmp}"

mkdir -p "$SIF_DIR"
sudo mkdir -p "$TMP_DIR"

echo "=== docker build $IMAGE (repo @ $GIT_SHA) ==="
sudo docker build -t "$IMAGE" --label "org.flash_attn.ci.git_sha=$GIT_SHA" \
  -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"

echo "=== apptainer build $SIF ==="
rm -f "$SIF"
sudo APPTAINER_TMPDIR="$TMP_DIR" apptainer build "$SIF" "docker-daemon://$IMAGE"
sudo chown "$(id -u):$(id -g)" "$SIF"

echo
echo "SIF ready: $SIF ($(du -sh "$SIF" | cut -f1))"
echo "Publish it to CI (repo Settings → Variables, or):"
echo "  gh variable set FA4_LOCAL_SIF -R Dao-AILab/flash-attention --body '$SIF'"
echo "Local run against it:"
echo "  FA4_SIF=$SIF CI_WORK_DIR=$OUT_DIR python3 tools/ci/run_fa4_ci.py --repo-root $REPO_ROOT --test-filter '<filter>'"
