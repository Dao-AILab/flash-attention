#!/bin/bash
set -e

DATE=$(date +%y.%m.%d)
IMAGE_NAME="flash-attn-4:flash-attn-cu13.0-${DATE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

echo "Building $IMAGE_NAME ..."
sudo docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
echo "Done: $IMAGE_NAME"
