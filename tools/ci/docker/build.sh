#!/bin/bash
set -e

DATE=$(date +%y.%m.%d)
IMAGE_NAME="flash-attn-4:flash-attn-cu13.0-${DATE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building $IMAGE_NAME ..."
sudo docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
echo "Done: $IMAGE_NAME"
