#!/bin/bash
set -e

DATE=$(date +%y.%m.%d)
LOCAL_TAG="flash-attn-4:flash-attn-cu12.9-${DATE}"
REMOTE_TAG="togethercomputer/training-performance:flash-attn-cu12.9-${DATE}"

docker tag "$LOCAL_TAG" "$REMOTE_TAG"
docker push "$REMOTE_TAG"
