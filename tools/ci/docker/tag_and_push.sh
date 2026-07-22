#!/bin/bash
set -e

DATE=$(date +%y.%m.%d)
LOCAL_TAG="flash-attn-4:flash-attn-cu13.0-${DATE}"
REMOTE_TAG="togethercomputer/training-performance:flash-attn-cu13.0-${DATE}"

docker tag "$LOCAL_TAG" "$REMOTE_TAG"
docker push "$REMOTE_TAG"
