#!/bin/bash
set -xe

# Default values
TAG='latest'
VERSION=''
MAX_JOBS=-1
GPU_ARCHS='native'
UNIT_TEST="false" # Default value

# Parse arguments in key=value format
for arg in "$@"; do
    case $arg in
        tag=*)
            TAG="${arg#*=}"
            shift
        ;;
        version=*)
            VERSION="${arg#*=}"
            shift
        ;;
        max-jobs=*)
            MAX_JOBS="${arg#*=}"
            shift
        ;;
        gpu-archs=*)
            GPU_ARCHS="${arg#*=}"
            shift
        ;;
        unit-test=*)
            UNIT_TEST="${arg#*=}"
            shift
        ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
    esac
done

# Check if unit-test is true and set FLASH_ATTENTION_INTERNAL_* variables
if [ "$UNIT_TEST" = "true" ]; then
    FLASH_ATTENTION_INTERNAL_USE_RTN=1
    FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE=1
    FLASH_ATTENTION_INTERNAL_DETERMINISTIC=1
fi

# Build arguments array
BUILD_ARGS=("--build-arg" "TAG=$TAG" "--build-arg" "VERSION=$VERSION")

IMAGE_NAME_TMP="tmp"
CONTAINER_NAME_TMP="tmp"

# Build the Docker image with the optional build arguments
docker build "${BUILD_ARGS[@]}" -f Dockerfile.rocm -t $IMAGE_NAME_TMP .

# Run the container with specific settings and environment variables
docker run --network host --ipc host --device /dev/dri --device /dev/kfd \
--cap-add SYS_PTRACE --group-add video --security-opt seccomp=unconfined \
--privileged --name $CONTAINER_NAME_TMP \
-e MAX_JOBS=$MAX_JOBS -e PYTORCH_ROCM_ARCH=$GPU_ARCHS \
-e FLASH_ATTENTION_INTERNAL_USE_RTN=$FLASH_ATTENTION_INTERNAL_USE_RTN \
$IMAGE_NAME_TMP /bin/bash -c "pip install ."

docker wait $CONTAINER_NAME_TMP

IMAGE_NAME="flash-attention"
CONTAINER_NAME="flash-attention"

docker commit $CONTAINER_NAME_TMP $IMAGE_NAME

docker rm $CONTAINER_NAME_TMP
docker image rm $IMAGE_NAME_TMP

# Run the final container with installed flash-attention
docker run -it --network host --ipc host --device /dev/dri --device /dev/kfd \
--cap-add SYS_PTRACE --group-add video --security-opt seccomp=unconfined \
--privileged --name $CONTAINER_NAME \
-e FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE=$FLASH_ATTENTION_INTERNAL_UNIT_TEST_MODE \
-e FLASH_ATTENTION_INTERNAL_DETERMINISTIC=$FLASH_ATTENTION_INTERNAL_DETERMINISTIC \
$IMAGE_NAME /bin/bash