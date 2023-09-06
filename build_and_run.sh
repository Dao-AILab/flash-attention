#! /bin/bash

echo "Build docker image"
docker build -t triton_fa_rocm .

echo "Start container"
docker run -it -d --device=/dev/kfd --device=/dev/dri --group-add video --name triton_fa_benchmark triton_fa_rocm

echo "Benchmarking flash attention forward kernel with 1 GCD"
docker exec --workdir /workspace/flash-attention/ triton_fa_benchmark python3 benchmarks/benchmark_flash_attention_forward.py

echo "Benchmarking flash attention forward kernel with 2 GCDs"
docker exec --workdir /workspace/triton/ triton_fa_benchmark ./scripts/amd/run_2gcd.sh fwd
