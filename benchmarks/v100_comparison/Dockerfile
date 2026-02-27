FROM flash_attn_v100_base

# xFormers: build from source against NGC PyTorch
# Record NGC torch version to verify it's not replaced
RUN python -c "import torch; open('/tmp/torch_ver.txt','w').write(torch.__version__)"

# Clone and build xFormers from source (pip install can still pull prebuilt wheels)
RUN git clone --depth 1 --branch v0.0.29.post2 \
        https://github.com/facebookresearch/xformers.git /tmp/xformers && \
    cd /tmp/xformers && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST="7.0" pip install --no-deps --no-build-isolation -e . && \
    cd / && rm -rf /tmp/xformers/.git

# Verify NGC torch was not replaced
RUN python -c "\
import torch; \
expected = open('/tmp/torch_ver.txt').read(); \
actual = torch.__version__; \
assert actual == expected, f'PyTorch was replaced: {expected} -> {actual}'; \
print(f'PyTorch OK: {actual}'); \
import flash_attn; print(f'flash_attn OK: {flash_attn.__version__}'); \
import xformers; print(f'xformers OK: {xformers.__version__}')"

COPY benchmarks/v100_comparison/ /workspace/flash-attention/benchmarks/v100_comparison/

WORKDIR /workspace/flash-attention/benchmarks/v100_comparison

# Build & Run:
#   docker build -t flash_attn_bench -f benchmarks/v100_comparison/Dockerfile .
#   docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py
