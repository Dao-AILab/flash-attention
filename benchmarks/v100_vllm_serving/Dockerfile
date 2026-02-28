# Use flash_attn_bench as base (already has NGC PyTorch + flash-attn SM70 + xFormers).
# This avoids the ~30min xFormers source build.
# To rebuild from scratch, change to flash_attn_v100_base and uncomment xFormers block below.
FROM flash_attn_bench

# Save versions for verification
RUN python -c "import torch; open('/tmp/torch_ver.txt','w').write(torch.__version__)"

# --- (Optional) xFormers from source, only if using flash_attn_v100_base ---
# RUN git clone --depth 1 --branch v0.0.29.post2 \
#         https://github.com/facebookresearch/xformers.git /tmp/xformers && \
#     cd /tmp/xformers && \
#     git submodule update --init --recursive && \
#     TORCH_CUDA_ARCH_LIST="7.0" pip install --no-deps --no-build-isolation -e . && \
#     cd / && rm -rf /tmp/xformers/.git

# --- vLLM v0.6.5 dependencies (excluding torch, flash-attn, xformers, triton) ---
# Install deps first, then vLLM with --no-deps to preserve our CUDA stack.
RUN pip install --no-cache-dir \
    psutil sentencepiece "requests>=2.26.0" tqdm py-cpuinfo \
    "transformers>=4.45.0,<4.48.0" "tokenizers>=0.19.1,<0.22" protobuf \
    "fastapi>=0.107.0" aiohttp "openai>=1.0" "uvicorn[standard]" \
    "pydantic>=2.9" pillow \
    "prometheus_client>=0.18.0" "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" "typing_extensions>=4.10" "filelock>=3.16.1" \
    partial-json-parser pyzmq msgspec importlib_metadata \
    "mistral_common>=1.5.0" pyyaml einops \
    cloudpickle blake3 interegular jsonschema \
    accelerate safetensors huggingface_hub \
    "outlines>=0.0.43,<0.1" "lm-format-enforcer>=0.10.9,<0.11" \
    "compressed-tensors==0.8.0" "depyf==0.18.0" "gguf==0.10.0" \
    lark ray

# --- vLLM v0.6.5 from source (prebuilt wheel has ABI mismatch with NGC PyTorch) ---
# Build against the NGC torch to get matching C++ ABI symbols.
RUN pip install --no-cache-dir setuptools-scm cmake ninja
RUN git clone --depth 1 --branch v0.6.5 \
        https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    TORCH_CUDA_ARCH_LIST="7.0" \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=4 \
    pip install --no-deps --no-build-isolation -e . && \
    cd / && rm -rf /tmp/vllm/.git

# Patch vLLM to use our SM70-compatible flash-attn instead of its bundled copy.
# 1. Selector: lower capability check from SM80 to SM70
RUN sed -i 's/has_device_capability(80)/has_device_capability(70)/' \
    /tmp/vllm/vllm/attention/selector.py && \
    grep -q 'has_device_capability(70)' /tmp/vllm/vllm/attention/selector.py && \
    echo "Patch 1/2: selector SM70 check OK"
# 2. Replace vllm_flash_attn module with shim that wraps our flash_attn.
#    vLLM's fork adds `out` parameter; our flash_attn doesn't have it.
COPY benchmarks/v100_vllm_serving/vllm_flash_attn_shim.py /tmp/vllm/vllm/vllm_flash_attn/__init__.py
RUN python -c "\
from vllm.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache; \
print('Patch 2/3: vllm_flash_attn shim OK')"
# 3. Allow block_size=256 for SM70 paged KV cache (default: 8,16,32 only)
RUN sed -i "s/choices=\[8, 16, 32\]/choices=[8, 16, 32, 256]/" \
    /tmp/vllm/vllm/engine/arg_utils.py && \
    grep -q '256' /tmp/vllm/vllm/engine/arg_utils.py && \
    echo "Patch 3/3: block_size=256 allowed"

# Verify full environment
RUN python -c "\
import torch, flash_attn, xformers, vllm; \
expected = open('/tmp/torch_ver.txt').read(); \
actual = torch.__version__; \
assert actual == expected, f'PyTorch was replaced: {expected} -> {actual}'; \
print(f'PyTorch OK:   {actual}'); \
print(f'flash_attn:   {flash_attn.__version__}'); \
print(f'xformers:     {xformers.__version__}'); \
print(f'vLLM:         {vllm.__version__}'); \
print('Environment OK')"

COPY benchmarks/v100_vllm_serving/ /workspace/flash-attention/benchmarks/v100_vllm_serving/
WORKDIR /workspace/flash-attention/benchmarks/v100_vllm_serving

# Build:
#   docker build -t flash_attn_v100_vllm -f benchmarks/v100_vllm_serving/Dockerfile .
#
# Run (mount HF cache to avoid re-downloading model each time):
#   docker run --gpus '"device=0"' --rm \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       flash_attn_v100_vllm python benchmark_serving.py --dry-run
