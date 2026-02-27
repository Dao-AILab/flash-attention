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

# --- vLLM v0.6.5 (no deps to preserve NGC torch + our flash-attn + xFormers) ---
RUN pip install vllm==0.6.5 --no-deps --no-cache-dir

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
