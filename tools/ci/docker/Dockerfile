FROM ubuntu:24.04

ENV PATH=/usr/local/cuda/bin:/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
        python3 python3.12 python3.12-dev python3.12-venv \
        curl git ca-certificates gcc && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# torch nightly cu129 (required for B200/SM100, pin to cu129 to match driver 12.9)
RUN uv pip install --system --break-system-packages --no-cache --pre \
        "torch==2.12.0.dev20260324+cu129" \
        --index-url https://download.pytorch.org/whl/nightly/cu129

# FA4 dependencies — torch pin included to prevent transitive deps from overriding cu129
RUN uv pip install --system --break-system-packages --no-cache \
        "torch==2.12.0.dev20260324+cu129" \
        "nvidia-cutlass-dsl>=4.4.2" \
        "quack-kernels>=0.3.3" \
        "apache-tvm-ffi>=0.1.5,<0.2" \
        "torch-c-dlpack-ext" \
        "einops" \
        "typing_extensions" \
        "pytest" \
        "pytest-xdist" \
        "ruff" \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu129

CMD ["/bin/bash"]
