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

# Install torch from the published cu130 nightly package page and let pip resolve
# the matching nightly Triton wheel from PyTorch's nightly triton page.
RUN uv pip install --system --break-system-packages --no-cache --pre \
        "torch>=2.12.0.dev0,<2.13.0" \
        --find-links https://download.pytorch.org/whl/nightly/cu130/torch \
        --find-links https://download.pytorch.org/whl/nightly/triton

# FA4 dependencies — derived from pyproject.toml to keep a single source of truth.
# torch stays pinned from the direct wheel install above.
# The package itself (version 0.0.0 via setuptools-scm fallback) is overwritten at
# step time by the editable install from the mounted repo.
COPY flash_attn/cute/ /tmp/fa4/
RUN uv pip install --system --break-system-packages --no-cache "/tmp/fa4[cu13,dev]"

CMD ["/bin/bash"]
