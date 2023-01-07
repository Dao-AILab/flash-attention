FROM rocm/pytorch:rocm5.3.2_ubuntu20.04_py3.7_pytorch_1.12.1
WORKDIR /flash_attn

USER root

ENV DEBIAN_FRONTEND noninteractive
ENV TZ "Asia/Shanghai"

RUN apt-get update \
    && apt install -y git-all \
    && git clone https://<your github username>:<your github token>@github.com/ROCmSoftwarePlatform/flash-attention_private \
    && cd /flash_attn/flash-attention_private \
    && git checkout flash_attention_for_rocm \
    && cd /flash_attn/flash-attention_private/csrc/flash_attn_rocm/composable_kernel \
    && git submodule init \
    && git submodule update \
    && cd /flash_attn/flash-attention_private/csrc/flash_attn_rocm \
    && mkdir build \
    && cd build \
    && cmake .. \
    && cd /flash_attn/flash-attention_private/csrc/flash_attn_rocm/build \
    && make -j64

