# Inspired by https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.10.0-cuda11.3-ubuntu20.04/Dockerfile
# ARG COMPAT=0
ARG PERSONAL=0
# FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as base-0
FROM nvcr.io/nvidia/pytorch:22.12-py3 as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# git for installing dependencies
# tzdata to set time zone
# wget and unzip to download data
# [2021-09-09] TD: zsh, stow, subversion, fasd are for setting up my personal environment.
# [2021-12-07] TD: openmpi-bin for MPI (multi-node training)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    zsh stow subversion fasd \
    && rm -rf /var/lib/apt/lists/*
    # openmpi-bin \

# Allow running runmpi as root
# ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# # Create a non-root user and switch to it
# RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
#     && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod 777 /home/user
WORKDIR /home/user

# Set up personal environment
# FROM base-${COMPAT} as env-0
FROM base as env-0
FROM env-0 as env-1
# Use ONBUILD so that the dotfiles dir doesn't need to exist unless we're building a personal image
# https://stackoverflow.com/questions/31528384/conditional-copy-add-in-dockerfile
ONBUILD COPY dotfiles ./dotfiles
ONBUILD RUN cd ~/dotfiles && stow bash zsh tmux && sudo chsh -s /usr/bin/zsh $(whoami)
# nvcr pytorch image sets SHELL=/bin/bash
ONBUILD ENV SHELL=/bin/zsh

FROM env-${PERSONAL} as packages

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1

# # apex and pytorch-fast-transformers take a while to compile so we install them first
# TD [2022-04-28] apex is already installed. In case we need a newer commit:
# RUN pip install --upgrade --force-reinstall --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" --global-option="--fmha" --global-option="--fast_layer_norm" --global-option="--xentropy" git+https://github.com/NVIDIA/apex.git#egg=apex

# xgboost conflicts with deepspeed
RUN pip uninstall -y xgboost && DS_BUILD_UTILS=1 DS_BUILD_FUSED_LAMB=1 pip install deepspeed==0.7.7

# General packages that we don't care about the version
# zstandard to extract the_pile dataset
# psutil to get the number of cpu physical cores
# twine to upload package to PyPI
RUN pip install pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops opt_einsum fvcore gsutil cmake pykeops zstandard psutil h5py twine gdown \
    && python -m spacy download en_core_web_sm
# hydra
RUN pip install hydra-core==1.3.1 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 pyrootutils rich
# Core packages
RUN pip install transformers==4.25.1 datasets==2.8.0 pytorch-lightning==1.8.6 triton==2.0.0.dev20221202 wandb==0.13.7 timm==0.6.12 torchmetrics==0.10.3
# torchmetrics 0.11.0 broke hydra's instantiate

# For MLPerf
RUN pip install git+https://github.com/mlcommons/logging.git@2.1.0

# Install FlashAttention
RUN pip install flash-attn==2.1.1

# Install CUDA extensions for cross-entropy, fused dense, layer norm
RUN git clone https://github.com/HazyResearch/flash-attention \
    && cd flash-attention && git checkout v2.1.1 \
    && cd csrc/fused_softmax && pip install . && cd ../../ \
    && cd csrc/rotary && pip install . && cd ../../ \
    && cd csrc/xentropy && pip install . && cd ../../ \
    && cd csrc/layer_norm && pip install . && cd ../../ \
    && cd csrc/fused_dense_lib && pip install . && cd ../../ \
    && cd csrc/ft_attention && pip install . && cd ../../ \
    && cd .. && rm -rf flash-attention
