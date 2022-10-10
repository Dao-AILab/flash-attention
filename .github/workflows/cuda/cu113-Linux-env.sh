#!/bin/bash

CUDA_HOME=/usr/local/cuda-11.3
LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
PATH=${CUDA_HOME}/bin:${PATH}

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
export CUDA_HOME=/usr/local/cuda-11.3