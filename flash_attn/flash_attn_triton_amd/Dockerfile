FROM rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0

WORKDIR /workspace

# install triton
RUN git clone https://github.com/triton-lang/triton &&\ 
    cd triton &&\
    git checkout f906b9b2f9a5efa0040f52f88ca1491526b6dc56 &&\
    pip uninstall -y triton &&\ 
    cd python &&\
    pip install matplotlib pandas pytest einops &&\
    pip install --verbose -e .

# install flash attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
    cd flash-attention &&\
    git checkout main_perf &&\
    python setup.py install

# set working dir
WORKDIR /workspace/flash-attention