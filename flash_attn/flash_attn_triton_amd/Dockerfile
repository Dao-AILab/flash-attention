FROM rocm/pytorch:rocm6.3.2_ubuntu22.04_py3.10_pytorch_release_2.4.0

WORKDIR /workspace

# install triton
RUN pip install triton=3.2.0

# install flash attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
    cd flash-attention &&\
    git checkout main_perf &&\
    python setup.py install

# set working dir
WORKDIR /workspace/flash-attention