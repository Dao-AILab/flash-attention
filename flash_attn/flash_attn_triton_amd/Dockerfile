FROM rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0

WORKDIR /workspace

# install triton
RUN git clone https://github.com/triton-lang/triton &&\ 
    cd triton &&\
    git checkout 3ca2f498e98ed7249b82722587c511a5610e00c4 &&\
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