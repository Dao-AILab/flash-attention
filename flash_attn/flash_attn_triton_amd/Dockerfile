FROM rocm/pytorch:rocm6.3.2_ubuntu22.04_py3.10_pytorch_release_2.4.0

WORKDIR /workspace

# install deps
RUN pip install numpy==1.24 matplotlib pandas pytest einops

# install triton
ARG TRITON_COMMIT=f906b9b2f9a5efa0040f52f88ca1491526b6dc56
RUN git clone https://github.com/triton-lang/triton && \
    cd triton && \
    git checkout ${TRITON_COMMIT} && \
    pip uninstall -y triton && \
    cd python && \
    pip install --verbose -e .

# install flash attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
    cd flash-attention &&\
    git checkout main_perf &&\
    python setup.py install

# set working dir
WORKDIR /workspace/flash-attention