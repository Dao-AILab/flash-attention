FROM rocm/pytorch:latest-release

WORKDIR /workspace
USER root

# Download and install triton
RUN git clone -b fa_fwd_benchmark_2gpus https://github.com/ROCmSoftwarePlatform/triton.git
RUN cd /workspace/triton/python \
    && pip install -e .

# git clone flash-attention and checkout branch benchmark_openai_triton_amd
RUN git clone -b benchmark_openai_triton_amd https://github.com/zhanglx13/flash-attention.git

RUN pip install gitpython
RUN apt install bc
RUN pip install einops

ENV PYTHONPATH "/workspace/flash-attention"
