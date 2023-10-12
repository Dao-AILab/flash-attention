FROM rocm/pytorch:latest-release

RUN pip uninstall -y triton
WORKDIR /workspace
USER root

# Download and install triton
RUN git clone -b fa_fwd_benchmark_2gpus https://github.com/ROCmSoftwarePlatform/triton.git
RUN cd /workspace/triton/python \
    && pip install -e .

RUN pip install gitpython
RUN apt install bc
RUN pip install einops
