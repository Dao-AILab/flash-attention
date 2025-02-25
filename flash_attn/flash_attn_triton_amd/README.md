Flash Attention Triton Kernel
===============

#### Introduction
The Triton implementation of the [Flash Attention v2](https://tridao.me/publications/flash2/flash2.pdf) is currently a work in progress.

It supports AMD's CDNA (MI200, MI300) and RDNA GPU's using fp16, bf16 and fp32 datatypes.

These features are supported in Fwd and Bwd
1) Fwd and Bwd with causal masking
2) Variable sequence lengths
3) Arbitrary Q and KV sequence lengths
4) Arbitrary head sizes
5) Multi and grouped query attention
6) Dropout
7) Rotary embeddings

These features are supported in Fwd for now. We will add them to backward soon.
1) ALiBi

These features are in development
1) FP8
2) Paged Attention 
3) Sliding Window
4) Performance Improvements

##### Getting Started
To get started with the triton backend for AMD, follow the steps below.

First install the recommended Triton version 

```
pip install triton==3.2.0
```
Then install Flash Attention with the flag `FLASH_ATTENTION_TRITON_AMD_ENABLE` set to `"TRUE"`.

```
cd flash-attention
git checkout main_perf
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
```

To test that things are working, you can run our tests. These tests take hours so you don't need to run the full thing.
```
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pytest tests/test_flash_attn_triton_amd.py
```

###### Docker
You can also use the Dockerfile below which does the above steps on top of the latest rocm/pytorch image.
```
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
```

To build the docker file
```
docker build -t fa_triton .
```

To run the docker image
```
docker run -it --network=host --user root --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 16G --device=/dev/kfd --device=/dev/dri fa_triton
```
##### Credits
AMD Triton kernels team

OpenAI kernel team
