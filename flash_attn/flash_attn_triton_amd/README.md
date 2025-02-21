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
2) ALiBi and matrix bias

These features are in development
1) Paged Attention 
2) Sliding Window
5) Performance Improvements

##### Getting Started
To get started with the triton backend for AMD, follow the steps below.

First install the recommended Triton version 

```
pip install triton==3.2.0
```
Then install and test Flash Attention with the flag `FLASH_ATTENTION_TRITON_AMD_ENABLE` set to `"TRUE"`.

```
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
cd flash-attention
python setup.py install
pytest tests/test_flash_attn_triton_amd.py
```

###### Docker
We have also created a Dockerfile.

To build the docker file
```
cd flash_attn/flash_attn_triton_amd
docker build -t fa_triton .
```

To run the docker image
```
docker run -it --network=host --user root --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 16G --device=/dev/kfd --device=/dev/dri fa_triton
```
Inside the docker, it should open to the flash attention repo with everything installed. You can run the following command to test things.
```
pytest tests/test_flash_attn_triton_amd.py
```

##### Credits
AMD Triton kernels team

OpenAI kernel team
