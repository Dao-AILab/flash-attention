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
8) ALiBi

We are working on the following things
1) Paged Attention 
2) Sliding Window
3) FP8
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

You can use autotune for better performance by using this flag `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"`
```
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE" python $PATH_TO_CODE
```

###### Docker
You can also use the Dockerfile below which does the above steps on top of the latest rocm/pytorch image.
```
FROM rocm/pytorch:latest

WORKDIR /workspace

# install triton
RUN pip install triton==3.2.0

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

###### FP8
In our fork We have created the following api functions that use fp8 to compute their values. These functions are `flash_attn_fp8_func`, `flash_attn_varlen_fp8_func`, `flash_attn_qkvpacked_fp8_func` and `flash_attn_varlen_qkvpacked_fp8_func`. To use these functions just call them with like the other api functions, the casting will be handled internally. For example

```
from flash_attn import flash_attn_qkvpacked_fp8_func

# forward pass
out, lse, S_dmask = flash_attn_qkvpacked_fp8_func(
                qkv,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

# backward pass
do = torch.randn_like(out)
dqkv = torch.autograd.grad(out, (qkv), do)
```

You can use the other api functions in a similar way.



##### Credits
AMD Triton kernels team

OpenAI kernel team
