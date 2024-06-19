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

These features are supported in Fwd for now. We will add them to backward soon.
1) Multi and grouped query attention
2) ALiBi and matrix bias

These features are in development
1) Paged Attention 
2) Sliding Window
3) Rotary embeddings
4) Dropout
5) Performance Improvements

#### Getting Started
To get started with the triton backend for AMD, follow the steps below.

First install the recommended Triton [commit](https://github.com/triton-lang/triton/commit/2e9f2c2d20601c24b91a4c32a7b97ad1f8a55d88).

```
git clone https://github.com/triton-lang/triton
cd triton
git checkout 2e9f2c2d20601c24b91a4c32a7b97ad1f8a55d88 
pip install --verbose -e python
```
Then install and test Flash Attention with the flag `FLASH_ATTENTION_TRITON_AMD_ENABLE` set to `"TRUE"`.

```
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
cd flash-attention
python setup.py install
pytest tests/test_flash_attn.py
```

#### Credits
AMD Triton kernels team

OpenAI kernel team
