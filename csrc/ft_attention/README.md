# Attention kernel from FasterTransformer

This CUDA extension wraps the single-query attention [kernel](https://github.com/NVIDIA/FasterTransformer/blob/release/v5.2.1_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp) from
FasterTransformer v5.2.1 for benchmarking purpose.

```sh
cd csrc/ft_attention && pip install .
```

As of 2023-09-17, this extension is no longer used in the FlashAttention repo.
FlashAttention now has implemented
[`flash_attn_with_kvcache`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attention_interface.py)
with all the features of this `ft_attention` kernel (and more).

