# Attention kernel from FasterTransformer

This CUDA extension wraps the single-query attention [kernel](https://github.com/NVIDIA/FasterTransformer/blob/release/v5.2.1_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp) from
FasterTransformer v5.2.1 for benchmarking purpose.

```sh
cd csrc/ft_attention && pip install .
```
