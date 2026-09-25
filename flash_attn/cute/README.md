# FlashAttention-4 (CuTeDSL)

FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.

## Installation

```sh
pip install flash-attn-4
```

If you're on CUDA 13, install with the `cu13` extra for best performance:

```sh
pip install "flash-attn-4[cu13]"
```

## Usage

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
```

### FP8 forward with head_dim 256 on Blackwell

On SM100 / SM103 (B200, B300, GB300), dense attention with `float8_e4m3fn` Q/K/V and
`head_dim == head_dim_v == 256` uses a dedicated persistent 2-CTA forward kernel whose
BF16 output is bitwise identical to the generic hd256 kernel. Variable-length inputs,
paged KV, local attention, descales and E5M2 keep using the generic kernels.

## Development

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
pytest tests/cute/
```
