# FlashAttention-4 (CuTeDSL)

FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.

## Installation

```sh
pip install flash-attn-4
```

## Usage

```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
```

## Development

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"
pytest tests/cute/
```
