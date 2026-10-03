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
from flash_attn.cute import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

out, _ = flash_attn_func(q, k, v, causal=True)

# Packed projection outputs can be passed without manually splitting them.
out_qkv, _ = flash_attn_qkvpacked_func(qkv, causal=True)
out_kv, _ = flash_attn_kvpacked_func(q, kv, causal=True)
```

Packed inputs use the same CuTe options and return the same ``(out, lse)`` tuple
as their unpacked counterparts. The packed dimensions are
``(batch, seqlen, 3, heads, dim)`` for QKV and
``(batch, seqlen_k, 2, heads_k, dim)`` for KV; the varlen forms omit the batch
dimension.

## Development

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
pytest tests/cute/
```
