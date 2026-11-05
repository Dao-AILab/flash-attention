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

Arithmetic opt-ins use `FwdConfig.approximate_softmax` for packed BF16 exponentials
and `BwdConfig.bf16_dq` (passed as `bwd_config=`) for direct BF16 dQ publication.
Both default to False. Packed softmax uses a device norm check and the config's
ordinary exponential path as fallback, with no minimum sequence length.
BF16 dQ rounds each KV tile contribution and currently uses the D128 two-CTA kernel.
Exponential emulation uses the existing `ex2_emu_freq` and `ex2_emu_start_frg` fields.

## Development

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
pytest tests/cute/
```
