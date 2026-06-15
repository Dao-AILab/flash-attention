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

## Consumer Blackwell (sm_120 / RTX PRO 6000, RTX 50-series)

FA4 runs on consumer Blackwell (compute capability 12.x), which exposes SM80-class
`mma.sync` tensor cores (no WGMMA/tcgen05/TMEM). Dispatch and tile selection are
auto-tuned for this arch; no environment variables are required for normal use.

**Supported:** forward and backward for dense, causal, and local/sliding-window
attention; MHA / GQA / MQA; variable-length (`flash_attn_varlen_func`); paged-KV;
block sparsity; `score_mod` / `mask_mod`; learnable sink. Head dims 64/96/128/192/256.

**fp8 KV-cache decode (e4m3/e5m2):** for decode (`seqlen_q == 1`) with a quantized
K/V cache and a bf16/fp16 query, pass fp8 `k`/`v` plus per-`(batch, kv_head)` fp32
`k_descale`/`v_descale`. This auto-routes to a memory-efficient GEMV decode kernel
(no env var needed) and is ~1.6–1.9× faster than bf16 at GQA ratios ≤ 4 while halving
KV-cache bandwidth. Accuracy is within ~2e-3 of an fp8-quantized reference.

**Environment flags:**
- `FLASH_ATTENTION_SM120_DECODE_KERNEL=1` — opt into the experimental **bf16** GEMV
  decode kernel for `seqlen_q == 1` (the fp8 decode path above is always on when fp8
  K/V is supplied). Off by default.
- `FLASH_ATTENTION_ARCH` — override the detected compute capability (testing/compile).

**Known performance floors vs FA2 on sm_120** (hardware-bound, not bugs):
- Causal *MHA* (`qhead_per_kvhead == 1`) at `seqlen ≥ 8192` is ~0.95× FA2 — a register/
  occupancy wall (255 regs/thread → 1 CTA/SM). GQA (the common case) is at parity or faster.
- fp8 KV-cache decode regresses below bf16 at GQA ratio ≥ 8 (the GEMV loop becomes
  compute-bound); it still halves KV memory, so it remains the only fp8-cache path.
- Backward is at ~parity with FA2; fp8 is forward/decode-only (no fp8 backward).

**Feature limitations on sm_120:**
- `learnable_sink` is incompatible with SplitKV (each split would double-count the sink
  in the combine step), so SplitKV is disabled when a sink is present — attention runs
  in a single split (correct, but without the decode SplitKV speedup).
- Negative-offset sliding windows (`window_size` with a negative bound, e.g. `(None, -X)`
  or `(-X, None)`) are forward-only: the backward raises `NotImplementedError` (its
  dK/dV are incorrect for these offset windows). Non-negative windows are fully supported.
- Deterministic backward (`deterministic=True`) is not supported (the SM80-base backward
  lacks the dQ-semaphore path).

## Development

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
pytest tests/cute/
```
