# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashAttention-4 (FA4) — fast, memory-efficient exact attention kernels written in Python using CuTeDSL (NVIDIA CUTLASS DSL). Kernels are compiled to PTX/CUBIN at runtime. Targets Hopper (SM90) and Blackwell (SM100/SM110) GPUs. Package name: `flash-attn-4`.

The repository also contains older generations (FA2 in top-level `csrc/`, FA3 in `hopper/`) but active development is on FA4 in `flash_attn/cute/`.

## Build & Install

```bash
pip install flash-attn-4
# or dev install:
pip install -e "flash_attn/cute[dev]"
```

Dependencies: `nvidia-cutlass-dsl>=4.4.1`, `torch`, `einops`, `apache-tvm-ffi`, `quack-kernels>=0.2.10`.

## Running Tests

```bash
pytest tests/cute/test_flash_attn.py
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x  # single test
pytest tests/cute/test_flash_attn_varlen.py
pytest tests/cute/test_mask_mod.py
pytest tests/cute/test_score_mod.py
pytest tests/cute/test_block_sparsity.py
```

### Fast two-pass testing

Compilation dominates test time. The fast workflow separates compilation (parallel, no GPU needed) from execution (uses cached binaries):

```bash
# Pass 1: compile all kernels in parallel using FakeTensorMode (no GPU memory allocation)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: run tests using cached compiled kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

- `FLASH_ATTENTION_FAKE_TENSOR=1` — uses PyTorch FakeTensorMode to compile kernels without allocating GPU memory or running them.
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` — enables persistent disk cache at `/tmp/${USER}/flash_attention_cute_dsl_cache/`.
- `-n 256` — pytest-xdist parallel workers (only useful in the compilation pass).

Tests are parametrized over dtype (fp16/bf16), head dimension (64, 96, 128), sequence length, causal/non-causal, and MHA/GQA/MQA.

If you get OOM errors running tests or benchmarks, use `nvidia-smi` to find a free GPU and select it with `CUDA_VISIBLE_DEVICES=<id>`.

## Linting

Pre-commit uses ruff on `flash_attn/cute/` files. Large kernel files (`flash_bwd.py`, `flash_fwd.py`, `flash_fwd_sm100.py`, `interface.py`) are excluded from auto-formatting.

```bash
ruff check flash_attn/cute/ --fix
ruff format flash_attn/cute/
```

## Code Architecture

### Public API (`flash_attn/cute/interface.py`)

Two entry points exported from `flash_attn/cute/__init__.py`:
- `flash_attn_func(q, k, v, ...)` — standard attention
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` — variable-length

Key parameters: `causal`, `window_size_left/right`, `softmax_scale`, `softcap`, `score_mod`, `mask_mod`, `block_sparse_tensors`, `num_splits`, `pack_gqa`, `m_block_size`, `n_block_size`, `num_threads`.

Tensor layout: `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

### Forward Kernels

- `flash_fwd.py` — `FlashAttentionForwardSm90`: Hopper forward. No SplitKV or paged KV.
- `flash_fwd_sm100.py` — `FlashAttentionForwardSm100`: Blackwell forward. Full features including SplitKV, paged KV cache, persistent kernels, 2CTA instructions.
- `flash_fwd_combine.py` — `FlashAttentionForwardCombine`: merges SplitKV partial results.

### Backward Kernels

- `flash_bwd.py` — `FlashAttentionBackwardSm80`: Ampere backward (base).
- `flash_bwd_sm90.py` — `FlashAttentionBackwardSm90`: Hopper backward.
- `flash_bwd_sm100.py` — `FlashAttentionBackwardSm100`: Blackwell backward with 2CTA and block sparse support.
- `flash_bwd_preprocess.py` / `flash_bwd_postprocess.py` — auxiliary backward kernels.

### Core Abstractions

- `softmax.py` — Online softmax with row_max/row_sum tracking, score modifier support.
- `mask.py` — `AttentionMask`: causal, local/sliding window, block sparse, mask_mod application.
- `block_info.py` — `BlockInfo`: tile dimensions, n/m block range computation for causal/local masking.
- `seqlen_info.py` — `SeqlenInfoQK`: sequence length and offset tracking for varlen.
- `pipeline.py` — `PipelineStateSimple`: circular buffer index/phase management for pipelined loads.
- `tile_scheduler.py` — Tile scheduling strategies (single tile, varlen-aware, persistent).
- `copy_utils.py` — Type-converting copies, shared-to-register loads, TMA copy atoms.
- `named_barrier.py` — Named barrier enums for warp synchronization.

### Architecture-Specific Helpers

- `hopper_helpers.py` — SM90 warp-group GEMM, shared memory layout creation, fence/commit/wait.
- `blackwell_helpers.py` — SM100 UMMA-based GEMM, PTX-optimized paths, 2CTA support.
- `mma_sm100_desc.py` — Hardware MMA descriptor enums (formats, saturation, scaling).

### Other Components

- `pack_gqa.py` — Packs multiple Q heads per KV head for efficient GQA.
- `paged_kv.py` — `PagedKVManager`: paged KV cache with TMA support.
- `fast_math.py` — exp2 polynomial coefficients, softcap score_mod creation.
- `utils.py` — Hash functions for compile cache keys, warp reductions, predicates.
- `cache_utils.py` — JIT compilation cache management.
- `cute_dsl_utils.py` — Patched `cute.compile` that optionally dumps SASS.

### Compilation & Caching

Kernels are JIT-compiled. Cache key includes dtype, head_dim, causal, mask/score_mod hashes, architecture, block sizes. Caching levels: in-memory LRU + optional disk cache via `get_jit_cache()`.

Env vars: `CUTE_CUBIN_PATH` (dump CUBIN/SASS), `CUTE_DSL_KEEP_PTX=1` (inspect PTX), `CUTE_DSL_PTXAS_PATH` (custom ptxas).

## Key Patterns

- Compile-time constants use `cutlass.Constexpr[type]` for kernel specialization.
- Score/mask modifiers are user-defined `@cute.jit` callables injected into the kernel at compile time.
- Forward execution: load Q tile → loop over K/V blocks (pipelined) → online softmax accumulation → store O and LSE.
- 2CTA instructions (SM100, hdim=128): both CTAs in a cluster coordinate via shared mbarriers; tx_count must be multiplied by `cta_group_size`.

## Debugging GPU Kernels

See `AI/DEBUG_2CTA.md` for kernel hang/deadlock debugging (printf bisection, pipeline barrier analysis, 2CTA pitfalls). See `AI/RACECHECK_TMA_HAZARD.md` for `compute-sanitizer` false positives with `cp.async.bulk`.

Key tools:
- `cute.printf` with thread guards (`tidx % 32 == 0`, `elect_one()`) for targeted output
- `compute-sanitizer --tool=racecheck` (beware false positives with raw TMA)
- `CUTE_DSL_KEEP_PTX=1` and `CUTE_DSL_LINEINFO=1` for PTX inspection and sanitizer source mapping
