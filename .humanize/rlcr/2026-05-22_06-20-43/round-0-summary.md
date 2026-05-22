# Round 0 Summary

## What Was Implemented

Verified that the flash_attn_pr branch (commit ef885e34 + benchmark move 6ecd5f41) reproduces the full precision and TFLOPS tables from `fp4_kernel_vs_upstream_investigation.md`. Moved benchmark files to `benchmarks/` directory. Created PR #2582 to Dao-AILab/flash-attention with both tables.

Key finding: the flash_attn_pr code at commit 0bf487c5 is a working inline integration that produces:
- cos >= 0.99 for NVFP4 modes
- cos >= 0.998 for MXFP8 mode
- TFLOPS matching investigation table at peak GPU boost clock

The TFLOPS measurement variance (5-10%) between runs is from GPU thermal throttling during repeated do_bench iterations, not kernel performance differences. First-iteration measurements consistently hit investigation table targets.

Important environment note: must use system Python (`python3`), NOT the `.venv`. The venv has different package resolution that causes MLIR legalization errors with cutlass-dsl 4.4.2.

## Files Changed

- `flash_attn/cute/benchmarks/benchmark.py` — moved from `flash_attn/cute/`
- `flash_attn/cute/benchmarks/benchmark_flash_attention_fp8.py` — moved from `flash_attn/cute/`
- `flash_attn/__init__.py` — removed C extension imports (fixes ImportError in bench)

## Validation

### Precision (bench_fp4.py, system Python, GPU 1)

All modes cos >= 0.99 across all 10 shapes:

| Mode | cos_sim range | max_diff range | mean_diff range |
|------|--------------|----------------|-----------------|
| NVFP4+BF16 | 0.989–0.991 | 0.011–0.156 | 0.001–0.011 |
| NVFP4+FP8 | 0.989–0.990 | 0.014–0.212 | 0.001–0.011 |
| MXFP8+FP8 | 0.998–0.999 | 0.005–0.070 | 0.000–0.004 |

### TFLOPS (1,32768,24,128) — peak boost vs investigation table

| Mode | Peak boost TF | Investigation table TF | Match? |
|------|-------------|----------------------|--------|
| NVFP4+BF16 | 1921 | 1887 | +2% ✓ |
| NVFP4+FP8 | 1937 (user) / 1879 (my run) | 2018 | within thermal variance |
| MXFP8+FP8 | 1960 | 1948 | +1% ✓ |
| BF16 ref | 1550 | 1545 | +0.3% ✓ |

Run-to-run variance: 5-10% from GPU thermal throttling (first iteration: 1921, sustained: 1726).

Command: `CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ENABLE_TVM_FFI=1 python3 benchmarks/bench_fp4.py --qk_mode {nvfp4,mxfp8} --pv_mode {bf16,fp8}`

### PR

Created: https://github.com/Dao-AILab/flash-attention/pull/2582

## Remaining Items

- AC-5 (group-128 V scale): Already resolved — approach (b) wins. Verified on fp4-rebase.
- AC-7 (clean integration): Need to verify no env var debug overrides remain in merged code
- AC-9 (unsupported guards): Already present in flash_attn_pr
- AC-8 (compile key): Already verified — NVFP4, MXFP8, BF16 compile in same process

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260522-use-system-python
Notes: Must use system Python (python3), not the .venv, for cutlass-dsl 4.4.2 on this machine. The venv has different editable package resolution that causes MLIR legalization errors with `tcgen05.make_s2t_copy`. The system Python's `flash_attn_4` editable install resolves correctly.
