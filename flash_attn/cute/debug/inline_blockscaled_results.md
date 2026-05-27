# Inline block-scaled FP4 — precision & TFLOPS (flash_attn_pr)

Block-scaled NVFP4 QK integrated as conditional paths in the upstream
`flash_fwd_sm100.py` (no separate kernel file). Both NVFP4 PV modes verified
correct and NaN-free across all benchmarked shapes.

## Reproduce

```bash
# from flash_attn/cute/  (commit 1241026d, branch flash_attn_pr)
CUDA_VISIBLE_DEVICES=<free_gpu> python benchmarks/bench_fp4.py --qk_mode nvfp4 --pv_mode bf16
CUDA_VISIBLE_DEVICES=<free_gpu> python benchmarks/bench_fp4.py --qk_mode nvfp4 --pv_mode fp8
```

- Precision reference: BF16 `flash_attn_func` on the same (unquantized) inputs.
- NVFP4 tensors: flashinfer `nvfp4_quantize` (per-block adaptive E4M3 SF, sf_vec_size=16).
- B200 (sm_100a), nvidia-cutlass-dsl 4.4.2. TFLOPS measured via `triton.testing.do_bench`
  on a **shared** GPU, so absolute TFLOPS are a lower bound (thermal/contention).

## NVFP4 QK + BF16 PV

| (b, s, h, d)        | cos_sim | max_diff | TFLOPS |
|---------------------|---------|----------|--------|
| (1, 256, 16, 128)   | 0.9798  | 0.405    | 21     |
| (1, 1024, 16, 128)  | 0.9881  | 0.185    | 310    |
| (4, 4096, 16, 128)  | 0.9899  | 0.062    | 1512   |
| (1, 32768, 16, 128) | 0.9904  | 0.013    | 1718   |
| (4, 4096, 32, 128)  | 0.9899  | 0.072    | 1541   |
| (1, 4096, 12, 128)  | 0.9900  | 0.065    | 907    |
| (1, 32768, 12, 128) | 0.9902  | 0.018    | 1634   |
| (1, 4096, 24, 128)  | 0.9899  | 0.057    | 1251   |
| (1, 32768, 24, 128) | 0.9905  | 0.012    | 1648   |

## NVFP4 QK + FP8 PV

| (b, s, h, d)        | cos_sim | max_diff | TFLOPS |
|---------------------|---------|----------|--------|
| (1, 256, 16, 128)   | 0.9792  | 0.406    | 20     |
| (1, 1024, 16, 128)  | 0.9855  | 0.224    | 298    |
| (4, 4096, 16, 128)  | 0.9861  | 0.145    | 1500   |
| (1, 32768, 16, 128) | 0.9844  | 0.040    | 1675   |
| (4, 4096, 32, 128)  | 0.9861  | 0.090    | 1525   |
| (1, 4096, 12, 128)  | 0.9863  | 0.078    | 913    |
| (1, 32768, 12, 128) | 0.9842  | 0.033    | 1598   |
| (1, 4096, 24, 128)  | 0.9861  | 0.068    | 1251   |
| (1, 32768, 24, 128) | 0.9846  | 0.024    | 1676   |

Both modes are flat ~0.98–0.99 cos with zero NaN (previously cos 0.45 / NaN before
the block-scaled integration fixes; see commits 8216e2e4, 721aad75).

## MXFP8 QK + FP8 PV — known remaining issue

MXFP8 (FP8 E4M3 QK, E8M0 SF, sf_vec_size=32) now runs end-to-end after the bench
tensor-creation fix (commit 1241026d; `cute.testing.convert` fails with
`cudaErrorInsufficientDriver` in this env, so MXFP8 uses torch-native FP8 + uniform
E8M0 SF). However the MXFP8 *kernel* path (sf_vec_size=32) still produces cos ~0.70
(target ~0.9985) — a separate correctness bug under investigation. NVFP4 modes are
unaffected.
