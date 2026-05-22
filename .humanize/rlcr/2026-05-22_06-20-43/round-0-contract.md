# Round 0 Contract

## Mainline Objective
Rebase flash_attn_pr onto latest public/main to inherit upstream pipeline optimizations, then verify all modes match the full TFLOPS table from fp4_kernel_vs_upstream_investigation.md within 3%.

## Target ACs
- AC-6: TFLOPS match investigation table within 3%
- AC-1: BF16 zero regression (must match upstream BF16 ref column)

## Blocking Side Issues In Scope
- flash_attn_pr is 2+ commits behind public/main (interface.py conflict in block_sparsity)
- Must use system Python (python3) not .venv for correct cutlass-dsl integration

## Queued Side Issues Out of Scope
- AC-5 group-128 V scale (already resolved)
- AC-9 unsupported-mode guards (already present)

## Round Success Criteria
1. bench_fp4.py all 3 modes produce cos >= 0.99
2. NVFP4+BF16 >= 1830 TF at (1,32768,24,128) (within 3% of 1887)
3. NVFP4+FP8 >= 1960 TF at (1,32768,24,128) (within 3% of 2018)
4. MXFP8+FP8 >= 1890 TF at (1,32768,24,128) (within 3% of 1948)
5. BF16 ref >= 1500 TF at (1,32768,24,128) (within 3% of 1545)
6. No env var debug overrides in final code
