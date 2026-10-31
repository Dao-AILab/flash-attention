# SM100 dense and packed-varlen forward validation

This benchmark reproduces LSE-producing forward comparisons from [issue #2764](https://github.com/Dao-AILab/flash-attention/issues/2764). A B200 validation on 2026-10-08 found that the original large gap is improved on pinned main `94e22c9`, consistent with the already merged [#2869](https://github.com/Dao-AILab/flash-attention/pull/2869). The companion script measures this workload and retains raw timing samples. The validation record was collected with an earlier isolated harness; it is not a full rerun with the packaged CLI.

## Run the benchmark

Run with the same Python environment for every source checkout. The default shape is BF16, noncausal B1/S86000/Hq=Hkv32/D128 with LSE and one split.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_sm100_dense_varlen.py \
  --include-1cta --output benchmarks/results/main.json

# Same source with the historical register table, in a fresh process.
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_sm100_dense_varlen.py \
  --include-1cta --restore-old-budget --output benchmarks/results/main-old-budget.json

# Select a separate pinned checkout without editing the benchmark's checkout.
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_sm100_dense_varlen.py \
  --source-root ../flash-attention-parent --include-1cta --output benchmarks/results/parent.json

# Small smoke case including multibatch packed-LSE layout.
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_sm100_dense_varlen.py \
  --batch-size 2 --seqlen 512 --nheads 4 --head-dim 128 \
  --include-1cta --output benchmarks/results/smoke.json

# Preserve the actual compiled PTX/CUBIN for disassembly, outside timing.
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_sm100_dense_varlen.py \
  --include-1cta --artifacts-dir benchmarks/results/kernel-artifacts \
  --output benchmarks/results/with-artifacts.json
```

The CLI exposes dtype, causality, batch/head counts, sequence length, device, seed, rounds, samples, replays, and warmup duration. `--include-1cta` overrides only forward CTA selection for a diagnostic arm; it preserves dense scheduling. `--restore-old-budget` applies before the first compilation in a fresh invocation. For b25 on current Quack, opt into `--b25-quack-compat`; its process-local native-operation alias corresponds to [#2787](https://github.com/Dao-AILab/flash-attention/pull/2787). Source checkouts and the global Python environment remain intact.

Full shape memory requirements are substantial; select a free B200. Compare revisions under the same compiler and include the emitted source/compiler metadata. To reproduce the original allocating-call timing, use `--do-bench`; its 20/100 ms budgets provide diagnostic corroboration only. Generated run outputs belong in the ignored `benchmarks/results/` directory; the curated reference dataset is versioned under `benchmarks/validation/`.

`--artifacts-dir` requires a new or empty directory and saves PTX/CUBIN before timed replay. Its JSON records artifact basenames, hashes, and the actual PTX compiler header without including local paths. Disassemble a saved CUBIN with `cuobjdump --dump-sass FILE.cubin` or `nvdisasm FILE.cubin`; compare source roles and loop bounds when counting spills.

## Measurement protocol

The recorded campaign uses preallocated O/LSE and one forward call per CUDA Graph, excluding allocation, compilation, capture, output comparisons, and warmup from timed replay. Each arm has seven rounds, seven samples per round, and three replays per sample: 49 samples and 147 timed calls. GPU execution is serialized; revision and mode order reverse on alternate rounds. The standalone CLI alternates its modes; a multi-revision campaign must also alternate revision order to reproduce that aspect of the archived protocol.

Reported values are medians of the seven round medians. Original S86000 arms warm up with three replays (about 270–375 ms), while the final S2048/S8192/S32768 and control campaign uses approximately 200 ms event-calibrated warmup. The CLI defaults to 200 ms warmup. The initial short-kernel campaign had order-sensitive variation even for identical binaries; extended warmup reduced paired GQA varlen changes from −13.48%…+13.04% to −1.30%…+2.88%. Clocks were not locked.

Percentage definitions:

- Varlen latency reduction: `100 * (1 - varlen / dense)`.
- Dense slowdown relative to varlen: `100 * (dense / varlen - 1)`.
- Forced dense1CTA change: `100 * (forced_1cta / dense - 1)`.

## Original reported shape

BF16, noncausal, return LSE, `(B,Sq,Sk,Hq,Hkv,Dqk,Dv)=(1,86000,86000,32,32,128,128)`.

| Version | Dense ms | Varlen ms | Varlen latency reduction |
| --- | --- | --- | --- |
| b25 source replay + API alias | 125.140 | 90.375 | 27.78% |
| #2869 parent | 125.879 | 90.263 | 28.29% |
| #2869 merged fix | 92.021 | 90.363 | 1.80% |
| Pinned main (`94e22c9`) | 92.179 | 90.201 | 2.15% |
| Pinned main, old budget restored | 126.325 | 90.658 | 28.23% |

Pinned-main dense/varlen differs by 2.19% in the reported direction, below 5% in every round. The #2869 parent/fix pair reduces dense latency by 26.90%, with varlen +0.11%. Restoring the old budget on pinned main restores the gap. The b25 campaign is separate; its dense/varlen ratio is supported directly, while cross-campaign absolute changes are descriptive rather than paired confidence estimates.

## Source and binary attribution

| Source | Commit |
| --- | --- |
| b25 source replay + API alias | [`c68c592`](https://github.com/Dao-AILab/flash-attention/commit/c68c592fd9da1e40a4fb0b56229caae6754ac5c9) |
| #2869 parent | [`62892fe`](https://github.com/Dao-AILab/flash-attention/commit/62892fe4a7582f1837ef72a206639a8563498e9e) |
| #2869 merged fix | [`117d189`](https://github.com/Dao-AILab/flash-attention/commit/117d189e871aa409e3cfb255b51dcb6f5c3f4f1f) |
| Pinned main (`94e22c9`) | [`94e22c9`](https://github.com/Dao-AILab/flash-attention/commit/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c) |

The old-budget arm uses the pinned-main commit with `_TUNING_CONFIG[(True,False,128,False)]` modified before compiling. It restores softmax=176/correction=88, with other=72 derived from the source formula. The updated budget is softmax=184/correction=80/other=64. [Merged register table](https://github.com/Dao-AILab/flash-attention/blob/117d189e871aa409e3cfb255b51dcb6f5c3f4f1f/flash_attn/cute/flash_fwd_sm100.py#L106), [other-role derivation](https://github.com/Dao-AILab/flash-attention/blob/117d189e871aa409e3cfb255b51dcb6f5c3f4f1f/flash_attn/cute/flash_fwd_sm100.py#L383).

All original-shape arms use 128×128 tiles and `q_stage=2`. Dense default is 2CTA / `StaticPersistentTileScheduler`; varlen is 1CTA / `SingleTileVarlenScheduler`. Forced dense1CTA retains the static persistent scheduler. CLC is disabled.

The parent/b25 dense binary has a 112-byte stack and 20 LDL/18 STL static instruction sites in its softmax KV scan, including spilled exponential values reloaded for sum reduction. Updated dense has a zero-byte stack and no LDL/STL. Parent/fix varlen and forced dense1CTA PTX/CUBIN are byte-identical; the same-main ablation's two control binaries also match. Current varlen retains an 8-byte stack outside softmax.

The [validation JSON](validation/sm100_dense_varlen_b200.json) records all 15 original-shape kernel configurations, PTX/CUBIN SHA256, stack/resource metadata, static spill counts and binary comparisons. Counts describe static instruction sites, not dynamic profiler traffic; the experiment does not assign every saved microsecond solely to local-memory traffic.

## Control measurements on pinned main

Defaults are BF16, noncausal, B1/Hq=Hkv32/D128. Named controls use S8192: FP16; causal; GQA Hkv8; D64; or B2/Hq=Hkv16. Negative dense/varlen values favor dense, and negative forced changes favor 1CTA.

| Case | Dense ms | Varlen ms | Dense/varlen change | Forced dense 1CTA change |
| --- | --- | --- | --- | --- |
| MHA, S86000 | 92.178741 | 90.200999 | +2.19% | -2.75% |
| MHA, S2048 | 0.064491 | 0.068555 | -5.93% | -2.48% |
| MHA, S8192 | 0.861419 | 0.882805 | -2.42% | -0.27% |
| MHA, S32768 | 13.403061 | 13.365718 | +0.28% | -1.98% |
| FP16, S8192 | 0.890869 | 0.914773 | -2.61% | -0.80% |
| Causal, S8192 | 0.475872 | 0.492053 | -3.29% | +0.00% |
| GQA (Hkv8), S8192 | 0.837163 | 0.883765 | -5.27% | +2.95% |
| D64, S8192 | 0.644939 | 0.790859 | -18.45% | +1.58% |
| B2/H16, S8192 | 0.855424 | 0.896853 | -4.62% | -0.07% |

All nine current-main cases stay below 5% dense slowdown in every round. Some controls favor dense by more than 5%, so these results do not establish universal dense/varlen parity or a universally optimal CTA selector. #2869 parent/fix control medians show no regression above 5%; the largest varlen increase is +0.13%.

## Numerical validation record

The archived campaign compiled 37 FakeTensor configurations, then passed 36 reduced real cases and ten repetitions of the original shape. Both real phases compiled zero kernels. Repetitions reuse one seeded input and the compiled kernels.

Reduced cases cover BF16/FP16 × causal/noncausal × MHA/GQA/MQA × `(Sq,Sk)=(257,513),(513,2049),(2048,2048)`, with B1/Hq8/D128. Both APIs check O/dQ/dK/dV against FP64 attention/autograd using twice the low-precision max/mean error plus dtype-rounding allowance. FP32 LSE uses `abs(error) <= 1e-5 + 1e-5*abs(FP64_reference)`; the largest normalized error is 0.485215. Three initial causal LSE checks exceeded a helper-specific absolute floor `2e-5`. Actual errors did not change; only the scratch reference criterion changed, and both real phases were rerun.

At S86000, complete dense/varlen maximum differences are O=`1.220703125e-4`, LSE=`4.76837158203125e-6`, and dQ/dK/dV each=`2.44140625e-4`. FP64 O/LSE checks stream all K/V in blocks of 4096 for ten query rows across all 32 heads. Maximum O error is `8.014509835352346e-5`; dense/varlen LSE maxima are `6.211859009397358e-6`/`5.245829717281936e-6`. Independent FP64 gradients are validated only on reduced shapes.

The numerical results are summarized in the validation JSON. The benchmark's O/LSE agreement check is a forward sanity check, not a replacement for those independent numerical checks or the existing public API test suite.

## Environment and replay qualification

NVIDIA B200/SM100; driver 580.126.20; Python 3.12.14; PyTorch 2.13.0+cu130; CUTLASS DSL 4.7.1; apache-tvm-ffi 0.1.11; quack-kernels 0.6.5; Triton 3.7.1. Actual saved PTX reports CUDA compilation tools 13.3/V13.3.27, based on NVVM 23.0.0, build CL-37800683, PTX 9.3 targeting `sm_100a`. Torch's CUDA version is not the actual CuTe codegen version.

All 52 Python CuTe modules in the PyPI `4.0.0b25` wheel match `c68c592`. Wheel SHA256: `2e6c00179c5e017bc9db81df410487485229e3be744fd3a0dc87cf3b0d0d76f3`; [release metadata](https://pypi.org/pypi/flash-attn-4/4.0.0b25/json). Replay uses historical source under the current toolchain with the explicit Quack alias, rather than reproducing the reporter's entire environment. All three adapted b25 kernels are PTX/CUBIN byte-identical to the #2869 parent.

SM110, other DSL/compiler versions, and the full repository parameterized suite were not exercised. The record contains 111 measurement arms with every final timing sample, nine configurations, five source arms, and numerical/binary summaries. It contains no workstation paths, usernames, or GPU UUIDs.
