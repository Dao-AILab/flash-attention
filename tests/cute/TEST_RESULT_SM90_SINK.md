# SM90 learnable-sink normalization test result

Baseline: `616b0e8abab13b87b01525b3916d5a863ab02ae0`.
Tested patch: `4566e1b988727a24d3c1edf430986340a23820b4`.
The final report commit changes validation artifacts only.

The final denominator, output multiplier and LSE are rebased against the
larger of the attention maximum and the sink logit. No-sink specialization
and the online KV loop are unchanged.

## Correctness

| Check | Baseline | Patch |
| --- | --- | --- |
| Real SM90 analytic sink=100 regression | Failed: infinite LSE | Passed: finite LSE=100 |
| New analytic, masking, GQA and forward/backward cases | — | 76 passed, no skips/xfails |
| Existing sink dtype and varlen/LSE backward regressions | — | 6 passed, original criteria |
| Qwen3-0.6B complete inference, both H20s | Native FA4 reference | Exact logits and generated tokens |
| Syntax, Ruff lint/format and whitespace checks | — | Passed |

Coverage includes FP16/BF16 QKV, FP16/BF16/FP32 sinks, packed/unpacked GQA,
dense/causal/local masks, dimensions 64/96/128/192/256, empty/fully masked
rows, sink=-inf, non-default scale, and a dominant-sink pure-LSE gradient.
No production backward implementation was changed.

Reference calibration was required: the initial strict FP64 elementwise
gradient checks produced 72 passes and four failures; the same four cases
also failed on the baseline. Using the unchanged upstream reordered-PT QKV
gradient criteria passed all QKV checks. For dsink, cancellation also required
a per-head precision floor derived only from independent FP64/PT forward
outputs and dout. Existing tests and their tolerances were unchanged. The
new dsink check uses that reference-only budget; it does not construct a
tolerance from the kernel output. Intermediate failures remain in the
validation history JSON.

## CUDA graph forward latency

Each GPU/mode uses one fresh-process A/P/P/A quartet. Each process records
15 rounds, each 20 graph replays of 10 calls, after compilation and warmup.
Values below are geometric means of the two process medians per source.
Relative speed = baseline / patch; values below 1 mean a slowdown. One
quartet per GPU/mode does not establish a statistically significant gain.
All shapes use B=1, Hq=16, Hkv=4, BF16, packed GQA, one split, non-causal,
and fixed allocations. This measures steady warm graph replay, not cold JIT
or streaming HBM bandwidth. Units are microseconds per forward call.

| GPU | LSE | Sq/Sk/D | Sink | Baseline us | Patch us | Relative speed |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 1/4096/64 | none | 113.117 | 113.134 | 0.9998x |
| 0 | 0 | 1/4096/64 | 0 | 113.444 | 113.133 | 1.0028x |
| 0 | 0 | 1/4096/64 | 8 | 113.404 | 112.879 | 1.0046x |
| 0 | 0 | 128/128/64 | none | 6.329 | 6.323 | 1.0009x |
| 0 | 0 | 128/128/64 | 0 | 6.442 | 6.523 | 0.9876x |
| 0 | 0 | 128/128/64 | 8 | 6.446 | 6.542 | 0.9853x |
| 0 | 0 | 1024/1024/128 | none | 80.301 | 80.246 | 1.0007x |
| 0 | 0 | 1024/1024/128 | 0 | 80.422 | 79.808 | 1.0077x |
| 0 | 0 | 1024/1024/128 | 8 | 80.397 | 79.791 | 1.0076x |
| 0 | 0 | 4096/4096/128 | none | 1052.587 | 1052.596 | 1.0000x |
| 0 | 0 | 4096/4096/128 | 0 | 1055.582 | 1053.410 | 1.0021x |
| 0 | 0 | 4096/4096/128 | 8 | 1055.519 | 1053.369 | 1.0020x |
| 0 | 1 | 1/4096/64 | none | 113.201 | 113.173 | 1.0003x |
| 0 | 1 | 1/4096/64 | 0 | 113.560 | 113.523 | 1.0003x |
| 0 | 1 | 1/4096/64 | 8 | 113.493 | 113.480 | 1.0001x |
| 0 | 1 | 128/128/64 | none | 6.558 | 6.584 | 0.9960x |
| 0 | 1 | 128/128/64 | 0 | 6.579 | 6.575 | 1.0006x |
| 0 | 1 | 128/128/64 | 8 | 6.594 | 6.589 | 1.0007x |
| 0 | 1 | 1024/1024/128 | none | 80.049 | 80.089 | 0.9995x |
| 0 | 1 | 1024/1024/128 | 0 | 80.329 | 80.188 | 1.0018x |
| 0 | 1 | 1024/1024/128 | 8 | 80.302 | 80.167 | 1.0017x |
| 0 | 1 | 4096/4096/128 | none | 1052.008 | 1052.037 | 1.0000x |
| 0 | 1 | 4096/4096/128 | 0 | 1052.283 | 1051.807 | 1.0005x |
| 0 | 1 | 4096/4096/128 | 8 | 1052.215 | 1051.795 | 1.0004x |
| 1 | 0 | 1/4096/64 | none | 111.825 | 111.828 | 1.0000x |
| 1 | 0 | 1/4096/64 | 0 | 112.150 | 111.680 | 1.0042x |
| 1 | 0 | 1/4096/64 | 8 | 112.110 | 111.713 | 1.0035x |
| 1 | 0 | 128/128/64 | none | 6.267 | 6.283 | 0.9975x |
| 1 | 0 | 128/128/64 | 0 | 6.365 | 6.477 | 0.9826x |
| 1 | 0 | 128/128/64 | 8 | 6.401 | 6.490 | 0.9864x |
| 1 | 0 | 1024/1024/128 | none | 79.053 | 79.333 | 0.9965x |
| 1 | 0 | 1024/1024/128 | 0 | 79.356 | 79.009 | 1.0044x |
| 1 | 0 | 1024/1024/128 | 8 | 79.330 | 78.995 | 1.0042x |
| 1 | 0 | 4096/4096/128 | none | 1040.151 | 1035.169 | 1.0048x |
| 1 | 0 | 4096/4096/128 | 0 | 1039.171 | 1036.104 | 1.0030x |
| 1 | 0 | 4096/4096/128 | 8 | 1038.142 | 1036.012 | 1.0021x |
| 1 | 1 | 1/4096/64 | none | 111.524 | 111.510 | 1.0001x |
| 1 | 1 | 1/4096/64 | 0 | 111.863 | 111.861 | 1.0000x |
| 1 | 1 | 1/4096/64 | 8 | 111.822 | 111.808 | 1.0001x |
| 1 | 1 | 128/128/64 | none | 6.541 | 6.503 | 1.0059x |
| 1 | 1 | 128/128/64 | 0 | 6.533 | 6.519 | 1.0021x |
| 1 | 1 | 128/128/64 | 8 | 6.540 | 6.535 | 1.0008x |
| 1 | 1 | 1024/1024/128 | none | 78.966 | 78.972 | 0.9999x |
| 1 | 1 | 1024/1024/128 | 0 | 79.248 | 79.104 | 1.0018x |
| 1 | 1 | 1024/1024/128 | 8 | 79.224 | 79.080 | 1.0018x |
| 1 | 1 | 4096/4096/128 | none | 1034.678 | 1034.718 | 1.0000x |
| 1 | 1 | 4096/4096/128 | 0 | 1034.914 | 1034.486 | 1.0004x |
| 1 | 1 | 4096/4096/128 | 8 | 1034.861 | 1034.404 | 1.0004x |

Sink=100 timings are retained below, but the baseline has an invalid LSE;
these are not valid speedup claims.

| GPU | LSE | Sq/Sk/D | Sink | Baseline us | Patch us | Relative speed |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 1/4096/64 | 100 | 113.411 | 113.030 | N/A (invalid baseline LSE) |
| 0 | 0 | 128/128/64 | 100 | 6.402 | 6.504 | N/A (invalid baseline LSE) |
| 0 | 0 | 1024/1024/128 | 100 | 80.443 | 79.834 | N/A (invalid baseline LSE) |
| 0 | 0 | 4096/4096/128 | 100 | 1055.597 | 1053.398 | N/A (invalid baseline LSE) |
| 0 | 1 | 1/4096/64 | 100 | 113.499 | 113.474 | N/A (invalid baseline LSE) |
| 0 | 1 | 128/128/64 | 100 | 6.559 | 6.552 | N/A (invalid baseline LSE) |
| 0 | 1 | 1024/1024/128 | 100 | 80.344 | 80.204 | N/A (invalid baseline LSE) |
| 0 | 1 | 4096/4096/128 | 100 | 1052.261 | 1051.816 | N/A (invalid baseline LSE) |
| 1 | 0 | 1/4096/64 | 100 | 112.117 | 111.583 | N/A (invalid baseline LSE) |
| 1 | 0 | 128/128/64 | 100 | 6.353 | 6.448 | N/A (invalid baseline LSE) |
| 1 | 0 | 1024/1024/128 | 100 | 79.636 | 79.039 | N/A (invalid baseline LSE) |
| 1 | 0 | 4096/4096/128 | 100 | 1038.215 | 1036.089 | N/A (invalid baseline LSE) |
| 1 | 1 | 1/4096/64 | 100 | 111.811 | 111.826 | N/A (invalid baseline LSE) |
| 1 | 1 | 128/128/64 | 100 | 6.509 | 6.499 | N/A (invalid baseline LSE) |
| 1 | 1 | 1024/1024/128 | 100 | 79.263 | 79.125 | N/A (invalid baseline LSE) |
| 1 | 1 | 4096/4096/128 | 100 | 1034.925 | 1034.492 | N/A (invalid baseline LSE) |

## Full-model end-to-end latency

Full `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
checkpoint, BF16, eager Transformers with every attention layer calling
native FA4, batch 1, unpadded input, greedy generation of exactly 32 tokens.
Each GPU uses a separate A/P/P/A quartet with two warmups and five complete
generations per process. Each length has 4,480 measured FA4 calls per process;
weights retain their storage addresses. JIT/model load is outside this steady
generation timer. Units are milliseconds per complete generation.

| GPU | Input tokens | Output tokens | Baseline ms | Patch ms | Relative speed | Logits/tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 128 | 32 | 647.096 | 646.585 | 1.0008x | Exact |
| 0 | 512 | 32 | 646.676 | 643.931 | 1.0043x | Exact |
| 1 | 128 | 32 | 642.545 | 648.315 | 0.9911x | Exact |
| 1 | 512 | 32 | 632.538 | 652.723 | 0.9691x | Exact |

The full model has no learnable sink, so it checks the no-sink regression
path. Sink semantics are checked independently above. The reference is the
same native upstream FA4 backend. An initial BF16 SDPA comparison diverged
after 20 of 32 common tokens; layer-level FP64 checks found comparable local
errors for both backends. SDPA bitwise/model equality is not claimed.

## Provenance and reproduction

Measured on 2026-10-01, two NVIDIA H20 (SM90, 97,871 MiB each),
driver 580.105.08. Torch 2.7.0+cu128 and CUDA 12.8 were reused unchanged;
CuTeDSL 4.8.0, Quack 0.5.3, TVM-FFI 0.1.12, torch-c-dlpack-ext 0.1.5,
and Transformers 4.57.1 were installed in a task-owned dependency directory.
Quack 0.6.5 requires Torch dtypes absent from Torch 2.7. Quack 0.5.3's
metadata pins DSL 4.6.0.dev0, which conflicts with FA4's DSL minimum. This
known metadata conflict remains; the unmodified Quack 0.5.3 package imported
and ran the actual kernels. `pip check` is not clean.

GPU UUIDs, process IDs, source/checkpoint hashes, raw samples and the exact
ordered process plan are in `results/sm90_sink_measurements.json`.
All processes finished naturally, with no overlapping timed arms. Only the
task-owned model checkpoint was removed after the final model arm.
This finalizer has an SM80 caller; validation here covers H20/SM90 only.

Run `test_sm90_sink_stability.py` and the existing
`test_flash_attn.py -k 'learnable_sink_backward_dtype or varlen_learnable_sink_backward_with_lse'`
under the pinned runtime. The replay and model harnesses are in
`validation_sm90/benchmark_sink.py` and `validation_sm90/model_e2e.py`.
Run each baseline/patch arm in its own source checkout and Python process;
disable persistent caching for these timing arms. Model arms use
`--model <checkpoint-directory> --label <arm> --rounds 5 --reference <json>`;
only the first baseline arm also uses `--write-reference`.
