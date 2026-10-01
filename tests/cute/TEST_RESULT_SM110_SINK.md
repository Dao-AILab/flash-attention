# Native SM110 sink normalization — test result

Tested 2026-10-01 on NVIDIA Thor, native SM110/aarch64, driver 595.78,
Torch 2.13.0+cu130/CUDA 13.0, CUTLASS DSL 4.8.0, TVM FFI 0.1.12 and Quack 0.5.3.
Dependencies were installed in task-local target directories; the existing environment was retained.

Baseline: `616b0e8abab13b87b01525b3916d5a863ab02ae0`.
Production change: `a44ce298fb671076db44baa5f0cd63bff8e32e78`.
Regression tests: `bd965234c324e8b9e6b726c49964ec874409232d`.
This independent branch starts at upstream main; it does not depend on the SM90 sink PR.

## Correctness

The native baseline fails the zero-score, sink=100 analytic case: LSE is infinite and dSink
is zero instead of 7. The patch matches the CPU FP64 reference for output, LSE and all four gradients.
156 analytic/random-gradient cases passed (FP16/BF16 QKV; FP16/BF16/FP32 sink; packed/unpacked
GQA; rectangular causal masking and fully masked rows). Another 18 SplitKV/empty-key cases
passed, including first-split sink ownership and a disabled sink (-infinity). Six unchanged
upstream sink dtype/varlen-LSE cases passed. Backward production code is unchanged.
Aligned contiguous, offset-8 and outer-strided tensors passed eager checks; offset-8 also passed
nondefault-stream and warmed CUDA-graph checks. These checks do not establish torch.compile fullgraph support.

## Forward timing

Causal BF16, batch 1, 16 Q heads / 4 KV heads, packed GQA, num_splits=1, returned LSE.
Each arm is a fresh A/P/P/A process. Each case warms a captured graph for at least 0.5 s;
15 rounds × 20 replays use 100 calls/graph for lengths <=1024 and 10 for length 4096.
JIT compilation is outside timing. Columns pool the two per-process medians for each variant.
No competing GPU processes were observed at the arm boundaries. GPU clocks were not locked.
The initial five-replay warmup showed a timing transition on short shapes; that pilot is retained
in the audit archive and excluded from this table. This is a correctness fix, not a speedup claim.
The sink=100 baseline is numerically invalid; its latency only describes the cost of the repair.

| Q × K, D | Sink | Baseline us | Patch us | Baseline / patch |
| --- | ---: | ---: | ---: | ---: |
| 1 × 4096, 64 | none | 36.015 | 36.018 | 0.9999x |
| 1 × 4096, 64 | 0.0 | 35.940 | 36.103 | 0.9955x |
| 1 × 4096, 64 | 8.0 | 35.939 | 35.984 | 0.9987x |
| 1 × 4096, 64 | 100.0 | 35.961 | 35.986 | 0.9993x |
| 128 × 128, 64 | none | 5.506 | 5.501 | 1.0008x |
| 128 × 128, 64 | 0.0 | 5.628 | 5.804 | 0.9698x |
| 128 × 128, 64 | 8.0 | 5.628 | 5.691 | 0.9891x |
| 128 × 128, 64 | 100.0 | 5.629 | 5.703 | 0.9872x |
| 1024 × 1024, 128 | none | 47.038 | 47.248 | 0.9956x |
| 1024 × 1024, 128 | 0.0 | 47.278 | 47.278 | 1.0000x |
| 1024 × 1024, 128 | 8.0 | 47.476 | 47.572 | 0.9980x |
| 1024 × 1024, 128 | 100.0 | 47.009 | 47.149 | 0.9970x |
| 4096 × 4096, 128 | none | 655.601 | 655.877 | 0.9996x |
| 4096 × 4096, 128 | 0.0 | 656.978 | 657.577 | 0.9991x |
| 4096 × 4096, 128 | 8.0 | 658.070 | 658.813 | 0.9989x |
| 4096 × 4096, 128 | 100.0 | 655.246 | 656.145 | 0.9986x |

## Complete Qwen3-0.6B generation

All original checkpoint weights were loaded in BF16: revision
`c1899de289a04d12100db370d81485cdf75e47ca`, safetensors SHA256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` (1,503,300,328 bytes).
Transformers 5.17.0 registers native FA4 attention with the standard SDPA mask builder.
Batch 1, all-valid attention mask, causal packed GQA, 128/512 input tokens, exactly 32 greedy
output tokens. The model's layers, weights, RoPE and cache behavior are retained.
Qwen3 has no learnable-sink parameter: this full-model run checks the unchanged no-sink path;
the analytic and random-gradient tests above exercise the repaired sink path.

Four fresh A/P/P/A processes each execute both input lengths. First prefill/JIT and model load
are outside steady timing; each arm records five complete generate wall-clock samples with a
CUDA synchronization. All arms match upstream FA4 last-prefix logits, all 32 greedy tokens,
and all vocabulary logits for the 32-token teacher-forced continuation. Every arm records
12,656 FA4 attention calls and unchanged parameter storage addresses. Cache is disabled.

| Full-model workload | Baseline ms | Patch ms | Baseline / patch | Correctness |
| --- | ---: | ---: | ---: | --- |
| 128 + 32 tokens | 783.540 | 776.529 | 1.0090x | PASS |
| 512 + 32 tokens | 779.681 | 785.584 | 0.9925x | PASS |

Ratios are descriptive: one quartet does not establish an inference speedup.
This is offline complete-model generation, without HTTP scheduling/network latency.
Native hardware validation covers SM110 FP16/BF16. SM100 and FP8 are not validated here.

## Reproduction and audit

The regression test is `test_sm110_sink_stability.py`; run it from an installed FA4 namespace,
not the legacy FA2 repository-root package. The benchmark drivers and pinned package hashes
are in the execution audit. Driver SHA256: `ee21c09298ed273b9053a2d3c666d7e835864a8c621e72f11d638fd8997129a5`.
The committed `SM110_SINK_RESULTS.json` contains all forward samples, complete-model samples,
PIDs, exact imports/source hashes and the checkpoint/cleanup manifest. Each arm started with
no compute process on the GPU, and recorded no foreign GPU process at its end.
No pre-existing processes or environments were changed.

All eight complete-model sink/cache arms exited naturally before cleanup. The task-owned
checkpoint directory (1,519,184,758 bytes including tokenizer and local metadata) was removed.
Reference logits/token outputs and audit files were retained; they contain no model weights.
