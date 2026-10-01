# Native SM110 short packed-GQA prefill — test result

Tested 2026-10-01 on native NVIDIA Thor/aarch64, driver 595.78, Torch 2.13.0+cu130,
CUTLASS DSL 4.8.0, TVM FFI 0.1.12 and Quack 0.5.3. Dependencies use task-local targets;
existing environments were retained. Baseline: `616b0e8abab13b87b01525b3916d5a863ab02ae0`.
Production: `69264336fb13fac29d2351cc9ac9c2467d33b505`; tested head: `9eb202c6162e8cc408cc2ffe9071131692e80252`.

## Change and correctness

Short causal packed-GQA D=128 prefill can issue only one double-Q-stage CTA per KV head.
When packed Q length is in (128,256] and two single-Q-stage CTAs per KV head fit in one SM
wave, SM110 now selects one Q stage. The production diff is nine lines in `interface.py`.
Longer sequences, larger batches and other head dimensions keep the existing selection.
Both baseline and patch pass all 28 native FP16/BF16 cases comparing output, LSE, dQ, dK,
dV and optional dSink with independent FP32 attention. Cases cover GQA ratios 2/4/8,
nonmultiple sequence lengths, the packed-Q threshold, batch 2 and sink=8.
Aligned contiguous, offset-8 and outer-strided views pass independent CPU FP64 checks on
the selected Q=128, K=129, GQA=2 configuration. Offset-8 also passes nondefault-stream
ordering and warmed CUDA-graph checks with mutated inputs. No torch.compile fullgraph claim.

## Steady forward timing

Causal, 16 Q heads, packed GQA, num_splits=1, no sink, no returned LSE.
Four fresh A/P/P/A processes; each case warms a CUDA graph for >=0.5 s.
15 rounds ×20 replays, 100 calls/graph (10 for Q=4096); JIT is excluded.
Columns average the two per-process medians. No competing compute processes were observed
at arm boundaries. GPU clocks were not locked. Unchanged controls are included.

| Dtype | B | Q × K | KV heads | D | Baseline us | Patch us | Baseline / patch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | 1 | 128 × 128 | 8 | 128 | 6.678 | 5.899 | 1.1320x |
| bfloat16 | 1 | 128 × 257 | 8 | 128 | 10.415 | 8.866 | 1.1746x |
| bfloat16 | 1 | 65 × 129 | 8 | 128 | 8.310 | 6.687 | 1.2427x |
| bfloat16 | 1 | 64 × 128 | 4 | 128 | 6.576 | 5.579 | 1.1786x |
| bfloat16 | 1 | 32 × 64 | 2 | 128 | 6.561 | 5.496 | 1.1937x |
| bfloat16 | 2 | 128 × 128 | 8 | 128 | 7.035 | 7.033 | 1.0003x |
| bfloat16 | 1 | 129 × 129 | 8 | 128 | 7.830 | 7.832 | 0.9997x |
| bfloat16 | 1 | 512 × 512 | 8 | 128 | 16.648 | 16.643 | 1.0003x |
| bfloat16 | 1 | 1024 × 1024 | 8 | 128 | 47.293 | 47.420 | 0.9973x |
| bfloat16 | 1 | 4096 × 4096 | 8 | 128 | 636.654 | 637.686 | 0.9984x |
| bfloat16 | 1 | 1 × 512 | 8 | 128 | 9.137 | 9.139 | 0.9998x |
| bfloat16 | 1 | 128 × 128 | 8 | 64 | 5.553 | 5.557 | 0.9992x |
| float16 | 1 | 128 × 128 | 8 | 128 | 6.682 | 5.974 | 1.1186x |
| float16 | 1 | 128 × 257 | 8 | 128 | 10.338 | 8.908 | 1.1605x |
| float16 | 1 | 65 × 129 | 8 | 128 | 8.325 | 6.697 | 1.2430x |
| float16 | 1 | 64 × 128 | 4 | 128 | 6.612 | 5.667 | 1.1668x |
| float16 | 1 | 32 × 64 | 2 | 128 | 6.591 | 5.566 | 1.1842x |
| float16 | 2 | 128 × 128 | 8 | 128 | 7.098 | 7.106 | 0.9988x |
| float16 | 1 | 129 × 129 | 8 | 128 | 7.916 | 7.917 | 0.9998x |
| float16 | 1 | 512 × 512 | 8 | 128 | 16.633 | 16.629 | 1.0003x |
| float16 | 1 | 1024 × 1024 | 8 | 128 | 51.306 | 51.281 | 1.0005x |
| float16 | 1 | 4096 × 4096 | 8 | 128 | 693.452 | 689.979 | 1.0050x |
| float16 | 1 | 1 × 512 | 8 | 128 | 9.153 | 9.153 | 1.0000x |
| float16 | 1 | 128 × 128 | 8 | 64 | 5.451 | 5.452 | 0.9999x |

The five selected shapes improve by 1.1186–1.2430x across FP16/BF16.

## NCU confirmation

Nsight Compute 2026.1.1, native SM110, full set (41 replay passes), warmed B=1,
Q=K=128, 16 Q /8 KV heads, BF16. Clock/cache controls are disabled.
NCU profiling times are excluded from the steady timing table. The launch metrics confirm
the intended change in work distribution; the existing kernel variants execute the work.

| NCU metric | Baseline | Patch |
| --- | ---: | ---: |
| Grid CTAs | 8 | 16 |
| SM waves | 0.40 | 0.80 |
| Registers/thread | 128 | 128 |
| Allocated shared memory KiB/CTA | 228 | 227 |

## Complete Qwen3-0.6B e2e

Original complete BF16 checkpoint: revision `c1899de289a04d12100db370d81485cdf75e47ca`;
safetensors SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`
(1,503,300,328 bytes). All files match the pinned checkpoint manifest. Transformers 5.17.0
uses native FA4 with the standard SDPA mask builder; model layers, RoPE, weights and KV cache
are retained. Batch 1, all-valid mask, 128/512 input tokens, exactly 32 greedy output tokens.
Only the 128-token prefill selects the new configuration; decode and 512-token prefill are controls.

Four fresh A/P/P/A processes pass prefix logits, exact 32 greedy tokens, and all vocabulary
logits for the 32-token teacher-forced continuation. Each records 16,632 actual FA4 calls
and unchanged parameter storage addresses. Seven steady complete-generate wall-time samples
and seven full-model prefill samples are CUDA-synchronized; model load and first prefill/JIT
are excluded. Persistent FA4 cache is disabled. This is offline full-model inference.

| Full-model workload | Baseline ms | Patch ms | Baseline / patch | Correctness |
| --- | ---: | ---: | ---: | --- |
| 128-token prefill | 23.544 | 23.940 | 0.9835x | PASS |
| 128-token +32-token generate | 772.089 | 790.193 | 0.9771x | PASS |
| 512-token prefill | 24.131 | 24.648 | 0.9790x | PASS |
| 512-token +32-token generate | 764.297 | 771.964 | 0.9901x | PASS |

Complete-generation wall times drift across processes, including the unchanged 512-token
control: baseline medians range 745–783 ms and patch medians 757–787 ms there.
This quartet does not establish a full-model speedup. The measured gain is in the selected
GPU attention kernels above; the e2e runs establish model-level numerical compatibility.

## Audit and cleanup

`SM110_SHORT_PREFILL_RESULTS.json` retains every timing sample, exact source/import hashes,
PIDs, package versions, NCU metrics, checkpoint manifest and model-worker exit records.
Benchmark driver SHA256: `9380461de5cd39d390fd2f665e0b52f591f49d82870ebcdaf14672ce15d45793`.
Model driver SHA256: `c109f3f9b8f09171bfe0b127eddf9c4ec3702bc54f6e7678b8533e610cba58d8`.
Full NCU reports and drivers are retained in the execution audit.
The verified checkpoint weight file was removed immediately after all four model workers
exited naturally. The original download writer also exited naturally; the cleanup worker
removed its remaining 1,519,184,759 bytes and all 24 files in the task-owned model directory.
The model directory no longer exists. Pre-existing processes and environments were retained.
