# Complete-model inference performance on native Thor

The final change improves full Qwen3-0.6B greedy generation by **1.0680×** for
128 input tokens and **1.0847×** for 512 input tokens, each producing exactly 32 tokens.
This is complete-model offline inference with all original weights and all 28 layers.

## Change and provenance

Dense inference calls previously created an autograd context and populated saved tensors
and backward attributes even under `no_grad`. Ordinary attention also constructed an MLA
route and queried target-SM metadata on every call. The change uses the existing forward
implementation directly when gradients are disabled and builds the MLA route only for MLA.
The differentiable API retains `FlashAttnFunc.apply`.

On native SM110, the existing short D=128 causal packed-GQA tuning selects one Q stage
when the two query CTAs per KV head fit in a single SM wave. This spreads an underfilled
short prefill across more SMs. The kernel arithmetic and backward implementation are unchanged.

| Item | Frozen value |
|---|---|
| Upstream baseline | `616b0e8abab13b87b01525b3916d5a863ab02ae0` |
| Tested patch, including tests | `9267a996522e33e9fdb2117acb75c50d6d0f7602` |
| Patched interface SHA-256 | `04346763cbe832dcc6b6139c98005ceb6eb986ab666cc6a823a278138e88f69f` |
| Forward kernel source, both arms | `24ff53a4d10d92ac0f86f3484cc978462686b795b62fd15c13d7c3cb001889b3` |
| Full-model driver SHA-256 | `4d84f06873b9526344ee17d6670335799c70019e610aeaa5dba210431ad3befd` |
| Checkpoint | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Complete weight file | 1,503,300,328 bytes, SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Hardware | NVIDIA Thor, native SM110, 20 SMs, aarch64 |
| GPU UUID | `a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600` |
| Runtime | Torch 2.13.0+cu130; CUDA 13.0; DSL 4.8.0; TVM-FFI 0.1.12; Quack 0.5.3 |
| Model / execution | Original BF16 weights, batch 1, eager inference, Torch threads=8 |

## Complete-model speed comparison

| Thor workload | Baseline ms | Patch ms | Speedup | 95% block-bootstrap CI | Output tokens/s A → P |
|---|---:|---:|---:|---:|---:|
| 128 full-model prefill | 23.210 | 22.095 | 1.0505× | 1.0247–1.0741× | — |
| 128 input + 32 greedy tokens | 766.272 | 717.495 | 1.0680× | 1.0459–1.0898× | 41.76 → 44.60 |
| 512 full-model prefill | 23.868 | 21.787 | 1.0955× | 1.0760–1.1114× | — |
| 512 input + 32 greedy tokens | 758.452 | 699.223 | 1.0847× | 1.0517–1.1145× | 42.19 → 45.77 |

Each arm is a fresh process importing one immutable checkout. The fixed order is
**A/P/P/A × 3**, with three warm generations and 11 synchronized samples per workload
per process. Each generation creates fresh request/KV state. Weight loading, first-use
compilation and correctness checks are outside steady timing. Persistent FA4 and DSL file
caches are disabled; all arms record three compiles and zero disk hits.

Table values are geometric means of the six process medians for each variant. The interval
is the exact nonparametric bootstrap over **three independent APPA blocks** (27 draws),
with linear percentile interpolation. Inner samples are not counted as independent replicas.
The preregistered minimum gain is 1%; both complete-generation intervals clear that gate.
No foreign GPU processes were present at any measured-process boundary. Host load and GPU
snapshots are retained for every arm.

All 12 processes pass prefix-logit, exact greedy-token and all-vocabulary teacher-forced
continuation-logit checks. Every compared logits tensor is bitwise equal to the first
baseline, with maximum absolute difference **0**. Each process records **27,608 real FA4
calls**; model parameter storage addresses remain unchanged. No layers, weights or generated
tokens are reduced. Full-model prefill uses `use_cache=False`; complete generation uses
`use_cache=True`. This report measures offline execution rather than an HTTP serving system.

## Public eager API latency

B=1, 16 Q heads / 8 KV heads, D=128, causal packed GQA, no sink, one split. K/V use
head-major views as in a growing dense KV cache. Four fresh A/P/P/A processes each collect
15 batches of 200 calls after compilation/warmup; wall time includes dispatch, allocation,
launches and final device synchronization.

| dtype | Q × K | Baseline us | Patch us | Eager API speedup |
|---|---:|---:|---:|---:|
| bfloat16 | 1 × 129 | 48.468 | 32.334 | 1.4990× |
| bfloat16 | 1 × 512 | 48.115 | 32.166 | 1.4958× |
| bfloat16 | 128 × 128 | 49.942 | 36.923 | 1.3526× |
| bfloat16 | 512 × 512 | 50.629 | 33.706 | 1.5021× |
| float16 | 1 × 129 | 49.011 | 32.462 | 1.5098× |
| float16 | 1 × 512 | 48.708 | 31.974 | 1.5233× |
| float16 | 128 × 128 | 50.044 | 36.367 | 1.3761× |
| float16 | 512 × 512 | 50.436 | 33.833 | 1.4907× |

These eager API measurements include host overhead. The earlier isolated GPU-kernel
speed/control matrix and native NCU reports remain in
[TEST_RESULT_SM110_SHORT_PREFILL.md](TEST_RESULT_SM110_SHORT_PREFILL.md).
That kernel-only revision had no established complete-model speedup; the table above tests
the final dispatch and scheduling changes together.

## Correctness and integration

| Validation | Result |
|---|---|
| Final Thor native output/LSE/gradient tests | 44 GPU cases passed; 2 additional CPU dispatch cases passed |
| Final H20 inference tests | 16 GPU cases passed; 2 additional CPU dispatch cases passed |
| Final Thor selected-prefill views/streams/graphs | 5 fresh-process probes passed, each with two independently checked mutated inputs |
| Unaligned offset1 eager diagnostic | Passed on native H20 and Thor; the FA input path canonicalizes alignment |
| Fullgraph compilation diagnostic | Unmodified baseline fails on H20/Thor at the Dynamo-skipped `active_fake_mode`; final Thor has the same failure |
| Complete-model inference | 12/12 processes passed; all output tokens and compared logits exactly match |

The native tests cover FP16/BF16, causal/noncausal dense GQA, optional FP32 sinks, inputs with
and without `requires_grad`, both `no_grad` and `inference_mode`, and the differentiable
output-plus-LSE gradient route. Shared-KV MLA argument normalization is checked by two CPU
dispatch tests. Those two tests do not claim native MLA kernel coverage.

A separate CPU profile of the complete model observes 2,772 FA4 calls in either arm:
autograd `apply` calls fall from 2,772 to zero; SM-selection helper calls fall from 2,772 to
168 short-prefill selections. Profiled call medians fall from 196.654 to 136.654 us. These
instrumented durations are excluded from performance tables. Native NCU evidence for the
selected 128×128 prefill confirms 8→16 CTAs and 0.40→0.80 SM waves.

H20 complete-model replication is queued around other workloads. Two completed arms pass
exact token/logit checks, but the patched arm ends with a foreign GPU process present.
Its timing is excluded from performance claims. No H20 complete-model speedup is claimed.
Other GPU architectures, native MLA, FP8 and fullgraph integration are unvalidated here.

## Reproduction and retained artifacts

Native regression commands from a checkout using the intended FA4 import path:

```bash
python -m pytest -q tests/cute/test_inference_dispatch.py
python -m pytest -q tests/cute/test_sm110_short_prefill.py  # native Thor
```

`INFERENCE_PERFORMANCE_RESULTS.json` contains every process/sample, source hashes,
checkpoint manifest, correctness differences, boundary snapshots, profiles, integration
results and the exact frozen benchmark script texts. Extract the scripts for reproduction:

```python
import json
from pathlib import Path

record = json.loads(Path("tests/cute/INFERENCE_PERFORMANCE_RESULTS.json").read_text())
for name, source in record["harnesses"].items():
    Path(name).write_text(source)
```

Run `model_e2e_final.py --mode generation --model /path/to/full-checkpoint
--label A0 --reference reference.pt --output A0.json --expected-sm 110
--write-reference --rounds 11 --warm-generations 3` with the baseline checkout as cwd and
its FA4 import path. Subsequent A/P/P/A arms use their own checkout/import path and omit
`--write-reference`. The preserved runner records the hardware/process boundaries.

The local binary/source audit archive SHA-256 is
`fe38c50f3411a28f5553a6b9e7b39dca970d1673071f035cf84b2666304dd0ae`.
After the final full-model profiler exited naturally, the task-owned Thor checkpoint was
removed: **1,519,182,365 bytes in seven files**, with the model directory absent. H20's
previous download cleanup also completed naturally. A separate reader-aware cleanup worker
covers the still-queued H20 replication checkpoint. Existing environments and caches are
preserved; no process was killed and lcpu NFS was untouched.
