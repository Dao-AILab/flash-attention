# Native Thor persistent-cache validation

This supplements the H20 results in `TEST_RESULT_QUACK_CACHE.md` with an actual installed
Quack version transition, not a mocked stamp. Native SM110/aarch64; driver 595.78;
Torch 2.13.0+cu130, CUDA 13.0, CUTLASS DSL 4.8.0, TVM FFI 0.1.12. Source is
`a9b862268a3fd8ce289dfbafceb2d27ac3738ded`; all dependency changes are task-local
Python target directories, preserving the original environment.

The six actual imported fingerprint tests pass against the patch. Upstream baseline passes
three and fails three (Quack fingerprint, namespace isolation and missing-version handling).
Fresh processes use actual Quack 0.5.3 and 0.6.0 on the same FA source/runtime, with one shared
persistent-cache root per quartet. A1 compiles, A2 loads from disk, B1 misses A's namespace
and recompiles, B2 reuses B. Standalone fingerprints are:

- 0.5.3: `f5c6508bf072602a080e75bfd1c974d9f8db479a4aa2339850dc2ffa911b8009`
- 0.6.0: `ca663996d30101f0c135097d83d116dc278f846092fd153263c1afc06d52e633`

## First-call timing

These times exclude Python imports and model/weight loading. They measure the first real
forward/prefill, including JIT or persistent loading. Compiles and hits are direct observations
of CuTe compilation and the FA cache loader during that measured call.

| Fresh-process workload | Cold ms | Disk reuse ms | Relative first-call speed | Compiles | Disk hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quack 0.5.3, standalone forward | 2269.105 | 4.532 | 500.74x | 1 → 0 | 0 → 1 |
| Quack 0.6.0, standalone forward | 2329.626 | 4.455 | 522.91x | 1 → 0 | 0 → 1 |
| Quack 0.5.3, full Qwen3 first 128-token prefill | 3528.252 | 591.893 | 5.961x | 1 → 0 | 0 → 1 |
| Quack 0.6.0, full Qwen3 first 128-token prefill | 3431.869 | 611.381 | 5.613x | 1 → 0 | 0 → 1 |

Across the complete 128/512-token generation cases each producer compiles three FA programs
with zero disk hits; each consumer has three disk hits and zero compiles. Changing Quack
selects a new namespace despite retained A binaries. Every full-model arm matches upstream
FA4 last-prefix logits, all 32 generated tokens and all 32 teacher-forced vocabulary-logit rows.
The full unchanged Qwen3-0.6B checkpoint/revision and BF16 execution match the SM110 sink report.

## Steady complete-model generation

Five complete-generate wall samples per arm, batch 1 and exactly 32 output tokens; model/JIT
warmup is outside timing. Cold/disk arms below are separate processes, not A/P speed tests.

| Quack | Input + output tokens | Cold-process steady ms | Reuse-process steady ms | Cold / reuse |
| --- | --- | ---: | ---: | ---: |
| 0.5.3 | 128 + 32 | 782.886 | 774.437 | 1.0109x |
| 0.5.3 | 512 + 32 | 786.252 | 776.047 | 1.0131x |
| 0.6.0 | 128 + 32 | 781.985 | 782.500 | 0.9993x |
| 0.6.0 | 512 + 32 | 801.005 | 754.060 | 1.0623x |

Individual cold/reuse pairs establish cache behavior and first-call latency on this runtime;
they do not establish a steady inference speedup caused by the fingerprint change.
All processes use the same driver SHA256 `ee21c09298ed273b9053a2d3c666d7e835864a8c621e72f11d638fd8997129a5`.
`THOR_QUACK_CACHE_RESULTS.json` retains raw timing samples, exact source paths/hashes,
versions, PIDs, observed compile/load counts and checkpoint/cleanup manifests.
Initial audit metadata accidentally chose a shadowed distribution's version; those pilot
records were retained privately and replaced here by fresh runs with active-path metadata.

The task-owned 1,519,184,758-byte checkpoint/tokenizer directory was removed after all eight
complete-model validation arms finished naturally. Existing global model caches and environments, and
unrelated processes were untouched.
