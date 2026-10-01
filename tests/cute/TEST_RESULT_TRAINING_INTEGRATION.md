# Complete-model training integration on native Thor

The inference performance change passes full-model training integration. This
pilot establishes **no material training speedup** and does not extend the
validated inference performance claim to training.

All original Qwen3-0.6B parameters (596,049,920) and all 28 layers are present.
FP32 parameters use BF16 autocast, deterministic FA4 backward, batch 1 and
fused capturable AdamW at lr=1e-4. Each fresh process performs three warm steps
and five measured graph replays per length, for 24 actual optimizer steps.
The graph contains 28 FA forward nodes, 28 backward nodes and five fused AdamW
kernel occurrences. Loading, compilation and graph capture are outside timing.

| Complete graph training | Baseline ms | Patch ms | Pilot speed ratio |
| --- | ---: | ---: | ---: |
| 256 tokens | 157.514213 | 157.354988 | 1.0010x |
| 2048 tokens | 477.993236 | 477.732525 | 1.0005x |
| 8192 tokens | 2101.748750 | 2098.743883 | 1.0014x |

Values are geometric means of the two clean baseline endpoints and two clean
patched process medians. The fixed order was A/A/P/P/A; the second A overlaps
foreign GPU work in 26 monitoring samples and its entire timing arm is excluded.
This single pilot supplies no confidence interval or formal performance claim.

All five processes pass original-weight loss/logit/all-parameter gradient checks,
finite parameter/state checks, unchanged parameter storage, exact trained
32-token greedy output and training loss comparisons. Compared measured loss
curves are bitwise equal (maximum absolute difference 0). The native regression
matrix also passes 192 cases on each build: FP16/BF16, output or output-plus-LSE
gradients, deterministic/default backward, three GQA ratios and eight square or
rectangular lengths against an independent FP64 reference.

| Validation | Baseline | Patch |
| --- | ---: | ---: |
| Native gradient cases | 192 passed | 192 passed |
| Maximum relative RMS, nonzero reference | 0.0027025514 | 0.0027025514 |
| Maximum absolute RMS | 0.0036933939 | 0.0036933939 |
| Complete-model quality | 3 processes passed | 2 processes passed |

Baseline `616b0e8abab13b87b01525b3916d5a863ab02ae0`; tested patch
`9267a996522e33e9fdb2117acb75c50d6d0f7602` has the same production sources as
the reported inference change. Frozen training driver SHA-256:
`408ea958a3bbc4891a6a982b6e3645660ff1678306d0c24a6df6a5469a1e2858`.
Checkpoint revision `c1899de289a04d12100db370d81485cdf75e47ca`, complete
weight-file SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
Hardware: native SM110 Thor, 20 SMs, UUID `a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600`.
Runtime: Torch 2.13.0+cu130, CUDA 13.0, DSL 4.8.0, FFI 0.1.12, Quack 0.5.3.

`TRAINING_INTEGRATION_RESULTS.json` preserves raw process samples, losses,
correctness, graph topology summaries, source hashes, runner contention, JUnit hashes,
contracts and the exact harness texts. The task-owned round-4 checkpoint is
still needed by the subsequent backward iteration; cleanup remains pending.

Full raw graph and JUnit data remain in the local audit snapshot
`thor-training-round4-audit-v1.tar.gz`, SHA-256 `7fd9a4ae53363a0d632ff7cf070e6835cb8f12b52e62fafa22560b9446d97c2f`.
