# SM90 learnable-sink normalization test result

Baseline: `616b0e8abab13b87b01525b3916d5a863ab02ae0`.

The patch rebases the final denominator and output multiplier against the larger
of the attention maximum and the sink logit. The sink remains in natural-log
units. The no-sink specialization and the online KV loop are unchanged.

| Check | Result |
| --- | --- |
| Python syntax, production edit and new regressions | Passed |
| Ruff lint and format, repository configuration | Passed |
| Whitespace validation | Passed |
| New sink regression cases | 76 cases added; GPU execution pending |
| Existing sink dtype and varlen/LSE backward regressions | GPU execution pending |

## Speed comparison

| Workload | Baseline latency | Patched latency | Relative speed | Status |
| --- | ---: | ---: | ---: | --- |
| H20 CUDA graph forward, no sink | — | — | — | Not measured |
| H20 CUDA graph forward, sink 0 / 8 | — | — | — | Not measured |
| Qwen3-0.6B, 128 / 512 input tokens and 32 output tokens | — | — | — | Not measured |

Both provisioned GPUs are NVIDIA H20 with compute capability 9.0. Existing
Torch is `2.7.0+cu128`. CuTeDSL, Quack, TVM-FFI and pytest are absent from that
environment. Dependency setup is pending; no GPU correctness or speed result
is claimed. The full-model checkpoint is pinned to
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.

The full-model check is a no-sink regression. The sink assertions independently
cover output, finite LSE, dominant-sink LSE gradients, causal/local masking,
packed GQA, and head dimensions 64, 96, 128, 192 and 256. This shared finalizer
also has an SM80 caller; no cross-architecture validation has been performed.
