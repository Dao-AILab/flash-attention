# Quack persistent-cache fingerprint test result

Baseline: `616b0e8abab13b87b01525b3916d5a863ab02ae0`.
Tested patch: `ce65708e6254395996e4720eb07fabb2b643c13d`.
The final report commit changes validation artifacts only.

The installed Quack distribution version joins the existing source/runtime
fingerprint. Process-local memoization and disabled-cache behavior are preserved.

| Check | Baseline | Patch |
| --- | --- | --- |
| Actual imported pytest: Quack-version invalidation | Failed | Passed |
| Six new and four existing cache tests | — | 10 passed |
| Same stamp reused; changed stamp isolated | — | Passed in four fresh GPU processes |
| Full-model inference after export/reuse/invalidation | — | Exact upstream FA4 logits and tokens |
| Missing installed Quack metadata | Ignored | Standard PackageNotFoundError |
| Syntax, Ruff lint/format and whitespace | — | Passed |

## Fresh-process forward/backward cache lifecycle

Installed Quack stayed at 0.5.3. Test-only metadata stamps A and B simulate
version identity before importing FA4; this is not a real package upgrade.
All four arms use the same patched source and task-owned cache root on H20 GPU1.
All output, LSE and QKV-gradient hashes match exactly; there are no load failures.
First forward/backward includes JIT/export or disk loading, not interpreter startup.
Units are milliseconds.

| Arm | Misses | Exports | Disk loads | First fwd+bwd ms | Outputs/grads |
| --- | --- | --- | --- | --- | --- |
| A1 | 4 | 4 | 0 | 3276.185 | Exact |
| A2 | 0 | 0 | 4 | 18.822 | Exact |
| B1 | 4 | 4 | 0 | 3396.957 | Exact |
| B2 | 0 | 0 | 4 | 17.628 | Exact |

Reuse speed: A 174.06x; B 192.71x. Each is one cold/reuse pair, not a statistical estimate.

## Full-checkpoint end-to-end cache lifecycle

Full Qwen3-0.6B checkpoint, BF16, eager Transformers with native FA4,
B=1, 128/512 input tokens, exactly 32 greedy output tokens. A1/A2/B1/B2 are
four fresh processes on GPU1, with a separate initially empty task-owned model
JIT cache. Every arm matches native upstream FA4 logits/tokens exactly.
The first-prefill timer includes compilation or disk loading; steady complete
generation has two warmups and five samples per length. Units are milliseconds.
Lengths run in order within each process: the 512-token first-prefill row
already benefits from the 128-token arm's JIT work, so its ratio is not a
separate cold-start result.

| Stamp | Input | Cold prefill ms | Reuse prefill ms | Startup speed | Cold-arm steady gen ms | Reuse-arm steady gen ms | Steady ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 128 | 2243.852 | 325.720 | 6.889x | 634.909 | 633.665 | 1.0020x |
| A | 512 | 29.650 | 28.766 | 1.031x | 641.542 | 637.317 | 1.0066x |
| B | 128 | 2245.410 | 319.439 | 7.029x | 643.232 | 645.176 | 0.9970x |
| B | 512 | 29.822 | 28.216 | 1.057x | 644.417 | 656.688 | 0.9813x |

| Arm | Misses | Exports | Disk loads | Load failures |
| --- | --- | --- | --- | --- |
| A1 | 2 | 2 | 0 | 0 |
| A2 | 0 | 0 | 2 | 0 |
| B1 | 2 | 2 | 0 | 0 |
| B2 | 0 | 0 | 2 | 0 |

No steady-state inference speedup is attributed to this fingerprint change.
The table separates startup reuse from complete steady generation. The stamp
contract concerns fresh processes with changed installed distribution versions;
same-version source edits and replacement inside a running process are outside it.

## Provenance and reproduction

Measured on 2026-10-01, two NVIDIA H20 (SM90, 97,871 MiB each),
driver 580.105.08. Torch 2.7.0+cu128 and CUDA 12.8 were reused unchanged;
CuTeDSL 4.8.0, Quack 0.5.3, TVM-FFI 0.1.12, torch-c-dlpack-ext 0.1.5,
and Transformers 4.57.1 were installed in a task-owned dependency directory.
Quack 0.6.5 requires Torch dtypes absent from Torch 2.7. Quack 0.5.3's
metadata pins DSL 4.6.0.dev0, which conflicts with FA4's DSL minimum. This
known metadata conflict remains; the unmodified Quack 0.5.3 package imported
and ran the actual kernels. `pip check` is not clean.

Raw samples, source hashes, cache fingerprints, PIDs, checkpoint identity and
log counts are in `results/quack_cache_measurements.json`. All processes
finished naturally. The task-owned model checkpoint was removed afterwards.

Run `test_cache_utils.py` and `test_quack_cache_fingerprint.py` under the pinned
runtime. Launch `validation_quack/cache_lifecycle.py` four times with
`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`, `FA_LOG_LEVEL=1`, the same empty
`FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=<task-cache>` root, and
`FA_TEST_QUACK_STAMP=A,A,B,B` in order. Launch
`validation_quack/model_e2e.py` in the same order using a second empty root,
the pinned checkpoint and `--model <checkpoint> --label <arm> --rounds 5
--reference <json>`. Generate that JSON first by running the model harness
on the pinned upstream baseline with `--write-reference` and persistent
caching disabled. The stamp overrides exist only in these validation harnesses.
