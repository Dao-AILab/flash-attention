---
id: environment
kind: command
command: /environment
script: ../harness/environment/setup_fa4_editable.sh
---

# Environment

Reset, install, or verify the repo-local CuteDSL FA4 editable runtime.

## Invocation

| Mode | Command |
| ---- | ------- |
| First setup, repair, or environment-suspect verify | `bash harness/harness/environment/setup_fa4_editable.sh` |
| Repeated gate verify after a successful setup | `bash harness/harness/environment/setup_fa4_editable.sh --verify-only` |
| Install-chain verify only | `bash harness/harness/environment/setup_fa4_editable.sh --skip-interface` |

## Behavior

| Step | Script Action |
| ---- | ------------- |
| Clean | Uninstalls stale `flash-attn-4`, `nvidia-cutlass-dsl`, `nvidia-cutlass-dsl-libs-base`, `quack-kernels`, `apache-tvm-ffi`, and `torch-c-dlpack-ext`. |
| Runtime install | Reinstalls repo-supported CuteDSL runtime components with `--no-deps` to avoid CUDA/Torch dependency drift. |
| Editable install | Installs only this checkout's `flash_attn/cute` with `python3 -m pip install --no-deps -e flash_attn/cute`. |
| Verify | Confirms CUTLASS `.pth`, `cutlass`, `quack`, and repo-local `flash_attn.cute.interface` imports. |

`--verify-only` runs only the Verify step. Use it for repeated UT/benchmark
gates in the same validated workspace so the gate does not reinstall CUTLASS
DSL on every pass. If verification fails, the environment is suspicious; rerun
the full setup without `--verify-only`.

## Notes

Use `--skip-interface` only when validating the installation chain while the
current source tree has a known syntax or import error. Normal gates must use
the full setup and interface verification.

Run import verification from outside the repo, normally `/tmp`. Starting Python
from the repo root can place root `flash_attn/__init__.py` before the editable
FA4 namespace hook and incorrectly require FA2 `flash_attn_2_cuda`.

The CUTLASS DSL version is pinned to the official repo-local
`flash_attn/cute/pyproject.toml` lower bound, currently
`nvidia-cutlass-dsl==4.4.2` with `nvidia-cutlass-dsl-libs-base==4.4.2`.
Do not silently upgrade this pin to the newest PyPI version.

Pipeline transaction-count support may come from repo-local wrappers such as
`flash_attn.cute.pipeline.PipelineTmaUmma`, not from the raw pip
`cutlass.pipeline.PipelineTmaUmma` class. When target code uses the wrapper,
fix imports/object construction to match that target API before suspecting the
installed CuteDSL version.
