---
id: environment
kind: skill
triggers:
  - environment
  - setup
  - editable
  - FA4
  - flash_attn/cute
  - import path
  - wheel
---

# Environment Skill

Repo-local runtime rules for CuteDSL FA4 (`flash_attn/cute`).

## When to Use

| Signal | Action |
| ------ | ------ |
| workflow start | Apply this skill before analysis, refactor, UT, benchmark, or commit gates. |
| editable, import path | Load `../commands/environment.md`; install and verify repo-local `flash_attn/cute`. |
| wheel, package | Load `../commands/wheel.md`. |

## Editable Runtime

Editable setup is the first environment action in a fresh or suspicious
workspace. Before the first import-path check, preflight, UT, benchmark, wheel
build, or commit gate, run `bash harness/harness/environment/setup_fa4_editable.sh`.
This resets stale CuteDSL runtime components, reinstalls the repo-supported
runtime packages, and installs the current checkout's CuteDSL FA4 package with
`python3 -m pip install --no-deps -e flash_attn/cute`.

After one full setup has passed in the same workspace, repeated gates should run
`bash harness/harness/environment/setup_fa4_editable.sh --verify-only` instead
of reinstalling. `--verify-only` preserves the installed packages and confirms
the package versions plus repo-local `flash_attn/cute` import path. If
verification fails, treat the environment as suspicious and rerun the full
setup. All downstream gates must validate and use this repo-local editable
package, not an already-installed wheel or a sibling checkout.

| Step | Command or Check | Requirement |
| ---- | ---------------- | ----------- |
| E1 First setup or repair | `bash harness/harness/environment/setup_fa4_editable.sh` | Reinstalls CuteDSL runtime components and this checkout's editable FA4 package. |
| E2 Repeated gate verify | `bash harness/harness/environment/setup_fa4_editable.sh --verify-only` | Does not reinstall; confirms installed runtime and repo-local editable imports. |
| E3 Verify | `import cutlass`, `import quack`, `import flash_attn.cute.interface` | `interface.__file__` must resolve under `flash_attn/cute/`. |
| E4 Repo Imports | Set `PYTHONPATH=$REPO:$PYTHONPATH` when scripts need repo modules such as `benchmarks/`. | Do not rely on sibling checkouts. |

## Version Pin

Use the dependency floor declared by this checkout's
`flash_attn/cute/pyproject.toml` as the harness pin. At the time of writing this
is `nvidia-cutlass-dsl==4.4.2` plus matching
`nvidia-cutlass-dsl-libs-base==4.4.2`. Do not auto-upgrade to the latest
available CUTLASS DSL during validation. For apparent API mismatches, first
check whether the target file uses repo-local wrappers such as
`flash_attn.cute.pipeline`.

## Import Path Trap

FA4 editable installs `flash_attn.cute` as an editable namespace package. If a
Python process starts from the repository root, `sys.path[0]` points at this
checkout and Python may import root `flash_attn/__init__.py` before the editable
namespace hook. That parent package imports FA2 `flash_attn_2_cuda`, which is
not part of the FA4 runtime and must not be required for CuteDSL tests.

Harness environment checks, preflight, and pytest entrypoints must start from a
directory outside the repo, such as `/tmp`, when importing `flash_attn.cute`.
When pytest needs repo-relative collection behavior, pass absolute test paths
and `--rootdir=$REPO` instead of changing cwd to `$REPO`.

## Pipeline API Trap

Do not infer supported pipeline call signatures from the pip package's raw
`cutlass.pipeline` classes alone. Some official FA4 kernels intentionally use
repo-local wrappers from `flash_attn.cute.pipeline`; for example,
`PipelineTmaUmma.producer_acquire(..., extra_tx_count=...)` is provided by that
wrapper, not by the raw pip `cutlass.pipeline.PipelineTmaUmma` API. When a
target file uses `from flash_attn.cute import pipeline`, align the edited kernel
to the same wrapper class instead of searching for a different CuteDSL version.

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No parent init patch | Do not edit `flash_attn/__init__.py` to fix CuteDSL imports. |
| Install before verify | Run full `/environment` first in a fresh or suspicious workspace; use `/environment --verify-only` for repeated gates after setup has passed. |
| No repo-root import cwd | Do not run FA4 import checks or pytest from `$REPO`; use `/tmp` plus absolute paths. |
| No sibling FA4 | FA4 means this repo's `flash_attn/cute`, not another project directory. |
| Match target wrappers | If the target file uses `flash_attn.cute.pipeline`, use that wrapper for matching pipeline objects. |
| No dependency install drift | Use `--no-deps` for editable setup unless explicitly asked to repair dependencies. |
| No latest CUTLASS drift | Keep CUTLASS DSL pinned to the repo-supported version, not PyPI latest. |
| Wheel handoff | Use `../commands/wheel.md` for wheel builds; do not invent ad hoc packaging paths. |
