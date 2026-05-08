# Harness

AI-agent entrypoint for the local CuteDSL HD256 engineering harness.

## Context

| Item | Value |
| ---- | ----- |
| Project | `flash-attention` checkout |
| Focus | CuteDSL `head_dim=256` forward/backward kernels |
| Kernel directory | `../flash_attn/cute/` |
| Upstream guides | `../AGENTS.md` and `../CLAUDE.md` are read-only |
| Deprecated | `../agent_space` is legacy context only |

## Skills

Auto-invoke based on "When to Use" conditions.

| Skill | When to Use |
| ----- | ----------- |
| [workflow](skills/workflow.md) | Development flow, planning, validation gates, benchmark gates, version lock |
| [environment](skills/environment.md) | CuteDSL editable setup, FA4 import path, repo-local runtime |
| [test](skills/test.md) | UT execution policy, fail-fast monitoring, rerun loop |
| [hang_detect_fix](skills/hang_detect_fix.md) | Hang detection, kill policy, cuda-gdb debug workflow |
| [benchmark](skills/benchmark.md) | Benchmark gate, previous-run comparison, regression handling, SASS export |
| [refactor](skills/refactor.md) | Refactor requirements, target files, merge-ready code-level alignment, HD256 kernels |
| [commit](skills/commit.md) | Commit checks, allowed files, feature exception, git identity |

## Commands

| Command | File | Description |
| ------- | ---- | ----------- |
| `/test` | [commands/test.md](commands/test.md) | Run monitored UT with fail-fast and hang detection |
| `/hang_detect_fix` | [commands/hang_detect_fix.md](commands/hang_detect_fix.md) | Capture cuda-gdb hang diagnostics for a reproduced hang |
| `/benchmark` | [commands/benchmark.md](commands/benchmark.md) | Run repeated HD256 benchmark and compare with previous run |
| `/wheel` | [commands/wheel.md](commands/wheel.md) | Build only the repo-local `flash_attn/cute` FA4 wheel into `dist/` |
| `/commit` | [commands/commit.md](commands/commit.md) | Commit after validation while enforcing harness commit scope |

## Routing

| User Intent | Load | Rule |
| ----------- | ---- | ---- |
| Environment, editable, FA4 import path | `skills/environment.md` | Use repo-local `flash_attn/cute`; never patch `flash_attn/__init__.py`. |
| Refactor, align, target file | `skills/workflow.md` + `skills/refactor.md` | Follow workflow gates, refactor edit allowlist, and merge-ready target alignment. |
| UT, correctness, precision test | `skills/test.md` + `commands/test.md` | Use monitored `/test`; do not call raw pytest from memory. |
| Hang, stuck UT, GPU 100% | `skills/hang_detect_fix.md` + `commands/hang_detect_fix.md` | Kill broad UT, reproduce case, capture cuda-gdb diagnostics. |
| Benchmark or performance gate | `skills/benchmark.md` + `commands/benchmark.md` | Run benchmark only after UT passes; block systemic regression. |
| FA4 wheel or package | `commands/wheel.md` | Build only `../flash_attn/cute` into `dist/`; delete old wheels first. |
| Commit | `skills/commit.md` + `commands/commit.md` | Enforce commit scope and git identity. |
| Optimization or feature completion | Matching skill + command | Do not drift away from target-file direction. |

## Directories

| Directory | Contents |
| --------- | -------- |
| `skills/` | Methodology, requirements, and gates |
| `commands/` | Command descriptions, test selection, outputs |
| `harness/` | Executable scripts and code |
| `dist/` | Generated FA4 wheel artifacts |
| `logs/` | Generated logs and reports |
| `harness/logs/` | Benchmark current/previous logs produced by benchmark harness scripts |
