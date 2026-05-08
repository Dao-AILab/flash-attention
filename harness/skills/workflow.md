---
id: workflow
kind: skill
triggers:
  - workflow
  - development workflow
  - refactor workflow
  - environment
  - editable
  - FA4
  - plan
  - validation gate
  - version lock
  - lock version
  - UT
  - test
  - hang
  - benchmark
  - commit
---

# Workflow Skill

Overall development gate for CuteDSL HD256 work.

## When to Use

| Signal | Action |
| ------ | ------ |
| workflow, development flow | Use the gate sequence below. |
| environment, editable, FA4 | Load `environment.md` before any downstream gate. |
| plan | Produce major steps before editing. |
| validation gate, UT, test | Load `test.md` and run monitored tests. |
| benchmark | Load `benchmark.md`; run only after UT passes. |
| version lock, lock version | Lock only after UT passes and benchmark does not regress. |
| commit | Load `commit.md` after version lock. |

## Gate Sequence

| Gate | Required Action | Exit Criteria |
| ---- | --------------- | ------------- |
| W0 Environment | Load `environment.md`. Ensure `flash_attn/cute` editable runtime is active and repo-local import path is verified. | `flash_attn.cute.interface` resolves under this checkout. |
| W1 Analyze | Read source plus target/reference files. Classify differences as alignment work, HD256 divergence, behavior risk, or missing feature. | Analysis exists before edits. |
| W2 Plan | Create major steps. Avoid tiny edits and unrelated mixed layers. | Each step is independently gateable and rollbackable. |
| W3 Implement | Implement one major step. For refactor, obey `refactor.md` edit allowlist. | Scope matches loaded skill. |
| W4 Validate | Run monitored UT via `test.md`, then benchmark via `benchmark.md`. | UT passes and benchmark does not regress. |
| W5 Lock | Record locked state only after W4 passes. | Next step may start only from locked state. |
| W6 Commit | Load `commit.md`, then `../commands/commit.md`. | Commit scope and identity checks pass. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No environment skip | W0 must pass before refactor, UT, benchmark, or commit. |
| No ad hoc edits | Analysis and plan come before code changes. |
| No tiny-step plans | A step must represent a coherent architectural layer. |
| No raw UT gate | Use `test.md`; it owns UT selection, fail-fast, and hang detection. |
| No benchmark-before-UT | Benchmark runs after UT passes. |
| No skipped benchmark | Performance-sensitive work cannot lock without `benchmark.md`. |
| No unlocked continuation | If UT fails or benchmark regresses, fix or roll back before the next step. |
| No commit-before-lock | Commit only after the current step is locked. |
| Unknown command | Stop and load or create the matching command doc under `../commands/`. |
