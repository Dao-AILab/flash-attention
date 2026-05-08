---
id: benchmark
kind: skill
triggers:
  - benchmark
  - performance
  - perf
  - regression
  - SASS
  - compare-baseline
---

# Benchmark Skill

Performance gate for CuteDSL HD256 work.

## When to Use

| Signal | Action |
| ------ | ------ |
| benchmark, performance, perf | Load `../commands/benchmark.md`. |
| after UT passes | Run benchmark gate before version lock or commit. |
| regression | Repeat benchmark, compare medians, export SASS or run equivalent experiment analysis. |
| SASS | Use benchmark command helper for before/after SASS export. |

## Benchmark Gate

| Step | Requirement |
| ---- | ----------- |
| B1 Run | Use `../commands/benchmark.md`; do not invent benchmark commands. |
| B2 Record | Store benchmark results under `harness/harness/logs/benchmark/`. |
| B3 Rotate | Keep only current run and previous run for comparison. |
| B4 Compare | Compare current repeated-run medians against previous repeated-run medians. |
| B5 Decide | This round passes only if no systemic performance regression is detected. |

## Regression Policy

| Condition | Required Action |
| --------- | --------------- |
| Normal run-to-run noise | Do not treat as regression; rely on repeated-run median report. |
| Systemic regression | Export before/after SASS or perform equivalent experiment analysis. |
| Regression root cause found | Modify/optimize until performance recovers. |
| Regression unresolved | Do not lock version, do not commit, and do not revert code as a substitute for analysis and optimization. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No skipped benchmark | Benchmark is mandatory after UT passes for performance-sensitive work. |
| No benchmark regression | Code cannot pass this round with systemic performance regression. |
| No rollback-to-pass | Do not skip benchmark or revert code just to pass this round; diagnose and optimize until performance recovers. |
| Command handoff | Use `../commands/benchmark.md`; benchmark scripts live under `../harness/benchmark/`. |
