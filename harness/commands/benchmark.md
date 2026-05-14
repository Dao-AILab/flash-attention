---
id: benchmark
kind: command
command: /benchmark
script: ../harness/benchmark/run_benchmark.sh
logs: ../harness/logs/benchmark/
---

# Benchmark

HD256 benchmark command with repeated-run report and previous-run comparison.

## Invocation

| Mode | Command |
| ---- | ------- |
| Full benchmark | `bash harness/harness/benchmark/run_benchmark.sh` |
| Dry run | `bash harness/harness/benchmark/run_benchmark.sh --dry-run` |
| Override repetitions | `BENCHMARK_RUNS=5 bash harness/harness/benchmark/run_benchmark.sh` |

## Fixed Benchmark Command

```bash
python3 harness/harness/benchmark/bench_sm100_hd256.py --compare-baseline --nheads 16 --nheads-kv 16 --rep 50 --warmup 10
```

## Environment

| Requirement | Command Or Setting |
| ----------- | ------------------ |
| Editable CuteDSL package | `python3 -m pip install --no-deps -e flash_attn/cute` |
| FA4 import routing | Wrapper runs from `/tmp` and verifies editable `flash_attn.cute.interface`. |
| Benchmark helper imports | Benchmark script must not import FA2 benchmark helpers that require root `flash_attn`. |
| Do not modify | Do not edit `flash_attn/__init__.py` for benchmark import routing. |

## Files

| File | Purpose |
| ---- | ------- |
| `harness/harness/benchmark/bench_sm100_hd256.py` | Benchmark implementation copied into harness. |
| `harness/harness/benchmark/run_benchmark.sh` | Fixed command wrapper, log rotation, repeated runs. |
| `harness/harness/benchmark/compare_benchmark.py` | Parses logs, compares current vs previous medians. |
| `harness/harness/benchmark/export_sass.sh` | Helper to export before/after SASS from provided cubin/shared-object paths. |

## Logs

| Path | Meaning |
| ---- | ------- |
| `harness/harness/logs/benchmark/current/` | Current benchmark run logs and report. |
| `harness/harness/logs/benchmark/previous/` | Previous benchmark run logs used for comparison. |
| `harness/harness/logs/benchmark/current/benchmark_report.md` | Median comparison and regression decision. |

Each benchmark run prints the same source stamp used by UT: package versions
plus path/SHA/mtime for key CuteDSL modules. Check the beginning of `run_*.log`
to prove the benchmark used the latest working-tree kernel files.

Only current and previous benchmark generations are retained.

## Result Gate

| Result | Meaning |
| ------ | ------- |
| Exit `0` | Benchmark gate passed or no previous baseline exists yet. |
| Nonzero exit | Benchmark failed or systemic regression detected. Do not lock or commit. |
