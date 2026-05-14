---
id: test
kind: command
command: /test
script: ../harness/test/monitor_ut.py
logs: ../logs/test/
---

# Test

Monitored UT command for HD256 validation.

## Invocation

| Mode | Command |
| ---- | ------- |
| Full monitored UT | `python3 harness/harness/test/monitor_ut.py` |
| Preflight through UT script | `python3 harness/harness/test/monitor_ut.py --preflight-only` |

## Behavior

| Event | Script Action |
| ----- | ------------- |
| Run starts | Calls internal runner `bash harness/harness/test/run_hd256_ut.sh`. |
| Failure pattern appears in UT logs | Kills the UT process group and writes failure summary. |
| No new command or UT-log output for 45s while GPU utilization remains 100% | Kills the UT process group and writes hang summary. |
| No failure or hang | Waits until UT process exits. |

Hang detection is based on idle time, not total runtime. Long HD256 groups can
keep the GPU busy for several minutes; do not treat normal log progress as a
hang.

## Internal Runner

| Runner | Scope |
| ------ | ----- |
| `harness/harness/test/run_hd256_ut.sh` | Runs the three full CuteDSL `head_dim=256` UT groups. Internal only; use `/test` as the command entry. |

The runner must execute import preflight and pytest from `/tmp`, not from the
repo root. It passes absolute test paths plus `--rootdir=$REPO` and
`--import-mode=importlib` so editable FA4 resolves as a namespace package and
root `flash_attn/__init__.py` does not import FA2 `flash_attn_2_cuda`.

## Preflight

Preflight logs a source stamp before any pytest group runs. The stamp includes
package versions, git HEAD/dirty files, and path/SHA/mtime for key modules:
`interface`, `pipeline`, `flash_fwd`, `sm100_hd256_bwd`, `sm100_hd256_dkdv`,
and `sm100_hd256_dq`. Use this section of `preflight.log` to prove each run is
using the latest working-tree kernel files.

| Test | Required Param |
| ---- | -------------- |
| `tests/cute/test_flash_attn.py::test_flash_attn_output` | `d == [256]` |
| `tests/cute/test_flash_attn.py::test_flash_attn_varlen_output` | `d == [256]` |
| `tests/cute/test_flash_attn_varlen.py::test_varlen` | `D == [256]` |

If any active parametrization is not `[256]`, the internal runner temporarily
patches it before running.

## Test Groups

| Group | Pytest Node | Log |
| ----- | ----------- | --- |
| HD256 output | `tests/cute/test_flash_attn.py::test_flash_attn_output` | `harness/logs/test/ut_hd256_output.log` |
| HD256 varlen output | `tests/cute/test_flash_attn.py::test_flash_attn_varlen_output` | `harness/logs/test/ut_hd256_varlen_output.log` |
| Varlen | `tests/cute/test_flash_attn_varlen.py::test_varlen` | `harness/logs/test/ut_varlen.log` |

## Outputs

| File | Purpose |
| ---- | ------- |
| `harness/logs/test/monitor.log` | Monitor events |
| `harness/logs/test/preflight.log` | Import and head_dim preflight |
| `harness/logs/test/failure_cases.txt` | Failed case or group hints |
| `harness/logs/test/hang_report.txt` | Hang detection summary |

## Follow-Up

| Result | Next Step |
| ------ | --------- |
| UT pass | Proceed to `../skills/benchmark.md` and `../commands/benchmark.md`. |
| Failed | Reproduce failed case, fix, rerun `/test`. |
| Hang | Load `../skills/hang_detect_fix.md` and `hang_detect_fix.md`. |
