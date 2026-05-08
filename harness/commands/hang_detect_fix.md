---
id: hang_detect_fix
kind: command
command: /hang_detect_fix
script: ../harness/hang_detect_fix/cuda_gdb_snapshot.sh
logs: ../logs/hang_detect_fix/
---

# Hang Detect Fix

Helper command for cuda-gdb diagnostics after reproducing a hang.

## Invocation

| Mode | Command |
| ---- | ------- |
| Capture live PID | `bash harness/harness/hang_detect_fix/cuda_gdb_snapshot.sh <pid>` |
| Custom log | `bash harness/harness/hang_detect_fix/cuda_gdb_snapshot.sh <pid> harness/logs/hang_detect_fix/gdb_hang.log` |

## Required Flow

| Step | Action |
| ---- | ------ |
| 1 | Let `/test` kill the broad UT run after hang detection. |
| 2 | Reproduce the smallest hanging case manually. |
| 3 | Find the reproduced process PID. |
| 4 | Run this command to collect cuda-gdb diagnostics. |
| 5 | Fix the hang point, then rerun `/test`. |
