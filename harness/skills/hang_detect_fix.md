---
id: hang_detect_fix
kind: skill
triggers:
  - hang
  - hang detect
  - hang fix
  - hang detection fix
  - stuck
  - gpu 100
  - cuda-gdb
---

# Hang Detect Fix Skill

Hang detection and repair flow for HD256 UT.

## When to Use

| Signal | Action |
| ------ | ------ |
| UT runtime > 45s and GPU util stays 100% | Treat as hang. |
| hang detected by `test.md` monitor | Kill broad UT run, reproduce the case, debug with cuda-gdb. |
| cuda-gdb needed | Load `../commands/hang_detect_fix.md`. |

## Hang Rule

| Condition | Decision |
| --------- | -------- |
| A UT is not complete after 45s and GPU utilization samples stay at 100% | Hang |
| GPU util is below 100% or process exits | Not hang |

## Fix Flow

| Step | Required Action |
| ---- | --------------- |
| H1 Stop | Kill the broad UT run as soon as hang is detected. |
| H2 Reproduce | Re-run the smallest case that reproduces the hang. |
| H3 Attach | Use `cuda-gdb --pid <pid>` on the reproduced hanging process. |
| H4 Capture | Collect kernels, warps, threads, disassembly, and backtrace. |
| H5 Fix | Locate the hang point, patch the kernel, then rerun `/test`. |

## cuda-gdb Commands

```gdb
set logging file gdb_hang.log
set logging on
info cuda kernels
info cuda warps
info cuda threads sm 0
x/8i $pc
bt
set logging off
```

Use `../commands/hang_detect_fix.md` for the executable helper.
