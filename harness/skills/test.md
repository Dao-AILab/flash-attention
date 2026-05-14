---
id: test
kind: skill
triggers:
  - test
  - UT
  - unit test
  - correctness
  - precision
  - failed
  - failure
  - rerun
---

# Test Skill

UT policy for CuteDSL HD256 validation.

## When to Use

| Signal | Action |
| ------ | ------ |
| UT, test, correctness, precision | Load `../commands/test.md`; use monitored test execution. |
| failed, failure | Stop all UT, reproduce failed case, fix, rerun monitored UT. |
| hang, stuck, GPU 100% | Jump to `hang_detect_fix.md`. |

## Test Loop

| Event | Required Action |
| ----- | --------------- |
| Start validation | Invoke `/test`; do not call pytest directly from memory. |
| UT failed detected in logs | Immediately stop the full UT run. |
| After failure stop | Reproduce the failing case, fix the bug, rerun `/test`. |
| Repeated failure | Repeat reproduce-fix-rerun until all UT pass. |
| Potential hang | Confirm there has been no command or UT-log progress before following `hang_detect_fix.md`. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No dead waiting | Do not wait indefinitely after failed logs or hang signal. |
| Fail fast | Kill the running UT process group when failure is detected. |
| Hang means idle | GPU 100% with continuing pytest log progress is not a hang. |
| Fix before rerun | Rerun only after investigating and attempting a fix. |
| Command handoff | Monitoring must be triggered through `../commands/test.md`. |
