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
| editable, import path | Install and verify repo-local `flash_attn/cute`. |
| wheel, package | Load `../commands/wheel.md`. |

## Editable Runtime

| Step | Command or Check | Requirement |
| ---- | ---------------- | ----------- |
| E1 Install | `python3 -m pip install --no-deps -e flash_attn/cute` | Installs only the CuteDSL FA4 package from this checkout. |
| E2 Verify | `import flash_attn.cute.interface` | `interface.__file__` must resolve under `flash_attn/cute/`. |
| E3 Repo Imports | Set `PYTHONPATH=$REPO:$PYTHONPATH` when scripts need repo modules such as `benchmarks/`. | Do not rely on sibling checkouts. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No parent init patch | Do not edit `flash_attn/__init__.py` to fix CuteDSL imports. |
| No sibling FA4 | FA4 means this repo's `flash_attn/cute`, not another project directory. |
| No dependency install drift | Use `--no-deps` for editable setup unless explicitly asked to repair dependencies. |
| Wheel handoff | Use `../commands/wheel.md` for wheel builds; do not invent ad hoc packaging paths. |
