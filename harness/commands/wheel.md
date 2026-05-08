---
id: wheel
kind: command
command: /wheel
script: ../harness/package_fa4_wheel.sh
outputs: ../dist/
---

# Wheel

Build only the repo-local CuteDSL FA4 wheel from `flash_attn/cute`.

## Invocation

| Mode | Command |
| ---- | ------- |
| Build FA4 wheel | `bash harness/harness/package_fa4_wheel.sh` |

## Behavior

| Step | Script Action |
| ---- | ------------- |
| Clean | Deletes old `harness/dist/*.whl` before building. |
| Build | Runs a wheel-only build for `flash_attn/cute`. |
| Verify | Requires exactly one generated `flash_attn_4-*.whl`. |

## Outputs

| Path | Purpose |
| ---- | ------- |
| `harness/dist/` | Fixed wheel output directory. |
| `harness/dist/flash_attn_4-*.whl` | Generated CuteDSL FA4 wheel artifact. |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| Scope | Build only `flash_attn/cute`; do not package root `flash-attention`. |
| Clean build dir | Remove old wheels before every new build. |
| No init patch | Do not edit `flash_attn/__init__.py` for packaging. |
