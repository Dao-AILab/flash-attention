---
id: commit
kind: command
command: /commit
script: ../harness/commit/commit.sh
---

# Commit

Commit command with harness scope checks.

## Invocation

| Mode | Command |
| ---- | ------- |
| Refactor commit | `bash harness/harness/commit/commit.sh --type refactor -m "message"` |
| Harness commit | `bash harness/harness/commit/commit.sh --type harness -m "message"` |
| Feature commit | `bash harness/harness/commit/commit.sh --type feature -m "message"` |
| Dry run | `bash harness/harness/commit/commit.sh --type refactor --dry-run -m "message"` |

## Behavior

| Type | Staging Behavior | Scope Check |
| ---- | ---------------- | ----------- |
| `refactor` | Script stages only the three allowed HD256 kernel files. | Fails if staged files include anything else or any `tests/` path. |
| `harness` | Script stages only `harness/` files. | Fails if staged files include anything outside `harness/` or any `tests/` path. |
| `feature` | Script does not auto-stage. Caller must stage intended files first. | Tests are allowed only in this explicit mode. |

If both kernel/source and harness files changed in one task, run separate
commits. Never mix `harness/` changes into the refactor commit.

## Identity

| Detected User | Git Config |
| ------------- | ---------- |
| `wangsiyu` | `user.name=wangsiyu`, `user.email=siyu.wsy@gmail.com` |
| `siyu.wsy` | `user.name=wangsiyu`, `user.email=siyu.wsy@gmail.com` |

## Refactor Allowlist

| File |
| ---- |
| `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` |
| `flash_attn/cute/sm100_hd256_2cta_fmha_backward.py` |
| `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py` |
| `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` |

## Result Gate

| Result | Meaning |
| ------ | ------- |
| Exit `0` | Commit succeeded or dry-run checks passed. |
| Nonzero exit | Commit blocked by scope, identity, staging, or git failure. |
