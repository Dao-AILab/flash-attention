---
id: commit
kind: skill
triggers:
  - commit
  - git commit
  - submit
  - lock version
  - version lock
---

# Commit Skill

Commit gate for CuteDSL HD256 harness work.

## When to Use

| Signal | Action |
| ------ | ------ |
| commit, git commit | Load this skill, then `../commands/commit.md`. |
| refactor commit | Enforce the three-kernel allowlist. |
| feature commit | Allow tests only when the work is explicitly a new Feature. |
| username `wangsiyu` or `siyu.wsy` | Use git name `wangsiyu` and email `siyu.wsy@gmail.com`. |

## Commit Scope

| Commit Type | Allowed Files | Test Files |
| ----------- | ------------- | ---------- |
| Refactor default | Only the three HD256 kernel files below | Forbidden |
| Feature | Explicitly staged feature files | Allowed only for new Feature work |

## Refactor Allowlist

| Editable In Refactor Commit |
| --------------------------- |
| `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` |
| `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py` |
| `flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` |

## Hard Rules

| Rule | Requirement |
| ---- | ----------- |
| No test commit by default | Do not commit `tests/` unless commit type is explicitly Feature. |
| Refactor commit scope | Do not commit files outside the three-kernel allowlist. |
| Feature exception | Tests may be committed only when the user explicitly says the change is a new Feature. |
| Identity | If current system user or git user is `wangsiyu` or `siyu.wsy`, set local git identity to `wangsiyu <siyu.wsy@gmail.com>`. |
| Command handoff | Use `../commands/commit.md`; do not hand-roll commit commands. |
