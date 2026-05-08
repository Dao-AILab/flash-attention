---
id: refactor
kind: skill
triggers:
  - refactor
  - target file
  - target files
  - align
  - alignment
  - architecture alignment
  - forward kernel alignment
  - backward kernel alignment
  - dQ alignment
  - dK/dV alignment
  - flash_fwd_sm100.py
  - flash_bwd_sm100.py
---

# Refactor Skill

Strict merge-ready code-level and architecture-level alignment for CuteDSL
`head_dim=256` kernels under `../flash_attn/cute/`.

## When to Use

| Signal | Action |
| ------ | ------ |
| `refactor` | Load this skill plus `workflow.md`. |
| target file, target files | Resolve source and target from the tables below. |
| align, alignment, architecture alignment | Compare structure, naming, signatures, formatting, control flow, launch architecture. |
| forward alignment | Align HD256 forward to `flash_fwd_sm100.py`. |
| backward alignment | Align HD256 backward path to `flash_bwd_sm100.py`. |
| dQ alignment | Structure may follow forward; formulas and gradients follow backward. |
| dK/dV alignment | Keep responsibilities traceable to backward target. |

## Kernel Map

| Role | HD256 File |
| ---- | ---------- |
| Forward kernel | `../flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` |
| Backward entry | `../flash_attn/cute/sm100_hd256_2cta_fmha_backward.py` |
| Backward dQ kernel | `../flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py` |
| Backward dK/dV kernel | `../flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` |

| Legacy Or Imprecise Reference | Treat As |
| ----------------------------- | -------- |
| `sm100_hd256_2cta_fmha_forward_dqkernel.py` | `../flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py` unless the repository has changed. |
| `sm100_hd256_2cta_fmha_forward_dkdvkernel.py` | `../flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` unless the repository has changed. |

## Edit Scope

| Type | Files |
| ---- | ----- |
| Editable | `../flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` |
| Editable | `../flash_attn/cute/sm100_hd256_2cta_fmha_backward_dqkernel.py` |
| Editable | `../flash_attn/cute/sm100_hd256_2cta_fmha_backward_dkdvkernel.py` |
| Read-only | All other files, including target files, tests, harness files, shared CuteDSL files, generated files, and `../flash_attn/cute/sm100_hd256_2cta_fmha_backward.py` |

If refactor needs any read-only file edit, stop and ask for explicit approval.

## Target Alignment

| HD256 Area | Target File | Alignment Rule |
| ---------- | ----------- | -------------- |
| Forward kernel | `../flash_attn/cute/flash_fwd_sm100.py` | Merge-ready align module structure, helper boundaries, signatures, argument order, naming, formatting, control flow, launch contract, and implementation organization so the HD256 forward is recognizable as the HD256 specialization of the forward target. |
| Backward entry | `../flash_attn/cute/flash_bwd_sm100.py` | Merge-ready align entry organization, shared argument preparation, scheduling, launch structure, gradient contracts, helper boundaries, signatures, argument order, naming, and formatting. |
| Backward dQ kernel | `../flash_attn/cute/flash_fwd_sm100.py` + `../flash_attn/cute/flash_bwd_sm100.py` | Either target may be relevant. The dQ kernel is structurally closer to forward, so code organization, recomputation flow, mainloop shape, and data movement may align with `flash_fwd_sm100.py`; formulas, gradient semantics, API contracts, naming, signatures, and correctness obligations align with `flash_bwd_sm100.py`. |
| Backward dK/dV kernel | `../flash_attn/cute/flash_bwd_sm100.py` | Merge-ready align split-kernel responsibility, helper boundaries, scheduling, softmax/dS flow, epilogue/writeback, signatures, argument order, naming, and formatting with the backward target. |

## Code-Level Requirements

| Area | Requirement |
| ---- | ----------- |
| Scope | Follow `workflow.md` before editing. |
| Alignment depth | Merge-ready code-level alignment, not formatting-only cleanup. |
| Structure | Align module organization, imports, constants, helpers, signatures, params, tiling, scheduling, mainloop, epilogue, launch wrappers, guards, dispatch, generated-code boundaries. |
| Preservation | Preserve APIs, tensor layouts, launch contracts, numerical behavior, and shape constraints unless explicitly changed. |
| Divergence | Keep HD256-specific differences explicit; comment only when non-obvious. |
| Purity | Do not mix feature completion into pure refactor unless explicitly requested. |

## Merge-Ready Requirements

| Surface | Requirement |
| ------- | ----------- |
| Naming | Prefer the target file's names for matching roles, helpers, tensors, pipeline states, flags, and variables. Rename HD256-only names unless the old name describes a real specialization. |
| Function signatures | Matching responsibilities should keep target-like function names, parameter order, grouping, defaults, and annotations. Extra HD256-only parameters should be appended or grouped after target-equivalent parameters. |
| Softmax and dS | Keep mask order, `softmax_scale_log2` flow, LSE/dPsum handling, P generation, dS generation, and score/mask hooks as close to the target implementation as supported by HD256 constraints. |
| Formatting | Keep section order, spacing, comments, and local layout close to the target where code is equivalent. Formatting differences must not hide semantic differences. |
| Merge delta | The eventual diff against the target should read like an HD256 specialization. Every remaining divergence must be attributable to an HD256-only constraint or explicit unsupported feature. |
| Analysis | Before editing, list target-equivalent names/signatures/flows and the exact HD256 divergences that will remain after the step. |
