# GitHub Workflow Tagging Flow

This repository uses separate tag lanes so FA2 and FA4 publishing do not collide.

## Release lanes

| Tag pattern | Workflow | Package target | Version source |
| --- | --- | --- | --- |
| `v*` | `.github/workflows/publish.yml` | Root package (`flash-attn`) | Root package version metadata |
| `fa4-v*` | `.github/workflows/publish-fa4.yml` | `flash_attn/cute` package (`flash-attn-4`) | `setuptools-scm` with `fa4-v*` tags |

## How to publish

### FA2 / root package lane

1. Create a tag matching `v*` (example: `v2.9.0`).
2. Push that tag.
3. `publish.yml` creates a release, builds wheel matrix artifacts, and publishes to PyPI.

### FA4 / CUTE package lane

1. Create a tag matching `fa4-v*` (example: `fa4-v0.1.0`).
2. Push that tag.
3. `publish-fa4.yml` builds from `flash_attn/cute`, creates a GitHub release, and uploads `flash-attn-4` to PyPI.

## Guardrails

- Do not use `v*` tags for FA4 releases.
- Do not use `fa4-v*` tags for FA2 releases.
- Keep `flash_attn/cute/pyproject.toml` tag parsing in sync with the FA4 tag prefix.
- The workflow filename (`publish-fa4.yml`) is part of the PyPI trusted publishing OIDC identity — do not rename without updating PyPI.
