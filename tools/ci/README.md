# FA4 CI

CI runs on a self-hosted GPU runner using an Apptainer (SIF) container pulled from Docker Hub.
Triggered on every push to `main`.

## Two-pass test strategy

- **Pass 1** — compile kernels in parallel via `FakeTensorMode` (no GPU memory needed)
- **Pass 2** — run tests using cached compiled kernels on real GPU

See `run_fa4_ci.py` for the shared logic used by both CI and `test_ci_local.sh`.

## Required GitHub secrets / variables

| Name | Kind | Value |
|------|------|-------|
| `DOCKERHUB_USERNAME` | Secret | Docker Hub username |
| `DOCKERHUB_TOKEN` | Secret | Docker Hub access token |
| `CI_WORK_DIR` | Variable | Large-disk path on runner, e.g. `/scratch/user/johnson` |

`CI_WORK_DIR` is used for SIF caching and Apptainer temp files. Falls back to `/scratch/user/<github-actor>` if unset.

## Updating the container image

1. Build and push a new image via `tools/ci/docker/build.sh` + `tag_and_push.sh`.
2. Update `FA4_IMAGE` in `.github/workflows/ci.yml` with the new tag and `sha256` digest.
3. The old SIF is automatically deleted from the runner on the next CI run.

## Expanding test coverage

Edit `FA4_TEST_FILTER` in `.github/workflows/ci.yml`. To run the full suite, set it to an empty string and increase `compile-workers` in the `gpu-test` action call.

Alternatively, edit `run_fa4_ci.py` to change `DEFAULT_TEST_TARGET` or worker defaults — changes there apply to both CI and local runs.

## FA2 import isolation

Tests run inside the Apptainer container. The repo's `flash_attn/__init__.py` imports the FA2 C extension (`flash_attn_2_cuda`) which is absent in the container. `run_fa4_ci.py` works around this by:

1. Installing FA4 from the current repo into the container at runtime (`uv pip install -e flash_attn/cute`).
2. Running pytest from `/tmp` with absolute test paths — this keeps the repo root out of `sys.path[0]` so the installed FA4 package is found instead of the FA2 `__init__.py`.

`flash_attn/__init__.py` is intentionally not modified; isolation is handled entirely in CI.

## Adding a new runner / GPU type

1. Register a self-hosted runner on the machine with the desired label (e.g. `h100`).
2. Add the label to the `gpu` matrix in `.github/workflows/ci.yml`.
3. Set `CI_WORK_DIR` for the new machine if its scratch path differs.
