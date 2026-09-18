# FA4 CI

CI runs on a self-hosted GPU runner inside an Apptainer (SIF) container. The container is either
pulled from Docker Hub (default) or, when the runner has one, a SIF built locally from
`tools/ci/docker/Dockerfile` (registry-free mode, see below). Triggered on every push to `main`.

## Two-pass test strategy

- **Pass 1** — compile kernels in parallel via `FakeTensorMode` (no GPU memory needed)
- **Pass 2** — run tests using cached compiled kernels on real GPU

See `run_fa4_ci.py` for the shared logic used by both CI and `test_ci_local.sh`.

Before Pass 1 the driver provisions a disk-backed Apptainer overlay (replaces the SIF-baked cutlass-dsl /
quack / FA4 with the versions pinned in `flash_attn/cute/pyproject.toml` plus the checked-out FA4) and
**verifies it from a fresh session**: one distribution per package, `flash_attn.cute` from the checkout,
and the CuTeDSL runtime owned by the installed cutlass-dsl. On failure it re-provisions once, then fails
with `OVERLAY VERIFY FAILED`. The verified runtime is pinned via `CUTE_DSL_LIBS`. This guards against
an apptainer session hand-off race where the SIF-baked packages reappear next to the installed ones.

## Required GitHub secrets / variables

| Name | Kind | Value |
|------|------|-------|
| `DOCKERHUB_USERNAME` | Secret | Docker Hub username |
| `DOCKERHUB_TOKEN` | Secret | Docker Hub access token |
| `CI_WORK_DIR` | Variable | Large-disk path on runner, e.g. `/scratch/user/johnson` |
| `FA4_LOCAL_SIF` | Variable | Optional. Absolute path of a runner-local SIF; enables registry-free mode |

`CI_WORK_DIR` is used for SIF caching and Apptainer temp files. Falls back to `/scratch/user/<github-actor>` if unset.

## Registry-free mode (runner-local SIF)

When `FA4_LOCAL_SIF` is set, the GPU job skips the Docker Hub login and pull and runs in that SIF, so
CI needs no registry credentials. The Docker Hub secrets are then unused.

1. On the runner: `tools/ci/docker/build_local_sif.sh` builds `tools/ci/docker/Dockerfile` and writes a SIF
   to `CI_WORK_DIR/local-sif/` (keep it there: registry mode prunes `*.sif` directly under `CI_WORK_DIR`).
2. Set the repo variable `FA4_LOCAL_SIF` to the printed path. A missing file fails the job explicitly.

The image only provides OS, Python and torch; `run_fa4_ci.py` installs cutlass-dsl, quack and FA4 at job
time, so DSL pin bumps never need a rebuild. Rebuild when the Dockerfile changes.

## Updating the container image (registry mode)

1. Build and push a new image via `tools/ci/docker/build.sh` + `tag_and_push.sh`.
2. Update `fa4_image_cu129` and/or `fa4_image_cu130` on the `gpu-test` action call in `.github/workflows/ci.yml` with the new tag and `sha256` digest. The action picks between them from the runner CUDA version and exports `FA4_IMAGE` internally.
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
