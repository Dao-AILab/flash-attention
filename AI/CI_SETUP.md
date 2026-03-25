# FA4 CI Setup Guide

This document describes how to set up the GitHub Actions CI for flash-attn-4 on a self-hosted GPU machine.

## Overview

CI runs on a self-hosted runner using an Apptainer (SIF) container pulled from Docker Hub. Two-pass test strategy:
- **Pass 1**: compile kernels in parallel using `FakeTensorMode` (no GPU memory needed)
- **Pass 2**: run tests using cached compiled kernels on real GPU

Triggered on every push to `main`.

---

## Prerequisites

- Linux x86_64, GPU with >= 40GB VRAM (tested on B200)
- CUDA 12.9+ driver
- `apptainer` installed on the runner
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

---

## Step 1 — Dev Environment

```bash
cd ~/scratch/flash-attention
uv venv .venv --python 3.12
source .venv/bin/activate

# Install PyTorch nightly matching your CUDA driver
# Check your CUDA version first: cat /usr/local/cuda/version.json
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129

# Install flash-attn-4 + dev dependencies
uv pip install -e "flash_attn/cute[dev]"

# Install pytest-xdist for parallel compilation pass
uv pip install pytest-xdist

# Verify
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_capability())"
# Expected: 2.12.x+cu129, (10, 0) for B200
```

---

## Step 2 — Self-hosted Runner

### 2a. Get Registration Token

Go to:
```
https://github.com/<owner>/flash-attention/settings/actions/runners/new?runnerOs=linux
```
Copy the token from the page (valid for 1 hour).

### 2b. Download and Register

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner

# Check the page for the latest runner version
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

./config.sh \
  --url https://github.com/<owner>/flash-attention \
  --token <TOKEN> \
  --name <machine-name> \
  --labels <gpu-label> \
  --work /home/<user>/scratch/flash-attention/_runner_work \
  --unattended
```

Label examples: `b200`, `h100`, `b300` — must match `ci.yml` matrix.

### 2c. Install as System Service

```bash
cd ~/actions-runner
sudo ./svc.sh install <username>
sudo ./svc.sh start
sudo ./svc.sh status   # should show: active (running)
```

### 2d. Verify on GitHub

Check runner is online at:
```
https://github.com/<owner>/flash-attention/settings/actions/runners
```
Status should show **Idle**.

### Managing the Service

```bash
sudo ./svc.sh stop      # stop runner
sudo ./svc.sh start     # start runner
sudo ./svc.sh status    # check status
sudo ./svc.sh uninstall # remove service (runner stays registered)

# To fully remove runner from GitHub:
./config.sh remove --token <TOKEN>
```

---

## Step 3 — CI Files

Files are already in the repo under `.github/` and `tools/ci/`:

```
.github/
├── workflows/
│   └── ci.yml                  # triggers on push to main
└── actions/
    └── gpu-test/
        └── action.yml          # pulls SIF from Docker Hub, runs two-pass tests
tools/
└── ci/
    ├── run_fa4_ci.py           # shared two-pass test logic
    ├── build_sif.sh            # builds Apptainer SIF from fa4.def
    ├── fa4.def                 # Apptainer definition file
    └── docker/
        ├── Dockerfile
        ├── build.sh
        └── tag_and_push.sh
```

### Docker credentials

The CI pulls the container image from Docker Hub. Add two secrets to your GitHub repo (`Settings → Secrets and variables → Actions → Secrets`):

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

### Work directory (`CI_WORK_DIR`)

The CI needs a writable directory on the runner for SIF caching and Apptainer temp/cache files. Set this as a repository variable (`Settings → Secrets and variables → Actions → Variables`):

| Variable | Value |
|----------|-------|
| `CI_WORK_DIR` | Large-disk path on the runner, e.g. `/scratch/user/johnson` |

Verify the path exists and has sufficient space on the runner:
```bash
df -h /scratch/user/johnson
```

If `CI_WORK_DIR` is not set, the workflow falls back to `/scratch/user/<github-actor>`.

### FA2 import isolation

Tests run inside the Apptainer container. The repo's `flash_attn/__init__.py` imports the FA2 C extension (`flash_attn_2_cuda`) which is not present in the SIF. `run_fa4_ci.py` avoids this by:

1. Installing FA4 from the current repo into the container at the start of each step (`uv pip install -e flash_attn/cute`)
2. Running pytest from `/tmp` with absolute test paths — this prevents Python from inserting the repo root into `sys.path[0]`, so the installed FA4 package is found instead of the repo's FA2 `__init__.py`

> **Note**: `flash_attn/__init__.py` is intentionally not modified — the isolation is handled entirely in CI.

### Update runner label

Edit `.github/workflows/ci.yml` and set the label to match your machine:

```yaml
matrix:
  gpu: [b200]   # change to h100, b300, etc.
```

### Expand coverage or restore the full suite

Update `tools/ci/run_fa4_ci.py` rather than editing the workflow and local script separately.

Recommended next step once the broader suite is ready:

```python
DEFAULT_TEST_TARGET = "tests/cute/"
```

Then increase the worker counts passed by the wrappers, for example:

```bash
python3 tools/ci/run_fa4_ci.py --compile-workers 64 --run-workers <num-gpus> --use-all-free-gpus
```

---

## Step 4 — Push and Trigger CI

```bash
cd ~/scratch/flash-attention
git add .github/
git commit -m "Add FA4 CI workflow with self-hosted runner"
git push origin main
```

CI will trigger automatically. Monitor at:
```
https://github.com/<owner>/flash-attention/actions
```

---

## Local Testing (without CI)

Use `test_ci_local.sh` to run the same logic manually:

```bash
~/scratch/flash-attention/test_ci_local.sh
```

The wrapper forwards extra flags to `tools/ci/run_fa4_ci.py`, for example:

```bash
~/scratch/flash-attention/test_ci_local.sh --skip-benchmark
~/scratch/flash-attention/test_ci_local.sh --use-all-free-gpus --run-workers 4
```

For Apptainer mode locally, set `FA4_SIF` to your local SIF path:

```bash
FA4_SIF=/scratch/user/johnson/my_fa4.sif ~/scratch/flash-attention/test_ci_local.sh
```

---

## Migration to Princeton / Dao-AILab

When moving to Princeton machine and `Dao-AILab/flash-attention`:

1. Check if Princeton machine has Apptainer — if yes, switch `gpu-test/action.yml` to use container (see quack's `action.yml` as reference)
2. Register a new runner on the Princeton machine following Step 2 above
3. Update runner label in `ci.yml` (e.g. `h100` or `princeton-h100`)
4. Set `CI_WORK_DIR` repo variable to the large-disk mount on the Princeton machine
5. Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets to the new repo
6. Restore full test suite (see Step 3 above)
7. Open a PR to `Dao-AILab/flash-attention` with the `.github/` changes
8. Add `environment: gpu-ci` to the PR workflow (`ci-pr.yml`) for security (see quack's `ci-pr.yml`)
