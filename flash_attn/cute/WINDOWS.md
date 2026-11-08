# Native Windows x64 validation

The CuTe package uses a local LockFileEx helper on Windows and fcntl on POSIX.
Runtime library loading does not assume ctypes.RTLD_GLOBAL exists. The Windows
helper is included in ordinary wheels and does not import a private QuACK module.

Validated on RTX 5060 Laptop (SM120, 8 GB), Windows 11 x64, driver 610.62,
CUDA toolkit 13.4, Python 3.12.16, CUDA PyTorch 2.14.1+cu130, CuTe-DSL 4.8.1,
Windows-adapted QuACK 0.6.5, apache-tvm-ffi 0.1.15.post2 and triton-windows 3.9.0.post30.
Install the paired Windows-adapted QuACK build and CUDA-enabled PyTorch first.

From the repository directory, after installing the CuTe package:

    python -I -m pytest --import-mode=importlib tests/cute/test_windows_filelock.py tests/cute/test_windows_sm120.py -q

Isolated Python mode prevents the repository's FA2 top-level package from
shadowing the installed FA4 namespace; no FA2 extension is required by these tests.
Numerical tests compare FP16/BF16 MHA/GQA, head dimensions 64/128, causal and
noncausal dense attention, ragged causal attention, and dQ/dK/dV against FP32
references. Lock tests cover readers, exclusive contention, unlock, blocking
acquisition and invalid operations.

Paired regular wheels were installed without editable finders in an isolated
venv and ran 59 integration numerical checks in each of two processes. The
second process rejected new QuACK/FA4 kernel compilation and loaded x64 COFF
cache objects. Dependency files were shared with the tested development env;
this does not establish a clean-machine dependency installation.

The tested SM120 subset does not certify other GPUs, all sizes, paged KV,
SplitKV, MLA, block sparsity, score/mask modifiers or every upstream feature.
Cold compilation currently emits upstream AuxData adapter warnings; the
tested numerical results and gradients pass.
