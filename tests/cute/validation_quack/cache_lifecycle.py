"""Fresh-process cache probe. Optional stamp overrides are TEST-ONLY simulation."""

import json
import os
import time
import hashlib
import subprocess
from importlib import metadata
from pathlib import Path

started = time.perf_counter()

import torch  # noqa: E402 - time the interpreter's imports as part of the cold probe

if os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED") != "1":
    raise RuntimeError("Set FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 before launch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
    raise RuntimeError("A real SM90 GPU is required")

real_version = metadata.version
installed_quack = real_version("quack-kernels")
override = os.getenv("FA_TEST_QUACK_STAMP")
if override is not None:

    def version(name):
        return (
            installed_quack + "+test." + override if name == "quack-kernels" else real_version(name)
        )

    metadata.version = version

# Override metadata BEFORE importing FA: its module-level caches initialize at import.
from flash_attn.cute import flash_attn_func  # noqa: E402
from flash_attn.cute import cache_utils  # noqa: E402

torch.manual_seed(2026)
q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
torch.cuda.synchronize()
operation_start = time.perf_counter()
out, lse = flash_attn_func(q, k, v, num_splits=1, return_lse=True)
grads = torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
torch.cuda.synchronize()
operation_s = time.perf_counter() - operation_start
assert lse is not None
assert all(torch.isfinite(t).all() for t in (out, lse, *grads))
print(
    json.dumps(
        {
            "pid": os.getpid(),
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "cache_source_sha256": hashlib.sha256(
                Path(cache_utils.__file__).read_bytes()
            ).hexdigest(),
            "gpu": torch.cuda.get_device_name(),
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "installed_quack": installed_quack,
            "simulated_stamp": override,
            "fingerprint": cache_utils._compute_source_fingerprint(),
            "out_checksum": out.float().sum().item(),
            "output_gradient_sha256": hashlib.sha256(
                b"".join(t.detach().float().cpu().numpy().tobytes() for t in (out, lse, *grads))
            ).hexdigest(),
            "first_forward_backward_s": operation_s,
            "probe_s": time.perf_counter() - started,
        },
        sort_keys=True,
    )
)
