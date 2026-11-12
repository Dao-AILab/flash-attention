"""Compare D512 training with PyTorch's memory-efficient SDPA on SM89.

Run from the repository root after building FlashAttention:
    python benchmarks/benchmark_flash_attention_d512.py
"""
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn import flash_attn_func


def measure(fn, repeats=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return {"ms": start.elapsed_time(end) / repeats,
            "peak_extra_mib": (torch.cuda.max_memory_allocated() - before) / 2**20}


def main():
    assert torch.cuda.get_device_capability() == (8, 9)
    torch.manual_seed(0)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "dtype": "bfloat16",
                      "batch": 1, "heads_q": 8, "heads_kv": 2, "head_dim": 512,
                      "scale": 1.0, "causal": True}))
    for n in (512, 2048, 4096, 8192):
        q = (torch.randn(1, n, 8, 512, device="cuda", dtype=torch.bfloat16) * .2).requires_grad_()
        k = (torch.randn(1, n, 2, 512, device="cuda", dtype=torch.bfloat16) * .2).requires_grad_()
        v = torch.randn_like(k).requires_grad_()
        do = torch.randn_like(q)

        def flash():
            return flash_attn_func(q, k, v, softmax_scale=1.0, causal=True)

        def sdpa():
            # The efficient backend needs equal Q/KV head counts. Include both
            # KV expansion and its backward reduction in the measured call.
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.repeat_interleave(4, 2).transpose(1, 2),
                v.repeat_interleave(4, 2).transpose(1, 2),
                scale=1.0, is_causal=True).transpose(1, 2)

        result = {"seqlen": n}
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for name, fn in (("flash", flash), ("sdpa_efficient", sdpa)):
                with torch.no_grad():
                    result[name + "_forward"] = measure(fn)
                result[name + "_forward_backward"] = measure(
                    lambda: torch.autograd.grad(fn(), (q, k, v), do))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
