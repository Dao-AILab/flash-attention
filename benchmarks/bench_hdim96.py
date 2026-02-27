"""Benchmark backward pass for d=64, d=96, d=128.

Verifies:
  (a) d=64 and d=128 performance is unchanged (compare with/without changes)
  (b) d=96 performance falls between d=64 and d=128

Usage:
    python benchmarks/bench_hdim96.py
"""
import types, sys, math
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))

import torch

from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward
from flash_attn.cute.interface import flash_attn_func

torch.manual_seed(0)

repeats = 30


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def make_qkv(batch, seqlen, nheads, d):
    return [torch.randn(batch, seqlen, nheads, d, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True) for _ in range(3)]


def main():
    results = []
    for causal in [True, False]:
        for d in [64, 96, 128]:
            for seqlen in [1024, 2048, 4096]:
                batch = max(1, 32768 // seqlen)
                nheads = 16
                tag = "causal" if causal else "full  "

                try:
                    q, k, v = make_qkv(batch, seqlen, nheads, d)

                    # Warmup: trigger cute.compile cache before timed runs
                    out = flash_attn_func(q, k, v, causal=causal)
                    out[0].backward(torch.randn_like(out[0]))
                    for x in (q, k, v):
                        x.grad = None
                    torch.cuda.synchronize()

                    _, mf = benchmark_forward(flash_attn_func, q, k, v, causal=causal,
                                              repeats=repeats, verbose=False)
                    _, mb = benchmark_backward(flash_attn_func, q, k, v, causal=causal,
                                               repeats=repeats, verbose=False)
                except Exception as e:
                    print(f"{tag} d={d:3d} seqlen={seqlen:4d} batch={batch:2d}  SKIPPED ({e})")
                    continue

                fwd_tflops = efficiency(flops(batch, seqlen, d, nheads, causal, mode="fwd"), mf.mean)
                bwd_tflops = efficiency(flops(batch, seqlen, d, nheads, causal, mode="bwd"), mb.mean)
                line = (
                    f"{tag} d={d:3d} seqlen={seqlen:4d} batch={batch:2d}  "
                    f"fwd {mf.mean*1e3:7.2f}ms ({fwd_tflops:6.1f} TF/s)  "
                    f"bwd {mb.mean*1e3:7.2f}ms ({bwd_tflops:6.1f} TF/s)"
                )
                print(line)
                results.append((causal, d, seqlen, fwd_tflops, bwd_tflops))

    # Summary
    print("\n--- Summary ---")
    for causal in [True, False]:
        tag = "causal" if causal else "full  "
        for seqlen in [1024, 2048, 4096]:
            rows = {d: (ft, bt) for c, d, s, ft, bt in results if c == causal and s == seqlen}
            parts = [f"d{d}={rows[d][1]:.1f}" for d in [64, 96, 128] if d in rows]
            if parts:
                print(f"{tag} seqlen={seqlen:4d}  bwd TF/s: {'  '.join(parts)}")


if __name__ == "__main__":
    main()
