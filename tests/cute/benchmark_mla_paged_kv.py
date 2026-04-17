# Copyright (c) 2025, Johnsonms.

# We recommend locking GPU clocks before running the benchmark to ensure consistent results.
# This can be done using the following commands (2619 MHz is the max clock for B200):
# sudo nvidia-smi -i 0 -pm 1
# sudo nvidia-smi -i 0 --lock-gpu-clocks 2619,2619
# See more here: https://github.com/triton-lang/triton/blob/d9f10ebdc5da53f73eb852fde73d8d7d80b679d1/python/triton/testing.py#L487

import time
import torch

from triton.testing import do_bench

from flash_attn.cute.interface import flash_attn_varlen_func


device = "cuda"
dtype = torch.bfloat16
seqlen_q = 1
nheads_q = 128
nheads_kv = 1  # MQA-128
headdim = 64
headdim_v = 512
causal = True

batch_size = 128
page_sizes = [None, 16, 64, 128]  # None = non-paged baseline

torch.manual_seed(0)

print(f"\nMLA paged KV, nheads_q = {nheads_q}, nheads_kv = {nheads_kv}, headdim = {headdim}, headdim_v = {headdim_v}, causal = {causal}")

for seqlen in [s * 1024 for s in [1, 2, 4, 8, 16, 32, 64]]:
    # Varlen format: (total_tokens, nheads, hdim)
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen

    try:
        q = torch.randn(total_q, nheads_q, headdim, dtype=dtype, device=device)
        k = torch.randn(total_k, nheads_kv, headdim, dtype=dtype, device=device)
        v = torch.randn(total_k, nheads_kv, headdim_v, dtype=dtype, device=device)
        qv = torch.randn(total_q, nheads_q, headdim_v, dtype=dtype, device=device)
    except torch.OutOfMemoryError:
        continue

    cu_seqlens_q = torch.arange(0, total_q + seqlen_q, seqlen_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, total_k + seqlen, seqlen, dtype=torch.int32, device=device)

    # Mem I/O: KV read + Q/QV read + O write
    total_seqlen = seqlen * batch_size
    mem_io = (
        total_seqlen * nheads_kv * (headdim + headdim_v) * 2  # K + V read
        + q.numel() * 2 + qv.numel() * 2  # Q + QV read
        + total_q * nheads_q * headdim_v * 2  # O write
    )
    # FLOPs: QK^T + PV (with qv, PV uses headdim_v)
    flops = seqlen_q * total_seqlen * nheads_q * (headdim + headdim_v * 2) * 2

    for page_size in page_sizes:
        if page_size is None:
            # Non-paged baseline
            fn = lambda: flash_attn_varlen_func(
                q, k, v, qv=qv,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=seqlen_q, max_seqlen_k=seqlen,
                causal=causal,
            )
            label = "non-paged"
        else:
            # Create paged KV
            num_pages_per_seq = (seqlen + page_size - 1) // page_size
            total_pages = num_pages_per_seq * batch_size
            k_paged = torch.zeros(total_pages, page_size, nheads_kv, headdim, device=device, dtype=dtype)
            v_paged = torch.zeros(total_pages, page_size, nheads_kv, headdim_v, device=device, dtype=dtype)
            page_table = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)
            for b in range(batch_size):
                for p in range(num_pages_per_seq):
                    page_idx = b * num_pages_per_seq + p
                    start = p * page_size
                    end = min(start + page_size, seqlen)
                    k_offset = b * seqlen
                    if start < seqlen:
                        k_paged[page_idx, :end - start] = k[k_offset + start:k_offset + end]
                        v_paged[page_idx, :end - start] = v[k_offset + start:k_offset + end]
                    page_table[b, p] = page_idx
            seqused_k = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)

            fn = lambda kp=k_paged, vp=v_paged, pt=page_table, su=seqused_k: flash_attn_varlen_func(
                q, kp, vp, qv=qv,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
                max_seqlen_q=seqlen_q, max_seqlen_k=None,
                seqused_k=su, page_table=pt,
                causal=causal,
            )
            path = "TMA" if page_size == 128 else "cp.async"
            label = f"paged-{page_size} ({path})"

        fn()  # warmup / compile
        time.sleep(1)  # avoid power throttling
        t = do_bench(fn, warmup=1, rep=10)
        print(
            f"Seqlen = {seqlen}, {label}: {t * 1e3:.1f} us, "
            f"{mem_io * 1e-9 / (t * 1e-3):.0f} GB/s, "
            f"{flops * 1e-12 / (t * 1e-3):.0f} TFLOPS/s"
        )

    print(f"Arithmetic intensity: {flops / mem_io:.1f}")
    print()
