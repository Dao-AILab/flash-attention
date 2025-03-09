import time
import torch
import torch.nn.functional as F

from triton.testing import do_bench, do_bench_cudagraph

from einops import rearrange

from flash_attn_interface import flash_attn_with_kvcache

try:
    from flash_attn.utils.benchmark import pytorch_profiler
except ImportError:
    pytorch_profiler = None

device = "cuda"
dtype = torch.bfloat16
# seqlen = 64 * 1024
seqlen = 8192
nheads = 128
nheads_kv = 1
headdim = 64
headdim_v = 512
has_qv = True
seqlen_q = 1
# page_size = None
page_size = 64

use_bench_cudagraph = False

torch.manual_seed(0)

batch_size = 128
cache_seqlens = None
# cache_seqlens = torch.tensor([seqlen] * batch_size, device=device, dtype=torch.int)
# cache_seqlens = torch.tensor([seqlen - 1, 1024, 1024, 1024], device=device, dtype=torch.int32)
# cache_seqlens = torch.tensor([1024] * batch_size, device=device, dtype=torch.int)
# cache_seqlens = torch.tensor([seqlen - 1, 1024, 1024, 1024], device=device, dtype=torch.int)
# cache_seqlens = torch.tensor([4500, 45000, 1800, 1800], dtype=torch.int32, device=device)

for seqlen in [s * 1024 for s in [1, 2, 4, 8, 16, 32, 64]]:
# for seqlen in [s * 1024 for s in [1]]:
    cache_seqlens = torch.tensor([seqlen] * batch_size, device=device, dtype=torch.int)
    num_splits = 0
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    v_cache = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, dtype=dtype, device=device)
    k_cache = torch.randn(batch_size, seqlen, nheads_kv, headdim, dtype=dtype, device=device)
    if page_size is not None:
        assert seqlen % page_size == 0
        k_cache, v_cache = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k_cache, v_cache]]
        page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                            "(b s) -> b s", s=seqlen // page_size)
    else:
        page_table = None
    qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, dtype=dtype, device=device) if has_qv else None

    # Time in ms
    fn = lambda: flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, num_splits=num_splits, qv=qv, page_table=page_table, causal=True)
    time.sleep(1)  # to avoid power throttling
    if not use_bench_cudagraph:
        t0 = do_bench(fn, warmup=1, rep=10)
    else:
        with torch.cuda.stream(torch.cuda.Stream()):
            t0 = do_bench_cudagraph(fn, rep=10)
    # exit(0)

    total_seqlen = seqlen * batch_size if cache_seqlens is None else cache_seqlens.sum().item()
    mem_io = total_seqlen * nheads_kv * (headdim + headdim_v) * 2 + q.numel() * 2 + qv.numel() * 4
    flops = seqlen_q * total_seqlen * nheads * (headdim + headdim_v * (2 if has_qv else 1)) * 2
    ideal_h100_time_mem = mem_io / 3.35e12 * 1e6
    ideal_h100_time_flop = flops / 989e12 * 1e6
    ideal_h100_time = max(ideal_h100_time_mem, ideal_h100_time_flop)
    print(f"Time{'' if not use_bench_cudagraph else ' w CUDA Graph'}: {t0 * 1e3:.0f} us, {mem_io * 1e-9 / (t0 * 1e-3):.0f} GB/s, {flops * 1e-12 / (t0 * 1e-3):.0f} TFLOPS/s")
    print(f"Ideal time: {ideal_h100_time:.0f} us")

    # if pytorch_profiler is not None:
    #     time.sleep(1)  # to avoid power throttling
    #     pytorch_profiler(fn)
