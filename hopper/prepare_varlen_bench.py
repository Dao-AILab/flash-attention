from collections import namedtuple
from functools import partial
import math
import os
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

Timing = NamedTuple('timing', [('mean', float)])

from einops import rearrange, repeat

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
# from flash_attn_interface import flash_attn_func as flash_attn_func_v3
from flash_attn_interface import flash_attn_with_kvcache as flash_attn_func_v3, get_scheduler_metadata
from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

from triton.testing import do_bench

cudnn = None
triton_attention = None

DISABLE_BACKWARD = True

def time_fwd(func, *args, repeats=10, verbose=True, desc="", **kwargs):
    # Warmup
    # for _ in range(5):
    #     func(*args, **kwargs)
    # time.sleep(1)
    # return benchmark_forward(func, *args, **kwargs, repeats=repeats, verbose=verbose, desc=desc)[1]
    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     for _ in range(2):
    #         out = func(*args, **kwargs)
    # torch.cuda.current_stream().wait_stream(s)
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph):
    #     out = func(*args, **kwargs)
    # time_f = benchmark_forward(lambda: graph.replay(), repeats=repeats, verbose=verbose, desc=desc)
    # # return time_f[1].mean
    # return time_f[1]
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3)

def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(-1, -1)):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (-1, -1):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0))
            col_right = torch.minimum(row_idx + seqlen_k - seqlen_q - window_size[1], torch.tensor(seqlen_k - 1))
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)

torch.manual_seed(0)
repeats = 10
dropout_p = 0.0
causal = False
dtype = torch.bfloat16
# dtype = torch.float8_e4m3fn
dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
device = 'cuda'
verbose = True
varlen = True
page_size = None
softcap = 0.0
V_colmajor = False
deterministic = False

# decode_batches = 128
# prefill_batches = 1
# batch_size = decode_batches + prefill_batches
# decode_seqlen_k = 8192 if decode_batches > 0 else 0
# prefill_seqlen_k = 2048 if prefill_batches > 0 else 0
# # seqlen_k = decode_seqlen_k * decode_batches + prefill_seqlen_k * prefill_batches # for cumulative
# seqlen_k = max(decode_seqlen_k, prefill_seqlen_k)
# decode_seqlen_q = 1 if decode_batches > 0 else 0
# prefill_seqlen_q = prefill_seqlen_k if prefill_batches > 0 else 0
# seqlen_q = decode_seqlen_q * decode_batches + prefill_seqlen_q * prefill_batches
# max_seqlen_q = max(decode_seqlen_q, prefill_seqlen_q)

time_f = {}
time_b = {}

prefill_first_vals = [False, True]
# prefill_first_vals = [False]

for headdim in [128]:
    for prefill_batches in [0]:
        # for decode_batches in range(32, 128 + 32, 8):
        for decode_batches in [128]:
            for sort_batches in [False, True]:
                
                batch_size = decode_batches + prefill_batches
                decode_seqlen_k = 8192 if decode_batches > 0 else 0
                prefill_seqlen_k = 1024 if prefill_batches > 0 else 0
                # seqlen_k = decode_seqlen_k * decode_batches + prefill_seqlen_k * prefill_batches # for cumulative
                seqlen_k = max(decode_seqlen_k, prefill_seqlen_k)
                decode_seqlen_q = 1 if decode_batches > 0 else 0
                prefill_seqlen_q = prefill_seqlen_k if prefill_batches > 0 else 0
                seqlen_q = decode_seqlen_q * decode_batches + prefill_seqlen_q * prefill_batches
                max_seqlen_q = max(decode_seqlen_q, prefill_seqlen_q)

                tp_degree=1
                nheads = 64//tp_degree
                nheads_kv = 8//tp_degree
                # nheads = 1
                # nheads_kv = 1
                headdim_v = headdim
                has_qv = False
                
                # window_size = (128, 0)
                window_size = (-1, -1)
                # window_size = (seqlen // 2 - 1, 0)

                # print("Window size: ", window_size)
                # print(f"Num query heads = {nheads}, kv heads = {nheads_kv}")
                # print("Head dim: ", headdim)
                # print("Batch size: ", batch_size)
                # print("Prefill seqlen k: ", prefill_seqlen_k)
                # print("Decode seqlen k: ", decode_seqlen_k)
                # print("Seqlen k (max): ", seqlen_k)
                # print("Prefill seqlen q: ", prefill_seqlen_q)
                # print("Decode seqlen q: ", decode_seqlen_q)
                # print("Seqlen q (total): ", seqlen_q)

                num_splits = 1
                pack_gqa = None

                # print(f"Num splits = {num_splits}, Pack GQA = {pack_gqa}")
                
                if prefill_batches == 0:
                    this_prefill_first_vals = [False]
                else:
                    this_prefill_first_vals = prefill_first_vals

                for prefill_first in this_prefill_first_vals:
                    leftpad_k = None
                    # leftpad_k = torch.full((batch_size,), 0, device=device, dtype=torch.int32)
                    q = torch.randn(seqlen_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=True)
                    k = torch.randn(batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=True)
                    v = torch.randn(batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
                    q, k, v = [x.detach().to(dtype).requires_grad_() for x in [q, k, v]]
                    v_colmajor = v.detach().transpose(-1, -3).contiguous().transpose(-1, -3).requires_grad_()
                    v_fa3 = v if not V_colmajor else v_colmajor
                    qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen) if has_qv else None
                    
                    seqlen_q_decode_offset = decode_seqlen_q * decode_batches
                    seqlen_q_prefill_offset = prefill_seqlen_q * prefill_batches
                    seqlen_k_decode_offset = decode_seqlen_k * decode_batches
                    
                    if prefill_first:
                        cu_seqlens_q_prefill = torch.arange(prefill_batches, device=device, dtype=torch.int32) * prefill_seqlen_q
                        cu_seqlens_q_decode = torch.arange(decode_batches + 1, device=device, dtype=torch.int32) * decode_seqlen_q + seqlen_q_prefill_offset
                        cu_seqlens_q = torch.cat((cu_seqlens_q_prefill, cu_seqlens_q_decode), dim=0)
                    else:
                        cu_seqlens_q_decode = torch.arange(decode_batches, device=device, dtype=torch.int32) * decode_seqlen_q
                        cu_seqlens_q_prefill = torch.arange(prefill_batches + 1, device=device, dtype=torch.int32) * prefill_seqlen_q + seqlen_q_decode_offset
                        cu_seqlens_q = torch.cat((cu_seqlens_q_decode, cu_seqlens_q_prefill), dim=0)

                    cache_seqlens_decode = torch.ones(decode_batches, dtype=torch.int32, device=device) * decode_seqlen_k
                    cache_seqlens_prefill = torch.ones(prefill_batches, dtype=torch.int32, device=device) * prefill_seqlen_k 

                    if prefill_first:
                        cache_seqlens = torch.cat((cache_seqlens_prefill, cache_seqlens_decode), dim=0)
                    else:
                        cache_seqlens = torch.cat((cache_seqlens_decode, cache_seqlens_prefill), dim=0)
                    

                    # print("q: ", q.shape)
                    # print("k: ", k.shape)
                    # print("v: ", v.shape)
                    # print("cu seqlens q: ", cu_seqlens_q.shape)
                    # print("cache seqlens: ", cache_seqlens.shape)
                    # print("cu seqlens q vals: ", cu_seqlens_q)
                    # print("cache seqlens vals: ", cache_seqlens)
                    
                    page_table = None

                    # for causal in [False, True]:
                    for causal in [True]:
                        print(f"\n### {headdim = }, {nheads = }, {nheads_kv = }, {causal = }, {prefill_seqlen_k = }, {decode_seqlen_k = }, {num_splits = }, {prefill_first = }, {decode_batches = }, {prefill_batches = } ###")
                        # nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen_k, headdim if not has_qv else headdim + headdim_v, headdim_v, causal=causal, window_size=window_size)
                        decode_nFLOPS = flops(decode_batches, nheads, decode_seqlen_q, decode_seqlen_k, headdim, headdim_v, causal=causal, window_size=window_size)
                        prefill_nFLOPS = flops(prefill_batches, nheads, prefill_seqlen_q, prefill_seqlen_k, headdim, headdim_v, causal=causal, window_size=window_size)
                        nFLOPS = decode_nFLOPS + prefill_nFLOPS

                        bytes_kv = (decode_seqlen_k * decode_batches + prefill_seqlen_k * prefill_batches) * (nheads_kv * headdim * 4)
                        bytes_qo = (decode_seqlen_q * decode_batches + prefill_seqlen_q * prefill_batches) * (nheads * headdim * 4) # don't count split partials
                        bytes = bytes_kv + bytes_qo
                        # print(f'{nFLOPS * 1e-9:.1f} GFLOPs, {bytes * 1e-9: .2f} GB, {nFLOPS/bytes:.1f} AI')

                        # time.sleep(1)
                        # m1 = time_fwd(flash_attn_func_v3,
                        #             q,
                        #             k,
                        #             v,
                        #             cache_seqlens=cache_seqlens,
                        #             cu_seqlens_q=cu_seqlens_q,
                        #             max_seqlen_q=max_seqlen_q,
                        #             causal=causal,
                        #             window_size=window_size,
                        #             softcap=softcap,
                        #             num_splits=num_splits,
                        #             pack_gqa=pack_gqa,
                        #             repeats=repeats, verbose=verbose, desc='Fav3')

                        # time_f[(causal, headdim, batch_size, seqlen_k), "Flash3"] = m1.mean
                        
                        # print(f'Fav3 fwd: {m1.mean * 1e3:.3f}ms, {(nFLOPS / m1.mean * 1e-12):.1f} TFLOPS, {bytes/ m1.mean * 1e-9: .2f} GB/s')

                        scheduler_metadata = get_scheduler_metadata(
                            batch_size, max_seqlen_q, seqlen_k, nheads, nheads_kv, headdim,
                            cache_seqlens, q.dtype, headdim_v=headdim, cu_seqlens_q=cu_seqlens_q,
                            causal=causal, num_splits=num_splits, varlen_sort_batches=sort_batches, )
                        
                        # m1 = time_fwd(get_scheduler_metadata,
                        #     batch_size, max_seqlen_q, seqlen_k, nheads, nheads_kv, headdim,
                        #     cache_seqlens, q.dtype, headdim_v=headdim, cu_seqlens_q=cu_seqlens_q,
                        #     causal=causal, num_splits=num_splits, sort_batches=sort_batches,
                        #     repeats=repeats, verbose=verbose, desc='Prepare'
                        # )

                        # time_f[(causal, headdim, batch_size, seqlen_k), "Prepare"] = m1.mean
                        
                        # print(f'Prepare: {m1.mean * 1e3:.3f}ms')
