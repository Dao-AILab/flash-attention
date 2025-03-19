from collections import namedtuple
from functools import partial
import math
import os
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

try:
    import cudnn
except ImportError:
    cudnn = None
# cudnn = None

Timing = NamedTuple('timing', [('mean', float)])


from einops import rearrange, repeat

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from flash_attn_interface import flash_attn_func as flash_attn_func_v3
# from flash_attn_interface import flash_attn_with_kvcache as flash_attn_func_v3
from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

from triton.testing import do_bench

try:
    from triton_fused_attention import attention as triton_attention
except ImportError:
    triton_attention = None
triton_attention = None

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"


def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    # # Warmup
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
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=3, rep=repeats) * 1e-3)


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


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


def cudnn_spda_setup(q, k, v, causal=False, window_size_left=-1):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads_k, seqlen_k, headdim)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu = torch.empty_like(q_gpu)
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=q.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left >= 0,
        use_causal_mask=causal or window_size_left >= 0,
        sliding_window_length=window_size_left if window_size_left >= 0 and not causal else None,
    )

    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


def cudnn_spda_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=-1):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads_k, seqlen_k, headdim)
    assert g.shape == (b, nheads, seqlen_q, headdim)
    assert o.shape == (b, nheads, seqlen_q, headdim)
    assert lse.shape == (b, nheads, seqlen_q, 1)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu, o_gpu, g_gpu = q, k, v, o, g
    dq_gpu = torch.empty_like(q_gpu)
    dk_gpu = torch.empty_like(k_gpu)
    dv_gpu = torch.empty_like(v_gpu)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())
    o = graph.tensor_like(o_gpu.detach())
    g = graph.tensor_like(g_gpu.detach())
    stats = graph.tensor_like(lse.detach())

    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=o,
        dO=g,
        stats=stats,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left >= 0,
        use_causal_mask=causal or window_size_left >= 0,
        sliding_window_length=window_size_left if window_size_left >= 0 and not causal else None,
    )

    dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
    dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
    dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        g: g_gpu,
        stats: lse,
        dq: dq_gpu,
        dk: dk_gpu,
        dv: dv_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return dq_gpu, dk_gpu, dv_gpu

    return run


torch.manual_seed(0)
repeats = 10
dropout_p = 0.0
causal = False
dtype = torch.bfloat16
# dtype = torch.float8_e4m3fn
dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
device = 'cuda'
verbose = True
varlen = False
page_size = None
softcap = 0.0
V_colmajor = False
deterministic = False
batch_size = 2
# seqlen = 2048
seqlen = 8192
# seqlen = 4096
# seqlen = 2047
dim = 2048
# headdim = 128
# headdim = 64
headdim = 256
# for headdim in [64, 128, 256]:
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
# bs_seqlen_vals = [(16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
# bs_seqlen_vals = [(32, 512), (16, 1024)]
# bs_seqlen_vals = [(2, 64 * 132)]
bs_seqlen_vals = [(2, 8192)]
# bs_seqlen_vals = [(1, 16 * 1024)]
time_f = {}
time_b = {}

# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192]:
# for headdim in [64, 96, 128, 192, 256]:
# for headdim in [64, 96, 128]:
# for headdim in [64, 128, 256]:
# for headdim in [64, 96, 128, 192, 256]:
for headdim in [128]:
    nheads = dim // headdim
    # nheads = 128
    # headdim = 64
    # batch_size = 64
    # seqlen = 512
    # nheads = 8
    # headdim = 128
    nheads_kv = nheads
    # nheads_kv = nheads // 4
    # nheads_kv = 1
    headdim_v = headdim
    # headdim_v = 512
    has_qv = headdim == 64 and headdim_v == 512
    # has_qv = False

    for batch_size, seqlen in bs_seqlen_vals:
        num_splits = 0
        window_size = (-1, -1)
        # window_size = (seqlen // 2 - 1, 0)
        pack_gqa = None
        # seqlen_q = 64
        seqlen_q = seqlen
        leftpad_k = None
        # leftpad_k = torch.full((batch_size,), 0, device=device, dtype=torch.int32)
        q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=True)
        k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=True)
        v = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in [q, k, v]]
        v_colmajor = v.detach().transpose(-1, -3).contiguous().transpose(-1, -3).requires_grad_()
        v_fa3 = v if not V_colmajor else v_colmajor
        qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen) if has_qv else None
        # q = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # k = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # v = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim_v), device=device, dtype=torch.int32).to(dtype)
        g = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
        o = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen, requires_grad=True)
        stats = torch.randn(batch_size, seqlen_q, nheads, 1, device=device, dtype=torch.float32)
        if varlen:
            q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_() for x in [q, k, v]]
            cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
            cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
            # cu_seqlens_q = torch.tensor([0, 248, 249, 250, 251, 252, 253, 254, 255, 256], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:256]
            # seqlen_q = 256
            # cu_seqlens_q = torch.tensor([0, 376, 377, 378, 379, 380, 381, 382, 383, 384], device=device, dtype=torch.int32)
            # q_unpad = q_unpad[:384]
            # seqlen_q = 384
        if page_size is not None:
            assert seqlen % page_size == 0
            k_paged, v_paged = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k, v]]
            page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                                   "(b s) -> b s", s=seqlen // page_size)
        else:
            page_table = None

        for causal in [False, True]:
        # for causal in [True]:
            print(f"\n### {headdim = }, {causal = }, {seqlen = } ###")
            nFLOPS = flops(batch_size, nheads, seqlen_q, seqlen, headdim if not has_qv else headdim + headdim_v, headdim_v, causal=causal, window_size=window_size)
            if cudnn is not None:
            # if False:
                if headdim <= 256 and dtype != torch.float8_e4m3fn and headdim == headdim_v:
                    cudnn_spda = cudnn_spda_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal, window_size_left=window_size[0])
                    cudnn_spda_bwd = cudnn_spda_bwd_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), o.transpose(1, 2), g.transpose(1, 2), stats.transpose(1, 2), causal=causal, window_size_left=window_size[0])
            # _, m0 = benchmark_forward(flash_attn_func, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=verbose, desc='Fav2')
            if dtype != torch.float8_e4m3fn and headdim == headdim_v:
            # if False:
                if not varlen:
                    m0 = time_fwd(flash_attn_func, q, k, v, dropout_p, causal=causal, window_size=window_size, softcap=softcap, repeats=repeats, verbose=verbose, desc='Fav2')
                else:
                    m0 = time_fwd(flash_attn_varlen_func, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, dropout_p, causal=causal, window_size=window_size, softcap=softcap, repeats=repeats, verbose=verbose, desc='Fav2')
                time_f[(causal, headdim, batch_size, seqlen), "Flash2"] = m0.mean
                time.sleep(1)
                if not varlen:
                    _, m0b = benchmark_backward(flash_attn_func, q, k, v, dropout_p, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic,
                                                repeats=repeats, verbose=False, desc='Fav2')
                else:
                    _, m0b = benchmark_backward(flash_attn_varlen_func, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, dropout_p, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic,
                                                repeats=repeats, verbose=False, desc='Fav2')
                time_b[(causal, headdim, batch_size, seqlen), "Flash2"] = m0b.mean
            # pytorch_profiler(flash_attn_func, q, k, v, dropout_p, causal=causal, backward=True)
            if headdim <= 256 and dtype != torch.float8_e4m3fn and headdim == headdim_v:
                if triton_attention is not None:
                    qt, kt, vt = [x.detach().transpose(1, 2).contiguous().requires_grad_() for x in [q, k, v]]
                    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                    m3 = time_fwd(triton_attention, qt, kt, vt, causal, 1 / math.sqrt(headdim), repeats=repeats, verbose=verbose, desc='Triton')
                    time_f[(causal, headdim, batch_size, seqlen), "Triton"] = m3.mean
                    # if causal: # triton bwd only works w causal for now
                    #     time.sleep(1)
                    #     _, m3b = benchmark_backward(triton_attention, qt, kt, vt, causal, 1 / math.sqrt(headdim), repeats=repeats, verbose=verbose, desc='Triton')
                    #     time_b[(causal, headdim, batch_size, seqlen), "Triton"] = m3b.mean
                    # # pytorch_profiler(triton_attention, q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), causal, 1 / math.sqrt(headdim), backward=True)
            if cudnn is not None:
            # if False:
                if headdim <= 256 and dtype != torch.float8_e4m3fn and headdim == headdim_v:
                    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                    m2 = time_fwd(cudnn_spda, repeats=repeats, verbose=verbose, desc='CuDNN')
                    time_f[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2.mean
                    time.sleep(1)
                    m2b = time_fwd(cudnn_spda_bwd, repeats=repeats, verbose=verbose, desc='CuDNN')
                    time_b[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2b.mean
                # pytorch_profiler(cudnn_spda, backward=False)
                # pytorch_profiler(cudnn_spda_bwd, backward=False)

            time.sleep(1)
            if not varlen:
                # m1 = time_fwd(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, cache_leftpad = leftpad_k, page_table=page_table, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
                m1 = time_fwd(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, qv=qv, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
                # pytorch_profiler(flash_attn_func_v3, q, k if page_size is None else k_paged, v_fa3 if page_size is None else v_paged, page_table=page_table, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa)
            else:
                m1 = time_fwd(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, repeats=repeats, verbose=verbose, desc='Fav3')
                # pytorch_profiler(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, num_splits=num_splits)
            time_f[(causal, headdim, batch_size, seqlen), "Flash3"] = m1.mean
            if dtype != torch.float8_e4m3fn and headdim == headdim_v and not DISABLE_BACKWARD:
                time.sleep(1)
                if not varlen:
                    _, m1b = benchmark_backward(flash_attn_func_v3, q, k, v, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic,
                                                repeats=repeats, verbose=False, desc='Fav3')
                else:
                    _, m1b = benchmark_backward(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, window_size=window_size, softcap=softcap, deterministic=deterministic,
                                                repeats=repeats, verbose=False, desc='Fav3')
                time_b[(causal, headdim, batch_size, seqlen), "Flash3"] = m1b.mean
                # time.sleep(1)
                # if not varlen:
                #     pytorch_profiler(flash_attn_func_v3, q, k, v, causal=causal, deterministic=deterministic, backward=True)
                # else:
                #     pytorch_profiler(flash_attn_varlen_func_v3, q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen, causal=causal, deterministic=deterministic, backward=True)
            # benchmark_forward(torch.clone, k, repeats=repeats, verbose=verbose, desc='Memcpy')

            if dtype != torch.float8_e4m3fn and headdim == headdim_v:
            # if False:
                print(f'Fav2 fwd: {m0.mean * 1e3:.3f}ms, {(nFLOPS / m0.mean * 1e-12):.1f} TFLOPS')
                print(f'Fav2 bwd: {m0b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m0b.mean * 1e-12):.1f} TFLOPS')
            if headdim <= 256 and dtype != torch.float8_e4m3fn and headdim == headdim_v:
                if triton_attention is not None:
                    print(f'Triton fwd: {m3.mean * 1e3:.3f}ms, {(nFLOPS / m3.mean * 1e-12):.1f} TFLOPS')
                    # if causal:
                    #     print(f'Triton bwd: {m3b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m3b.mean * 1e-12):.1f} TFLOPS')
                if cudnn is not None:
                    print(f'CuDNN fwd: {m2.mean * 1e3:.3f}ms, {(nFLOPS / m2.mean * 1e-12):.1f} TFLOPS')
                    print(f'CuDNN bwd: {m2b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m2b.mean * 1e-12):.1f} TFLOPS')
            print(f'Fav3 fwd: {m1.mean * 1e3:.3f}ms, {(nFLOPS / m1.mean * 1e-12):.1f} TFLOPS')
            if dtype != torch.float8_e4m3fn and headdim == headdim_v and not DISABLE_BACKWARD:
                print(f'Fav3 bwd: {m1b.mean * 1e3:.3f}ms, {(2.5 * nFLOPS / m1b.mean * 1e-12):.1f} TFLOPS')
            # benchmark_forward(torch.square, k)
            # print(f'cuBLAS: {m5.mean * 1e3:.3f}ms, {(nFLOPS_matmul / m5.mean * 1e-12):.1f} TFLOPS')
    # print(time_f)
    # print(time_b)

    # import pickle
    # # with open(f'flash3_attn_time_h100_hdim{headdim}_causal.plk', 'wb') as fp:
    # # with open(f'flash3_attn_time_h100_cudnn_triton_20241208.plk', 'wb') as fp:
    # with open(f'flash3_attn_time_h100_fa3_20250313.plk', 'wb') as fp:
    # # with open(f'flash3_attn_time_h100_fa3_fp8_20250313.plk', 'wb') as fp:
    # # with open(f'flash3_attn_time_h100_fp8_hdim{headdim}.plk', 'wb') as fp:
    # # with open(f'flash3_attn_time_h100_hdim{headdim}_1031.plk', 'wb') as fp:
    #     pickle.dump((time_f, time_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
