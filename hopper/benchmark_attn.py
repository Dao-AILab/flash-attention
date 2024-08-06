from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import time

try:
    import cudnn
except ImportError:
    cudnn = None


from einops import rearrange, repeat

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn_interface import flash_attn_func as flash_attn_func_v3, flash_attn_varlen_func as flash_attn_varlen_func_v3

# Need to install triton nightly:
# pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

try:
    from triton_fused_attention import attention as triton_attention
except ImportError:
    triton_attention = None

def flops(batch, nheads, seqlen_q, seqlen_k, headdim, causal=False, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


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


def cudnn_sdpa_setup(q, k, v, grad, o, stats, causal=False, varlen=False, seqlens=None):
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_kv, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads_kv, seqlen_k, headdim)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu, stats_gpu = o, stats
    graph_forward = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_forward = graph_forward.tensor_like(q_gpu.detach())
    k_forward = graph_forward.tensor_like(k_gpu.detach())
    v_forward = graph_forward.tensor_like(v_gpu.detach())

    seqlens_reshaped = seqlens if varlen else None
    seq_len_q = graph_forward.tensor_like(seqlens_reshaped.detach()) if varlen else None
    seq_len_kv = graph_forward.tensor_like(seqlens_reshaped.detach()) if varlen else None

    o_forward, stats_forward = graph_forward.sdpa(
        name="sdpa",
        q=q_forward,
        k=k_forward,
        v=v_forward,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal,
        use_padding_mask=varlen,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
    )

    o_forward.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats_forward.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph_forward.validate()
    graph_forward.build_operation_graph()
    graph_forward.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph_forward.check_support()
    graph_forward.build_plans()

    variant_pack_forward = {
        q_forward: q_gpu,
        k_forward: k_gpu,
        v_forward: v_gpu,
        o_forward: o_gpu,
        stats_forward: stats_gpu,
        seq_len_q: seqlens_reshaped,
        seq_len_kv: seqlens_reshaped,
    }

    dQ_gpu = torch.empty_like(q_gpu)
    dK_gpu = torch.empty_like(k_gpu)
    dV_gpu = torch.empty_like(v_gpu)
    dO_gpu = grad

    graph_backward = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    
    q_backward = graph_backward.tensor_like(q_gpu.detach())
    k_backward = graph_backward.tensor_like(k_gpu.detach())
    v_backward = graph_backward.tensor_like(v_gpu.detach())
    o_backward = graph_backward.tensor_like(o_gpu.detach())
    dO_backward = graph_backward.tensor_like(dO_gpu.detach())
    stats_backward = graph_backward.tensor_like(stats_gpu.detach())
    seq_len_q = graph_backward.tensor_like(seqlens_reshaped.detach()) if varlen else None
    seq_len_kv = graph_backward.tensor_like(seqlens_reshaped.detach()) if varlen else None
    
    dQ_backward, dK_backward, dV_backward = graph_backward.sdpa_backward(
        name="sdpa_backward",
        q=q_backward,
        k=k_backward,
        v=v_backward,
        o=o_backward,
        dO=dO_backward,
        stats=stats_backward,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal,
        use_padding_mask=varlen,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
    )
    
    dQ_backward.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
    dK_backward.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
    dV_backward.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())
    
    graph_backward.validate()
    graph_backward.build_operation_graph()
    graph_backward.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph_backward.check_support()
    graph_backward.build_plans()

    variant_pack_backward = {
        q_backward: q_gpu,
        k_backward: k_gpu,
        v_backward: v_gpu,
        o_backward: o_gpu,
        dO_backward: dO_gpu,
        stats_backward: stats_gpu,
        dQ_backward: dQ_gpu,
        dK_backward: dK_gpu,
        dV_backward: dV_gpu,
        seq_len_q: seqlens_reshaped,
        seq_len_kv: seqlens_reshaped,
    }

    workspace = torch.empty(
        max(graph_forward.get_workspace_size(), graph_backward.get_workspace_size()), 
        device="cuda", dtype=torch.uint8
    )

    def run_fwd(*args, **kwargs):
        graph_forward.execute(variant_pack_forward, workspace)
        return o_gpu, stats_gpu

    def run_bwd(*args, **kwargs):
        graph_backward.execute(variant_pack_backward, workspace)
        return dQ_gpu, dK_gpu, dV_gpu

    return run_fwd, run_bwd


torch.manual_seed(0)
repeats = 100
dropout_p = 0.0
causal = False
dtype = torch.float16
device = 'cuda'
verbose = False
batch_size = 2
# seqlen = 2048
seqlen = 8192
# seqlen = 4096
# seqlen = 2047
dim = 2048
# headdim = 128
# headdim = 64
headdim = 256

for mode in ['fwd', 'bwd']:
# for mode in ['bwd']:
    for headdim in [64, 128, 256]:
    # for headdim in [128]:
        for seqlen in [1024, 2048, 4096, 8192, 16384, 32768]:
        # for seqlen in [8192]:
            nheads = dim // headdim
            # nheads = 24
            # headdim = 64
            # batch_size = 64
            # seqlen = 512
            # nheads = 8
            # headdim = 128
            # nheads = 16
            # headdim = 128
            nheads_kv = nheads
            # nheads_kv = 1
    
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                            requires_grad=True)
            q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype, requires_grad=True)
            q_t = q.transpose(1, 2).contiguous().detach().requires_grad_()
            k_t = k.transpose(1, 2).contiguous().detach().requires_grad_()
            v_t = k.transpose(1, 2).contiguous().detach().requires_grad_()
            grad = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
            grad_t = grad.transpose(1, 2).contiguous()
            o_t = torch.empty_like(q.transpose(1, 2))
            stats = torch.empty(batch_size, nheads, seqlen, 1, dtype=torch.float32, device=q.device)
    
            bench_fn = benchmark_forward if mode == 'fwd' else partial(benchmark_backward, grad=grad)

            for causal in [False, True]:
            # for causal in [True]:
                print(f"\n### {mode = }, {batch_size = }, {headdim = }, {seqlen = }, {causal = } ###")
                # For var-seq-len
                lens = torch.full([q.shape[0]], seqlen, dtype=torch.int32)
                seqlens_cudnn = lens.reshape(batch_size, 1, 1, 1).contiguous().cuda()
                cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(lens, dim=0, dtype=torch.int32)]).cuda()
                if headdim <= 128 and cudnn is not None:
                    cudnn_sdpa_fwd, cudnn_sdpa_bwd = cudnn_sdpa_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), grad.transpose(1, 2), o_t, stats, causal=causal)
                    cudnn_sdpa_fwd_varlen, cudnn_sdpa_bwd_varlen = cudnn_sdpa_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), grad.transpose(1, 2), o_t, stats, causal=causal, varlen=True, seqlens=seqlens_cudnn)
                f = flops(batch_size, nheads, seqlen, seqlen, headdim, causal=causal, mode=mode)
                ref_o = flash_attn_func(q, k, v, dropout_p, causal=causal)
                _, m0 = bench_fn(flash_attn_func, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=verbose, desc='Fav2')
                if mode == 'bwd':
                    ref_dv, v.grad = v.grad.clone(), None
                    ref_dk, k.grad = k.grad.clone(), None
                    ref_dq, q.grad = q.grad.clone(), None
                # pytorch_profiler(flash_attn_func, q, k, v, dropout_p, causal=causal, backward=False)
                if headdim <= 128:
                    if triton_attention is not None and nheads_kv == nheads:
                        if mode == 'fwd':
                            time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                            _, m3 = benchmark_forward(triton_attention, q_t, k_t, v_t, causal, 1 / math.sqrt(headdim), repeats=repeats, verbose=verbose, desc='Triton')
                        # TODO: fix Triton numeric errors.
                        # if mode == 'bwd':
                        #     dv, v_t.grad = v_t.grad.clone(), None
                        #     dk, k_t.grad = k_t.grad.clone(), None
                        #     dq, q_t.grad = q_t.grad.clone(), None
                        #     torch.testing.assert_close(ref_dv, dv.transpose(1, 2), atol=0.05, rtol=0.05)
                        #     torch.testing.assert_close(ref_dk, dk.transpose(1, 2), atol=0.05, rtol=0.05)
                        #     torch.testing.assert_close(ref_dq, dq.transpose(1, 2), atol=0.05, rtol=0.05)
                    if cudnn is not None:
                        time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                        if mode == 'fwd':
                            _, m2 = benchmark_forward(cudnn_sdpa_fwd, repeats=repeats, verbose=verbose, desc='CuDNN')
                            _, m2_var = benchmark_forward(cudnn_sdpa_fwd_varlen, repeats=repeats, verbose=verbose, desc='CuDNN')
                            cudnn_sdpa_fwd()
                            torch.testing.assert_close(ref_o, o_t.transpose(1, 2), atol=0.05, rtol=0.05)
                            cudnn_sdpa_fwd_varlen()
                            torch.testing.assert_close(ref_o, o_t.transpose(1, 2), atol=0.05, rtol=0.05)
                        else:
                            cudnn_sdpa_fwd()
                            _, m2 = benchmark_forward(cudnn_sdpa_bwd, repeats=repeats, verbose=verbose, desc='CuDNN')
                            _, m2_var = benchmark_forward(cudnn_sdpa_bwd_varlen, repeats=repeats, verbose=verbose, desc='CuDNN')
                            dq, dk, dv = cudnn_sdpa_bwd()
                            torch.testing.assert_close(ref_dv, dv.transpose(1, 2), atol=0.05, rtol=0.05)
                            torch.testing.assert_close(ref_dk, dk.transpose(1, 2), atol=0.05, rtol=0.05)
                            torch.testing.assert_close(ref_dq, dq.transpose(1, 2), atol=0.05, rtol=0.05)
                            dq, dk, dv = cudnn_sdpa_bwd_varlen()
                            torch.testing.assert_close(ref_dv, dv.transpose(1, 2), atol=0.05, rtol=0.05)
                            torch.testing.assert_close(ref_dk, dk.transpose(1, 2), atol=0.05, rtol=0.05)
                            torch.testing.assert_close(ref_dq, dq.transpose(1, 2), atol=0.05, rtol=0.05)
                        # pytorch_profiler(cudnn_sdpa, backward=False)

                if headdim <= 128 or mode == 'fwd':
                    time.sleep(1)
                    _, m1 = bench_fn(flash_attn_func_v3, q, k, v, causal=causal, repeats=repeats, verbose=verbose, desc='Fav3')
                    q_var = q.reshape(-1, q.shape[-2], q.shape[-1])
                    k_var = k.reshape(-1, k.shape[-2], k.shape[-1])
                    v_var = v.reshape(-1, v.shape[-2], v.shape[-1])
                    time.sleep(1)
                    if mode == 'bwd':
                        dv, v.grad = v.grad.clone(), None
                        dk, k.grad = k.grad.clone(), None
                        dq, q.grad = q.grad.clone(), None
                        torch.testing.assert_close(ref_dv, dv, atol=0.05, rtol=0.05)
                        torch.testing.assert_close(ref_dk, dk, atol=0.05, rtol=0.05)
                        torch.testing.assert_close(ref_dq, dq, atol=0.05, rtol=0.05)
 
                    bench_var_fn = bench_fn
                    if mode == 'bwd':
                        grad_var = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                        bench_var_fn = partial(benchmark_backward, grad=grad_var)
                    _, m1_var = bench_var_fn(flash_attn_varlen_func_v3, q_var, k_var, v_var, cu_seqlens, cu_seqlens, seqlen, seqlen, causal=causal, repeats=repeats, verbose=verbose, desc='Fav3 var len')

                # pytorch_profiler(flash_attn_func_v3, q, k, v, causal=causal, backward=False)
                print(f'Fav2: {m0.mean * 1e3:.3f}ms, {(f / m0.mean * 1e-12):.1f} TFLOPS')
                if headdim <= 128:
                    if mode == 'fwd' and triton_attention is not None and nheads_kv == nheads:
                        print(f'Triton: {m3.mean * 1e3:.3f}ms, {(f / m3.mean * 1e-12):.1f} TFLOPS')
                    if cudnn is not None:
                        print(f'CuDNN: {m2.mean * 1e3:.3f}ms, {(f / m2.mean * 1e-12):.1f} TFLOPS')
                        print(f'CuDNN varlen: {m2_var.mean * 1e3:.3f}ms, {(f / m2_var.mean * 1e-12):.1f} TFLOPS')
                if headdim <= 128 or mode == 'fwd':
                    print(f'Fav3: {m1.mean * 1e3:.3f}ms, {(f / m1.mean * 1e-12):.1f} TFLOPS')
                    print(f'Fav3 varlen: {m1_var.mean * 1e3:.3f}ms, {(f / m1_var.mean * 1e-12):.1f} TFLOPS')
    