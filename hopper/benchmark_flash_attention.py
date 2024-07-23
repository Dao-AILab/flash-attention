# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func
from flash_attn_interface import flash_attn_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    import cudnn
except ImportError:
    cudnn = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


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


def cudnn_spda_setup(q, k, v, causal=False):
    b, nheads, seqlen_q, headdim = q.shape
    _, _, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads, seqlen_k, headdim)
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
        use_causal_mask=causal,
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


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

# Ideally, seq-len should be divisible by 132 to avoid wave quantization.
# However, the existing Triton implementation doesn't support seq-len like 8448.
bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192)]
# bs_seqlen_vals = [(2, 8192)]
causal_vals = [False]
# headdim_vals = [64, 128]
headdim_vals = [128]
dim = 128
dropout_p = 0.0

methods = (["Flash2", "Pytorch", "Flash3"]
           + (["Triton"] if attention_triton is not None else [])
           + (["xformers.c"] if xops is not None else [])
           + (["xformers.f"] if xops is not None else [])
           + (["cudnn"] if cudnn is not None else []))

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            f, b = time_fwd_bwd(
                flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            time_f[config, "Flash2"] = f
            time_b[config, "Flash2"] = b

            try:
                qkv = qkv.detach().requires_grad_(True)
                f, b = time_fwd_bwd(
                    attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
                )
                res_baseline = attention_pytorch(qkv, dropout_p, causal=causal)
            except:  # Skip if OOM
                f, b = float('nan'), float('nan')
            time_f[config, "Pytorch"] = f
            time_b[config, "Pytorch"] = b

            q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                    requires_grad=True) for _ in range(3)]
            f, b = time_fwd_bwd(flash_attn_func, q, k, v, causal=causal, repeats=repeats, verbose=False)
            res = flash_attn_func(q, k, v, causal=causal)

            time_f[config, "Flash3"] = f
            time_b[config, "Flash3"] = b

            if cudnn is not None:
                time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                res = benchmark_forward(
                    cudnn_spda_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal),
                    repeats=repeats, verbose=False
                )
                f = res[1].mean
                time_f[config, "cudnn"] = f
                time_b[config, "cudnn"] = math.inf

            if attention_triton is not None:
                q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                                    requires_grad=True) for _ in range(3)]
                # Try both values of sequence_parallel and pick the faster one
                try:
                    f, b = time_fwd_bwd(
                        attention_triton, q, k, v, causal, headdim**(-0.5),
                        False, repeats=repeats, verbose=False
                    )
                except:
                    f, b = float('nan'), float('inf')
                try:
                    _, b0 = time_fwd_bwd(
                        attention_triton, q, k, v, causal, headdim**(-0.5),
                        True, repeats=repeats, verbose=False
                    )
                except:
                    b0 = float('inf')
                time_f[config, "Triton"] = f
                time_b[config, "Triton"] = min(b, b0) if min(b, b0) < float('inf') else float('nan')

            if xops is not None:
                q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                    requires_grad=True) for _ in range(3)]
                f, b = time_fwd_bwd(
                    xops.memory_efficient_attention, q, k, v,
                    attn_bias=xops.LowerTriangularMask() if causal else None,
                    op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp)
                )
                time_f[config, "xformers.c"] = f
                time_b[config, "xformers.c"] = b

            if xops is not None:
                q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                    requires_grad=True) for _ in range(3)]
                f, b = time_fwd_bwd(
                    xops.memory_efficient_attention, q, k, v,
                    attn_bias=xops.LowerTriangularMask() if causal else None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
                )
                time_f[config, "xformers.f"] = f
                time_b[config, "xformers.f"] = b

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                speed_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                    time_b[config, method]
                )
                speed_f_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                    time_f_b[config, method]
                )
                #print (time_f[config,method])
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                    f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                )


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
