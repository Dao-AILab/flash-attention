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
from flash_attn_interface import flash_attn_func, _flash_attn_forward

try:
    from triton_fused_attention import attention as attention_triton
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
    elif torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif torch_type == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    else:
        raise ValueError("Unsupported tensor data type.")

def cudnn_spda_setup(qkv, seqlen_q, seqlen_k, causal=False):
    b, _, _, nheads, headdim = qkv.shape
    assert cudnn is not None, 'CUDNN is not available'
    o_gpu = torch.zeros(b, seqlen_q, nheads, headdim, dtype=qkv.dtype, device=qkv.device)
    o_gpu_transposed = torch.as_strided(
        o_gpu,
        [b, nheads, seqlen_q, headdim],
        [nheads * seqlen_q * headdim, headdim, nheads * headdim, 1],
    )
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=qkv.device)
    amax_s_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
    amax_o_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(qkv.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    new_q = torch.as_strided(
        qkv,
        [b, nheads, seqlen_q, headdim],
        [seqlen_q * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=0,
    )
    q = graph.tensor(
        name = "Q",
        dim = list(new_q.shape),
        stride = list(new_q.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_k = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim,
    )
    k = graph.tensor(
        name = "K",
        dim = list(new_k.shape),
        stride = list(new_k.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_v = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim * 2,
    )
    v = graph.tensor(
        name = "V",
        dim = list(new_v.shape),
        stride = list(new_v.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )

    def get_default_scale_tensor():
        return graph.tensor(
            dim = [1, 1, 1, 1],
            stride = [1, 1, 1, 1],
            data_type=cudnn.data_type.FLOAT
        )

    default_scale_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    descale_q = get_default_scale_tensor()
    descale_k = get_default_scale_tensor()
    descale_v = get_default_scale_tensor()
    descale_s = get_default_scale_tensor()
    scale_s = get_default_scale_tensor()
    scale_o = get_default_scale_tensor()

    o, _, amax_s, amax_o = graph.sdpa_fp8(
        q=q,
        k=k,
        v=v,
        descale_q=descale_q,
        descale_k=descale_k,
        descale_v=descale_v,
        descale_s=descale_s,
        scale_s=scale_s,
        scale_o=scale_o,
        is_inference=True,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal,
        name="sdpa",
    )

    o.set_output(True).set_dim(o_gpu_transposed.shape).set_stride(o_gpu_transposed.stride())

    amax_s.set_output(False).set_dim(amax_s_gpu.shape).set_stride(amax_s_gpu.stride())
    amax_o.set_output(False).set_dim(amax_o_gpu.shape).set_stride(amax_o_gpu.stride())
    # stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: new_q,
        k: new_k,
        v: new_v,
        descale_q: default_scale_gpu,
        descale_k: default_scale_gpu,
        descale_v: default_scale_gpu,
        descale_s: default_scale_gpu,
        scale_s: default_scale_gpu,
        scale_o: default_scale_gpu,
        o: o_gpu_transposed,
        amax_s: amax_s_gpu,
        amax_o: amax_o_gpu,
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu, amax_o_gpu

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

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_fwd(func, *args, **kwargs):
    time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


torch.manual_seed(0)

repeats = 30
device = 'cuda'
# dtype = torch.float16
dtype = torch.float8_e4m3fn

# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4224), (2, 8448), (1, 8448 * 2)]
bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 8192 * 2)]
# bs_seqlen_vals = [(4, 4096), (2, 8192), (1, 8192 * 2)]
# bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048)]
causal_vals = [False, True]
headdim_vals = [64, 128, 256]
dim = 2048
# dim = 256
dropout_p = 0.0

methods = (["Pytorch", "Flash3"]
        + (["cuDNN"] if cudnn is not None else [])
        # + (["Triton"] if attention_triton is not None else [])
        #    + (["xformers.c"] if xops is not None else [])
        #    + (["xformers.f"] if xops is not None else [])
           )

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            torch.cuda.empty_cache()
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16, requires_grad=False) for _ in range(3)]
            
            qkv = torch.stack([q, k, v], dim=2)
            qkv = qkv.to(torch.bfloat16)
            f = time_fwd(attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False)
            time_f[config, "Pytorch"] = f
            res_baseline = attention_pytorch(qkv, dropout_p, causal=causal)

            if attention_triton is not None:
                q_transposed = q.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
                k_transposed = k.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
                v_transposed = v.transpose(1, 2).contiguous().permute(0, 1, 3, 2).to(torch.float8_e4m3fn)
                scale = 1 / math.sqrt(headdim)
                f = time_fwd(
                    attention_triton, q_transposed, k_transposed, v_transposed,
                    causal, scale, repeats=5, verbose=False, desc='Triton'
                )
                f = time_fwd(
                    attention_triton, q_transposed, k_transposed, v_transposed,
                    causal, scale, repeats=repeats, verbose=False, desc='Triton'
                )
                time_f[config, "Triton"] = f
                res = attention_triton(
                    q_transposed, k_transposed, v_transposed.permute(0, 1, 3, 2),
                    causal, scale
                ).half().transpose(1, 2)
                torch.testing.assert_close(res, res_baseline, atol=0.5, rtol=0.5)

            # out = torch.empty_like(q)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            softmax_scale = q.shape[-1] ** (-0.5)
            descale_q = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            descale_k = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            descale_v = torch.tensor([1.0], dtype=torch.float32, device='cuda')

            # f = time_fwd(flash_attn_func, q, k, v, causal=causal, repeats=repeats, verbose=False)
            f = time_fwd(
                _flash_attn_forward,
                q, 
                k, 
                v, 
                softmax_scale, 
                causal=causal,
                window_size=(-1,-1),
                descale_q=descale_q, 
                descale_k=descale_k, 
                descale_v=descale_v, 
                repeats=repeats, 
                verbose=False
            )

            # res = flash_attn_func(q, k, v, causal=causal)
            # torch.testing.assert_close(res.half(), res_baseline, atol=0.05, rtol=0.05)

            time_f[config, "Flash3"] = f

            if cudnn is not None:
                qkv_fp8 = qkv.to(dtype)
                time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
                f = time_fwd(
                    cudnn_spda_setup(
                        qkv_fp8, seqlen, seqlen,
                        causal=causal
                    ),
                    repeats=repeats, verbose=False
                )
                time_f[config, "cuDNN"] = f
                # res, amax_o = cudnn_spda_setup(
                #     qkv_fp8, seqlen, seqlen,
                #     causal=causal
                # )()
                # res = res.half()
                # TODO: CUDNN has numerics issues when
                # num_heads=16, dim=128, seq_len=1024, batch_size=2
                # or larger sizes.
                # res_cpu = res.cpu().reshape(-1)
                # res_baseline_cpu = res_baseline.cpu().reshape(-1)
                # print(amax_o)
                # print(res)
                # print(res_baseline)
                # for i in range(len(res_cpu)):
                #     item = res_cpu[i]
                #     item_baseline = res_baseline_cpu[i]
                #     if abs(item - item_baseline) > 0.5:
                #         print(i)
                #         print(item)
                #         print(item_baseline)
                # torch.testing.assert_close(res, res_baseline, atol=0.05, rtol=0.05)

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            for method in methods:
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )
                #print (time_f[config,method])
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1e3} ms, "
                )


# with open('flash3_attn_time.plk', 'wb') as fp:
#     pickle.dump((time_f, time_b, time_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
