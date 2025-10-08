from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

# from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined, benchmark_all, benchmark_fwd_bwd, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# # from flash_attn.triton.fused_attention import attention as attention
# from flash_attn.flash_attn_triton import flash_attn_qkvpacked_func
# from flash_attn.flash_attn_triton_og import attention as attention_og

# from triton.ops.flash_attention import attention as attention_triton

from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func

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


torch.manual_seed(0)
repeats = 30
batch_size = 8
seqlen = 2048
nheads = 12
headdim = 128
# nheads = 24
# headdim = 64
# batch_size = 64
# seqlen = 512
# nheads = 8
# headdim = 128
dropout_p = 0.0
causal = True
dtype = torch.float16
device = 'cuda'

qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                  requires_grad=True)
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                          device=qkv.device)

qkv_unpad = rearrange(qkv, 'b s ... -> (b s) ...').detach().requires_grad_(True)
# benchmark_all(flash_attn_varlen_qkvpacked_func, qkv_unpad,
#               cu_seqlens, seqlen, dropout_p, causal=causal, repeats=repeats, desc='FlashAttention')
# pytorch_profiler(flash_attn_varlen_qkvpacked_func, qkv_unpad,
#                  cu_seqlens, seqlen, dropout_p, causal=causal, backward=True)
benchmark_forward(flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, desc='Fav2')
pytorch_profiler(flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, backward=False)

# for dropout_p in [0.1, 0.0]:
#     for causal in [False, True]:
#         print(f"### {dropout_p = }, {causal = } ###")
#         pytorch_profiler(fav2_qkvpacked_func, qkv, dropout_p, causal=causal, backward=True)


# nheads_k = 2
# q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
# kv = torch.randn(batch_size, seqlen, 2, nheads_k, headdim, device=device, dtype=dtype,
#                  requires_grad=True)
# if fav2_kvpacked_func is not None:
#     benchmark_all(fav2_kvpacked_func, q, kv, dropout_p, causal=causal, repeats=repeats, desc='Fav2')
#     pytorch_profiler(fav2_kvpacked_func, q, kv, dropout_p, causal=causal, backward=True)

# dropout_p = 0.0
# causal = False
# benchmark_all(attention_pytorch, qkv, dropout_p, causal=causal,
#               repeats=repeats, desc='PyTorch Attention')

# benchmark_all(flash_attn_qkvpacked_func, qkv, None, causal, repeats=repeats, desc='FlashAttention Triton')
# pytorch_profiler(flash_attn_qkvpacked_func, qkv, None, causal, backward=True)

# q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
#                        requires_grad=True) for _ in range(3)]
# benchmark_all(attention_og, q, k, v, 1.0, repeats=repeats, desc='FlashAttention Triton OG')
# # pytorch_profiler(attention, q, k, v, 1.0, backward=True)

# from src.ops.fftconv import fftconv_func

# dim = nheads * headdim
# u = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype, requires_grad=True)
# k = torch.randn(dim, seqlen, device=device, requires_grad=True)
# D = torch.randn(dim, device=device, requires_grad=True)
# benchmark_all(fftconv_func, u, k, D, repeats=repeats, desc='FFTConv')
# pytorch_profiler(fftconv_func, u, k, D, backward=True)
# pytorch_profiler(torch.fft.rfft, u.float())

flops = 4 * batch_size * seqlen ** 2 * nheads * headdim
ideal_a100_time = flops / 312 / 1e9
print(f"Ideal A100 fwd time: {ideal_a100_time:.3f}ms, bwd time: {ideal_a100_time * 2.5:.3f}ms")
exit(0)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

time_f = {}
time_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            nheads = dim // headdim
            qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=qkv.device)
            qkv_unpad = rearrange(qkv, 'b s ... -> (b s) ...').detach().requires_grad_(True)
            f, b = time_fwd_bwd(
                flash_attn_varlen_qkvpacked_func, qkv_unpad, cu_seqlens, seqlen, dropout_p,
                causal=causal, repeats=repeats, verbose=False
            )
            time_f[(causal, headdim, batch_size, seqlen), "Flash"] = f
            time_b[(causal, headdim, batch_size, seqlen), "Flash"] = b

            qkv = qkv.detach().requires_grad_(True)
            f, b = time_fwd_bwd(
                fav2_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            time_f[(causal, headdim, batch_size, seqlen), "Flash2"] = f
            time_b[(causal, headdim, batch_size, seqlen), "Flash2"] = b

            # q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
            #                        requires_grad=True) for _ in range(3)]
            # # Try both values of sequence_parallel and pick the faster one
            # f, b = time_fwd_bwd(
            #     attention_triton, q, k, v, causal, headdim**(-0.5),
            #     False, repeats=repeats, verbose=False
            # )
            # _, b0 = time_fwd_bwd(
            #     attention_triton, q, k, v, causal, headdim**(-0.5),
            #     True, repeats=repeats, verbose=False
            # )
            # time_f[(causal, headdim, batch_size, seqlen), "Triton"] = f
            # time_b[(causal, headdim, batch_size, seqlen), "Triton"] = min(b, b0)

            if seqlen <= 8 * 1024:
                qkv = qkv.detach().requires_grad_(True)
                f, b = time_fwd_bwd(
                    attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
                )
            else:
                f, b = float('nan'), float('nan')
            time_f[(causal, headdim, batch_size, seqlen), "Pytorch"] = f
            time_b[(causal, headdim, batch_size, seqlen), "Pytorch"] = b

            # q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
            #                        requires_grad=True) for _ in range(3)]
            # import xformers.ops as xops
            # f, b = time_fwd_bwd(
            #     xops.memory_efficient_attention, q, k, v,
            #     attn_bias=xops.LowerTriangularMask() if causal else None,
            #     op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp)
            # )
            # time_f[(causal, headdim, batch_size, seqlen), "xformers"] = f
            # time_b[(causal, headdim, batch_size, seqlen), "xformers"] = b


import pickle
with open('flash2_attn_time_h100.plk', 'wb') as fp:
    pickle.dump((time_f, time_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
