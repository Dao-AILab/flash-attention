# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn_sycl  as flash_attn_gpu

from einops import rearrange, repeat
import torch.utils.benchmark as benchmark

from flash_attn import flash_attn_func



def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**9) if not math.isnan(time) else 0.0


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
    
def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    for _ in range(5):
        y = fn(*inputs, **kwinputs)    
        
    torch.xpu.synchronize()    
    elapsed_time = [0] * repeats
    for i in range(repeats):
        start_time = time.perf_counter()
        y = fn(*inputs, **kwinputs)
        torch.xpu.synchronize()
        end_time = time.perf_counter()
        elapsed_time[i] = (end_time - start_time) * 1e3
    
    min_time = min(elapsed_time)
    avg = sum(elapsed_time) / repeats
    max_time = max(elapsed_time)
    if verbose:
        print(f"min: {min_time:.3f} ms, avg: {avg:.3f} ms, max: {max_time:.3f} ms")
    
    return min_time
    

def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'xpu'
dtype = torch.float16

bs_seqlen_vals = [(4, 1024), (4, 2048)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 1024
dropout_p = 0.0

methods = (["Flash2"])
time_f = {}
speed_f = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads = dim // headdim
            q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                    requires_grad=False) for _ in range(3)]
            f = time_fwd(
                flash_attn_func, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=True
            )
            time_f[config, "Flash2"] = f

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, nheads={nheads}, seqlen={seqlen} ###")
            for method in methods:
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                    time_f[config, method]
                )

                print(
                    f"{method} fwd: time: {f:.3f} ms, TFLOPS:  {speed_f[config, method]:.2f} TFLOPs/s, "
                )
