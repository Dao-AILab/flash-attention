# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import csv
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, v_headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 2 * batch * seqlen**2 * nheads * (headdim+v_headdim) // (2 if causal else 1)
    b = 2 * batch * seqlen**2 * nheads * (3*headdim+2*v_headdim) // (2 if causal else 1)
    return f if mode == "fwd" else (b if mode == "bwd" else f+b)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(q, k, v, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, nheads, d = q.shape
    nheads_k = k.shape[2]
    nheads_v = v.shape[2]
    if nheads_k < nheads:
        k = repeat(k, 'b s h d -> b s (h g) d', g=nheads//nheads_k)
    if nheads_v < nheads:
        v = repeat(v, 'b s h d -> b s (h g) d', g=nheads//nheads_v)
    v_d = v.shape[-1]
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=q.dtype, device=q.device)
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
    return output.to(dtype=q.dtype)


def flash_attention_pad(q,k,v, dropout_p=0.0, causal=True):
    batch_size, seqlen, nheads, d = q.shape
    nheads_k = k.shape[2]
    nheads_v = v.shape[2]
    if nheads_k < nheads_v:
        k = repeat(k, 'b s h d -> b s (h g) d', g=nheads_v//nheads_k)
    elif nheads_k > nheads_v:
        v = repeat(v, 'b s h d -> b s (h g) d', g=nheads_k//nheads_v)
    v_d = v.shape[-1]
    if d == v_d:
        return flash_attn_func(q, k, v, dropout_p, causal)
    if d < v_d:
        q = F.pad(q, (0, v_d-d))
        k = F.pad(k, (0, v_d-d))
        return flash_attn_func(q, k, v, dropout_p, causal)
    elif d > v_d:
        v = F.pad(v, (0, d-v_d))
        o = flash_attn_func(q, k, v, dropout_p, causal)
        return o[:,:,:,:v_d]
        


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

save_csv = True

repeats = 30
device = 'cuda'
dtype = torch.float16
torch.cuda.set_device(0)

bs_seqlen_vals = [(4, 512), (4, 1024), (4, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [ (32,64),(64,128)]
nheads_qkv = (32, 4, 16)
dropout_p = 0.0

methods = (["CustomFlash2", "Pytorch", "Flash2_Pad"])

if save_csv:
    csvfile =  open('flash2_attn_time.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow([
            "causal", "qk_headdim", "v_headdim","nheads_q", "nheads_k", "nheads_v", "batch_size", "seqlen",
            "time_fwd_CustomFlash2", "time_bwd_CustomFlash2", "time_fwd_bwd_CustomFlash2",
            "time_fwd_Pytorch", "time_bwd_Pytorch", "time_fwd_bwd_Pytorch",
            "time_fwd_Flash2_Pad", "time_bwd_Flash2_Pad", "time_fwd_bwd_Flash2_Pad",
            "flops_fwd_CustomFlash2", "flops_bwd_CustomFlash2", "flops_fwd_bwd_CustomFlash2",
            "flops_fwd_Pytorch", "flops_bwd_Pytorch", "flops_fwd_bwd_Pytorch",
            "flops_fwd_Flash2_Pad", "flops_bwd_Flash2_Pad", "flops_fwd_bwd_Flash2_Pad",
    ])

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim,v_headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (causal, headdim, batch_size, seqlen)
            nheads_q, nheads_k, nheads_v = nheads_qkv
            q = torch.randn(batch_size, seqlen, nheads_q, headdim, device=device, dtype=dtype,
                              requires_grad=True)
            k = torch.randn(batch_size, seqlen, nheads_k, headdim, device=device, dtype=dtype,
                                requires_grad=True)
            v = torch.randn(batch_size, seqlen, nheads_v, v_headdim, device=device, dtype=dtype,
                                requires_grad=True)
            f, b = time_fwd_bwd(
                flash_attn_func, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            time_f[config, "CustomFlash2"] = f
            time_b[config, "CustomFlash2"] = b

            try:
                q = q.detach().requires_grad_(True)
                k = k.detach().requires_grad_(True)
                v = v.detach().requires_grad_(True)
                f, b = time_fwd_bwd(
                    attention_pytorch, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=False
                )
            except:  # Skip if OOM
                f, b = float('nan'), float('nan')
            time_f[config, "Pytorch"] = f
            time_b[config, "Pytorch"] = b

            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            f, b = time_fwd_bwd(
                flash_attention_pad, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=False
            )
            time_f[config, "Flash2_Pad"] = f
            time_b[config, "Flash2_Pad"] = b

            print(f"### causal={causal}, qk_headdim={headdim}, v_headdim={v_headdim}, batch_size={batch_size}, seqlen={seqlen}, head_qkv={nheads_qkv} ###")
            for method in methods:
                time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                speed_f[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, v_headdim, nheads_q, causal, mode="fwd"),
                    time_f[config, method]
                )
                speed_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, v_headdim, nheads_q, causal, mode="bwd"),
                    time_b[config, method]
                )
                speed_f_b[config, method] = efficiency(
                    flops(batch_size, seqlen, headdim, v_headdim, nheads_q, causal, mode="fwd_bwd"),
                    time_f_b[config, method]
                )
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                    f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                )
            if save_csv:
                writer.writerow([
                    causal, headdim, v_headdim, *nheads_qkv, batch_size, seqlen,
                    time_f[config, "CustomFlash2"], time_b[config, "CustomFlash2"], time_f_b[config, "CustomFlash2"],
                    time_f[config, "Pytorch"], time_b[config, "Pytorch"], time_f_b[config, "Pytorch"],
                    time_f[config, "Flash2_Pad"], time_b[config, "Flash2_Pad"], time_f_b[config, "Flash2_Pad"],
                    speed_f[config, "CustomFlash2"], speed_b[config, "CustomFlash2"], speed_f_b[config, "CustomFlash2"],
                    speed_f[config, "Pytorch"], speed_b[config, "Pytorch"], speed_f_b[config, "Pytorch"],
                    speed_f[config, "Flash2_Pad"], speed_b[config, "Flash2_Pad"], speed_f_b[config, "Flash2_Pad"],
                ])
        
if save_csv:
    csvfile.close()
                


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
