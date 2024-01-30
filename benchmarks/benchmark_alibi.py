# Copyright (c) 2024, Sanghun Cho, Tri Dao.

import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from flash_attn.layers.rotary import apply_rotary_emb

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def generate_cos_sin(seqlen, rotary_dim, device, dtype):
    assert rotary_dim % 2 == 0
    angle = torch.rand(seqlen * 2, rotary_dim // 2, device=device) * 2 * math.pi
    cos = torch.cos(angle).to(dtype=dtype)
    sin = torch.sin(angle).to(dtype=dtype)
    return cos, sin


def flash_rotary(q, k, v, cos, sin, causal=False):
    # corrected by @tridao comments
    q = apply_rotary_emb(
        q, cos, sin, seqlen_offsets=0, interleaved=False, inplace=True
    )
    k = apply_rotary_emb(
        k, cos, sin, seqlen_offsets=0, interleaved=False, inplace=True
    )

    return flash_attn_func(q, k, v, causal=causal)


def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(q, k, v, dropout_p=0.0, causal=True, attn_bias=None):
    """
    Arguments:
        q, k, v: (batch_size, seqlen, nheads, head_dim)
        dropout_p: float
        attn_bias: (batch_size, nheads, seqlen, seqlen) or (1, nheads, seqlen, seqlen)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, nheads, d = q.shape
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    if attn_bias is not None:
        scores = rearrange(attn_bias, 'b h t s -> (b h) t s')
    else:
        scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=q.dtype, device=q.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
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


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

methods = (["fa2_alibi", "torch"]
           + (["xformers"] if xops is not None else [])
           + ["sdpa"]
           + ["fa2_baseline"]
           + ["fa2_rotary"])

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
            q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                    requires_grad=True) for _ in range(3)]
            # alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
            alibi_slopes = torch.rand(1, nheads, device=device, dtype=torch.float32) * 0.3
            attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen, seqlen, causal=causal).to(dtype)
            attn_bias = repeat(attn_bias, "1 ... -> b ...", b=batch_size)
            f, b = time_fwd_bwd(
                flash_attn_func,
                q, k, v,
                dropout_p,
                causal=causal,
                # alibi_slopes=alibi_slopes,
                alibi_slopes=None,
                repeats=repeats,
                verbose=False
            )
            time_f[config, "fa2_baseline"] = f
            time_b[config, "fa2_baseline"] = b

            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            f, b = time_fwd_bwd(
                flash_attn_func,
                q, k, v,
                dropout_p,
                causal=causal,
                alibi_slopes=rearrange(alibi_slopes, "1 h -> h"),
                # alibi_slopes=None,
                repeats=repeats,
                verbose=False
            )
            time_f[config, "fa2_alibi"] = f
            time_b[config, "fa2_alibi"] = b

            try:
                q = q.detach().requires_grad_(True)
                k = k.detach().requires_grad_(True)
                v = v.detach().requires_grad_(True)
                f, b = time_fwd_bwd(
                    attention_pytorch,
                    q, k, v,
                    dropout_p,
                    causal=causal,
                    attn_bias=attn_bias,
                    repeats=repeats,
                    verbose=False
                )
            except:  # Skip if OOM
                f, b = float('nan'), float('nan')
            time_f[config, "torch"] = f
            time_b[config, "torch"] = b

            # F.sdpa doesn't currently (torch 2.1) dispatch to flash-attn but just to be safe
            with torch.backends.cuda.sdp_kernel(enable_flash=False):
                q_pt = q.detach().requires_grad_(True).transpose(1, 2)
                k_pt = k.detach().requires_grad_(True).transpose(1, 2)
                v_pt = v.detach().requires_grad_(True).transpose(1, 2)
                f, b = time_fwd_bwd(
                    F.scaled_dot_product_attention,
                    q_pt, k_pt, v_pt,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                    is_causal=causal,
                    repeats=repeats,
                    verbose=False
                )
                time_f[config, "sdpa"] = f
                time_b[config, "sdpa"] = b

            if xops is not None:
                q = q.detach().requires_grad_(True)
                k = k.detach().requires_grad_(True)
                v = v.detach().requires_grad_(True)
                if causal:
                    attn_bias_xops = xops.LowerTriangularMask().add_bias(attn_bias.expand(-1, -1, seqlen, -1).to(dtype=q.dtype))
                    # NotImplementedError: No operator found for `memory_efficient_attention_backward` with inputs:
                    # `flshattB@v2.3.6` is not supported because:
                    #     attn_bias type is <class 'xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias'>
                    # `cutlassB` is not supported because:
                    #     attn_bias type is <class 'xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias'>
                    attn_bias_xops = attn_bias_xops.materialize((batch_size, nheads, seqlen, seqlen), dtype=q.dtype, device=device)
                else:
                    attn_bias_xops = attn_bias.to(dtype=q.dtype)
                f, b = time_fwd_bwd(
                    xops.memory_efficient_attention,
                    q, k, v,
                    attn_bias_xops,
                    dropout_p,
                    repeats=repeats,
                    verbose=False
                )
                time_f[config, "xformers"] = f
                time_b[config, "xformers"] = b

            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            cos, sin = generate_cos_sin(seqlen, headdim, device, dtype)
            f, b = time_fwd_bwd(
                flash_rotary,
                q, k, v,
                cos, sin,
                causal,
                repeats=repeats,
                verbose=False
            )
            time_f[config, "fa2_rotary"] = f
            time_b[config, "fa2_rotary"] = b

            print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
            csv_output = ""
            csv_output += f"{causal},{headdim},{batch_size},{seqlen},"
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
                print(
                    f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                    f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                )
                csv_output += f"{speed_f[config, method]:.2f},{speed_b[config, method]:.2f},{speed_f_b[config, method]:.2f},"
            print(csv_output)
