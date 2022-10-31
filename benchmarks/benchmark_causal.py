from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_forward, benchmark_all, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
# from flash_attn.triton.fused_attention import attention as attention
from flash_attn.flash_attn_triton import flash_attn_qkvpacked_func
from flash_attn.flash_attn_triton_og import attention as attention_og

try:
    from flash_attn.fused_softmax import scaled_upper_triang_masked_softmax
except ImportError:
    scaled_upper_triang_masked_softmax = None


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


def attention_megatron(qkv):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
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
    attention = scaled_upper_triang_masked_softmax(scores, None, scale=1.0)
    output = torch.einsum('bhts,bshd->bthd', attention, v)
    return output.to(dtype=qkv.dtype)


torch.manual_seed(0)
repeats = 30
batch_size = 2
seqlen = 4096
nheads = 12
headdim = 128
# batch_size = 64
# seqlen = 512
# nheads = 8
# headdim = 128
dropout_p = 0.0
causal = True
dtype = torch.bfloat16
device = 'cuda'

qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                  requires_grad=True)
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                          device=qkv.device)

benchmark_all(flash_attn_unpadded_qkvpacked_func, rearrange(qkv, 'b s ... -> (b s) ...'),
              cu_seqlens, seqlen, dropout_p, causal=causal, repeats=repeats, desc='FlashAttention')
benchmark_all(attention_pytorch, qkv, dropout_p, causal=causal,
              repeats=repeats, desc='PyTorch Attention')

benchmark_all(flash_attn_qkvpacked_func, qkv, causal, repeats=repeats, desc='FlashAttention Triton')
pytorch_profiler(flash_attn_qkvpacked_func, qkv, causal, backward=True)

q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                       requires_grad=True) for _ in range(3)]
benchmark_all(attention_og, q, k, v, 1.0, repeats=repeats, desc='FlashAttention Triton OG')
# pytorch_profiler(attention, q, k, v, 1.0, backward=True)

if scaled_upper_triang_masked_softmax is not None:
    benchmark_all(attention_megatron, qkv, repeats=repeats, desc='Megatron Attention')
