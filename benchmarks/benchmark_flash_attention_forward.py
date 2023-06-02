from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward, benchmark_combined
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func


def attention_ref(qkv, attn_mask, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = (qkv.float() if upcast else qkv).unbind(dim=2)
    seqlen = qkv.shape[1]
    d = qkv.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=qkv.dtype)


torch.manual_seed(0)
repeats = 250
batch_size = [1,32,64,128]
nheads = 16
seqlen = [1024,2048,4096]
n = 1024
d = n // nheads
dropout_p = 0.1
causal = False
dtype = torch.float16
device = 'cuda'

result_summary = []

for bs in batch_size:
    for sq in seqlen:
        if (bs > 32 and sq > 2048) or (bs > 64 and sq > 1024):
            continue
        x = torch.randn(bs, sq, n, device='cuda', dtype=dtype, requires_grad=True)
        Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

        lengths = torch.randint(sq - 20, sq, (bs, 1), device='cuda')
        attention_mask_bool = repeat(torch.arange(sq, device='cuda'), 's -> b s', b=bs) < lengths
        attention_mask = torch.zeros(bs, sq, device='cuda', dtype=dtype)
        attention_mask[~attention_mask_bool] = -10000.0
        attention_mask = rearrange(attention_mask, 'b s -> b 1 1 s')

        x_unpad, indices, cu_sqs, max_sq_in_batch = unpad_input(x, attention_mask_bool)
        qkv_unpad = rearrange(Wqkv(x_unpad), 'nnz (t h d) -> nnz t h d', t=3,
                              h=nheads).detach().requires_grad_()
        qkv = rearrange(Wqkv(x), 'b s (t h d) -> b s t h d', t=3, h=nheads).detach().requires_grad_()

        print(f'Batch size: {bs}, Sequence Length: {sq}')
        
        fn = lambda qkv_unpad: flash_attn_unpadded_qkvpacked_func(
            qkv_unpad, cu_sqs, max_sq_in_batch, dropout_p, causal=causal
        )
        fa_time,fa_measurement = benchmark_forward(fn, qkv_unpad, repeats=repeats, desc='FlashAttention')
        fn = lambda qkv: attention_ref(qkv, attention_mask_bool, dropout_p, causal=causal)
        pyt_time,pyt_measurement = benchmark_forward(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')

        relative_perf = ((pyt_measurement.mean-fa_measurement.mean)/pyt_measurement.mean) * 100

        result_summary.append([bs,sq,pyt_measurement.mean,fa_measurement.mean,relative_perf])
        
        print(f'Flash Attention Speedup: {relative_perf}\n')

print(f'batch size, sequence length, PyTorch Standard Attention, FlashAttention, speedup relative to PyTorch\n {result_summary}')
