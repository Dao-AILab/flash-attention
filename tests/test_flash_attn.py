# BSD 3 Clause
# Copyright 2023 Advanced Micro Devices, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import torch
import torch.nn.functional as F
import torch.nn as nn

import pytest

from einops import rearrange, repeat

from flash_attn import flash_attn_func, flash_attn_kvpacked_func, flash_attn_qkvpacked_func
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import _get_block_size
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis


<<<<<<< HEAD
flash_attn_func = None


is_sm75 = False #torch.cuda.get_device_capability('cuda') == (7, 5)
is_sm80 = True #torch.cuda.get_device_capability('cuda') == (8, 0)
=======
MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability('cuda') == (7, 5)
is_sm8x = torch.cuda.get_device_capability('cuda')[0] == 8
is_sm80 = torch.cuda.get_device_capability('cuda') == (8, 0)
is_sm90 = torch.cuda.get_device_capability('cuda') == (9, 0)
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc


def generate_random_padding_mask(max_seqlen, batch_size, device, mode='random'):
    assert mode in ['full', 'random', 'third']
    if mode == 'full':
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == 'random':
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen, (batch_size, 1), device=device)
    elif mode == 'third':
        lengths = torch.randint(max_seqlen // 3, max_seqlen, (batch_size, 1), device=device)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), 's -> b s', b=batch_size) < lengths
    return padding_mask


def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None,
                 kvpacked=False, qkvpacked=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen_q)
    else:
        q_unpad = rearrange(q, 'b s h d -> (b s) h d')
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q_unpad.device)
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(output_unpad, '(b s) h d -> b s h d', b=batch_size)

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, 'b s h d -> (b s) h d')
        v_unpad = rearrange(v, 'b s h d -> (b s) h d')
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=k_unpad.device)
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(dqkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (qkv_unpad.detach().requires_grad_(), cu_seqlens_q, max_seqlen_q,
                qkv.detach().requires_grad_(), output_pad_fn, dqkv_pad_fn)
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(dkv_unpad, '(b s) t h d -> b s t h d', b=batch_size)
        return (q_unpad.detach().requires_grad_(), kv_unpad.detach().requires_grad_(),
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                q.detach().requires_grad_(), kv.detach().requires_grad_(),
                output_pad_fn, dq_pad_fn, dkv_pad_fn)
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, '(b s) h d -> b s h d', b=batch_size)
        return (q_unpad.detach().requires_grad_(), k_unpad.detach().requires_grad_(),
                v_unpad.detach().requires_grad_(),
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                q.detach().requires_grad_(), k.detach().requires_grad_(),
                v.detach().requires_grad_(),
                output_pad_fn, dq_pad_fn, dk_pad_fn)


def attention_ref(q, k, v, query_padding_mask=None, key_padding_mask=None, dropout_p=0.0,
                  dropout_mask=None, causal=False, upcast=True, reorder_ops=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(d), k)
    else:
        scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, 'b s -> b s 1 1'), 0.0)
        attention = attention.masked_fill(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(q, kv, query_padding_mask=None, key_padding_mask=None, dropout_p=0.0,
                           dropout_mask=None, causal=False, upcast=True, reorder_ops=False):
    return attention_ref(q, kv[:, :, 0], kv[:, :, 1], query_padding_mask,
                         key_padding_mask, dropout_p, dropout_mask, upcast=upcast, causal=causal,
                         reorder_ops=reorder_ops)


def attention_qkvpacked_ref(qkv, key_padding_mask=None, dropout_p=0.0,
                            dropout_mask=None, causal=False, upcast=True, reorder_ops=False):
    return attention_ref(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], key_padding_mask,
                         key_padding_mask, dropout_p, dropout_mask, upcast=upcast, causal=causal,
                         reorder_ops=reorder_ops)


def generate_sparsity_mask(seqlen, sparsity=0.3):
    repeats = seqlen // 16 // 2
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([0, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    nrow, ncol = seqlen // 16, seqlen // 256
    mask = torch.rand(nrow, ncol, device='cuda') < sparsity
    return mask


def attention_blocksparse_ref(qkv, blockmask, attn_mask, dropout_p, dropout_mask):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        blockmask: (seqlen / 16, seqlen / 256)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen, seqlen)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = qkv.float().unbind(dim=2)
    d = qkv.shape[-1]
    seqlen = qkv.shape[1]
    scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(d), k)
    scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    blockmask = repeat(blockmask, 's_16 s_256 -> (s_16 16) (s_256 256)')
    blockmask = blockmask[:seqlen, :seqlen]
    scores.masked_fill_(rearrange(~blockmask, 't s -> 1 1 t s'), float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(rearrange(~attn_mask, 'b s -> b 1 s 1'), 0.0)
    attention = attention.masked_fill_(rearrange(~blockmask, 't s -> 1 1 t s'), 0.0)
    attention_drop = attention.masked_fill(~dropout_mask, 0.0) / (1 - dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    output.masked_fill_(rearrange(~attn_mask, 'b s -> b s 1 1'), 0)
    return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)


def convert_flash_attn_S_to_softmax(S, query_padding_mask, key_padding_mask, head_dim, is_dropout,
                                    causal=False):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
    """
    seqlen_q, seqlen_k = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(S.device, head_dim, is_dropout, causal)
    nblocks_n = (seqlen_k + blocksize_n - 1) // blocksize_n
    nblocks_m = (seqlen_q + blocksize_m - 1) // blocksize_m
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(S, 'b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)',
                       blocksize_m=blocksize_m, blocksize_n=blocksize_n)
    S_converted = rearrange(S_flat, 'b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)',
                            mmas_n=mmas_n, warps_n=warps_n, eight=8, c0=2, c1=2, c2=2, four=4)
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=S.device), 1)
        S_converted.masked_fill_(causal_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q
    if query_padding_mask is not None:
        if seqlen_q_og < seqlen_q:
            query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q - seqlen_q_og))
        else:
            query_padding_mask = query_padding_mask[:, :seqlen_q]
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        if seqlen_k_og < seqlen_k:
            key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k - seqlen_k_og))
        else:
            key_padding_mask = key_padding_mask[:, :seqlen_k]
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), 0.0)
    if seqlen_q_og < seqlen_q:
        S_converted = S_converted[:, :, :seqlen_q_og, :]
    else:
        S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q))
    if seqlen_k_og < seqlen_k:
        S_converted = S_converted[:, :, :, :seqlen_k_og]
    else:
        S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k))
    return S_converted


def normalize_flash_attn_S(attn_unnorm, q, k, v, query_padding_mask=None, key_padding_mask=None,
                           is_dropout=False, causal=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum('bthd,bshd->bhts', q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    _, block_size_n = _get_block_size(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat([a / rearrange(torch.exp(lse - m), 'b h s -> b h s 1')
                           for a, m in zip(attn_unnorm_block, cummax_block)], dim=-1)
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(dropout_mask, query_padding_mask=None, key_padding_mask=None, causal=False):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), False)
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool,
                                            device=dropout_mask.device), 1)
        dropped.masked_fill_(causal_mask, False)
    dropped_total = dropped.sum()
    query_lengths = (query_padding_mask.sum(dim=-1) if query_padding_mask is not None
                     else torch.full((batch_size,), seqlen_q, device=dropout_mask.device))
    key_lengths = (key_padding_mask.sum(dim=-1) if key_padding_mask is not None
                   else torch.full((batch_size,), seqlen_k, device=dropout_mask.device))
    if not causal:
        numel_per_batch = query_lengths * key_lengths
    else:
        numel_per_batch = torch.where(
            query_lengths <= key_lengths,
            query_lengths * (query_lengths + 1) / 2,
            query_lengths * key_lengths - (key_lengths * (key_lengths - 1) / 2)
        )
    return dropped_total / (numel_per_batch.sum() * nheads)

def get_dropout_mask(S_dmask, dropout_p, cu_seqlens_q, cu_seqlens_k, batch_size, nheads, seqlen):
    if(dropout_p == 0.0):
        dropout_mask = torch.full([batch_size, nheads, seqlen, seqlen], True , device=S_dmask.device)
    else:
        S_dmask_converted = torch.full([batch_size, nheads, seqlen, seqlen], 0, dtype=torch.int32 , device=S_dmask.device)
        for i in range(batch_size):
            current_seqlen_q = cu_seqlens_q[i+1] - cu_seqlens_q[i]
            current_seqlen_k = cu_seqlens_k[i+1] - cu_seqlens_k[i]
            #print(f'current_seqlen: {current_seqlen}')
            S_dmask_each = S_dmask[i].view(-1).contiguous()
            #print(f'S_dmask_each.size(): {S_dmask_each.size()}')
            for j in range(nheads):
                ZeroPad = nn.ZeroPad2d(padding=(0, seqlen - current_seqlen_k, 0, seqlen - current_seqlen_q))
                S_dmask_converted_each = S_dmask_each[j * current_seqlen_q * current_seqlen_k : (j+1) * current_seqlen_q * current_seqlen_k]
                S_dmask_converted_each_2d = S_dmask_converted_each.view(current_seqlen_q, current_seqlen_k).contiguous()
                S_dmask_converted_each_2d_pad = ZeroPad(S_dmask_converted_each_2d)
                S_dmask_converted[i][j] = S_dmask_converted_each_2d_pad
                #for k in range(current_seqlen_q):
                #    index_for_S_dmask_start = j * current_seqlen_q * current_seqlen_k + k* current_seqlen_k
                #    index_for_S_dmask_end = j * current_seqlen_q * current_seqlen_k + k* current_seqlen_k + current_seqlen_k
                #    S_dmask_converted[i][j][k][0 : current_seqlen_k] = S_dmask_each[index_for_S_dmask_start : index_for_S_dmask_end]
                #    #for m in range(current_seqlen_k):
                #    #    index_for_S_dmask = j * current_seqlen_q * current_seqlen_k + k* current_seqlen_k + m
                #    #    S_dmask_converted[i][j][k][m] = S_dmask_each[index_for_S_dmask]
        dropout_mask_t = S_dmask_converted <= ((1 - dropout_p) * 65535)
        dropout_mask = dropout_mask_t.contiguous()
    return dropout_mask

@pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('causal', [False, True])
<<<<<<< HEAD
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('d', [128, 64, 80, 40, 32, 16])
# @pytest.mark.parametrize('d', [128])
=======
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize('d', [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128])
# @pytest.mark.parametrize('d', [64])
# @pytest.mark.parametrize('seqlen', [128, 256, 384, 512, 768, 1024, 2048])
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc
@pytest.mark.parametrize('seqlen', [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048])
# @pytest.mark.parametrize('seqlen', [97])
@pytest.mark.parametrize('dropout_p', [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.17])
<<<<<<< HEAD
def test_flash_attn_unpadded_qkvpacked(seqlen, d, dropout_p, causal, dtype):
=======
def test_flash_attn_qkvpacked(seqlen, d, dropout_p, causal, dtype):
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc
    if seqlen >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
<<<<<<< HEAD
    # Set smaller batch size so it would trigger num_splits > 1
    batch_size = 8
    nheads = 4
    x = torch.randn(batch_size, seqlen, nheads * d, device=device, dtype=dtype, requires_grad=True)
    Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='random')
    #key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='full')

    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        x, Wqkv, nheads, key_padding_mask, key_padding_mask, qkvpacked=True
    )

    output_unpad, sm_lse, S_dmask = flash_attn_unpadded_qkvpacked_func(
        qkv_unpad, cu_seqlens, max_seqlen, dropout_p, return_attn_probs=True, causal=causal
    )
    output = output_pad_fn(output_unpad)

    dropout_mask = get_dropout_mask(S_dmask, dropout_p, cu_seqlens, cu_seqlens, batch_size, nheads, seqlen)

    dropout_fraction = get_dropout_fraction(dropout_mask, key_padding_mask, key_padding_mask,
                                            causal=causal).item()

    output_ref, attn_ref = attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                                   causal=causal)
    output_pt, attn_pt = attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                                 causal=causal, upcast=False, reorder_ops=True)
    print(f'Actual dropout fraction: {dropout_fraction}')
    print(f'Output max diff: {(output - output_ref).abs().max().item()}')
    print(f'Output mean diff: {(output - output_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(output_pt - output_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(output_pt - output_ref).abs().mean().item()}')
    #print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
    #print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
        g = torch.randn_like(output)
        dqkv_unpad, = torch.autograd.grad(output, qkv_unpad, g)
        dqkv = dqkv_pad_fn(dqkv_unpad)
        dqkv_ref, = torch.autograd.grad(output_ref, qkv, g)
        dqkv_pt, = torch.autograd.grad(output_pt, qkv, g)
=======
    batch_size = 16
    nheads = 9
    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype,
                      requires_grad=True)
    out, lse, S_dmask = flash_attn_qkvpacked_func(
        qkv, dropout_p, return_attn_probs=True, causal=causal
    )
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask, None, None, d, dropout_p > 0.0, causal=causal
        )[:, :, :seqlen, :seqlen]
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        attn = normalize_flash_attn_S(attn_unnorm, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                                      None, None, dropout_p > 0.0, causal=causal)
        dropout_fraction = get_dropout_fraction(dropout_mask, None, None, causal=causal).item()
        print(f'Actual dropout fraction: {dropout_fraction}')
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(qkv, None, dropout_p, dropout_mask, causal=causal)
    out_pt, attn_pt = attention_qkvpacked_ref(qkv, None, dropout_p, dropout_mask, causal=causal,
                                              upcast=False, reorder_ops=True)
    # v = qkv[:, :, 2].float()
    # qk = torch.einsum('bshd,bthd->bhst', qkv[:, :, 0], qkv[:, :, 1]).float()
    # if causal:
    #     causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
    #     qk.masked_fill_(causal_mask, float('-inf'))
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # p_tmp = torch.softmax(qk / math.sqrt(d), -1)
    # p_dropped = p_tmp if dropout_mask is None else p_tmp.masked_fill(~dropout_mask, 0)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # qk_max1 = torch.max(qk[:, :, 128:, 192:], -1, keepdim=True).values
    # qk_max2 = torch.max(qk[:, :, 128:, 128:], -1, keepdim=True).values
    # qk_max3 = torch.max(qk[:, :, 128:, 64:], -1, keepdim=True).values
    # qk_max4 = torch.max(qk[:, :, 128:, :], -1, keepdim=True).values
    # o1 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 192:] - qk_max1) / math.sqrt(d)), v[:, 192:])
    # o2 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 128:] - qk_max2) / math.sqrt(d)), v[:, 128:])
    # o3 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 64:] - qk_max3) / math.sqrt(d)), v[:, 64:])
    # o4 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, :] - qk_max4) / math.sqrt(d)), v[:, :])
    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(out_pt - out_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}')
    if dropout_p > 0.0:
        print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
        print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    g = torch.randn_like(out)
    # do_o = (g.float() * out.float()).sum(-1)
    # dv_tmp = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, :64], g[:, :64])
    # dv_tmp1 = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, 64:], g[:, 64:])
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        dqkv, = torch.autograd.grad(out, qkv, g)
        dqkv_ref, = torch.autograd.grad(out_ref, qkv, g)
        dqkv_pt, = torch.autograd.grad(out_pt, qkv, g)
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc
        print(f'dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
        print(f'dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
        print(f'dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}')
        print(f'dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
        print(f'dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
        print(f'dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
<<<<<<< HEAD
    assert (output - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item()
    # assert torch.allclose(output, output_ref, rtol=rtol, atol=atol)
    # assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    # assert torch.allclose(attn, attn_ref, rtol=rtol, atol=atol)
    if dropout_p == 0.0:
        assert dropout_mask.all()
    else:
        assert 0.98 <= dropout_fraction / dropout_p <= 1.02
=======
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        assert abs(dropout_fraction - dropout_p) <= 0.01

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (dqkv_pt - dqkv_ref).abs().max().item()
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc



@pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('causal', [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('d', [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('seqlen', [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize('dropout_p', [0.0, 0.17])
<<<<<<< HEAD
# @pytest.mark.parametrize('dropout_p', [0.17])
def test_flash_attn_unpadded_kvpacked(seqlen, d, dropout_p, causal, dtype):
=======
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_varlen_qkvpacked(seqlen, d, dropout_p, causal, dtype):
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc
    if seqlen >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 6
    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype,
                      requires_grad=True)

    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='random')
    # key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='full')

    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        *qkv.unbind(dim=2), key_padding_mask, key_padding_mask, qkvpacked=True
    )
<<<<<<< HEAD
    output = output_pad_fn(output_unpad)

    dropout_mask = get_dropout_mask(S_dmask, dropout_p, cu_seqlens_q, cu_seqlens_k, batch_size, nheads, seqlen)

    dropout_fraction = get_dropout_fraction(dropout_mask, query_padding_mask, key_padding_mask,
                                            causal=causal)

    output_ref, attn_ref = attention_kvpacked_ref(q, kv, query_padding_mask, key_padding_mask,
                                                  dropout_p, dropout_mask, causal=causal)
    output_pt, attn_pt = attention_kvpacked_ref(q, kv, query_padding_mask, key_padding_mask,
                                                dropout_p, dropout_mask, causal=causal,
                                                upcast=False, reorder_ops=True)
    print(f'Actual dropout fraction: {dropout_fraction}')
    print(f'Output max diff: {(output - output_ref).abs().max().item()}')
    print(f'Output mean diff: {(output - output_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(output_pt - output_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(output_pt - output_ref).abs().mean().item()}')
    #print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
    #print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
        g = torch.randn_like(output)
        dq_unpad, dkv_unpad, = torch.autograd.grad(output, (q_unpad, kv_unpad), g)
        dq = dq_pad_fn(dq_unpad)
        dkv = dkv_pad_fn(dkv_unpad)
        dq_ref, dkv_ref, = torch.autograd.grad(output_ref, (q, kv), g)
        dq_pt, dkv_pt = torch.autograd.grad(output_pt, (q, kv), g)
        print(f'dQ max diff: {(dq - dq_ref).abs().max().item()}')
        print(f'dK max diff: {(dkv[:, :, 0] - dkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dV max diff: {(dkv[:, :, 1] - dkv_ref[:, :, 1]).abs().max().item()}')
        print(f'dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}')
        print(f'dK Pytorch max diff: {(dkv_pt[:, :, 0] - dkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dV Pytorch max diff: {(dkv_pt[:, :, 1] - dkv_ref[:, :, 1]).abs().max().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (output - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item()
    # assert torch.allclose(output, output_ref, rtol=rtol, atol=atol)
    # assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    # assert torch.allclose(attn, attn_ref, rtol=rtol, atol=atol)
    if dropout_p == 0.0:
        assert dropout_mask.all()
    else:
        assert 0.99 <= dropout_fraction / dropout_p <= 1.01
=======

    out_unpad, sm_lse, S_dmask = flash_attn_varlen_qkvpacked_func(
        qkv_unpad, cu_seqlens, max_seqlen, dropout_p, return_attn_probs=True, causal=causal
    )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask, key_padding_mask, key_padding_mask, d, dropout_p > 0.0, causal=causal
        )[:, :, :seqlen, :seqlen]
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        attn = normalize_flash_attn_S(attn_unnorm, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                                      key_padding_mask, key_padding_mask, dropout_p > 0.0,
                                      causal=causal)
        dropout_fraction = get_dropout_fraction(dropout_mask, key_padding_mask, key_padding_mask,
                                                causal=causal).item()
        print(f'Actual dropout fraction: {dropout_fraction}')
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                                causal=causal)
    out_pt, attn_pt = attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                              causal=causal, upcast=False, reorder_ops=True)
    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(out_pt - out_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}')
    if dropout_p > 0.0:
        print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
        print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    g = torch.randn_like(out)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        dqkv_unpad, = torch.autograd.grad(out, qkv_unpad, g)
        dqkv = dqkv_pad_fn(dqkv_unpad)
        dqkv_ref, = torch.autograd.grad(out_ref, qkv, g)
        dqkv_pt, = torch.autograd.grad(out_pt, qkv, g)
        print(f'dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
        print(f'dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
        print(f'dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}')
        print(f'dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
        print(f'dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
        print(f'dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
        print(f'dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        assert abs(dropout_fraction - dropout_p) <= 0.01

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (dqkv_pt - dqkv_ref).abs().max().item()


@pytest.mark.parametrize('kvpacked', [True, False])
# @pytest.mark.parametrize('kvpacked', [False])
@pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('mha_type', ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize('mha_type', ["mha"])
@pytest.mark.parametrize('causal', [False, True])
<<<<<<< HEAD
@pytest.mark.parametrize('d', [128, 64, 80, 40, 32, 16])
# @pytest.mark.parametrize('d', [16])
@pytest.mark.parametrize('seqlen', [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048])
# @pytest.mark.parametrize('seqlen', [16])
=======
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('d', [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('seqlen_q,seqlen_k', [(113, 203), (128, 217), (113, 211), (108, 256), (256, 512), (512, 256), (1024, 1024), (1023, 1024), (1024, 1023), (2048, 2048)])
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc
@pytest.mark.parametrize('dropout_p', [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_output(seqlen_q, seqlen_k, d, dropout_p, causal, mha_type, dtype, kvpacked):
    if max(seqlen_q, seqlen_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 16
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype,
                         requires_grad=True)
    else:
        k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype,
                        requires_grad=True)
        v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype,
                        requires_grad=True)

    if kvpacked:
        out, lse, S_dmask = flash_attn_kvpacked_func(
            q, kv, dropout_p, return_attn_probs=True, causal=causal
        )
    else:
        out, lse, S_dmask = flash_attn_func(
            q, k, v, dropout_p, return_attn_probs=True, causal=causal
        )
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask, None, None, d, dropout_p > 0.0, causal=causal
        )[:, :, :seqlen_q, :seqlen_k]
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(attn_unnorm, q, k_rep, v_rep,
                                      None, None, dropout_p > 0.0, causal=causal)
        dropout_fraction = get_dropout_fraction(dropout_mask, None, None, causal=causal).item()
        print(f'Actual dropout fraction: {dropout_fraction}')
    else:
        dropout_mask = None

<<<<<<< HEAD
    output_unpad, sm_lse, S_dmask = flash_attn_unpadded_func(
        q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, return_attn_probs=True, causal=causal
    )

    output = output_pad_fn(output_unpad)

    dropout_mask = get_dropout_mask(S_dmask, dropout_p, cu_seqlens_q, cu_seqlens_k, batch_size, nheads, seqlen)

    dropout_fraction = get_dropout_fraction(dropout_mask, query_padding_mask, key_padding_mask,
                                            causal=causal)

    output_ref, attn_ref = attention_ref(q, k, v, query_padding_mask, key_padding_mask,
                                         dropout_p, dropout_mask, causal=causal)
    output_pt, attn_pt = attention_ref(q, k, v, query_padding_mask, key_padding_mask,
                                       dropout_p, dropout_mask, causal=causal,
                                       upcast=False, reorder_ops=True)
    print(f'Actual dropout fraction: {dropout_fraction}')
    print(f'Output max diff: {(output - output_ref).abs().max().item()}')
    print(f'Output mean diff: {(output - output_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(output_pt - output_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(output_pt - output_ref).abs().mean().item()}')
    #print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
    #print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
        g = torch.randn_like(output)
        dq_unpad, dk_unpad, dv_unpad, = torch.autograd.grad(output, (q_unpad, k_unpad, v_unpad), g)
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        dq_ref, dk_ref, dv_ref, = torch.autograd.grad(output_ref, (q, k, v), g)
        dq_pt, dk_pt, dv_pt, = torch.autograd.grad(output_pt, (q, k, v), g)
=======
    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(q, kv, None, None, dropout_p, dropout_mask,
                                                causal=causal)
        out_pt, attn_pt = attention_kvpacked_ref(q, kv, None, None, dropout_p, dropout_mask,
                                                causal=causal, upcast=False, reorder_ops=True)
    else:
        out_ref, attn_ref = attention_ref(q, k, v, None, None, dropout_p, dropout_mask,
                                          causal=causal)
        out_pt, attn_pt = attention_ref(q, k, v, None, None, dropout_p, dropout_mask,
                                        causal=causal, upcast=False, reorder_ops=True)

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(out_pt - out_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}')
    if dropout_p > 0.0:
        print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
        print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        if kvpacked:
            dq, dkv, = torch.autograd.grad(out, (q, kv), g)
            dk, dv = dkv.unbind(2)
            dq_ref, dkv_ref, = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            dq_pt, dkv_pt, = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        else:
            dq, dk, dv, = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref, = torch.autograd.grad(out_ref, (q, k, v), g)
            dq_pt, dk_pt, dv_pt, = torch.autograd.grad(out_pt, (q, k, v), g)
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc
        print(f'dQ max diff: {(dq - dq_ref).abs().max().item()}')
        print(f'dK max diff: {(dk - dk_ref).abs().max().item()}')
        print(f'dV max diff: {(dv - dv_ref).abs().max().item()}')
        print(f'dQ mean diff: {(dq - dq_ref).abs().mean().item()}')
        print(f'dK mean diff: {(dk - dk_ref).abs().mean().item()}')
        print(f'dV mean diff: {(dv - dv_ref).abs().mean().item()}')
        print(f'dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}')
        print(f'dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}')
        print(f'dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}')
        print(f'dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}')
        print(f'dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}')
        print(f'dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
<<<<<<< HEAD
    assert (output - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item()
    # assert torch.allclose(output, output_ref, rtol=rtol, atol=atol)
    # assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    # assert torch.allclose(attn, attn_ref, rtol=rtol, atol=atol)
    if dropout_p == 0.0:
        assert dropout_mask.all()
    else:
        assert 0.99 <= dropout_fraction / dropout_p <= 1.01
=======
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        assert abs(dropout_fraction - dropout_p) <= 0.01
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize('kvpacked', [True, False])
# @pytest.mark.parametrize('kvpacked', [False])
@pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('mha_type', ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize('mha_type', ["mqa"])
@pytest.mark.parametrize('causal', [False, True])
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize('d', [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('seqlen_q,seqlen_k', [(113, 203), (128, 217), (113, 211), (108, 256), (256, 512), (512, 256), (1024, 1024), (1023, 1024), (1024, 1023), (2048, 2048)])
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
@pytest.mark.parametrize('dropout_p', [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_varlen_output(seqlen_q, seqlen_k, d, dropout_p, causal, mha_type, dtype,
                                  kvpacked):
    if max(seqlen_q, seqlen_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 16
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype,
                         requires_grad=True)
    else:
<<<<<<< HEAD
        assert 0.99 <= dropout_fraction / dropout_p <= 1.01

    if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (dqkv_pt - dqkv_ref).abs().max().item()
        # assert torch.allclose(dqkv, dqkv_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('d', [128, 64, 80, 40, 32, 16])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('seqlen', [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize('dropout_p', [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_race_condition(seqlen, d, dropout_p, causal, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    nheads = 4
    x = torch.randn(batch_size, seqlen, nheads * d, device=device, dtype=dtype, requires_grad=True)
    Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

    query_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='random')
    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='random')

    (q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, k, v,
     output_pad_fn, dq_pad_fn, dk_pad_fn) = generate_qkv(
         x, Wqkv, nheads, query_padding_mask, key_padding_mask
     )

    torch.random.manual_seed(0)
    output_unpad_0, sm_lse_0, S_dmask_0 = flash_attn_unpadded_func(
        q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, return_attn_probs=True, causal=causal
    )
    # S_dmask_converted_0 = convert_flash_attn_S_to_softmax(
    #     S_dmask_0, query_padding_mask, key_padding_mask, d, dropout_p > 0.0, causal=causal
    # )

    if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
        g = torch.randn_like(output_unpad_0)
        dq_unpad_0, dk_unpad_0, dv_unpad_0, = torch.autograd.grad(output_unpad_0,
                                                                  (q_unpad, k_unpad, v_unpad), g)
        # Parallelizing over seqlen_k makes dq non-deterministic
        deterministic_dq = False
        # Numerical error if we just do any arithmetic on dq
        dq_atol = ((dq_unpad_0 + 0.3 - 0.3) - dq_unpad_0).abs().max().item()
        equal_fn = torch.equal if deterministic_dq else partial(torch.allclose, atol=dq_atol)

    for _ in range(10):
        torch.random.manual_seed(0)
        output_unpad, sm_lse, S_dmask = flash_attn_unpadded_func(
            q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, return_attn_probs=True, causal=causal
        )
        # S_dmask_converted = convert_flash_attn_S_to_softmax(
        #     S_dmask, query_padding_mask, key_padding_mask, d, dropout_p > 0.0, causal=causal
        # )
        assert torch.equal(output_unpad, output_unpad_0)
        # sm_lse has some parts that are uninitialized from torch.empty
        # assert torch.equal(sm_lse, sm_lse_0)
        # assert torch.equal(S_dmask_converted, S_dmask_converted_0)

        if is_sm80 or d <= 64:  # Only run backward for d=128 on A100
            dq_unpad, dk_unpad, dv_unpad, = torch.autograd.grad(output_unpad,
                                                                (q_unpad, k_unpad, v_unpad), g)
            assert equal_fn(dq_unpad, dq_unpad_0)
            assert torch.equal(dk_unpad, dk_unpad_0)
            assert torch.equal(dv_unpad, dv_unpad_0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires multiple GPUs')
def test_flash_attn_multigpu():
    seqlen = 256
    d = 64
    dropout_p = 0.0
    causal = False
    dtype = torch.float16
    device = 'cuda:1'
    torch.random.manual_seed(0)
    batch_size = 32
    nheads = 4
    x = torch.randn(batch_size, seqlen, nheads * d, device=device, dtype=dtype, requires_grad=True)
    Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='random')
    # key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='full')

    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        x, Wqkv, nheads, key_padding_mask, key_padding_mask, qkvpacked=True
    )

    output_unpad, sm_lse, S_dmask = flash_attn_unpadded_qkvpacked_func(
        qkv_unpad, cu_seqlens, max_seqlen, dropout_p, return_attn_probs=True, causal=causal
    )
    output = output_pad_fn(output_unpad)

    dropout_mask = get_dropout_mask(S_dmask, dropout_p, cu_seqlens, cu_seqlens, batch_size, nheads, seqlen)

    # S_dmask_converted = convert_flash_attn_S_to_softmax(
    #     S_dmask, key_padding_mask, key_padding_mask, d, dropout_p > 0.0, causal=causal
    # )
    # dropout_mask = S_dmask_converted >= 0
    # attn_unnorm = S_dmask_converted.abs()
    # attn = normalize_flash_attn_S(attn_unnorm, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
    #                               key_padding_mask, key_padding_mask, dropout_p > 0.0, causal=causal)
    dropout_fraction = get_dropout_fraction(dropout_mask, key_padding_mask, key_padding_mask,
                                            causal=causal).item()

    output_ref, attn_ref = attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                                   causal=causal)
    output_pt, attn_pt = attention_qkvpacked_ref(qkv, key_padding_mask, dropout_p, dropout_mask,
                                                 causal=causal, upcast=False, reorder_ops=True)
    print(f'Actual dropout fraction: {dropout_fraction}')
    print(f'Output max diff: {(output - output_ref).abs().max().item()}')
    print(f'Output mean diff: {(output - output_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(output_pt - output_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(output_pt - output_ref).abs().mean().item()}')
    #print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
    #print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    g = torch.randn_like(output)
    dqkv_unpad, = torch.autograd.grad(output, qkv_unpad, g)
    dqkv = dqkv_pad_fn(dqkv_unpad)
    dqkv_ref, = torch.autograd.grad(output_ref, qkv, g)
    dqkv_pt, = torch.autograd.grad(output_pt, qkv, g)
    print(f'dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
    print(f'dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
    print(f'dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
    print(f'dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}')
    print(f'dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}')
    print(f'dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}')
    print(f'dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}')
    print(f'dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (output - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item()
    # assert torch.allclose(output, output_ref, rtol=rtol, atol=atol)
    # assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    # assert torch.allclose(attn, attn_ref, rtol=rtol, atol=atol)
    if dropout_p == 0.0:
        assert dropout_mask.all()
    else:
        assert 0.99 <= dropout_fraction / dropout_p <= 1.01
=======
        k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype,
                        requires_grad=True)
        v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype,
                        requires_grad=True)

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode='random')
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='random')
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')

    if kvpacked:
        (q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, kv,
        output_pad_fn, dq_pad_fn, dkv_pad_fn) = generate_qkv(
            q, *kv.unbind(dim=2), query_padding_mask, key_padding_mask, kvpacked=True
        )
        out_unpad, sm_lse, S_dmask = flash_attn_varlen_kvpacked_func(
            q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, return_attn_probs=True, causal=causal
        )
    else:
        (q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, q, k, v,
        output_pad_fn, dq_pad_fn, dk_pad_fn) = generate_qkv(
            q, k, v, query_padding_mask, key_padding_mask, kvpacked=False
        )
        out_unpad, sm_lse, S_dmask = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, return_attn_probs=True, causal=causal
        )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask, query_padding_mask, key_padding_mask, d, dropout_p > 0.0, causal=causal
        )[:, :, :seqlen_q, :seqlen_k]
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(attn_unnorm, q, k_rep, v_rep,
                                      query_padding_mask, key_padding_mask,
                                      dropout_p > 0.0, causal=causal)
        dropout_fraction = get_dropout_fraction(dropout_mask, query_padding_mask,
                                                key_padding_mask, causal=causal).item()
        print(f'Actual dropout fraction: {dropout_fraction}')
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(q, kv, query_padding_mask, key_padding_mask,
                                                dropout_p, dropout_mask, causal=causal)
        out_pt, attn_pt = attention_kvpacked_ref(q, kv, query_padding_mask, key_padding_mask,
                                                dropout_p, dropout_mask,
                                                causal=causal, upcast=False, reorder_ops=True)
    else:
        out_ref, attn_ref = attention_ref(q, k, v, query_padding_mask, key_padding_mask,
                                          dropout_p, dropout_mask, causal=causal)
        out_pt, attn_pt = attention_ref(q, k, v, query_padding_mask, key_padding_mask,
                                        dropout_p, dropout_mask,
                                        causal=causal, upcast=False, reorder_ops=True)

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    print(f'Pytorch max diff: {(out_pt - out_ref).abs().max().item()}')
    print(f'Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}')
    if dropout_p > 0.0:
        print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
        print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    g = torch.randn_like(out)
    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        if kvpacked:
            dq_unpad, dkv_unpad, = torch.autograd.grad(out, (q_unpad, kv_unpad), g)
            dk, dv = dkv_pad_fn(dkv_unpad).unbind(2)
            dq_ref, dkv_ref, = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            dq_pt, dkv_pt, = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        else:
            dq_unpad, dk_unpad, dv_unpad, = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            dq_ref, dk_ref, dv_ref, = torch.autograd.grad(out_ref, (q, k, v), g)
            dq_pt, dk_pt, dv_pt, = torch.autograd.grad(out_pt, (q, k, v), g)
        dq = dq_pad_fn(dq_unpad)
        print(f'dQ max diff: {(dq - dq_ref).abs().max().item()}')
        print(f'dK max diff: {(dk - dk_ref).abs().max().item()}')
        print(f'dV max diff: {(dv - dv_ref).abs().max().item()}')
        print(f'dQ mean diff: {(dq - dq_ref).abs().mean().item()}')
        print(f'dK mean diff: {(dk - dk_ref).abs().mean().item()}')
        print(f'dV mean diff: {(dv - dv_ref).abs().mean().item()}')
        print(f'dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}')
        print(f'dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}')
        print(f'dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}')
        print(f'dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}')
        print(f'dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}')
        print(f'dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc

    if dropout_p > 0.0:
        assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
        assert abs(dropout_fraction - dropout_p) <= 0.01

    if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()


# @pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('causal', [False, True])
# @pytest.mark.parametrize('causal', [True])
# @pytest.mark.parametrize('d', [32, 56, 64, 80, 96, 128])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
@pytest.mark.parametrize('d', [64])
# @pytest.mark.parametrize('seqlen', [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048])
@pytest.mark.parametrize('seqlen', [128, 256, 384, 512, 768, 1024, 2048])
# @pytest.mark.parametrize('seqlen', [193])
# @pytest.mark.parametrize('dropout_p', [0.0, 0.17])
@pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_race_condition(seqlen, d, dropout_p, causal, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    nheads = 4
    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    out0, lse0, _ = flash_attn_qkvpacked_func(
        qkv, dropout_p, return_attn_probs=True, causal=causal
    )
    g = torch.randn_like(out0)
    dqkv0, = torch.autograd.grad(out0, qkv, g)

    for _ in range(200):
        torch.random.manual_seed(0)
        out, lse, S_dmask = flash_attn_qkvpacked_func(
            qkv, dropout_p, return_attn_probs=True, causal=causal
        )
        assert torch.equal(out, out0)
        assert torch.equal(lse, lse0)
        # sm_lse has some parts that are uninitialized from torch.empty
        # assert torch.equal(sm_lse, sm_lse_0)

<<<<<<< HEAD
@pytest.mark.skipif(flash_attn_func is None, reason='Triton is not installed or is too old')
@pytest.mark.skipif(not is_sm80, reason='Triton version is only tested on A100')
@pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('causal', [False, True])
# @pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize('d', [40, 48, 64, 128, 80, 88, 96])
# @pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('seqlen_q,seqlen_k', [(113, 203), (128, 217), (91, 211), (108, 256), (256, 512), (512, 256), (1024, 1024), (1023, 1024), (1024, 1023), (2048, 2048)])
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(113, 203)])
@pytest.mark.parametrize('bias_shape', ([None, '1h1k', '1hqk', 'b11k', 'b1qk']))
# @pytest.mark.parametrize('bias_shape', (['b1qk']))
def test_flash_attn_triton_race_condition(seqlen_q, seqlen_k, d, causal, dtype, bias_shape):
    if seqlen_q >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k, v = torch.randn(batch_size, seqlen_k, 2, nheads, d, device=device, dtype=dtype).unbind(dim=2)
    if bias_shape == '1h1k':
        bias = torch.randn(1, nheads, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == '1hqk':
        bias = torch.randn(1, nheads, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == 'b11k':
        bias = torch.randn(batch_size, 1, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == 'b1qk':
        bias = torch.randn(batch_size, 1, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    else:
        bias = None

    q, k, v = [x.detach().requires_grad_() for x in [q, k, v]]
    output_0 = flash_attn_func(q, k, v, bias, causal)

    g = torch.randn_like(output_0)
    dq_0, dk_0, dv_0 = torch.autograd.grad(output_0, (q, k, v), g)

    # The SEQUENCE_PARALLEL option for the bwd to makes dq non-deterministic
    deterministic_dq = False
    # Numerical error if we just do any arithmetic on dq
    dq_atol = ((dq_0 + 0.3 - 0.3) - dq_0).abs().max().item()
    equal_fn = torch.equal if deterministic_dq else partial(torch.allclose, atol=dq_atol)
    # Run 10000 times and check that the results don't change
    for i in range(10000):
        output = flash_attn_func(q, k, v, bias, causal)
        output_equal = torch.equal(output, output_0)
        if not output_equal:  # Printing / computing diff sometimes makes the race condition disappear
            #print(f'{dtype = }, {causal = }, {d = }, {seqlen_q = }, {seqlen_k = }, {bias_shape = }, {i = }')
            print(f'Output max diff: {(output - output_0).abs().max().item()}')
        assert torch.equal(output, output_0)
        dq, dk, dv = torch.autograd.grad(output, (q, k, v), g)
        dq_equal = equal_fn(dq, dq_0)
        dk_equal = torch.equal(dk, dk_0)
        dv_equal = torch.equal(dv, dv_0)
        if not (dq_equal and dk_equal and dv_equal):
            #print(f'{dtype = }, {causal = }, {d = }, {seqlen_q = }, {seqlen_k = }, {bias_shape = }, {i = }')
            print(f'dQ max diff: {(dq - dq_0).abs().max().item()}')
            print(f'dK max diff: {(dk - dk_0).abs().max().item()}')
            print(f'dV max diff: {(dv - dv_0).abs().max().item()}')
        assert equal_fn(dq, dq_0)
        assert torch.equal(dk, dk_0)
        assert torch.equal(dv, dv_0)
=======
        if not (is_sm75 and d == 128):
            dqkv, = torch.autograd.grad(out, qkv, g)
            assert torch.equal(dqkv[:, :, 0], dqkv0[:, :, 0])
            assert torch.equal(dqkv[:, :, 1], dqkv0[:, :, 1])
            assert torch.equal(dqkv[:, :, 2], dqkv0[:, :, 2])
>>>>>>> 4f285b354796fb17df8636485b9a04df3ebbb7dc
