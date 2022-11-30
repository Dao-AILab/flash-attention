# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from typing import Optional, Tuple
import torch
import math
from torch.nn import Linear
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn.flash_attn_triton import flash_attn_func
def attention_ref(q, k, v, bias=None, mask=None, num_heads=1):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        bias: (batch_size, nheads, seqlen_q, seqlen_k)
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
    q = rearrange(q, "b s (n d) -> b s n d",n=num_heads)
    k = rearrange(k, "b s (n d) -> b s n d",n=num_heads)
    v = rearrange(v, "b s (n d) -> b s n d",n=num_heads)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    if bias is not None:
        scores = (scores + bias).to(dtype=scores.dtype)
    if mask is None:
        mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1)
    scores.masked_fill_(mask==False, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    attention_drop = attention
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v )
    return output.to(dtype=dtype_og)
def attn_ref(q,k,v,bias,mask,num_heads):
    
    h_q = rearrange(q, "b s (n d) -> b n s d",n=num_heads)
    h_k = rearrange(k, "b s (n d) -> b n s d",n=num_heads)
    h_v = rearrange(v, "b s (n d) -> b n s d",n=num_heads)
    # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
    score = torch.matmul(h_q, h_k.transpose(-1, -2)) / math.sqrt(h_q.shape[3])
    if bias is not None:
        score = score + bias
    score = torch.masked_fill(
            score,
            mask==False,
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )
    score = F.softmax(score, dim = -1)
    score = torch.masked_fill(
            score,
            mask==False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )

    # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
    score = torch.matmul(score, h_v)

    # score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
    score = rearrange(score, "b n s d -> b s (n d)").contiguous()
    return score

# @pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('dtype', [torch.float16])
# @pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('causal', [True])
# @pytest.mark.parametrize('d', [128, 64, 80, 40, 32, 16])
@pytest.mark.parametrize('d', [2])
# @pytest.mark.parametrize('seqlen', [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048])
@pytest.mark.parametrize('seqlen', [5])
# @pytest.mark.parametrize('dropout_p', [0.0, 0.17])
@pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_unpadded_qkvpacked(seqlen, d, dropout_p, causal, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    torch.random.manual_seed(1)
    batch_size = 512
    nheads = 127


    data = torch.randn(batch_size, seqlen, 3 * nheads * d, device=device, dtype=dtype).abs()
    qkv = rearrange(data, "b s (t h) -> b s t h",t=3)
    q = qkv[:, :, 0].detach().clone().requires_grad_(True)
    k = qkv[:, :, 1].detach().clone().requires_grad_(True)
    v = qkv[:, :, 2].detach().clone().requires_grad_(True)
    bias = torch.randn((1,nheads,seqlen,seqlen),dtype=dtype,device=device)
    q_f = rearrange(q, "b s (n h) -> b s n h", n=nheads).detach().clone().requires_grad_(True)
    k_f = rearrange(k, "b s (n h) -> b s n h", n=nheads).detach().clone().requires_grad_(True)
    v_f = rearrange(v, "b s (n h) -> b s n h", n=nheads).detach().clone().requires_grad_(True)
    mask = torch.arange(seqlen, device=device, dtype=torch.long).view(1, 1, 1, seqlen) >= torch.arange(seqlen, device=device, dtype=torch.long).view(1, 1, seqlen, 1)
    mask_ = mask.clone().detach().requires_grad_(False)
    bias_ = bias.masked_fill(mask==False, float('-inf'))
    mask = repeat(mask,"1 ... -> b ...",b=batch_size)
    # mask = torch.randint(0,2,(batch_size,seqlen,seqlen),dtype=torch.bool, device=device).view(batch_size,1,seqlen,seqlen)
    mask[:] = True
    import time
    start = time.time()
    output = flash_attn_func(
        q_f,k_f,v_f,mask, bias_,causal
    )
    end = time.time()
    print("flash time",end-start)
    start = time.time()
    # output = rearrange(output, "b s n d -> b s (n d)")
    # output_ref = attn_ref(q,k,v,bias,mask,nheads)
    output_ref = attention_ref(q,k,v,bias,mask_,num_heads=nheads)
    print(output_ref.shape)
    end = time.time()
    print("ref time",end-start)
    print(f'Output max diff: {(output - output_ref).abs().max().item()}')
    print(f'Output sum diff: {(output - output_ref).abs().mean().item()}')
    g = torch.randn_like(output)
    dq_f,dk_f,dv_f =  torch.autograd.grad(output, [q_f, k_f, v_f], g, retain_graph=True)
    dq,dk,dv  = torch.autograd.grad(output_ref, [q, k, v], g, retain_graph=True)
    # print(q.grad[0][5])
    # print(q_f.grad[0][5].flatten())
    print(f'Q max diff in a seq: {(dv[0][0] - dv_f[0][0].flatten()).abs().max().item()}')
    print(f'Q max diff in a batch: {(dq[1].flatten() - dq_f[1].flatten()).abs().max().item()}')
    print(f"q grad mean diff: {(dq.flatten() - dq_f.flatten()).abs().mean().item()}")