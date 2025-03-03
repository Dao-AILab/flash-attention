import os
import math
import itertools
import random

import pytest
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from padding import pad_input, unpad_input
from test_util import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_combine, flash_attn_with_kvcache, flex_flash_attn_func


def get_mask_from_ranges(q_ranges, k_ranges, q_len, k_len):
    bsz = q_ranges.shape[0]
    mask = torch.zeros((q_len, k_len), device='cuda', dtype=torch.bool)
    for i in range(bsz):
        mask[q_ranges[i, 0]:q_ranges[i, 1], k_ranges[i, 0]:k_ranges[i, 1]] = True
    return mask


def torch_attn_ref(q, k, v, mask, layout="thd", high_precision=True):
    if layout == "thd":
        q = rearrange(q, "t h d -> 1 h t d")
        k = rearrange(k, "t h d -> 1 h t d")
        v = rearrange(v, "t h d -> 1 h t d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        out = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64), attn_mask=mask)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if layout == "thd":
        out = rearrange(out, "1 h t d -> t h d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        out = out.to(q.dtype)
    return out


def generate_qk_ranges(seqlen_q, seqlen_k, bsz, device='cuda'):
    """生成q和k的ranges
    
    Args:
        seqlen: 序列长度
        bsz: batch size
        device: 设备,默认为'cuda'
        
    Returns:
        q_range: q的ranges张量,形状为(bsz, 2)
        k_range: k的ranges张量,形状为(bsz, 2)
    """
    
    random.seed(42)
    
    if bsz == 1:
        # bsz为1时直接使用完整序列
        q_ranges = [[0, seqlen_q]]
        max_seqlen_q = seqlen_q
        
        # 随机生成k_range
        start = random.randint(0, seqlen_k-1)
        end = random.randint(start+1, seqlen_k)
        k_ranges = [[start, end]]
        max_seqlen_k = end - start
        
    else:
        # 随机获取bsz-1个整数作为q的分割点
        points = sorted(random.sample(range(seqlen_q), bsz-1))
        
        max_seqlen_q = 0
        max_seqlen_k = 0

        # 构建q_range
        q_ranges = [[0, points[0]]]
        for i in range(bsz-2):
            q_ranges.append([points[i], points[i+1]])
        q_ranges.append([points[-1], seqlen_q])
        for q_range in q_ranges:
            max_seqlen_q = max(max_seqlen_q, q_range[1] - q_range[0])
        
        # 随机生成k_ranges
        k_ranges = []
        for i in range(bsz):
            start = random.randint(0, seqlen_k-1)
            end = random.randint(start+1, seqlen_k)
            k_ranges.append([start, end])
            max_seqlen_k = max(max_seqlen_k, end - start)
            
    q_ranges = torch.tensor(q_ranges, device=device, dtype=torch.int32)
    k_ranges = torch.tensor(k_ranges, device=device, dtype=torch.int32)

    print(f"DEBUG q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}", flush=True)
    
    return q_ranges, k_ranges, max_seqlen_q, max_seqlen_k

# @pytest.mark.skip(reason="skipped")
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen_q", [8, 256, 551, 1234, 1999]) # hang when seqlen is smaller than 7
@pytest.mark.parametrize("seqlen_k", [8, 256, 551, 1234]) # hang when seqlen is smaller than 7
@pytest.mark.parametrize("bsz", [1, 2])
def test_flex_flash_attn_output(
    seqlen_q, 
    seqlen_k, 
    bsz,
    d,
    mha_type, 
    dtype
):
    device = 'cuda'
    torch.random.manual_seed(42)

    q_ranges, k_ranges, max_seqlen_q, max_seqlen_k = generate_qk_ranges(seqlen_q * bsz, seqlen_k * bsz, bsz, device)

    # print(f"q_ranges: {q_ranges}, k_ranges: {k_ranges}, max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    q = torch.randn(bsz * seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz * seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(bsz * seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(bsz * seqlen_q, nheads, d, device=device, dtype=dtype)

    out, _ = flex_flash_attn_func(q, k, v, q_ranges, k_ranges, max_seqlen_q, max_seqlen_k, softmax_scale=None, deterministic=False)
    # out.backward(g)
    # dq, dk, dv = q.grad, k.grad, v.grad
    # q.grad, k.grad, v.grad = None, None, None

    out_ref = torch_attn_ref(q, k, v, mask=get_mask_from_ranges(q_ranges, k_ranges, seqlen_q * bsz, seqlen_k * bsz), layout="thd", high_precision=True)
    # out_ref.backward(g)
    # dq_ref, dk_ref, dv_ref = q.grad, k.grad, v.grad
    # q.grad, k.grad, v.grad = None, None, None
    
    out_ref_low_precision = torch_attn_ref(q, k, v, mask=get_mask_from_ranges(q_ranges, k_ranges, seqlen_q * bsz, seqlen_k * bsz), layout="thd", high_precision=False)
    # out_ref_low_precision.backward(g)
    # dq_ref_low_precision, dk_ref_low_precision, dv_ref_low_precision = q.grad, k.grad, v.grad
    # q.grad, k.grad, v.grad = None, None, None

    assert (out - out_ref_low_precision).abs().max().item() <= 2 * (out_ref_low_precision - out_ref).abs().max().item()
    # assert (dq - dq_ref_low_precision).abs().max().item() <= 2 * (dq_ref_low_precision - dq_ref).abs().max().item()

    # if d <= 128:
        # assert (dk - dk_ref_low_precision).abs().max().item() < 1e-4 or (dk - dk_ref_low_precision).abs().max().item() <= 3 * (dk_ref_low_precision - dk_ref).abs().max().item()
        # assert (dv - dv_ref_low_precision).abs().max().item() < 1e-4 or (dv - dv_ref_low_precision).abs().max().item() <= 3 * (dv_ref_low_precision - dv_ref).abs().max().item()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    elsit = []
    print("\n", flush=True)
    print(f"=========================START=========================", flush=True)
    try:
        torch.testing.assert_close(out, out_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    except Exception as e:
        print(f"---------------------------Start Out check---------------------------", flush=True)
        print(f"Failed out check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
        print(e, flush=True)
        print(f"---------------------------End Out check---------------------------", flush=True)
        elsit.append(e)
    # try:
    #     torch.testing.assert_close(dq, dq_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    # except Exception as e:
    #     print(f"---------------------------Start dq check---------------------------", flush=True)
    #     print(f"Failed dq check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
    #     print(e, flush=True)
    #     print(f"---------------------------End dq check---------------------------", flush=True)
    #     elsit.append(e)
    # try:
    #     torch.testing.assert_close(dk, dk_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    # except Exception as e:
    #     print(f"---------------------------Start dk check---------------------------", flush=True)
    #     print(f"Failed dk check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
    #     print(e, flush=True)
    #     print(f"---------------------------End dk check---------------------------", flush=True)
    #     elsit.append(e)
    # try:
    #     torch.testing.assert_close(dv, dv_ref, atol=torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps)
    # except Exception as e:
    #     print(f"---------------------------Start dv check---------------------------", flush=True)
    #     print(f"Failed dv check for mha_type: {mha_type}, dtype: {dtype}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, bsz: {bsz}", flush=True)
    #     print(e, flush=True)
    #     print(f"---------------------------End dv check---------------------------", flush=True)
    #     elsit.append(e)
    # print(f"=========================END=========================", flush=True)

    # for e in elsit:
    #     raise e