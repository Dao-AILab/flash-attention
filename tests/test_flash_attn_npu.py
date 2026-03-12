import sys
import os
import torch
import torch_npu
import pytest
import flash_attn_2_cuda
from flash_attn import flash_attn_with_kvcache

def group_matmul(head, kv_head, left, right, high_prec = 1):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        if high_prec == 0:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32)).to(torch.float32)
        else:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score

def softmax1( 
    qk_result,
    is_first,
    gm,
    interm_dtype = torch.float16
    ):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm

def qkMM1( 
    query,
    key
    ):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def pvMM2( 
    p,
    value
    ):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :] 
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def ref_flash_attention( 
    query,
    key,
    value,
    scale,
    mask,
    data_type,
    ):
    inner_prec = 0
    interm_dtype = torch.float16 if inner_prec == 1 else torch.float32
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale)
    scale = scale.to(torch.float16) if inner_prec == 1 else scale.to(torch.float32)
    context_len = key.shape[2]
    context_size = 512
    group_num = query.shape[0] // key.shape[0]
    gl = None
    gl_high = None
    go = None
    go_high = None
    if mask is not None:
        mask = mask.cpu()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            sub_mask = mask[:query.shape[1], kv_start : kv_start + sub_len].to(interm_dtype) * (-1e4)
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key).to(interm_dtype)
        qk_result = qk_result * scale
        if mask is not None:
            qk_result += sub_mask
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.to(data_type)
        if kv_start == 0:
            gm_high = None
        lo = pvMM2(p_result, sub_value).to(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm
            gl = gl + row_sum
            go = go * dm
            go = go + lo
    go = go / gl
    go = go.permute(1, 0, 2)
    lse = torch.squeeze((torch.log(gl) + gm), dim=-1).to(torch.float32)
    return go.to(data_type), lse

test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal)
    (torch.bfloat16, 1, 1, 1, 1024, 1024, 128, 1, 128, False),
    (torch.bfloat16, 5, 4, 4, 1024, 1024, 128, 0, 128, True),
    (torch.float16, 7, 1, 1, 512, 512, 128, 1, 128, False),
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal", test_cases)
def test_fa_custom_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, cache_mode, block_size, is_causal):
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    block_size = 128
    num_blocks = 64
    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size, q_seqlen, num_heads, head_size)).to(data_type).npu()
    key_cache = None
    value_cache = None
    block_tables = []
    if cache_mode == 1:
        key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)).to(data_type).npu()
        value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(num_blocks, block_size, kv_heads, head_size)).to(data_type).npu()
        max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
        for i in range(batch_size):
            block_table = [
                max_num_blocks_per_seq * i + j
                for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int32).npu()
    else:
        key_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
        value_cache = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size, kv_seqlen, kv_heads, head_size)).to(data_type).npu()
        block_tables = None
    kv_seqlen_list = [kv_seqlen] * batch_size
    scale = 1.0 / (head_size ** 0.5)
    window_size_left = -1
    window_size_right = -1
    is_rotary_interleaved = False
    softcap = 0
    num_splits = 0
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int64).cpu()
    rotary_cos = None
    rotary_sin = None
    cache_batch_idx = None
    leftpad_k = None
    alibi_slopes = None
    out_out, softmax_lse = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=kv_seqlen_list,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        block_table=block_tables,
        causal=is_causal,
        window_size=[window_size_left, window_size_right],
        rotary_interleaved=is_rotary_interleaved,
        alibi_slopes=alibi_slopes,
        num_splits=num_splits,
        return_softmax_lse=True
    )
    golden_out = torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=data_type)
    golden_lseL = torch.empty((batch_size, q_seqlen, num_heads), dtype=torch.float32)
    atten_mask = None
    if is_causal:
        atten_mask = torch.triu(torch.ones(q_seqlen, kv_seqlen), diagonal=1).bool()
    for i in range(batch_size):
        key_cache_per_batch = None
        value_cache_per_batch = None
        if cache_mode == 1:
            keys = []
            values = []
            block_table = block_tables.cpu()[i]
            for j in range(kv_seqlen):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = key_cache.detach().cpu()[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)
                v = value_cache.detach().cpu()[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size)
                values.append(v)
            key_cache_per_batch = torch.stack(keys, dim=0)
            value_cache_per_batch = torch.stack(values, dim=0)
        else:
            key_cache_per_batch = key_cache.detach().cpu()[i]
            value_cache_per_batch = value_cache.detach().cpu()[i]
        query_cpu = query.detach().cpu()[i]
        if is_causal:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, scale, atten_mask, data_type)
        else:
            output, golden_lse = ref_flash_attention(query_cpu, key_cache_per_batch, value_cache_per_batch, scale, None, data_type)
        out = output.reshape(q_seqlen, num_heads, head_size)
        golden_out[i:i+1] = out
        golden_lseL[i:i+1] = torch.transpose(golden_lse.reshape(num_heads, q_seqlen), 0, 1)
    rtol = 1e-2
    atol = 1e-2
    torch.testing.assert_close(out_out.cpu(), golden_out.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(softmax_lse.cpu(), golden_lseL.cpu(), rtol=rtol, atol=atol)