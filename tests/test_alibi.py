import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from flash_attn import (flash_attn_func, flash_attn_kvpacked_func,
                        flash_attn_qkvpacked_func, flash_attn_varlen_func,
                        flash_attn_varlen_kvpacked_func,
                        flash_attn_varlen_qkvpacked_func,
                        flash_attn_with_kvcache)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import _get_block_size
from flash_attn.flash_attn_triton import \
    flash_attn_func as flash_attn_func_triton
from flash_attn.layers.rotary import apply_rotary_emb

MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def generate_alibi(max_seq_len, num_attention_heads, tp_world_size, tp_index, key_padding_mask=None, device="cuda"):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                :n - closest_power_of_2]

    slopes = torch.tensor(get_slopes(num_attention_heads)).to(device=device)
    # Select the part of the tensor that corresponds to our tensor parallel index.
    assert (num_attention_heads/tp_world_size).is_integer(
    ), "it works only when (num_attention_heads/tp_world_size) is integer"
    nh_tp = num_attention_heads // tp_world_size
    slopes = slopes[nh_tp * tp_index:nh_tp * (tp_index + 1)]

    if (key_padding_mask is None):
        arange_tensor = rearrange(torch.arange(max_seq_len), "sqk -> 1 sqk").to(device=device)
    else:
        arange_tensor = (key_padding_mask.cumsum(dim=-1, dtype=slopes.dtype) - 1) \
            .masked_fill_(~key_padding_mask, torch.finfo(torch.float).min).to(device=device)
        
    arange_tensor = rearrange(arange_tensor, 'b sqk -> b 1 1 sqk')
    # (1, nheads, 1, seqlen_k) or (batch, nheads, 1, seqlen_k)
    alibi_tensor = rearrange(slopes, 'nh -> 1 nh 1 1') * arange_tensor

    return alibi_tensor, slopes
    

def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random", right_padding=True):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen,
                             device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(
            max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    if right_padding:
        padding_mask = (
            repeat(torch.arange(max_seqlen, device=device),
                "s -> b s", b=batch_size) < lengths
        )
    else:
        padding_mask = (
            repeat(torch.arange(start=max_seqlen-1, end=-1, step=-1, device=device),
                "s -> b s", b=batch_size) < lengths
        )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
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
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            q, query_padding_mask)

        def output_pad_fn(output_unpad): return pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q

        def output_pad_fn(output_unpad): return rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
            k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            def dqkv_pad_fn(dqkv_unpad): return pad_input(
                dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            def dqkv_pad_fn(dqkv_unpad): return rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            def dkv_pad_fn(dkv_unpad): return pad_input(
                dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            def dkv_pad_fn(dkv_unpad): return rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            def dk_pad_fn(dk_unpad): return pad_input(
                dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            def dk_pad_fn(dk_unpad): return rearrange(
                dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(
        seqlen_q, device=device, dtype=torch.long), "s -> s 1")
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
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(
            col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
    bias=None
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if bias is not None:
        bias = bias.to(scores.dtype)
        scores += bias
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask,
                            "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum(
        "bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(
            rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(
    q,
    kv,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        query_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def generate_sparsity_mask(seqlen, sparsity=0.3):
    repeats = seqlen // 16 // 2
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([0, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    nrow, ncol = seqlen // 16, seqlen // 256
    mask = torch.rand(nrow, ncol, device="cuda") < sparsity
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
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    blockmask = repeat(blockmask, "s_16 s_256 -> (s_16 16) (s_256 256)")
    blockmask = blockmask[:seqlen, :seqlen]
    scores.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(
        rearrange(~attn_mask, "b s -> b 1 s 1"), 0.0)
    attention = attention.masked_fill_(
        rearrange(~blockmask, "t s -> 1 1 t s"), 0.0)
    attention_drop = attention.masked_fill(
        ~dropout_mask, 0.0) / (1 - dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    output.masked_fill_(rearrange(~attn_mask, "b s -> b s 1 1"), 0)
    return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(
        S.device, head_dim, is_dropout, causal)
    nblocks_n = (seqlen_k_rounded + blocksize_n - 1) // blocksize_n
    nblocks_m = (seqlen_q_rounded + blocksize_m - 1) // blocksize_m
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(
        S,
        "b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)",
        blocksize_m=blocksize_m,
        blocksize_n=blocksize_n,
    )
    S_converted = rearrange(
        S_flat,
        "b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)",
        mmas_n=mmas_n,
        warps_n=warps_n,
        eight=8,
        c0=2,
        c1=2,
        c2=2,
        four=4,
    )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted.masked_fill_(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(
            query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(
            key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask,
                            "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    _, block_size_n = _get_block_size(
        scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1)
                            for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack(
        [torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(
        scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()


@pytest.mark.parametrize(
    "dtype", [torch.float16]
)
@pytest.mark.parametrize(
    "b_sq", 
    [
        (32, 512), 
        (16, 1024), 
        (8, 2048),
        (4, 4096), 
        (2, 8192), 
        (1, 16384)
    ]
)
@pytest.mark.parametrize(
    "nh_hd", 
    [
        (32, 64),
        (16, 128),
        (40, 128) # non power of 2 nh
    ]
)
@pytest.mark.parametrize(
    "tp_world_size", [1, 2, 4]
)
def test_flash_attn_func(b_sq, nh_hd, tp_world_size, dtype):
    b, sq = b_sq
    nh, hd = nh_hd
    nh_tp = nh // tp_world_size
    q, k, v = [torch.randn(b, sq, nh_tp, hd, device="cuda", 
                           dtype=dtype, requires_grad=True) for _ in range(3)]
    dout = torch.rand_like(q)

    for tp_index in range(tp_world_size):
        alibi, alibi_slopes = generate_alibi(
            max_seq_len=sq,
            num_attention_heads=nh,
            tp_world_size=tp_world_size,
            tp_index=tp_index,
            key_padding_mask=None,
            device="cuda"
        )

        triton_out = flash_attn_func_triton(
            q, k, v, alibi, True, hd**(-0.5))
        triton_out.backward(dout)
        triton_dq, q.grad = q.grad.clone(), None
        triton_dk, k.grad = k.grad.clone(), None
        triton_dv, v.grad = v.grad.clone(), None

        flash_out = flash_attn_func(q, k, v, causal=True, alibi_slopes=repeat(alibi_slopes, "nh -> b nh", b=b))
        flash_out.backward(dout)
        flash_dq, q.grad = q.grad.clone(), None
        flash_dk, k.grad = k.grad.clone(), None
        flash_dv, v.grad = v.grad.clone(), None

        assert torch.allclose(flash_out, triton_out, atol=1e-2, rtol=0.)
        assert torch.allclose(flash_dq, triton_dq, atol=1e-2, rtol=0.)
        assert torch.allclose(flash_dk, triton_dk, atol=1e-2, rtol=0.)
        assert torch.allclose(flash_dv, triton_dv, atol=1e-2, rtol=0.)


@pytest.mark.parametrize(
    "dtype", [torch.float16]
)
@pytest.mark.parametrize(
    "right_padding", [True, False]
)
@pytest.mark.parametrize(
    "b_sq", 
    [
        (32, 512), 
        (16, 1024), 
        (8, 2048),
        (4, 4096), 
        (2, 8192), 
        (1, 16384)
    ]
)
@pytest.mark.parametrize(
    "nh_hd", 
    [
        (32, 64),
        (16, 128),
        (40, 128) # non power of 2 nh
    ]
)
@pytest.mark.parametrize(
    "tp_world_size", [1, 2, 4]
)
def test_flash_attn_varlen_func(b_sq, nh_hd, tp_world_size, right_padding, dtype):
    b, sqk = b_sq
    nh, hd = nh_hd
    nh_tp = nh // tp_world_size
    # flash_attn_func_triton(), flash-attention v2 (above v2.1) causal logic are different
    # so only (seqlen_q == 1, causal=False to triton ver.) shows correct results
    # https://github.com/huggingface/text-generation-inference/blob/v1.1.1/server/text_generation_server/models/custom_modeling/mpt_modeling.py#L53-L63
    q = torch.randn(b, 1, nh_tp, hd, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [torch.randn(b, sqk, nh_tp, hd, device="cuda",
                        dtype=dtype, requires_grad=True) for _ in range(2)]
    dout = torch.rand_like(q)

    padding_mask = generate_random_padding_mask(sqk, b, "cuda", "random", right_padding)
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, None, padding_mask, kvpacked=False)

    for tp_index in range(tp_world_size):
        alibi, alibi_slopes = generate_alibi(
            max_seq_len=sqk,
            num_attention_heads=nh,
            tp_world_size=tp_world_size,
            tp_index=tp_index,
            key_padding_mask=padding_mask,
            device="cuda"
        )

        triton_out = flash_attn_func_triton(
            q, k, v, alibi, False, hd**(-0.5))
        triton_out.backward(dout)
        triton_dq, q.grad = q.grad.clone(), None
        triton_dk, k.grad = k.grad.clone(), None
        triton_dv, v.grad = v.grad.clone(), None

        flash_out_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            alibi_slopes=repeat(alibi_slopes, "nh -> b nh", b=b)
        )
        flash_out = output_pad_fn(flash_out_unpad)
        flash_out.backward(dout)
        flash_dq_unpad, q_unpad.grad = q_unpad.grad.clone(), None
        flash_dk_unpad, k_unpad.grad = k_unpad.grad.clone(), None
        flash_dv_unpad, v_unpad.grad = v_unpad.grad.clone(), None
        flash_dq = dq_pad_fn(flash_dq_unpad)
        flash_dk = dk_pad_fn(flash_dk_unpad)
        flash_dv = dk_pad_fn(flash_dv_unpad)

        assert torch.allclose(flash_out, triton_out, atol=1e-2, rtol=0.)
        assert torch.allclose(flash_dq, triton_dq, atol=1e-2, rtol=0.)
        assert torch.allclose(flash_dk, triton_dk, atol=1e-2, rtol=0.)
        assert torch.allclose(flash_dv, triton_dv, atol=1e-2, rtol=0.)


@pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("num_splits", [1, 0])
# @pytest.mark.parametrize("num_splits", [0])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("new_kv", [False, True])
# @pytest.mark.parametrize("new_kv", [True])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
@pytest.mark.parametrize("rotary_interleaved", [False, True])
# @pytest.mark.parametrize("rotary_interleaved", [False])
@pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0])
# @pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize("has_batch_idx", [False, True])
# @pytest.mark.parametrize("has_batch_idx", [True])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (3, 799),
        (64, 2048),
        (16, 20000),
        (1, 128 * 1024),
        (16, 128 * 1024),
        (128, 128),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    mha_type,
    num_splits,
    dtype,
    alibi,
):
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 8
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 4)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads,
                    d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(
        1, seqlen_q + 1, (1,)).item()
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k,
                        d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k,
                        d, device=device, dtype=dtype)
    else:
        k, v = None, None
    k_cache = torch.randn(batch_size_cache, seqlen_k,
                          nheads_k, d, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size_cache, seqlen_k,
                          nheads_k, d, device=device, dtype=dtype)
    cache_seqlens = torch.randint(
        0,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        (seqlen_k - (seqlen_q if (causal or local)
         and rotary_dim > 1 else seqlen_new) + 1)
        if new_kv
        else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    if has_batch_idx:
        cache_batch_idx = torch.randperm(
            batch_size_cache, dtype=torch.int32, device=device)[:batch_size]
    else:
        cache_batch_idx = None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    if rotary_dim > 0:
        angle = torch.rand(seqlen_k, rotary_dim // 2,
                           device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=dtype)
        sin = torch.sin(angle).to(dtype=dtype)
        if causal or local:
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            q_ro = rearrange(
                apply_rotary_emb(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=seqlen_q,
            )
        # q_ro = q
        k_ro = apply_rotary_emb(
            k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (
        k_cache if not has_batch_idx else k_cache[cache_batch_idx]).clone()
    v_cache_ref = (
        v_cache if not has_batch_idx else v_cache[cache_batch_idx]).clone()
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(
        k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(
        v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    if alibi:
        seqlen_alibi = k_cache_rep.shape[1]
        alibi_tensor, alibi_slopes = generate_alibi(
            max_seq_len=seqlen_alibi,
            num_attention_heads=nheads,
            tp_world_size=1,
            tp_index=0,
            key_padding_mask=None,
            device="cuda"
        )
        # alibi_tensor = alibi_tensor.expand(batch_size, -1, seqlen_q, -1)
        alibi_slopes = repeat(alibi_slopes, "nh -> b nh", b=batch_size)
        if alibi_tensor.abs().max().item() >= torch.finfo(dtype).max:
            pytest.skip()
    else:
        alibi_tensor, alibi_slopes = None, None
    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cos,
        sin,
        cache_seqlens,
        cache_batch_idx,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
        alibi_slopes=alibi_slopes
    )
    # out = flash_attn_with_kvcache(
    #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
    # )
    # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
    # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # probs = torch.softmax(qk, dim=-1)
    key_padding_mask = arange < cache_seqlens_expanded + \
        (seqlen_new if new_kv else 0)
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        bias=alibi_tensor
    )
    out_pt, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
        bias=alibi_tensor
    )
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        k_cache_select = k_cache if not has_batch_idx else k_cache[cache_batch_idx]
        v_cache_select = v_cache if not has_batch_idx else v_cache[cache_batch_idx]
        assert torch.allclose(k_cache_select, k_cache_ref,
                              rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    assert (out - out_ref).abs().max().item() <= 3 * \
        (out_pt - out_ref).abs().max().item() + 1e-5
