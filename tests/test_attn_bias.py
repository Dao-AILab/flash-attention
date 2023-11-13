import pytest
import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_func

#torch.set_printoptions(precision=2, threshold=100000)
torch.manual_seed(0)

## Follow flash attention test

def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
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
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    bias=None,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
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
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
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
        if bias is not None:
            bias = bias.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))

    if bias is not None:
        scores += bias

    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
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
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

def get_tensors(batch_size, seq_len, num_heads, head_dim, dtype):
    q, k, v = [torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda",
                           dtype=dtype, requires_grad=True) for _ in range(3)]

    bias = torch.randn(batch_size, num_heads, seq_len, seq_len, device="cuda",
                           dtype=dtype, requires_grad=True)

    attention_mask = generate_random_padding_mask(seq_len, batch_size, device=q.device)
    return q, k, v, attention_mask, bias


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize(
    "bs_seqlen", [(8, 128), (8,256), (8, 512), (4, 1024), (2, 2048),
                  (1, 4096), (5, 127), (5, 257), (5, 513)]
)
@pytest.mark.parametrize(
    "nh_headdim", [(16,32), (16, 64), (16, 96),
                    (16, 128), (16, 160), (8, 192),
                    (8, 224), (8, 256)]
)
@pytest.mark.parametrize(
    "causal", [True, False]
)
def test_bias_attention(bs_seqlen, nh_headdim, dtype, causal):

    q, k, v, _, bias = get_tensors(bs_seqlen[0], bs_seqlen[1], nh_headdim[0], nh_headdim[1], dtype)
    dout = torch.rand_like(q)

    out_ref, attn_ref = attention_ref(q, k, v,
        bias=bias,
        query_padding_mask=None,
        key_padding_mask=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=causal
    )

    out_pt, attn_pt = attention_ref(q, k, v,
        bias=bias,
        query_padding_mask=None,
        key_padding_mask=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=causal,
        upcast=False,
        reorder_ops=True
    )

    out_fl = flash_attn_func(q, k, v,
        dropout_p=0.,
        causal=causal,
        attn_bias=bias
        )

    # Compute gradients
    (dq, dk, dv, dbias) = torch.autograd.grad(out_fl, (q, k, v, bias), dout)
    (dq_ref, dk_ref, dv_ref, dbias_ref) = torch.autograd.grad(out_ref, (q, k, v, bias), dout)
    (dq_pt, dk_pt, dv_pt, dbias_pt) = torch.autograd.grad(out_pt, (q, k, v, bias), dout)

    assert (out_fl - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()
    assert (dbias - dbias_ref).abs().max().item() <= 2 * (dbias_pt - dbias_ref).abs().max().item()
