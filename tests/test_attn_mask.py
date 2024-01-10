import pytest
import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_func
import test_flash_attn

torch.manual_seed(0)


def attention_ref(
    q,
    k,
    v,
    mask=None,
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
        mask: (batch_size, seqlen_k, nheads_k, head_dim)
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
        if mask is not None:
            mask = mask.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))

    if mask is not None:
        scores.masked_fill_(mask < 0.5, float("-inf"))

    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = test_flash_attn.construct_local_mask(
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
    # If we mask a full seqlen_k for a q we get NaNs. Set them to 0.
    if mask is not None:
        if not causal:
            attention = attention.masked_fill(torch.all(mask < 0.5, dim=-1, keepdim=True), 0.0)
        else:
            causal_mask = torch.logical_not(torch.tril(torch.ones(mask.shape))).cuda()
            attention = attention.masked_fill(torch.all(torch.logical_or(mask<0.5, causal_mask), dim=-1, keepdim=True), 0.0)
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


def get_tensors(batch_size, seq_len, num_heads, num_heads_k,head_dim, dtype):
    q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
    k, v = [
        torch.randn(
            batch_size, seq_len, num_heads_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        for _ in range(2)
    ]

    mask = (
        (torch.randn(batch_size, num_heads, seq_len, seq_len, requires_grad=False) > 0.0)
        .type(dtype)
        .cuda()
    )
    mask.requires_grad = False

    attention_mask = test_flash_attn.generate_random_padding_mask(
        seq_len, batch_size, device=q.device
    )
    return q, k, v, attention_mask, mask


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "seqlen",
    [1024, 2048],
)
@pytest.mark.parametrize(
    "num_heads",
    [32, 64],
)
@pytest.mark.parametrize("gqa_factor", [1, 4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_mask_attention(batch_size, seqlen, num_heads, gqa_factor, head_dim, dtype, causal):
    #  TODO(cyprien): add test for varlen
    num_heads_k = num_heads // gqa_factor
    q, k, v, _, mask = get_tensors(batch_size, seqlen, num_heads, num_heads_k, head_dim, dtype)
    dout = torch.rand_like(q)

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        mask=mask,
        query_padding_mask=None,
        key_padding_mask=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=causal,
    )

    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        mask=mask,
        query_padding_mask=None,
        key_padding_mask=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    out_fl = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal, attn_mask=mask)

    # Compute gradients
    (dq, dk, dv) = torch.autograd.grad(out_fl, (q, k, v), dout)
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(out_ref, (q, k, v), dout)
    (dq_pt, dk_pt, dv_pt) = torch.autograd.grad(out_pt, (q, k, v), dout)

    assert (out_fl - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()
