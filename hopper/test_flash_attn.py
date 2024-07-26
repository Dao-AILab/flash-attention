import math

import pytest
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from tests.test_util import generate_random_padding_mask, generate_qkv, construct_local_mask, attention_ref

ABS_TOL = 5e-3
REL_TOL = 1e-1

def print_diffs(out, out_ref):
    out_1d = out.flatten()
    out_ref_1d = out_ref.flatten()
    for idx, (e_o, e_o_ref) in enumerate(zip(out_1d, out_ref_1d)):
        diff = e_o - e_o_ref
        abs_diff = abs(diff)
        abs_ref = abs(e_o_ref + 1e-5)
        relative_diff = abs_diff / abs_ref
        if abs_diff > ABS_TOL or relative_diff > REL_TOL:
            print(f"==== diff ==== {idx}, test: {e_o}, ref: {e_o_ref}")


<<<<<<< HEAD
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype_to", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["gqa"])
@pytest.mark.parametrize("causal", [False, True])
=======
def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads, head_dim)
        v: (batch_size, seqlen_k, nheads, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
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
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if causal:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            (-1, 0),
            None,
            None,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if causal:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
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



# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype_to", [torch.float8_e4m3fn])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("causal", [False])
>>>>>>> 17dd4b7e5ebe53f433298c923fd1d8200bfc94a5
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("d", [64, 128, 256])
<<<<<<< HEAD
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (257, 1),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
=======
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 1)
        # (64, 128),
        # (128, 128),
        # (256, 256),
        # (113, 203),
        # (128, 217),
        # (113, 211),
        # (108, 256),
        # (256, 512),
        # (384, 256),
        # (640, 128),
        # (512, 256),
        # (1024, 1024),
        # (1023, 1024),
        # (1024, 1023),
        (4096, 4096),
>>>>>>> 17dd4b7e5ebe53f433298c923fd1d8200bfc94a5
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
<<<<<<< HEAD
    seqlen_q, seqlen_k, d, causal, mha_type,
    dtype,
    # dtype_to
=======
    seqlen_q, seqlen_k, d, causal, mha_type, dtype_to
>>>>>>> 17dd4b7e5ebe53f433298c923fd1d8200bfc94a5
):
    device = "cuda"
    dtype = torch.float16
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 4
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    # nheads_kv = 2
    # batch_size = 9
    # nheads = 6
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True)

<<<<<<< HEAD
    # q = q.to(dtype_to)
    # k = k.to(dtype_to)
    # v = v.to(dtype_to)
=======
    q = q.to(dtype_to)
    k = k.to(dtype_to)
    v = v.to(dtype_to)
>>>>>>> 17dd4b7e5ebe53f433298c923fd1d8200bfc94a5

    print(q.dtype)

    out, lse = flash_attn_func(q, k, v, causal=causal)

<<<<<<< HEAD
    # q = q.to(dtype)
    # k = k.to(dtype)
    # v = v.to(dtype)
=======
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
>>>>>>> 17dd4b7e5ebe53f433298c923fd1d8200bfc94a5
    
    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        None,
        None,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    # qk = torch.einsum('bshd,bthd->bhst', q, k).float()
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # exp_sum = s_tmp.sum(-1)
    # qk = torch.einsum('bthd,bshd->bhts', q.float() / math.sqrt(d), k.float())
    # lse_ref = torch.logsumexp(qk, dim=-1)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    # if d <= 128:
    #     g = torch.randn_like(out)
    #     do_o = (g.float() * out.float()).sum(-1)
    #     dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    #     dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    #     dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
    #     print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    #     print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    #     print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    #     print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    #     print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    #     print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
    #     print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    #     print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    #     print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    #     print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
    #     print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
    #     print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
    # P = torch.softmax(qk, -1)
    # dP = P * (dS - do_o.unsqueeze(1))
    # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
    # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
    # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())
    # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
<<<<<<< HEAD
    # breakpoint()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
=======
    # assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
>>>>>>> 17dd4b7e5ebe53f433298c923fd1d8200bfc94a5

    # if d <= 128:
    #     assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()
    #     assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()
    #     assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize('causal', [True])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (1, 3),
        (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (113, 203),
        (128, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (512, 256),
        (640, 128),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, causal, mha_type, dtype
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 1
    # nheads = 1
    batch_size = 9
    nheads = 6
    nheads_kv = 6 if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
 
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')

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
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    # print("cu_seqlens_q: ", cu_seqlens_q)
    # print("cu_seqlens_k: ", cu_seqlens_k)
    # print("q_unpad, shape: ", q_unpad.shape)
    # print("k_unpad, shape: ", k_unpad.shape)
    # print("v_unpad, shape: ", v_unpad.shape)
    out_unpad, sm_lse = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
    )
    out = output_pad_fn(out_unpad)
    dropout_mask = None

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
    )
    out_pt, attn_pt = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # g = torch.randn_like(out)
    # if d <= 128:
    #     (
    #         dq_unpad,
    #         dk_unpad,
    #         dv_unpad,
    #     ) = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), g)
    #     dk = dk_pad_fn(dk_unpad)
    #     dv = dk_pad_fn(dv_unpad)
    #     (
    #         dq_ref,
    #         dk_ref,
    #         dv_ref,
    #     ) = torch.autograd.grad(out_ref, (q, k, v), g)
    #     (
    #         dq_pt,
    #         dk_pt,
    #         dv_pt,
    #     ) = torch.autograd.grad(out_pt, (q, k, v), g)
    #     dq = dq_pad_fn(dq_unpad)
    #     print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    #     print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    #     print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    #     print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    #     print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    #     print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
    #     print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    #     print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    #     print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
    #     print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
    #     print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
    #     print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    # if d <= 128:
    #     assert (dq - dq_ref).abs().max().item() < 1e-4 or (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
    #     assert (dk - dk_ref).abs().max().item() < 1e-4 or (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
    #     assert (dk - dk_ref).abs().max().item() < 1e-4 or (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()
