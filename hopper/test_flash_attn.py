import math

import pytest
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from flash_attn.bert_padding import pad_input, unpad_input

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

ABS_TOL = 5e-3
REL_TOL = 1e-1


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
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *rest = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, *rest = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _, *rest = unpad_input(v, key_padding_mask)
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
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
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
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
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
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
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
    q_scale=None, k_scale=None, v_scale=None,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
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
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    if q_scale is not None:
        q = (q.float() * q_scale).to(dtype=q.dtype)
    if k_scale is not None:
        k = (k.float() * k_scale).to(dtype=k.dtype)
    if v_scale is not None:
        v = (v.float() * v_scale).to(dtype=v.dtype)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
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
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)



@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0, 50.0])
# @pytest.mark.parametrize("softcap", [50.0])
@pytest.mark.parametrize("causal,local", [(False, False), (True, False), (False, True)])
# @pytest.mark.parametrize("causal,local", [(False, False)])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("V_colmajor", [False, True])
# @pytest.mark.parametrize("V_colmajor", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128, 192])
@pytest.mark.parametrize("d", [64, 96, 128, 192, 256])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 128),
        (128, 192),
        (256, 256),
        (113, 203),
        (113, 128),
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
        (8192, 8192),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
        seqlen_q, seqlen_k, d, causal, local, softcap, V_colmajor, deterministic, mha_type, dtype
):
    if V_colmajor and (seqlen_k % 16 != 0 or dtype != torch.float8_e4m3fn):
        pytest.skip("V_colmajor requires seqlen_k to be a multiple of 16 and dtype to be float8_e4m3fn")
    if softcap > 0.0 and dtype == torch.float8_e4m3fn:
        pytest.skip("Softcap is not supported for float8_e4m3fn")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_k <= 2048 else 2
    nheads = 6
    # batch_size = 1
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    if softcap > 0.0:
        # Ensure the values of qk are at least within softcap range.
        q_ref = (q_ref * softcap / 2).detach().requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    # Put window_size after QKV randn so that window_size changes from test to test
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    if dtype == torch.float8_e4m3fn:
        q_scale, k_scale, v_scale = [torch.rand(1, device=device, dtype=torch.float32) * 2 for _ in range(3)]
    else:
        q_scale, k_scale, v_scale = None, None, None
    q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
    if V_colmajor:
        v = rearrange(rearrange(v.detach(), "b s h d -> b h d s").contiguous(), "b h d s -> b s h d").requires_grad_()
    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
        window_size=window_size,
        softcap=softcap,
    )
    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        None,
        None,
        causal=causal,
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
        window_size=window_size,
        softcap=softcap
    )
    out_pt, attn_pt = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        None,
        None,
        causal=causal,
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
        window_size=window_size,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
        intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
    )

    # qk = torch.einsum('bshd,bthd->bhst', q_ref, k_ref).float()
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # exp_sum = s_tmp.sum(-1)
    # qk = torch.einsum('bthd,bshd->bhts', q_ref.float() / math.sqrt(d), k_ref.float())
    # lse_ref = torch.logsumexp(qk, dim=-1)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()


    if dtype != torch.float8_e4m3fn and not V_colmajor:
        g = torch.randn_like(out)
        do_o = ((g.float() * out.float()).sum(-1)).transpose(1, 2)
        import flashattn_hopper_cuda
        dq, dk, dv, softmax_d, dq_accum, dk_accum, dv_accum = flashattn_hopper_cuda.bwd(
            g,
            q,
            k,
            v,
            out,
            lse,
            None,
            None,
            None,
            d ** (-0.5),
            causal,
            window_size[0], window_size[1],
            softcap,
            deterministic,
        )
        # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
        # assert (softmax_d - do_o).abs().max().item() <= 1e-5
        # assert dq_accum.abs().max().item() == 0.0

        # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
        # P = torch.softmax(qk, -1)
        # dP = P * (dS - do_o.transpose(1, 2).unsqueeze(1))
        # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
        # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
        # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())

        # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
        # breakpoint()


    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    # multiple = 2 if dtype != torch.float8_e4m3fn else 3
    multiple = 2
    assert (out - out_ref).abs().max().item() <= multiple * (out_pt - out_ref).abs().max().item()

    if dtype != torch.float8_e4m3fn and not V_colmajor:
        multiple = 2 if softcap == 0.0 else 4
        assert (dq - dq_ref).abs().max().item() <= multiple * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= multiple * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= multiple * (dv_pt - dv_ref).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0, 50.0])
# @pytest.mark.parametrize("softcap", [50.0])
@pytest.mark.parametrize("causal,local", [(False, False), (True, False), (False, True)])
# @pytest.mark.parametrize("causal,local", [(False, False)])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128])
@pytest.mark.parametrize("d", [64, 96, 128, 192, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
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
        (8192, 8192),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_varlen_output(
        seqlen_q, seqlen_k, d, causal, local, softcap, deterministic, mha_type, dtype
):
    if softcap > 0.0 and dtype == torch.float8_e4m3fn:
        pytest.skip("Softcap is not supported for float8_e4m3fn")
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal))
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_q <= 2048 else 1
    nheads = 6
    # batch_size = 2
    # nheads = 2
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    if softcap > 0.0:
        # Ensure the values of qk are at least within softcap range.
        q_ref = (q_ref * softcap / 2).detach().requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    # Put window_size after QKV randn so that window_size changes from test to test
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    if dtype == torch.float8_e4m3fn:
        q_scale, k_scale, v_scale = [torch.rand(1, device=device, dtype=torch.float32) * 2 for _ in range(3)]
    else:
        q_scale, k_scale, v_scale = None, None, None
    q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
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
    q_unpad, k_unpad, v_unpad = [x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)]
    out_unpad, lse = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        q_scale=q_scale,
        k_scale=k_scale, v_scale=v_scale,
        window_size=window_size,
        softcap=softcap,
    )
    out = output_pad_fn(out_unpad)
    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
        window_size=window_size,
        softcap=softcap
    )
    out_pt, attn_pt = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
        window_size=window_size,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
        intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    if dtype != torch.float8_e4m3fn:
        g_unpad = torch.randn_like(out_unpad)
        do_o = ((g_unpad.float() * out_unpad.float()).sum(-1)).transpose(-1, -2)
        import flashattn_hopper_cuda
        dq_unpad, dk_unpad, dv_unpad, softmax_d, dq_accum, lse_log2 = flashattn_hopper_cuda.bwd_varlen(
            g_unpad,
            q_unpad,
            k_unpad,
            v_unpad,
            out_unpad,
            lse,
            None,
            None,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            None, None,
            max_seqlen_q,
            max_seqlen_k,
            d ** (-0.5),
            causal,
            window_size[0], window_size[1],
            softcap,
            deterministic,
        )
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
        # assert (softmax_d - do_o).abs().max().item() <= 1e-5
        # assert dq_accum.abs().max().item() == 0.0
        g = output_pad_fn(g_unpad)

        # qk = torch.einsum('bthd,bshd->bhts', q / (d ** 0.5), k).float()
        # qk = torch.masked_fill(qk, rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
        # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
        # P = torch.softmax(qk, -1)
        # dP = P * (dS - (g.float() * out.float()).sum(-1).transpose(1, 2).unsqueeze(-1))
        # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
        # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
        # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())


        # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
        # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if dtype != torch.float8_e4m3fn:
        multiple = 2 if softcap == 0.0 else 4
        assert (dq - dq_ref).abs().max().item() <= multiple * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= multiple * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= multiple * (dv_pt - dv_ref).abs().max().item()
