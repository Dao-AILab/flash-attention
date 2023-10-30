import math

import pytest
import torch
from einops import rearrange, repeat

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_triton import \
    flash_attn_func as flash_attn_func_triton


def generate_alibi(max_seq_len, num_attention_heads, tp_world_size, tp_index, device="cuda"):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        assert math.log2(n).is_integer(
        ), "it works only when num_attention_heads is power of 2"

        return get_slopes_power_of_2(n)

    slopes = torch.tensor(get_slopes(num_attention_heads))
    alibi_tensor = slopes.unsqueeze(1).unsqueeze(
        1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(num_attention_heads, -1, -1)
    # Select the part of the tensor that corresponds to our tensor parallel index.
    alibi_tensor = alibi_tensor.reshape(
        (tp_world_size, -1, *alibi_tensor.shape[1:]))[tp_index]
    # (1, nheads, 1, seqlen_k)
    alibi_tensor = alibi_tensor.unsqueeze(0).contiguous().to(
        device=device, dtype=torch.float32)

    assert (num_attention_heads/tp_world_size).is_integer(
    ), "it works only when (num_attention_heads/tp_world_size) is integer"
    nh_tp = num_attention_heads // tp_world_size
    alibi_ratio = (2 ** (-2 ** -(math.log2(num_attention_heads) - 3)))
    alibi_start = (2 ** (-2 ** -(math.log2(num_attention_heads) - 3))
                   ) * alibi_ratio ** (nh_tp * tp_index)

    return alibi_tensor, alibi_start, alibi_ratio


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
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
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device),
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


@pytest.mark.parametrize(
    "dtype", [torch.float16]
)
@pytest.mark.parametrize(
    "bs_seqlen", [(32, 512), (16, 1024), (8, 2048),
                  (4, 4096), (2, 8192), (1, 16384)]
)
@pytest.mark.parametrize(
    "headdim", [64, 128]
)
@pytest.mark.parametrize(
    "tp_world_size", [1, 2, 4, 8]
)
def test_flash_attn_func(bs_seqlen, headdim, tp_world_size, dtype):
    bs, seqlen = bs_seqlen
    nh = 2048 // headdim
    nh_tp = nh // tp_world_size
    q, k, v = [torch.randn(bs, seqlen, nh_tp, headdim, device="cuda",
                           dtype=dtype, requires_grad=True) for _ in range(3)]
    dout = torch.rand_like(q)

    for tp_index in range(tp_world_size):
        alibi, alibi_start, alibi_ratio = generate_alibi(
            seqlen, nh, tp_world_size, tp_index, "cuda")

        triton_out = flash_attn_func_triton(
            q, k, v, alibi, True, headdim**(-0.5))
        triton_out.backward(dout)
        triton_dq, q.grad = q.grad.clone(), None
        triton_dk, k.grad = k.grad.clone(), None
        triton_dv, v.grad = v.grad.clone(), None

        flash_out = flash_attn_func(
            q, k, v, causal=True, alibi=True, alibi_start=alibi_start, alibi_ratio=alibi_ratio)
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
    "bs_seqlen", [(32, 512), (16, 1024), (8, 2048),
                  (4, 4096), (2, 8192), (1, 16384)]
)
@pytest.mark.parametrize(
    "headdim", [64, 128]
)
@pytest.mark.parametrize(
    "tp_world_size", [1, 2, 4, 8]
)
def test_flash_attn_varlen_func(bs_seqlen, headdim, tp_world_size, dtype):
    bs, seqlen = bs_seqlen
    nh = 2048 // headdim
    nh_tp = nh // tp_world_size
    q, k, v = [torch.randn(bs, seqlen, nh_tp, headdim, device="cuda",
                           dtype=dtype, requires_grad=True) for _ in range(3)]
    dout = torch.rand_like(q)

    padding_mask = generate_random_padding_mask(seqlen, bs, "cuda", "random")
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
    ) = generate_qkv(q, k, v, padding_mask, padding_mask, kvpacked=False)

    for tp_index in range(tp_world_size):
        alibi, alibi_start, alibi_ratio = generate_alibi(
            seqlen, nh, tp_world_size, tp_index, "cuda")
        alibi_masked = alibi.expand(bs, -1, -1, -1).masked_fill(~(rearrange(
            padding_mask, "b sqk -> b 1 1 sqk")), torch.finfo(torch.float).min)

        triton_out = flash_attn_func_triton(
            q, k, v, alibi_masked, True, headdim**(-0.5))
        triton_out = triton_out.masked_fill(
            ~rearrange(padding_mask, "b sq -> b sq 1 1"), 0.0)
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
            alibi=True,
            alibi_start=alibi_start,
            alibi_ratio=alibi_ratio
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
