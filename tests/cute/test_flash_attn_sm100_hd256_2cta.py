# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

# Integrated into flash-attention: exercises SM100 Blackwell head_dim=256 2CTA Cute-DSL kernels
# via `flash_attn.cute.interface` (524 parametrized cases from the upstream Aliyun test suite).

r"""Unit test for FMHA (SM100 hdim=256 2CTA)."""

import itertools
import math
import random

import pytest
import torch
import torch.nn.functional as f
from einops import rearrange, repeat

from flash_attn.cute.interface import flash_attn_varlen_func
from flash_attn.cute.interface import (
    flash_attn_bwd_sm100_hd256_2cta as _flash_attn_backward_sm100,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM100 Blackwell required for head_dim=256 2CTA kernel tests",
)


@pytest.fixture(autouse=True)
def _reset_hd256_2cta_trace():
    import flash_attn.cute.sm100_hd256_2cta_trace as _hd256_tr

    _hd256_tr.reset()
    yield


class IndexFirstAxis(torch.autograd.Function):
    """Helper class for UT."""

    @staticmethod
    def forward(ctx, input_t, indices):
        """Forward pass."""
        ctx.save_for_backward(indices)
        assert input_t.ndim >= 2  # noqa: PLR2004
        ctx.first_axis_dim, other_shape = input_t.shape[0], input_t.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input_t, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2  # noqa: PLR2004
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    """Helper class for UT."""

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        """Forward pass."""
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2  # noqa: PLR2004
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Unpad input."""
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = f.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """Pad input."""
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random", zero_lengths=False):
    """Generate padding mask."""
    assert mode in {"full", "random", "third"}
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(0 if zero_lengths else 1, max_seqlen - 20),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )
    else:
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)

    if zero_lengths:
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    return repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    qv=None,
    kvpacked=False,
    qkvpacked=False,
    query_unused_mask=None,
    key_unused_mask=None,
):
    """Generate Q,K and V."""
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    d_v = v.shape[-1]
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d_v)
    if query_unused_mask is not None or key_unused_mask is not None:
        assert not kvpacked
        assert not qkvpacked

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, seqused_q = unpad_input(
            q, query_padding_mask, query_unused_mask
        )

        def output_pad_fn(output_unpad):
            return pad_input(output_unpad, indices_q, batch_size, seqlen_q)

        qv_unpad = rearrange(qv, "b s ... -> (b s) ...")[indices_q] if qv is not None else None
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        seqused_q = None
        max_seqlen_q = seqlen_q

        def output_pad_fn(output_unpad):
            return rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

        qv_unpad = rearrange(qv, "b s ... -> (b s) ...") if qv is not None else None

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, seqused_k = unpad_input(k, key_padding_mask, key_unused_mask)
        v_unpad, *_ = unpad_input(v, key_padding_mask, key_unused_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        seqused_k = None
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:

            def dqkv_pad_fn(dqkv_unpad):
                return pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)

        else:

            def dqkv_pad_fn(dqkv_unpad):
                return rearrange(dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)

        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    if kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:

            def dkv_pad_fn(dkv_unpad):
                return pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)

        else:

            def dkv_pad_fn(dkv_unpad):
                return rearrange(dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)

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
    dq_pad_fn = output_pad_fn
    if key_padding_mask is not None:

        def dk_pad_fn(dk_unpad):
            return pad_input(dk_unpad, indices_k, batch_size, seqlen_k)

    else:

        def dk_pad_fn(dk_unpad):
            return rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)

    return (
        q_unpad.detach().requires_grad_(),
        k_unpad.detach().requires_grad_(),
        v_unpad.detach().requires_grad_(),
        qv_unpad.detach() if qv is not None else None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        q.detach().requires_grad_(),
        k.detach().requires_grad_(),
        v.detach().requires_grad_(),
        qv.detach() if qv is not None else None,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    """Construct local mask."""
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    if window_size[0] == -1:
        return col_idx > row_idx + sk - sq + window_size[1]
    sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk

    return torch.logical_or(
        col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
        torch.logical_and(col_idx < row_idx + sk - sq - window_size[0], col_idx >= sink_token_length),
    )


def construct_chunk_mask(
    seqlen_q,
    seqlen_k,
    attention_chunk,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    """Helper class for construct chunk mask."""
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
    col_limit_left_chunk = row_idx + sk - sq - (row_idx + sk - sq) % attention_chunk
    return torch.logical_or(col_idx < col_limit_left_chunk, col_idx >= col_limit_left_chunk + attention_chunk)


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    sink_token_length=0,
    learnable_sink: torch.Tensor | None = None,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
):
    """Groudtruth of FMHA."""
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        qv = qv.float() if qv is not None else None
    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2])
        q = (q.float() * q_descale).to(q.dtype)
        qv = (qv.float() * q_descale).to(qv.dtype) if qv is not None else None
    if k_descale is not None:
        k = (k.float() * rearrange(k_descale, "b h -> b 1 h 1")).to(dtype=k.dtype)
    if v_descale is not None:
        v = (v.float() * rearrange(v_descale, "b h -> b 1 h 1")).to(dtype=v.dtype)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d if qv is None else d + dv)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if qv is not None:
        scores += torch.einsum("bthd,bshd->bhts", qv * softmax_scale, v)
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    local_mask = None
    if window_size[0] != -1 or window_size[1] != -1:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            sink_token_length,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
        )
    if attention_chunk > 0:
        chunk_mask = construct_chunk_mask(
            seqlen_q,
            seqlen_k,
            attention_chunk,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
        )
        local_mask = torch.logical_or(local_mask, chunk_mask) if local_mask is not None else chunk_mask
    if local_mask is not None:
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores += attn_bias
    if learnable_sink is None:
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
    else:
        scores_fp32 = scores.to(torch.float32)
        logits_max = torch.amax(scores_fp32, dim=-1, keepdim=True)
        learnable_sink = rearrange(learnable_sink, "h -> h 1 1")
        logits_or_sinks_max = torch.maximum(learnable_sink, logits_max)
        unnormalized_scores = torch.exp(scores_fp32 - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(learnable_sink - logits_or_sinks_max)
        attention = (unnormalized_scores / normalizer).to(v.dtype)
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    if key_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    if local_mask is not None:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    attention_drop = attention.masked_fill(~dropout_mask, 0.0) if dropout_mask is not None else attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("has_learnable_sink", [False])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",  # noqa : PT006
    [
        (64, 128),
        (128, 192),
        (256, 256),
        (64, 1),
        (799, 3),
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
        (255, 256),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
def test_flash_attn_output(
    seqlen_q, seqlen_k, causal, local, softcap, deterministic, has_qv, has_learnable_sink, mha_type, dtype
):
    """Testing forward and backward of FMHA in fixed length."""
    d = 256
    if (causal or local) and seqlen_k < seqlen_q:
        pytest.skip("Causal attention requires seqlen_k >= seqlen_q")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 9 if seqlen_k <= 2048 else 2  # noqa: PLR2004
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])  # noqa: PLR2004
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = q_ref * softcap / 4
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = (
            torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        v_ref = (
            torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        if has_qv:
            qv_ref = (
                torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            )
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device) if has_learnable_sink else None
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [
                torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)
            ]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
        _ = qv_ref.detach().to(dtype).requires_grad_() if has_qv else None
        out_ref, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
        )
        out_pt, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        pack_gqa_vals = [False, True, None]
        num_splits_vals = [1]
        for _, _ in itertools.product(pack_gqa_vals, num_splits_vals):
            out, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                None,
                None,
                None,
                None,
                None,
                None,
                causal=causal,
                window_size=window_size,
            )
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

            # Check that FlashAttention's numerical error is at most twice the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol

        if (
            dtype != torch.float8_e4m3fn
            and not has_qv
            and dv <= 256  # noqa: PLR2004
            and attention_chunk == 0
            and softcap == 0.0
            and not local
            and dv == d
            and learnable_sink is None
        ):
            g = torch.randn_like(out)
            d_q, d_k, d_v = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            print(f"dQ max diff: {(d_q - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(d_k - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(d_v - dv_ref).abs().max().item()}")
            print(f"dQ mean diff: {(d_q - dq_ref).abs().mean().item()}")
            print(f"dK mean diff: {(d_k - dk_ref).abs().mean().item()}")
            print(f"dV mean diff: {(d_v - dv_ref).abs().mean().item()}")
            print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
            print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
            print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
            print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
            print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
            print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (d_q - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (d_k - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (d_v - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("add_unused_qkv", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",  # noqa : PT006
    [
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (1, 1),
        (64, 1),
        (64, 10),
        (1, 63),
        (1, 128),
        (1204, 102),
        (64, 128),
        (128, 192),
        (256, 256),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (255, 256),
        (4096, 4096),
        (4224, 4224),
        (4012, 10235),
    ],
)
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, add_unused_qkv, causal, local, softcap, deterministic, has_qv, mha_type, dtype
):
    """Testing forward and backward of FMHA in variable length."""
    d = 256
    dv_vals = [d]
    if (causal or local) and seqlen_k < seqlen_q:
        pytest.skip("Causal attention requires seqlen_k >= seqlen_q")
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    batch_size = random.randint(1, 10)
    nheads = 16
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    assert nheads % nheads_kv == 0
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype

    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4).detach().requires_grad_()
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = (
            torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        v_ref = (
            torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        if has_qv:
            qv_ref = (
                torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            )
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        learnable_sink = None
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [
                torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)
            ]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        # TODO: test zero_lengths
        key_padding_mask = generate_random_padding_mask(
            seqlen_k,
            batch_size,
            device,
            mode="random",
            zero_lengths=False,
        )

        def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
            if add_unused:
                another_mask = generate_random_padding_mask(max_seq_len, bs, device)
                attn_mask = torch.logical_and(padding_mask, another_mask)
                unused_mask = torch.logical_xor(torch.logical_or(padding_mask, another_mask), attn_mask)
            else:
                attn_mask = padding_mask
                unused_mask = None
            return attn_mask, unused_mask

        query_padding_mask = None
        query_unused_mask = None
        key_padding_mask = None
        key_unused_mask = None

        if causal or local:
            key_padding_mask = query_padding_mask

        (
            q_unpad,
            k_unpad,
            v_unpad,
            _,
            cu_seqlens_q,
            cu_seqlens_k,
            _,
            _,
            _,
            _,
            q,
            k,
            v,
            qv,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            qv=qv,
            kvpacked=False,
            query_unused_mask=query_unused_mask,
            key_unused_mask=key_unused_mask,
        )
        q_unpad, k_unpad, v_unpad = [x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)]
        out_ref, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
        )

        out_pt, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        if query_unused_mask is not None:
            q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref.detach() + 0.3 - 0.3 - out_ref.detach()).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        pack_gqa_vals = [False, True, None]
        num_splits_vals = [1]
        for _, _ in itertools.product(pack_gqa_vals, num_splits_vals):
            out_unpad, _ = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                None,
                None,
                None,
                None,
                causal=causal,
                window_size=window_size,
            )

            # Check that FlashAttention's numerical error is at most 3x the numerical error
            # of a Pytorch implementation.
            out_copy = out_unpad.detach().reshape(-1)
            out_ref_copy = out_ref.detach().reshape(-1)
            out_pt_copy = out_pt.detach().reshape(-1)

            assert (out_copy - out_ref_copy).abs().max().item() <= rtol * (
                out_pt_copy - out_ref_copy
            ).abs().max().item() + fwd_atol

        print(f"attention_chunk : {attention_chunk}", flush=True)
        if (
            dtype != torch.float8_e4m3fn
            and not has_qv
            and dv <= 256  # noqa: PLR2004
            and attention_chunk == 0
            and dv == d
        ):
            g_unpad = torch.randn_like(out_unpad)

            dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad)
            d_q = dq_pad_fn(dq_unpad)
            d_k = dk_pad_fn(dk_unpad)
            d_v = dk_pad_fn(dv_unpad)
            if key_unused_mask is not None:
                k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
                d_k.masked_fill_(k_zero_masking, 0.0)
                d_v.masked_fill_(k_zero_masking, 0.0)
            if query_unused_mask is not None:
                d_q.masked_fill_(q_zero_masking, 0.0)
            g = output_pad_fn(g_unpad)

            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            print(f"dQ max diff: {(d_q - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(d_k - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(d_v - dv_ref).abs().max().item()}")
            print(f"dQ mean diff: {(d_q - dq_ref).abs().mean().item()}")
            print(f"dK mean diff: {(d_k - dk_ref).abs().mean().item()}")
            print(f"dV mean diff: {(d_v - dv_ref).abs().mean().item()}")
            print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
            print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
            print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
            print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
            print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
            print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (d_q - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (d_k - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (d_v - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol


@torch.no_grad()
def _stats(name, a, b, atol, rtol):
    diff = (a - b).float()
    mean_abs = diff.abs().mean().item()
    mean_rel = diff.abs().mean() / b.abs().clamp_min(1e-6).mean().item()
    print(f"{name}: mean_abs={mean_abs:.4e}, mean_rel={mean_rel:.4e}, sum_fa={a.sum()}, sum_ref={b.sum()}")
    return mean_abs < atol and mean_rel < rtol


def generate_varlen_args(
    batch_size=8,
    n_heads=16,
    d_head=128,
    min_len=32,
    max_len=64,
    mha_type="mha",
    dtype=torch.bfloat16,
):
    """Generate variable length input data."""
    torch.manual_seed(0)
    device = "cuda"

    assert mha_type in {"mha", "mqa", "gqa"}

    lens_q = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    lens_k = lens_q.clone()

    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lens_q.cumsum(0)])
    cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32), lens_k.cumsum(0)])

    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]

    cu_seqlens_q = cu_seqlens_q.contiguous().to(dtype=torch.int32, device=device)
    cu_seqlens_k = cu_seqlens_k.contiguous().to(dtype=torch.int32, device=device)

    if mha_type == "gqa":
        h = 3 * n_heads
        h_kv = n_heads
    elif mha_type == "mha":
        h = h_kv = n_heads
    else:  # MQA
        h = n_heads
        h_kv = 1

    d_head_v = d_head

    q = torch.randn(total_q, h, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, h_kv, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, h_kv, d_head_v, device=device, dtype=dtype, requires_grad=True)

    return q, k, v, cu_seqlens_q, cu_seqlens_k, total_q, total_k


# Simple for loop over batch dim implementation
def torch_flash_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    total_q: int = 0,
    total_k: int = 0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **kwargs,
):
    """Groundtruth of FMHA.

    q: (total_q, H, d) if cu_seqlens_q is not None, otherwise (B, L, H, d).
    k: (total_k, H_kv, d) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d).
    v: (total_k, H_kv, d_v) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d_v).
    cu_seqlens_q: (B+1,) int32, cumulative.
    cu_seqlens_k: (B+1,) int32, cumulative.

    seqused_q: (B+1,) int32.
    seqused_k: (B+1,) int32.

    Returns:
        out packed like q: (total_q, H, d_v).
    """
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.dim() == 1
        assert total_q == q.shape[0]
        assert q.dim() == 3  # noqa: PLR2004
        h = q.shape[1]
        b = cu_seqlens_q.shape[0] - 1
    else:
        assert q.dim() == 4  # noqa: PLR2004
        h = q.shape[2]
        b = q.shape[0]

    if cu_seqlens_k is not None:
        assert cu_seqlens_k.dim() == 1
        assert total_k == k.shape[0] == v.shape[0]
        assert k.dim() == v.dim() == 3  # noqa: PLR2004
        h_kv = k.shape[1]
        b_kv = cu_seqlens_k.shape[0] - 1
    else:
        assert k.dim() == v.dim() == 4  # noqa: PLR2004
        assert k.shape[0] == v.shape[0]
        h_kv = k.shape[2]
        b_kv = k.shape[0]

    d = q.shape[-1]

    assert h_kv == v.shape[-2]
    assert d == k.shape[-1]
    assert b_kv == b

    assert q.device == k.device == v.device
    assert q.is_floating_point()
    assert k.is_floating_point()
    assert v.is_floating_point()

    device = q.device
    dtype = q.dtype

    hcseq_q = cu_seqlens_q.to(device="cpu") if cu_seqlens_q is not None else cu_seqlens_q
    hcseq_k = cu_seqlens_k.to(device="cpu") if cu_seqlens_k is not None else cu_seqlens_k

    outs = []
    for bid in range(b):
        if hcseq_q is not None:
            q_start, q_end = int(hcseq_q[bid]), int(hcseq_q[bid + 1])
            qb = q[q_start:q_end]
        else:
            qb = q[bid]

        if hcseq_k is not None:
            k_start, k_end = int(hcseq_k[bid]), int(hcseq_k[bid + 1])
            kb = k[k_start:k_end]
            vb = v[k_start:k_end]
        else:
            kb = k[bid]
            vb = v[bid]

        qb = qb.permute(1, 0, 2).unsqueeze(0)
        kb = kb.permute(1, 0, 2).unsqueeze(0)
        vb = vb.permute(1, 0, 2).unsqueeze(0)

        ob = f.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=softmax_scale, enable_gqa=h_kv != h
        )

        ob = ob.squeeze(0).permute(1, 0, 2).contiguous()
        outs.append(ob)

    if cu_seqlens_q is not None:
        out = torch.cat(outs, dim=0).to(device=device, dtype=dtype)
    else:
        out = torch.stack(outs, dim=0).to(device=device, dtype=dtype)
    return out


def check_backward_vs_torch_flash(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    total_q=None,
    total_k=None,
    softmax_scale=None,
    causal=True,
    mha_type="mha",
    softcap=0.0,
    atol=3e-2,
    rtol=3e-2,
):
    """Groudtruth of FMHA backward."""
    assert q.requires_grad, "Set requires_grad=True on inputs"
    assert k.requires_grad, "Set requires_grad=True on inputs"
    assert v.requires_grad, "Set requires_grad=True on inputs"

    def clone_like(t):
        return t.clone().detach().requires_grad_(True)

    q_fa, k_fa, v_fa = map(clone_like, (q, k, v))
    q_t, k_t, v_t = map(clone_like, (q, k, v))

    if cu_seqlens_q is not None:
        cu_seqlens_q_fa = cu_seqlens_q.clone()
        cu_seqlens_q_t = cu_seqlens_q.clone()
    else:
        cu_seqlens_q_fa = None
        cu_seqlens_q_t = None

    if cu_seqlens_k is not None:
        cu_seqlens_k_fa = cu_seqlens_k.clone()
        cu_seqlens_k_t = cu_seqlens_k.clone()
    else:
        cu_seqlens_k_fa = None
        cu_seqlens_k_t = None

    out_fa, _ = flash_attn_varlen_func(
        q_fa,
        k_fa,
        v_fa,
        cu_seqlens_q=cu_seqlens_q_fa,
        cu_seqlens_k=cu_seqlens_k_fa,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=(1.0 / q.shape[-1] ** 0.5) if softmax_scale is None else softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )

    out_t = torch_flash_ref(
        q_t,
        k_t,
        v_t,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_k=cu_seqlens_k_t,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        total_q=total_q,
        total_k=total_k,
        softmax_scale=softmax_scale,
        causal=causal,
        mha_type=mha_type,
    )

    # Check the correctness of the forward pass
    torch.testing.assert_close(out_fa.float(), out_t.float(), atol=atol, rtol=rtol)

    # Use the same upstream gradient to compare backward paths
    grad_out = torch.randn_like(out_fa)

    grad_fa = clone_like(grad_out)
    grad_t = clone_like(grad_out)

    # Cute bwd
    out_fa.backward(grad_fa, retain_graph=False)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    # Ref bwd
    out_t.backward(grad_t, retain_graph=False)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    ok_q = torch.allclose(dq_fa.float(), dq_t.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_fa.float(), dk_t.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_fa.float(), dv_t.float(), atol=atol, rtol=rtol)
    return ok_q and ok_k and ok_v


@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("h", [4, 6])
@pytest.mark.parametrize("min_seq_len", [32, 128])
@pytest.mark.parametrize("max_seq_len", [64, 2048])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("softmax_scale", [None, 0.1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
def test_varlen(
    b,
    h,
    min_seq_len,
    max_seq_len,
    causal,
    softmax_scale,
    dtype,
    mha_type,
):
    """Testing forward and backward of FMHA in random variable length."""
    d = 256
    if min_seq_len > max_seq_len:
        pytest.skip("Skipping min_seq_len > max_seq_len")

    q, k, v, cu_seqlens_q, cu_seqlens_k, total_q, total_k = generate_varlen_args(
        batch_size=b, n_heads=h, d_head=d, min_len=min_seq_len, max_len=max_seq_len, mha_type=mha_type, dtype=dtype
    )

    ok = check_backward_vs_torch_flash(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q=total_q,
        total_k=total_k,
        softmax_scale=softmax_scale,
        causal=causal,
        mha_type=mha_type,
    )
    assert ok


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",  # noqa : PT006
    [
        (16384, 1048576),
        (32768, 1048576),
        (65536, 1048576),
    ],
)
def test_extremely_large_sequence(seqlen_q, seqlen_k, causal, mha_type, dtype):
    """Testing extremely large sequence, only smoke test."""
    d = 256
    if causal and seqlen_k < seqlen_q:
        pytest.skip("Causal attention requires seqlen_k >= seqlen_q")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 9 if seqlen_k <= 2048 else 2  # noqa: PLR2004
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])  # noqa: PLR2004
    attention_chunk_vals = [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = (
            torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        v_ref = (
            torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref)
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1)
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]

        pack_gqa_vals = [False, True, None]
        num_splits_vals = [1]
        for _, _ in itertools.product(pack_gqa_vals, num_splits_vals):
            out, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                None,
                None,
                None,
                None,
                None,
                None,
                causal=causal,
                window_size=window_size,
            )
            assert not out.isnan().any()
        if (
            dv <= 256  # noqa: PLR2004
            and attention_chunk == 0
            and dv == d
        ):
            g = torch.randn_like(out)
            d_q, d_k, d_v = torch.autograd.grad(out, (q, k, v), g)
            assert not d_q.isnan().any()
            assert not d_k.isnan().any()
            assert not d_v.isnan().any()


def generate_cu_seqlens_concise(total_tokens, batch_size, device="cuda"):
    """Generate varlen under total_tokens' constraint."""
    if total_tokens > batch_size:
        weights = torch.rand(batch_size, device=device)
        weights /= weights.sum()
        remaining_tokens = total_tokens - batch_size
        additional_lengths = (weights * remaining_tokens).floor().long()

        diff = remaining_tokens - additional_lengths.sum().item()
        additional_lengths[-1] += diff

        seq_lengths = torch.ones(batch_size, dtype=torch.long, device=device) + additional_lengths
    else:
        seq_lengths = torch.ones(batch_size, dtype=torch.long, device=device)

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seq_lengths, dim=0)

    return cu_seqlens_q


@pytest.mark.parametrize("num_docs", [157])
@pytest.mark.parametrize("num_q_heads", [16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "total_q,total_k",  # noqa : PT006
    [
        (16384, 16384),
    ],
)
def test_large_batch_packed_gqa_backward(total_q, total_k, causal, num_docs, num_q_heads):
    """Testing large batch size of variable length case."""
    d = 256
    device = "cuda"
    q = torch.randn(total_q, num_q_heads, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(total_k, num_q_heads, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(total_k, num_q_heads, d, device=device, dtype=torch.bfloat16)
    softmax_lse = torch.randn(num_q_heads, total_q, device=device, dtype=torch.float32)
    out = torch.randn(total_q, num_q_heads, d, device=device, dtype=torch.bfloat16)
    dout = torch.randn_like(out)

    cu_seqlens_q = generate_cu_seqlens_concise(total_q, num_docs, device)
    cu_seqlens_k = cu_seqlens_q

    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    _flash_attn_backward_sm100(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        dq,
        dk,
        dv,
        softmax_scale=1.0 / math.sqrt(d),
        causal=causal,
        window_size=[-1, -1],
    )

    assert not dq.isnan().any()
    assert not dk.isnan().any()
    assert not dv.isnan().any()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "batch_size,min_len_q,max_len_q,min_len_k,max_len_k",
    [
        (4, 32, 128, 128, 512),       # typical asymmetric varlen
        (2, 128, 256, 512, 2048),      # larger K
        (8, 32, 64, 64, 256),          # many short sequences
        (1, 512, 512, 2048, 2048),     # single batch, fixed ratio
        (2, 128, 128, 128, 128),       # s_q == s_k baseline
    ],
)
def test_varlen_asymmetric(batch_size, min_len_q, max_len_q, min_len_k, max_len_k, causal, mha_type, dtype):
    """Test forward and backward with s_q <= s_k varlen sequences (per-sequence asymmetric lengths).

    Uses attention_ref (right-aligned causal) as ground truth — NOT torch SDPA which is
    left-aligned when seqlen_q != seqlen_k.
    """
    d = 256
    n_heads = 4
    torch.manual_seed(42)
    device = "cuda"

    if mha_type == "gqa":
        nheads, nheads_kv = 4 * n_heads, n_heads
    elif mha_type == "mha":
        nheads = nheads_kv = n_heads
    else:  # MQA
        nheads, nheads_kv = n_heads, 1

    lens_q = torch.randint(low=min_len_q, high=max_len_q + 1, size=(batch_size,))
    lens_k = torch.randint(low=min_len_k, high=max_len_k + 1, size=(batch_size,))
    lens_k = torch.maximum(lens_k, lens_q)  # ensure s_q <= s_k

    max_sq, max_sk = lens_q.max().item(), lens_k.max().item()

    # Padded tensors for reference (batch, max_seqlen, heads, d)
    q_ref = torch.zeros(batch_size, max_sq, nheads, d, device=device, dtype=dtype)
    k_ref = torch.zeros(batch_size, max_sk, nheads_kv, d, device=device, dtype=dtype)
    v_ref = torch.zeros(batch_size, max_sk, nheads_kv, d, device=device, dtype=dtype)

    # Fill with random data per sequence
    for i in range(batch_size):
        q_ref[i, :lens_q[i]] = torch.randn(lens_q[i], nheads, d, device=device, dtype=dtype)
        k_ref[i, :lens_k[i]] = torch.randn(lens_k[i], nheads_kv, d, device=device, dtype=dtype)
        v_ref[i, :lens_k[i]] = torch.randn(lens_k[i], nheads_kv, d, device=device, dtype=dtype)

    q_ref = q_ref.requires_grad_(True)
    k_ref = k_ref.requires_grad_(True)
    v_ref = v_ref.requires_grad_(True)

    # Build padding masks (True = valid token)
    query_padding_mask = torch.arange(max_sq, device=device).unsqueeze(0) < lens_q.to(device).unsqueeze(1)
    key_padding_mask = torch.arange(max_sk, device=device).unsqueeze(0) < lens_k.to(device).unsqueeze(1)

    # Reference: attention_ref with right-aligned causal mask
    window_size = (-1, -1)
    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref,
        query_padding_mask, key_padding_mask,
        causal=causal, window_size=window_size,
    )
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref,
        query_padding_mask, key_padding_mask,
        causal=causal, window_size=window_size,
        upcast=False, reorder_ops=True,
    )
    fwd_atol = 2 * (out_ref.detach() + 0.3 - 0.3 - out_ref.detach()).abs().max().item()

    # Pack into varlen format
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.tensor(lens_q, device=device).cumsum(0)
    cu_seqlens_k[1:] = torch.tensor(lens_k, device=device).cumsum(0)
    total_q, total_k = cu_seqlens_q[-1].item(), cu_seqlens_k[-1].item()

    q_unpad = torch.cat([q_ref[i, :lens_q[i]] for i in range(batch_size)], dim=0)
    k_unpad = torch.cat([k_ref[i, :lens_k[i]] for i in range(batch_size)], dim=0)
    v_unpad = torch.cat([v_ref[i, :lens_k[i]] for i in range(batch_size)], dim=0)
    q_unpad = q_unpad.detach().requires_grad_(True)
    k_unpad = k_unpad.detach().requires_grad_(True)
    v_unpad = v_unpad.detach().requires_grad_(True)

    out_fa, _ = flash_attn_varlen_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        None, None, None, None,
        causal=causal, window_size=window_size,
    )

    # Unpack and compare per-sequence
    for i in range(batch_size):
        sq_i = lens_q[i].item()
        q_start = cu_seqlens_q[i].item()
        out_fa_i = out_fa[q_start:q_start + sq_i]
        out_ref_i = out_ref[i, :sq_i]
        out_pt_i = out_pt[i, :sq_i]
        fa_err = (out_fa_i.float() - out_ref_i.float()).abs().max().item()
        pt_err = (out_pt_i.float() - out_ref_i.float()).abs().max().item()
        assert fa_err <= 2 * pt_err + fwd_atol, (
            f"Fwd seq {i}: fa_err={fa_err:.6f} > 2*pt_err({pt_err:.6f})+atol({fwd_atol:.6f})"
        )

    # Backward check
    g = torch.randn_like(out_fa)
    out_fa.backward(g, retain_graph=False)

    # Pad grad back to reference shape
    g_ref = torch.zeros_like(out_ref)
    for i in range(batch_size):
        sq_i = lens_q[i].item()
        q_start = cu_seqlens_q[i].item()
        g_ref[i, :sq_i] = g[q_start:q_start + sq_i]

    out_ref.backward(g_ref, retain_graph=False)

    for i in range(batch_size):
        sq_i, sk_i = lens_q[i].item(), lens_k[i].item()
        q_start, k_start = cu_seqlens_q[i].item(), cu_seqlens_k[i].item()
        dq_atol = 2 * (q_ref.grad[i, :sq_i].detach() + 0.3 - 0.3 - q_ref.grad[i, :sq_i].detach()).abs().max().item()
        dk_atol = 2 * (k_ref.grad[i, :sk_i].detach() + 0.3 - 0.3 - k_ref.grad[i, :sk_i].detach()).abs().max().item()
        dv_atol = 2 * (v_ref.grad[i, :sk_i].detach() + 0.3 - 0.3 - v_ref.grad[i, :sk_i].detach()).abs().max().item()
        dq_err = (q_unpad.grad[q_start:q_start+sq_i].float() - q_ref.grad[i, :sq_i].float()).abs().max().item()
        dk_err = (k_unpad.grad[k_start:k_start+sk_i].float() - k_ref.grad[i, :sk_i].float()).abs().max().item()
        dv_err = (v_unpad.grad[k_start:k_start+sk_i].float() - v_ref.grad[i, :sk_i].float()).abs().max().item()
        assert dq_err <= 2 * 0.015 + dq_atol, f"dQ seq {i}: err={dq_err:.6f}"
        assert dk_err <= 2 * 0.015 + dk_atol, f"dK seq {i}: err={dk_err:.6f}"
        assert dv_err <= 2 * 0.015 + dv_atol, f"dV seq {i}: err={dv_err:.6f}"
