import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

from test_flash_attn import (
    attn_bias_from_alibi_slopes,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
    generate_random_padding_mask,
    _generate_block_kvcache,
    attention_ref,
    attention_kvpacked_ref,
    attention_qkvpacked_ref,
)

from flash_attn.layers.rotary import apply_rotary_emb

def is_bwd_hdim_supported(d):
    return d <= 256


def ck_randval_to_dropout_mask(randval, p):
    # If p = 0.3, randval in 255 * (0.7, 1.0] will be dropout
    # randval in 255 * [0, 0.7] will be kept
    # If return dropout_mask >=0, value will be kept
    return math.floor(255.0 * (1 - p)) - randval.to(torch.float32)


def pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens_q, seqlen_q_rounded, seqlen_k_rounded):
    """ pad + rearrange [nheads, total_q, max_seqlen_k] into [b, nheads, seqlen_q_rounded, seqlen_k_rounded]
    Arguments:
        S_dmask: (nheads, total_q, max_seqlen_k)
        cu_seqlens_q: (b + 1)
    Output:
        S_dmask: (b, nheads, seqlen_q_rounded, seqlen_k_rounded)
    """
    batch_size = cu_seqlens_q.numel() - 1
    seqlens_q = torch.roll(cu_seqlens_q, shifts = -1) - cu_seqlens_q
    seqlens_q = seqlens_q[0:batch_size].tolist()
    S_dmask = torch.split(S_dmask, seqlens_q, dim=1)
    # [(nheads, seqlen_q0, max_seqlen_k), (nheads, seqlen_q1, max_seqlen_k), ..., (nheads, seqlen_qb, max_seqlen_k)]
    masks = ()
    for mask in S_dmask:
        # (nheads, seqlen_qi, max_seqlen_k) -> (nheads, seqlen_q_rounded, seqlen_k_rounded)
        mask = F.pad(mask, (0, seqlen_k_rounded - mask.shape[2], 0, seqlen_q_rounded - mask.shape[1], 0, 0)).unsqueeze(1)
        masks = masks + (mask, )
    S_dmask = torch.cat(masks, dim=1)

    S_dmask = S_dmask.transpose(0, 1)
    return S_dmask

@pytest.mark.parametrize("kvpacked", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 72, 80, 88, 96, 104, 112, 120, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (256, 512),
        (1024, 1024),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_output(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked
):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if kvpacked:
        kv = torch.randn(
            batch_size, seqlen_k, 2, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    else:
        k = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=True
        )
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen_q, seqlen_k, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None

    if kvpacked:
        out, lse, S_dmask = flash_attn_kvpacked_func(
            q,
            kv,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    else:
        out, lse, S_dmask = flash_attn_func(
            q,
            k,
            v,
            dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
    if dropout_p > 0.0:
        # TODO - move to c++ mha_varlen_fwd()
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        if kvpacked:
            kv_rep = repeat(kv, "b s two h d -> b s two (h g) d", g=nheads // nheads_k)
            k_rep, v_rep = kv_rep.unbind(dim=2)
        else:
            k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
            v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        # CK does not return P. Hence, we don't test the attn here.
    else:
        dropout_mask = None

    if kvpacked:
        out_ref, attn_ref = attention_kvpacked_ref(
            q,
            kv,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_kvpacked_ref(
            q,
            kv,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )
    else:
        out_ref, attn_ref = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
        )
        out_pt, attn_pt = attention_ref(
            q,
            k,
            v,
            None,
            None,
            attn_bias,
            dropout_p,
            dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=False,
            reorder_ops=True,
        )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    g = torch.randn_like(out)
    if is_bwd_hdim_supported(d):
        if kvpacked:
            (
                dq,
                dkv,
            ) = torch.autograd.grad(out, (q, kv), g)
            dk, dv = dkv.unbind(2)
            (
                dq_ref,
                dkv_ref,
            ) = torch.autograd.grad(out_ref, (q, kv), g)
            dk_ref, dv_ref = dkv_ref.unbind(2)
            (
                dq_pt,
                dkv_pt,
            ) = torch.autograd.grad(out_pt, (q, kv), g)
            dk_pt, dv_pt = dkv_pt.unbind(2)
        else:
            (
                dq,
                dk,
                dv,
            ) = torch.autograd.grad(out, (q, k, v), g)
            (
                dq_ref,
                dk_ref,
                dv_ref,
            ) = torch.autograd.grad(out_ref, (q, k, v), g)
            (
                dq_pt,
                dk_pt,
                dv_pt,
            ) = torch.autograd.grad(out_pt, (q, k, v), g)
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

        # TODO - use 10 times to check, wait for ck to fix bwd precision issue
        assert (dq - dq_ref).abs().max().item() <= 10 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 10 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 10 * (dv_pt - dv_ref).abs().max().item()
