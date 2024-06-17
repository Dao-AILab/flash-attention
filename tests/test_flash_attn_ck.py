import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

from test_flash_attn import (
    attn_bias_from_alibi_slopes,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
    attention_qkvpacked_ref,
    generate_random_padding_mask
)

is_gfx94x = torch.cuda.get_device_capability("cuda") == (9, 4)


def ck_randval_to_dropout_mask(randval, p):
    # If p = 0.3, randval in 255 * (0.7, 1.0] will be dropout
    # randval in 255 * [0, 0.7] will be kept
    # If return dropout_mask >=0, value will be kept
    return torch.floor(255.0 * (1 - p) - randval)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 384, 768, 1024, 1025, 2048])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_flash_attn_qkvpacked(seqlen, d, dropout_p, causal, local, alibi, deterministic, dtype):
    if d > 256:
        pytest.skip()

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(alibi_slopes, seqlen, seqlen, causal=causal)
    else:
        alibi_slopes, attn_bias = None, None
    out, lse, S_dmask = flash_attn_qkvpacked_func(
        qkv,
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
            seqlen,
            seqlen,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(
        qkv, None, attn_bias, dropout_p, dropout_mask, causal=causal, window_size=window_size
    )
    out_pt, attn_pt = attention_qkvpacked_ref(
        qkv,
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

    if d <= 128 and d % 2 == 0:
        (dqkv,) = torch.autograd.grad(out, qkv, g)
        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
        (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
        print(f"dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
        print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

        # TODO - use 5 times to check, wait for ck to change dq type to f32
        assert (dqkv - dqkv_ref).abs().max().item() <= 5 * (dqkv_pt - dqkv_ref).abs().max().item()


def pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens, seqlen):
    # pad + rearrange [nheads, total_q, max_seqlen_k] into [b, nheads, seqlen_q, seqlen_kv]
    batch_size = cu_seqlens.numel() - 1
    seqlens = torch.roll(cu_seqlens, shifts = -1) - cu_seqlens
    seqlens = seqlens[0:batch_size].tolist()
    S_dmask = torch.split(S_dmask, seqlens, dim=1)
    masks = ()
    for mask in S_dmask:
        mask = F.pad(mask, (0, 0, 0, seqlen - mask.shape[1])).unsqueeze(1)
        masks = masks + (mask, )
    S_dmask = torch.cat(masks, dim=1)

    S_dmask = S_dmask.transpose(0, 1)
    return S_dmask


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 257, 384, 512, 768, 1025, 2048])
@pytest.mark.parametrize("dropout_p", [0, 0.17])
def test_flash_attn_varlen_qkvpacked(seqlen, d, dropout_p, causal, local, alibi, deterministic, dtype):
    if d > 256:
        pytest.skip()

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 6
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen, seqlen, key_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        *qkv.unbind(dim=2), key_padding_mask, key_padding_mask, qkvpacked=True
    )

    out_unpad, sm_lse, S_dmask = flash_attn_varlen_qkvpacked_func(
        qkv_unpad,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    out = output_pad_fn(out_unpad)
    if dropout_p > 0.0:
        # TODO - move to c++ mha_varlen_fwd()
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask = pad_rearrange_dropout_mask_hts_to_bhss(S_dmask, cu_seqlens, seqlen)

        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen,
            seqlen,
            key_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )

        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(
        qkv,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
    )
    out_pt, attn_pt = attention_qkvpacked_ref(
        qkv,
        key_padding_mask,
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
