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
    #Causal is the special case where window_size_right == 0 and window_size_left < 0.
    window_size = (-1, 0) if causal else window_size

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
        S_dmask = torch.floor(255.0 - 255.0 * dropout_p - S_dmask)
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

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 59, 64, 80, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 257, 384, 512, 768, 1025, 2048])
@pytest.mark.parametrize("dropout_p", [0.0])
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
        # [nheads, total_q, max_seqlen_k]
        # TODO - pad and rearrange p into [b, nheads, seqlen_q, seqlen_kv]
        # S_dmask = rearrange(S_dmask, "h (b sq) sk -> b h sq sk", b=batch_size)
        S_dmask = torch.floor(255.0 - 255.0 * dropout_p - S_dmask)
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
