import math

import pytest
import torch

from flash_attn import flash_attn_func
from flash_attn.flash_attn_triton import \
    flash_attn_func as triton_flash_attn_func


def generate_alibi(max_seq_len, num_attention_heads, batch_size, use_flash_attn, tp_world_size, tp_index):
    # Based on https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        assert math.log2(n).is_integer(
        ), "it works only when num_attention_heads is power of 2"
        return get_slopes_power_of_2(n)

    slopes = torch.Tensor(get_slopes(num_attention_heads))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
        num_attention_heads, -1, -1)

    # Select the part of the tensor that corresponds to our tensor parallel index.
    alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

    if use_flash_attn:
        alibi = alibi.unsqueeze(0).contiguous()
        # (1, nheads, 1, seqlen_k)
    else:
        alibi = alibi.repeat(batch_size, 1, 1).contiguous()

    assert (num_attention_heads/tp_world_size).is_integer(
    ), "it works only when (num_attention_heads/tp_world_size) is integer"
    nh_tp = num_attention_heads // tp_world_size
    alibi_ratio = (2 ** (-2 ** -(math.log2(num_attention_heads) - 3)))
    alibi_start = (2 ** (-2 ** -(math.log2(num_attention_heads) - 3))
                   ) * alibi_ratio ** (nh_tp * tp_index)

    return alibi, alibi_start, alibi_ratio


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
def test_alibi(bs_seqlen, headdim, tp_world_size, dtype):
    bs, seqlen = bs_seqlen
    nh = 2048 // headdim
    nh_tp = nh // tp_world_size
    q, k, v = [torch.randn(bs, seqlen, nh_tp, headdim, device="cuda",
                           dtype=dtype, requires_grad=True) for _ in range(3)]
    dout = torch.rand_like(q)

    for tp_index in range(tp_world_size):
        alibi, alibi_start, alibi_ratio = generate_alibi(
            seqlen, nh, bs, True, tp_world_size, tp_index)
        alibi = alibi.to(device="cuda", dtype=torch.float32)

        triton_out = triton_flash_attn_func(
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
