# Copyright (c) 2026, Tri Dao.

import math

import pytest
import torch

from flash_attn.cute.interface import (
    _flash_attn_bwd,
    _flash_attn_bwd_two_section_causal,
    _flash_attn_fwd,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="two-section causal backward requires SM100 or SM110",
)
def test_two_section_causal_backward_matches_independent_sections():
    torch.manual_seed(20260722)
    q_length, prefix1, prefix2 = 1024, 1024, 2048
    heads, head_dim = 20, 128
    scale = 1.0 / math.sqrt(head_dim)
    device = torch.device("cuda")

    q = torch.randn(
        2, q_length, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        1, prefix2, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    seqused_k = torch.tensor([prefix1, prefix2], device=device, dtype=torch.int32)
    out, lse, *_ = _flash_attn_fwd(
        q,
        k.expand(2, -1, -1, -1),
        v.expand(2, -1, -1, -1),
        causal=True,
        seqused_k=seqused_k,
        softmax_scale=scale,
        return_lse=True,
    )
    dout = torch.randn_like(out)

    dq1, dq2, dk, dv = _flash_attn_bwd_two_section_causal(
        q[:1],
        q[1:],
        k,
        v,
        out[:1],
        out[1:],
        dout[:1],
        dout[1:],
        lse[:1],
        lse[1:],
        prefix1,
        prefix2,
        softmax_scale=scale,
    )

    ref_dq1, ref_dk1, ref_dv1 = _flash_attn_bwd(
        q[:1],
        k[:, :prefix1],
        v[:, :prefix1],
        out[:1],
        dout[:1],
        lse[:1],
        causal=True,
        softmax_scale=scale,
    )
    ref_dq2, ref_dk2, ref_dv2 = _flash_attn_bwd(
        q[1:],
        k,
        v,
        out[1:],
        dout[1:],
        lse[1:],
        causal=True,
        softmax_scale=scale,
    )
    ref_dk = ref_dk2.clone()
    ref_dv = ref_dv2.clone()
    ref_dk[:, :prefix1].add_(ref_dk1)
    ref_dv[:, :prefix1].add_(ref_dv1)

    for actual, expected in (
        (dq1, ref_dq1),
        (dq2, ref_dq2),
        (dk, ref_dk),
        (dv, ref_dv),
    ):
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
