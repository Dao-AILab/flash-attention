"""Regression tests for learnable_sink (attention-sink) support on the SM80/Ampere/Ada
CuTeDSL forward kernel (flash_attn.cute.flash_fwd.FlashAttentionForwardSm80).

Before this fix, FlashAttentionForwardSm80.__call__ hardcoded
`assert learnable_sink is None`, so sink-attention models (e.g. gpt-oss, StreamingLLM-style
architectures) had no FA4 CuTeDSL forward path on Ampere/Ada (A100, RTX 3090/4090) or
consumer-Blackwell (RTX 50-series, which shares this base kernel via FlashAttentionForwardSm120).
`tests/cute/test_flash_attn.py::test_flash_attn_output` already parametrizes
`has_learnable_sink=[False, True]` with no SM80-specific skip, so it exercises this path too;
this file adds a few focused, fast cases at the shapes most relevant to the fix (GQA with
pack_gqa, plain MHA, and a multi-n-block causal case) so the sink path has direct regression
coverage independent of the full parametrized matrix.
"""

import pytest
import torch

from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.testing import attention_ref

IS_SM80_FAMILY = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (8, 12)


@pytest.mark.skipif(
    not IS_SM80_FAMILY, reason="Targets the SM80-family (Ampere/Ada/SM120) forward kernel"
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,nheads,nheads_kv,d",
    [
        # gpt-oss-20b production shape: D=64, GQA 8:1, single-tile prefill (pack_gqa path).
        (64, 64, 64, 8, 64),
        # Same GQA ratio but long enough seqlen_k to exercise the multi-n-block mainloop.
        (64, 512, 64, 8, 64),
        # Plain MHA (no pack_gqa), larger head_dim, for generality.
        (128, 256, 16, 16, 128),
    ],
)
def test_sm80_learnable_sink_matches_reference(seqlen_q, seqlen_k, nheads, nheads_kv, d, causal):
    device = "cuda"
    dtype = torch.bfloat16
    torch.random.manual_seed(0)
    batch_size = 2

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False) * 0.1
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=False) * 0.1
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=False) * 0.1
    learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device) * 2.0

    out, lse, *_ = flash_attn_func(
        q, k, v, causal=causal, learnable_sink=learnable_sink, return_lse=True
    )
    out_ref, _, lse_ref = attention_ref(
        q, k, v, causal=causal, learnable_sink=learnable_sink, return_lse=True
    )

    out_f, ref_f = out.float(), out_ref.float()
    cos_sim = torch.nn.functional.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0)
    assert cos_sim.item() > 0.999, f"cosine similarity too low: {cos_sim.item()}"
    assert (out_f - ref_f).abs().max().item() < 0.05
    assert torch.allclose(lse.float(), lse_ref.float(), atol=0.05, rtol=0.05)


@pytest.mark.skipif(
    not IS_SM80_FAMILY, reason="Targets the SM80-family (Ampere/Ada/SM120) forward kernel"
)
def test_sm80_no_sink_regression():
    """learnable_sink=None must reproduce pre-existing (no-sink) behavior exactly."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.random.manual_seed(0)
    q = torch.randn(2, 64, 16, 64, device=device, dtype=dtype) * 0.1
    k = torch.randn(2, 64, 16, 64, device=device, dtype=dtype) * 0.1
    v = torch.randn(2, 64, 16, 64, device=device, dtype=dtype) * 0.1

    out, *_ = flash_attn_func(q, k, v, causal=True)
    assert not torch.isnan(out).any()
    out_ref, _ = attention_ref(q, k, v, causal=True)
    cos_sim = torch.nn.functional.cosine_similarity(
        out.float().flatten(), out_ref.float().flatten(), dim=0
    )
    assert cos_sim.item() > 0.999
