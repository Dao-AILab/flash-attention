"""MMA operands above 256 KB of shared memory (sm_107 only).

sm_107 SMEM descriptors carry a 15-bit start address. Masking it to 14 bits wraps any K/V
stage placed above 256 KB onto low shared memory and silently corrupts the output.
"""

import pytest
import torch

import flash_attn.cute.flash_fwd_sm100 as fwd_sm100
from flash_attn.cute import interface
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.interface import flash_attn_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs the sm_107 shared memory capacity (more than 256 KB per CTA)",
)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
def test_hd256_kv_stages_above_256kb(monkeypatch, causal):
    setup = fwd_sm100.FlashAttentionForwardSm100._setup_attributes

    def setup_deep_kv(self):
        setup(self)
        # 64 KB of Q plus 8 x 32 KB K/V stages: the last stages start above 256 KB.
        self.kv_stage = 8

    monkeypatch.setattr(fwd_sm100.FlashAttentionForwardSm100, "_setup_attributes", setup_deep_kv)
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", get_jit_cache("fwd_sm107_deep_kv"))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2048, 4, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    q32, k32, v32 = (t.float().transpose(1, 2) for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(q32, k32, v32, is_causal=causal).transpose(1, 2)
    ref_bf16 = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, k, v)), is_causal=causal
    ).transpose(1, 2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / 16.0
    if causal:
        scores.masked_fill_(torch.ones(2048, 2048, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1)
    assert (out.float() - ref).abs().max() <= 2 * (ref_bf16.float() - ref).abs().max() + 1e-5
    assert (lse - lse_ref).abs().max() < 1e-3
