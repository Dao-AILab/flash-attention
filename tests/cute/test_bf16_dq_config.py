from dataclasses import replace

import pytest
import torch

from flash_attn.cute.config import BwdHeuristicInputs, select_bwd_config
from flash_attn.cute.interface import _flash_attn_bwd, flash_attn_func


def bwd_inputs(**changes):
    return BwdHeuristicInputs(
        device_arch=100, head_dim=128, head_dim_v=128, num_heads=4, num_heads_kv=4,
    )._replace(**changes)


@pytest.mark.parametrize("arch", [90, 100, 103, 110, 120])
def test_bf16_dq_default_remains_off(arch):
    assert not select_bwd_config(bwd_inputs(device_arch=arch)).bf16_dq


def blackwell_arch():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("Blackwell required")
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def test_bf16_dq_empty_gpu():
    arch = blackwell_arch()
    q = torch.empty(1, 0, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 129, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lse = torch.empty(1, 4, 0, device="cuda", dtype=torch.float32)
    config = replace(select_bwd_config(bwd_inputs(device_arch=arch)), bf16_dq=True)
    dq, dk, dv = _flash_attn_bwd(q, k, v, q, q, lse, config=config)
    assert dq.shape == q.shape
    assert torch.count_nonzero(dk) == torch.count_nonzero(dv) == 0


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 4])
@pytest.mark.parametrize("with_dlse", [False, True])
def test_bf16_dq_gpu(causal, kv_heads, with_dlse):
    arch = blackwell_arch()
    torch.manual_seed(7)
    q = torch.randn(1, 257, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 385, kv_heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    inputs = bwd_inputs(
        device_arch=arch, num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        num_heads_kv=kv_heads, causal=causal, max_seqlen_q=257, max_seqlen_k=385,
    )
    config = replace(select_bwd_config(inputs), bf16_dq=True, pack_gqa=False)
    out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True, bwd_config=config)
    dout = torch.randn_like(out)
    dlse = torch.randn_like(lse) if with_dlse else None
    actual = (torch.autograd.grad((out, lse), (q, k, v), (dout, dlse))
              if with_dlse else torch.autograd.grad(out, (q, k, v), dout))
    expected = _flash_attn_bwd(q, k, v, out, dout, lse, causal=causal, dlse=dlse,
                              config=replace(config, bf16_dq=False))
    for got, ref in zip(actual, expected):
        assert (got.float() - ref.float()).norm() / ref.float().norm() < 0.01
        torch.testing.assert_close(got, ref, atol=0.01, rtol=0.03)
