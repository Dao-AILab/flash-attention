from dataclasses import replace

import pytest
import torch

from flash_attn.cute.config import FwdHeuristicInputs, select_fwd_config
from flash_attn.cute.exp_approximation import should_use_exp_approximation
from flash_attn.cute.interface import _flash_attn_fwd


def fwd_inputs(**changes):
    return FwdHeuristicInputs(
        device_arch=100, num_sms=132, dtype="torch.bfloat16", head_dim=128, head_dim_v=128,
        num_heads=1, num_heads_kv=1, batch_size=1, total_q=16384, total_k=16384,
        max_seqlen_q=16384, max_seqlen_k=16384, requested_num_splits=1,
        requested_use_swapab=False,
    )._replace(**changes)


@pytest.mark.parametrize("arch", [90, 100, 103, 110, 120])
def test_packed_softmax_default_remains_off(arch):
    assert not select_fwd_config(fwd_inputs(device_arch=arch)).approximate_softmax


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("norm_ok", [False, True])
@pytest.mark.parametrize("seqlen,num_splits,kv_heads,varlen", [
    (128, 1, 1, False), (257, 1, 2, False), (2048, 2, 1, False),
    (16384, 1, 1, False), (257, 1, 1, True),
])
def test_packed_softmax_gpu(causal, norm_ok, seqlen, num_splits, kv_heads, varlen):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("Blackwell required")
    properties = torch.cuda.get_device_properties(0)
    torch.manual_seed(8)
    num_heads = 4 if kv_heads == 2 else 1
    q = torch.randn(1, seqlen, num_heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, seqlen + 1, kv_heads, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if not norm_ok:
        q.mul_(8)
    kwargs = {}
    if varlen:
        q, k, v = [t.flatten(0, 1) for t in (q, k, v)]
        kwargs = {
            "cu_seqlens_q": torch.tensor([0, seqlen // 2, seqlen], device="cuda", dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, (seqlen + 1) // 2, seqlen + 1], device="cuda", dtype=torch.int32),
            "max_seqlen_q": (seqlen + 1) // 2, "max_seqlen_k": (seqlen + 2) // 2,
        }
    inputs = fwd_inputs(
        device_arch=properties.major * 10 + properties.minor, causal=causal,
        num_sms=properties.multi_processor_count, num_heads=num_heads, num_heads_kv=kv_heads,
        total_q=seqlen, total_k=seqlen + 1, batch_size=2 if varlen else 1,
        max_seqlen_q=kwargs.get("max_seqlen_q", seqlen),
        max_seqlen_k=kwargs.get("max_seqlen_k", seqlen + 1),
        has_cu_seqlens_q=varlen, has_cu_seqlens_k=varlen, requested_num_splits=num_splits,
    )
    config = replace(select_fwd_config(inputs), use_s_ping_pong=False)
    assert bool(should_use_exp_approximation(q, k, v, 128 ** -0.5).item()) is norm_ok
    baseline, baseline_lse, *_ = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True,
                                               config=config, **kwargs)
    packed, packed_lse, *_ = _flash_attn_fwd(q, k, v, causal=causal, return_lse=True,
        config=replace(config, approximate_softmax=True), **kwargs)
    if norm_ok:
        assert (packed.float() - baseline.float()).norm() / baseline.float().norm() < 0.01
        torch.testing.assert_close(packed, baseline, atol=0.01, rtol=0.03)
        torch.testing.assert_close(packed_lse, baseline_lse, atol=0.02, rtol=0.002)
    else:
        torch.testing.assert_close(packed, baseline, atol=0, rtol=0)
        torch.testing.assert_close(packed_lse, baseline_lse, atol=0, rtol=0)
