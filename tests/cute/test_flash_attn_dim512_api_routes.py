"""Equivalent unbounded API representations must retain native D512 routing."""

from functools import wraps

import pytest
import torch

from flash_attn.cute import flash_fwd_sm100_hd512 as fused
from flash_attn.cute.interface import _flash_attn_fwd, flash_attn_func, flash_attn_varlen_func


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires SM100",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("api", ["internal", "public"])
@pytest.mark.parametrize("packed", [False, True])
def test_unbounded_api_forms_share_fused_dispatch_and_cache(dtype, causal, api, packed, monkeypatch):
    """(-1,-1) is the legacy unbounded pair; one-sided signed bounds differ.

    This tests equivalent API forms, not a timing threshold. The baseline PR
    produces equal outputs but misses native fusion before normalization.
    """
    torch.manual_seed(20261024)
    q = torch.randn(2, 129, 8, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(2, 193, 2, 512, device="cuda", dtype=dtype) for _ in range(2)]
    options = dict(causal=causal, return_lse=True)
    if packed:
        q = torch.cat((q[0], q[1, :65]))
        k, v = [torch.cat((tensor[0], tensor[1, :97])) for tensor in (k, v)]
        options.update(cu_seqlens_q=torch.tensor([0, 129, 194], device="cuda", dtype=torch.int32),
                       cu_seqlens_k=torch.tensor([0, 193, 290], device="cuda", dtype=torch.int32),
                       max_seqlen_q=129, max_seqlen_k=193)
    monkeypatch.setattr(fused, "_forward_cache", {})
    calls = []
    original = fused.forward_sm100_d512

    @wraps(original)
    def observed(*args, **kwargs):
        calls.append((kwargs.get("window_size_left"), kwargs.get("window_size_right")))
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "forward_sm100_d512", observed)

    def invoke(window):
        if api == "internal":
            return _flash_attn_fwd(q, k, v, window_size_left=window[0],
                                   window_size_right=window[1], **options)[:2]
        fn = flash_attn_varlen_func if packed else flash_attn_func
        return fn(q, k, v, window_size=window, **options)

    canonical = invoke((None, None))
    assert len(calls) == 1
    assert len(fused._forward_cache) == 1
    legacy = invoke((-1, -1))
    assert len(calls) == 2, "Legacy unbounded pair must reach the same fused forward"
    assert len(fused._forward_cache) == 1, "Equivalent masks must reuse the same compilation"
    for actual, expected in zip(legacy, canonical):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()
