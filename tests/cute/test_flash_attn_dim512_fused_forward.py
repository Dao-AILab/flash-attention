"""Fused D512 forward coverage for independently packed query and KV tensors."""

from itertools import accumulate

import pytest
import torch

from flash_attn.cute import flash_fwd_sm100_hd512 as fused
from flash_attn.cute.interface import _flash_attn_fwd
from test_flash_attn_dim512 import check, reference
from test_flash_attn_dim512 import make_case, verify


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires SM100",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("packed_q,packed_k", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("ratio", [1, 4])
def test_fused_independent_varlen(dtype, causal, packed_q, packed_k, ratio, monkeypatch):
    """Ragged stores must preserve adjacent sequences, including empty KV rows."""
    torch.manual_seed(513)
    monkeypatch.setattr(fused, "_forward_cache", {})
    lq, lk = [1, 129, 193, 0, 5], [17, 131, 65, 9, 0]
    hq, hk = 2 * ratio, 2
    qshape = (sum(lq),) if packed_q else (len(lq), max(lq))
    kshape = (sum(lk),) if packed_k else (len(lk), max(lk))
    q = torch.randn(*qshape, hq, 512, device="cuda", dtype=dtype)
    k = torch.randn(*kshape, hk, 512, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cuq = torch.tensor([0, *accumulate(lq)], device="cuda", dtype=torch.int32) if packed_q else None
    cuk = torch.tensor([0, *accumulate(lk)], device="cuda", dtype=torch.int32) if packed_k else None
    usedq = torch.tensor(lq, device="cuda", dtype=torch.int32)
    usedk = torch.tensor(lk, device="cuda", dtype=torch.int32)
    supplied = torch.full_like(q, 11)
    out, lse, *_ = _flash_attn_fwd(
        q, k, v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
        seqused_q=usedq, seqused_k=usedk,
        max_seqlen_q=max(lq), max_seqlen_k=max(lk),
        causal=causal, return_lse=True, out=supplied,
    )
    assert out is supplied
    assert len(fused._forward_cache) == 1, "The fused path must handle all packing combinations"
    qparts = q.split(lq) if packed_q else [q[i, :n] for i, n in enumerate(lq)]
    kparts = k.split(lk) if packed_k else [k[i, :n] for i, n in enumerate(lk)]
    vparts = v.split(lk) if packed_k else [v[i, :n] for i, n in enumerate(lk)]
    start = 0
    for i, (qi, ki, vi) in enumerate(zip(qparts, kparts, vparts)):
        if not lq[i]:
            continue
        ref, ref_lse = reference(qi[None], ki[None], vi[None], causal=causal)
        actual = out[start:start + lq[i]] if packed_q else out[i, :lq[i]]
        actual_lse = lse[:, start:start + lq[i]] if packed_q else lse[i, :, :lq[i]]
        check(actual, ref[0], dtype)
        torch.testing.assert_close(actual_lse.double(), ref_lse[0], atol=0.003, rtol=0.003)
        start += lq[i]
    torch.cuda.synchronize()


@pytest.mark.parametrize("causal", [False, True])
def test_fused_varlen_dynamic_cache_reuse(causal, monkeypatch):
    """Packed token and batch counts stay dynamic in the fused compile key."""
    monkeypatch.setattr(fused, "_forward_cache", {})
    for lq, lk in [([129], [137]), ([193, 17], [65, 29]), ([1, 7, 131], [9, 0, 129])]:
        q = torch.randn(sum(lq), 4, 512, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(sum(lk), 2, 512, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cuq = torch.tensor([0, *accumulate(lq)], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, *accumulate(lk)], device="cuda", dtype=torch.int32)
        out, *_ = _flash_attn_fwd(q, k, v, cu_seqlens_q=cuq, cu_seqlens_k=cuk, causal=causal)
        refs = [reference(qi[None], ki[None], vi[None], causal=causal)[0][0]
                for qi, ki, vi in zip(q.split(lq), k.split(lk), v.split(lk))]
        check(out, torch.cat(refs), torch.bfloat16)
        assert len(fused._forward_cache) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("window,causal,softcap", [
    ((33, 7), False, None), ((33, None), True, None),
    ((None, 7), False, None), ((33, 7), False, 8.0),
    ((None, None), True, 8.0), ((None, None), False, 8.0),
])
def test_fused_window_softcap(dtype, packed, window, causal, softcap, monkeypatch):
    torch.manual_seed(514)
    monkeypatch.setattr(fused, "_forward_cache", {})
    q = torch.randn(2, 193, 4, 512, device="cuda", dtype=dtype)
    k = torch.randn(2, 513, 1, 512, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    lq, lk = [193, 129], [513, 259]
    if packed:
        q = torch.cat([q[i, :n] for i, n in enumerate(lq)])
        k = torch.cat([k[i, :n] for i, n in enumerate(lk)])
        v = torch.cat([v[i, :n] for i, n in enumerate(lk)])
        cuq = torch.tensor([0, *accumulate(lq)], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, *accumulate(lk)], device="cuda", dtype=torch.int32)
    else:
        lq, lk = [193, 193], [513, 513]
        cuq = cuk = None
    out, lse, *_ = _flash_attn_fwd(
        q, k, v, cu_seqlens_q=cuq, cu_seqlens_k=cuk, return_lse=True,
        causal=causal, window_size_left=window[0], window_size_right=window[1],
        softcap=softcap,
    )
    assert len(fused._forward_cache) == 1
    qparts = q.split(lq) if packed else q.unbind(0)
    kparts = k.split(lk) if packed else k.unbind(0)
    vparts = v.split(lk) if packed else v.unbind(0)
    start = 0
    ref_window = tuple(-1 if x is None else x for x in window)
    for i, (qi, ki, vi) in enumerate(zip(qparts, kparts, vparts)):
        ref, ref_lse = reference(qi[None], ki[None], vi[None], causal=causal,
                                 window=ref_window, softcap=softcap or 0.0)
        actual = out[start:start + lq[i]] if packed else out[i]
        actual_lse = lse[:, start:start + lq[i]] if packed else lse[i]
        check(actual, ref[0], dtype)
        torch.testing.assert_close(actual_lse.double(), ref_lse[0], atol=0.003, rtol=0.003)
        start += lq[i]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [16, 32, 64, 96, 128, 256])
@pytest.mark.parametrize("mode", ["dense", "packed", "window", "softcap"])
def test_fused_paged(dtype, page_size, mode, monkeypatch):
    monkeypatch.setattr(fused, "_forward_cache", {})
    verify(
        *make_case(dtype, page_size, 65), packed=mode == "packed",
        window=(33, 7) if mode == "window" else (-1, -1),
        softcap=8.0 if mode == "softcap" else 0.0,
    )
    assert len(fused._forward_cache) == 1, "Paged forward must share the QK/softmax work"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.017, 0.13])
def test_fused_softcap_saturated_scores(dtype, scale, monkeypatch):
    """Exercise saturated tanh values and fully masked rows with custom scales."""
    torch.manual_seed(515)
    monkeypatch.setattr(fused, "_forward_cache", {})
    q = torch.randn(1, 193, 4, 512, device="cuda", dtype=dtype) * 8
    k = torch.randn(1, 65, 2, 512, device="cuda", dtype=dtype) * 8
    v = torch.randn_like(k)
    out, lse, *_ = _flash_attn_fwd(
        q, k, v, causal=True, softcap=4.0, softmax_scale=scale, return_lse=True,
    )
    ref, ref_lse = reference(q, k, v, causal=True, softcap=4.0, scale=scale)
    check(out, ref, dtype)
    torch.testing.assert_close(lse.double(), ref_lse, atol=0.003, rtol=0.003)
    assert len(fused._forward_cache) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_legacy_unbounded_window_cache(dtype, packed, causal, monkeypatch):
    """Legacy unbounded-window spellings must not silently select V256 fallback."""
    monkeypatch.setattr(fused, "_forward_cache", {})
    shape = (258,) if packed else (2, 129)
    q = torch.randn(*shape, 4, 512, device="cuda", dtype=dtype)
    k = torch.randn(*shape, 1, 512, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, 129, 258], device="cuda", dtype=torch.int32) if packed else None
    expected = None
    for bounds in [(None, None), (-1, -1), (-257, -257)]:
        out, lse, *_ = _flash_attn_fwd(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, causal=causal,
            window_size_left=bounds[0], window_size_right=bounds[1], return_lse=True,
        )
        assert len(fused._forward_cache) == 1
        if expected is None:
            expected = out, lse
        else:
            torch.testing.assert_close(out, expected[0], atol=0, rtol=0)
            torch.testing.assert_close(lse, expected[1], atol=0, rtol=0)
