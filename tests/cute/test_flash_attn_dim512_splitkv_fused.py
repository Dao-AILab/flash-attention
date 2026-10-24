"""Fused D512 SplitKV partials, empty splits, packed GQA, and backward parity."""
from itertools import accumulate

import pytest
import torch

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute import flash_fwd_sm100_hd512 as fused
from flash_attn.cute.interface import _flash_attn_fwd
from test_flash_attn_dim512 import check, reference, make_case, verify


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires SM100",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [1, 2, 4])
@pytest.mark.parametrize("layout", ["dense", "packed", "qpacked", "kpacked", "used"])
@pytest.mark.parametrize("feature", ["full", "causal", "window", "softcap"])
@pytest.mark.parametrize("pack_gqa", [False, True])
def test_splitkv_independent_layouts(dtype, splits, layout, feature, pack_gqa, monkeypatch):
    torch.manual_seed(516)
    monkeypatch.setattr(fused, "_forward_cache", {})
    lq, lk = [1, 65, 0, 3], [129, 17, 9, 0]
    if layout == "dense":
        lq, lk = [65] * 4, [129] * 4
    qpacked, kpacked = layout in ("packed", "qpacked"), layout in ("packed", "kpacked")
    q = torch.randn(*( (sum(lq),) if qpacked else (4, max(lq)) ), 4, 512,
                    device="cuda", dtype=dtype)
    k = torch.randn(*( (sum(lk),) if kpacked else (4, max(lk)) ), 1, 512,
                    device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cuq = torch.tensor([0, *accumulate(lq)], device="cuda", dtype=torch.int32) if qpacked else None
    cuk = torch.tensor([0, *accumulate(lk)], device="cuda", dtype=torch.int32) if kpacked else None
    # Independent used lengths deliberately leave nonzero storage behind inactive rows.
    usedq = torch.tensor(lq, device="cuda", dtype=torch.int32) if layout != "dense" else None
    usedk = torch.tensor(lk, device="cuda", dtype=torch.int32) if layout != "dense" else None
    window = (13, 7) if feature == "window" else (None, None)
    softcap = 2.5 if feature == "softcap" else None
    causal = feature == "causal"
    out, lse, *_ = _flash_attn_fwd(
        q, k, v, cu_seqlens_q=cuq, cu_seqlens_k=cuk, seqused_q=usedq, seqused_k=usedk,
        max_seqlen_q=max(lq), max_seqlen_k=max(lk), num_splits=splits,
        pack_gqa=pack_gqa, causal=causal, window_size_left=window[0],
        window_size_right=window[1], softcap=softcap, return_lse=True,
    )
    assert len(fused._forward_cache) == 1
    qparts = q.split(lq) if qpacked else [q[i, :n] for i, n in enumerate(lq)]
    kparts = k.split(lk) if kpacked else [k[i, :n] for i, n in enumerate(lk)]
    vparts = v.split(lk) if kpacked else [v[i, :n] for i, n in enumerate(lk)]
    offset = 0
    for i, (qi, ki, vi) in enumerate(zip(qparts, kparts, vparts)):
        if not lq[i]:
            continue
        ref, ref_lse = reference(qi[None], ki[None], vi[None], causal=causal,
                                 window=tuple(-1 if x is None else x for x in window),
                                 softcap=softcap or 0.0)
        actual = out[offset:offset + lq[i]] if qpacked else out[i, :lq[i]]
        actual_lse = lse[:, offset:offset + lq[i]] if qpacked else lse[i, :, :lq[i]]
        check(actual, ref[0], dtype)
        torch.testing.assert_close(actual_lse.double(), ref_lse[0], atol=0.003, rtol=0.003)
        offset += lq[i]
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize("page_size", [16, 128])
@pytest.mark.parametrize("feature", ["full", "causal", "window", "softcap"])
def test_splitkv_paged(dtype, splits, page_size, feature, monkeypatch):
    monkeypatch.setattr(fused, "_forward_cache", {})
    verify(*make_case(dtype, page_size, 65), splits=splits,
           causal=feature == "causal", window=(13, 7) if feature == "window" else (-1, -1),
           softcap=2.5 if feature == "softcap" else 0.0)
    assert len(fused._forward_cache) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("ratio", [1, 4])
def test_splitkv_training_backward(dtype, splits, causal, packed, ratio):
    torch.manual_seed(517)
    shape = (129,) if packed else (1, 129)
    kshape = (65,) if packed else (1, 65)
    q = torch.randn(*shape, 4, 512, device="cuda", dtype=dtype, requires_grad=True)
    kvheads = 4 // ratio
    k = torch.randn(*kshape, kvheads, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    if packed:
        out, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=torch.tensor([0, 129], device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 65], device="cuda", dtype=torch.int32),
            num_splits=splits, causal=causal,
        )
    else:
        out, _ = flash_attn_func(q, k, v, num_splits=splits, causal=causal)
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    ref, _ = reference(qr.reshape(1, 129, 4, 512), kr.reshape(1, 65, kvheads, 512),
                       vr.reshape(1, 65, kvheads, 512), causal=causal)
    ref = ref.reshape_as(out)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), grad)
    expected = torch.autograd.grad(ref, (qr, kr, vr), grad.double())
    check(out, ref, dtype)
    for result, exact in zip(actual, expected):
        check(result, exact, dtype)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize("layout,feature", [
    ("dense", "full"), ("packed", "causal"), ("paged16", "window"), ("paged128", "softcap")
])
def test_splitkv_fp8(dtype, splits, layout, feature, monkeypatch):
    import test_flash_attn_dim512_fp8_fused as fp8_tests
    original = fp8_tests._flash_attn_fwd

    def split_forward(*args, **kwargs):
        kwargs.update(num_splits=splits, pack_gqa=False)
        return original(*args, **kwargs)
    monkeypatch.setattr(fp8_tests, "_flash_attn_fwd", split_forward)
    fp8_tests._run_fp8_case(dtype, layout, 4, feature, (True, True, True), monkeypatch)


@pytest.mark.parametrize("ratio", [2, 8, 64, 128])
@pytest.mark.parametrize("layout", ["dense", "qpacked", "paged16", "paged128"])
@pytest.mark.parametrize("splits", [1, 4])
def test_pack_gqa_decode(ratio, layout, splits, monkeypatch):
    """Large MQA head groups exercise the packed-varlen token scheduler."""
    monkeypatch.setattr(fused, "_forward_cache", {})
    torch.manual_seed(518)
    dtype = torch.bfloat16
    q = torch.randn(2, 1, ratio, 512, device="cuda", dtype=dtype)
    kwargs = dict(causal=True, num_splits=splits, pack_gqa=True, return_lse=True)
    packed = layout == "qpacked"
    if layout.startswith("paged"):
        _, k, v, table, used, lengths = make_case(dtype, int(layout[5:]), 1, hk=1)
        kparts = [k[table[i].long()].flatten(0, 1)[:n] for i, n in enumerate(lengths)]
        vparts = [v[table[i].long()].flatten(0, 1)[:n] for i, n in enumerate(lengths)]
        kwargs.update(page_table=table, seqused_k=used)
    else:
        k = torch.randn(2, 259, 1, 512, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        kparts, vparts = k.unbind(0), v.unbind(0)
    if packed:
        q = q.flatten(0, 1)
        kwargs.update(cu_seqlens_q=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32))
    out, lse, *_ = _flash_attn_fwd(q, k, v, **kwargs)
    for i, (ki, vi) in enumerate(zip(kparts, vparts)):
        qi = q[i:i + 1] if packed else q[i]
        ref, ref_lse = reference(qi[None], ki[None], vi[None], causal=True)
        actual = out[i:i + 1] if packed else out[i]
        actual_lse = lse[:, i:i + 1] if packed else lse[i]
        check(actual, ref[0], dtype)
        torch.testing.assert_close(actual_lse.double(), ref_lse[0], atol=0.003, rtol=0.003)
    assert len(fused._forward_cache) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_splitkv_optional_lse(dtype, splits, packed, monkeypatch):
    monkeypatch.setattr(fused, "_forward_cache", {})
    q = torch.randn(1, 3, 4, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
    k = torch.randn(1, 259, 1, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
    v = torch.randn_like(k.to(torch.bfloat16)).to(dtype)
    kwargs = dict(num_splits=splits, pack_gqa=True)
    if packed:
        q = q[0]
        kwargs["cu_seqlens_q"] = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    first, first_lse, *_ = _flash_attn_fwd(q, k, v, return_lse=False, **kwargs)
    second, second_lse, *_ = _flash_attn_fwd(q, k, v, return_lse=True, **kwargs)
    assert first_lse is None and second_lse is not None
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert len(fused._forward_cache) == 1


@pytest.mark.parametrize("pack_gqa,window,expected_splits", [
    (True, (None, None), 2), (False, (None, None), 1), (True, (33, None), 1),
    (None, (None, None), 2), (None, (33, None), 1),
])
def test_auto_splits_native_occupancy_cache(pack_gqa, window, expected_splits, monkeypatch):
    from flash_attn.cute import interface
    monkeypatch.setattr(fused, "_forward_cache", {})
    sms_calls = []
    monkeypatch.setattr(interface, "get_num_sms_for_selection",
                        lambda device, arch: sms_calls.append((device, arch)) or 8)
    q = torch.randn(1, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1024, 1, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qhint = torch.tensor(1, device="cuda", dtype=torch.int32)
    khint = torch.tensor(1024, device="cuda", dtype=torch.int32)
    original_item = torch.Tensor.item

    def guarded_item(tensor, *args):
        assert tensor is not qhint and tensor is not khint, "Planner read a GPU length scalar"
        return original_item(tensor, *args)
    monkeypatch.setattr(torch.Tensor, "item", guarded_item)
    kwargs = dict(pack_gqa=pack_gqa, return_lse=True, max_seqlen_q=qhint,
                  max_seqlen_k=khint, window_size_left=window[0], window_size_right=window[1])
    automatic, alse, *_ = _flash_attn_fwd(q, k, v, num_splits=0, **kwargs)
    if pack_gqa is None:
        kwargs["pack_gqa"] = True
    explicit, else_, *_ = _flash_attn_fwd(q, k, v, num_splits=expected_splits, **kwargs)
    assert len(sms_calls) == 1
    assert len(fused._forward_cache) == 1, "auto and the chosen explicit plan must reuse compiled code"
    torch.testing.assert_close(automatic, explicit, atol=0, rtol=0)
    torch.testing.assert_close(alse, else_, atol=0, rtol=0)


@pytest.mark.parametrize("packed,total_q,batch,qhint,heads,kvheads,khint,expected", [
    (True, 1, 1, 1, 128, 1, 8192, 16),
    (True, 2, 2, 1, 128, 1, 8192, 16),
    (False, 16384, 2, 8192, 8, 2, 8192, 1),
    (True, 100, 100, None, 128, 1, 8192, 1),
    (True, 2, 2, 1, 128, 1, 512, 1),
])
def test_auto_splits_physical_two_cta_counts(packed, total_q, batch, qhint,
                                            heads, kvheads, khint, expected):
    splits, actual_pack = fused.select_sm100_d512_config(
        num_splits=0, pack_gqa=packed, batch_size=batch, total_q=total_q,
        num_head=heads, num_head_kv=kvheads, seqlen_q_hint=qhint,
        seqlen_k_hint=khint, window_size_left=None, window_size_right=None, num_sms=148,
    )
    assert splits == expected
    assert actual_pack is packed


@pytest.mark.parametrize("batch,ctas", [(1, 2), (2, 4)])
def test_auto_splits_passes_physical_occupancy_to_native(batch, ctas, monkeypatch):
    from flash_attn.cute import interface
    observed = []
    monkeypatch.setattr(interface, "num_splits_heuristic", lambda *args: observed.append(args) or 1)
    fused.select_sm100_d512_config(
        num_splits=0, pack_gqa=None, batch_size=batch, total_q=batch,
        num_head=128, num_head_kv=1, seqlen_q_hint=1, seqlen_k_hint=8192,
        window_size_left=None, window_size_right=None, num_sms=148,
    )
    assert observed == [(ctas, 148, 64, 16)]


def test_auto_splits_gqa_keeps_native_occupancy_policy(monkeypatch):
    from flash_attn.cute import interface
    native = interface.num_splits_heuristic
    observed = []

    def record(*args):
        observed.append(args)
        return native(*args)
    monkeypatch.setattr(interface, "num_splits_heuristic", record)
    splits, packed = fused.select_sm100_d512_config(
        num_splits=0, pack_gqa=None, batch_size=1, total_q=1,
        num_head=128, num_head_kv=2, seqlen_q_hint=1, seqlen_k_hint=8192,
        window_size_left=None, window_size_right=None, num_sms=148,
    )
    assert observed == [(4, 148, 64, 128)]
    assert splits == 37 and packed


@pytest.mark.parametrize("is_fp8,kvheads,ratio,splits,requested,expected", [
    (True, 1, 16, 1, None, False), (True, 1, 32, 1, None, False),
    (True, 1, 32, 0, None, True), (False, 1, 32, 1, None, True),
    (True, 2, 32, 1, None, True), (True, 1, 128, 1, None, True),
    (True, 1, 32, 1, True, True), (True, 1, 32, 1, False, False),
])
def test_default_pack_policy_precision_boundaries(is_fp8, kvheads, ratio, splits, requested, expected):
    _, packed = fused.select_sm100_d512_config(
        num_splits=splits, pack_gqa=requested, batch_size=1, total_q=2,
        num_head=ratio * kvheads, num_head_kv=kvheads, seqlen_q_hint=2,
        seqlen_k_hint=8192, window_size_left=None, window_size_right=None,
        num_sms=148, is_fp8=is_fp8,
    )
    assert packed is expected


@pytest.mark.parametrize("q_length,ratio", [(2, 32), (4, 16)])
@pytest.mark.parametrize("override,expected_pack", [
    ({}, False), ({"pack_gqa": True}, True), ({"pack_gqa": False}, False),
    ({"num_splits": 0}, True), ({"is_bf16": False}, True),
    ({"is_causal": False}, True), ({"is_plain_dense": False}, True),
    ({"batch_size": 2}, True), ({"num_head_kv": 1}, True),
    ({"window_size_left": 33}, True),
    ({"seqlen_k_hint": 4096}, True),
])
def test_default_bf16_gqa_decode_policy_boundaries(q_length, ratio, override, expected_pack):
    kwargs = dict(
        num_splits=1, pack_gqa=None, batch_size=1, total_q=q_length,
        num_head=ratio * 2, num_head_kv=2, seqlen_q_hint=q_length,
        seqlen_k_hint=8192, window_size_left=None, window_size_right=None,
        num_sms=148, is_bf16=True, is_causal=True, is_plain_dense=True,
    )
    kwargs.update(override)
    # A changed KV-head count still describes the same GQA ratio.
    kwargs["num_head"] = ratio * kwargs["num_head_kv"]
    _, packed = fused.select_sm100_d512_config(**kwargs)
    assert packed is expected_pack


@pytest.mark.parametrize("q_length,ratio", [(2, 32), (4, 16)])
def test_default_bf16_gqa_decode_route_cache(q_length, ratio, monkeypatch):
    """The default measured plan reuses explicit False and keeps True available."""
    monkeypatch.setattr(fused, "_forward_cache", {})
    original = fused.forward_sm100_d512
    chosen = []

    def record(*args, **kwargs):
        chosen.append(kwargs["pack_gqa"])
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "forward_sm100_d512", record)
    q = torch.randn(1, q_length, ratio * 2, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8192, 2, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = dict(causal=True, num_splits=1, return_lse=True)
    automatic, alse, *_ = _flash_attn_fwd(q, k, v, pack_gqa=None, **kwargs)
    explicit, else_, *_ = _flash_attn_fwd(q, k, v, pack_gqa=False, **kwargs)
    assert chosen == [False, False]
    assert len(fused._forward_cache) == 1
    torch.testing.assert_close(automatic, explicit, atol=0, rtol=0)
    torch.testing.assert_close(alse, else_, atol=0, rtol=0)
    packed, plse, *_ = _flash_attn_fwd(q, k, v, pack_gqa=True, **kwargs)
    assert chosen == [False, False, True]
    assert len(fused._forward_cache) == 2
    ref, ref_lse = reference(q, k, v, causal=True)
    for out, lse in ((automatic, alse), (packed, plse)):
        check(out, ref, torch.bfloat16)
        torch.testing.assert_close(lse.double(), ref_lse, atol=0.003, rtol=0.003)


@pytest.mark.parametrize("dtype,expected_pack", [
    (torch.bfloat16, True), (torch.float8_e4m3fn, False), (torch.float8_e5m2, False),
])
def test_default_pack_policy_gpu_route_cache(dtype, expected_pack, monkeypatch):
    monkeypatch.setattr(fused, "_forward_cache", {})
    original = fused.forward_sm100_d512
    chosen = []

    def record(*args, **kwargs):
        chosen.append(kwargs["pack_gqa"])
        return original(*args, **kwargs)
    monkeypatch.setattr(fused, "forward_sm100_d512", record)
    q = torch.randn(1, 2, 32, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
    k = torch.randn(1, 257, 1, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
    v = torch.randn_like(k.to(torch.bfloat16)).to(dtype)
    implicit, ilse, *_ = _flash_attn_fwd(q, k, v, pack_gqa=None, return_lse=True)
    explicit, else_, *_ = _flash_attn_fwd(q, k, v, pack_gqa=expected_pack, return_lse=True)
    assert chosen == [expected_pack, expected_pack]
    assert len(fused._forward_cache) == 1
    torch.testing.assert_close(implicit, explicit, atol=0, rtol=0)
    torch.testing.assert_close(ilse, else_, atol=0, rtol=0)
