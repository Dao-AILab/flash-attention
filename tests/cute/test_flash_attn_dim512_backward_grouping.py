"""Grouped D512 backward across packed, mixed and used-length layouts."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from flash_attn.cute import flash_bwd_sm100_hd512 as native
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from flash_attn.cute.testing import attention_ref
from test_flash_attn_dim512 import check_grouped_error, reference


@pytest.mark.parametrize("layout", ["dense", "packed_q", "packed_k", "packed_both"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("ratio", [2, 4, 8])
def test_metadata_head_group_policy(monkeypatch, layout, causal, ratio):
    """Selection reads tensor shapes, never GPU metadata or a maximum value."""
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=148),
    )
    packed_q, packed_k = (
        layout in ("packed_q", "packed_both"),
        layout in ("packed_k", "packed_both"),
    )
    q = SimpleNamespace(
        shape=(8192, ratio * 4, 512) if packed_q else (8, 1024, ratio * 4, 512),
        device="cuda",
    )
    k = SimpleNamespace(shape=(8192, 4, 512) if packed_k else (8, 1024, 4, 512))
    # Objects deliberately have no item()/tolist()/shape; touching values fails.
    cuq, cuk = object() if packed_q else None, object() if packed_k else None
    group = native._select_head_group(q, k, causal, cuq, cuk, object(), object(), 1024, 1024)
    assert group in (2, 4)
    assert ratio % group == 0


@pytest.mark.parametrize("ratio", [1, 3])
def test_metadata_head_group_keeps_indivisible_heads(ratio):
    q = SimpleNamespace(shape=(128, ratio * 2, 512))
    k = SimpleNamespace(shape=(128, 2, 512))
    assert native._select_head_group(q, k, True, object(), object(), None, None, 128, 128) == 1


def test_local_head_group_uses_traversal_width(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=148),
    )
    q = SimpleNamespace(shape=(1, 8192, 16, 512), device="cuda")
    k = SimpleNamespace(shape=(1, 8192, 4, 512))
    assert native._select_head_group(q, k, False, None, None, None, None, 8192, 8192) == 1
    assert native._select_head_group(q, k, False, None, None, None, None, 8192, 8192, 128, 128) == 4


def test_short_causal_group_retains_partial_wave(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=148),
    )
    q = SimpleNamespace(shape=(8, 128, 32, 512), device="cuda")
    k = SimpleNamespace(shape=(8, 128, 8, 512))
    assert native._select_head_group(q, k, True, None, None, None, None, 128, 128) == 4


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("ratio", [2, 4, 8])
def test_medium_packed_head_group_policy(monkeypatch, causal, ratio):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=148),
    )
    q = SimpleNamespace(shape=(3065, 8, 512), device="cuda")
    k = SimpleNamespace(shape=(3065, 8 // ratio, 512))
    # Three possible members make these maxima compatible with the same
    # physical token count.  Opaque metadata forbids host length readbacks.
    for maxq, expected in ((1536, 2), (1537, 1), (2048, 1), (2049, 2 if causal else 1)):
        if ratio == 2 and maxq <= 2048:
            expected = 2
        assert (
            native._select_head_group(
                q, k, causal, object(), object(), object(), object(), maxq, 2048
            )
            == expected
        )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("layout", "tokens", "window", "softcap", "sms", "full_group", "causal_group"),
    [
        ("packed", 3456, (None, None), 0.0, 148, 1, 1),
        ("packed", 3457, (None, None), 0.0, 148, 4, 2),
        ("packed", 3065, (None, None), 5.0, 148, 2, 2),
        ("packed", 3065, (None, 0), 0.0, 148, 2, 2),
        ("packed_q", 3065, (None, None), 0.0, 148, 4, 2),
        ("dense", 3065, (None, None), 0.0, 148, 4, 2),
        ("packed", 3065, (None, None), 0.0, 120, 4, 2),
    ],
)
def test_medium_packed_group_preserves_distinct_paths(
    monkeypatch, causal, layout, tokens, window, softcap, sms, full_group, causal_group
):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=sms),
    )
    packed_q, packed_k = layout != "dense", layout == "packed"
    q = SimpleNamespace(shape=(tokens, 8, 512) if packed_q else (2, 2048, 8, 512), device="cuda")
    k = SimpleNamespace(shape=(tokens, 2, 512) if packed_k else (2, 2048, 2, 512))
    assert native._select_head_group(
        q,
        k,
        causal,
        object() if packed_q else None,
        object() if packed_k else None,
        None,
        None,
        2048,
        2048,
        *window,
        softcap,
    ) == (causal_group if causal else full_group)


@pytest.mark.parametrize(
    ("causal", "window", "maxq", "maxk", "heads", "expected"),
    [
        (False, (None, None), 4096, 4096, 8, True),
        (True, (None, None), 4096, 4096, 8, False),
        (False, (128, 128), 4096, 4096, 8, False),
        (False, (None, None), 2048, 2048, 32, False),
        (False, (None, None), 8192, 4096, 8, False),
    ],
)
def test_head_major_policy_uses_launch_maxima(causal, window, maxq, maxk, heads, expected):
    assert native._select_head_major(causal, window, maxq, maxk, heads) == expected


sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires SM100",
)


@sm100
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("layout", "ratio", "head_group"),
    [
        ("dense", 4, 4),
        ("packed_q", 4, 2),
        ("packed_k", 4, 4),
        ("packed_both", 2, 2),
        ("packed_both", 8, 4),
        ("packed_both", 8, 8),
        ("packed_both_major", 8, 4),
    ],
)
def test_grouped_metadata_backward(monkeypatch, dtype, causal, layout, ratio, head_group):
    """Independent offsets, empty rows, softcap, window, LSE and output views compose."""
    monkeypatch.setattr(native, "_select_head_group", lambda *args: head_group)
    if layout.endswith("_major"):
        monkeypatch.setattr(native, "_select_head_major", lambda *args: True)
        layout = layout.removesuffix("_major")
    torch.manual_seed(20261024)
    packed_q, packed_k = (
        layout in ("packed_q", "packed_both"),
        layout in ("packed_k", "packed_both"),
    )
    qs = [137, 19, 73] if packed_q else [137] * 3
    ks = [101, 41, 11] if packed_k else [101] * 3
    uq, uk = [129, 0, 65], [97, 33, 0]
    q_shape = (sum(qs),) if packed_q else (3, qs[0])
    k_shape = (sum(ks),) if packed_k else (3, ks[0])
    q = torch.randn(*q_shape, ratio, 512, device="cuda", dtype=dtype)
    k, v = [torch.randn(*k_shape, 1, 512, device="cuda", dtype=dtype) for _ in range(2)]
    cuq = torch.tensor([0, *accumulate(qs)], device="cuda", dtype=torch.int32) if packed_q else None
    cuk = torch.tensor([0, *accumulate(ks)], device="cuda", dtype=torch.int32) if packed_k else None
    opts = dict(
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        seqused_q=torch.tensor(uq, device="cuda", dtype=torch.int32),
        seqused_k=torch.tensor(uk, device="cuda", dtype=torch.int32),
        max_seqlen_q=max(qs),
        max_seqlen_k=max(ks),
        causal=causal,
        softcap=5.0,
        window_size_left=32,
        window_size_right=16,
    )
    out, lse, *_ = _flash_attn_fwd(q, k, v, return_lse=True, **opts)
    dout, dlse = torch.randn_like(out), torch.randn_like(lse)
    # A strided last axis goes through the real output-buffer adaptation path.
    buffers = [
        torch.full((*t.shape[:-1], 1024), 7, device="cuda", dtype=dtype)[..., ::2]
        for t in (q, k, v)
    ]
    actual = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        dlse=dlse,
        dq=buffers[0],
        dk=buffers[1],
        dv=buffers[2],
        **opts,
    )
    assert all(a is b for a, b in zip(actual, buffers))
    qr, kr, vr = [t.double().requires_grad_() for t in (q, k, v)]
    qe, ke, ve = [t.detach().clone().requires_grad_() for t in (q, k, v)]

    def rows(t, sizes, packed):
        return t.split(sizes) if packed else t.unbind(0)

    exact_loss = sum(t.sum() * 0 for t in (qr, kr, vr))
    eager_loss = sum(t.float().sum() * 0 for t in (qe, ke, ve))
    exact_rows = zip(rows(qr, qs, packed_q), rows(kr, ks, packed_k), rows(vr, ks, packed_k))
    eager_rows = list(zip(rows(qe, qs, packed_q), rows(ke, ks, packed_k), rows(ve, ks, packed_k)))
    do_rows = rows(dout, qs, packed_q)
    dlse_rows = dlse.split(qs, -1) if packed_q else dlse.unbind(0)
    for i, (a, b, c) in enumerate(exact_rows):
        nq, nk = uq[i], uk[i]
        if not nq or not nk:
            continue
        ref, rlse = reference(
            a[None, :nq],
            b[None, :nk],
            c[None, :nk],
            causal=causal,
            window=(32, 16),
            softcap=5.0,
        )
        do, dl = do_rows[i][:nq], dlse_rows[i][:, :nq]
        finite = torch.isfinite(rlse[0])
        exact_loss = (
            exact_loss
            + (ref[0] * do.double()).sum()
            + (rlse[0][finite] * dl.double()[finite]).sum()
        )
        a, b, c = eager_rows[i]
        eo, _ = attention_ref(
            a[None, :nq],
            b[None, :nk],
            c[None, :nk],
            causal=causal,
            window_size=(32, 16),
            softcap=5.0,
            upcast=False,
            reorder_ops=True,
        )
        _, el = reference(
            a[None, :nq],
            b[None, :nk],
            c[None, :nk],
            causal=causal,
            window=(32, 16),
            softcap=5.0,
        )
        eager_loss = (
            eager_loss
            + (eo[0].float() * do.float()).sum()
            + (el[0][finite] * dl.double()[finite]).sum()
        )
    exact = torch.autograd.grad(exact_loss, (qr, kr, vr))
    eager = torch.autograd.grad(eager_loss, (qe, ke, ve))
    for name, result, expected, er in zip(("dq", "dk", "dv"), actual, exact, eager):
        check_grouped_error(name, result, expected, er, dtype)
    torch.cuda.synchronize()
