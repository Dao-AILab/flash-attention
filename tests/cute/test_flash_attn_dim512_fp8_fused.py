"""Real FP8 D512 fusion, independent descales, and packed/paged boundaries."""

from itertools import accumulate, product

import pytest
import torch

from flash_attn.cute import flash_fwd_sm100_hd512 as fused
from flash_attn.cute.interface import _flash_attn_fwd
from test_flash_attn_dim512 import reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires SM100",
)


def _run_fp8_case(dtype, layout, ratio, feature, components, monkeypatch, *, sq=65, sk=193):
    torch.manual_seed(512809)
    monkeypatch.setattr(fused, "_forward_cache", {})
    batch, hk = 2, 1 if ratio == 8 else 2
    hq = hk * ratio
    q = torch.randn(batch, sq, hq, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
    k, v = [
        torch.randn(batch, sk, hk, 512, device="cuda", dtype=torch.bfloat16).to(dtype)
        for _ in range(2)
    ]
    scales = [
        torch.tensor(values, device="cuda", dtype=torch.float32)[:, :hk].contiguous()
        for values in (
            [[0.75, 1.25], [1.5, 0.5]],
            [[1.125, 0.625], [0.875, 1.375]],
            [[0.5, 1.75], [1.25, 0.875]],
        )
    ]
    qr, kr, vr = [x.double() for x in (q, k, v)]
    if components[0]:
        qr *= scales[0].repeat_interleave(ratio, -1)[:, None, :, None]
    if components[1]:
        kr *= scales[1][:, None, :, None]
    if components[2]:
        vr *= scales[2][:, None, :, None]
    kwargs = dict(return_lse=True, num_splits=1)
    kwargs.update(
        (name, scale)
        for name, scale, present in zip(
            ("q_descale", "k_descale", "v_descale"), scales, components
        )
        if present
    )
    causal = feature == "causal"
    kwargs["causal"] = causal
    window = (31, 7) if feature == "window" else (-1, -1)
    if feature == "window":
        kwargs.update(window_size_left=31, window_size_right=7)
    softcap = 2.5 if feature == "softcap" else 0.0
    if softcap:
        kwargs["softcap"] = softcap
    lq = [sq, min(33, sq)] if layout != "dense" else [sq, sq]
    lk = [sk, min(129, sk)] if layout != "dense" else [sk, sk]
    packed_q = layout in ("packed", "qpacked")
    packed_k = layout in ("packed", "kpacked")
    if packed_q or packed_k:
        if packed_q:
            q = torch.cat([q[i, :length] for i, length in enumerate(lq)])
            kwargs["cu_seqlens_q"] = torch.tensor(
                [0, *accumulate(lq)], device="cuda", dtype=torch.int32
            )
        else:
            kwargs["seqused_q"] = torch.tensor(lq, device="cuda", dtype=torch.int32)
        if packed_k:
            k, v = [
                torch.cat([x[i, :length] for i, length in enumerate(lk)]) for x in (k, v)
            ]
            kwargs["cu_seqlens_k"] = torch.tensor(
                [0, *accumulate(lk)], device="cuda", dtype=torch.int32
            )
        else:
            kwargs["seqused_k"] = torch.tensor(lk, device="cuda", dtype=torch.int32)
        kwargs.update(max_seqlen_q=max(lq), max_seqlen_k=max(lk))
    elif layout.startswith("paged"):
        page_size = int(layout.removeprefix("paged"))
        width = (sk + page_size - 1) // page_size
        table = torch.randperm(batch * width + 3, device="cuda")[:batch * width]
        table = table.reshape(batch, width).int()
        pages = []
        for x in (k, v):
            padded = torch.zeros(batch, width * page_size, hk, 512, device="cuda", dtype=dtype)
            padded.view(torch.uint8)[:, :sk].copy_(x.view(torch.uint8))
            pool = torch.zeros(batch * width + 3, page_size, hk, 512, device="cuda", dtype=dtype)
            pool.view(torch.uint8)[table.long()] = padded.view(torch.uint8).reshape(
                batch, width, page_size, hk, 512
            )
            pages.append(pool)
        k, v = pages
        kwargs.update(
            page_table=table,
            seqused_q=torch.tensor(lq, device="cuda", dtype=torch.int32),
            seqused_k=torch.tensor(lk, device="cuda", dtype=torch.int32),
            max_seqlen_q=sq, max_seqlen_k=sk,
        )
    out, lse, *_ = _flash_attn_fwd(q, k, v, **kwargs)
    assert out.dtype == torch.bfloat16
    assert len(fused._forward_cache) == 1, "FP8 must reach the fused QK/softmax path"
    tolerance = 0.06 if dtype == torch.float8_e4m3fn else 0.14
    offset = 0
    for i, (nq, nk) in enumerate(zip(lq, lk)):
        ref, ref_lse = reference(
            qr[i:i + 1, :nq], kr[i:i + 1, :nk], vr[i:i + 1, :nk],
            causal=causal, window=window, softcap=softcap,
        )
        actual = out[offset:offset + nq] if packed_q else out[i, :nq]
        actual_lse = lse[:, offset:offset + nq] if packed_q else lse[i, :, :nq]
        error = (actual.double() - ref[0]).abs()
        assert actual.isfinite().all()
        assert error.square().mean().sqrt() <= tolerance * ref.square().mean().sqrt()
        assert error.max() <= tolerance * ref.abs().max()
        torch.testing.assert_close(actual_lse.double(), ref_lse[0], atol=0.006, rtol=0.006)
        offset += nq
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("layout", ["dense", "packed", "paged16", "paged128"])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("feature", ["full", "causal", "window", "softcap"])
def test_fused_fp8_common_paths(dtype, layout, ratio, feature, monkeypatch):
    _run_fp8_case(dtype, layout, ratio, feature, (True, True, True), monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("components", list(product((False, True), repeat=3)))
def test_fused_fp8_partial_descales(dtype, components, monkeypatch):
    _run_fp8_case(dtype, "dense", 4, "full", components, monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("layout", ["packed", "paged16"])
def test_fused_fp8_mqa(dtype, layout, monkeypatch):
    _run_fp8_case(dtype, layout, 8, "causal", (True, True, True), monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("layout", ["qpacked", "kpacked"])
@pytest.mark.parametrize("feature", ["full", "causal"])
def test_fused_fp8_independent_qk_lengths(dtype, layout, feature, monkeypatch):
    _run_fp8_case(dtype, layout, 4, feature, (True, True, True), monkeypatch)


@pytest.mark.parametrize("dtype,layout,feature", [
    (torch.float8_e4m3fn, "dense", "full"),
    (torch.float8_e5m2, "dense", "full"),
    (torch.float8_e4m3fn, "dense", "causal"),
    (torch.float8_e5m2, "packed", "causal"),
    (torch.float8_e4m3fn, "paged16", "window"),
    (torch.float8_e5m2, "paged128", "softcap"),
])
def test_fused_fp8_pipeline_wraparound(dtype, layout, feature, monkeypatch):
    """More K/V tiles than pipeline slots exercise repeated phase wraparound."""
    _run_fp8_case(dtype, layout, 4, feature, (True, True, True), monkeypatch, sq=129, sk=1025)
