# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SM12x backward regression tests: dense / varlen, MHA / GQA / MQA, masks, sinks, softcap,
and deterministic mode."""

import itertools
import math
import os

import pytest
import torch

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode
from mask_mod_definitions import (
    cute_global_packed_doc_mask,
    get_cute_sliding_window_mask,
    get_mask_pair,
)
from score_mod_definitions import (
    score_mod_bwd_identity,
    score_mod_bwd_scaled_squared,
    score_mod_dual_buffer,
    score_mod_scaled_squared,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
    pytest.skip("SM12x backward tests require CUDA SM 12.x.", allow_module_level=True)


def _reference(q, k, v, lens_q, lens_k, causal, window, sink, softcap, scale, upcast,
               score_bias=None, score_mask=None):
    """Attention over packed (total, heads, dim) tensors, per-sequence Q / K lengths.

    Masks are aligned bottom-right (query i sees key j <= i + len_k - len_q when causal); rows
    without a visible key give zero output. score_mask(off_q, off_k, n_q, n_k) -> (n_q, n_k)
    bool, True where a sequence starting at packed offsets off_q / off_k sees the key.
    """
    outs, off_q, off_k = [], 0, 0
    g = q.shape[1] // k.shape[1]
    for n_q, n_k in zip(lens_q, lens_k):
        qi = q[off_q:off_q + n_q]
        ki, vi = (t[off_k:off_k + n_k] for t in (k, v))
        if upcast:
            qi, ki, vi = qi.float(), ki.float(), vi.float()
        qi = qi.transpose(0, 1)
        ki = ki.repeat_interleave(g, 1).transpose(0, 1)
        vi = vi.repeat_interleave(g, 1).transpose(0, 1)
        s = (qi @ ki.transpose(-1, -2)).float() * scale
        if score_bias is not None:
            s = s + score_bias(n_q, n_k)
        if softcap > 0:
            s = torch.tanh(s / softcap) * softcap
        rows = torch.arange(n_q, device=q.device)[:, None] + (n_k - n_q)
        cols = torch.arange(n_k, device=q.device)[None, :]
        mask = torch.zeros(n_q, n_k, dtype=torch.bool, device=q.device)
        if causal:
            mask |= cols > rows
        if window is not None:
            mask |= cols < rows - window[0]
            mask |= cols > rows + window[1]
        if score_mask is not None:
            mask |= ~score_mask(off_q, off_k, n_q, n_k)
        empty = mask.all(-1, keepdim=True)
        s = s.masked_fill(mask, float("-inf")).masked_fill(empty, 0.0)
        if sink is not None:
            s = torch.cat([s, sink.float()[:, None, None].expand(-1, n_q, 1)], dim=-1)
        p = torch.softmax(s, dim=-1)
        if sink is not None:
            p = p[..., :-1]
        p = p.masked_fill(empty, 0.0)
        outs.append((p.to(vi.dtype) @ vi).transpose(0, 1))
        off_q += n_q
        off_k += n_k
    return torch.cat(outs)


def _grads(fn, inputs, dout):
    inputs = [t.detach().clone().requires_grad_() for t in inputs]
    out = fn(*inputs)
    out = out[0] if isinstance(out, tuple) else out
    return torch.autograd.grad(out, inputs, dout)


def _bwd_cases():
    """Each head-dim pair with MHA / GQA / MQA, deterministic or not; the dense / varlen and
    noncausal / causal / local layouts rotate through each pair's cases."""
    layouts = [(v, m) for v in (False, True) for m in ("noncausal", "causal", "local")]
    cases = []
    for head_dim, head_dim_v in ((64, 64), (96, 96), (128, 128), (192, 128), (256, 256)):
        for heads, kind in (((8, 8), "mha"), ((8, 2), "gqa"), ((8, 1), "mqa")):
            if (head_dim, head_dim_v) == (192, 128) and kind == "mqa":
                continue
            for deterministic in (False, True):
                varlen, mode = layouts[len(cases) % len(layouts)]
                cases.append(pytest.param(
                    head_dim, head_dim_v, heads, varlen, mode, deterministic,
                    id=f"d{head_dim}x{head_dim_v}-{kind}-{'varlen' if varlen else 'dense'}-{mode}"
                    f"{'-det' if deterministic else ''}",
                ))
    return cases


@pytest.mark.parametrize("head_dim,head_dim_v,heads,varlen,mode,deterministic", _bwd_cases())
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_matches_reference(head_dim, head_dim_v, heads, varlen, mode, deterministic):
    _check(torch.bfloat16, head_dim, heads, varlen, mode, deterministic, sink=False, softcap=0.0,
           head_dim_v=head_dim_v)


@pytest.mark.parametrize(
    "head_dim,varlen,mode,lens_k,deterministic",
    [
        (64, False, "noncausal", None, False),
        (96, False, "noncausal", None, False),
        (128, False, "noncausal", None, False),
        (256, False, "noncausal", None, False),
        (128, False, "causal", None, True),
        (256, False, "causal", None, True),
        (128, True, "causal", None, False),
        (64, False, "noncausal", [77, 77], False),
        (128, False, "causal", [77, 77], False),
        (64, False, "noncausal", [200, 200], False),
        (128, False, "noncausal", [200, 200], False),
    ],
    ids=["d64", "d96", "d128", "d256", "d128_det", "d256_det", "d128_varlen", "d64_sk77",
         "d128_sk77", "d64_sk200", "d128_sk200"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_fp16(monkeypatch, head_dim, varlen, mode, lens_k, deterministic):
    """fp16 on every route (the other tests use bf16): TMA (d64 / d96 / d128), the wide-head-dim
    and deterministic split passes, packed TMA, and the direct dQ stores of one- and two-tile
    keys."""
    seen = _record_dkdv_kernels(monkeypatch)
    _check(torch.float16, head_dim, (8, 2), varlen, mode, deterministic, sink=False, softcap=0.0,
           lens_q=[1000, 1000] if lens_k else None, lens_k=lens_k)
    if is_fake_mode():
        return
    assert seen and all(seen.direct) == (lens_k is not None), seen.keys


# fp16 on each feature and layout the bf16 matrices cover (name: _check arguments, plus a
# forced head split "split" / "det_split" and a modifier "mods").
_FP16_FEATURE_CASES = {
    "head_split_mqa_s4_d128": dict(head_dim=128, heads=(8, 1), mode="noncausal", split=4),
    "head_split_mqa_s2_d64_varlen": dict(head_dim=64, heads=(8, 1), varlen=True, mode="causal",
                                         split=2),
    "head_split_mqa_s8_d96": dict(head_dim=96, heads=(8, 1), mode="causal", split=8),
    "head_split_gqa_s4_d256_local": dict(head_dim=256, heads=(8, 2), mode="local", split=4),
    "det_split_mqa_s4_d128_varlen": dict(head_dim=128, heads=(8, 1), varlen=True,
                                         mode="causal", det_split=4),
    "sink_d128_varlen": dict(head_dim=128, varlen=True, mode="causal", sink=True),
    "sink_d256_det": dict(head_dim=256, mode="causal", sink=True, deterministic=True),
    "softcap_d64": dict(head_dim=64, mode="causal", softcap=15.0),
    "softcap_d256_det": dict(head_dim=256, mode="causal", softcap=15.0, deterministic=True),
    "mask_mod_d128": dict(head_dim=128, mods="mask_mod"),
    "mask_mod_d256_det": dict(head_dim=256, mods="mask_mod", deterministic=True),
    "score_mod_d64": dict(head_dim=64, mode="causal", mods="score_mod"),
    "score_mod_d256_det": dict(head_dim=256, mode="causal", mods="score_mod",
                               deterministic=True),
    "hd192x128_gqa_local_varlen": dict(head_dim=192, head_dim_v=128, varlen=True, mode="local"),
    "hd192_mqa_causal": dict(head_dim=192, heads=(8, 1), mode="causal"),
    "hd256_mqa_local_varlen": dict(head_dim=256, heads=(8, 1), varlen=True, mode="local"),
    "envelope_72x136_varlen": dict(head_dim=72, head_dim_v=136, varlen=True, mode="causal",
                                   lens_q=[31, 33, 65, 200], lens_k=[32, 63, 33, 129],
                                   unused_k=[0, 5, 0, 31]),
}


@pytest.mark.parametrize("case", list(_FP16_FEATURE_CASES))
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_fp16_features(monkeypatch, case):
    import flash_attn.cute.sm120.bwd_runtime as rt

    args = dict(heads=(8, 2), varlen=False, mode="noncausal", deterministic=False, sink=False,
                softcap=0.0)
    args.update(_FP16_FEATURE_CASES[case])
    split, det_split = args.pop("split", None), args.pop("det_split", None)
    if split is not None:
        monkeypatch.setattr(rt, "_head_split", lambda *a, **kw: split)
    if det_split is not None:
        args["deterministic"] = True
        monkeypatch.setattr(rt, "_deterministic_head_split", lambda *a, **kw: det_split)
    mods = args.pop("mods", None)
    if mods == "mask_mod":
        args["mods"] = dict(mask_mod=get_cute_sliding_window_mask(64, 32, 0),
                            ref_window=(64, 32))
    elif mods == "score_mod":
        head_bias = torch.randn(8, device="cuda", dtype=torch.float32) * 0.5
        pos_bias = torch.randn(512, device="cuda", dtype=torch.float32) * 0.5
        args["mods"] = dict(
            score_mod=score_mod_dual_buffer, score_mod_bwd=score_mod_bwd_identity,
            aux_tensors=[head_bias, pos_bias],
            ref_bias=lambda n_q, n_k: head_bias[:, None, None] + pos_bias[:n_q][None, :, None],
        )
    head_dim = args.pop("head_dim")
    _check(torch.float16, head_dim, args.pop("heads"), args.pop("varlen"), args.pop("mode"),
           args.pop("deterministic"), **args)


@pytest.mark.parametrize("head_dim,deterministic", [(64, False), (128, False), (256, True)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_learnable_sink(head_dim, deterministic):
    """Packed sequences with a learnable sink (dense calls are covered by the generic tests)."""
    _check(torch.bfloat16, head_dim, (8, 2), True, "causal", deterministic, sink=True,
           softcap=0.0)


@pytest.mark.parametrize("mode,deterministic", [("local", False), ("noncausal", True)])
@pytest.mark.parametrize(
    "head_dims",
    [(256, 256), (256, 128), (128, 256), (192, 192), (200, 128), (72, 136), (192, 128)],
    ids=lambda d: f"{d[0]}x{d[1]}",
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_head_dim_envelope(head_dims, mode, deterministic):
    """Head-dim pairs across the routing rule, ragged Q != K lengths around the 32 / 64
    tile edges, K tokens past seqused_k, and (local, Q longer than K) rows with no keys."""
    _check(torch.bfloat16, head_dims[0], (8, 2), True, mode, deterministic, sink=False,
           softcap=0.0, head_dim_v=head_dims[1], lens_q=[31, 33, 65, 200, 64, 1],
           lens_k=[32, 63, 33, 129, 64, 97], unused_k=[0, 5, 0, 31, 1, 0])



@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
@pytest.mark.parametrize(
    "heads,split,varlen,mode",
    [((8, 1), 2, False, "noncausal"), ((8, 1), 8, True, "causal"), ((8, 2), 4, True, "noncausal"),
     ((12, 3), 2, False, "causal")],
    ids=["mqa_s2", "mqa_s8", "gqa_s4", "gqa3_s2"],
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_sm120_bwd_deterministic_head_split(monkeypatch, head_dim, heads, split, varlen, mode):
    """Deterministic head split: per-split fp32 slots, summed by one reduction (bit-identical
    across runs, checked by _check). The split is forced past the selector."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    used = []

    def head_split(*args, **kwargs):
        used.append(split)
        return split

    monkeypatch.setattr(rt, "_deterministic_head_split", head_split)
    _check(torch.bfloat16, head_dim, heads, varlen, mode, True, sink=False, softcap=0.0)
    assert used, "the head-split rule was not consulted"


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="captures a CUDA graph")
def test_sm120_bwd_deterministic_head_split_graph(monkeypatch):
    """Slot accumulators start from zero on every call and graph replay: replays with new dout
    match an eager run of the same configuration bit for bit."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    monkeypatch.setattr(rt, "_deterministic_head_split", lambda *a, **kw: 4)
    torch.manual_seed(0)
    b, s, d = 2, 320, 64
    q = torch.randn(b, s, 8, d, device="cuda", dtype=torch.bfloat16).requires_grad_()
    k, v = (
        torch.randn(b, s, 1, d, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(2)
    )
    dout = torch.randn(b, s, 8, d, device="cuda", dtype=torch.bfloat16)

    def grads():
        out = flash_attn_func(q, k, v, causal=True, deterministic=True)
        out = out[0] if isinstance(out, tuple) else out
        return torch.autograd.grad(out, (q, k, v), dout)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            grads()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = grads()
    for step in range(3):
        dout.copy_(torch.randn_like(dout) * (step + 1))
        graph.replay()
        torch.cuda.synchronize()
        replayed = [g.clone() for g in captured]
        for a, e in zip(replayed, grads()):
            assert torch.equal(a, e), "graph replay differs from eager"


def test_sm120_slot_split_choice():
    """Host rules of the deterministic split: no split for MHA, divisors of the group only,
    the scratch cap at its boundary, and the slot bytes the allocation uses."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    base = dict(ctas=64, m_tiles=128, tile_m=64, tile_n=128, head_dim=64, head_dim_v=64,
                causal=False, num_sms=188, dram_bytes_per_us=1.8e6)
    assert rt._slot_split_choice(group=1, slot_bytes=1 << 20, **base) == 1
    for group in (6, 8, 12):
        split = rt._slot_split_choice(group=group, slot_bytes=1 << 20, **base)
        assert split > 1 and group % split == 0
    budget = rt._SLOT_SCRATCH_BUDGET
    # One CTA, compute-dominated: the largest split wins whenever its slots plus their sum fit.
    big = dict(base, ctas=1, m_tiles=1 << 20, dram_bytes_per_us=1e12)
    assert rt._slot_split_choice(group=8, slot_bytes=budget // 9, **big) == 8
    assert rt._slot_split_choice(group=8, slot_bytes=budget // 9 + 1, **big) == 4
    assert rt._slot_split_choice(group=8, slot_bytes=budget // 2, **big) == 1
    # The cap counts the bytes _dkv_accum_shapes allocates for one slot per KV head.
    k = torch.empty(2, 300, 3, 64, device="cuda")
    one = sum(math.prod(sh) for sh in rt._dkv_accum_shapes(k, None, 128, 3, (64, 64)))
    four = sum(math.prod(sh) for sh in rt._dkv_accum_shapes(k, None, 128, 12, (64, 64)))
    assert four == 4 * one == 4 * 2 * 3 * 384 * 128


def test_sm120_dq_head_group_scope():
    """dQ longest-first grouping: every dense causal call on discrete parts, in whole KV-head
    groups; non-causal, local and varlen / seqused calls, and integrated parts, keep the
    original order (0)."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    def group(d=64, dv=64, hq=8, hk=1, causal=True, local=False, seq=(None,) * 4):
        q = torch.empty(2, 4096, hq, d, device="cuda", dtype=torch.bfloat16)
        k = torch.empty(2, 4096, hk, d, device="cuda", dtype=torch.bfloat16)
        return rt._dq_head_group(q, k, causal, local, seq, d, dv, hq // hk)

    if rt._is_integrated(torch.cuda.current_device()):
        assert group() == 0
        return
    for kwargs in (dict(), dict(hq=32, hk=8), dict(d=128, dv=128), dict(hq=16, hk=16),
                   dict(d=128, dv=128, hq=16, hk=16), dict(d=256, dv=256), dict(d=256, dv=128),
                   dict(d=192, dv=128), dict(d=96, dv=96)):
        g = group(**kwargs)
        assert g > 0 and g % (kwargs.get("hq", 8) // kwargs.get("hk", 1)) == 0, kwargs
    cu = torch.zeros(2, dtype=torch.int32, device="cuda")
    for kwargs in (dict(causal=False), dict(local=True), dict(seq=(cu, cu, None, None)),
                   dict(seq=(None, None, None, cu))):
        assert group(**kwargs) == 0, kwargs


def test_sm120_deterministic_head_split_scope():
    """Shapes the split model does not cover keep the unsplit deterministic pass."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    def split(d=64, dv=64, seq=(None,) * 4, local=False, mods=False, causal=False, seqlen_k=8192):
        q = torch.empty(1, 8192, 8, d, device="cuda", dtype=torch.bfloat16)
        k = torch.empty(1, seqlen_k, 1, d, device="cuda", dtype=torch.bfloat16)
        v = torch.empty(1, seqlen_k, 1, dv, device="cuda", dtype=torch.bfloat16)
        return rt._deterministic_head_split(q, k, v, seq, local, mods, 64, 128, 8, 1, causal,
                                            (d, dv))

    cu = torch.zeros(2, dtype=torch.int32, device="cuda")
    assert split() > 1  # MQA, 64 CTAs: the modelled case splits
    assert split(causal=True) > 1
    assert split(seqlen_k=4096) > 1  # rectangular non-causal is still a full sweep
    for kwargs in (dict(d=96, dv=96), dict(d=256, dv=256), dict(seq=(cu, cu, None, None)),
                   dict(seq=(None, None, None, cu)), dict(local=True), dict(mods=True),
                   dict(causal=True, seqlen_k=4096), dict(causal=True, seqlen_k=128)):
        assert split(**kwargs) == 1, kwargs


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
@pytest.mark.parametrize("head_dim,deterministic", [(64, False), (128, True), (256, True)])
def test_sm120_bwd_mask_mod(head_dim, deterministic):
    """mask_mod through both dQ paths (fused and the dQ kernel) and the hd-256 layout."""
    left, right = 64, 32
    mods = dict(mask_mod=get_cute_sliding_window_mask(left, right, 0), ref_window=(left, right))
    _check(torch.bfloat16, head_dim, (8, 2), False, "noncausal", deterministic, sink=False,
           softcap=0.0, mods=mods)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
@pytest.mark.parametrize("head_dim,deterministic", [(64, False), (128, True), (256, True)])
def test_sm120_bwd_score_mod_aux_tensors(head_dim, deterministic):
    """score_mod reading aux tensors indexed by head and query position."""
    head_bias = torch.randn(8, device="cuda", dtype=torch.float32) * 0.5
    pos_bias = torch.randn(512, device="cuda", dtype=torch.float32) * 0.5

    def ref_bias(n_q, n_k):
        return head_bias[:, None, None] + pos_bias[:n_q][None, :, None]

    mods = dict(score_mod=score_mod_dual_buffer, score_mod_bwd=score_mod_bwd_identity,
                aux_tensors=[head_bias, pos_bias], ref_bias=ref_bias)
    _check(torch.bfloat16, head_dim, (8, 2), False, "causal", deterministic, sink=False,
           softcap=0.0, mods=mods)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_sm120_bwd_score_mod_bwd_masked_aux_reads(head_dim, deterministic):
    """A score_mod_bwd reading an aux tensor at global key indices also runs on the masked
    columns past the last sequence's end, where the read lands past the end of the tensor.
    Whatever is stored there (NaN here) must not reach the gradients."""
    torch.manual_seed(0)
    lens = [17, 33, 65]
    total = sum(lens)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device="cuda", dtype=torch.int32)
    q = torch.randn(total, 8, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    scale = torch.randn(total, device="cuda", dtype=torch.float32) * 0.1 + 1.0

    def grads(pad):
        buf = torch.full((total + 1024,), pad, device="cuda", dtype=torch.float32)
        buf[:total] = scale

        def fa(q, k, v):
            return flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(lens),
                max_seqlen_k=max(lens), score_mod=score_mod_scaled_squared,
                score_mod_bwd=score_mod_bwd_scaled_squared, aux_tensors=[buf[:total]],
                deterministic=deterministic,
            )

        return _grads(fa, [q, k, v], dout)

    g_nan = grads(float("nan"))
    g_zero = grads(0.0)
    if is_fake_mode():
        return
    for name, a, b in zip(["dq", "dk", "dv"], g_nan, g_zero):
        assert torch.isfinite(a).all(), f"{name}: non-finite"
        if deterministic:
            assert torch.equal(a, b), f"{name}: depends on memory past the aux tensor"
        else:
            # Run-to-run accumulation order differs; values past the aux tensor would not.
            torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
@pytest.mark.parametrize("head_dim", [128])
def test_sm120_bwd_varlen_offset_mask_mod(head_dim):
    """A mask_mod reading packed aux tensors at seqlen_info.offset_q / offset_k, with equal
    sequence lengths (so offset 0 for every sequence would be in bounds but wrong)."""
    lens = [256, 256]
    # Per-token document ids: sequence 0 has a striped pattern, sequence 1 one document, so
    # reading sequence 1 at offset 0 would mask most of its keys.
    pos = torch.arange(lens[0], device="cuda", dtype=torch.int32)
    doc_q = torch.cat([(pos // 64) % 2, torch.full_like(pos, 2)])
    doc_k = torch.cat([(pos // 32) % 2, torch.full_like(pos, 2)])

    def ref_mask(off_q, off_k, n_q, n_k):
        return doc_q[off_q:off_q + n_q, None] == doc_k[None, off_k:off_k + n_k]

    mods = dict(mask_mod=cute_global_packed_doc_mask, aux_tensors=[doc_q, doc_k],
                ref_mask=ref_mask)
    _check(torch.bfloat16, head_dim, (8, 2), True, "noncausal", False, sink=False, softcap=0.0,
           lens_q=lens, mods=mods)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_green_context(monkeypatch):
    """Launch choices use the SMs of the stream's green context, and results stay correct."""
    from flash_attn.cute.cute_dsl_utils import get_num_sms_for_selection

    green_sms = 32
    if USE_FAKE_TENSOR:
        # Compilation pass: a green context needs a real device context; pinning the SM count
        # issues the same kernels.
        monkeypatch.setenv("FLASH_ATTENTION_NUM_SMS", str(green_sms))
        _green_case()
        return
    green_contexts = pytest.importorskip("torch.cuda.green_contexts")
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    total = torch.cuda.get_device_properties(device).multi_processor_count
    torch.zeros(1, device="cuda")
    try:
        ctx = green_contexts.GreenContext.create(num_sms=green_sms, device_id=device)
    except RuntimeError as e:
        pytest.skip(f"green contexts unavailable: {e}")
    stream = ctx.Stream()
    with torch.cuda.stream(stream):
        assert get_num_sms_for_selection(device, major * 10 + minor) == green_sms
        _green_case()
    stream.synchronize()
    assert get_num_sms_for_selection(device, major * 10 + minor) == total


def _green_case():
    # MQA: the head split is chosen from the SM count.
    _check(torch.bfloat16, 128, (8, 1), True, "causal", False, sink=False, softcap=0.0)


def _check(dtype, head_dim, heads, varlen, mode, deterministic, sink, softcap, head_dim_v=None,
           lens_q=None, lens_k=None, unused_k=None, mods=None, seed=0, unused_q=None):
    """Gradients against an fp32 reference, within 2x the error of a same-precision reference.

    unused_k / unused_q: per-sequence K / V and Q tokens past seqused_k / seqused_q (varlen
    only); their gradients are not compared.
    """
    torch.manual_seed(seed)
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    hq, hk = heads
    if lens_q is None:
        lens_q = [300, 37, 1025, 640] if varlen else [512, 512]
    lens_k = lens_q if lens_k is None else lens_k
    unused_k = [0] * len(lens_k) if unused_k is None else unused_k
    alloc_k = [n + u for n, u in zip(lens_k, unused_k)]
    unused_q = [0] * len(lens_q) if unused_q is None else unused_q
    alloc_q = [n + u for n, u in zip(lens_q, unused_q)]
    total_q, total_k = sum(alloc_q), sum(alloc_k)
    q = torch.randn(total_q, hq, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total_k, hk, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(total_k, hk, head_dim_v, device="cuda", dtype=dtype)
    dout = torch.randn(total_q, hq, head_dim_v, device="cuda", dtype=dtype)
    # Positions of the used Q tokens in the packed allocation; unused rows get no gradient.
    keep_q = torch.cat([torch.arange(a, a + n, device="cuda")
                        for a, n in zip(itertools.accumulate([0] + alloc_q[:-1]), lens_q)])
    if any(unused_q):
        dout_unused = torch.ones(total_q, dtype=torch.bool, device="cuda")
        dout_unused[keep_q] = False
        dout[dout_unused] = 0
    sink_t = torch.randn(hq, device="cuda", dtype=torch.float32) if sink else None
    causal = mode in ("causal", "local")
    # "local": causal with a left window; "window": a two-sided window without causal.
    window = {"local": (96, 0), "window": (96, 64)}.get(mode)
    # mods: extra flash_attn kwargs (score_mod / mask_mod / aux_tensors) plus their reference:
    # "ref_window" for a mask_mod window, "ref_bias" (n_q, n_k) -> per-head score bias,
    # "ref_mask" a _reference score_mask.
    mods = dict(mods or {})
    ref_window, ref_bias = mods.pop("ref_window", None), mods.pop("ref_bias", None)
    ref_mask = mods.pop("ref_mask", None)
    scale = 1.0 / math.sqrt(head_dim)
    cu = lambda lens: torch.tensor(
        [0] + list(itertools.accumulate(lens)), device="cuda", dtype=torch.int32
    )
    used = torch.tensor(lens_k, device="cuda", dtype=torch.int32) if any(unused_k) else None
    used_q = torch.tensor(lens_q, device="cuda", dtype=torch.int32) if any(unused_q) else None
    # Positions of the used K / V tokens in the packed allocation.
    keep_k = torch.cat([torch.arange(a, a + n, device="cuda")
                        for a, n in zip(itertools.accumulate([0] + alloc_k[:-1]), lens_k)])

    def fa(q, k, v, *extra):
        kwargs = dict(
            causal=causal,
            window_size=window or (None, None),
            learnable_sink=extra[0] if extra else None,
            softcap=softcap,
            deterministic=deterministic,
            **mods,
        )
        if varlen:
            return flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu(alloc_q), cu_seqlens_k=cu(alloc_k), seqused_k=used,
                seqused_q=used_q, max_seqlen_q=max(alloc_q), max_seqlen_k=max(alloc_k), **kwargs,
            )
        assert len(set(lens_q)) == 1 and len(set(lens_k)) == 1 and used is None and used_q is None
        shape = lambda t, n: t.view(len(lens_q), n, *t.shape[1:])
        out = flash_attn_func(
            shape(q, lens_q[0]), shape(k, lens_k[0]), shape(v, lens_k[0]), **kwargs
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.reshape(total_q, *out.shape[2:])

    def ref(q, k, v, *extra, upcast=True):
        return _reference(q[keep_q], k[keep_k], v[keep_k], lens_q, lens_k, causal,
                          ref_window or window, extra[0] if extra else None, softcap, scale,
                          upcast, ref_bias, ref_mask)

    inputs = [q, k, v] + ([sink_t] if sink else [])
    g_fa = _grads(fa, inputs, dout)
    if is_fake_mode():
        # Compilation pass: every kernel of this case is issued above (a deterministic
        # rerun hits the same cache entries).
        return
    g_ref = _grads(ref, [t.float() for t in inputs], dout[keep_q].float())
    g_lp = _grads(lambda *a: ref(*a, upcast=False), inputs, dout[keep_q])
    for name, a, r, lp in zip(["dq", "dk", "dv", "dsink"], g_fa, g_ref, g_lp):
        if name in ("dk", "dv"):
            a, r, lp = a[keep_k], r[keep_k], lp[keep_k]
        elif name == "dq":
            a, r, lp = a[keep_q], r[keep_q], lp[keep_q]
        assert torch.isfinite(a).all(), f"{name}: non-finite"
        err = (a.float() - r).abs().max().item()
        bound = 2 * (lp.float() - r).abs().max().item() + 1e-2
        assert err <= bound, f"{name}: {err} > {bound}"
    if deterministic:
        g_again = _grads(fa, inputs, dout)
        for name, a, b in zip(["dq", "dk", "dv", "dsink"], g_fa, g_again):
            if name in ("dk", "dv"):
                a, b = a[keep_k], b[keep_k]
            elif name == "dq":
                a, b = a[keep_q], b[keep_q]
            assert torch.equal(a, b), f"{name}: not bit-identical across runs"


def _record_dkdv_kernels(monkeypatch):
    """(tma_ws, head_split, stages_dO) of every dK / dV kernel the backward looks up, with
    per-lookup key fields in seen.tail (head_split, tail_split), seen.l2, seen.direct,
    seen.sweep, seen.sparse, seen.dense (no sequence tensors) and the whole key in seen.keys.

    Wraps the compile cache, so cache hits (in-memory or persistent) are seen as well as
    compiles.
    """
    import flash_attn.cute.interface as interface
    from flash_attn.cute.sm120.bwd_runtime import DKDV_KEY_FIELDS

    field = {name: i for i, name in enumerate(DKDV_KEY_FIELDS)}

    class Seen(list):
        pass

    seen = Seen()
    seen.tail = []
    seen.l2 = []
    seen.direct = []
    seen.sweep = []
    seen.sparse = []
    seen.dense = []
    seen.keys = []
    cache = interface._flash_attn_bwd.compile_cache

    class Recording:
        def _note(self, key):
            if isinstance(key, tuple) and key and key[0] == "sm120_dkdv":
                f = {name: key[i] for name, i in field.items()}
                seen.append((f["tma_ws"], f["head_split"], f["stages_do"]))
                seen.tail.append((f["head_split"], f["tail_split"]))
                seen.l2.append(f["l2_group_bytes"])
                seen.direct.append(f["direct_dq"])
                seen.sweep.append(f["kv_sweep"])
                seen.sparse.append(f["sparse_key"])
                # seq_key: True per absent cu_seqlens_q / cu_seqlens_k / seqused_q / seqused_k.
                seen.dense.append(all(f["seq_key"]))
                seen.keys.append(key)

        def __contains__(self, key):
            self._note(key)
            return key in cache

        def __getitem__(self, key):
            return cache[key]

        def __setitem__(self, key, value):
            cache[key] = value

        def __getattr__(self, name):
            return getattr(cache, name)

    monkeypatch.setattr(interface._flash_attn_bwd, "compile_cache", Recording())
    return seen


@pytest.mark.parametrize(
    "head_dim,heads,mode",
    [(64, (8, 2), "noncausal"), (64, (8, 1), "causal"), (128, (8, 2), "causal"),
     (128, (8, 1), "noncausal")],
    ids=["d64-gqa", "d64-mqa", "d128-gqa", "d128-mqa"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tma_multi_head(monkeypatch, head_dim, heads, mode):
    """TMA producer path with several Q heads per CTA (split forced to 1), so the producer
    resets its barriers at head boundaries; checked against the reference."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    monkeypatch.setattr(rt, "_head_split", lambda *args, **kwargs: 1)
    seen = _record_dkdv_kernels(monkeypatch)
    # Keys over two KV tiles, so d64 does not take the direct-dQ route of short keys.
    lens = [1024, 1024] if head_dim == 64 else None
    _check(torch.bfloat16, head_dim, heads, False, mode, False, sink=False, softcap=0.0,
           lens_q=lens)
    if is_fake_mode():
        return
    # The TMA producer refills dO once both groups' dV is done: one dO stage.
    assert seen and all(k == (True, 1, 1) for k in seen), seen


@pytest.mark.parametrize(
    "case",
    ["local_causal", "local", "causal_q_lt_k", "causal_q_gt_k", "mask_mod"],
)
@pytest.mark.parametrize("heads", [(8, 2)], ids=["gqa"])
@pytest.mark.parametrize("head_dim", [64, 128])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tma_masks_and_mods(monkeypatch, head_dim, heads, case):
    """Local windows, causal calls with seqlen_q != seqlen_k and mask_mod on the TMA producer
    path, against the reference."""
    lens_q, lens_k, mode, mods = [512, 512], None, "noncausal", None
    if case == "local_causal":
        mode = "local"
    elif case == "local":
        mode = "window"
    elif case == "causal_q_lt_k":
        lens_q, lens_k, mode = [300, 300], [700, 700], "causal"
    elif case == "causal_q_gt_k":
        lens_q, lens_k, mode = [700, 700], [300, 300], "causal"
    else:
        mods = dict(mask_mod=get_cute_sliding_window_mask(64, 32, 0), ref_window=(64, 32))
    # 188 SMs for the split choice: the mask_mod case (64 base CTAs) then splits its Q ranges,
    # covering a modifier with the tail fixup.
    monkeypatch.setattr(
        "flash_attn.cute.cute_dsl_utils.get_num_sms_for_selection", lambda *args, **kwargs: 188
    )
    seen = _record_dkdv_kernels(monkeypatch)
    _check(torch.bfloat16, head_dim, heads, False, mode, False, sink=False, softcap=0.0,
           lens_q=lens_q, lens_k=lens_k, mods=mods)
    if is_fake_mode():
        return
    assert seen and all(k[0] is True for k in seen), seen
    if case == "mask_mod":
        assert seen.tail and all(parts > 1 for _, parts in seen.tail), seen.tail


@pytest.mark.parametrize("head_dim,seqlen", [(64, 40), (128, 17), (128, 40), (64, 961), (128, 481)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tma_short(monkeypatch, head_dim, seqlen):
    """Short and partial-tile sequences on the TMA producer path: one partial Q tile at d64 S40
    and d128 S17, two at d128 S40."""
    seen = _record_dkdv_kernels(monkeypatch)
    for mode in ("noncausal", "causal"):
        _check(torch.bfloat16, head_dim, (8, 2), False, mode, False, sink=False, softcap=0.0,
               lens_q=[seqlen, seqlen])
    if is_fake_mode():
        return
    assert seen and all(k[0] is True and k[2] == 1 for k in seen), seen


@pytest.mark.parametrize(
    "head_dim,heads,lens_q,lens_k,unused_k,mode",
    [
        (64, (8, 8), [512, 512], None, None, "noncausal"),
        (128, (8, 8), [300, 37, 1025, 640], None, None, "noncausal"),
        (64, (8, 2), [300, 37, 1025, 640], None, None, "causal"),
        (128, (8, 2), [300, 300, 300], [512, 512, 512], None, "noncausal"),
        (64, (8, 8), [300, 300, 300], [512, 512, 512], None, "causal"),
        (128, (8, 8), [1, 17, 33, 200], None, None, "causal"),
        (64, (8, 2), [300, 37, 1025, 640], None, [0, 64, 0, 13], "noncausal"),
    ],
    ids=["equal", "ragged", "ragged_causal", "rect", "rect_causal", "short", "seqused_k"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tma_varlen(monkeypatch, head_dim, heads, lens_q, lens_k, unused_k, mode):
    """Packed calls on the TMA producer path: tiles start at each sequence's row offset, and
    a sequence's last Q tile reads the next sequence's rows (or TMA's zero fill at the end)."""
    seen = _record_dkdv_kernels(monkeypatch)
    _check(torch.bfloat16, head_dim, heads, True, mode, False, sink=False, softcap=0.0,
           lens_q=lens_q, lens_k=lens_k, unused_k=unused_k)
    if is_fake_mode():
        return
    assert seen and not any(seen.dense) and all(k[0] is True for k in seen), seen.keys


@pytest.mark.parametrize(
    "lens_q,lens_k",
    [([1000, 1000], [77, 77]), ([4096], [128]), ([33], [1])],
    ids=["77", "128", "1"],
)
@pytest.mark.parametrize(
    "head_dim,heads,mode", [(64, (8, 8), "noncausal"), (128, (8, 2), "causal")],
    ids=["d64-mha-noncausal", "d128-gqa-causal"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_single_kv(monkeypatch, head_dim, heads, mode, lens_q, lens_k):
    """Keys within one KV tile: the kernel stores dQ itself (no fp32 accumulator, zero fill
    or postprocess), with the Q range split over CTAs for small grids."""
    seen = _record_dkdv_kernels(monkeypatch)
    _check(torch.bfloat16, head_dim, heads, False, mode, False, sink=False, softcap=0.0,
           lens_q=lens_q, lens_k=lens_k)
    if is_fake_mode():
        return
    assert seen and all(seen.direct), seen.keys


@pytest.mark.parametrize(
    "lens_q,lens_k", [([1000, 1000], [200, 200]), ([4096], [256]), ([700, 700], [129, 129])],
    ids=["200", "256", "129"],
)
@pytest.mark.parametrize(
    "head_dim,heads,mode",
    [(64, (8, 2), "noncausal"), (128, (8, 8), "softcap"), (128, (8, 2), "causal")],
    ids=["d64-gqa-noncausal", "d128-mha-softcap", "d128-gqa-causal"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_kv_sweep(monkeypatch, head_dim, heads, mode, lens_q, lens_k):
    """Keys over two KV tiles: non-causal calls sweep both for each Q range and store dQ
    themselves (fp32 partials between the sweeps); causal ones keep the dQ accumulator."""
    monkeypatch.setattr(
        "flash_attn.cute.cute_dsl_utils.get_num_sms_for_selection", lambda *args, **kwargs: 188
    )
    seen = _record_dkdv_kernels(monkeypatch)
    softcap = 30.0 if mode == "softcap" else 0.0
    mode = "noncausal" if mode == "softcap" else mode
    _check(torch.bfloat16, head_dim, heads, False, mode, False, sink=False, softcap=softcap,
           lens_q=lens_q, lens_k=lens_k)
    if is_fake_mode():
        return
    if mode == "causal":
        assert seen and not any(seen.sweep), seen.keys
    else:
        assert seen and all(seen.sweep), seen.keys


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="captures a CUDA graph")
def test_sm120_bwd_kv_sweep_graph(monkeypatch):
    """The sweep's dQ partials need no initial value: graph replays with new dout (split Q
    ranges, so also the per-KV-tile dK / dV fixups) match eager calls."""
    monkeypatch.setattr(
        "flash_attn.cute.cute_dsl_utils.get_num_sms_for_selection", lambda *args, **kwargs: 188
    )
    seen = _record_dkdv_kernels(monkeypatch)
    torch.manual_seed(0)
    q = torch.randn(2, 2048, 8, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    k, v = (
        torch.randn(2, 192, 8, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(2)
    )
    dout = torch.randn(2, 2048, 8, 128, device="cuda", dtype=torch.bfloat16)

    def grads():
        out = flash_attn_func(q, k, v)
        out = out[0] if isinstance(out, tuple) else out
        return torch.autograd.grad(out, (q, k, v), dout)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            grads()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = grads()
    assert seen and all(seen.sweep), seen.keys
    assert all(parts > 1 for _, parts in seen.tail), seen.tail
    for step in range(3):
        dout.copy_(torch.randn_like(dout) * (step + 1))
        graph.replay()
        torch.cuda.synchronize()
        replayed = [g.clone() for g in captured]
        for name, a, e in zip(("dq", "dk", "dv"), replayed, grads()):
            # The dK / dV fixup order differs between runs: compare within bf16 rounding.
            tol = 1e-2 * e.float().abs().max().item()
            assert (a.float() - e.float()).abs().max().item() <= tol, name


@pytest.mark.parametrize("mode", ["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tma_seqused_q(monkeypatch, mode):
    """Packed calls with seqused_q shorter than the allocation (one sequence empty) on the TMA
    path: valid dQ rows and dK / dV against the reference."""
    seen = _record_dkdv_kernels(monkeypatch)
    _check(torch.bfloat16, 128, (8, 2), True, mode, False, sink=False, softcap=0.0,
           lens_q=[300, 0, 640], unused_q=[37, 50, 0], lens_k=[400, 200, 640])
    if is_fake_mode():
        return
    assert seen and all(k[0] is True for k in seen), seen.keys


@pytest.mark.parametrize("Sk", [100, 200])
@pytest.mark.parametrize(
    "case,head_dim", [("local", 64), ("mask_mod", 128), ("mask_mod_and_window", 64)]
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_single_kv_local_mask_mod_dlse(monkeypatch, head_dim, case, Sk):
    """Single-KV-tile path (and at Sk 200 the two-tile sweep, except for local windows) with
    a local window or a mask_mod and a nonzero LSE gradient, with its Q ranges split (tail
    fixup): dQ / dK / dV against an fp32 reference. With both, the mask_mod replaces the
    window, as in the forward."""
    torch.manual_seed(0)
    B, Sq, H = 2, 1024, 8
    win, mwin = (300, 200), (250, 400)
    q = torch.randn(B, Sq, H, head_dim, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(B, Sk, H, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    dout = torch.randn_like(q)
    dlse = torch.randn(B, H, Sq, device="cuda", dtype=torch.float32)
    kwargs = dict(return_lse=True)
    if case != "mask_mod":
        kwargs["window_size"] = win
    if case != "local":
        kwargs["mask_mod"] = get_cute_sliding_window_mask(*mwin, 0)
    left, right = win if case == "local" else mwin
    monkeypatch.setattr(
        "flash_attn.cute.cute_dsl_utils.get_num_sms_for_selection", lambda *args, **kwargs: 188
    )
    seen = _record_dkdv_kernels(monkeypatch)

    def ref(q, k, v):
        # Bottom-right aligned windows; rows without a visible key give zero output and LSE.
        s = torch.einsum("bshd,bthd->bhst", q, k) / math.sqrt(head_dim)
        rows = torch.arange(Sq, device="cuda")[:, None] + (Sk - Sq)
        cols = torch.arange(Sk, device="cuda")[None, :]
        s = s.masked_fill((cols < rows - left) | (cols > rows + right), float("-inf"))
        lse = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse[..., None]).nan_to_num(0.0)
        return torch.einsum("bhst,bthd->bshd", p, v), lse.nan_to_num(0.0, neginf=0.0)

    inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    out, lse = flash_attn_func(*inputs, **kwargs)
    g_fa = torch.autograd.grad((out, lse), inputs, (dout, dlse))
    if is_fake_mode():
        return
    inputs_ref = [t.detach().float().requires_grad_() for t in (q, k, v)]
    g_ref = torch.autograd.grad(ref(*inputs_ref), inputs_ref, (dout.float(), dlse))
    for name, a, r in zip(("dq", "dk", "dv"), g_fa, g_ref):
        assert torch.isfinite(a).all(), name
        err = (a.float() - r).abs().max().item() / r.abs().max().item()
        assert err < 2e-2, f"{name}: {err}"
    if Sk > 128 and case == "local":
        assert seen and not any(seen.direct), seen.keys
        return
    assert seen and all(seen.direct) and all(seen.sweep) == (Sk > 128), seen.keys
    assert seen.tail and all(parts > 1 for _, parts in seen.tail), seen.tail


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("heads", [(8, 8), (16, 4)], ids=["mha", "gqa"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tail_split(monkeypatch, heads, head_dim):
    """Tail split: the partial last wave's KV tiles run as several CTAs; their
    dK / dV meet in fp32 scratch (MHA) or the head-split accumulators (GQA)."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    # 16 KV tiles x 8 (KV head, split) slots per sequence: 3 sequences make 384 CTAs, a last
    # wave of 8 on the 188 SMs the rule is told about, split in 4.
    monkeypatch.setattr(rt, "_head_split", lambda *args, **kwargs: 1 if heads[0] == heads[1] else 2)
    monkeypatch.setattr(
        "flash_attn.cute.cute_dsl_utils.get_num_sms_for_selection", lambda *args, **kwargs: 188
    )
    seen = _record_dkdv_kernels(monkeypatch)
    _check(torch.bfloat16, head_dim, heads, False, "noncausal", False, sink=False, softcap=0.0,
           lens_q=[2048] * 3)
    if is_fake_mode():
        return
    assert seen.tail and all(t[1] == 4 for t in seen.tail), seen.tail

# (case id, SMs the rule is told about, heads, forced head split, lens_q, lens_k, expected
# (tail_tiles, parts)). 128-row KV tiles; each case has its own tail.
_TAIL_EDGE_CASES = [
    ("parts2", 154, (8, 8), 1, [2048] * 3, None, (76, 2)),
    ("parts3", 170, (8, 8), 1, [2048] * 3, None, (44, 3)),
    ("ragged_qk", 188, (8, 8), 1, [1500] * 3, [2000] * 3, (8, 4)),
    ("gqa_scratch", 170, (16, 4), 1, [2048] * 3, None, (22, 4)),
    # One Q tile per KV tile: the 4 parts of each tail tile get 0, 0, 0 and 1 Q tiles.
    ("empty_shares", 188, (8, 8), 1, [17] * 3, [2048] * 3, (8, 4)),
]


def _tail_case_setup(monkeypatch, sms, heads, split):
    import flash_attn.cute.sm120.bwd_runtime as rt

    monkeypatch.setattr(rt, "_head_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(
        "flash_attn.cute.cute_dsl_utils.get_num_sms_for_selection", lambda *args, **kwargs: sms
    )
    plans = []
    real = rt._tail_split

    def recording(*args, **kwargs):
        plans.append(real(*args, **kwargs))
        return plans[-1]

    monkeypatch.setattr(rt, "_tail_split", recording)
    return plans


@pytest.mark.parametrize("case", _TAIL_EDGE_CASES, ids=[c[0] for c in _TAIL_EDGE_CASES])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tail_split_edges(monkeypatch, case):
    """Tail split: 2 / 3 parts, ragged Q / K, head-split-1 GQA through the scratch fixup;
    a second call with new inputs (same kernel, fresh scratch and counters)."""
    _, sms, heads, split, lens_q, lens_k, expect = case
    plans = _tail_case_setup(monkeypatch, sms, heads, split)
    seen = _record_dkdv_kernels(monkeypatch)
    for seed in (0, 1):
        _check(torch.bfloat16, 128, heads, False, "noncausal", False, sink=False, softcap=0.0,
               lens_q=lens_q, lens_k=lens_k, seed=seed)
    if is_fake_mode():
        return
    assert plans and all(p == expect for p in plans), plans
    assert seen.tail and all(t == (split, expect[1]) for t in seen.tail), seen.tail


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_tail_split_same_key(monkeypatch):
    """Two shapes that share the compiled kernel but not the tail tile count."""
    plans = _tail_case_setup(monkeypatch, 188, (8, 8), 1)
    seen = _record_dkdv_kernels(monkeypatch)
    # 384 CTAs (tail 8) and 392 CTAs (tail 16), both in 4 parts.
    _check(torch.float16, 128, (8, 8), False, "noncausal", False, sink=False, softcap=0.0,
           lens_q=[2048] * 3)
    _check(torch.float16, 128, (8, 8), False, "noncausal", False, sink=False, softcap=0.0,
           lens_q=[6272])
    if is_fake_mode():
        return
    assert plans == [(8, 4), (16, 4)], plans
    assert len(set(seen)) == 1 and len(set(seen.tail)) == 1, (seen, seen.tail)


@pytest.mark.parametrize("seqlen_q", [2048, 17], ids=["full", "empty_shares"])
@pytest.mark.skipif(USE_FAKE_TENSOR, reason="captures a CUDA graph")
def test_sm120_bwd_tail_split_graph(monkeypatch, seqlen_q):
    """Tail scratch and counters start from zero on every graph replay (also with tail parts
    that get no Q tiles)."""
    plans = _tail_case_setup(monkeypatch, 188, (8, 8), 1)
    torch.manual_seed(0)
    q = torch.randn(3, seqlen_q, 8, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    k, v = (
        torch.randn(3, 2048, 8, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(2)
    )
    dout = torch.randn(3, seqlen_q, 8, 128, device="cuda", dtype=torch.bfloat16)

    def grads():
        out = flash_attn_func(q, k, v)
        out = out[0] if isinstance(out, tuple) else out
        return torch.autograd.grad(out, (q, k, v), dout)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            grads()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = grads()
    assert plans and all(p == (8, 4) for p in plans), plans
    for step in range(3):
        dout.copy_(torch.randn_like(dout) * (step + 1))
        graph.replay()
        torch.cuda.synchronize()
        replayed = [g.clone() for g in captured]
        for name, a, e in zip(("dq", "dk", "dv"), replayed, grads()):
            # Atomic order differs between runs: compare to eager within bf16 rounding.
            tol = 1e-2 * e.float().abs().max().item()
            assert (a.float() - e.float()).abs().max().item() <= tol, name


@pytest.mark.parametrize("group_mb", [None, 2, 1024])
@pytest.mark.parametrize("head_dim", [128])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_causal_l2_groups(monkeypatch, head_dim, group_mb):
    """Causal dK/dV launch orders over L2 head groups, against the reference: the device
    default (half the L2), groups of one (head, batch) and one group over all of them.
    12 Q heads x 2 sequences: a 2 MB-per-head budget of 16 heads leaves a residual group."""
    import flash_attn.cute.sm120.bwd_runtime as rt

    if group_mb is not None:
        monkeypatch.setattr(rt, "_l2_group_bytes", lambda *args: group_mb << 20)
    seen = _record_dkdv_kernels(monkeypatch)
    lens = [4096, 4096] if head_dim == 64 else [2048, 2048]
    _check(torch.float16, head_dim, (12, 12), False, "causal", False, sink=False, softcap=0.0,
           lens_q=lens)
    if is_fake_mode():
        return
    # The launch uses the runtime's group size: the device default, or the override.
    expect = rt._l2_group_bytes(0) if group_mb is None else group_mb << 20
    assert seen and all(k[0] is True for k in seen), seen
    assert seen.l2 and all(v == expect for v in seen.l2), seen.l2


@pytest.mark.parametrize(
    "mask_name,head_dim,heads,block_size",
    # Every head dim and head layout on one mask, the narrower Q block once, the other masks
    # once.
    [("sliding_window", d, h, (128, 128)) for d in (64, 96, 128) for h in ((8, 8), (8, 2))]
    + [("sliding_window", 128, (8, 2), (64, 128))]
    + [(m, 128, (8, 2), (128, 128)) for m in ("block_diagonal", "mini_causal", "causal")],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_block_sparse(monkeypatch, mask_name, head_dim, heads, block_size):
    _check_bwd_block_sparse(monkeypatch, torch.bfloat16, mask_name, head_dim, heads, block_size)


@pytest.mark.parametrize("mask_name,head_dim", [("mini_causal", 64), ("sliding_window", 128)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_block_sparse_fp16(monkeypatch, mask_name, head_dim):
    _check_bwd_block_sparse(monkeypatch, torch.float16, mask_name, head_dim, (8, 2), (128, 128))


def _check_bwd_block_sparse(monkeypatch, dtype, mask_name, head_dim, heads, block_size):
    """Block-sparse backward (Q-direction lists from create_block_mask, partial and full
    blocks; dense mask_mod forward) against the reference. Q 1000 / K 1100: the last Q block
    is cut at seqlen_q."""
    from torch.nn.attention.flex_attention import create_block_mask

    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    lens_q, lens_k = [1000, 1000], [1100, 1100]
    cute_mask, flex_mask = get_mask_pair(mask_name, lens_q[0], lens_k[0], window_size=200)
    t = create_block_mask(flex_mask, len(lens_q), heads[0], lens_q[0], lens_k[0], device="cuda",
                          BLOCK_SIZE=block_size).as_tuple()
    sparse = BlockSparseTensorsTorch(mask_block_cnt=t[6], mask_block_idx=t[7], full_block_cnt=t[8],
                                     full_block_idx=t[9], block_size=block_size)

    def ref_mask(off_q, off_k, n_q, n_k):
        rows = torch.arange(n_q, device="cuda")[:, None]
        return flex_mask(0, 0, rows, torch.arange(n_k, device="cuda")[None, :]).expand(n_q, n_k)

    seen = _record_dkdv_kernels(monkeypatch)
    mods = dict(mask_mod=cute_mask, block_sparse_tensors_bwd=sparse, ref_mask=ref_mask)
    _check(dtype, head_dim, heads, False, "noncausal", False, sink=False, softcap=0.0,
           lens_q=lens_q, lens_k=lens_k, mods=mods)
    if is_fake_mode():
        return
    assert seen and all(key is not None for key in seen.sparse), seen.keys


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_bwd_block_sparse_unsupported():
    """Block-sparse calls the SM12x backward rejects: deterministic, wide head dims, and
    block sizes that are not a multiple of the kernel tile."""
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_bwd

    def call(head_dim=128, block_size=(128, 128), deterministic=False, head_dim_v=None):
        b, s, h = 1, 256, 2
        q = torch.randn(b, s, h, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, s, h, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(b, s, h, head_dim_v or head_dim, device="cuda", dtype=torch.bfloat16)
        out = torch.zeros_like(q[..., : v.shape[-1]])
        lse = torch.zeros(b, h, s, device="cuda")
        nm, nn = -(-s // block_size[0]), -(-s // block_size[1])
        sparse = BlockSparseTensorsTorch(
            mask_block_cnt=torch.zeros(b, h, nn, dtype=torch.int32, device="cuda"),
            mask_block_idx=torch.zeros(b, h, nn, nm, dtype=torch.int32, device="cuda"),
            block_size=block_size,
        )
        _flash_attn_bwd(q, k, v, out, torch.zeros_like(out), lse, deterministic=deterministic,
                        block_sparse_tensors=sparse)

    with pytest.raises(NotImplementedError, match="deterministic"):
        call(deterministic=True)
    with pytest.raises(NotImplementedError, match="head_dim"):
        call(head_dim=256)
    with pytest.raises(ValueError, match="multiple of the backward tile"):
        call(block_size=(128, 64))
