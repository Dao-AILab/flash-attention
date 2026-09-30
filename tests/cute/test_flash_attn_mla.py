# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# MLA (qv) attention tests: absorbed dense and sparse (top-k gather) forward / backward, paged,
# the 1CTA / kb64 kernels (FLASH_ATTENTION_MLA_1CTA=1) and the 2CTA kernel. Split out of
# test_flash_attn.py so an MLA-only run collects a few thousand cases instead of ~418K.

import itertools
import math
import os
import random

import pytest
import torch
from einops import rearrange, repeat

from flash_attn.cute.testing import (
    attention_ref,
    check_dsink_vs_ref,
    check_tensor_vs_ref,
    generate_qkv,
    generate_random_padding_mask,
    is_fake_mode,
    maybe_fake_tensor_mode,
    unpad_input,
)
from flash_attn.cute.interface import (
    _flash_attn_bwd_sparse_mla,
    _flash_attn_fwd,
    flash_attn_func,
    flash_attn_varlen_func,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
# Routes MLA-absorbed (qv) calls to the 1CTA kernel where it applies. Sparse (top-k) MLA
# goes to 1CTA for <= 64 heads (training: only with recompute-P); every other
# sparse case falls back to the 2CTA kernel, so the sparse tests run under either setting.
MLA_1CTA = os.environ.get("FLASH_ATTENTION_MLA_1CTA", "0") == "1"
IS_SM90 = torch.cuda.get_device_capability()[0] == 9
IS_SM100 = torch.cuda.get_device_capability()[0] == 10
IS_SM110 = torch.cuda.get_device_capability()[0] == 11
IS_SM120 = torch.cuda.get_device_capability()[0] == 12


def print_diff_stats(name, actual, ref, pt=None, verbose=True):
    if actual is None:
        return
    if pt is not None:
        diff_pt = (pt - ref).abs()
        print(f"{name} Pytorch max diff: {diff_pt.max().item()}")
        print(f"{name} Pytorch mean diff: {diff_pt.mean().item()}")
    diff = (actual - ref).abs()
    print(f"{name} max diff: {diff.max().item()}")
    print(f"{name} mean diff: {diff.mean().item()}")
    if verbose:
        coords = torch.unravel_index(diff.argmax(), diff.shape)
        print(f"  at coordinates {tuple(c.item() for c in coords)}: {name}={actual[coords].item()}, {name}_ref={ref[coords].item()}")


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mqa"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("local_enum", [0])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("hdim", [64])
@pytest.mark.parametrize("kv_sparsity", [False, True])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (3, 3),
        (3, 128),
        (128, 3),
        (256, 256),
        (1025, 255),
        (255, 1025),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (1, 8192),
        (4096, 4096),
    ],
)
# @pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize("seed", [0])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_absorbed(
    seqlen_q,
    seqlen_k,
    hdim,
    causal,
    local_enum,
    deterministic,
    has_learnable_sink,
    mha_type,
    dtype,
    kv_sparsity,
    shared_kv,
    seed,
):
    check_fwd_deterministic = True
    test_bwd = kv_sparsity is True
    hdimv = 512
    if not IS_SM100:
        pytest.skip()
    local = local_enum > 0
    if local and causal:
        pytest.skip()
    if local:
        pytest.xfail("mla absorbed: local not supported yet")
    device = "cuda"
    # set seed
    seed = seed
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    batch_size = 12 if seqlen_q <= 512 else 3 if seqlen_q <= 2048 else 1
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    # 24 heads pad to the 64-head bwd tile and exercise the sink reduction on padded dpsum.
    nheads_vals = [128, 24] if kv_sparsity else [16, 128]
    seqlen_k_base = max(min(seqlen_k // 256 * 256, 1024), 256)
    gather_kv_lengths = [seqlen_k_base - 128, seqlen_k_base] if kv_sparsity else [0]
    seqlen_k_og = seqlen_k
    for nheads, gather_kv_length in itertools.product(nheads_vals, gather_kv_lengths):
        nheads_kv = nheads if mha_type == "mha" else (8 if mha_type == "gqa" else 1)
        print(f"\n{batch_size=}, {nheads=}, {nheads_kv=}, {gather_kv_length=}")
        if kv_sparsity and seqlen_k < gather_kv_length:
            seqlen_k = seqlen_k_og + gather_kv_length
        q_ref = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdim, device=device, dtype=dtype).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdimv, device=device, dtype=dtype).requires_grad_()
        qv_ref = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype).requires_grad_()
        if kv_sparsity:
            gather_kv_indices = torch.rand(batch_size, seqlen_q, gather_kv_length, device=device).argsort(dim=-1).to(torch.int32)
        else:
            gather_kv_indices = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (
            (None, None) if not local else tuple(random.randrange(0, seqlen_k) for _ in range(2))
        )
        if local_enum == 2:
            window_size = (None, -window_size[1])
        elif local_enum == 3:
            window_size = (-window_size[0], None)
        if local:
            print("window size = ", window_size)
        # window_size = (-1, -1) if not local else (16, 0)
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device, requires_grad=True)
        else:
            learnable_sink = None
        q, k, v, qv = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref, qv_ref)]
        if shared_kv:
            q, k, qv = qv, v, None
            q_ref, k_ref, qv_ref = qv_ref, v_ref, None
        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            causal=causal,
            qv=qv_ref,
            window_size=window_size,
            learnable_sink=learnable_sink,
            gather_kv_indices=gather_kv_indices,
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            causal=causal,
            qv=qv_ref,
            window_size=window_size,
            learnable_sink=learnable_sink,
            upcast=False,
            reorder_ops=True,
            gather_kv_indices=gather_kv_indices,
        )

        # Numerical error if we just do any arithmetic on out_ref
        if not is_fake_mode():
            fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
            rtol = 2
            print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
            print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        # SplitKV with qv is 1CTA-only, and never for sparse MLA (rejected on both kernels).
        num_splits_vals = [1, 3] if MLA_1CTA and not DISABLE_SPLIT and not kv_sparsity else [1]
        pack_gqa_vals = [True]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out, lse = flash_attn_func(
                q,
                k,
                v,
                qv=qv,
                gather_kv_indices=gather_kv_indices,
                causal=causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                pack_gqa=pack_gqa,
                num_splits=num_splits,
                deterministic=deterministic,
            )
            if is_fake_mode():
                # no more flash_attn cutedsl calls for the rest of the loop
                # skip data-dependent postprocessing
                continue
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most twice the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (
                out_pt - out_ref
            ).abs().max().item() + fwd_atol
            assert not torch.isnan(lse).any(), "LSE contains NaN"

            repeats = 10 if check_fwd_deterministic else 0
            for iter in range(repeats):
                out2, lse2 = flash_attn_func(
                    q,
                    k,
                    v,
                    qv=qv,
                    gather_kv_indices=gather_kv_indices,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    pack_gqa=pack_gqa,
                    num_splits=num_splits,
                )
                assert torch.equal(out, out2), f"non-deterministic with max diff = {(out - out2).abs().max().item()} on {iter=}"
        
        if test_bwd:
            print("BWD SPARSE MLA")
            g = torch.randn_like(out)
            sink_inputs = (learnable_sink,) if has_learnable_sink else ()
            inputs = ((q, k) if shared_kv else (q, k, v, qv)) + sink_inputs
            inputs_ref = ((q_ref, k_ref) if shared_kv else (q_ref, k_ref, v_ref, qv_ref)) + sink_inputs
            grads = torch.autograd.grad(out, inputs, g)

            if is_fake_mode():
                continue

            grads_ref = torch.autograd.grad(out_ref, inputs_ref, g)
            grads_pt = torch.autograd.grad(out_pt, inputs_ref, g)
            for name, actual, ref, pt in zip(("dQ", "dK", "dV", "dQv"), grads, grads_ref, grads_pt):
                print_diff_stats(name, actual, ref, pt)
                check_tensor_vs_ref(name, actual, ref, pt)
            if has_learnable_sink:
                check_dsink_vs_ref(grads[-1], grads_ref[-1], grads_pt[-1])


@pytest.mark.skipif(not MLA_1CTA, reason="SplitKV with qv is implemented by the 1CTA MLA kernel only")
@pytest.mark.parametrize("varlen_q", ["none", "cu_seqlens_q", "seqused_q"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_split_distribution(varlen_q, monkeypatch):
    """Every split must own KV work, not just produce the right combined output.

    Catches the silent failure modes where split-KV degrades to one split doing all the
    work: the interface forcing num_splits=1 (combine never runs), and the kernel's KV
    range ignoring num_splits (splits >= 1 come out empty with LSE = -inf).
    """
    if not IS_SM100:
        pytest.skip()
    import flash_attn.cute.interface as fa_interface

    captured = []
    real_combine = fa_interface._flash_attn_fwd_combine

    def recording_combine(out_partial, lse_partial, *args, **kwargs):
        captured.append(lse_partial.clone())
        return real_combine(out_partial, lse_partial, *args, **kwargs)

    # _flash_attn_fwd_combine reaches its JIT cache through its module-global name.
    recording_combine.compile_cache = real_combine.compile_cache

    torch.random.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    batch_size, seqlen_q, seqlen_k, nheads, hdim, hdimv, num_splits = 2, 4, 8192, 128, 64, 512, 3
    q = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype)
    qv = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, 1, hdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, 1, hdimv, device=device, dtype=dtype)
    seqused_q = torch.tensor([seqlen_q, seqlen_q - 1], device=device, dtype=torch.int32)

    def run(num_splits):
        if varlen_q == "none":
            return flash_attn_func(q, k, v, qv=qv, num_splits=num_splits, return_lse=True)
        if varlen_q == "cu_seqlens_q":
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32)
            out, lse = flash_attn_varlen_func(
                q.flatten(0, 1), k, v, qv=qv.flatten(0, 1), cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=seqlen_q, num_splits=num_splits, return_lse=True,
            )
            return out.unflatten(0, (batch_size, seqlen_q)), lse.unflatten(0, (batch_size, seqlen_q))
        return flash_attn_varlen_func(
            q, k, v, qv=qv, seqused_q=seqused_q, num_splits=num_splits, return_lse=True,
        )

    out_1, lse_1 = run(1)
    monkeypatch.setattr(fa_interface, "_flash_attn_fwd_combine", recording_combine)
    out_s, lse_s = run(num_splits)

    assert len(captured) == 1, "num_splits=3 did not reach the combine kernel (split-KV collapsed to 1)"
    lse_partial = captured[0]  # (num_splits, batch, seqlen_q, nheads) or (num_splits, total_q, nheads)
    assert lse_partial.shape[0] == num_splits
    if is_fake_mode():  # the combine kernel is reached (and compiled) in fake mode too
        return
    if varlen_q == "cu_seqlens_q":
        lse_partial = lse_partial.unflatten(1, (batch_size, seqlen_q))
    for b in range(batch_size):
        rows = int(seqused_q[b]) if varlen_q == "seqused_q" else seqlen_q
        for split in range(num_splits):
            assert torch.isfinite(lse_partial[split, b, :rows]).all(), (
                f"split {split} of batch {b} did no KV work (non-finite partial LSE)"
            )
    for b in range(batch_size):
        rows = int(seqused_q[b]) if varlen_q == "seqused_q" else seqlen_q
        torch.testing.assert_close(out_s[b, :rows], out_1[b, :rows], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse_s[b, :rows], lse_1[b, :rows], atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA MLA cp.async KV path")
@pytest.mark.parametrize("has_qk", [True, False])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_paged_clc_bitwise(has_qk):
    """The persistent (CLC) scheduler only changes which CTA runs a tile, so its output
    must be bitwise identical to the non-persistent one. Regression test for the sO /
    V-stage-0 overlap: the cp.async KV gather (paged, page_size != tile_n) must wait for
    the previous tile's TMA O store (sO overlays V).
    Needs >1 tile per CTA (CLC, causal so the interface keeps CLC on) and the TMA O store
    (dense batched Q)."""
    if not IS_SM100:
        pytest.skip()
    import flash_attn.cute.utils as fa_utils

    torch.random.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    b, s, h, page_size = 4, 2048, 16, 16
    q = torch.randn(b, s, h, 64, device=device, dtype=dtype) if has_qk else None
    qv = torch.randn(b, s, h, 512, device=device, dtype=dtype)
    k = torch.randn(b, s, 1, 64, device=device, dtype=dtype)
    v = torch.randn(b, s, 1, 512, device=device, dtype=dtype)
    num_pages = b * s // page_size
    perm = torch.randperm(num_pages, device=device)
    page_table = perm.view(b, s // page_size).to(torch.int32)
    to_pages = lambda x: x.reshape(num_pages, page_size, 1, x.shape[-1])[perm.argsort()].contiguous()  # noqa: E731
    k_p, v_p = to_pages(k), to_pages(v)
    if not has_qk:
        q, k_p, qv = qv, v_p, None  # shared_kv: interface routes it as qv-only MLA
    seqused_k = torch.full((b,), s, dtype=torch.int32, device=device)
    outs = []
    saved = fa_utils._fa_clc_enabled
    try:
        for clc in (False, True):
            fa_utils._fa_clc_enabled = clc
            outs.append(flash_attn_varlen_func(
                q, k_p, v_p, qv=qv, causal=True, page_table=page_table,
                seqused_k=seqused_k, max_seqlen_q=s, return_lse=True,
            ))
    finally:
        fa_utils._fa_clc_enabled = saved
    if is_fake_mode():
        return
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def _mla_sink_ref(q, qv, k, v, softmax_scale, causal, sink):
    """fp32 MLA-absorbed reference returning (out, lse); the sink is one extra logit per Q head."""
    q, qv, k, v = [t.float() for t in (q, qv, k, v)]
    nheads, nheads_kv = q.shape[2], k.shape[2]
    k, v = [repeat(t, "b s h d -> b s (h g) d", g=nheads // nheads_kv) for t in (k, v)]
    scores = (torch.einsum("bshd,bthd->bhst", q, k) + torch.einsum("bshd,bthd->bhst", qv, v)) * softmax_scale
    seqlen_q, seqlen_k = scores.shape[-2:]
    if causal:
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        scores = scores.masked_fill(col > row + seqlen_k - seqlen_q, float("-inf"))
    sink_logit = sink.float().view(1, nheads, 1, 1).expand(*scores.shape[:-1], 1)
    logits = torch.cat([scores, sink_logit], dim=-1)
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :-1]
    out = torch.einsum("bhst,bthd->bshd", probs, v)
    return out, lse.transpose(1, 2)


@pytest.mark.skipif(not MLA_1CTA, reason="learnable sink test for the 1CTA MLA kernel")
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("causal", [False, True])
# 48 Q heads per KV head does not divide the 64-row tile; 64 runs the dense kb64 mainloop
@pytest.mark.parametrize("nheads", [16, 48, 64])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(64, 1024), (300, 200)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_learnable_sink(seqlen_q, seqlen_k, nheads, causal, num_splits, return_lse):
    """Sink folded into the 1CTA epilogue: pack_gqa head indexing, split-KV (split 0 owns
    the sink), fully masked rows, and the no-LSE path (row_max still has to reach the
    epilogue when mLSE is None)."""
    if not IS_SM100:
        pytest.skip()
    torch.random.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    batch_size, hdim, hdimv = 2, 64, 512
    q = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype)
    qv = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, 1, hdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, 1, hdimv, device=device, dtype=dtype)
    sink = torch.randn(nheads, device=device, dtype=dtype) * 4
    softmax_scale = 1.0 / math.sqrt(hdim + hdimv)
    out, lse = flash_attn_func(
        q, k, v, qv=qv, causal=causal, learnable_sink=sink, num_splits=num_splits,
        return_lse=return_lse,
    )
    if is_fake_mode():
        return
    out_ref, lse_ref = _mla_sink_ref(q, qv, k, v, softmax_scale, causal, sink)
    assert (out.float() - out_ref).abs().max().item() <= 2e-2
    # Rows with no visible key (causal, seqlen_q > seqlen_k) keep only the sink: O = 0.
    masked_rows = seqlen_q - seqlen_k if causal and seqlen_q > seqlen_k else 0
    if masked_rows > 0:
        assert (out[:, :masked_rows] == 0).all()
    if not return_lse:
        assert lse is None
        return
    if num_splits > 1:
        # A split-KV tile with no KV blocks at all never reaches split 0's epilogue, so its
        # LSE stays -inf instead of the sink (same as flash_fwd_sm100); O is still 0.
        lse, lse_ref = lse[:, masked_rows:], lse_ref[:, masked_rows:]
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-3)


def _mla_dense_ref(q, qv, k, v, causal, seqlen_k_used=None):
    """fp32 MLA-absorbed reference, (out, lse) with lse (b, s_q, h); q may be None (no rope).
    Rows with no visible key are NaN in out / -inf in lse."""
    k_lat = v.float()[:, :, 0]
    scale = (qv.shape[-1] + (q.shape[-1] if q is not None else 0)) ** -0.5
    s = torch.einsum("bqhd,bkd->bhqk", qv.float(), k_lat)
    if q is not None:
        s = s + torch.einsum("bqhd,bkd->bhqk", q.float(), k.float()[:, :, 0])
    s = s * scale
    s_q, s_k = s.shape[-2:]
    lim = torch.full((s_q,), s_k if seqlen_k_used is None else seqlen_k_used, device=s.device)
    if causal:
        lim = torch.minimum(lim, torch.arange(s_q, device=s.device) + 1 + lim - s_q)
    s = s.masked_fill(torch.arange(s_k, device=s.device)[None, :] >= lim[:, None], float("-inf"))
    lse = torch.logsumexp(s, -1)
    out = torch.einsum("bhqk,bkd->bqhd", torch.softmax(s, -1), k_lat)
    return out, lse.transpose(1, 2)


def _check_mla_vs_ref(out, lse, out_ref, lse_ref, what=""):
    valid = torch.isfinite(lse_ref)  # (b, s_q, h): rows with >= 1 visible key
    assert (out[~valid] == 0).all() and torch.isneginf(lse[~valid]).all(), what
    if valid.any():
        o, r = out.float()[valid], out_ref[valid]
        rel = ((o - r).norm() / r.norm()).item()
        assert rel < 1e-2, f"{what}: out rel-L2 vs reference {rel:.2e}"
        assert (lse[valid] - lse_ref[valid]).abs().max().item() < 1e-3, what


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA dense MLA, 64-key-block mainloop")
@pytest.mark.parametrize("num_splits", [1, 3, 0])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(1, 8192), (1, 100), (1, 63), (1, 64), (1, 3), (4, 777), (300, 200), (128, 1024)],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("has_qk", [True, False])
@pytest.mark.parametrize("nheads", [64, 16, 24])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_dense_kb64(nheads, has_qk, causal, seqlen_q, seqlen_k, num_splits,
                                        monkeypatch):
    """Dense MLA at 64 heads runs the 64-key-block (kb64) mainloop: TMA loads per latent part,
    positional masking, a runtime block count (1 block, odd counts, fully masked causal rows
    with a dummy block) and split-KV (explicit, heuristic, and empty splits: s_k = 3 has one
    block for 3 splits). Checked against the fp32 reference, bitwise run to run, and against
    the 2CTA kernel (FLASH_ATTENTION_MLA_1CTA=0, unsplit: 2CTA has no MLA split-KV) under the
    bf16-rounding contract (head counts dividing 128). Fewer than 64 heads pad the one-token tile in-kernel; with fewer
    than 64 heads only decode (seqlen_q = 1) routes to kb64, prefill shapes run the 128-key
    mainloop."""
    if not IS_SM100:
        pytest.skip()
    if not _mla_kb64_active(nheads):
        pytest.skip("kb64 mainloop disabled")
    b = 2
    kw, (q_r, k_r, v_r, qv_r) = _mla_inputs(b, seqlen_q, seqlen_k, nheads, has_qk)
    call = dict(q=kw["q"] if has_qk else None, k=kw["k"] if has_qk else None, v=kw["v"],
                qv=kw["qv"] if has_qk else kw["q"], causal=causal, num_splits=num_splits,
                return_lse=True)
    # every variant first (fake mode compiles them all, then returns)
    out, lse, *_ = _flash_attn_fwd(**call)
    out_again, lse_again, *_ = _flash_attn_fwd(**call)
    # the 2CTA dense kernel packs heads into its 128-row tile only for ratios dividing 128
    vs_2cta = 128 % nheads == 0
    if vs_2cta:
        monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
        out_2cta, lse_2cta, *_ = _flash_attn_fwd(**dict(call, num_splits=1))
    if is_fake_mode():
        return
    assert torch.equal(out, out_again) and torch.equal(lse, lse_again), "not deterministic"
    out_ref, lse_ref = _mla_dense_ref(q_r if has_qk else None, qv_r if has_qk else q_r, k_r, v_r, causal)
    _check_mla_vs_ref(out, lse, out_ref, lse_ref, "kb64 vs reference")
    if vs_2cta:
        _assert_mla_fwd_close(out, out_2cta, lse, lse_2cta, "kb64 vs 2CTA")


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA dense MLA, 64-key-block mainloop")
@pytest.mark.parametrize("num_splits", [1, 3, 0])
@pytest.mark.parametrize("nheads", [64, 16])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_dense_kb64_packed_varlen_decode(nheads, num_splits, monkeypatch):
    """Dense kb64 with cu_seqlens_q schedules a flat grid over the tokens (packed varlen, one
    token per tile). Many single-token sequences with ragged s_k (including 1 and multiples of
    64), split-KV: close to the 2CTA kernel (unsplit) and a sample of sequences against the
    reference."""
    if not IS_SM100:
        pytest.skip()
    if not _mla_kb64_active(nheads):
        pytest.skip("kb64 mainloop disabled")
    device, dtype = "cuda", torch.bfloat16
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 300
    with torch._subclasses.fake_tensor.unset_fake_temporarily():  # host-side lengths, real
        seqlens_k = torch.randint(1, 3000, (n,), generator=g).tolist()
    seqlens_k[:4] = [1, 64, 128, 2999]
    cu = lambda lens: torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=device)  # noqa: E731
    torch.random.manual_seed(0)
    q = torch.randn(n, nheads, 64, device=device, dtype=dtype)
    qv = torch.randn(n, nheads, 512, device=device, dtype=dtype)
    k = torch.randn(sum(seqlens_k), 1, 64, device=device, dtype=dtype)
    v = torch.randn(sum(seqlens_k), 1, 512, device=device, dtype=dtype)
    call = dict(qv=qv, num_splits=num_splits, cu_seqlens_q=cu([1] * n), cu_seqlens_k=cu(seqlens_k),
                max_seqlen_q=1, max_seqlen_k=max(seqlens_k), return_lse=True)
    # every variant first (fake mode compiles them all, then returns)
    out, lse, *_ = _flash_attn_fwd(q, k, v, **call)
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    out_2cta, lse_2cta, *_ = _flash_attn_fwd(q, k, v, **dict(call, num_splits=1))
    if is_fake_mode():
        return
    starts = list(itertools.accumulate([0] + seqlens_k))
    for i in [0, 1, 2, 3, 57, 150, 299]:
        ks, ke = starts[i], starts[i + 1]
        o_ref, l_ref = _mla_dense_ref(q[i][None, None], qv[i][None, None], k[ks:ke][None], v[ks:ke][None], False)
        _check_mla_vs_ref(out[i][None, None], lse[i][None, None], o_ref, l_ref, f"sequence {i}")
    _assert_mla_fwd_close(out, out_2cta, lse, lse_2cta, "kb64 vs 2CTA")


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA dense MLA, 64-key-block mainloop")
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("nheads", [1, 24, 48])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_dense_kb64_padded_head_canary(nheads, has_learnable_sink):
    """Dense kb64 with fewer than 64 heads pads each token's tile: the padded rows alias the next
    token's heads (and run past the tensor for the last token), so they must never be written.
    out / lse are views into canary-filled buffers whose tails must survive; every row matches
    the reference. Decode (seqlen_q = 1): the shape that routes fewer than 64 heads to kb64."""
    if not IS_SM100:
        pytest.skip()
    if not _mla_kb64_active(nheads):
        pytest.skip("kb64 mainloop disabled")
    b, s_q, s_k = 33, 1, 1000
    kw, (q_r, k_r, v_r, qv_r) = _mla_inputs(b, s_q, s_k, nheads, has_qk=True)
    sink = torch.randn(nheads, device="cuda", dtype=torch.bfloat16) * 4 if has_learnable_sink else None
    canary_out, canary_lse, pad = -777.0, -555.0, 64 * 512 * 4
    n_out, n_lse = b * s_q * nheads * 512, b * s_q * nheads
    buf_out = torch.full((n_out + pad,), canary_out, device="cuda", dtype=torch.bfloat16)
    buf_lse = torch.full((n_lse + pad,), canary_lse, device="cuda", dtype=torch.float32)
    out = buf_out[:n_out].view(b, s_q, nheads, 512)
    lse = buf_lse[:n_lse].view(b, s_q, nheads)
    _flash_attn_fwd(kw["q"], kw["k"], kw["v"], qv=kw["qv"], learnable_sink=sink, out=out, lse=lse,
                    causal=True, return_lse=True)
    if is_fake_mode():
        return
    torch.cuda.synchronize()
    assert (buf_out[n_out:] == canary_out).all(), "padded-head O rows written past the tensor"
    assert (buf_lse[n_lse:] == canary_lse).all(), "padded-head LSE written past the tensor"
    if sink is None:
        out_ref, lse_ref = _mla_dense_ref(q_r, qv_r, k_r, v_r, causal=True)
        _check_mla_vs_ref(out, lse, out_ref, lse_ref, "padded kb64 vs reference")
    else:
        out_ref, lse_ref = _mla_sink_ref(q_r, qv_r, k_r, v_r, 1.0 / math.sqrt(576), True, sink)
        assert (out.float() - out_ref).abs().max().item() <= 2e-2
        torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA dense MLA, 64-key-block mainloop")
@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize(
    "mode",
    ["cu_seqlens", "seqused_q", "paged64", "paged128", "paged256",
     "paged1", "paged16", "paged48", "paged96"],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads", [64, 16])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_dense_kb64_varlen_paged(nheads, causal, mode, num_splits, monkeypatch):
    """Dense kb64 with varlen Q / K (ragged lengths incl. 0 and 1) and paged KV (shuffled
    pages): TMA for page_size 64 / 128 / 256 (1-4 blocks per page), the cp.async gather for
    pages that are not whole blocks (1 / 16 / 48 / 96: blocks span pages, pages straddle split
    boundaries). Paged runs are bitwise equal to the same kernel on contiguous KV (the gather
    zero-fills rows past seqlen_k, masked to P = 0 either way); every sequence matches the
    reference; the 2CTA kernel (unsplit) agrees under the bf16-rounding contract. Page 16 also
    checks the no-rope path bitwise."""
    if not IS_SM100:
        pytest.skip()
    if not _mla_kb64_active(64):
        pytest.skip("kb64 mainloop disabled")
    device, dtype, h = "cuda", torch.bfloat16, nheads
    torch.random.manual_seed(0)
    # fewer than 64 heads reach kb64 on decode only (max_seqlen_q = 1)
    seqlens_q = [3, 0, 1, 70, 1] if nheads == 64 else [1, 0, 1, 1, 1]
    seqlens_k = [900, 257, 64, 1500, 1]
    b, s_q_max, s_k_max = len(seqlens_q), max(seqlens_q), max(seqlens_k)
    cu = lambda lens: torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=device)  # noqa: E731
    qs = [torch.randn(s, h, 64, device=device, dtype=dtype) for s in seqlens_q]
    qvs = [torch.randn(s, h, 512, device=device, dtype=dtype) for s in seqlens_q]
    ks = [torch.randn(s, 1, 64, device=device, dtype=dtype) for s in seqlens_k]
    vs = [torch.randn(s, 1, 512, device=device, dtype=dtype) for s in seqlens_k]

    def pad(xs, s_max):
        out = torch.zeros(len(xs), s_max, *xs[0].shape[1:], device=device, dtype=xs[0].dtype)
        for i, x in enumerate(xs):
            out[i, : x.shape[0]] = x
        return out

    seqused_k = torch.tensor(seqlens_k, dtype=torch.int32, device=device)
    if mode == "cu_seqlens":
        q_in, qv_in = torch.cat(qs), torch.cat(qvs)
        extra = dict(cu_seqlens_q=cu(seqlens_q), max_seqlen_q=s_q_max, cu_seqlens_k=cu(seqlens_k),
                     max_seqlen_k=s_k_max)
        k_in, v_in = torch.cat(ks), torch.cat(vs)
    else:
        q_in, qv_in = pad(qs, s_q_max), pad(qvs, s_q_max)
        extra = dict(seqused_q=torch.tensor(seqlens_q, dtype=torch.int32, device=device),
                     max_seqlen_q=s_q_max, seqused_k=seqused_k)
        k_in, v_in = pad(ks, s_k_max), pad(vs, s_k_max)
    call = dict(q=q_in, k=k_in, v=v_in, qv=qv_in, causal=causal, num_splits=num_splits,
                return_lse=True, **extra)
    # every variant first (fake mode compiles them all, then returns), the checks after
    out, lse, *_ = _flash_attn_fwd(**call)
    paged = {}
    if mode.startswith("paged"):
        ps = int(mode[len("paged"):])
        npg = (s_k_max + ps - 1) // ps
        perm = torch.randperm(b * npg, device=device).to(torch.int32)
        page_table = perm.view(b, npg)
        # page j of sequence i holds its keys j*ps ..: scatter the zero-padded KV page-wise
        k_cache = torch.empty(b * npg, ps, 1, 64, device=device, dtype=dtype)
        v_cache = torch.empty(b * npg, ps, 1, 512, device=device, dtype=dtype)
        k_cache[page_table.view(-1).long()] = pad(ks, npg * ps).view(b * npg, ps, 1, 64)
        v_cache[page_table.view(-1).long()] = pad(vs, npg * ps).view(b * npg, ps, 1, 512)
        call_p = dict(call, k=k_cache, v=v_cache, page_table=page_table)
        paged["paged != contiguous"] = (_flash_attn_fwd(**call_p)[:2], (out, lse))
        if ps == 16:
            # no rope part: the gather skips the K rows
            nope = dict(q=None, k=None)
            paged["no rope"] = (_flash_attn_fwd(**dict(call_p, **nope))[:2],
                                _flash_attn_fwd(**dict(call, **nope))[:2])
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    out_2cta, lse_2cta, *_ = _flash_attn_fwd(**dict(call, num_splits=1))
    if is_fake_mode():
        return
    # rows past seqused_q are never written (uninitialized in both)
    vq = torch.arange(s_q_max, device=device)[None] < torch.tensor(seqlens_q, device=device)[:, None]
    for what, ((o_a, l_a), (o_b, l_b)) in paged.items():
        assert torch.equal(o_a[vq], o_b[vq]) and torch.equal(l_a[vq], l_b[vq]), what
    starts_q = list(itertools.accumulate([0] + seqlens_q))
    for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        if sq == 0:
            continue
        if mode == "cu_seqlens":
            rows = slice(starts_q[i], starts_q[i + 1])
            o, l = out[rows][None], lse[rows][None]
        else:
            o, l = out[i: i + 1, :sq], lse[i: i + 1, :sq]
        o_ref, l_ref = _mla_dense_ref(qs[i][None], qvs[i][None], ks[i][None], vs[i][None], causal)
        _check_mla_vs_ref(o, l, o_ref, l_ref, f"sequence {i}")
    if mode != "cu_seqlens":
        # rows past seqused_q are not written by either kernel
        out, out_2cta, lse, lse_2cta = out[vq], out_2cta[vq], lse[vq], lse_2cta[vq]
    _assert_mla_fwd_close(out, out_2cta, lse, lse_2cta, "kb64 vs 2CTA")


@pytest.mark.skipif(not MLA_1CTA, reason="learnable sink test for the 1CTA MLA kernel")
@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("causal", [False, True])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_learnable_sink_fp8(causal, num_splits):
    """fp8 + descales + sink: the sink must enter with the fp8 max_offset pre-scale and the
    descale-folded softmax scale. Checked exactly against the same fp8 call without a sink
    (lse' = logaddexp(lse, sink), O' = O * exp(lse - lse')), then loosely against a
    dequantized fp32 reference."""
    if not IS_SM100:
        pytest.skip()
    torch.random.manual_seed(0)
    device, fp8 = "cuda", torch.float8_e4m3fn
    batch_size, seqlen_q, seqlen_k, nheads, hdim, hdimv = 2, 64, 1024, 16, 64, 512
    q, qv, k, v = [
        torch.randn(*shape, device=device).to(fp8)
        for shape in (
            (batch_size, seqlen_q, nheads, hdim),
            (batch_size, seqlen_q, nheads, hdimv),
            (batch_size, seqlen_k, 1, hdim),
            (batch_size, seqlen_k, 1, hdimv),
        )
    ]
    q_descale = torch.rand(batch_size, 1, device=device) + 0.5
    kv_descale = torch.rand(batch_size, 1, device=device) + 0.5
    sink = torch.randn(nheads, device=device, dtype=torch.bfloat16) * 4
    softmax_scale = 1.0 / math.sqrt(hdim + hdimv)

    def run(learnable_sink):
        # Descales are only exposed by the internal entry point.
        out, lse, *_ = _flash_attn_fwd(
            q, k, v, qv=qv, causal=causal, learnable_sink=learnable_sink, num_splits=num_splits,
            q_descale=q_descale, k_descale=kv_descale, v_descale=kv_descale, return_lse=True,
        )
        return out, lse

    out, lse = run(None)
    out_sink, lse_sink = run(sink)
    if is_fake_mode():
        return
    lse_expected = torch.logaddexp(lse, sink.float().view(1, 1, nheads))
    torch.testing.assert_close(lse_sink, lse_expected, atol=1e-3, rtol=1e-4)
    out_expected = out.float() * torch.exp(lse - lse_expected).unsqueeze(-1)
    torch.testing.assert_close(out_sink.float(), out_expected, atol=1e-2, rtol=1e-2)

    deq = lambda t, d: t.float() * d.view(batch_size, 1, 1, 1)  # noqa: E731
    out_ref, lse_ref = _mla_sink_ref(
        deq(q, q_descale), deq(qv, q_descale), deq(k, kv_descale), deq(v, kv_descale),
        softmax_scale, causal, sink,
    )
    torch.testing.assert_close(lse_sink, lse_ref, atol=5e-2, rtol=1e-2)
    assert (out_sink.float() - out_ref).abs().max().item() <= 0.1 * out_ref.abs().max().item()


def _mla_1cta_fp8_inputs(batch_size=2, seqlen_q=17, seqlen_k=257, nheads=6, nheads_kv=2):
    device, fp8 = "cuda", torch.float8_e4m3fn
    torch.random.manual_seed(0)
    q, qv, k, v = [
        torch.randn(*shape, device=device).to(fp8)
        for shape in (
            (batch_size, seqlen_q, nheads, 64),
            (batch_size, seqlen_q, nheads, 512),
            (batch_size, seqlen_k, nheads_kv, 64),
            (batch_size, seqlen_k, nheads_kv, 512),
        )
    ]
    q_descale = torch.rand(batch_size, nheads_kv, device=device) + 0.5
    kv_descale = torch.rand(batch_size, nheads_kv, device=device) + 0.5
    sink = torch.linspace(-2, 7, nheads, device=device)
    return q, qv, k, v, q_descale, kv_descale, sink


@pytest.mark.skipif(not MLA_1CTA, reason="fp8 descales with qv are 1CTA-only")
@pytest.mark.parametrize("num_splits", [1, 3])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_fp8_descales_cuda_graph(num_splits):
    """The descale checks must not sync the device: capture and replay a CUDA graph of
    the fp8 + descales + sink forward and compare it to the eager result."""
    if not IS_SM100:
        pytest.skip()
    q, qv, k, v, q_descale, kv_descale, sink = _mla_1cta_fp8_inputs()

    def call():
        out, lse, *_ = _flash_attn_fwd(
            q, k, v, qv=qv, causal=True, learnable_sink=sink, num_splits=num_splits,
            q_descale=q_descale, k_descale=kv_descale, v_descale=kv_descale, return_lse=True,
        )
        return out, lse

    out_eager, lse_eager = call()  # also compiles, which must happen outside capture
    if is_fake_mode():
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph, lse_graph = call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_graph, out_eager)
    assert torch.equal(lse_graph, lse_eager)


@pytest.mark.skipif(not MLA_1CTA, reason="fp8 descales with qv are 1CTA-only")
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_fp8_descales_must_be_shared():
    """K and V descales fold into one softmax scale, so they must be one tensor; equal
    values in separate allocations are rejected (checking values would sync the device)."""
    if not IS_SM100:
        pytest.skip()
    q, qv, k, v, q_descale, kv_descale, sink = _mla_1cta_fp8_inputs()
    with pytest.raises(AssertionError, match="k_descale and v_descale to be the same tensor"):
        _flash_attn_fwd(
            q, k, v, qv=qv, q_descale=q_descale, k_descale=kv_descale,
            v_descale=kv_descale.clone(),
        )


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_fp8_descale_without_rope(monkeypatch):
    """Without a rope part (q = k = None) S = Qv @ V^T, so the latent cache's descale is
    v_descale; a lone k_descale would be ignored, so it must be rejected."""
    if not IS_SM100:
        pytest.skip()
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "1")  # fp8 MLA is 1CTA-only
    _, qv, _, v, q_descale, kv_descale, _ = _mla_1cta_fp8_inputs()
    with pytest.raises(AssertionError, match="no rope part"):
        _flash_attn_fwd(None, None, v, qv=qv, q_descale=q_descale, k_descale=kv_descale)
    out, *_ = _flash_attn_fwd(None, None, v, qv=qv, q_descale=q_descale, v_descale=kv_descale)
    out_shared, *_ = _flash_attn_fwd(
        None, None, v, qv=qv, q_descale=q_descale, k_descale=kv_descale, v_descale=kv_descale
    )
    if is_fake_mode():
        return
    assert torch.equal(out, out_shared)


@pytest.mark.parametrize("present", ["q", "kv"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_fp8_partial_descales(present, monkeypatch):
    """q_descale alone or the shared k / v descale alone: a missing one is 1 (bitwise equal
    to passing ones), with no member shifting slots across the kernel boundary."""
    if not IS_SM100:
        pytest.skip()
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "1")  # fp8 MLA is 1CTA-only
    q, qv, k, v, q_descale, kv_descale, sink = _mla_1cta_fp8_inputs()
    ones = torch.ones_like(q_descale)

    def call(qd, kvd):
        return _flash_attn_fwd(
            q, k, v, qv=qv, causal=True, learnable_sink=sink,
            q_descale=qd, k_descale=kvd, v_descale=kvd, return_lse=True,
        )[:2]

    if present == "q":
        (out, lse), (out_ref, lse_ref) = call(q_descale, None), call(q_descale, ones)
    else:
        (out, lse), (out_ref, lse_ref) = call(None, kv_descale), call(ones, kv_descale)
    if is_fake_mode():
        return
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


@pytest.mark.parametrize("max_seqlen_q", ["tensor", None, 1])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_dispatch_varlen_decode_split(max_seqlen_q, monkeypatch):
    """Varlen decode with the split heuristic and no host max_seqlen_q: max_seqlen_q is then
    total_q, a bound on one sequence, so the 1CTA tile count must be bounded by the tokens
    (one kb64 tile each), not batch x total_q. 64 single-token sequences on 64 heads get split
    KV on 1CTA kb64 whether or not the caller passes the host hint."""
    if not IS_SM100:
        pytest.skip()
    import flash_attn.cute.interface as fa_interface
    monkeypatch.delenv("FLASH_ATTENTION_MLA_1CTA", raising=False)
    device, dtype, b, h, s_k = "cuda", torch.bfloat16, 64, 64, 2048
    torch.random.manual_seed(0)
    q = torch.randn(b, h, 64, device=device, dtype=dtype)
    qv = torch.randn(b, h, 512, device=device, dtype=dtype)
    k = torch.randn(b * s_k, 1, 64, device=device, dtype=dtype)
    v = torch.randn(b * s_k, 1, 512, device=device, dtype=dtype)
    cu_q = torch.arange(0, b + 1, device=device, dtype=torch.int32)
    cu_k = torch.arange(0, b + 1, device=device, dtype=torch.int32) * s_k
    m = torch.tensor(1, device=device, dtype=torch.int32) if max_seqlen_q == "tensor" else max_seqlen_q
    real_cache = fa_interface._flash_attn_fwd.compile_cache
    keys = []

    class Spy:
        def __contains__(self, key):
            keys.append(key)
            return key in real_cache

        def __getitem__(self, key):
            return real_cache[key]

        def __setitem__(self, key, value):
            real_cache[key] = value

    monkeypatch.setattr(fa_interface._flash_attn_fwd, "compile_cache", Spy())
    out, *_ = _flash_attn_fwd(q, k, v, qv=qv, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                              max_seqlen_q=m, max_seqlen_k=s_k, num_splits=0)
    monkeypatch.setattr(fa_interface._flash_attn_fwd, "compile_cache", real_cache)
    # first two key entries: the 1CTA route and the kb64 mainloop; split KV is in the key too
    assert keys and all(key[0] and key[1] for key in keys), keys
    if is_fake_mode():
        return
    out_ref, _ = attention_ref(q.view(b, 1, h, 64), k.view(b, s_k, 1, 64), v.view(b, s_k, 1, 512),
                               qv=qv.view(b, 1, h, 512))
    out_pt, _ = attention_ref(q.view(b, 1, h, 64), k.view(b, s_k, 1, 64), v.view(b, s_k, 1, 512),
                              qv=qv.view(b, 1, h, 512), upcast=False, reorder_ops=True)
    err = (out.view(b, 1, h, 512).float() - out_ref.float()).abs().max().item()
    err_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert err <= 2 * err_pt + 1e-3, (err, err_pt)


@pytest.mark.parametrize(
    "route",
    [
        ("1", 64, 64, 1), ("1", 64, 64, 3),      # 1CTA kb64, TMA pages
        ("1", 64, 128, 1),                       # 1CTA kb64, TMA pages spanning two blocks
        ("1", 128, 128, 1), ("1", 128, 128, 3),  # 1CTA 128-key, TMA pages
        ("0", 64, 128, 1),                       # 2CTA, TMA pages
        ("1", 64, 16, 1),                        # 1CTA kb64, cp.async gather
    ],
    ids=lambda r: f"flag{r[0]}-h{r[1]}-page{r[2]}-splits{r[3]}",
)
@pytest.mark.parametrize("padding", ["unused_page_nan", "tail_finite_garbage"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_paged_padding_contract(route, padding, monkeypatch):
    """The paged-KV padding contract. Rows past seqused_k inside a partially used page must be
    finite (TMA pages load them, and P = 0 times NaN is NaN), but may hold any finite garbage;
    pages a batch entry does not use may hold anything, including NaN: an entry with
    seqused_k = 0 still loads a fully masked dummy tile from its first page-table entry, and the
    epilogue must write O = 0 for it rather than scale the accumulator. Either way the output
    equals the run with zero padding, bitwise."""
    if not IS_SM100:
        pytest.skip()
    flag, h, page_size, num_splits = route
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", flag)
    device, dtype = "cuda", torch.bfloat16
    k_lens = [70, 0, 130, 0]
    b, cap = len(k_lens), 256
    pages_per = cap // page_size
    num_pages = b * pages_per + 1
    junk = num_pages - 1  # a page no entry uses
    torch.random.manual_seed(0)
    k = torch.randn(num_pages, page_size, 1, 64, device=device, dtype=dtype) * 0.5
    v = torch.randn(num_pages, page_size, 1, 512, device=device, dtype=dtype) * 0.5
    page_table = torch.arange(b * pages_per, device=device, dtype=torch.int32).view(b, pages_per)
    for i, s_k in enumerate(k_lens):
        if s_k == 0:
            page_table[i, 0] = junk
    q = torch.randn(b, 1, h, 64, device=device, dtype=dtype)
    qv = torch.randn(b, 1, h, 512, device=device, dtype=dtype)
    seqused_k = torch.tensor(k_lens, device=device, dtype=torch.int32)
    # the clean cache: zeros in every row past seqused_k and in the unused page
    k_clean, v_clean = k.clone(), v.clone()
    k_bad, v_bad = k.clone(), v.clone()
    if not is_fake_mode():
        k_clean[junk] = 0
        v_clean[junk] = 0
        for i, s_k in enumerate(k_lens):
            if s_k > 0 and s_k % page_size:
                p, off = page_table[i, s_k // page_size].item(), s_k % page_size
                for t in (k_clean, v_clean):
                    t[p, off:] = 0
                for t in (k_bad, v_bad):
                    t[p, off:] = 0 if padding == "unused_page_nan" else 3e4
        if padding == "unused_page_nan":
            k_bad[junk] = float("nan")
            v_bad[junk] = float("nan")
        else:
            k_bad[junk] = 0
            v_bad[junk] = 0

    def run(k_cache, v_cache):
        return _flash_attn_fwd(q, k_cache, v_cache, qv=qv, page_table=page_table,
                               seqused_k=seqused_k, num_splits=num_splits, return_lse=True)[:2]

    out_clean, lse_clean = run(k_clean, v_clean)
    out_bad, lse_bad = run(k_bad, v_bad)
    if is_fake_mode():
        return
    assert torch.equal(out_bad, out_clean), (out_bad - out_clean).abs().nan_to_num(1e9).max()
    assert torch.equal(lse_bad, lse_clean)
    for i, s_k in enumerate(k_lens):
        if s_k == 0:
            assert torch.count_nonzero(out_bad[i]) == 0
            assert torch.isneginf(lse_bad[i]).all()


def rect_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, causal, device, *,
                      fill_frac=1.0, shuffle_slots=True, oob_frac=0.0, seed=0):
    """Top-k index lists for rectangular (s_q != s_k) sparse attention tests.

    Query t may attend keys [0, limit_t) with limit_t = t + 1 + s_k - s_q when causal
    (bottom-right aligned, the kernel's seqlen_k_limit) and s_k otherwise. Each row draws
    min(topk, fill_frac * limit_t) DISTINCT keys spread over its whole valid range and pads
    with -1 (so topk > seqlen_k works). shuffle_slots scatters the valid entries across
    slots; oob_frac turns that fraction of the -1 padding into indices >= limit_t (keys
    the kernel must mask by range, not by sentinel).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.full((batch_size, seqlen_q, topk_len), -1, dtype=torch.int32)
    for b in range(batch_size):
        for t in range(seqlen_q):
            limit = max(0, min(seqlen_k, t + 1 + seqlen_k - seqlen_q) if causal else seqlen_k)
            n = min(topk_len, int(limit * fill_frac))
            if n > 0:
                idx[b, t, :n] = torch.randperm(limit, generator=g)[:n].to(torch.int32)
            n_oob = int((topk_len - n) * oob_frac)
            if n_oob > 0 and limit < seqlen_k:
                idx[b, t, n:n + n_oob] = torch.randint(
                    limit, seqlen_k, (n_oob,), generator=g, dtype=torch.int32
                )
            if shuffle_slots:
                idx[b, t] = idx[b, t][torch.randperm(topk_len, generator=g)]
    return idx.to(device).contiguous()


def _topk_valid_rows(idx, seqlen_q, seqlen_k, causal):
    """(b, s_q) bool: row has at least one slot the kernel treats as valid."""
    t = torch.arange(seqlen_q, device=idx.device).view(1, -1, 1)
    limit = (t + 1 + seqlen_k - seqlen_q).clamp(max=seqlen_k) if causal else seqlen_k
    return ((idx >= 0) & (idx < limit)).any(-1)


def _mla_inputs(b, s_q, s_k, h, has_qk, dtype=torch.bfloat16, seed=0):
    torch.random.manual_seed(seed)
    q = torch.randn(b, s_q, h, 64, device="cuda", dtype=dtype)
    qv = torch.randn(b, s_q, h, 512, device="cuda", dtype=dtype)
    k = torch.randn(b, s_k, 1, 64, device="cuda", dtype=dtype)
    v = torch.randn(b, s_k, 1, 512, device="cuda", dtype=dtype)
    if has_qk:
        return dict(q=q, k=k, v=v, qv=qv), (q, k, v, qv)
    # shared_kv: the interface routes (q=qv, k=v, v=v) as qv-only MLA
    return dict(q=qv, k=v, v=v), (qv, v, v, None)


def _mla_kb64_active(nheads, dtype=torch.bfloat16):
    """Whether an MLA forward with FLASH_ATTENTION_MLA_1CTA=1 runs the 1CTA 64-key-block
    mainloop (at most 64 heads, padded in-kernel, 16-bit; for training only the sparse
    recompute-P route reaches it). Its running max advances per 64 keys
    instead of the 2CTA kernel's 128, so the two agree to bf16 rounding, not bitwise."""
    return (
        MLA_1CTA
        and nheads <= 64
        and dtype in (torch.float16, torch.bfloat16)
    )


def _assert_mla_fwd_close(out, out_other, lse=None, lse_other=None, what=""):
    """The bf16-rounding contract between two sparse-MLA forwards with different block
    orders (PR 2914's): out rel-L2 < 5e-3, the same -inf LSE pattern, finite LSE within 1e-4.
    Measured: out ~2e-3 (two independent bf16 roundings), LSE ~1e-6."""
    rel = ((out.float() - out_other.float()).norm() / out_other.float().norm().clamp_min(1e-30)).item()
    assert rel < 5e-3, f"{what}: out rel-L2 {rel:.2e}"
    if lse is not None:
        fin = torch.isfinite(lse_other)
        assert torch.equal(torch.isfinite(lse), fin), f"{what}: LSE -inf pattern differs"
        if fin.any():
            err = (lse[fin] - lse_other[fin]).abs().max().item()
            assert err < 1e-4, f"{what}: LSE max-abs {err:.2e}"


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA sparse MLA forward")
@pytest.mark.parametrize("gen", ["rect", "rect_oob"])
@pytest.mark.parametrize("topk", [128, 1024])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(1, 8192), (128, 1024), (300, 200), (1024, 1024)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("has_qk", [True, False])
@pytest.mark.parametrize("nheads", [1, 16, 24, 48, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_sparse_fwd(nheads, has_qk, causal, seqlen_q, seqlen_k, topk, gen,
                                        monkeypatch):
    """Inference sparse forward on the 1CTA kernel (<= 64 heads, padded to the 64-row tile):
    vs the reference, and vs the 2CTA kernel (padded to 128) on the same inputs."""
    if not IS_SM100:
        pytest.skip()
    b = 2
    kw, (q_r, k_r, v_r, qv_r) = _mla_inputs(b, seqlen_q, seqlen_k, nheads, has_qk)
    idx = rect_topk_indices(b, seqlen_q, seqlen_k, topk, causal, "cuda",
                            fill_frac=0.5 if gen == "rect_oob" else 1.0,
                            oob_frac=0.5 if gen == "rect_oob" else 0.0)
    kw.update(gather_kv_indices=idx, causal=causal, return_lse=True)
    out, lse = flash_attn_func(**kw)
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    out_2cta, lse_2cta = flash_attn_func(**kw)
    if is_fake_mode():
        return
    # bf16, <= 64 heads under the flag: the kb64 mainloop, whose 64-key blocks give a different
    # running-max sequence than the 2CTA kernel's
    _assert_mla_fwd_close(out, out_2cta, lse, lse_2cta, "1CTA kb64 vs 2CTA")

    valid = _topk_valid_rows(idx, seqlen_q, seqlen_k, causal)
    # rows with no valid slot: O = 0, LSE = -inf (the reference NaNs there)
    assert (out[~valid] == 0).all()
    assert torch.isneginf(lse[~valid]).all()
    ref_args = dict(causal=causal, gather_kv_indices=idx)
    out_ref, _ = attention_ref(q_r, k_r, v_r, qv=qv_r, **ref_args)
    out_pt, _ = attention_ref(q_r, k_r, v_r, qv=qv_r, upcast=False, reorder_ops=True, **ref_args)
    if not valid.any():
        return
    # compare valid rows only: the reference is NaN on rows with no valid slot
    err = (out.float() - out_ref.float()).abs()[valid].max().item()
    err_pt = (out_pt.float() - out_ref.float()).abs()[valid].max().item()
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref)[valid].abs().max().item()
    assert err <= 2 * err_pt + fwd_atol, (err, err_pt, fwd_atol)


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA sparse MLA forward")
@pytest.mark.parametrize("gen", ["rect", "rect_oob"])
@pytest.mark.parametrize("topk", [128, 1024, 2048])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("has_qk", [True, False])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_sparse_kb64(has_qk, causal, topk, gen, monkeypatch):
    """The 64-key-block mainloop (64 heads) against the 2CTA kernel on the same inputs: the
    bf16-rounding contract, bitwise run-to-run, O = 0 / LSE = -inf on rows with no valid
    slot. Its 256-bit O stores need 32-B aligned rows: a caller's `out` that is only 16-B
    aligned is rejected."""
    if not IS_SM100:
        pytest.skip()
    if not _mla_kb64_active(64):
        pytest.skip("kb64 mainloop disabled")
    b, s_q, s_k, h = 2, 97, 4096, 64
    kw, _ = _mla_inputs(b, s_q, s_k, h, has_qk)
    idx = rect_topk_indices(b, s_q, s_k, topk, causal, "cuda",
                            fill_frac=0.5 if gen == "rect_oob" else 1.0,
                            oob_frac=0.5 if gen == "rect_oob" else 0.0)
    kw.update(gather_kv_indices=idx, causal=causal, return_lse=True)
    # every variant first (fake mode compiles them all, then returns)
    out, lse = flash_attn_func(**kw)
    out_again, lse_again = flash_attn_func(**kw)
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    out_2cta, lse_2cta = flash_attn_func(**kw)
    if is_fake_mode():
        return
    assert torch.equal(out, out_again) and torch.equal(lse, lse_again), "not deterministic"
    valid = _topk_valid_rows(idx, s_q, s_k, causal)
    assert (out[~valid] == 0).all() and torch.isneginf(lse[~valid]).all()
    if has_qk:
        # the alignment check reads the data pointer: real tensors only
        buf = torch.empty(out.numel() + 8, device="cuda", dtype=out.dtype)
        out_unaligned = buf[8:].view_as(out)
        assert out_unaligned.data_ptr() % 32 == 16
        monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "1")
        with pytest.raises(AssertionError, match="out must have aligned strides"):
            _flash_attn_fwd(kw["q"], kw["k"], kw["v"], qv=kw["qv"], gather_kv_indices=idx,
                            causal=causal, out=out_unaligned, return_lse=True)
    _assert_mla_fwd_close(out, out_2cta, lse, lse_2cta, "kb64 vs 2CTA")


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA sparse MLA forward")
@pytest.mark.parametrize("has_qk", [True, False])
@pytest.mark.parametrize("nheads", [16, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_sparse_bitmask_mapping(nheads, has_qk):
    """The bit -> S-column mapping of the validity bitmask (128-key mainloop, Layout E: each
    thread owns 64 columns, two 32-slot words; 64-key kb64 mainloop at 64 heads: one word per
    32-key half). The slots below sit on the word boundaries of both geometries. Uniform
    patterns can't catch swapped words or halves, so:
    (1) one valid slot at each word/half boundary, in the first and the last processed
    n_block -- the output must then be exactly that key's V row; (2) a distinct random
    pattern per 32-slot word, checked against the reference."""
    if not IS_SM100:
        pytest.skip()
    b, s_q, s_k, topk = 1, 2, 512, 256
    kw, (q_r, k_r, v_r, qv_r) = _mla_inputs(b, s_q, s_k, nheads, has_qk)
    v_rows = kw["v"]
    for n_block in (0, topk // 128 - 1):
        for slot in (0, 31, 32, 63, 64, 95, 96, 127):
            idx = torch.full((b, s_q, topk), -1, dtype=torch.int32, device="cuda")
            key = 37 + slot  # any distinct key per case
            idx[..., n_block * 128 + slot] = key
            out, _ = flash_attn_func(**kw, gather_kv_indices=idx, return_lse=True)
            if is_fake_mode():  # one compile key for every call here
                return
            want = v_rows[0, key, 0].expand(s_q, nheads, -1)
            assert torch.equal(out[0], want), f"n_block={n_block} slot={slot}"
    # distinct per-word patterns: every 32-slot word differs, no half is a copy of the other
    g = torch.Generator(device="cpu").manual_seed(1)
    words = torch.randint(0, 2**32, (topk // 32,), generator=g, dtype=torch.int64)
    assert len(set(words.tolist())) == len(words)
    bits = ((words.view(-1, 1) >> torch.arange(32)) & 1).flatten().bool()
    keys = torch.randperm(s_k, generator=g)[:topk].to(torch.int32)
    idx = torch.where(bits, keys, torch.full_like(keys, -1)).view(1, 1, topk).expand(b, s_q, topk)
    idx = idx.contiguous().cuda()
    out, _ = flash_attn_func(**kw, gather_kv_indices=idx, return_lse=True)
    out_ref, _ = attention_ref(q_r, k_r, v_r, qv=qv_r, gather_kv_indices=idx)
    out_pt, _ = attention_ref(q_r, k_r, v_r, qv=qv_r, gather_kv_indices=idx, upcast=False,
                              reorder_ops=True)
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA sparse MLA forward")
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("nheads", [1, 24, 48, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_sparse_padded_head_canary(nheads, has_learnable_sink, train, monkeypatch):
    """Padded heads must never be written: with fewer than 64 heads the packed layout maps a
    token's padded rows onto the next token's heads (and past the tensor for the last
    token). out/lse are views into canary-filled buffers; the tails must survive, and every
    real row must match the 2CTA kernel (to bf16 rounding). Covers both the TMA O store (dense Q) and the
    guarded LSE / sink paths. train: the recompute-P training forward (exact max, the o_lo
    residual under the same row guard as O, checked to half an ulp of O)."""
    if not IS_SM100:
        pytest.skip()
    b, s_q, s_k, topk = 2, 33, 1024, 256
    kw, _ = _mla_inputs(b, s_q, s_k, nheads, has_qk=True)
    if train:
        kw = {n: x.detach().requires_grad_() for n, x in kw.items()}
    mode = dict(gather_bwd_recompute_p=True) if train else {}
    idx = rect_topk_indices(b, s_q, s_k, topk, False, "cuda", fill_frac=0.3)
    sink = (torch.randn(nheads, device="cuda", dtype=torch.bfloat16) * 4
            if has_learnable_sink else None)
    canary_out, canary_lse, pad = -777.0, -555.0, 64 * 512 * 4
    n_out, n_lse = b * s_q * nheads * 512, b * s_q * nheads
    buf_out = torch.full((n_out + pad,), canary_out, device="cuda", dtype=torch.bfloat16)
    buf_lse = torch.full((n_lse + pad,), canary_lse, device="cuda", dtype=torch.float32)
    out = buf_out[:n_out].view(b, s_q, nheads, 512)
    lse = buf_lse[:n_lse].view(b, s_q, nheads)
    _, _, _, _, o_lo = _flash_attn_fwd(kw["q"], kw["k"], kw["v"], qv=kw["qv"], gather_kv_indices=idx,
                                       learnable_sink=sink, out=out, lse=lse, return_lse=True, **mode)
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    out_2cta, lse_2cta, *_ = _flash_attn_fwd(
        kw["q"], kw["k"], kw["v"], qv=kw["qv"], gather_kv_indices=idx,
        learnable_sink=sink, return_lse=True, **mode,
    )
    if is_fake_mode():
        return
    torch.cuda.synchronize()
    assert (buf_out[n_out:] == canary_out).all(), "padded-head O rows written past the tensor"
    assert (buf_lse[n_lse:] == canary_lse).all(), "padded-head LSE written past the tensor"
    assert (o_lo is not None) == train
    if train:
        # a padded row's o_lo written onto the next token's heads would break this bound
        _, e = torch.frexp(out.float())
        half_ulp = torch.ldexp(torch.ones_like(out, dtype=torch.float32), e - 9)
        assert (o_lo.float().abs() <= half_ulp)[out != 0].all(), "o_lo"
    # bf16, <= 64 heads under the flag: the kb64 mainloop
    _assert_mla_fwd_close(out, out_2cta, lse, lse_2cta, "1CTA kb64 vs 2CTA")


@pytest.mark.parametrize("nheads", [64, 128])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_no_split_kv(nheads, monkeypatch):
    """Split-KV is disabled for sparse MLA on both kernels. Under FLASH_ATTENTION_MLA_1CTA=1,
    64 heads route to 1CTA and 128 fall back to 2CTA. An explicit num_splits > 1 must
    raise -- checked before config selection, which would otherwise silently turn it into 1
    on the 2CTA route -- and the heuristic (num_splits=0) must resolve to one split: the
    combine kernel never runs."""
    if not IS_SM100:
        pytest.skip()
    import flash_attn.cute.interface as fa_interface
    b, s_q, s_k, topk = 1, 1, 8192, 2048
    kw, _ = _mla_inputs(b, s_q, s_k, nheads, has_qk=True)
    idx = rect_topk_indices(b, s_q, s_k, topk, False, "cuda")
    with pytest.raises(ValueError, match="num_splits=3 is not supported with gather_kv_indices"):
        flash_attn_func(**kw, gather_kv_indices=idx, num_splits=3)
    calls = []
    real_combine = fa_interface._flash_attn_fwd_combine

    def recording_combine(*args, **kwargs):
        calls.append(1)
        return real_combine(*args, **kwargs)

    recording_combine.compile_cache = real_combine.compile_cache
    monkeypatch.setattr(fa_interface, "_flash_attn_fwd_combine", recording_combine)
    out_h, lse_h = flash_attn_func(**kw, gather_kv_indices=idx, num_splits=0, return_lse=True)
    assert not calls, "sparse MLA ran split-KV (combine was called)"
    out_1, lse_1 = flash_attn_func(**kw, gather_kv_indices=idx, num_splits=1, return_lse=True)
    if is_fake_mode():
        return
    assert torch.equal(out_h, out_1) and torch.equal(lse_h, lse_1)


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA sparse MLA forward")
@pytest.mark.parametrize("varlen_k", [False, True])
@pytest.mark.parametrize("q_mode", ["cu_seqlens_q", "seqused_q"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("has_qk", [True, False])
@pytest.mark.parametrize("nheads", [16, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_sparse_varlen(nheads, has_qk, causal, q_mode, varlen_k, monkeypatch):
    """Sparse forward with varlen Q: cu_seqlens_q runs the packed (flat over tokens)
    scheduler with batch-local indexing and keeps the TMA O store (one-token tiles cannot
    straddle sequences; the 64-head kb64 mainloop stores O per row); seqused_q runs the
    varlen scheduler. Ragged lengths include 0 and 1. Checked per sequence against the
    reference, against 2CTA (to bf16 rounding: these run the kb64 mainloop), and for
    writes past the last token (canary tail)."""
    if not IS_SM100:
        pytest.skip()
    device, dtype, topk = "cuda", torch.bfloat16, 256
    seqlens_q = [37, 0, 1, 200, 64, 5]
    seqlens_k = [1024, 300, 256, 777, 1024, 129]
    b = len(seqlens_q)
    s_q_max, s_k_max = max(seqlens_q), max(seqlens_k)
    torch.random.manual_seed(0)
    d_q = 64 if has_qk else 512
    qs = [torch.randn(sq, nheads, d_q, device=device, dtype=dtype) for sq in seqlens_q]
    qvs = [torch.randn(sq, nheads, 512, device=device, dtype=dtype) for sq in seqlens_q]
    ks = [torch.randn(sk, 1, 64, device=device, dtype=dtype) for sk in seqlens_k]
    vs = [torch.randn(sk, 1, 512, device=device, dtype=dtype) for sk in seqlens_k]
    idxs = [rect_topk_indices(1, sq, sk, topk, causal, device, fill_frac=0.7, oob_frac=0.3,
                              seed=i)[0] for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k))]
    cu = lambda lens: torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=device)  # noqa: E731

    def pad_batch(xs, s_max):
        out = torch.zeros(len(xs), s_max, *xs[0].shape[1:], device=device, dtype=xs[0].dtype)
        for i, x in enumerate(xs):
            out[i, : x.shape[0]] = x
        return out

    if varlen_k:
        k_in, v_in = torch.cat(ks), torch.cat(vs)
        kv_kw = dict(cu_seqlens_k=cu(seqlens_k), max_seqlen_k=s_k_max)
    else:
        k_in, v_in = pad_batch(ks, s_k_max), pad_batch(vs, s_k_max)
        kv_kw = dict(seqused_k=torch.tensor(seqlens_k, dtype=torch.int32, device=device))
    if q_mode == "cu_seqlens_q":
        q_in, qv_in, idx_in = torch.cat(qs), torch.cat(qvs), torch.cat(idxs)
        q_kw = dict(cu_seqlens_q=cu(seqlens_q), max_seqlen_q=s_q_max)
    else:
        q_in, qv_in = pad_batch(qs, s_q_max), pad_batch(qvs, s_q_max)
        idx_in = pad_batch(idxs, s_q_max).contiguous()
        q_kw = dict(seqused_q=torch.tensor(seqlens_q, dtype=torch.int32, device=device),
                    max_seqlen_q=s_q_max)
    if has_qk:
        args = dict(q=q_in, k=k_in, v=v_in, qv=qv_in)
    else:
        args = dict(q=qv_in, k=v_in, v=v_in)
    call = dict(**args, **q_kw, **kv_kw, gather_kv_indices=idx_in, causal=causal, return_lse=True)
    # out as a view into a canary-filled buffer: a TMA O box straddling past the last
    # token (cu_seqlens_q) would overwrite the tail
    canary, pad = -777.0, 64 * 512 * 2
    n_out = q_in.shape[:-1].numel() * 512
    buf = torch.full((n_out + pad,), canary, device=device, dtype=dtype)
    out_view = buf[:n_out].view(*q_in.shape[:-1], 512)
    # _flash_attn_fwd takes shared_kv as (q=None, k=None, v, qv); only the autograd
    # wrappers rewrite (q, k=v, v) into that form
    out, lse, *_ = _flash_attn_fwd(
        q_in if has_qk else None, k_in if has_qk else None, v_in, qv=qv_in, out=out_view,
        **q_kw, **kv_kw, gather_kv_indices=idx_in, causal=causal, return_lse=True,
    )
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    out_2cta, lse_2cta = flash_attn_varlen_func(**call)
    if is_fake_mode():
        return
    torch.cuda.synchronize()
    assert (buf[n_out:] == canary).all(), "O written past the last token"
    starts_q = list(itertools.accumulate([0] + seqlens_q))
    for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        if sq == 0:
            continue
        if q_mode == "cu_seqlens_q":
            rows = slice(starts_q[i], starts_q[i + 1])
            o, o2, l, l2 = out[rows], out_2cta[rows], lse[rows], lse_2cta[rows]
        else:
            o, o2, l, l2 = out[i, :sq], out_2cta[i, :sq], lse[i, :sq], lse_2cta[i, :sq]
        # bf16, <= 64 heads under the flag: the kb64 mainloop
        _assert_mla_fwd_close(o, o2, l, l2, f"sequence {i}: 1CTA kb64 vs 2CTA")
        q_r = qs[i][None] if has_qk else None
        k_r = ks[i][None] if has_qk else vs[i][None]
        ref_q = q_r if has_qk else qvs[i][None]
        ref_qv = qvs[i][None] if has_qk else None
        valid = _topk_valid_rows(idxs[i][None], sq, sk, causal)[0]
        assert (o[~valid] == 0).all() and torch.isneginf(l[~valid]).all()
        if valid.any():
            out_ref, _ = attention_ref(ref_q, k_r, vs[i][None], qv=ref_qv, causal=causal,
                                       gather_kv_indices=idxs[i][None])
            out_pt, _ = attention_ref(ref_q, k_r, vs[i][None], qv=ref_qv, causal=causal,
                                      gather_kv_indices=idxs[i][None], upcast=False,
                                      reorder_ops=True)
            err = (o.float() - out_ref[0].float()).abs()[valid].max().item()
            err_pt = (out_pt[0].float() - out_ref[0].float()).abs()[valid].max().item()
            atol = 2 * (out_ref[0] + 0.3 - 0.3 - out_ref[0])[valid].abs().max().item()
            assert err <= 2 * err_pt + atol, (i, err, err_pt, atol)


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA sparse MLA forward (fp8 is 1CTA-only)")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads", [16, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_sparse_fp8(nheads, causal):
    """fp8 sparse gather (4 V stages, 16-element cp.async chunks) with descales, with and
    without a sink: exact sink identity against the sink-free fp8 call, and a loose check
    against the dequantized fp32 reference."""
    if not IS_SM100:
        pytest.skip()
    device, fp8 = "cuda", torch.float8_e4m3fn
    b, s_q, s_k, topk = 2, 64, 2048, 512
    torch.random.manual_seed(0)
    q, qv, k, v = [
        torch.randn(*shape, device=device).to(fp8)
        for shape in ((b, s_q, nheads, 64), (b, s_q, nheads, 512), (b, s_k, 1, 64), (b, s_k, 1, 512))
    ]
    q_descale = torch.rand(b, 1, device=device) + 0.5
    kv_descale = torch.rand(b, 1, device=device) + 0.5
    sink = torch.randn(nheads, device=device, dtype=torch.bfloat16) * 4
    idx = rect_topk_indices(b, s_q, s_k, topk, causal, device, fill_frac=0.8, oob_frac=0.5)

    def run(learnable_sink):
        out, lse, *_ = _flash_attn_fwd(
            q, k, v, qv=qv, causal=causal, gather_kv_indices=idx, learnable_sink=learnable_sink,
            q_descale=q_descale, k_descale=kv_descale, v_descale=kv_descale, return_lse=True,
        )
        return out, lse

    out, lse = run(None)
    out_sink, lse_sink = run(sink)
    if is_fake_mode():
        return
    valid = _topk_valid_rows(idx, s_q, s_k, causal)
    lse_expected = torch.logaddexp(lse, sink.float().view(1, 1, nheads))
    torch.testing.assert_close(lse_sink, lse_expected, atol=1e-3, rtol=1e-4)
    out_expected = out.float() * torch.exp(lse - lse_expected).unsqueeze(-1)
    out_expected[~valid] = 0
    torch.testing.assert_close(out_sink.float(), out_expected, atol=1e-2, rtol=1e-2)

    deq = lambda t, d: (t.float() * d.view(b, 1, 1, 1)).to(torch.bfloat16)  # noqa: E731
    out_ref, _ = attention_ref(deq(q, q_descale), deq(k, kv_descale), deq(v, kv_descale),
                               qv=deq(qv, q_descale), causal=causal, gather_kv_indices=idx)
    assert (out[~valid] == 0).all()
    err = (out.float() - out_ref.float())[valid].abs().max().item()
    assert err <= 0.1 * out_ref.float()[valid].abs().max().item(), err


@pytest.mark.skipif(not MLA_1CTA, reason="1CTA sparse MLA training forward")
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("has_qk", [True, False])
# fewer than 64 heads pad the forward's and the backward's 64-row tiles
@pytest.mark.parametrize("h", [64, 24, 1])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_1cta_sparse_train_recompute_p(h, has_qk, causal, varlen, has_learnable_sink,
                                                      monkeypatch):
    """Sparse training with the recompute-P backward at 1..64 heads runs its forward on
    the 1CTA kernel: exact running max (rescale_threshold 0), LSE and the O residual, no
    P / row_max. The 64-key-block mainloop agrees with the 2CTA kernel to bf16 rounding:
    out / LSE under the forward contract, out + o_lo and every gradient by relative L2."""
    if not IS_SM100:
        pytest.skip()
    import flash_attn.cute.interface as fa_interface
    device, dtype, topk = "cuda", torch.bfloat16, 256
    seqlens = [300, 1, 177] if varlen else [512]
    total = sum(seqlens)
    torch.random.manual_seed(0)
    d_q = 64 if has_qk else 512
    mk = lambda *shape: torch.randn(*shape, device=device, dtype=dtype).requires_grad_()  # noqa: E731
    if varlen:
        q, qv = mk(total, h, d_q), mk(total, h, 512)
        k, v = mk(total, 1, 64), mk(total, 1, 512)
        idx = torch.cat([rect_topk_indices(1, s, s, topk, causal, device, fill_frac=0.8, seed=i)[0]
                         for i, s in enumerate(seqlens)])
        cu = torch.tensor([0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device=device)
        extra = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens))
        fn = flash_attn_varlen_func
    else:
        q, qv = mk(1, total, h, d_q), mk(1, total, h, 512)
        k, v = mk(1, total, 1, 64), mk(1, total, 1, 512)
        idx = rect_topk_indices(1, total, total, topk, causal, device, fill_frac=0.8)
        extra = {}
        fn = flash_attn_func
    sink = (torch.randn(h, device=device, dtype=torch.bfloat16) * 4).requires_grad_() \
        if has_learnable_sink else None
    if has_qk:
        fwd_args, grad_inputs = (q, k, v), (q, k, v, qv)
        call = dict(q=q, k=k, v=v, qv=qv)
    else:
        fwd_args, grad_inputs = (None, None, v), (qv, v)
        call = dict(q=qv, k=v, v=v)  # shared_kv
    if sink is not None:
        grad_inputs = grad_inputs + (sink,)
    kw = dict(causal=causal, gather_kv_indices=idx, learnable_sink=sink, **extra)
    g = torch.randn(*qv.shape[:-1], 512, device=device, dtype=dtype)

    # Which forward kernel ran: the first compile-key element is mla_1cta. Spy on the cache
    # lookups -- kernel construction alone is not observable once the JIT cache has it. The
    # membership test runs in fake mode too (the lookup of the compiled function does not).
    real_cache = fa_interface._flash_attn_fwd.compile_cache

    class CacheSpy:
        def __init__(self):
            self.keys = []

        def __contains__(self, key):
            self.keys.append(key)
            return key in real_cache

        def __getitem__(self, key):
            return real_cache[key]

        def __setitem__(self, key, value):
            real_cache[key] = value

    results = {}
    for flag in ("1", "0"):
        spy = CacheSpy()
        monkeypatch.setattr(fa_interface._flash_attn_fwd, "compile_cache", spy)
        monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", flag)
        out, lse, p, row_max, o_lo = _flash_attn_fwd(
            *fwd_args, qv=qv if has_qk else qv, gather_bwd_recompute_p=True, return_lse=True, **kw
        )
        assert p is None and row_max is None and o_lo is not None
        out_ag, _ = fn(**call, gather_bwd_recompute_p=True, **kw)
        grads = torch.autograd.grad(out_ag, grad_inputs, g)
        results[flag] = (out, lse, o_lo, grads)
        assert spy.keys and all(key[0] == (flag == "1") for key in spy.keys), (
            f"FLASH_ATTENTION_MLA_1CTA={flag}: forward ran on the wrong kernel"
        )
    monkeypatch.setattr(fa_interface._flash_attn_fwd, "compile_cache", real_cache)
    if is_fake_mode():
        return
    rel_l2 = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()  # noqa: E731
    assert _mla_kb64_active(h, dtype)  # sparse bf16 <= 64 heads: the kb64 mainloop
    # 64-key blocks: bf16-rounding agreement with the 2CTA kernel, not bitwise
    _assert_mla_fwd_close(results["1"][0], results["0"][0], results["1"][1], results["0"][1],
                          "1CTA kb64 vs 2CTA")
    for flag in ("1", "0"):
        out_f, _, o_lo_f, _ = results[flag]
        # o_lo is the rounding residual: at most half an ulp of out
        _, e = torch.frexp(out_f.float())
        half_ulp = torch.ldexp(torch.ones_like(out_f, dtype=torch.float32), e - 9)
        assert (o_lo_f.float().abs() <= half_ulp)[out_f != 0].all(), flag
    # the near-fp32 O the backward's dpsum uses; measured <= 8e-4
    o32 = [results[f][0].float() + results[f][2].float() for f in ("1", "0")]
    assert rel_l2(*o32) < 2e-3, "out + o_lo"
    names = ("dq", "dk", "dv", "dqv") if has_qk else ("dqv", "dv")
    if sink is not None:
        names = names + ("dsink",)
    for name, a, b in zip(names, results["1"][3], results["0"][3]):
        # P is recomputed from a slightly different LSE and dpsum from a slightly different
        # O: measured rel-L2 <= 1.6e-3 on the grads, 3.3e-3 on dsink. At one head dsink is a
        # single bf16 scalar summing ~500 signed terms: 2CTA is 1.4e-2 off the fp32 reference
        # there (kb64 0, the bf16 torch reference 4.6e-2).
        dsink_tol = 5e-2 if h == 1 else 1e-2
        assert rel_l2(a, b) < (dsink_tol if name == "dsink" else 5e-3), name


def causal_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, device):
    """Top-k indices as produced by a causal sparse-attention selector: query t
    gets min(t+1, seqlen_k, topk_len) valid keys drawn from [0, t], with
    trailing -1 sentinel padding (the documented marker for invalid slots)."""
    n_keys = max(seqlen_k, topk_len)
    scores = torch.rand(batch_size, seqlen_q, n_keys, device=device)
    key_idx = torch.arange(n_keys, device=device)
    query_idx = torch.arange(seqlen_q, device=device)
    invalid = (key_idx[None, None, :] > query_idx[None, :, None]) | (key_idx >= seqlen_k)[None, None, :]
    scores.masked_fill_(invalid, float("-inf"))
    val, idx = scores.topk(topk_len, dim=-1)
    idx = idx.masked_fill(torch.isinf(val), -1)
    return idx.to(torch.int32).contiguous()


def plant_canary(shape, pad_words, device):
    """Allocate a float32 buffer of `shape` with `pad_words` extra words on
    each side holding int32 patterns 1..pad_words. Every pattern value is a
    subnormal fp32 bit pattern, so a single misdirected red.add.f32 (even of
    +0.0) flushes it to zero."""
    numel = math.prod(shape)
    parent = torch.zeros(pad_words + numel + pad_words, dtype=torch.float32, device=device)
    pattern = torch.arange(1, pad_words + 1, dtype=torch.int32, device=device)
    parent[:pad_words].view(torch.int32).copy_(pattern)
    parent[-pad_words:].view(torch.int32).copy_(pattern)
    return parent, parent[pad_words:-pad_words].view(shape)


def check_canary(name, parent, pad_words):
    expected = torch.arange(1, pad_words + 1, dtype=torch.int32)
    for side, sl in (("before", slice(None, pad_words)), ("after", slice(-pad_words, None))):
        got = parent[sl].view(torch.int32).cpu()
        n_bad = (got != expected).sum().item()
        assert n_bad == 0, (
            f"{name}: {n_bad}/{pad_words} canary words {side} the buffer were "
            f"corrupted by an out-of-bounds scatter"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
# recompute_p at 64 heads: dK_rope is scattered by the main kernel's epilogue
# (fused), at 128 heads and in load-p mode by the separate dk kernel.
@pytest.mark.parametrize("recompute_p", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize("nheads", [128, 64, 96, 24, 1])
# 130 rows: a partial last preprocess tile when padded heads use per-head packing.
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(130, 258), (512, 512), (1024, 1024)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_sentinel(seqlen_q, seqlen_k, nheads, shared_kv, causal, recompute_p, dtype):
    """Sparse-MLA backward with -1-padded gather_kv_indices, the padding any
    causal top-k selector produces for early queries.

    nheads < 128 covers in-kernel head padding (pack_gqa.qheads_first_tma_view):
    96 pads to 128 with a partial second CTA, 24 and 1 pad to the 64-head bwd tile.
    recompute_p runs the canaries against the recompute-P backward (padded counts:
    zero-filled Q / Qv rows and +inf lse_log2 pads make P = 0 in the padded rows), whose
    64-row main kernel (1..64 heads) scatters dK_rope itself.

    Regression test for unguarded sentinel scatters: the dV/dK backward
    epilogues used to atomically accumulate at row -1 — out of bounds of the
    (batch-sliced) buffer — corrupting adjacent memory even though the addend
    is exactly 0.0, because red.add.f32 flushes subnormal destinations to zero
    and canonicalizes NaN payloads. Checks that
      1. grads match the reference through the public autograd path, and
      2. int32 canaries planted directly before/after preallocated dk/dv
         buffers are untouched by the scatter epilogues.
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 2
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256

    q_ref = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdim, device=device, dtype=dtype).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdimv, device=device, dtype=dtype).requires_grad_()
    qv_ref = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype).requires_grad_()
    gather_kv_indices = causal_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, device)

    q, k, v, qv = [x.detach().clone().requires_grad_() for x in (q_ref, k_ref, v_ref, qv_ref)]
    if shared_kv:
        q, k, qv = qv, v, None
        q_ref, k_ref, qv_ref = qv_ref, v_ref, None

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, causal=causal, qv=qv_ref, gather_kv_indices=gather_kv_indices
    )
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, causal=causal, qv=qv_ref, gather_kv_indices=gather_kv_indices,
        upcast=False, reorder_ops=True,
    )

    out, lse = flash_attn_func(
        q, k, v, qv=qv, gather_kv_indices=gather_kv_indices, causal=causal, pack_gqa=True,
        gather_bwd_recompute_p=recompute_p,
    )

    g = torch.randn_like(out)
    if shared_kv:
        dq, dk = torch.autograd.grad(out, (q, k), g)
        dv = dqv = None
    else:
        dq, dk, dv, dqv = torch.autograd.grad(out, (q, k, v, qv), g)

    # Rerun the backward with preallocated dk/dv surrounded by canaries. The
    # scatter epilogues write rows of hdim/hdimv fp32 words, so an unguarded
    # -1 sentinel lands exactly in the pad before the buffer.
    dv_parent, dv_buf = plant_canary((batch_size, seqlen_k, nheads_kv, hdimv), hdimv, device)
    if shared_kv:
        dk_parent = dk_buf = None
    else:
        dk_parent, dk_buf = plant_canary((batch_size, seqlen_k, nheads_kv, hdim), hdim, device)
    with torch.no_grad():
        fq, fk, fqv = (None, None, q) if shared_kv else (q, k, qv)
        out2, lse2, p2, row_max2, o_lo2 = _flash_attn_fwd(
            fq, fk, v, qv=fqv, causal=causal, gather_kv_indices=gather_kv_indices, pack_gqa=True,
            gather_bwd_recompute_p=recompute_p,
        )
        dq2, dk2, dv2, dqv2, _ = _flash_attn_bwd_sparse_mla(
            fq, fk, v, fqv, out2, g, lse2, p2, row_max2, gather_kv_indices,
            causal=causal, dk=dk_buf, dv=dv_buf, recompute_p=recompute_p, o_lo=o_lo2,
        )

    if is_fake_mode():
        # no more flash_attn cutedsl calls; skip data-dependent checks
        return

    assert (gather_kv_indices == -1).any(), "test must exercise sentinel slots"

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol
    assert not torch.isnan(lse).any(), "LSE contains NaN"

    if shared_kv:
        dq_ref, dk_ref = torch.autograd.grad(out_ref, (q_ref, k_ref), g)
        dq_pt, dk_pt = torch.autograd.grad(out_pt, (q_ref, k_ref), g)
        dv_ref = dqv_ref = dv_pt = dqv_pt = None
    else:
        dq_ref, dk_ref, dv_ref, dqv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref, qv_ref), g)
        dq_pt, dk_pt, dv_pt, dqv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref, qv_ref), g)

    print_diff_stats("dQ", dq, dq_ref, dq_pt)
    print_diff_stats("dK", dk, dk_ref, dk_pt)
    print_diff_stats("dV", dv, dv_ref, dv_pt)
    print_diff_stats("dQv", dqv, dqv_ref, dqv_pt)

    check_tensor_vs_ref("dQ", dq, dq_ref, dq_pt)
    check_tensor_vs_ref("dK", dk, dk_ref, dk_pt)
    check_tensor_vs_ref("dV", dv, dv_ref, dv_pt)
    check_tensor_vs_ref("dQv", dqv, dqv_ref, dqv_pt)

    check_canary("dV", dv_parent, hdimv)
    if not shared_kv:
        check_canary("dK", dk_parent, hdim)
    # preallocated buffers must produce the same grads as internal allocation
    if shared_kv:
        check_tensor_vs_ref("dV(prealloc)", dv2, dk_ref, dk_pt)
        check_tensor_vs_ref("dQv(prealloc)", dqv2, dq_ref, dq_pt)
    else:
        check_tensor_vs_ref("dQ(prealloc)", dq2, dq_ref, dq_pt)
        check_tensor_vs_ref("dK(prealloc)", dk2, dk_ref, dk_pt)
        check_tensor_vs_ref("dV(prealloc)", dv2, dv_ref, dv_pt)
        check_tensor_vs_ref("dQv(prealloc)", dqv2, dqv_ref, dqv_pt)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
# recompute_p at 1..64 heads: dK_rope is scattered by the main kernel's epilogue (fused).
@pytest.mark.parametrize("recompute_p", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shared_kv", [False, True])
# 24 heads: padded per-head preprocess tiles must not spill into the next packed sequence
# (recompute_p: nor the padded lse_log2 columns, which must keep their +inf).
# 64 heads: the native 64-row (tile_m == 64) backward specialization.
@pytest.mark.parametrize("nheads", [128, 64, 24])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_sentinel_varlen(nheads, shared_kv, causal, recompute_p, dtype):
    """Varlen counterpart of test_flash_attn_mla_sparse_bwd_sentinel.

    The varlen kernels are separate compile-time specializations, and the dK
    epilogue applies the sentinel guard to the doc-relative index before
    adding seqlen_k_offset — a -1 that slipped past the guard would land in
    the previous doc's last row (in bounds, so only doc 0's row -1 is
    canary-visible, same as batch 0 in the non-varlen test). Includes a doc
    shorter than topk_len whose index rows are almost entirely sentinels.
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256
    seqlens = [512, 4, 1024] if nheads in (128, 64) else [130, 4, 258]
    total = sum(seqlens)
    cu_bounds = [0] + list(itertools.accumulate(seqlens))
    cu_seqlens = torch.tensor(cu_bounds, dtype=torch.int32, device=device)
    max_seqlen = max(seqlens)

    q_ref = torch.randn(total, nheads, hdim, device=device, dtype=dtype).requires_grad_()
    k_ref = torch.randn(total, nheads_kv, hdim, device=device, dtype=dtype).requires_grad_()
    v_ref = torch.randn(total, nheads_kv, hdimv, device=device, dtype=dtype).requires_grad_()
    qv_ref = torch.randn(total, nheads, hdimv, device=device, dtype=dtype).requires_grad_()
    # doc-relative key indices with -1 tail padding, packed along the token dim
    gather_kv_indices = torch.cat(
        [causal_topk_indices(1, L, L, topk_len, device)[0] for L in seqlens], dim=0
    ).contiguous()

    q, k, v, qv = [x.detach().clone().requires_grad_() for x in (q_ref, k_ref, v_ref, qv_ref)]
    if shared_kv:
        q, k, qv = qv, v, None
        q_ref, k_ref, qv_ref = qv_ref, v_ref, None

    # reference per doc (each doc is an independent attention problem)
    outs_ref, outs_pt = [], []
    for i in range(len(seqlens)):
        s, e = cu_bounds[i], cu_bounds[i + 1]
        doc = dict(causal=causal, gather_kv_indices=gather_kv_indices[s:e].unsqueeze(0))
        o_ref, _ = attention_ref(
            q_ref[s:e].unsqueeze(0), k_ref[s:e].unsqueeze(0), v_ref[s:e].unsqueeze(0),
            qv=qv_ref[s:e].unsqueeze(0) if qv_ref is not None else None, **doc,
        )
        o_pt, _ = attention_ref(
            q_ref[s:e].unsqueeze(0), k_ref[s:e].unsqueeze(0), v_ref[s:e].unsqueeze(0),
            qv=qv_ref[s:e].unsqueeze(0) if qv_ref is not None else None,
            upcast=False, reorder_ops=True, **doc,
        )
        outs_ref.append(o_ref[0])
        outs_pt.append(o_pt[0])
    out_ref = torch.cat(outs_ref, dim=0)
    out_pt = torch.cat(outs_pt, dim=0)

    out, lse = flash_attn_varlen_func(
        q, k, v, qv=qv, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        gather_kv_indices=gather_kv_indices, causal=causal, pack_gqa=True,
        gather_bwd_recompute_p=recompute_p,
    )

    g = torch.randn_like(out)
    if shared_kv:
        dq, dk = torch.autograd.grad(out, (q, k), g)
        dv = dqv = None
    else:
        dq, dk, dv, dqv = torch.autograd.grad(out, (q, k, v, qv), g)

    # canary rerun with preallocated dk/dv (see non-varlen test)
    dv_parent, dv_buf = plant_canary((total, nheads_kv, hdimv), hdimv, device)
    if shared_kv:
        dk_parent = dk_buf = None
    else:
        dk_parent, dk_buf = plant_canary((total, nheads_kv, hdim), hdim, device)
    with torch.no_grad():
        fq, fk, fqv = (None, None, q) if shared_kv else (q, k, qv)
        out2, lse2, p2, row_max2, o_lo2 = _flash_attn_fwd(
            fq, fk, v, qv=fqv, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=causal, gather_kv_indices=gather_kv_indices, pack_gqa=True,
            gather_bwd_recompute_p=recompute_p,
        )
        dq2, dk2, dv2, dqv2, _ = _flash_attn_bwd_sparse_mla(
            fq, fk, v, fqv, out2, g, lse2, p2, row_max2, gather_kv_indices,
            causal=causal,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dk=dk_buf, dv=dv_buf, recompute_p=recompute_p, o_lo=o_lo2,
        )

    if is_fake_mode():
        # no more flash_attn cutedsl calls; skip data-dependent checks
        return

    assert (gather_kv_indices == -1).any(), "test must exercise sentinel slots"

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol
    assert not torch.isnan(lse).any(), "LSE contains NaN"

    if shared_kv:
        dq_ref, dk_ref = torch.autograd.grad(out_ref, (q_ref, k_ref), g)
        dq_pt, dk_pt = torch.autograd.grad(out_pt, (q_ref, k_ref), g)
        dv_ref = dqv_ref = dv_pt = dqv_pt = None
    else:
        dq_ref, dk_ref, dv_ref, dqv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref, qv_ref), g)
        dq_pt, dk_pt, dv_pt, dqv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref, qv_ref), g)

    print_diff_stats("dQ", dq, dq_ref, dq_pt)
    print_diff_stats("dK", dk, dk_ref, dk_pt)
    print_diff_stats("dV", dv, dv_ref, dv_pt)
    print_diff_stats("dQv", dqv, dqv_ref, dqv_pt)

    check_tensor_vs_ref("dQ", dq, dq_ref, dq_pt)
    check_tensor_vs_ref("dK", dk, dk_ref, dk_pt)
    check_tensor_vs_ref("dV", dv, dv_ref, dv_pt)
    check_tensor_vs_ref("dQv", dqv, dqv_ref, dqv_pt)

    check_canary("dV", dv_parent, hdimv)
    if not shared_kv:
        check_canary("dK", dk_parent, hdim)
    if shared_kv:
        check_tensor_vs_ref("dV(prealloc)", dv2, dk_ref, dk_pt)
        check_tensor_vs_ref("dQv(prealloc)", dqv2, dq_ref, dq_pt)
    else:
        check_tensor_vs_ref("dQ(prealloc)", dq2, dq_ref, dq_pt)
        check_tensor_vs_ref("dK(prealloc)", dk2, dk_ref, dk_pt)
        check_tensor_vs_ref("dV(prealloc)", dv2, dv_ref, dv_pt)
        check_tensor_vs_ref("dQv(prealloc)", dqv2, dqv_ref, dqv_pt)


@pytest.mark.parametrize("nheads", [24, 1])
@pytest.mark.parametrize("has_qk", [True, False])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_recompute_p_padded(nheads, has_qk, monkeypatch):
    """Recompute-P with a padded head tile (1..63 heads -> 64 rows). The padded rows load
    zero Q / Qv / dO and +inf lse_log2, so their P and dS are exactly 0:
      1. recompute-P vs load-P grads on the same forward (bf16 contract), all finite;
      2. zero contribution: the same call at 64 heads with the extra heads' dO zeroed gives
         bitwise-equal dq / dqv on the real heads and dk / dv up to the atomic scatter order;
      3. the 64-row dQ / dQv kernel (1..64 heads) is bitwise equal to the generic 128-row one.
    Runs the 2CTA forward (the flag-off route); the kb64 route is covered by
    test_flash_attn_mla_1cta_sparse_train_recompute_p."""
    if not IS_SM100:
        pytest.skip()
    import flash_attn.cute.interface as fa_interface
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    device, dtype, topk = "cuda", torch.bfloat16, 256
    seqlens = [300, 1, 177]
    total, h_full = sum(seqlens), 64
    cu = torch.tensor([0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device=device)
    idx = torch.cat([rect_topk_indices(1, s, s, topk, True, device, fill_frac=0.8, seed=i)[0]
                     for i, s in enumerate(seqlens)])
    torch.random.manual_seed(0)
    d_q = 64 if has_qk else 512
    q_full = torch.randn(total, h_full, d_q, device=device, dtype=dtype)
    qv_full = torch.randn(total, h_full, 512, device=device, dtype=dtype)
    k, v = (torch.randn(total, 1, d, device=device, dtype=dtype) for d in (64, 512))
    g_full = torch.randn(total, h_full, 512, device=device, dtype=dtype)
    extra = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
                 causal=True, gather_kv_indices=idx)

    def grads(h, recompute, g):
        ins = [q_full[:, :h].contiguous(), k, v, qv_full[:, :h].contiguous()]
        if not has_qk:
            ins = [ins[3], v]  # shared_kv: (q=qv, k=v, v=v)
        ins = [x.detach().clone().requires_grad_() for x in ins]
        call = (ins[0], ins[1], ins[2]) if has_qk else (ins[0], ins[1], ins[1])
        out, _ = flash_attn_varlen_func(*call, qv=ins[3] if has_qk else None,
                                        gather_bwd_recompute_p=recompute, **extra)
        names = ("dq", "dk", "dv", "dqv") if has_qk else ("dqv", "dv")
        return dict(zip(names, torch.autograd.grad(out, ins, g)))

    rel_l2 = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()  # noqa: E731
    g = g_full[:, :nheads].contiguous()
    g_pad = g_full.clone()
    g_pad[:, nheads:] = 0
    # every variant first (fake mode compiles them all, then returns)
    rp, lp = grads(nheads, True, g), grads(nheads, False, g)
    full = grads(h_full, True, g_pad)
    # the generic 128-row dQ / dQv kernel for the same call (its own compile key)
    with monkeypatch.context() as m:
        m.setattr(fa_interface, "dQdQvGemmKernelH64", fa_interface.dQdQvGemmKernel)
        generic = grads(nheads, True, g)
    if is_fake_mode():
        return
    for name in rp:
        assert torch.isfinite(rp[name]).all(), name
        assert rel_l2(rp[name], lp[name]) < 5e-3, name  # one fewer P rounding; measured ~3e-3
    for name in rp:
        if name in ("dq", "dqv"):
            assert torch.equal(rp[name], full[name][:, :nheads]), name
        else:
            assert rel_l2(rp[name], full[name]) < 5e-4, name  # atomic scatter-add order

    for name in ("dq", "dqv"):
        if name in rp:
            assert torch.equal(rp[name], generic[name]), name


def random_cutoff_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, device):
    """Top-k indices drawn from [0, cutoff_t) with a per-row random cutoff and
    trailing -1 padding. Deliberately NOT causal: many rows contain indices
    beyond their own position, so under causal=True the kernel's causal key
    limit (not just the -1 sentinels) must mask entries. Rows with a small
    cutoff get -1 tails."""
    n_keys = max(seqlen_k, topk_len)
    scores = torch.rand(batch_size, seqlen_q, n_keys, device=device)
    # key 0 is causally valid for every row; always select it so no row ends up
    # fully masked under causal=True (the reference NaNs on all--inf rows)
    scores[..., 0] = 2.0
    key_idx = torch.arange(n_keys, device=device)
    cutoff = torch.randint(
        topk_len // 2, seqlen_k + 1, (batch_size, seqlen_q, 1), device=device
    )
    invalid = (key_idx[None, None, :] >= cutoff) | (key_idx >= seqlen_k)[None, None, :]
    scores.masked_fill_(invalid, float("-inf"))
    val, idx = scores.topk(topk_len, dim=-1)
    idx = idx.masked_fill(torch.isinf(val), -1)
    return idx.to(torch.int32).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(512, 512)])
# 64 heads: the 64-row backward tile with no padded head rows.
@pytest.mark.parametrize("nheads", [128, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_recompute_p(seqlen_q, seqlen_k, nheads, shared_kv, causal, dtype):
    """Sparse-MLA backward with gather_bwd_recompute_p and gather_bwd_token_chunk.

    With recompute_p the forward saves only out+lse (no p/row_max) and the
    backward main kernel reconstructs P = exp2(softmax_scale*log2(e)*S - lse)
    in-kernel; token_chunk additionally bounds the dS transient to a token
    chunk. Checks:
      1. the recompute forward is bitwise-identical to the default forward
         (same kernel math, only the p/row_max stores are skipped) -- to bf16
         rounding when FLASH_ATTENTION_MLA_1CTA=1 routes 64-head recompute-P to
         the 1CTA 64-key-block forward;
      2. grads match the fp32 reference within the standard tolerance;
      3. the token-chunked backward matches the unchunked one bitwise on
         dq/dqv (dk/dv only up to fp32-atomic accumulation order). The causal
         cases use non-causal indices, so this exercises the chunked path's
         per-chunk key-extent shrink: the recomputed mask must reproduce the
         forward's causal limit exactly, or recomputed P is nonzero where the
         forward had -inf;
      4. backward can run twice (retain_graph): nothing saved is consumed.
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 1  # token_chunk requires varlen or batch 1
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256
    token_chunk = 200  # 512 = 200 + 200 + 112: exercises tail chunks

    q_ref = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdim, device=device, dtype=dtype).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdimv, device=device, dtype=dtype).requires_grad_()
    qv_ref = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype).requires_grad_()
    if causal:
        gather_kv_indices = random_cutoff_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, device)
    else:
        gather_kv_indices = causal_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, device)

    q, k, v, qv = [x.detach().clone().requires_grad_() for x in (q_ref, k_ref, v_ref, qv_ref)]
    if shared_kv:
        q, k, qv = qv, v, None
        q_ref, k_ref, qv_ref = qv_ref, v_ref, None
    grad_inputs = (q, k) if shared_kv else (q, k, v, qv)

    out_default, lse_default = flash_attn_func(
        q, k, v, qv=qv, gather_kv_indices=gather_kv_indices, causal=causal, pack_gqa=True
    )
    out, lse = flash_attn_func(
        q, k, v, qv=qv, gather_kv_indices=gather_kv_indices, causal=causal, pack_gqa=True,
        gather_bwd_recompute_p=True,
    )
    g = torch.randn_like(out)
    grads = torch.autograd.grad(out, grad_inputs, g, retain_graph=True)
    grads_again = torch.autograd.grad(out, grad_inputs, g)

    out_ck, _ = flash_attn_func(
        q, k, v, qv=qv, gather_kv_indices=gather_kv_indices, causal=causal, pack_gqa=True,
        gather_bwd_recompute_p=True, gather_bwd_token_chunk=token_chunk,
    )
    grads_ck = torch.autograd.grad(out_ck, grad_inputs, g)

    # Default (load-p) path token chunking: same chunked-vs-unchunked contract
    # without recompute_p. This covers the non-varlen chunked load-p
    # configuration (per-chunk p/scale_p slicing interacting with the causal
    # k_end view shrink), which no other test exercises.
    out_default_ck, _ = flash_attn_func(
        q, k, v, qv=qv, gather_kv_indices=gather_kv_indices, causal=causal, pack_gqa=True,
        gather_bwd_token_chunk=token_chunk,
    )
    grads_default = torch.autograd.grad(out_default, grad_inputs, g, retain_graph=True)
    grads_default_ck = torch.autograd.grad(out_default_ck, grad_inputs, g)

    if is_fake_mode():
        # no more flash_attn cutedsl calls; skip data-dependent checks
        return

    assert (gather_kv_indices == -1).any(), "test must exercise sentinel slots"
    if _mla_kb64_active(nheads, dtype):
        # FLASH_ATTENTION_MLA_1CTA=1: recompute-P runs the 1CTA 64-key-block forward, the
        # default (load-P) path the 2CTA one
        _assert_mla_fwd_close(out, out_default, lse, lse_default, "recompute vs default fwd")
    else:
        assert torch.equal(out, out_default), "recompute fwd out must be bitwise-identical"
        assert torch.equal(lse, lse_default), "recompute fwd lse must be bitwise-identical"

    # dq/dqv (pure GEMM consumers of identical dS tiles) are bitwise-stable
    # across retain_graph reruns and across chunking; dk/dv accumulate with
    # fp32 atomics whose order changes with launch partitioning.
    atomic_grads = (1,) if shared_kv else (1, 2)
    for i, (a, b) in enumerate(zip(grads_again, grads)):
        if i not in atomic_grads:
            assert torch.equal(a, b), f"second backward grad {i} not bitwise"
    for i, (a, b) in enumerate(zip(grads_ck, grads)):
        if i in atomic_grads:
            rel = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
            assert rel < 5e-4, f"chunked grad {i} rel_l2 {rel} beyond atomic noise"
        else:
            assert torch.equal(a, b), f"chunked grad {i} not bitwise vs unchunked"
    for i, (a, b) in enumerate(zip(grads_default_ck, grads_default)):
        if i in atomic_grads:
            rel = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
            assert rel < 5e-4, f"default-path chunked grad {i} rel_l2 {rel} beyond atomic noise"
        else:
            assert torch.equal(a, b), f"default-path chunked grad {i} not bitwise vs unchunked"

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, causal=causal, qv=qv_ref, gather_kv_indices=gather_kv_indices
    )
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, causal=causal, qv=qv_ref, gather_kv_indices=gather_kv_indices,
        upcast=False, reorder_ops=True,
    )
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol
    assert not torch.isnan(lse).any(), "LSE contains NaN"

    ref_inputs = (q_ref, k_ref) if shared_kv else (q_ref, k_ref, v_ref, qv_ref)
    grads_ref = torch.autograd.grad(out_ref, ref_inputs, g)
    grads_pt = torch.autograd.grad(out_pt, ref_inputs, g)
    names = ("dQv", "dV") if shared_kv else ("dQ", "dK", "dV", "dQv")
    for variant, gs in (("", grads), ("(chunked)", grads_ck)):
        for name, a, r, p_ in zip(names, gs, grads_ref, grads_pt):
            print_diff_stats(name + variant, a, r, p_)
            check_tensor_vs_ref(name + variant, a, r, p_)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("recompute_p", [False, True])
@pytest.mark.parametrize("token_chunk", [None, 200])
# both backward head tiles: 128 (2 x 64-row halves) and the native 64-row specialization
@pytest.mark.parametrize("nheads", [128, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_fully_masked_rows(nheads, token_chunk, recompute_p, dtype):
    """Rows whose every top-k slot is the -1 sentinel (fully masked).

    The forward must emit out = 0 / lse = -inf for those rows, and the
    backward must produce exact zeros for them with no NaN anywhere. In
    recompute mode this exercises the delicate chain the feature added for
    this case: lse = -inf -> lse_log2 sanitized to 0 in the preprocess ->
    bitmask forces every exponent to -inf (P = 0) -> non-finite-dP hardening
    keeps dS = 0. In load-p mode it relies on the fwd-saved p rows being 0
    and the sV smem zero-fill (stale-smem dP garbage would otherwise make
    dS = 0 * NaN). dq/dqv/out rows of OTHER queries must be bitwise-identical
    to a run where the masked rows are given valid indices (they are per-row
    functions of dS and the gather indices). The masked row range straddles a
    token_chunk boundary in the chunked variant."""
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size, seqlen_q, seqlen_k = 1, 512, 512
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256
    masked_rows = slice(150, 260)  # straddles the chunk boundary at 200

    q = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, hdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, hdimv, device=device, dtype=dtype, requires_grad=True)
    qv = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype, requires_grad=True)
    idx_valid = random_cutoff_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, device)
    idx_masked = idx_valid.clone()
    idx_masked[:, masked_rows] = -1
    g = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype)

    def run(idx):
        out, lse = flash_attn_func(
            q, k, v, qv=qv, gather_kv_indices=idx, causal=True, pack_gqa=True,
            gather_bwd_recompute_p=recompute_p, gather_bwd_token_chunk=token_chunk,
        )
        grads = torch.autograd.grad(out, (q, k, v, qv), g)
        return out, lse, grads

    out_m, lse_m, grads_m = run(idx_masked)
    out_v, _, grads_v = run(idx_valid)

    if is_fake_mode():
        return

    # lse layout with qv is (batch, seqlen_q, nheads)
    assert (out_m[:, masked_rows] == 0).all(), "fully-masked rows must produce out = 0"
    # exactly -inf: the preprocess sanitizes only lse == -inf (a +inf
    # regression would bypass it), see flash_bwd_preprocess.py
    assert (lse_m[:, masked_rows] == float("-inf")).all(), "fully-masked rows must produce lse = -inf"
    for name, t in zip(("dQ", "dK", "dV", "dQv"), grads_m):
        assert not t.isnan().any(), f"{name} has NaN with fully-masked rows"
    dq_m, _, _, dqv_m = grads_m
    dq_v, _, _, dqv_v = grads_v
    assert (dq_m[:, masked_rows] == 0).all(), "dq rows of fully-masked queries must be 0"
    assert (dqv_m[:, masked_rows] == 0).all(), "dqv rows of fully-masked queries must be 0"
    keep = torch.ones(seqlen_q, dtype=torch.bool, device=device)
    keep[masked_rows] = False
    assert torch.equal(out_m[:, keep], out_v[:, keep]), "unmasked out rows must be unaffected"
    assert torch.equal(dq_m[:, keep], dq_v[:, keep]), "unmasked dq rows must be unaffected"
    assert torch.equal(dqv_m[:, keep], dqv_v[:, keep]), "unmasked dqv rows must be unaffected"


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("recompute_p", [False, True])
@pytest.mark.parametrize("nheads", [128, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_token_chunk_rect(nheads, recompute_p, dtype):
    """Rectangular causal chunked backward with seqlen_q > seqlen_k.

    With bottom-right-aligned causal masking, query rows before
    seqlen_q - seqlen_k are fully masked in the forward (lse = -inf), so the
    leading token chunk has k_end = seqlen_k - seqlen_q + tok1 <= 0 and takes
    the skip_main early-continue in the chunk loop (zero dq/dqv directly,
    skip all three kernels); the middle chunk takes a partial K-extent shrink
    and the last chunk runs at the full extent. Chunked dq/dqv must stay
    bitwise-equal to the unchunked backward, fully-masked rows must get
    exactly zero grads, and nothing may NaN. No other test reaches skip_main
    (they all use seqlen_q == seqlen_k)."""
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size, seqlen_q, seqlen_k = 1, 512, 256
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256
    token_chunk = 200  # chunk 0: k_end = -56 (skip_main); chunk 1: 144; chunk 2: 256
    n_masked = seqlen_q - seqlen_k  # rows [0, 256) have an empty causal window

    q = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, hdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, hdimv, device=device, dtype=dtype, requires_grad=True)
    qv = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype, requires_grad=True)
    gather_kv_indices = random_cutoff_topk_indices(batch_size, seqlen_q, seqlen_k, topk_len, device)
    g = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype)

    def run(**kw):
        out, lse = flash_attn_func(
            q, k, v, qv=qv, gather_kv_indices=gather_kv_indices, causal=True, pack_gqa=True,
            gather_bwd_recompute_p=recompute_p, **kw,
        )
        return out, lse, torch.autograd.grad(out, (q, k, v, qv), g)

    out, lse, grads = run()
    _, _, grads_ck = run(gather_bwd_token_chunk=token_chunk)

    if is_fake_mode():
        return

    assert (out[:, :n_masked] == 0).all(), "causally-empty rows must produce out = 0"
    assert (lse[:, :n_masked] == float("-inf")).all(), "causally-empty rows must produce lse = -inf"
    for i, (name, a, b) in enumerate(zip(("dQ", "dK", "dV", "dQv"), grads_ck, grads)):
        assert not a.isnan().any(), f"{name} chunked has NaN"
        assert not b.isnan().any(), f"{name} unchunked has NaN"
        if i in (1, 2):  # dk/dv: fp32-atomic accumulation order differs across launches
            rel = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
            assert rel < 5e-4, f"{name} chunked rel_l2 {rel} beyond atomic noise"
        else:
            assert torch.equal(a, b), f"{name} chunked not bitwise vs unchunked"
    dq_ck, _, _, dqv_ck = grads_ck
    assert (dq_ck[:, :n_masked] == 0).all(), "skip_main chunk must zero its dq rows"
    assert (dqv_ck[:, :n_masked] == 0).all(), "skip_main chunk must zero its dqv rows"


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("recompute_p", [False, True])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize("nheads", [128, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_token_chunk_varlen(nheads, shared_kv, recompute_p, dtype):
    """Varlen token-chunked sparse-MLA backward, causal, with non-causal
    indices and both docs split across chunk boundaries.

    This is the coverage for the per-doc cu_seqlens_k end-offset clamp: with
    docs (600, 424) and token_chunk=256, doc 0 spans chunks 0-2 and doc 1
    spans chunks 2-4, so both the doc-continuing-past-tok1 shrink and the
    doc-starting-mid-chunk offset handling are exercised. Indices are drawn
    from the full key range (non-causal), so an incorrectly relaxed causal
    limit in any chunk would change the recomputed mask (recompute_p=True) or
    the gathered dP inputs, and the bitwise dq/dqv check against the
    unchunked backward would fail. recompute_p=False covers chunking of the
    default load-p path (p sliced per chunk). shared_kv=True is the no-rope
    kernel specialization, whose recompute_p smem layout keeps dS in its own
    buffer (the rope specialization merges dS into sP), so both recompute
    layouts get varlen chunked coverage.
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256
    doc_lens = (600, 424)
    total = sum(doc_lens)
    cu = torch.tensor([0, doc_lens[0], total], device=device, dtype=torch.int32)

    q = torch.randn(total, nheads, hdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total, nheads_kv, hdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total, nheads_kv, hdimv, device=device, dtype=dtype, requires_grad=True)
    qv = torch.randn(total, nheads, hdimv, device=device, dtype=dtype, requires_grad=True)
    if shared_kv:
        q, k, qv = qv, v, None
    grad_inputs = (q, k) if shared_kv else (q, k, v, qv)
    names = ("dQv", "dV") if shared_kv else ("dQ", "dK", "dV", "dQv")
    atomic_grads = (1,) if shared_kv else (1, 2)
    gather_kv_indices = torch.cat([
        random_cutoff_topk_indices(1, n, n, topk_len, device).squeeze(0) for n in doc_lens
    ])
    g = torch.randn(total, nheads, hdimv, device=device, dtype=dtype)

    def run(**kw):
        out, _ = flash_attn_varlen_func(
            q, k, v, qv=qv, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max(doc_lens), max_seqlen_k=max(doc_lens),
            gather_kv_indices=gather_kv_indices, causal=True, pack_gqa=True,
            gather_bwd_recompute_p=recompute_p, **kw,
        )
        return torch.autograd.grad(out, grad_inputs, g)

    grads = run()
    grads_ck = run(gather_bwd_token_chunk=256)

    if is_fake_mode():
        return

    assert (gather_kv_indices == -1).any(), "test must exercise sentinel slots"
    for i, (name, a, b) in enumerate(zip(names, grads_ck, grads)):
        assert not a.isnan().any(), f"{name} chunked has NaN"
        if i in atomic_grads:  # dk/dv: fp32-atomic accumulation order differs across launches
            rel = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
            assert rel < 5e-4, f"{name} chunked rel_l2 {rel} beyond atomic noise"
        else:
            assert torch.equal(a, b), f"{name} chunked not bitwise vs unchunked"

    # Reference check (per doc): the chunked-vs-unchunked assertions above are
    # purely self-consistent, so a varlen-specific bug that affects both runs
    # identically (e.g. a wrong per-doc lse_log2/dpsum offset in the packed
    # (total_q, h) layout) would pass them. Check the grads against the fp32
    # reference too — for both recompute_p settings.
    ref_inputs = tuple(x.detach().clone().requires_grad_() for x in grad_inputs)
    if shared_kv:
        q_ref, k_ref = ref_inputs
        v_ref, qv_ref = k_ref, None
    else:
        q_ref, k_ref, v_ref, qv_ref = ref_inputs
    outs_ref, outs_pt = [], []
    for b in range(len(doc_lens)):
        s = slice(int(cu[b].item()), int(cu[b + 1].item()))
        doc_args = dict(causal=True, qv=qv_ref[s].unsqueeze(0) if qv_ref is not None else None,
                        gather_kv_indices=gather_kv_indices[s].unsqueeze(0))
        o_ref, _ = attention_ref(
            q_ref[s].unsqueeze(0), k_ref[s].unsqueeze(0), v_ref[s].unsqueeze(0), **doc_args
        )
        o_pt, _ = attention_ref(
            q_ref[s].unsqueeze(0), k_ref[s].unsqueeze(0), v_ref[s].unsqueeze(0), **doc_args,
            upcast=False, reorder_ops=True,
        )
        outs_ref.append(o_ref.squeeze(0))
        outs_pt.append(o_pt.squeeze(0))
    out_ref = torch.cat(outs_ref)
    out_pt = torch.cat(outs_pt)
    grads_ref = torch.autograd.grad(out_ref, ref_inputs, g)
    grads_pt = torch.autograd.grad(out_pt, ref_inputs, g)
    for name, a, r, p_ in zip(names, grads, grads_ref, grads_pt):
        print_diff_stats(name, a, r, p_)
        check_tensor_vs_ref(name, a, r, p_)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("recompute_p", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_preprocess_tile_tail(varlen, recompute_p, dtype):
    """Sparse-MLA backward at 64 Q heads with an odd token count per sequence.

    The backward preprocess tiles the packed (token, head) rows 128 at a time, and the
    sparse-MLA path runs it without padded per-sequence offsets: sequences sit back to
    back with no slack between them. With 64 heads a tile spans two tokens, so a sequence
    with an odd token count ends in a half-filled tile whose tail rows ARE the next
    sequence's first token (or lie past the end of the buffer, for the last sequence).
    Every per-row store of that tile has to stop at the sequence's real row count; a
    tile-rounded store overwrites the next sequence's first token with the padding value.
    For lse_log2 that value is +inf: the recomputed P of that token is exp2(S - inf) = 0,
    so its dq/dqv vanish and its dk/dv contributions are lost. Whether the tail store or
    the next sequence's own store lands last is a race between CTAs, so a single boundary
    fails intermittently; this test packs many odd-length sequences (varlen) or batches
    them (non-varlen, where the batch elements are likewise contiguous) so a hit is
    near-certain. 128 heads fill every tile exactly and cannot hit this.

    Keys extend 64 tokens past the queries (bottom-right aligned causal), so the first
    token of every sequence has a non-trivial softmax and a non-zero gradient to lose.
    Grads are checked per sequence against the fp32 reference; dq/dqv additionally per
    token, so a failure names the corrupted tokens. recompute_p=False covers the load-P
    stores (dpsum, scale_p) of the same tiles.
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    nheads, nheads_kv, hdim, hdimv = 64, 1, 64, 512
    topk_len = 128
    kv_extra = 64  # keys before the first query token
    if varlen:
        # 11 sequence boundaries with a half-filled preprocess tile on the left, plus one
        # half-filled tile at the end of the buffer.
        q_lens = (129, 1, 65, 17, 3, 131, 33, 5, 99, 61, 127, 19)
    else:
        q_lens = (129,) * 4
    k_lens = tuple(n + kv_extra for n in q_lens)
    total_q, total_k = sum(q_lens), sum(k_lens)
    cu_q = [0] + list(itertools.accumulate(q_lens))
    cu_k = [0] + list(itertools.accumulate(k_lens))

    q = torch.randn(total_q, nheads, hdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, nheads_kv, hdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, nheads_kv, hdimv, device=device, dtype=dtype, requires_grad=True)
    qv = torch.randn(total_q, nheads, hdimv, device=device, dtype=dtype, requires_grad=True)
    g = torch.randn(total_q, nheads, hdimv, device=device, dtype=dtype)
    # Indices from the full key range: the causal limit masks some, -1 pads the rest.
    gather_kv_indices = torch.cat([
        random_cutoff_topk_indices(1, nq, nk, topk_len, device).squeeze(0)
        for nq, nk in zip(q_lens, k_lens)
    ])

    if varlen:
        cu_seqlens_q = torch.tensor(cu_q, device=device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor(cu_k, device=device, dtype=torch.int32)
        out, _ = flash_attn_varlen_func(
            q, k, v, qv=qv, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens),
            gather_kv_indices=gather_kv_indices, causal=True, pack_gqa=True,
            gather_bwd_recompute_p=recompute_p,
        )
        g_call = g
    else:
        # Same packed tensors viewed as (batch, seqlen, ...): the grads come back packed.
        batch = len(q_lens)
        unpack = lambda t, n: t.view(batch, n, *t.shape[1:])  # noqa: E731
        out, _ = flash_attn_func(
            unpack(q, q_lens[0]), unpack(k, k_lens[0]), unpack(v, k_lens[0]),
            qv=unpack(qv, q_lens[0]), gather_kv_indices=unpack(gather_kv_indices, q_lens[0]),
            causal=True, pack_gqa=True, gather_bwd_recompute_p=recompute_p,
        )
        g_call = unpack(g, q_lens[0])
    grads = torch.autograd.grad(out, (q, k, v, qv), g_call)

    if is_fake_mode():
        return

    assert (gather_kv_indices == -1).any(), "test must exercise sentinel slots"
    q_ref, k_ref, v_ref, qv_ref = [x.detach().clone().requires_grad_() for x in (q, k, v, qv)]
    outs_ref, outs_pt = [], []
    for b in range(len(q_lens)):
        sq, sk = slice(cu_q[b], cu_q[b + 1]), slice(cu_k[b], cu_k[b + 1])
        args = (q_ref[sq][None], k_ref[sk][None], v_ref[sk][None])
        kw = dict(causal=True, qv=qv_ref[sq][None], gather_kv_indices=gather_kv_indices[sq][None])
        outs_ref.append(attention_ref(*args, **kw)[0].squeeze(0))
        outs_pt.append(attention_ref(*args, **kw, upcast=False, reorder_ops=True)[0].squeeze(0))
    ref_inputs = (q_ref, k_ref, v_ref, qv_ref)
    grads_ref = torch.autograd.grad(torch.cat(outs_ref), ref_inputs, g)
    grads_pt = torch.autograd.grad(torch.cat(outs_pt), ref_inputs, g)
    for name, a, r, p_ in zip(("dQ", "dK", "dV", "dQv"), grads, grads_ref, grads_pt):
        assert not a.isnan().any(), f"{name} has NaN"
        print_diff_stats(name, a, r, p_)
        if name in ("dQ", "dQv"):
            rel_tok = (a.float() - r.float()).flatten(1).norm(dim=1) / r.float().flatten(1).norm(dim=1)
            print(f"{name} per-token rel-L2 max: {rel_tok.max().item():.3e}")
            lost = torch.nonzero(rel_tok > 0.5).flatten().tolist()
            assert not lost, f"{name}: tokens {lost} lost their gradient (sequence starts {cu_q[:-1]})"
        check_tensor_vs_ref(name, a, r, p_)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("nheads", [128, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_learnable_sink(nheads, varlen, shared_kv, causal, dtype):
    """Sparse-MLA backward with a learnable sink, across every gather_bwd mode.

    The sink only enters through lse (lse = log(exp(sink) + sum_j exp(s_j))),
    so recompute-P's P = exp2(scale_log2 * S - lse_log2) needs no sink-specific
    handling, and dsink = -sum_rows exp(sink - lse) * dpsum depends only on
    dpsum and lse, which the token-chunked backward keeps full-size (the
    preprocess runs once before the chunk loop, the dsink reduce once after
    it). Runs default (load-p), recompute_p, recompute_p + token_chunk and
    load-p + token_chunk with -1-padded indices (non-causal indices under
    causal=True, so the causal key limit must also be applied) and checks:
      1. out, lse and dsink are bitwise-identical across the four modes (with
         FLASH_ATTENTION_MLA_1CTA=1 at 64 heads the recompute modes run the 1CTA
         64-key-block forward: bf16-rounding agreement with the load-P modes,
         bitwise between the two recompute modes);
      2. chunked dq/dqv are bitwise-equal to unchunked (dk/dv within
         fp32-atomic accumulation noise);
      3. every grad including dsink is within the standard tolerance of the
         fp32 reference, in every mode.
    Covers both the rope (q/k present) and the shared-KV (no-rope) kernel
    specializations, non-varlen and varlen (docs split across chunks).
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256
    token_chunk = 200
    if varlen:
        doc_lens = (300, 212)  # doc 0 spans chunks 0-1, doc 1 spans chunks 1-2
        total = sum(doc_lens)
        cu = torch.tensor([0, doc_lens[0], total], device=device, dtype=torch.int32)
        tok_shape = (total,)
        docs = [(0, doc_lens[0], doc_lens[0]), (doc_lens[0], total, doc_lens[1])]
    else:
        seqlen = 512  # 512 = 200 + 200 + 112: exercises a tail chunk
        tok_shape = (1, seqlen)  # token_chunk requires varlen or batch 1
        docs = [(0, seqlen, seqlen)]
    index_fn = random_cutoff_topk_indices if causal else causal_topk_indices
    gather_kv_indices = torch.cat([
        index_fn(1, n, n, topk_len, device).squeeze(0) for _, _, n in docs
    ])
    if not varlen:
        gather_kv_indices = gather_kv_indices.unsqueeze(0)

    q = torch.randn(*tok_shape, nheads, hdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(*tok_shape, nheads_kv, hdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(*tok_shape, nheads_kv, hdimv, device=device, dtype=dtype, requires_grad=True)
    qv = torch.randn(*tok_shape, nheads, hdimv, device=device, dtype=dtype, requires_grad=True)
    sink = torch.randn(nheads, device=device, dtype=dtype, requires_grad=True)
    if shared_kv:
        q, k, qv = qv, v, None
    grad_inputs = ((q, k) if shared_kv else (q, k, v, qv)) + (sink,)
    names = (("dQv", "dV") if shared_kv else ("dQ", "dK", "dV", "dQv")) + ("dSink",)
    atomic_grads = (1,) if shared_kv else (1, 2)
    g = torch.randn(*tok_shape, nheads, hdimv, device=device, dtype=dtype)

    def run(recompute_p, chunk):
        kw = dict(
            qv=qv, gather_kv_indices=gather_kv_indices, causal=causal, pack_gqa=True,
            learnable_sink=sink, gather_bwd_recompute_p=recompute_p,
            gather_bwd_token_chunk=chunk,
        )
        if varlen:
            out, lse = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max(doc_lens), max_seqlen_k=max(doc_lens), **kw,
            )
        else:
            out, lse = flash_attn_func(q, k, v, **kw)
        return out, lse, torch.autograd.grad(out, grad_inputs, g)

    modes = {
        "default": (False, None),
        "recompute_p": (True, None),
        "recompute_p+chunk": (True, token_chunk),
        "load_p+chunk": (False, token_chunk),
    }
    results = {name: run(*args) for name, args in modes.items()}

    if is_fake_mode():
        return

    assert (gather_kv_indices == -1).any(), "test must exercise sentinel slots"
    out, lse, grads = results["default"]
    assert not lse.isnan().any(), "LSE contains NaN"
    kb64 = _mla_kb64_active(nheads, dtype)
    for mode, (out_m, lse_m, grads_m) in results.items():
        for name, t in zip(names, grads_m):
            assert not t.isnan().any(), f"{name} has NaN in mode {mode}"
        if kb64 and modes[mode][0]:
            _assert_mla_fwd_close(out_m, out, lse_m, lse, f"mode {mode} vs default")
            rel = ((grads_m[-1].float() - grads[-1].float()).norm()
                   / grads[-1].float().norm().clamp_min(1e-12)).item()
            assert rel < 1e-2, f"dsink rel_l2 {rel:.2e} in mode {mode}"
            continue
        # Same forward kernel math in every mode (recompute_p only skips the
        # p/row_max stores), and dsink is a function of (dpsum, lse, sink) only.
        assert torch.equal(out_m, out), f"out not bitwise-identical in mode {mode}"
        assert torch.equal(lse_m, lse), f"lse not bitwise-identical in mode {mode}"
        assert torch.equal(grads_m[-1], grads[-1]), f"dsink not bitwise-identical in mode {mode}"
    # the two recompute modes share one forward kernel
    (out_r, lse_r, grads_r), (out_rc, lse_rc, grads_rc) = results["recompute_p"], results["recompute_p+chunk"]
    assert torch.equal(out_rc, out_r) and torch.equal(lse_rc, lse_r), "recompute modes differ"
    assert torch.equal(grads_rc[-1], grads_r[-1]), "dsink differs between the recompute modes"
    for chunked, unchunked in (("recompute_p+chunk", "recompute_p"), ("load_p+chunk", "default")):
        for i, (name, a, b) in enumerate(zip(names[:-1], results[chunked][2], results[unchunked][2])):
            if i in atomic_grads:
                rel = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
                assert rel < 5e-4, f"{name} {chunked} rel_l2 {rel} beyond atomic noise"
            else:
                assert torch.equal(a, b), f"{name} {chunked} not bitwise vs {unchunked}"

    # fp32 reference (per doc for varlen), including dsink
    ref_inputs = tuple(x.detach().clone().requires_grad_() for x in grad_inputs)
    if shared_kv:
        q_ref, k_ref, sink_ref = ref_inputs
        v_ref, qv_ref = k_ref, None
    else:
        q_ref, k_ref, v_ref, qv_ref, sink_ref = ref_inputs
    outs_ref, outs_pt = [], []
    for start, end, _ in docs:
        s = slice(start, end) if varlen else (slice(None), slice(start, end))
        unbatch = (lambda t: t.unsqueeze(0)) if varlen else (lambda t: t)
        doc_args = dict(
            causal=causal, learnable_sink=sink_ref,
            qv=unbatch(qv_ref[s]) if qv_ref is not None else None,
            gather_kv_indices=unbatch(gather_kv_indices[s]),
        )
        o_ref, _ = attention_ref(unbatch(q_ref[s]), unbatch(k_ref[s]), unbatch(v_ref[s]), **doc_args)
        o_pt, _ = attention_ref(
            unbatch(q_ref[s]), unbatch(k_ref[s]), unbatch(v_ref[s]), **doc_args,
            upcast=False, reorder_ops=True,
        )
        outs_ref.append(o_ref.squeeze(0) if varlen else o_ref)
        outs_pt.append(o_pt.squeeze(0) if varlen else o_pt)
    out_ref = torch.cat(outs_ref, dim=0 if varlen else 1)
    out_pt = torch.cat(outs_pt, dim=0 if varlen else 1)
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + fwd_atol
    grads_ref = torch.autograd.grad(out_ref, ref_inputs, g)
    grads_pt = torch.autograd.grad(out_pt, ref_inputs, g)
    for mode, (_, _, grads_m) in results.items():
        for name, a, r, p_ in zip(names[:-1], grads_m, grads_ref, grads_pt):
            print_diff_stats(f"{name}({mode})", a, r, p_)
            check_tensor_vs_ref(f"{name}({mode})", a, r, p_)
        check_dsink_vs_ref(grads_m[-1], grads_ref[-1], grads_pt[-1])


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mqa"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("local_enum", [0])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("add_unused_qkv", [False])
# Under the 1CTA flag also run the dense (kv_sparsity=False) cases, which the 2CTA matrix
# leaves commented out by default.
@pytest.mark.parametrize(
    "kv_sparsity",
    [False, True] if MLA_1CTA else [True],
)
@pytest.mark.parametrize("hdim", [64])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (3, 3),
        (3, 128),
        (128, 3),
        (256, 256),
        (1025, 511),
        (511, 1025),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (4096, 4096),
        (1, 4096),
    ],
)
@pytest.mark.parametrize("varlen_mode", ["random", "full"])
# @pytest.mark.parametrize("varlen_mode", ["random"])
@pytest.mark.parametrize(
    "zero_lengths_q, zero_lengths_k",
    [
        (False, False),
        # (True, False),
    ],
)
@pytest.mark.parametrize(
    "unpad_q, unpad_kv",
    [
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ],
)
# @pytest.mark.parametrize("seed", [i for i in range(10)])
@pytest.mark.parametrize("seed", [0])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_absorbed_varlen(
    seqlen_q,
    seqlen_k,
    hdim,
    add_unused_qkv,
    causal,
    local_enum,
    deterministic,
    has_learnable_sink,
    mha_type,
    dtype,
    varlen_mode,
    zero_lengths_q,
    zero_lengths_k,
    unpad_q,
    unpad_kv,
    kv_sparsity,
    shared_kv,
    seed,
):
    check_fwd_deterministic = True
    test_bwd = unpad_q and unpad_kv and kv_sparsity
    hdimv = 512
    if not IS_SM100:
        pytest.skip()
    local = local_enum > 0
    if local and causal:
        pytest.skip()
    if local:
        pytest.xfail("mla absorbed: local not supported yet")
    device = "cuda"
    # set seed
    seed = seed + seqlen_q + seqlen_k + hdim + int(causal) * 2 + int(local)
    random.seed(seed)
    torch.random.manual_seed(seed)
    nheads_vals = [128] if kv_sparsity else [16, 128]
    seqlen_k_base = max(min(seqlen_k // 256 * 256, 1024), 256)
    gather_kv_lengths = [seqlen_k_base - 128, seqlen_k_base] if kv_sparsity else [0]
    seqlen_q_og, seqlen_k_og = seqlen_q, seqlen_k
    for nheads, gather_kv_length in itertools.product(nheads_vals, gather_kv_lengths):
        nheads_kv = nheads if mha_type == "mha" else (8 if mha_type == "gqa" else 1)
        if kv_sparsity and seqlen_k_og < gather_kv_length:
            seqlen_k = gather_kv_length
        # varlen reference is set up to require this
        if causal or local:
            seqlen_q = max(seqlen_q_og, seqlen_k)
            seqlen_k = seqlen_q
        batch_size = 12 if seqlen_q <= 512 else 3 if seqlen_q <= 2048 else 1
        print(f"{batch_size=}, {nheads=}, {nheads_kv=}, {gather_kv_length=}, (max) {seqlen_q=}, (max) {seqlen_k=}")
        q_ref = torch.randn(batch_size, seqlen_q, nheads, hdim, device=device, dtype=dtype).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdim, device=device, dtype=dtype).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, hdimv, device=device, dtype=dtype).requires_grad_()
        qv_ref = torch.randn(batch_size, seqlen_q, nheads, hdimv, device=device, dtype=dtype).requires_grad_()
        if kv_sparsity:
            gather_kv_indices = torch.rand(batch_size, seqlen_q, gather_kv_length, device=device).argsort(dim=-1).to(torch.int32)
        else:
            gather_kv_indices = None
            
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (
            (None, None) if not local else tuple(random.randrange(0, seqlen_k) for _ in range(2))
        )
        if local_enum == 2:
            window_size = (None, window_size[1])
        elif local_enum == 3:
            window_size = (window_size[0], None)
        if local:
            print("window size = ", window_size)
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device, requires_grad=True)
        else:
            learnable_sink = None
        q, k, v, qv = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref, qv_ref)]
        query_padding_mask = generate_random_padding_mask(
            seqlen_q,
            batch_size,
            device,
            mode=varlen_mode,
            zero_lengths=zero_lengths_q,
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k,
            batch_size,
            device,
            mode=varlen_mode,
            zero_lengths=zero_lengths_k,
            min_seqlen=gather_kv_length if kv_sparsity else None,
        )
        def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
            if add_unused:
                another_mask = generate_random_padding_mask(max_seq_len, bs, device)
                attn_mask = torch.logical_and(padding_mask, another_mask)
                unused_mask = torch.logical_xor(
                    torch.logical_or(padding_mask, another_mask), attn_mask
                )
            else:
                attn_mask = padding_mask
                unused_mask = None
            return attn_mask, unused_mask

        query_padding_mask, query_unused_mask = _gen_unused_masks(
            query_padding_mask, add_unused_qkv, seqlen_q, batch_size, q.device
        )
        # query_padding_mask[:] = True
        # query_unused_mask = None
        key_padding_mask, key_unused_mask = _gen_unused_masks(
            key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device
        )
        if causal or local:
            key_padding_mask = query_padding_mask
        (
            q_unpad,
            k_unpad,
            v_unpad,
            qv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            qv,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            qv=qv,
            kvpacked=False,
            query_unused_mask=query_unused_mask,
            key_unused_mask=key_unused_mask,
        )
        # unpad gather_kv_indices
        if kv_sparsity:
            _, indices_q, _, _, _ = unpad_input(
                q, query_padding_mask, query_unused_mask
            )
            gather_kv_indices_unpad = rearrange(gather_kv_indices, "b s ... -> (b s) ...")[indices_q]
        else:
            gather_kv_indices_unpad = None
        if unpad_q:
            print("cu_seqlens_q = ", cu_seqlens_q)
        else:
            print("seqused_q = ", seqused_q)
        if unpad_kv:
            print("cu_seqlens_k = ", cu_seqlens_k)
        else:
            print("seqused_k = ", seqused_k)
        q_unpad, k_unpad, v_unpad, qv_unpad = [
            x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad, qv_unpad)
        ]
        if shared_kv:
            q, q_unpad = qv, qv_unpad
            k, k_unpad = v, v_unpad
            qv = qv_unpad = None
            q_ref = qv_ref
            k_ref = v_ref
            qv_ref = None

        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            window_size=window_size,
            learnable_sink=learnable_sink,
            gather_kv_indices=gather_kv_indices,
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            window_size=window_size,
            learnable_sink=learnable_sink,
            upcast=False,
            reorder_ops=True,
            gather_kv_indices=gather_kv_indices,
        )

        if not is_fake_mode():
            print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
            print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

            if query_unused_mask is not None:
                q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

            # Numerical error if we just do any arithmetic on out_ref
            fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
            rtol = 2

        pack_gqa_vals = [True]
        # SplitKV with qv is 1CTA-only and never sparse; unpad_q exercises both cu_seqlens_q
        # and seqused_q.
        num_splits_vals = [1, 3] if MLA_1CTA and not DISABLE_SPLIT and not kv_sparsity else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            # SplitKV not supported on SM90/SM120 - skip this iteration
            if (IS_SM90 or IS_SM120) and num_splits > 1:
                continue
            out_unpad, lse = flash_attn_varlen_func(
                q_unpad if unpad_q else q,
                k_unpad if unpad_kv else k,
                v_unpad if unpad_kv else v,
                qv_unpad if unpad_q else qv,
                cu_seqlens_q=cu_seqlens_q if unpad_q else None,
                cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
                max_seqlen_q=seqlen_q,
                max_seqlen_k=seqlen_k,
                min_seqlen_k=gather_kv_length if kv_sparsity else None,
                seqused_q=seqused_q if not unpad_q else None,
                seqused_k=seqused_k if not unpad_kv else None,
                causal=causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                deterministic=deterministic,
                gather_kv_indices=gather_kv_indices_unpad if unpad_q else gather_kv_indices,
            )
            out = output_pad_fn(out_unpad) if unpad_q else out_unpad
            if is_fake_mode():
                # no more flash_attn cutedsl calls for the rest of the loop
                # skip data-dependent postprocessing
                continue
            if query_unused_mask is not None:
                out.masked_fill_(q_zero_masking, 0.0)
            # When unpad_q=False with seqused_q, the kernel doesn't write positions
            # beyond seqused_q, so those contain uninitialized values. Mask them out
            # before comparing.
            out_cmp, out_ref_cmp, out_pt_cmp = out, out_ref, out_pt
            if not unpad_q and seqused_q is not None:
                seqused_mask = torch.arange(seqlen_q, device=device)[None, :] < seqused_q[:, None]
                seqused_mask = rearrange(seqused_mask, "b s -> b s 1 1")
                out_cmp = out.clone().masked_fill_(~seqused_mask, 0.0)
                out_ref_cmp = out_ref.clone().masked_fill_(~seqused_mask, 0.0)
                out_pt_cmp = out_pt.clone().masked_fill_(~seqused_mask, 0.0)
            print(f"Output max diff: {(out_cmp - out_ref_cmp).abs().max().item()}")
            print(f"Output mean diff: {(out_cmp - out_ref_cmp).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most 3x the numerical error
            # of a Pytorch implementation.
            assert (out_cmp - out_ref_cmp).abs().max().item() <= rtol * (
                out_pt_cmp - out_ref_cmp
            ).abs().max().item() + fwd_atol
            # LSE sanity: only valid positions (packed unpad path; padded path
            # can legitimately contain uninit tail beyond seqused_q).
            if unpad_q:
                assert not torch.isnan(lse).any(), "LSE contains NaN"

            repeats = 10 if check_fwd_deterministic else 0
            for iter in range(repeats):
                out_unpad2, lse = flash_attn_varlen_func(
                    q_unpad if unpad_q else q,
                    k_unpad if unpad_kv else k,
                    v_unpad if unpad_kv else v,
                    qv_unpad if unpad_q else qv,
                    cu_seqlens_q=cu_seqlens_q if unpad_q else None,
                    cu_seqlens_k=cu_seqlens_k if unpad_kv else None,
                    max_seqlen_q=seqlen_q,
                    max_seqlen_k=seqlen_k,
                    min_seqlen_k=gather_kv_length if kv_sparsity else None,
                    seqused_q=seqused_q if not unpad_q else None,
                    seqused_k=seqused_k if not unpad_kv else None,
                    causal=causal,
                    window_size=window_size,
                    learnable_sink=learnable_sink,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    deterministic=deterministic,
                    gather_kv_indices=gather_kv_indices_unpad if unpad_q else gather_kv_indices,
                )
                out2 = output_pad_fn(out_unpad2) if unpad_q else out_unpad2
                if query_unused_mask is not None:
                    out2.masked_fill_(q_zero_masking, 0.0)
                # When unpad_q=False with seqused_q, the kernel doesn't write positions
                # beyond seqused_q, so those contain uninitialized values. Mask them out
                # before comparing.
                if not unpad_q and seqused_q is not None:
                    seqused_mask = torch.arange(seqlen_q, device=device)[None, :] < seqused_q[:, None]
                    seqused_mask = rearrange(seqused_mask, "b s -> b s 1 1")
                    out2.masked_fill_(~seqused_mask, 0.0)
                # print(f"out2 max: {out2.abs().max().item()}, {iter=}")
                # print(f"out vs out2 max diff: {(out_cmp - out2).abs().max().item()}, {iter=}")
                # print(f"out vs out2 mean diff: {(out_cmp - out2).abs().mean().item()}, {iter=}")
                assert torch.equal(out_cmp, out2), f"non-deterministic with max diff = {(out_cmp - out2).abs().max().item()} on {iter=}"

        if test_bwd:
            print("VARLEN BWD SPARSE MLA")
            g_unpad = torch.randn_like(out_unpad)
            sink_inputs = (learnable_sink,) if has_learnable_sink else ()
            if shared_kv:
                dq_unpad, dk_unpad, *dsink = torch.autograd.grad(
                    out_unpad,
                    (
                        q_unpad if unpad_q else q,
                        k_unpad if unpad_kv else k,
                        *sink_inputs,
                    ),
                    g_unpad,
                    allow_unused=True,
                )
                dv_unpad, dqv_unpad = None, None
            else:
                dq_unpad, dk_unpad, dv_unpad, dqv_unpad, *dsink = torch.autograd.grad(
                    out_unpad,
                    (
                        q_unpad if unpad_q else q,
                        k_unpad if unpad_kv else k,
                        v_unpad if unpad_kv else v,
                        qv_unpad if unpad_q else qv,
                        *sink_inputs,
                    ),
                    g_unpad,
                    allow_unused=True,
                )

            if is_fake_mode():
                continue

            dq = dq_pad_fn(dq_unpad) if unpad_q else dq_unpad
            dk = dk_pad_fn(dk_unpad) if unpad_kv else dk_unpad
            dv = dk_pad_fn(dv_unpad) if unpad_kv and dv_unpad is not None else dv_unpad
            dqv = dq_pad_fn(dqv_unpad) if unpad_q and dqv_unpad is not None else dqv_unpad

            if key_unused_mask is not None:
                k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
                dk.masked_fill_(k_zero_masking, 0.0)
                if dv is not None:
                    dv.masked_fill_(k_zero_masking, 0.0)
            if query_unused_mask is not None:
                dq.masked_fill_(q_zero_masking, 0.0)
                if dqv is not None:
                    dqv.masked_fill_(q_zero_masking, 0.0)
            if not unpad_kv:
                dk.masked_fill_(rearrange(~key_padding_mask, "b s -> b s 1 1"), 0.0)
                if dv is not None:
                    dv.masked_fill_(rearrange(~key_padding_mask, "b s -> b s 1 1"), 0.0)
            if not unpad_q:
                dq.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
                if dqv is not None:
                    dqv.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

            g = output_pad_fn(g_unpad) if unpad_q else g_unpad

            if shared_kv:
                dq_ref, dk_ref, *dsink_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, *sink_inputs), g)
                dq_pt, dk_pt, *dsink_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, *sink_inputs), g)
                dv, dqv, dv_ref, dqv_ref, dv_pt, dqv_pt = None, None, None, None, None, None
            else:
                dq_ref, dk_ref, dv_ref, dqv_ref, *dsink_ref = torch.autograd.grad(
                    out_ref, (q_ref, k_ref, v_ref, qv_ref, *sink_inputs), g
                )
                dq_pt, dk_pt, dv_pt, dqv_pt, *dsink_pt = torch.autograd.grad(
                    out_pt, (q_ref, k_ref, v_ref, qv_ref, *sink_inputs), g
                )

            print_diff_stats("dQ", dq, dq_ref, dq_pt)
            print_diff_stats("dK", dk, dk_ref, dk_pt)
            print_diff_stats("dV", dv, dv_ref, dv_pt)
            print_diff_stats("dQv", dqv, dqv_ref, dqv_pt)

            check_tensor_vs_ref("dQ", dq, dq_ref, dq_pt)
            check_tensor_vs_ref("dK", dk, dk_ref, dk_pt)
            check_tensor_vs_ref("dV", dv, dv_ref, dv_pt)
            check_tensor_vs_ref("dQv", dqv, dqv_ref, dqv_pt)
            if has_learnable_sink:
                check_dsink_vs_ref(dsink[0], dsink_ref[0], dsink_pt[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1, 16, 64, 128])
@pytest.mark.parametrize("has_qk", [True, False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (4, 256),
        (64, 512),
        (1, 2048),
        (2048, 2048),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_paged(dtype, seqlen_q, seqlen_k, page_size, causal, has_qk):
    if not IS_SM100:
        pytest.skip("MLA paged KV only supported on SM100")
    device = "cuda"
    d, dv = 64, 512
    nheads = 128
    nheads_kv = 1
    batch_size = 49 if seqlen_k <= 512 else 7

    torch.random.manual_seed(0)

    # Non-paged reference tensors (varlen format)
    q = k = None
    if has_qk:
        q = torch.randn(batch_size * seqlen_q, nheads, d, device=device, dtype=dtype)
        k = torch.randn(batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size * seqlen_k, nheads_kv, dv, device=device, dtype=dtype)
    qv = torch.randn(batch_size * seqlen_q, nheads, dv, device=device, dtype=dtype)

    cu_seqlens_q = torch.tensor(
        [i * seqlen_q for i in range(batch_size + 1)], dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.tensor(
        [i * seqlen_k for i in range(batch_size + 1)], dtype=torch.int32, device=device
    )

    # Non-paged reference
    out_ref, _ = flash_attn_varlen_func(
        q, k, v, qv=qv,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
        causal=causal,
    )

    # Create paged K/V cache
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    total_pages = num_pages_per_seq * batch_size
    k_paged = None
    if has_qk:
        k_paged = torch.zeros(total_pages, page_size, nheads_kv, d, device=device, dtype=dtype)
    v_paged = torch.zeros(total_pages, page_size, nheads_kv, dv, device=device, dtype=dtype)
    page_table = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)

    # Fill paged K/V from contiguous K/V (sequential page assignment)
    for b in range(batch_size):
        for p in range(num_pages_per_seq):
            page_idx = b * num_pages_per_seq + p
            start = p * page_size
            end = min(start + page_size, seqlen_k)
            k_offset = b * seqlen_k
            if start < seqlen_k:
                if has_qk:
                    k_paged[page_idx, :end - start] = k[k_offset + start:k_offset + end]
                v_paged[page_idx, :end - start] = v[k_offset + start:k_offset + end]
            page_table[b, p] = page_idx

    seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)

    # Paged output (triggers cp.async path if page_size != 128)
    out, _ = flash_attn_varlen_func(
        q, k_paged, v_paged, qv=qv,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=None,
        seqused_k=seqused_k, page_table=page_table,
        causal=causal,
    )

    if is_fake_mode():
        return

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(out, out_ref)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(64, 64), (128, 128), (128, 256)])
@pytest.mark.skipif(not (IS_SM100 or IS_SM110), reason="MLA kernel requires SM100/SM110")
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_mla_sink_precision_vs_fp64(seqlen_q, seqlen_k, causal):
    """Sparse MLA + learnable sink: kernel fwd/bwd error vs an fp64 ground truth.

    `attention_ref` computes its sink path in fp32, so this is the only check that measures
    the kernel against an exact reference. The kernel (bf16 in, fp32 accumulate) must stay
    within 3x the error of an fp32 eager evaluation of the same bf16 inputs, after rounding
    the eager result to the kernel's output dtype.
    """
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)
    batch_size, nheads, d = 2, 128, 512
    gather_kv_length = ((seqlen_k + 127) // 128) * 128

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, 1, d, device=device, dtype=dtype)
    sink = torch.randn(nheads, device=device, dtype=dtype)
    g = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    def reference(q, v, sink):
        """softmax over [sink, scores] per row; the sink column has no value vector."""
        scores = torch.einsum("bthd,bsd->bhts", q / math.sqrt(d), v[:, :, 0])
        if causal:
            row_idx = torch.arange(seqlen_q, device=device)[:, None]
            col_idx = torch.arange(seqlen_k, device=device)[None, :]
            scores = scores.masked_fill(col_idx > row_idx + (seqlen_k - seqlen_q), float("-inf"))
        sink_logit = sink.view(1, nheads, 1, 1).expand(batch_size, nheads, seqlen_q, 1)
        attn = torch.softmax(torch.cat([sink_logit, scores], dim=-1), dim=-1)[..., 1:]
        return torch.einsum("bhts,bsd->bthd", attn, v[:, :, 0])

    inputs_64 = [t.double().requires_grad_() for t in (q, v, sink)]
    inputs_32 = [t.float().requires_grad_() for t in (q, v, sink)]
    inputs_kern = [t.clone().requires_grad_() for t in (q, v, sink)]

    indices = torch.full((batch_size, seqlen_q, gather_kv_length), -1, device=device, dtype=torch.int32)
    indices[:, :, :seqlen_k] = torch.arange(seqlen_k, device=device, dtype=torch.int32)
    q_kern, v_kern, sink_kern = inputs_kern
    out_kern, _ = flash_attn_func(
        q_kern, v_kern, v_kern, causal=causal, learnable_sink=sink_kern, gather_kv_indices=indices
    )
    grads_kern = torch.autograd.grad(out_kern, inputs_kern, g)
    if is_fake_mode():
        return

    out_64 = reference(*inputs_64)
    out_32 = reference(*inputs_32)
    grads_64 = torch.autograd.grad(out_64, inputs_64, g.double())
    grads_32 = torch.autograd.grad(out_32, inputs_32, g.float())

    for name, exact, eager, kern in zip(
        ("out", "dQ", "dV", "dSink"), (out_64, *grads_64), (out_32, *grads_32), (out_kern, *grads_kern)
    ):
        # Round the eager result to the kernel's output dtype so both sides pay the same
        # storage-precision cost and the ratio isolates the kernel's arithmetic error.
        eager_err = (eager.to(kern.dtype).double() - exact).abs().max().item()
        kern_err = (kern.double() - exact).abs().max().item()
        print(f"[causal={causal}, sq={seqlen_q}, sk={seqlen_k}] {name}: kernel {kern_err:.3e} vs eager fp32 {eager_err:.3e}")
        assert kern_err <= 3 * eager_err + 1e-6, f"{name}: kernel error {kern_err:.3e} > 3x eager fp32 {eager_err:.3e}"


def _sparse_mla_fp64_reference(q, k, v, qv, gather_kv_indices, softmax_scale, causal, g):
    """fp64 out and grads of one sparse-MLA problem (MQA, top-k gathered KV, no batch dim).

    q: (T, H, hdim), k: (S, 1, hdim), v: (S, 1, hdimv), qv: (T, H, hdimv),
    gather_kv_indices: (T, W) int32 with -1 sentinels, g: (T, H, hdimv).
    Returns out, dq, dk, dv, dqv in fp64 (dk/dv keep the kv-head dim), plus a dict with
    the grads of an emulated ideal bf16 pipeline (exact dpsum; P rounded to bf16 only as
    the dV operand, dS rounded to bf16 once as the dQ/dK operand, bf16 outputs) to
    calibrate what "at the bf16 floor" means for these inputs.
    """
    T, S = q.shape[0], k.shape[0]
    device = q.device
    qf, qvf, gf = q.double(), qv.double(), g.double()
    kf, vf = k[:, 0].double(), v[:, 0].double()
    ix = gather_kv_indices.long()
    valid = ix >= 0
    if causal:  # bottom-right aligned causal limit, as in the kernel
        valid &= ix <= (torch.arange(T, device=device) + (S - T))[:, None]
    ixs = ix.clamp_min(0)
    kg, vg = kf[ixs], vf[ixs]  # (T, W, hdim), (T, W, hdimv)
    s = torch.einsum("thd,twd->thw", qf, kg) + torch.einsum("thd,twd->thw", qvf, vg)
    s = (s * softmax_scale).masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1).nan_to_num(0.0)
    o = torch.einsum("thw,twd->thd", p, vg)
    dp = torch.einsum("thd,twd->thw", gf, vg)
    ds = p * (dp - (gf * o).sum(-1, keepdim=True)) * softmax_scale
    dq = torch.einsum("thw,twd->thd", ds, kg)
    dqv = torch.einsum("thw,twd->thd", ds, vg)
    dk = torch.zeros_like(kf).index_add_(0, ix[valid], torch.einsum("thw,thd->twd", ds, qf)[valid])
    dv_g = torch.einsum("thw,thd->twd", ds, qvf) + torch.einsum("thw,thd->twd", p, gf)
    dv = torch.zeros_like(vf).index_add_(0, ix[valid], dv_g[valid])

    def bf16(x):
        return x.to(torch.bfloat16).double()

    ds_b, p_b = bf16(ds), bf16(p)
    dv_gi = torch.einsum("thw,thd->twd", ds_b, qvf) + torch.einsum("thw,thd->twd", p_b, gf)
    ideal = dict(
        dq=bf16(torch.einsum("thw,twd->thd", ds_b, kg)),
        dqv=bf16(torch.einsum("thw,twd->thd", ds_b, vg)),
        dk=bf16(torch.zeros_like(kf).index_add_(0, ix[valid], torch.einsum("thw,thd->twd", ds_b, qf)[valid])).unsqueeze(1),
        dv=bf16(torch.zeros_like(vf).index_add_(0, ix[valid], dv_gi[valid])).unsqueeze(1),
    )
    return o, dq, dk.unsqueeze(1), dv.unsqueeze(1), dqv, ideal


def self_including_topk_indices(seqlen, topk_len, device):
    """Causal top-k indices that always contain the query's own key (plus random earlier
    keys), -1 padded: the selection an indexer makes for strongly self-attending tokens.
    Pure tensor ops (no data-dependent Python) so it also runs under FakeTensorMode."""
    n_keys = max(seqlen, topk_len)
    scores = torch.rand(seqlen, n_keys, device=device)
    query_idx = torch.arange(seqlen, device=device)
    scores[query_idx, query_idx] = 2.0  # the query's own key always wins
    key_idx = torch.arange(n_keys, device=device)
    invalid = (key_idx[None, :] > query_idx[:, None]) | (key_idx >= seqlen)[None, :]
    scores.masked_fill_(invalid, float("-inf"))
    val, idx = scores.topk(topk_len, dim=-1)
    idx = idx.masked_fill(torch.isinf(val), -1)
    return idx.to(torch.int32).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# 64 and 24 heads pad to the 128-row forward tile (pack_gqa.qheads_first_tma_view): the
# residual store must skip the padded rows. Recompute-P at 24 heads also pads the 64-row
# backward tile (+inf lse_log2 pads).
@pytest.mark.parametrize(
    "nheads,recompute_p",
    [(128, False), (128, True), (64, False), (64, True), (24, False), (24, True)],
)
@pytest.mark.parametrize("varlen", [False, True])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_bwd_precise_dpsum(varlen, nheads, recompute_p, causal, dtype):
    """Sparse-MLA training numerics: the forward writes o_lo = fp32(O) - bf16(O) and the
    backward preprocess forms dpsum = rowsum(dO * (O + o_lo)); the forward runs the online
    softmax with an exact running max.

    Inputs are built so every token attends ~80% to its own key: then dP ~ dpsum and the
    bf16 rounding of O, which is row-coherent in dS = P * (dP - dpsum), dominates the dq/dk
    error (~1% rel-L2 vs fp64, with a tail of rows at ~100%). Checks:
      1. the internal forward/backward entry points reproduce the autograd path;
      2. dq/dqv/dk rel-L2 vs an fp64 reference drops well below the error of the same
         backward run without the residual (internal entry point, dpsum from bf16 out
         only) and lands within 1.5x of an emulated ideal bf16 pipeline with exact dpsum
         (the bf16 floor for these inputs); dv does not get worse;
      3. o_lo is the bf16 rounding residual of out (|o_lo| <= half an ulp of out) and
         out + o_lo is closer to the fp64 out than out alone; with padded heads (nheads <
         128) this also catches padded rows wrapping into the next token's residual;
      4. composes with gather_bwd_token_chunk (dq/dqv bitwise vs unchunked).
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    nheads_kv, hdim, hdimv = 1, 64, 512
    topk_len = 256
    seqlens = [512, 384] if varlen else [512]
    total = sum(seqlens)
    softmax_scale = (hdim + hdimv) ** -0.5
    beta = 0.4  # self-key boost: q_t += beta * k_t, qv_t += beta * v_t

    k32 = torch.randn(total, nheads_kv, hdim, device=device)
    v32 = torch.randn(total, nheads_kv, hdimv, device=device)
    q32 = torch.randn(total, nheads, hdim, device=device) + beta * k32
    qv32 = torch.randn(total, nheads, hdimv, device=device) + beta * v32
    q, k, v, qv = [x.to(dtype).requires_grad_() for x in (q32, k32, v32, qv32)]
    g = torch.randn(total, nheads, hdimv, device=device, dtype=dtype)
    gather_kv_indices = torch.cat(
        [self_including_topk_indices(L, topk_len, device) for L in seqlens]
    ).contiguous()
    cu_bounds = [0] + list(itertools.accumulate(seqlens))

    if varlen:
        cu_seqlens = torch.tensor(cu_bounds, dtype=torch.int32, device=device)

        def run(**kw):
            return flash_attn_varlen_func(
                q, k, v, qv=qv, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
                gather_kv_indices=gather_kv_indices, softmax_scale=softmax_scale,
                causal=causal, pack_gqa=True, gather_bwd_recompute_p=recompute_p, **kw,
            )

        g_call = g
    else:

        def run(**kw):
            return flash_attn_func(
                q[None], k[None], v[None], qv=qv[None], gather_kv_indices=gather_kv_indices[None],
                softmax_scale=softmax_scale, causal=causal, pack_gqa=True,
                gather_bwd_recompute_p=recompute_p, **kw,
            )

        g_call = g[None]

    out1, lse1 = run()
    grads1 = torch.autograd.grad(out1, (q, k, v, qv), g_call)
    out_ck, _ = run(gather_bwd_token_chunk=200)
    grads_ck = torch.autograd.grad(out_ck, (q, k, v, qv), g_call)
    # Baseline without the residual through the internal entry points (the public API
    # always uses it for training forwards): same forward, dpsum from the bf16 out only.
    if varlen:
        q_c, k_c, v_c, qv_c, idx_c = q, k, v, qv, gather_kv_indices
        seq_kw = dict(
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        )
    else:
        q_c, k_c, v_c, qv_c, idx_c = q[None], k[None], v[None], qv[None], gather_kv_indices[None]
        seq_kw = {}
    with torch.no_grad():
        out0, lse0, p0, row_max0, o_lo = _flash_attn_fwd(
            q_c, k_c, v_c, qv=qv_c, gather_kv_indices=idx_c, softmax_scale=softmax_scale,
            causal=causal, pack_gqa=True, gather_bwd_recompute_p=recompute_p, **seq_kw,
        )
        dq0, dk0, dv0, dqv0, _ = _flash_attn_bwd_sparse_mla(
            q_c, k_c, v_c, qv_c, out0, g_call, lse0, p0, row_max0, idx_c,
            softmax_scale=softmax_scale, causal=causal, recompute_p=recompute_p, o_lo=None,
            **seq_kw,
        )
    grads0 = (dq0, dk0, dv0, dqv0)

    if is_fake_mode():
        # no more flash_attn cutedsl calls; skip data-dependent checks
        return

    assert torch.equal(out0, out1) and torch.equal(lse0, lse1), "internal fwd must match autograd fwd"
    assert torch.equal(out_ck, out1)
    for i, (a, b) in enumerate(zip(grads_ck, grads1)):
        if i in (0, 3):  # dq, dqv: pure GEMM consumers of identical dS tiles
            assert torch.equal(a, b), f"chunked grad {i} not bitwise vs unchunked"

    # fp64 reference, per document
    refs = []
    for i in range(len(seqlens)):
        s, e = cu_bounds[i], cu_bounds[i + 1]
        refs.append(
            _sparse_mla_fp64_reference(
                q[s:e].detach(), k[s:e].detach(), v[s:e].detach(), qv[s:e].detach(),
                gather_kv_indices[s:e], softmax_scale, causal, g[s:e],
            )
        )
    out_ref, dq_ref, dk_ref, dv_ref, dqv_ref = [torch.cat(x, dim=0) for x in list(zip(*refs))[:5]]
    ideal_ref = {n: torch.cat([r[5][n] for r in refs], dim=0) for n in ("dq", "dk", "dv", "dqv")}
    out_flat = out1.reshape(total, nheads, hdimv)

    def rel_l2(a, r):
        return ((a.reshape(r.shape).double() - r).norm() / r.norm()).item()

    names = ("dq", "dk", "dv", "dqv")
    refs_by_name = dict(zip(names, (dq_ref, dk_ref, dv_ref, dqv_ref)))
    err0 = {n: rel_l2(a, refs_by_name[n]) for n, a in zip(names, grads0)}
    err1 = {n: rel_l2(a, refs_by_name[n]) for n, a in zip(names, grads1)}
    err_ideal = {n: rel_l2(ideal_ref[n], refs_by_name[n]) for n in names}
    print("rel-L2 vs fp64 without/with O residual (ideal bf16 pipeline): "
          + ", ".join(f"{n} {err0[n]:.4%}/{err1[n]:.4%} ({err_ideal[n]:.4%})" for n in names))
    assert err0["dq"] > 5e-3, "test inputs must be peaked enough for the bf16-O dpsum error to dominate"
    for n in ("dq", "dqv", "dk"):
        assert err1[n] < 0.6 * err0[n], f"{n}: the O residual did not reduce the error ({err0[n]:.4%} -> {err1[n]:.4%})"
    for n in names:
        assert err1[n] <= 1.5 * err_ideal[n], f"{n}: rel-L2 {err1[n]:.4%} vs ideal bf16 pipeline {err_ideal[n]:.4%}"

    assert o_lo is not None and o_lo.shape == out1.shape and o_lo.dtype == out1.dtype
    half_ulp = out1.float().abs() * 2**-8
    assert (o_lo.float().abs() <= half_ulp + 1e-30).all(), "o_lo must be the bf16 rounding residual of out"
    assert rel_l2(out_flat.double() + o_lo.reshape(out_flat.shape).double(), out_ref) < rel_l2(out_flat, out_ref)


def _self_last_permutation(idx):
    """Move slot 0 of every row of a -1 padded top-k index tensor to the last valid slot
    (pure tensor ops so it also runs under FakeTensorMode). Same set per row, new order."""
    n_valid = (idx >= 0).sum(-1, keepdim=True)
    pos = torch.arange(idx.shape[-1], device=idx.device)[None, :]
    src = torch.where(pos < n_valid - 1, pos + 1, torch.where(pos == n_valid - 1, torch.zeros_like(pos), pos))
    return torch.gather(idx, -1, src.expand_as(idx)).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads", [128, 64])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_sparse_topk_order_invariance(nheads, causal, dtype):
    """The sparse-MLA training forward must not care about the ORDER of the per-row top-k
    indices beyond bf16 rounding noise.

    The kernel walks the index blocks from the last to the first, so an indexer that puts
    the query's own (dominant) key in slot 0 has it processed last. With a lazy running
    max the dominant P is exp2(delta) instead of exactly 1.0 and its bf16 rounding is a
    coherent gain error on the whole output row that flips with the order; with the exact
    running max used for training forwards it is exactly 1.0 in either order.

    Runs the same set with the dominant key first and last and checks that the per-row
    coherent component of the difference (projection of out_first - out_last onto the
    fp64 out row) is at the level expected from independent per-element rounding, and
    that both orderings are within 2x of the ideal bf16 output floor in that metric.
    """
    if not IS_SM100:
        pytest.skip()
    device = "cuda"
    torch.random.manual_seed(0)
    nheads_kv, hdim, hdimv = 1, 64, 512
    total, topk_len = 512, 256
    softmax_scale = (hdim + hdimv) ** -0.5
    beta = 0.25  # self-key boost: q_t += beta * k_t, qv_t += beta * v_t (~50% self weight)

    k32 = torch.randn(total, nheads_kv, hdim, device=device)
    v32 = torch.randn(total, nheads_kv, hdimv, device=device)
    q32 = torch.randn(total, nheads, hdim, device=device) + beta * k32
    qv32 = torch.randn(total, nheads, hdimv, device=device) + beta * v32
    q, k, v, qv = [x.to(dtype).requires_grad_() for x in (q32, k32, v32, qv32)]
    g = torch.randn(total, nheads, hdimv, device=device, dtype=dtype)
    idx_first = self_including_topk_indices(total, topk_len, device)  # own key in slot 0
    idx_last = _self_last_permutation(idx_first)

    def run(idx):
        out, _ = flash_attn_func(
            q[None], k[None], v[None], qv=qv[None], gather_kv_indices=idx[None],
            softmax_scale=softmax_scale, causal=causal, pack_gqa=True,
        )
        grads = torch.autograd.grad(out, (q, k, v, qv), g[None])
        return out[0], grads

    out_first, grads_first = run(idx_first)
    out_last, grads_last = run(idx_last)

    if is_fake_mode():
        return

    assert torch.equal(idx_first.sort(-1).values, idx_last.sort(-1).values)
    assert not torch.equal(out_first, out_last), "orderings must differ in bf16 rounding for the test to mean anything"

    o_ref, dq_ref, dk_ref, dv_ref, dqv_ref, _ = _sparse_mla_fp64_reference(
        q.detach(), k.detach(), v.detach(), qv.detach(), idx_first, softmax_scale, causal, g,
    )
    den = (o_ref * o_ref).sum(-1)  # (total, nheads)
    keep = den > 1e-12 * den.max()

    def row_gain(a, b):
        """Coherent per-row component of (a - b) along the reference out row, and the
        per-row elementwise relative rms of the same difference."""
        d = a.double() - b.double()
        gain = (d * o_ref).sum(-1) / den.clamp_min(1e-300)
        elem = (d * d).sum(-1).sqrt() / den.clamp_min(1e-300).sqrt()
        return gain[keep], elem[keep]

    def rms(x):
        return x.pow(2).mean().sqrt().item()

    # coherent component of the ordering difference vs its random-noise expectation
    gain_diff, elem_diff = row_gain(out_first, out_last)
    noise_floor = rms(elem_diff) * math.sqrt(3.0 / hdimv)
    # each ordering vs fp64, against the bf16 output-rounding floor for these rows
    gain_first, _ = row_gain(out_first, o_ref)
    gain_last, _ = row_gain(out_last, o_ref)
    gain_ideal, _ = row_gain(o_ref.to(dtype), o_ref)
    print(f"row-gain rms: first-vs-last {rms(gain_diff):.3e} (noise floor {noise_floor:.3e}); "
          f"vs fp64: first {rms(gain_first):.3e}, last {rms(gain_last):.3e}, ideal bf16 {rms(gain_ideal):.3e}")
    assert rms(gain_diff) < 2.0 * noise_floor, "top-k order changes the output rows coherently"
    assert rms(gain_first) < 2.0 * rms(gain_ideal)
    assert rms(gain_last) < 2.0 * rms(gain_ideal)

    # the gradient accuracy must be order-independent too
    def rel_l2(a, r):
        return ((a.reshape(r.shape).double() - r).norm() / r.norm()).item()

    for name, i, ref in (("dq", 0, dq_ref), ("dk", 1, dk_ref), ("dv", 2, dv_ref), ("dqv", 3, dqv_ref)):
        e_first, e_last = rel_l2(grads_first[i], ref), rel_l2(grads_last[i], ref)
        print(f"{name} rel-L2 vs fp64: first {e_first:.4%} last {e_last:.4%}")
        assert abs(e_first - e_last) < 0.05 * max(e_first, e_last), f"{name}: gradient accuracy depends on the top-k order"


@pytest.mark.parametrize(
    "case",
    [
        # (description, kwargs for _mla_1cta_route, expected 1CTA)
        # split: the split heuristic (num_splits < 1). Dense goes 1CTA provisionally at any shape;
        # the interface sizes the split for the 1CTA kernel and, if it picks one split, asks
        # again unsplit (the rows below without split)
        ("dense decode 64 heads, split heuristic", dict(topk=False, h=64, s_q=1, split=True, ctas=16), True),
        ("dense 64 heads x 2 tokens, split heuristic", dict(topk=False, h=64, s_q=2, split=True, ctas=16), True),
        ("dense decode 128 heads, split heuristic", dict(topk=False, h=128, s_q=1, split=True, ctas=16), True),
        ("dense prefill, split heuristic", dict(topk=False, h=64, s_q=4096, split=True), True),
        # unsplit: decode shapes (seqlen_q x heads per KV head <= 64) go 1CTA only once the 2CTA
        # kernel (2 CTAs per batch x KV head) exceeds one wave of num_sms
        ("dense decode 64 heads, 2CTA one wave", dict(topk=False, h=64, s_q=1, ctas=144), False),
        ("dense decode 64 heads, 2CTA two waves", dict(topk=False, h=64, s_q=1, ctas=160), True),
        ("dense decode 16 heads x 4 tokens (64 rows), two waves", dict(topk=False, h=16, s_q=4, ctas=160), True),
        ("dense decode 16 heads, one wave", dict(topk=False, h=16, s_q=1, ctas=128), False),
        ("dense 64 heads x 2 tokens (128 rows), two waves", dict(topk=False, h=64, s_q=2, ctas=160), False),
        ("dense decode 128 heads, two waves", dict(topk=False, h=128, s_q=1, ctas=160), False),
        ("dense prefill, two waves", dict(topk=False, h=64, s_q=4096, ctas=160), False),
        ("dense varlen without a host max_seqlen_q", dict(topk=False, h=64, s_q=None, ctas=160), False),
        ("dense prefill fp8 (2CTA has no fp8)", dict(topk=False, h=64, s_q=4096, needs=True), True),
        # sparse (never split): 2 CTAs per token on 2CTA; 1CTA only past one 2CTA wave
        ("sparse 64 heads", dict(topk=True, h=64, s_q=4096), True),
        ("sparse 24 heads decode", dict(topk=True, h=24, s_q=1), True),
        ("sparse decode, 2CTA one wave", dict(topk=True, h=64, s_q=1, ctas=152), False),
        ("sparse decode, 2CTA past one wave", dict(topk=True, h=64, s_q=1, ctas=160), True),
        ("sparse decode 16 heads, one wave", dict(topk=True, h=16, s_q=1, ctas=64), False),
        ("sparse fp8 below one wave (2CTA has no fp8)", dict(topk=True, h=64, s_q=1, ctas=64, needs=True), True),
        ("sparse 128 heads (unsupported on 1CTA)", dict(topk=True, h=128, s_q=1), False),
        ("sparse training, load-P (unsupported)", dict(topk=True, h=64, s_q=4096, grad=True), False),
        ("sparse training, recompute-P", dict(topk=True, h=64, s_q=4096, grad=True, rp=True), True),
    ],
    ids=lambda c: c[0].replace(" ", "_"),
)
def test_flash_attn_mla_dispatch_heuristic(case, monkeypatch):
    """The 1CTA / 2CTA MLA dispatch with FLASH_ATTENTION_MLA_1CTA unset: sparse -> 1CTA up to
    64 heads once 2CTA exceeds one wave (2 CTAs per token); dense with the split heuristic ->
    1CTA (provisionally), unsplit -> 1CTA on decode shapes (seqlen_q x heads per KV head <= 64)
    once 2CTA exceeds one wave, 2CTA otherwise; fp8 / descales / explicit split-KV -> 1CTA.
    The variable overrides it."""
    from flash_attn.cute.interface import _mla_1cta_route
    _, kw, expected = case
    route = lambda: _mla_1cta_route(  # noqa: E731
        kw["topk"], kw["h"], kw.get("grad", False), kw.get("rp", False),
        seqlen_q_hint=kw["s_q"], needs_1cta=kw.get("needs", False),
        split_kv=kw.get("split", False), ctas_2cta=kw.get("ctas"), num_sms=152,
    )
    monkeypatch.delenv("FLASH_ATTENTION_MLA_1CTA", raising=False)
    assert route() == expected
    supported = not (kw["topk"] and (kw["h"] > 64 or (kw.get("grad") and not kw.get("rp"))))
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "1")
    assert route() == supported
    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "0")
    assert not route()


@pytest.mark.parametrize(
    "shape",
    ["dense_decode", "dense_decode_128h", "dense_decode_128h_few_splits", "dense_decode_unsplit",
     "dense_prefill", "dense_prefill_auto", "sparse"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_mla_dispatch_heuristic_end_to_end(shape, monkeypatch):
    """With FLASH_ATTENTION_MLA_1CTA unset the call runs the kernel the heuristic picks (the
    first compile-key element is the 1CTA route) and matches the reference."""
    if not IS_SM100:
        pytest.skip()
    import flash_attn.cute.interface as fa_interface
    monkeypatch.delenv("FLASH_ATTENTION_MLA_1CTA", raising=False)
    b = 32 if shape == "dense_decode_128h_few_splits" else 2
    h = 128 if shape.startswith("dense_decode_128h") else 64
    s_q, s_k = {
        "dense_decode": (1, 2048), "dense_decode_128h": (1, 2048),
        "dense_decode_128h_few_splits": (1, 1024), "dense_decode_unsplit": (1, 2048),
        "dense_prefill": (256, 512), "dense_prefill_auto": (256, 512), "sparse": (256, 512),
    }[shape]
    kw, (q_r, k_r, v_r, qv_r) = _mla_inputs(b, s_q, s_k, h, has_qk=True)
    extra = {}
    # the split heuristic (num_splits=0), sized for 1CTA: decode splits -> 1CTA, at 128 rows
    # only with >= 5 splits (batch 2: 16; batch 32: 4 -> 2CTA); prefill gets one split ->
    # decided again unsplit -> 2CTA. Unsplit decode at batch 2 (far below one 2CTA wave) -> 2CTA
    fwd_kw = dict(num_splits=0) if shape not in ("dense_decode_unsplit", "dense_prefill", "sparse") else {}
    if shape == "sparse":
        extra = dict(gather_kv_indices=rect_topk_indices(b, s_q, s_k, 256, True, "cuda"), causal=True)
    real_cache = fa_interface._flash_attn_fwd.compile_cache

    class Spy:
        keys = []

        def __contains__(self, key):
            self.keys.append(key)
            return key in real_cache

        def __getitem__(self, key):
            return real_cache[key]

        def __setitem__(self, key, value):
            real_cache[key] = value

    spy = Spy()
    monkeypatch.setattr(fa_interface._flash_attn_fwd, "compile_cache", spy)
    out, _ = flash_attn_func(**kw, **extra, **fwd_kw)
    monkeypatch.setattr(fa_interface._flash_attn_fwd, "compile_cache", real_cache)
    expect_1cta = shape in ("dense_decode", "dense_decode_128h", "sparse")
    assert spy.keys and all(key[0] == expect_1cta for key in spy.keys), shape
    if is_fake_mode():
        return
    out_ref, _ = attention_ref(q_r, k_r, v_r, qv=qv_r, **extra)
    out_pt, _ = attention_ref(q_r, k_r, v_r, qv=qv_r, upcast=False, reorder_ops=True, **extra)
    valid = torch.isfinite(out_ref).all(-1)
    err = (out.float() - out_ref.float()).abs()[valid].max().item()
    err_pt = (out_pt.float() - out_ref.float()).abs()[valid].max().item()
    assert err <= 2 * err_pt + 1e-3, (err, err_pt)
