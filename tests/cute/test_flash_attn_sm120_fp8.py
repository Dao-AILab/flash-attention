# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SM12x FP8 (e4m3 / e5m2) forward tests.

Every check compares O and LSE with attention_ref on the FP32 dequantized inputs. The O bound
is the upstream FP8 rule (check_tensor_vs_ref): twice the error of the same reference with P
rounded to the input FP8 type, plus twice the BF16 rounding of the output, plus 1e-3.
"""

import itertools
import os
import random

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.testing import (
    attention_ref,
    check_tensor_vs_ref,
    is_fake_mode,
    maybe_fake_tensor_mode,
)

from mask_mod_definitions import (
    cute_block_diagonal_mask,
    cute_causal_mask,
    flex_block_diagonal_mask,
    get_cute_sliding_window_mask,
    get_flex_sliding_window_mask,
)
from score_mod_definitions import alibi_eager, score_mod_alibi

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
    pytest.skip(
        "SM12x FlashAttention-4 test requires CUDA SM 12.x.",
        allow_module_level=True,
    )

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1

# (head_dim, head_dim_v) the SM12x FP8 forward runs.
SHAPES = [(64, 64), (96, 96), (128, 128), (192, 192), (192, 128), (256, 256)]
FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
# Byte pattern that is NaN in both FP8 formats; fills cache rows a call must not read.
NAN_BYTE = 0x7F


def _shape_id(shape):
    return f"hd{shape[0]}x{shape[1]}"


def _cu(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)


def _sparse(block_mask, block):
    _, _, kv_cnt, kv_idx, full_cnt, full_idx, *_ = block_mask.as_tuple()
    return BlockSparseTensorsTorch(
        mask_block_cnt=kv_cnt, mask_block_idx=kv_idx, full_block_cnt=full_cnt,
        full_block_idx=full_idx, block_size=block,
    )


def _reference(q, k, v, *, causal=False, window=None, score_mod=None, mask=None, sink=None,
               softcap=0.0, p_dtype=None, batch_idx=0):
    """attention_ref of one sequence: q (sq, hq, d), k (sk, hk, d), v (sk, hk, dv), all FP32.
    score_mod (additive) and mask enter as an attention bias. Returns (O, O with P rounded to
    p_dtype, LSE (hq, sq))."""
    sq, hq, _ = q.shape
    sk = k.shape[0]
    bias = None
    if score_mod is not None or mask is not None:
        b = torch.tensor(batch_idx, device=q.device)
        h = torch.arange(hq, device=q.device)[:, None, None]
        qi = torch.arange(sq, device=q.device)[None, :, None]
        ki = torch.arange(sk, device=q.device)[None, None, :]
        bias = torch.zeros(hq, sq, sk, device=q.device)
        if score_mod is not None:
            bias = score_mod(bias, b, h, qi, ki)
        if mask is not None:
            bias = bias.masked_fill(~mask(b, h, qi, ki).expand_as(bias), float("-inf"))
        bias = bias[None]
    kw = dict(attn_bias=bias, causal=causal, window_size=window or (None, None),
              learnable_sink=sink, softcap=softcap, return_lse=True)
    out, _, lse = attention_ref(q[None], k[None], v[None], **kw)
    out_pt, _, _ = attention_ref(q[None], k[None], v[None], intermediate_dtype=p_dtype, **kw)
    return out[0], out_pt[0], lse[0]


def _check(out, lse, ref, label):
    """out (sq, hq, dv) BF16, lse (hq, sq) FP32 against _reference's (O, O_pt, LSE)."""
    out_ref, out_pt, lse_ref = ref
    assert torch.isfinite(out).all(), f"{label}: non-finite O"
    check_tensor_vs_ref(
        f"{label} O", out.float(), out_ref, out_pt,
        atol=2 * (out_ref.bfloat16().float() - out_ref).abs().max().item() + 1e-3,
    )
    finite = torch.isfinite(lse_ref)
    assert torch.isfinite(lse[finite]).all(), f"{label}: non-finite LSE where the reference is finite"
    # Rows without visible keys: exactly -inf (NaN and +inf must not pass).
    assert torch.isneginf(lse[~finite]).all(), f"{label}: LSE not -inf where the reference is -inf"
    if finite.any():
        lse_err = (lse[finite] - lse_ref[finite]).abs().max().item()
        assert lse_err < 2e-3, f"{label}: LSE error {lse_err:.3e}"


def _dequant(t, scale, rep=1):
    """FP32 values of an FP8 tensor (s, h, d) under a per-head descale (h // rep,)."""
    t = t.float()
    return t if scale is None else t * scale.repeat_interleave(rep)[None, :, None]


def _descales(batch, hk, enabled, g):
    if not enabled:
        return {"q": None, "k": None, "v": None}
    return {n: torch.rand(batch, hk, device="cuda", generator=g) + 0.5 for n in "qkv"}


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
@pytest.mark.parametrize(
    "fmt,mask,heads,lengths,descale",
    [
        ("e4m3", "causal", (16, 4), (2, 1024, 1024), True),
        ("e4m3", "local", (16, 4), (3, 333, 777), True),
        ("e4m3", "none", (16, 1), (1, 1, 2048), False),
        ("e5m2", "none", (8, 8), (2, 700, 700), False),
        ("e5m2", "causal", (16, 4), (1, 4096, 4096), False),
    ],
    ids=["e4m3-causal-gqa", "e4m3-local-irregular", "e4m3-mqa-sq1", "e5m2-full", "e5m2-causal-long"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_dense(shape, fmt, mask, heads, lengths, descale):
    """Dense batches: every FP8 shape, both formats, causal / local / full masks, descales,
    MHA / GQA / MQA, irregular and one-query lengths. The long causal case takes the 128-row
    hd 256 tile."""
    hd, hdv = shape
    dtype = FP8[fmt]
    (hq, hk), (b, sq, sk) = heads, lengths
    g = torch.Generator(device="cuda").manual_seed(hd + hdv + sq)
    q = torch.randn(b, sq, hq, hd, device="cuda", generator=g).to(dtype)
    k = torch.randn(b, sk, hk, hd, device="cuda", generator=g).to(dtype)
    v = torch.randn(b, sk, hk, hdv, device="cuda", generator=g).to(dtype)
    d = _descales(b, hk, descale, g)
    window = (300, 50) if mask == "local" else None
    out, lse = _flash_attn_fwd(
        q, k, v, causal=mask == "causal",
        window_size_left=None if window is None else window[0],
        window_size_right=None if window is None else window[1],
        return_lse=True, q_descale=d["q"], k_descale=d["k"], v_descale=d["v"],
    )[:2]
    if is_fake_mode():
        return
    for i in range(b):
        sc = {n: None if d[n] is None else d[n][i] for n in "qkv"}
        ref = _reference(
            _dequant(q[i], sc["q"], hq // hk), _dequant(k[i], sc["k"]), _dequant(v[i], sc["v"]),
            causal=mask == "causal", window=window, p_dtype=dtype,
        )
        _check(out[i], lse[i], ref, f"batch {i}")


def _paged_kv(k, v, lengths, page_size, rng):
    """Scatter per-sequence rows of k / v (b, s, h, d) into shuffled page pools under one page
    table. Rows past each length and the spare pages hold NaN bytes."""
    npages = [-(-n // page_size) for n in lengths]
    total = sum(npages) + 3
    order = list(range(total))
    rng.shuffle(order)
    rows_of = []  # (page, sequence, first row, row count)
    table = [[0] * max(max(npages), 1) for _ in lengths]
    used = 0
    for i, n in enumerate(lengths):
        for p in range(npages[i]):
            table[i][p] = order[used]
            rows_of.append((order[used], i, p * page_size, min(page_size, n - p * page_size)))
            used += 1
    pools = []
    for x in (k, v):
        h, d = x.shape[2], x.shape[3]
        pool = torch.full((total, page_size, h, d * x.element_size()), NAN_BYTE,
                          dtype=torch.uint8, device=x.device)
        for page, i, start, rows in rows_of:
            pool[page, :rows] = x[i, start : start + rows].view(torch.uint8)
        pools.append(pool.view(x.dtype))
    return pools[0], pools[1], torch.tensor(table, dtype=torch.int32, device=k.device)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
@pytest.mark.parametrize("page_size", [64, 256, None], ids=["page64", "page256", "dense"])
@pytest.mark.parametrize("num_splits", [1, 3, 0], ids=["split1", "split3", "split-auto"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_varlen_paged_splitkv(shape, page_size, num_splits):
    """Packed varlen Q against a paged or dense cache with seqused_k: one-query decode at
    GQA 12, two-query decode (a speculative token), and a short prefill; SplitKV fixed / auto
    (per-batch); descales; NaN bytes in every cache row past seqused_k; an empty sequence,
    whose rows give O = 0 and LSE = -inf."""
    hd, hdv = shape
    dtype = torch.float8_e4m3fn
    rng = random.Random(hd + hdv + (page_size or 0) + num_splits)
    g = torch.Generator(device="cuda").manual_seed(hd + hdv + num_splits)
    cases = [("decode", 1, (24, 2), (1000, 0, 37, 4097)),
             ("mtp", 2, (24, 2), (513, 4096, 7)),
             ("prefill", 200, (8, 8), (900, 200, 1500))]
    results = []
    for kind, sq, (hq, hk), lengths in cases:
        b, smax = len(lengths), max(lengths)
        q = torch.randn(b * sq, hq, hd, device="cuda", generator=g).to(dtype)
        k = torch.randn(b, smax, hk, hd, device="cuda", generator=g).to(dtype)
        v = torch.randn(b, smax, hk, hdv, device="cuda", generator=g).to(dtype)
        if page_size is None:
            k_cache, v_cache, table = k.clone(), v.clone(), None
            for i, n in enumerate(lengths):
                k_cache[i, n:].view(torch.uint8).fill_(NAN_BYTE)
                v_cache[i, n:].view(torch.uint8).fill_(NAN_BYTE)
        else:
            k_cache, v_cache, table = _paged_kv(k, v, lengths, page_size, rng)
        d = _descales(b, hk, True, g)
        cu_q = torch.arange(0, (b + 1) * sq, sq, device="cuda", dtype=torch.int32)
        seqused_k = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        out, lse = _flash_attn_fwd(
            q, k_cache, v_cache, cu_seqlens_q=cu_q, max_seqlen_q=sq, seqused_k=seqused_k,
            page_table=table, causal=True, num_splits=num_splits, return_lse=True,
            q_descale=d["q"], k_descale=d["k"], v_descale=d["v"],
        )[:2]
        results.append((kind, sq, hq, hk, lengths, q, k, v, d, out, lse))
    if is_fake_mode():
        return
    for kind, sq, hq, hk, lengths, q, k, v, d, out, lse in results:
        for i, n in enumerate(lengths):
            if n == 0:  # no keys: the kernel writes zeros and -inf
                assert (out[i * sq : (i + 1) * sq] == 0).all(), f"{kind} batch {i}: O not zero"
                assert torch.isneginf(lse[:, i * sq : (i + 1) * sq]).all(), f"{kind} batch {i}"
                continue
            sc = {name: d[name][i] for name in "qkv"}
            ref = _reference(
                _dequant(q[i * sq : (i + 1) * sq], sc["q"], hq // hk),
                _dequant(k[i, :n], sc["k"]), _dequant(v[i, :n], sc["v"]),
                causal=True, p_dtype=dtype,
            )
            _check(out[i * sq : (i + 1) * sq], lse[:, i * sq : (i + 1) * sq], ref,
                   f"{kind} batch {i}")


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_block_sparse_rejects_undividable_hd256():
    """The fp8 PV MMA steps 32 keys, so the hd 256 fp8 tiles are N = 64, or N = 32 at M = 128:
    none divides a (64, 96) sparse block, which the host must reject."""
    dtype, block = torch.float8_e4m3fn, (64, 96)
    q = torch.zeros(2, 1000, 16, 256, device="cuda").to(dtype)
    kv = torch.zeros(2, 1300, 4, 256, device="cuda").to(dtype)
    bm = create_block_mask(flex_block_diagonal_mask, 2, 1, 1000, 1300, device="cuda", BLOCK_SIZE=block)
    with pytest.raises(ValueError, match="no forward tile divides"):
        _flash_attn_fwd(q, kv, kv, mask_mod=cute_block_diagonal_mask,
                        block_sparse_tensors=_sparse(bm, block), pack_gqa=False)


@pytest.mark.parametrize(
    "shape, block",
    [
        pytest.param(s, blk, id=f"{_shape_id(s)}-blk{blk[0]}x{blk[1]}")
        for s in ((128, 128), (192, 128), (256, 256))
        for blk in ((128, 128), (64, 96))
        if not (s[0] == 256 and blk[1] % 64)
    ],
)
@pytest.mark.parametrize("num_splits", [1, 3], ids=["split1", "split3"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_block_sparse(shape, block, num_splits):
    """Block-sparse lists from flex block masks with the matching mask_mod, descales and
    GQA; the tile is fitted to the sparse block over the tiles FP8 runs."""
    hd, hdv = shape
    dtype = torch.float8_e4m3fn
    b, sq, sk, hq, hk = 2, 1000, 1300, 16, 4
    g = torch.Generator(device="cuda").manual_seed(hd + block[0] + block[1] + num_splits)
    q = torch.randn(b, sq, hq, hd, device="cuda", generator=g).to(dtype)
    k = torch.randn(b, sk, hk, hd, device="cuda", generator=g).to(dtype)
    v = torch.randn(b, sk, hk, hdv, device="cuda", generator=g).to(dtype)
    d = _descales(b, hk, True, g)
    results = []
    for name, cute_mask, flex_mask in (
        ("sliding", get_cute_sliding_window_mask(256, 64, sk - sq), get_flex_sliding_window_mask(256, 64, sk - sq)),
        ("block_diag", cute_block_diagonal_mask, flex_block_diagonal_mask),
    ):
        bm = create_block_mask(flex_mask, b, 1, sq, sk, device="cuda", BLOCK_SIZE=block)
        out, lse = _flash_attn_fwd(
            q, k, v, mask_mod=cute_mask, block_sparse_tensors=_sparse(bm, block), num_splits=num_splits,
            return_lse=True, pack_gqa=False, q_descale=d["q"], k_descale=d["k"], v_descale=d["v"],
        )[:2]
        results.append((name, flex_mask, out, lse))
    if is_fake_mode():
        return
    for name, flex_mask, out, lse in results:
        for i in range(b):
            sc = {n: d[n][i] for n in "qkv"}
            ref = _reference(
                _dequant(q[i], sc["q"], hq // hk), _dequant(k[i], sc["k"]), _dequant(v[i], sc["v"]),
                mask=lambda bb, h, qi, ki: flex_mask(bb, torch.zeros_like(h), qi, ki), p_dtype=dtype,
                batch_idx=i,
            )
            _check(out[i], lse[i], ref, f"{name} batch {i}")


@pytest.mark.parametrize("shape, block", [((192, 192), (128, 32)), ((256, 256), (64, 32))],
                         ids=["hd192-blk128x32", "hd256-blk64x32"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_block_sparse_paged(shape, block):
    """Paged block-sparse calls: the tile fitted to the sparse block runs with the paged
    layout (hd 192 at a (128, 32) block takes (64, 32), not the 64-column fallback)."""
    hd, hdv = shape
    dtype = torch.float8_e4m3fn
    b, sq, sk, hq, hk = 2, 128, 1280, 8, 2
    g = torch.Generator(device="cuda").manual_seed(hd + block[0])
    q = torch.randn(b, sq, hq, hd, device="cuda", generator=g).to(dtype)
    k = torch.randn(b, sk, hk, hd, device="cuda", generator=g).to(dtype)
    v = torch.randn(b, sk, hk, hdv, device="cuda", generator=g).to(dtype)
    k_cache, v_cache, table = _paged_kv(k, v, [sk] * b, 64, random.Random(hd))
    flex_mask = get_flex_sliding_window_mask(256, 64, sk - sq)
    bm = create_block_mask(flex_mask, b, 1, sq, sk, device="cuda", BLOCK_SIZE=block)
    out, lse = _flash_attn_fwd(
        q, k_cache, v_cache, page_table=table, mask_mod=get_cute_sliding_window_mask(256, 64, sk - sq),
        block_sparse_tensors=_sparse(bm, block), return_lse=True,
    )[:2]
    if is_fake_mode():
        return
    for i in range(b):
        ref = _reference(
            q[i].float(), k[i].float(), v[i].float(),
            mask=lambda bb, h, qi, ki: flex_mask(bb, torch.zeros_like(h), qi, ki), p_dtype=dtype,
            batch_idx=i,
        )
        _check(out[i], lse[i], ref, f"batch {i}")


@pytest.mark.parametrize("shape, block", [((192, 192), (64, 32)), ((256, 256), (16, 32))],
                         ids=["hd192-blk64x32", "hd256-blk16x32"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_block_sparse_varlen_paged(shape, block):
    """Varlen Q with block-sparse lists over a paged cache of ragged lengths (last pages part
    full, NaN past each length): the Q tile is the sparse block's rows, run with the paged
    layout's tile. Every KV block a Q block sees is listed as a partial block."""
    hd, hdv = shape
    dtype = torch.float8_e4m3fn
    lq, lk, hq, hk = [1, 37, 130], [700, 69, 1301], 8, 2
    bm, bn = block
    g = torch.Generator(device="cuda").manual_seed(hd + bm)
    qs = [torch.randn(n, hq, hd, device="cuda", generator=g).to(dtype) for n in lq]
    k = torch.randn(len(lk), max(lk), hk, hd, device="cuda", generator=g).to(dtype)
    v = torch.randn(len(lk), max(lk), hk, hdv, device="cuda", generator=g).to(dtype)
    k_cache, v_cache, table = _paged_kv(k, v, lk, 64, random.Random(hd))
    cnt, idx = [], []
    for a, c in zip(lq, lk):
        n = -(-c // bn)
        for m in range(-(-a // bm)):
            seen = min(n, (min((m + 1) * bm, a) - 1 + c - a) // bn + 1)
            cnt.append(seen)
            idx += list(range(seen)) + [0] * (n - seen)
    row = lambda x: torch.tensor([x], device="cuda", dtype=torch.int32)
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=row(cnt), mask_block_idx=row(idx), full_block_cnt=row([0] * len(cnt)),
        full_block_idx=row(idx), block_size=block, cu_total_m_blocks=_cu([-(-a // bm) for a in lq]),
        cu_block_idx_offsets=_cu([-(-a // bm) * -(-c // bn) for a, c in zip(lq, lk)]),
    )
    out, lse = _flash_attn_fwd(
        torch.cat(qs), k_cache, v_cache, page_table=table, cu_seqlens_q=_cu(lq),
        seqused_k=torch.tensor(lk, device="cuda", dtype=torch.int32), max_seqlen_q=max(lq),
        max_seqlen_k=max(lk), mask_mod=cute_causal_mask, block_sparse_tensors=sparse, return_lse=True,
    )[:2]
    if is_fake_mode():
        return
    for i, (a, c) in enumerate(zip(lq, lk)):
        s = sum(lq[:i])
        ref = _reference(qs[i].float(), k[i, :c].float(), v[i, :c].float(), causal=True, p_dtype=dtype)
        _check(out[s : s + a], lse[:, s : s + a], ref, f"sequence {i}")


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_softcap(shape):
    """softcap on scores large enough to saturate it: a causal prefill with descales and a
    one-query decode, GQA."""
    hd, hdv = shape
    dtype = torch.float8_e4m3fn
    hq, hk, softcap = 16, 4, 15.0
    g = torch.Generator(device="cuda").manual_seed(hd + hdv)
    results = []
    for (b, sq, sk), causal, descale in (((2, 300, 700), True, True), ((2, 1, 2048), False, False)):
        q = (torch.randn(b, sq, hq, hd, device="cuda", generator=g) * 3).to(dtype)
        k = (torch.randn(b, sk, hk, hd, device="cuda", generator=g) * 3).to(dtype)
        v = torch.randn(b, sk, hk, hdv, device="cuda", generator=g).to(dtype)
        d = _descales(b, hk, descale, g)
        out, lse = _flash_attn_fwd(
            q, k, v, causal=causal, softcap=softcap, return_lse=True,
            q_descale=d["q"], k_descale=d["k"], v_descale=d["v"],
        )[:2]
        results.append((b, sq, causal, q, k, v, d, out, lse))
    if is_fake_mode():
        return
    for b, sq, causal, q, k, v, d, out, lse in results:
        for i in range(b):
            sc = {n: None if d[n] is None else d[n][i] for n in "qkv"}
            ref = _reference(
                _dequant(q[i], sc["q"], hq // hk), _dequant(k[i], sc["k"]), _dequant(v[i], sc["v"]),
                causal=causal, softcap=softcap, p_dtype=dtype,
            )
            _check(out[i], lse[i], ref, f"sq {sq} batch {i}")


@pytest.mark.parametrize("shape", [(64, 64), (192, 128), (256, 256)], ids=_shape_id)
@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_score_mod_sink_varlen(shape, fmt):
    """score_mod (ALiBi), a learnable sink and varlen cu_seqlens_q / cu_seqlens_k with a
    one-query sequence, causal, GQA."""
    hd, hdv = shape
    dtype = FP8[fmt]
    hq, hk = 16, 4
    lq, lk = [300, 1, 777, 130], [1100, 64, 777, 2000]
    g = torch.Generator(device="cuda").manual_seed(hd + hdv + len(fmt))
    qs = [torch.randn(n, hq, hd, device="cuda", generator=g).to(dtype) for n in lq]
    ks = [torch.randn(n, hk, hd, device="cuda", generator=g).to(dtype) for n in lk]
    vs = [torch.randn(n, hk, hdv, device="cuda", generator=g).to(dtype) for n in lk]
    sink = (torch.randn(hq, device="cuda", generator=g) * 2).to(torch.bfloat16)
    out, lse = _flash_attn_fwd(
        torch.cat(qs), torch.cat(ks), torch.cat(vs), cu_seqlens_q=_cu(lq), cu_seqlens_k=_cu(lk),
        max_seqlen_q=max(lq), max_seqlen_k=max(lk), causal=True, learnable_sink=sink,
        score_mod=score_mod_alibi, return_lse=True,
    )[:2]
    if is_fake_mode():
        return
    starts = [0, *itertools.accumulate(lq)]
    for i in range(len(lq)):
        ref = _reference(
            qs[i].float(), ks[i].float(), vs[i].float(), causal=True, sink=sink, p_dtype=dtype,
            score_mod=alibi_eager, batch_idx=i,
        )
        rows = slice(starts[i], starts[i + 1])
        _check(out[rows], lse[:, rows], ref, f"batch {i}")


def test_sm120_fp8_check_rejects_nonfinite_lse():
    """The LSE check accepts only -inf where the reference has no visible keys."""
    out_ref = torch.zeros(2, 1, 4)
    ref = (out_ref, out_ref, torch.tensor([[0.5, float("-inf")]]))
    _check(out_ref, torch.tensor([[0.5, float("-inf")]]), ref, "exact")
    for bad in (float("nan"), float("inf"), 0.0):
        with pytest.raises(AssertionError):
            _check(out_ref, torch.tensor([[0.5, bad]]), ref, "bad")


@pytest.mark.parametrize("layout", ["fixed", "varlen"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fp8_block_sparse_decode_keeps_sparse_tile(layout):
    """One-query block-sparse hd 256 calls (pack_gqa off): the decode tile narrowing must not
    replace the tile fitted to the sparse block (fixed length: a (64, 128) block; varlen: Q
    tiles of the sparse block itself, 64)."""
    dtype = torch.float8_e4m3fn
    hq, hk, hd = 8, 2, 256
    g = torch.Generator(device="cuda").manual_seed(91)
    if layout == "fixed":
        b, sk = 2, 700
        q = torch.randn(b, 1, hq, hd, device="cuda", generator=g).to(dtype)
        k = torch.randn(b, sk, hk, hd, device="cuda", generator=g).to(dtype)
        v = torch.randn(b, sk, hk, hd, device="cuda", generator=g).to(dtype)
        flex_causal = lambda bb, h, qi, ki: ki <= qi + (sk - 1)
        bm = create_block_mask(flex_causal, b, 1, 1, sk, device="cuda", BLOCK_SIZE=(64, 128))
        out, lse = _flash_attn_fwd(
            q, k, v, mask_mod=cute_causal_mask, block_sparse_tensors=_sparse(bm, (64, 128)),
            return_lse=True, pack_gqa=False,
        )[:2]
        seqs = [(q[i], k[i], v[i], out[i], lse[i]) for i in range(b)]
    else:
        lq, lk = [1, 1, 1], [700, 64, 1300]
        cu_q, cu_k = _cu(lq), _cu(lk)
        cu_m = torch.tensor([0, *itertools.accumulate(-(-n // 64) for n in lq)], device="cuda", dtype=torch.int32)
        cu_idx = torch.tensor(
            [0, *itertools.accumulate(-(-a // 64) * -(-c // 64) for a, c in zip(lq, lk))],
            device="cuda", dtype=torch.int32,
        )
        # One query at the end of each sequence sees every key: all KV blocks are full blocks,
        # listed in order per sequence (the layout compute_block_sparsity produces).
        nblk = [-(-n // 64) for n in lk]
        kc = torch.zeros(1, len(lq), device="cuda", dtype=torch.int32)
        ki_ = torch.zeros(1, sum(nblk), device="cuda", dtype=torch.int32)
        fc = torch.tensor([nblk], device="cuda", dtype=torch.int32)
        fi = torch.tensor([[j for n in nblk for j in range(n)]], device="cuda", dtype=torch.int32)
        sparse = BlockSparseTensorsTorch(
            mask_block_cnt=kc, mask_block_idx=ki_, full_block_cnt=fc, full_block_idx=fi,
            block_size=(64, 64), cu_total_m_blocks=cu_m, cu_block_idx_offsets=cu_idx,
        )
        qs = [torch.randn(n, hq, hd, device="cuda", generator=g).to(dtype) for n in lq]
        ks = [torch.randn(n, hk, hd, device="cuda", generator=g).to(dtype) for n in lk]
        vs = [torch.randn(n, hk, hd, device="cuda", generator=g).to(dtype) for n in lk]
        out, lse = _flash_attn_fwd(
            torch.cat(qs), torch.cat(ks), torch.cat(vs), cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=max(lk), mask_mod=cute_causal_mask,
            block_sparse_tensors=sparse, return_lse=True, pack_gqa=False,
        )[:2]
        seqs = [(qs[i], ks[i], vs[i], out[i : i + 1], lse[:, i : i + 1]) for i in range(len(lq))]
    if is_fake_mode():
        return
    for i, (qi, ki, vi, oi, li) in enumerate(seqs):
        ref = _reference(qi.float(), ki.float(), vi.float(), causal=True, p_dtype=dtype)
        _check(oi, li, ref, f"{layout} sequence {i}")
