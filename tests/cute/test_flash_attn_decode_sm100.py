"""SM100 SwapAB decode: end-to-end checks of the shared forward path against an fp32 oracle.

Two-pass capable: under FLASH_ATTENTION_FAKE_TENSOR=1 every test issues all of its kernel
variants and returns before the first data-dependent check.
"""

import os
from functools import partial

import cutlass
from cutlass import cute
import pytest
import torch

from decode_test_utils import (
    SM100,
    assert_decode_close,
    decode_inputs,
    decode_reference,
    paged_inputs,
    poison_unused_kv,
    record_fwd_configs,
    require_swapab,
)
from flash_attn.cute import interface
from flash_attn.cute.config import select_fwd_config
from flash_attn.cute.interface import _flash_attn_fwd, flash_attn_func
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1

pytestmark = pytest.mark.skipif(not SM100, reason="SM100 SwapAB integration")


@pytest.fixture(autouse=True)
def force_swapab(monkeypatch):
    """Every forward requests SwapAB and fails if the request falls back to the standard kernel;
    tests that set another backend opt out per call."""
    require_swapab(monkeypatch)


@cute.jit
def keep_every_third(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    return kv_idx % 3 == 0


@cute.jit
def none_kept(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    return kv_idx < 0


@cute.jit
def batch_bias(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    bi = cute.make_rmem_tensor(1, cutlass.Int32)
    bi.store(b)
    bias = cute.make_rmem_tensor(1, cutlass.Float32)
    bias[0] = aux_tensors[0][bi[0]]
    return score + bias.load()


# Modifiers here do not call other @cute.jit modifiers: tracing a nested callee replaces its
# __code__, so utils.hash_callable keys it differently in later tests of the same process and
# the two-pass execution pass recompiles.
@cute.jit
def token_head_bias(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    """Score bias at (batch, query head, logical KV token)."""
    bi = cute.make_rmem_tensor(1, cutlass.Int32)
    hi = cute.make_rmem_tensor(1, cutlass.Int32)
    ki = cute.make_rmem_tensor(1, cutlass.Int32)
    bi.store(b)
    hi.store(h)
    ki.store(kv_idx)
    bias = cute.make_rmem_tensor(1, cutlass.Float32)
    bias[0] = aux_tensors[0][bi[0], hi[0], ki[0]]
    return score + bias.load()


def check(
    q,
    k,
    v,
    *,
    ns=3,
    used=None,
    page_table=None,
    window=None,
    softcap=None,
    softmax_scale=None,
    mask=False,
    bias=None,
    poison=False,
    **kwargs,
):
    """Run the forward once and, outside fake mode, compare O/LSE with the oracle."""
    if mask:
        kwargs["mask_mod"] = keep_every_third
    if bias is not None:
        kwargs.update(score_mod=batch_bias, aux_tensors=[bias])
    if poison and not is_fake_mode():
        poison_unused_kv(k, v, used, page_table)
    o, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        num_splits=ns,
        seqused_k=used,
        page_table=page_table,
        window_size_left=window,
        softcap=softcap,
        softmax_scale=softmax_scale,
        return_lse=True,
        **kwargs,
    )
    if is_fake_mode():
        return o, lse
    seqlen_k = k.shape[1] if page_table is None else page_table.shape[1] * k.shape[1]
    keep = None
    if mask:
        keep = torch.arange(seqlen_k, device="cuda")[None, :] % 3 == 0
        keep = keep.expand(q.shape[0], -1)
    ref = decode_reference(
        q,
        k,
        v,
        page_table=page_table,
        seqused=used,
        keep=keep,
        bias=None if bias is None else bias[:, None, None].expand(-1, q.shape[2], seqlen_k),
        causal=kwargs.get("causal", False),
        window=window,
        softcap=softcap,
        softmax_scale=softmax_scale,
    )
    assert_decode_close(o, lse, ref)
    return o, lse


ALL_DIMS = [
    (64, 64),
    (96, 96),
    (128, 128),
    (64, 128),
    (128, 64),
    (96, 128),
    (128, 96),
    (64, 96),
    (96, 64),
    (256, 256),
]
# Every (dq, dv) pair is a distinct kernel (smem/TMEM layout, TMA boxes); dtype only changes the
# convert/exp paths and ns > 1 only the partial-write/combine path, so those get a subset.
DENSE_CASES = (
    [(dims, torch.bfloat16, 1) for dims in ALL_DIMS]
    + [(dims, torch.float16, 1) for dims in [(64, 64), (128, 128), (256, 256)]]
    + [(dims, torch.float16, 3) for dims in [(128, 128), (96, 128), (64, 96), (256, 256)]]
)


@pytest.mark.parametrize(
    "dims,dtype,ns",
    [
        pytest.param(dims, dtype, ns, id=f"hd{dims[0]}x{dims[1]}-{str(dtype)[6:]}-ns{ns}")
        for dims, dtype, ns in DENSE_CASES
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_dense(dims, dtype, ns):
    check(*decode_inputs(*dims, dtype=dtype), ns=ns)


@pytest.mark.parametrize(
    "ratio,length,ns",
    [
        (1, 1, 1),
        (1, 0, 3),
        (3, 255, 3),
        (5, 256, 1),
        (8, 257, 3),
        (16, 512, 3),
        (32, 1537, 3),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_heads_and_blocks(ratio, length, ns):
    check(*decode_inputs(ratio=ratio, length=length), ns=ns)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_oversplit_with_empty_splits():
    """More splits than KV blocks, an empty batch entry, and a one-token split: the combine
    must ignore empty splits (zero O / -inf LSE) and the unused blocks may hold NaN."""
    q, k, v = decode_inputs(length=513)
    used = torch.tensor([0, 257], device="cuda", dtype=torch.int32)
    check(q, k, v, ns=8, used=used, poison=True)


@pytest.mark.parametrize(
    "dims",
    [pytest.param((d, d), id=f"hd{d}") for d in (64, 128, 256)],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_split_heuristic(dims):
    """num_splits=0 takes the decode split heuristic (per head dim) on a long KV."""
    check(*decode_inputs(*dims, ratio=8, length=16385), ns=0)


SEMANTICS = {
    "used": dict(used=[37, 789], poison=True),
    "empty": dict(used=[0, 789], poison=True),
    "causal": dict(causal=True),
    "local": dict(used=[37, 789], window=31),
    "softcap": dict(softcap=2.0, softmax_scale=0.1),
    "mask": dict(mask=True),
    "aux": dict(bias=[0.2, -0.3]),
}
# (kind, dims, ns, GQA ratio). hd256 and hd64 are separate kernels (own split targets /
# layouts): cover the mask, local, aux and NaN-poisoned unused KV paths there too, the latter
# with a wider query-head tile.
SEMANTIC_CASES = (
    [(kind, (128, 128), ns, 3) for kind in SEMANTICS for ns in (1, 3)]
    + [(kind, (256, 256), 3, 3) for kind in ("mask", "local", "aux")]
    + [("mask", (64, 64), 3, 3), ("used", (64, 64), 3, 8)]
)


@pytest.mark.parametrize(
    "kind,dims,ns,ratio",
    [
        pytest.param(kind, dims, ns, ratio, id=f"{kind}-hd{dims[0]}-ns{ns}-gqa{ratio}")
        for kind, dims, ns, ratio in SEMANTIC_CASES
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_semantics(kind, dims, ns, ratio):
    args = dict(SEMANTICS[kind])
    if "used" in args:
        args["used"] = torch.tensor(args["used"], device="cuda", dtype=torch.int32)
    if "bias" in args:
        args["bias"] = torch.tensor(args["bias"], device="cuda")
    tensors = decode_inputs(*dims, ratio=ratio)
    check(*tensors, ns=ns, **args)
    if kind == "aux":
        # Same compiled signature, different device pointer and values.
        args["bias"] = torch.tensor([-0.7, 0.8], device="cuda")
        check(*tensors, ns=ns, **args)


@pytest.mark.parametrize("ns", [1, 3])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_public_buffers_and_combine(ns):
    """Caller buffers are written in place; flash_attn_func matches the private forward with
    and without the LSE."""
    q, k, v = decode_inputs()
    out = torch.empty_like(q)
    lse = torch.empty((2, 6, 1), device="cuda", dtype=torch.float32)
    o, actual_lse = check(q, k, v, ns=ns, out=out, lse=lse)
    public_o, public_l = flash_attn_func(q, k, v, num_splits=ns, return_lse=True)
    no_lse_o, no_lse = flash_attn_func(q, k, v, num_splits=ns)
    if is_fake_mode():
        return
    assert o.data_ptr() == out.data_ptr() and actual_lse.data_ptr() == lse.data_ptr()
    torch.testing.assert_close(public_o, o)
    torch.testing.assert_close(public_l, actual_lse)
    assert no_lse is None
    torch.testing.assert_close(no_lse_o, o)


GRAPH_CASES = [
    pytest.param(layout, (128, 128), torch.bfloat16, ns, False, id=f"{layout}-hd128-ns{ns}")
    for layout in ("dense", "paged256", "paged768")
    for ns in (1, 3)
] + [
    pytest.param("dense", (256, 256), torch.float16, 3, False, id="dense-hd256-ns3"),
    pytest.param(
        "paged256", (128, 128), torch.bfloat16, 3, True, id="paged256-hd128-ns3-score_mod"
    ),
]


@pytest.mark.parametrize("layout,dims,dtype,ns,score_mod", GRAPH_CASES)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_graph_replay(layout, dims, dtype, ns, score_mod):
    """Replays read the mutated page table / used lengths / K/V (and score_mod aux tensor)
    through stable pointers, and the NaN-prefilled caller buffers are fully overwritten,
    including nonempty -> empty -> nonempty transitions. The paged score_mod sees the logical
    token, not the physical page slot."""
    if layout == "dense":
        q, k, v = decode_inputs(*dims, dtype=dtype, length=1025)
        table = None
        used = torch.tensor([1025, 513], device="cuda", dtype=torch.int32)
        steps = [(None, [1025, 513]), (None, [0, 1]), (None, [257, 0]), (None, [1, 1025])]
    else:
        page = int(layout[5:])
        q, k, v, table, used = paged_inputs(page, *dims, dtype=dtype)
        steps = [
            ([[7, 2, 5], [1, 7, 3]], [page + 17, 3 * page - 9]),
            ([[4, 0, 8], [8, 6, 4]], [0, 2 * page + 1]),
            ([[8, 5, 0], [2, 8, 1]], [2 * page + 1, 0]),
            ([[7, 2, 5], [1, 7, 3]], [1, 3 * page]),
        ]
    out = torch.empty_like(q)
    lse = torch.empty((2, q.shape[2], 1), device="cuda", dtype=torch.float32)
    modifier = {}
    if score_mod:
        # (batch, query head, logical token), oversized along tokens so a padded-lane read past
        # the table's capacity stays in bounds.
        capacity = table.shape[1] * k.shape[1]
        bias = torch.randn(2, q.shape[2], 2 * capacity, device="cuda")
        modifier = dict(score_mod=token_head_bias, aux_tensors=[bias])

    def run():
        return _flash_attn_fwd(
            q, k, v, page_table=table, seqused_k=used, num_splits=ns, out=out, lse=lse,
            return_lse=True, **modifier,
        )[:2]

    # Warm kernels and the allocator on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    if is_fake_mode():
        return
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_out, captured_lse = run()
    assert captured_out.data_ptr() == out.data_ptr()
    assert captured_lse.data_ptr() == lse.data_ptr()
    for mapping, lengths in steps:
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn_like(k))
        v.copy_(torch.randn_like(v))
        if mapping is not None:
            table.copy_(torch.tensor(mapping, device="cuda", dtype=torch.int32))
        used.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        poison_unused_kv(k, v, used, table)
        if score_mod:
            bias.copy_(torch.randn_like(bias))
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        ref = decode_reference(
            q, k, v, page_table=table, seqused=used,
            bias=bias[:, :, :capacity] if score_mod else None,
        )
        assert_decode_close(out, lse, ref)
    torch.cuda.synchronize()


@pytest.mark.parametrize("kind", ["prefill", "pack_gqa_off"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_swapab_request_falls_back(kind, monkeypatch):
    """A SwapAB request SwapAB cannot serve (prefill) or that a selection request vetoes
    (pack_gqa=False) runs the standard kernel, as auto does, with the same output."""
    # force_swapab fails such fallbacks; record selections from the real selector instead.
    monkeypatch.setattr(interface, "select_fwd_config", select_fwd_config)
    configs = record_fwd_configs(monkeypatch)
    q, k, v = decode_inputs()
    kw = {}
    if kind == "prefill":
        q = q.expand(-1, 2, -1, -1).contiguous()
    else:
        kw["pack_gqa"] = False
    outs = {}
    for backend in ("swapab", "auto", "standard"):
        monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", backend)
        outs[backend] = _flash_attn_fwd(q, k, v, **kw)[0]
    requested, auto, standard = configs
    assert not standard.use_swapab and requested == auto == standard
    if is_fake_mode():
        return
    assert torch.equal(outs["swapab"], outs["standard"])
    assert torch.equal(outs["auto"], outs["standard"])


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="drives FakeTensorMode itself")
def test_fake_to_real():
    """A fake-mode compile must produce the key the real call looks up (two-pass contract).
    Same kernel as test_dense[hd128x128-bfloat16-ns1], so the execution pass compiles nothing."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    q, k, v = decode_inputs()
    with FakeTensorMode() as mode:
        _flash_attn_fwd(
            mode.from_tensor(q), mode.from_tensor(k), mode.from_tensor(v),
            num_splits=1, return_lse=True,
        )
    size = len(_flash_attn_fwd.compile_cache.cache)
    check(q, k, v, ns=1)
    assert len(_flash_attn_fwd.compile_cache.cache) == size


@pytest.mark.parametrize("ns", [1, 3])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_normal_paging(ns, monkeypatch):
    """The standard kernel's non-TMA paged path (page size differs from the tile size) still works."""
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "standard")
    _, k, v, table, used = paged_inputs(512, 96, 64, seed=71)
    q = torch.randn(2, 17, 6, 96, device="cuda", dtype=torch.bfloat16)
    if not is_fake_mode():
        poison_unused_kv(k, v, used, table)
    o, lse, *_ = _flash_attn_fwd(
        q, k, v, page_table=table, seqused_k=used, num_splits=ns, return_lse=True
    )
    if is_fake_mode():
        return
    ref = decode_reference(q, k, v, page_table=table, seqused=used)
    assert_decode_close(o, lse, ref)


@pytest.mark.parametrize("ns", [1, 3])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_fully_masked_rows(ns):
    q, k, v = decode_inputs(length=257)
    o, lse, *_ = _flash_attn_fwd(
        q, k, v, mask_mod=none_kept, num_splits=ns, return_lse=True
    )
    if is_fake_mode():
        return
    assert torch.count_nonzero(o) == 0
    assert torch.isneginf(lse).all()


def packed_keep_mask(packed, seqlen_k):
    bits = (packed[:, 0, :, None].long() >> torch.arange(32, device=packed.device)) & 1
    return bits.reshape(packed.shape[0], -1)[:, :seqlen_k].bool()


# vec_size 1 / 8 / 32 / 64 are the scalar, sub-word, word and multi-word mask paths.
VECTORIZED_MASK_CASES = [(vec, 1025, 1) for vec in (1, 8, 32, 64)] + [
    (64, 300, 1),  # tail block not a multiple of 64 tokens
    (32, 1025, 3),
]


@pytest.mark.parametrize(
    "vec_size,length,ns",
    [
        pytest.param(vec, length, ns, id=f"vec{vec}-len{length}-ns{ns}")
        for vec, length, ns in VECTORIZED_MASK_CASES
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_vectorized_mask_mod(vec_size, length, ns):
    from mask_mod_definitions import (
        cute_packed_mask_aux,
        get_cute_packed_mask_aux_vec,
        make_packed_mask_aux_tensor,
    )

    if vec_size == 1:
        mask_mod = cute_packed_mask_aux
    else:
        mask_mod = get_cute_packed_mask_aux_vec(vec_size)
        mask_mod.__vec_size__ = vec_size
    q, k, v = decode_inputs(length=length)
    packed = make_packed_mask_aux_tensor(2, 1, length, seed=vec_size)
    o, lse, *_ = _flash_attn_fwd(
        q, k, v, mask_mod=mask_mod, aux_tensors=[packed], num_splits=ns, return_lse=True
    )
    if is_fake_mode():
        return
    ref = decode_reference(q, k, v, keep=packed_keep_mask(packed, length))
    assert_decode_close(o, lse, ref)


@pytest.mark.parametrize("backend", ["auto", "swapab"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_vectorized_mask_mod_reads_token_bit(backend, monkeypatch):
    """Mask word 0b10 keeps only token 1, even though the word is nonzero for token 0."""
    from mask_mod_definitions import get_cute_packed_mask_aux_vec

    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", backend)

    mask_mod = get_cute_packed_mask_aux_vec(32)
    mask_mod.__vec_size__ = 32
    q = torch.zeros(1, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v[:, 1] = 1
    packed = torch.zeros(1, 1, 4, device="cuda", dtype=torch.uint32)
    packed[..., 0] = 0b10
    packed.__assumed_align__ = 16
    o, *_ = _flash_attn_fwd(q, k, v, mask_mod=mask_mod, aux_tensors=[packed])
    if is_fake_mode():
        return
    torch.testing.assert_close(o.float(), torch.ones_like(o, dtype=torch.float32))


def test_empty_kv_returns_before_selection():
    """Empty inputs return before config selection, as for the standard kernel, and yield zero O
    and -inf LSE. pack_gqa=False vetoes SwapAB, so the autouse force_swapab fixture fails the
    call if it reaches selection."""
    q, k, v = decode_inputs(length=1)
    k, v = k[:, :0], v[:, :0]
    out, lse, *_ = _flash_attn_fwd(q, k, v, pack_gqa=False, return_lse=True)
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_explicit_config_selects_the_kernel(monkeypatch):
    """An explicit FwdConfig runs its kernel whatever FLASH_ATTENTION_SM100_BACKEND says."""
    q, k, v = decode_inputs(length=1025)
    selected = record_fwd_configs(monkeypatch)
    outs = {}
    for backend in ("swapab", "standard"):
        monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", backend)
        outs[backend] = _flash_attn_fwd(q, k, v, num_splits=3)[0]
    swapab, standard = selected
    assert swapab.use_swapab and not standard.use_swapab
    forced = {}
    for config, backend in ((swapab, "standard"), (standard, "swapab")):
        monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", backend)
        forced[config.use_swapab] = _flash_attn_fwd(q, k, v, num_splits=3, config=config)[0]
    if is_fake_mode():
        return
    assert torch.equal(forced[True], outs["swapab"])
    assert torch.equal(forced[False], outs["standard"])


PAGED_KINDS = {
    "used": {},
    # mask_mod and score_mod over logical tokens, an empty entry, and more splits than blocks.
    "masked_empty": dict(used=[0, 529], mask=True, bias=[0.2, -0.3], poison=True),
    # No seqused_k over three KV blocks: the final-max / normalized-stats handoff once
    # deadlocked here.
    "no_seqused": dict(used=None),
}
# Pages of 128 or 384 tokens: a 256-token KV block spans two pages, resolved per subtile.
PAGED_CASES = [(page, ns, "used") for page in (128, 384) for ns in (1, 3)] + [
    (256, 8, "masked_empty"),
    (256, 2, "no_seqused"),
]


@pytest.mark.parametrize(
    "page_size,ns,kind",
    [
        pytest.param(page, ns, kind, id=f"{kind}-page{page}-ns{ns}")
        for page, ns, kind in PAGED_CASES
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_paged(page_size, ns, kind):
    q, k, v, table, used = paged_inputs(page_size)
    args = {"used": used, **PAGED_KINDS[kind]}
    if isinstance(args["used"], list):
        args["used"] = torch.tensor(args["used"], device="cuda", dtype=torch.int32)
    if "bias" in args:
        args["bias"] = torch.tensor(args["bias"], device="cuda")
    check(q, k, v, ns=ns, page_table=table, **args)


VARLEN_CASES = [((128, 128), 1), ((128, 128), 3), ((256, 256), 3), ((128, 128), 0)]


@pytest.mark.parametrize(
    "dims,ns", [pytest.param(d, n, id=f"hd{d[0]}-ns{n}") for d, n in VARLEN_CASES]
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_varlen_decode_matches_dense(dims, ns):
    """Varlen Q with one token per sequence equals the dense call. Sequence 1 has no query token
    (CUDA-graph padding) and the trailing token belongs to no sequence: neither is written."""
    q, k, v = decode_inputs(*dims, ratio=8, length=1300, batch=4)
    used = torch.tensor([1300, 700, 1, 257], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1, 1, 2, 3], device="cuda", dtype=torch.int32)
    rows = [0, 2, 3]
    q_var = torch.cat([q[rows, 0], q[:1, 0]])  # (total_q = 4, h, d); token 3 is padding
    out = torch.full_like(q_var[..., : dims[1]], 7.0)
    lse = torch.full((q.shape[2], q_var.shape[0]), 7.0, device="cuda")
    _flash_attn_fwd(
        q_var,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=used,
        num_splits=ns,
        out=out,
        lse=lse,
    )
    dense_out, dense_lse, *_ = _flash_attn_fwd(
        q, k, v, seqused_k=used, num_splits=ns, return_lse=True
    )
    if is_fake_mode():
        return
    assert_decode_close(dense_out, dense_lse, decode_reference(q, k, v, seqused=used))
    torch.testing.assert_close(out[:3], dense_out[rows, 0], atol=1e-3, rtol=0)
    torch.testing.assert_close(lse[:, :3], dense_lse[rows, :, 0].T, atol=1e-5, rtol=0)
    assert torch.all(out[3] == 7.0) and torch.all(lse[:, 3] == 7.0)


def varlen_paged_inputs(dims=(128, 128), page_size=128, seed=5):
    """Serving-style decode: 4 sequences over 128-token pages, rank-3 varlen Q.

    The page table is a 3-wide view of a 4-wide table whose last column maps a NaN page, so a
    load past a row (or past seqused_k into an unowned page) poisons the output. Sequence 1 has
    no query token and the last token belongs to no sequence.
    """
    torch.manual_seed(seed)
    dq, dv = dims
    q = torch.randn(4, 1, 16, dq, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(17, page_size, 2, dq, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(17, page_size, 2, dv, device="cuda", dtype=torch.bfloat16)
    table_full = torch.randperm(16, device="cuda").to(torch.int32).view(4, 4)
    table_full[:, 3] = 16
    table = table_full[:, :3]
    used = torch.tensor([300, 37, 3 * page_size, 129], device="cuda", dtype=torch.int32)
    if not is_fake_mode():
        poison_unused_kv(k, v, used, table)
        k[16], v[16] = float("nan"), float("nan")
    cu_seqlens_q = torch.tensor([0, 1, 1, 2, 3], device="cuda", dtype=torch.int32)
    rows = [0, 2, 3]
    q_var = torch.cat([q[rows, 0], q[:1, 0]])
    return q, q_var, k, v, table, used, cu_seqlens_q, rows


@pytest.mark.parametrize("ns", [1, 3, 0])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_varlen_paged_decode(ns):
    """Varlen Q, 128-token pages, seqused_k and the split combine at a short capacity (static
    split counts; test_per_sequence_splits_graph_replay covers per-sequence counts)."""
    q, q_var, k, v, table, used, cu, rows = varlen_paged_inputs()
    out = torch.full_like(q_var, 7.0)
    _, lse, *_ = _flash_attn_fwd(
        q_var, k, v, cu_seqlens_q=cu, max_seqlen_q=1, page_table=table, seqused_k=used,
        num_splits=ns, out=out, return_lse=True,
    )
    if is_fake_mode():
        return
    ref = decode_reference(q, k, v, page_table=table, seqused=used)
    ref = ref._replace(out=ref.out[rows], lse=ref.lse[rows], out_pt=ref.out_pt[rows])
    assert_decode_close(out[:3, None], lse[:, :3].T[:, :, None], ref)
    assert torch.all(out[3] == 7.0)


@cute.jit
def global_q_bias(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    """Score bias read at the global query index, the FA4 varlen modifier convention."""
    q_frag = cute.make_rmem_tensor(1, cutlass.Int32)
    q_frag.store(q_idx + seqlen_info.offset_q)
    bias = cute.make_rmem_tensor(1, cutlass.Float32)
    bias[0] = aux_tensors[0][q_frag[0]]
    return score + bias.load()


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_varlen_modifier_sees_global_query_index(monkeypatch):
    """A varlen score_mod reads aux data at offset_q + q_idx, as on the standard kernel."""
    _, q_var, k, v, table, used, cu, _ = varlen_paged_inputs(seed=9)
    token_bias = torch.tensor([0.0, 2.5, -3.0, 0.0], device="cuda")
    kw = dict(
        cu_seqlens_q=cu, max_seqlen_q=1, page_table=table, seqused_k=used, num_splits=3,
        score_mod=global_q_bias, aux_tensors=[token_bias], return_lse=True,
    )
    out, lse, *_ = _flash_attn_fwd(q_var, k, v, **kw)
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "standard")
    std_out, std_lse, *_ = _flash_attn_fwd(q_var, k, v, **kw)
    if is_fake_mode():
        return
    torch.testing.assert_close(out[:3], std_out[:3], atol=2e-3, rtol=0)
    torch.testing.assert_close(lse[:, :3], std_lse[:, :3], atol=1e-4, rtol=0)


@pytest.fixture
def combine_split_counts(monkeypatch):
    """The per-sequence split-count tensors the forward hands to the combine, one per call."""
    counts = []
    combine = interface._flash_attn_fwd_combine

    def spy(*args, **kwargs):
        counts.append(kwargs.get("num_splits_dynamic_ptr"))
        return combine(*args, **kwargs)

    spy.compile_cache = combine.compile_cache
    monkeypatch.setattr(interface, "_flash_attn_fwd_combine", spy)
    return counts


def ragged_paged_inputs(lengths, page_size=256, pages_per_row=160, seed=3):
    """One query per sequence over permuted pages; unused pages hold NaN."""
    torch.manual_seed(seed)
    batch = len(lengths)
    num_pages = batch * pages_per_row
    q = torch.randn(batch, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(num_pages, page_size, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.randperm(num_pages, device="cuda").to(torch.int32).view(batch, pages_per_row)
    used = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    if not is_fake_mode():
        poison_unused_kv(k, v, used, table)
    return q, k, v, table, used


def ragged_dense_inputs(lengths, capacity=40960, seed=3):
    """One query per sequence over a dense KV cache; tokens past the lengths hold NaN."""
    torch.manual_seed(seed)
    batch = len(lengths)
    q = torch.randn(batch, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, capacity, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    used = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    if not is_fake_mode():
        poison_unused_kv(k, v, used)
    return q, k, v, None, used


# Empty, one-token and one-block rows next to long rows: the long rows get several splits.
RAGGED_LENGTHS = [0, 1, 300, 9000, 40000]


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_per_sequence_splits_follow_lengths(combine_split_counts):
    """With seqused_k the split count is chosen per sequence: short and empty rows take one
    split, longer rows more, at most num_splits; the output matches the reference."""
    q, k, v, table, used = ragged_paged_inputs(RAGGED_LENGTHS)
    o, lse, *_ = _flash_attn_fwd(
        q, k, v, page_table=table, seqused_k=used, num_splits=0, return_lse=True
    )
    if is_fake_mode():
        return
    assert_decode_close(o, lse, decode_reference(q, k, v, page_table=table, seqused=used))
    counts = combine_split_counts[-1].tolist()
    assert counts[:3] == [1, 1, 1]
    assert 1 < counts[3] < counts[4] <= 32


# Varlen Q replay steps: (cu_seqlens_q, seqused_k). An empty-query sequence moves between
# replays; the trailing token then belongs to no sequence. The lengths move the split counts and
# which split owns a partial last KV block; the last step ends every sequence on a block boundary.
VARLEN_REPLAY_STEPS = (
    ([0, 0, 1, 2, 3, 4], [40000, 0, 9000, 1, 300]),
    ([0, 1, 2, 2, 3, 4], [1, 1, 1, 1, 1]),
    ([0, 1, 2, 3, 4, 5], RAGGED_LENGTHS),
    ([0, 1, 1, 2, 3, 4], [9216, 40960, 1, 0, 256]),
)


RAGGED_KV_LAYOUTS = {
    "paged256": ragged_paged_inputs,
    # 128-token pages: a 256-token KV block spans two pages.
    "paged128": partial(ragged_paged_inputs, page_size=128, pages_per_row=320),
    "dense": ragged_dense_inputs,
}


@pytest.mark.parametrize(
    "kv_layout,varlen_q",
    [
        pytest.param(layout, varlen_q, id=f"{layout}-{'varlen_q' if varlen_q else 'dense_q'}")
        for layout, varlen_q in [
            ("paged256", False),
            ("paged256", True),
            ("dense", False),
            ("dense", True),
            ("paged128", True),
        ]
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_per_sequence_splits_graph_replay(kv_layout, varlen_q, combine_split_counts):
    """Under CUDA-graph replay the split counts follow the replayed seqused_k: they are
    recomputed each replay, before the forward and the combine read them. With varlen Q the
    replayed cu_seqlens_q also moves which token each sequence reads and writes. Dense KV
    bounds the partial last block's V loads by the replayed length (NaN past it)."""
    q, k, v, table, used = RAGGED_KV_LAYOUTS[kv_layout](RAGGED_LENGTHS)
    batch = q.shape[0]
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    if varlen_q:
        q_in = q[:, 0].clone()  # (total_q, h, d): one token slot per sequence
        lse = torch.empty((q.shape[2], batch), device="cuda", dtype=torch.float32)
        varlen_kwargs = dict(cu_seqlens_q=cu, max_seqlen_q=1)
    else:
        q_in = q
        lse = torch.empty((batch, q.shape[2], 1), device="cuda", dtype=torch.float32)
        varlen_kwargs = {}
    out = torch.empty_like(q_in)

    def run():
        _flash_attn_fwd(
            q_in, k, v, page_table=table, seqused_k=used, num_splits=0, out=out, lse=lse,
            return_lse=True, **varlen_kwargs,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    if is_fake_mode():
        return
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    counts = combine_split_counts[-1]
    assert counts is not None, "the capacity selects per-sequence split counts"
    for cu_list, lengths in VARLEN_REPLAY_STEPS:
        if varlen_q:
            cu.copy_(torch.tensor(cu_list, device="cuda", dtype=torch.int32))
            q_in.copy_(torch.randn_like(q_in))
        used.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        k.copy_(torch.randn_like(k))
        v.copy_(torch.randn_like(v))
        poison_unused_kv(k, v, used, table)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        if varlen_q:
            rows = [b for b in range(batch) if cu_list[b + 1] > cu_list[b]]
            tokens = [cu_list[b] for b in rows]
            if table is None:
                ref = decode_reference(q_in[tokens, None], k[rows], v[rows], seqused=used[rows])
            else:
                ref = decode_reference(
                    q_in[tokens, None], k, v, page_table=table[rows], seqused=used[rows]
                )
            assert_decode_close(out[tokens, None], lse[:, tokens].T[:, :, None], ref)
            continue
        assert_decode_close(out, lse, decode_reference(q, k, v, page_table=table, seqused=used))
        by_length = dict(zip(lengths, counts.tolist()))
        assert by_length[1] == 1
        if 40000 in by_length:
            assert by_length[0] == by_length[300] == 1 < by_length[9000] < by_length[40000]
    torch.cuda.synchronize()


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_per_sequence_splits_past_grid_limit():
    """256 sequences x 256 splits on one KV head: batch * num_splits = 65536 tiles, one past the
    grid Y/Z limit, still launch and match the reference."""
    torch.manual_seed(5)
    batch = 256
    q = torch.randn(batch, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 32768, 1, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # Short lengths keep the reference small; the capacity selects per-sequence splits.
    used = torch.randint(0, 4097, (batch,), device="cuda", dtype=torch.int32)
    if not is_fake_mode():
        used[:2] = torch.tensor([0, 4096])
        poison_unused_kv(k, v, used)
    o, lse, *_ = _flash_attn_fwd(q, k, v, seqused_k=used, num_splits=256, return_lse=True)
    if is_fake_mode():
        return
    ref = decode_reference(q, k[:, :4096], v[:, :4096], seqused=used)
    assert_decode_close(o, lse, ref)
