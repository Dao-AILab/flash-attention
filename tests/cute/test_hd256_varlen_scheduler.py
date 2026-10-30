"""HD256 varlen API, cache, and CUDA-graph regression tests on SM100/SM110.

None and fresh CUDA scalar maxima must share a device-driven specialization
without host reads; Python integers retain rectangular scheduling in the backward.
Set FLASH_ATTENTION_HD256_STRESS=1 for additional randomized cases.
"""

import os
from functools import partial
from unittest.mock import Mock

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens
from host_sync import RejectHostScalarExtraction

from flash_attn.cute import flash_attn_varlen_func, interface
from flash_attn.cute.cache_utils import JITCache
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1
RUN_STRESS = int(os.getenv("FLASH_ATTENTION_HD256_STRESS", "0")) == 1

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11),
        reason="SM100/SM110-only test",
    ),
    pytest.mark.skipif(
        USE_FAKE_TENSOR,
        reason="data-dependent CUDA maxima and CUDA graphs need real tensors",
    ),
]

HEAD_DIM = 256
SEED = 20260814

# (q_lens, k_lens, dtype). Zero-length entries are empty batch slots.
LENGTH_CASES = [
    pytest.param((128, 256), (128, 256), torch.bfloat16, id="complete"),
    pytest.param((129, 257), (129, 257), torch.bfloat16, id="ragged"),
    pytest.param((129, 0, 257), (129, 0, 257), torch.bfloat16, id="empty_slot"),
    pytest.param((65, 257), (129, 513), torch.bfloat16, id="mixed_qk"),
    pytest.param((129, 257), (129, 257), torch.float16, id="ragged_fp16"),
]
MAX_MODES = pytest.mark.parametrize(
    "max_mode", ["none", "int", "cuda"], ids=["max_none", "max_int", "max_cuda"]
)
CAUSAL = pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])


def make_maxima(max_mode, q_lens, k_lens, cu_seqlens_q, cu_seqlens_k):
    if max_mode == "none":
        return None, None
    if max_mode == "int":
        return max(q_lens), max(k_lens)
    assert max_mode == "cuda"
    # Fresh device scalars every call: tensor identity must not leak into the compile key.
    return (
        (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max(),
        (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max(),
    )


def make_inputs(q_lens, k_lens, dtype, num_heads_q=4, num_heads_kv=1):
    torch.manual_seed(SEED)
    q = torch.randn(
        sum(q_lens), num_heads_q, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        sum(k_lens), num_heads_kv, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    return q, k, v, dout


def run_varlen(q, k, v, dout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
    )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out.detach(), dq, dk, dv


# ---------------------------------------------------------------------------
# Correctness matrix: O/dQ/dK/dV vs FP64 eager for every maxima form.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_lens, k_lens, dtype", LENGTH_CASES)
@MAX_MODES
@CAUSAL
def test_hd256_varlen_maxima(q_lens, k_lens, dtype, max_mode, causal):
    q, k, v, dout = make_inputs(q_lens, k_lens, dtype)
    cu_q, cu_k = make_cu_seqlens(q_lens), make_cu_seqlens(k_lens)

    def run():
        max_q, max_k = make_maxima(max_mode, q_lens, k_lens, cu_q, cu_k)
        return run_varlen(q, k, v, dout, cu_q, cu_k, max_q, max_k, causal)

    first = run()  # may compile
    with RejectHostScalarExtraction():
        second = run()  # warmed: must not synchronize through .item()
    torch.cuda.synchronize()
    for name, a, b in zip(("O", "dQ", "dK", "dV"), first, second):
        assert torch.equal(a, b), f"{name} differs between two identical warmed calls"

    check_against_reference(second, q, k, v, dout, q_lens, k_lens, causal, dtype)


@pytest.mark.parametrize("layout", ["packed", "dense", "packed_q"])
@pytest.mark.parametrize("used", ["q", "k", "qk"])
@pytest.mark.parametrize("max_mode", ["none", "int"])
@pytest.mark.parametrize(
    "dtype, heads_q, heads_kv",
    [
        pytest.param(torch.bfloat16, 4, 1, id="bf16_gqa"),
        pytest.param(torch.float16, 2, 2, id="fp16_mha"),
    ],
)
@CAUSAL
def test_hd256_seqused_backward(
    layout,
    used,
    max_mode,
    dtype,
    heads_q,
    heads_kv,
    causal,
):
    """Check seqused outputs and gradients across layouts with static and varlen scheduling."""
    if layout == "dense" and max_mode == "int":
        pytest.skip("dense integer maxima match shape-derived defaults in max_mode='none'")
    torch.manual_seed(SEED)
    packed_q, packed_k = layout != "dense", layout == "packed"
    q_capacity = (193, 257, 385) if packed_q else (384,) * 3
    k_capacity = (577, 513, 641) if packed_k else (640,) * 3
    q_lens = (65, 0, 257) if "q" in used else q_capacity
    k_lens = (449, 385, 513) if "k" in used else k_capacity
    if used == "qk":
        k_lens = (449, 0, 513)
    cu_q = make_cu_seqlens(q_capacity) if packed_q else None
    cu_k = make_cu_seqlens(k_capacity) if packed_k else None
    used_q = torch.tensor(q_lens, device="cuda", dtype=torch.int32) if "q" in used else None
    used_k = torch.tensor(k_lens, device="cuda", dtype=torch.int32) if "k" in used else None
    q_shape = (sum(q_capacity),) if packed_q else (len(q_capacity), q_capacity[0])
    k_shape = (sum(k_capacity),) if packed_k else (len(k_capacity), k_capacity[0])
    q = torch.randn(*q_shape, heads_q, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(*k_shape, heads_kv, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)

    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        seqused_q=used_q,
        seqused_k=used_k,
        max_seqlen_q=max(q_capacity) if max_mode != "none" else None,
        max_seqlen_k=max(k_capacity) if max_mode != "none" else None,
        causal=causal,
    )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    torch.cuda.synchronize()

    def compact(t, capacity, lengths):
        batches = t.split(capacity) if t.ndim == 3 else t.unbind(0)
        return torch.cat([batch[:length] for batch, length in zip(batches, lengths)])

    actual = (
        compact(out.detach(), q_capacity, q_lens),
        compact(dq, q_capacity, q_lens),
        compact(dk, k_capacity, k_lens),
        compact(dv, k_capacity, k_lens),
    )
    check_against_reference(
        actual,
        compact(q, q_capacity, q_lens),
        compact(k, k_capacity, k_lens),
        compact(v, k_capacity, k_lens),
        compact(dout, q_capacity, q_lens),
        q_lens,
        k_lens,
        causal,
        dtype,
    )


# ---------------------------------------------------------------------------
# Compile-key behaviour: none/cuda share one specialization; int adds a backward one.
# ---------------------------------------------------------------------------


@CAUSAL
def test_hd256_varlen_maxima_compile_keys(causal, monkeypatch):
    q_lens = k_lens = (129, 257)
    dtype = torch.bfloat16
    q, k, v, dout = make_inputs(q_lens, k_lens, dtype)
    cu_q, cu_k = make_cu_seqlens(q_lens), make_cu_seqlens(k_lens)

    def run(max_mode):
        max_q, max_k = make_maxima(max_mode, q_lens, k_lens, cu_q, cu_k)
        return run_varlen(q, k, v, dout, cu_q, cu_k, max_q, max_k, causal)

    fwd_cache, bwd_cache = JITCache(), JITCache()
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", fwd_cache)
    monkeypatch.setattr(_flash_attn_bwd, "compile_cache", bwd_cache)
    by_mode = {"none": run("none")}
    assert (len(fwd_cache.cache), len(bwd_cache.cache)) == (1, 1)

    by_mode["cuda"] = run("cuda")
    with RejectHostScalarExtraction():
        cuda_again = run("cuda")
    torch.cuda.synchronize()
    assert (len(fwd_cache.cache), len(bwd_cache.cache)) == (1, 1), (
        "CUDA scalar maxima must reuse the max_seqlen=None compile keys"
    )
    for name, a, b, c in zip(("O", "dQ", "dK", "dV"), by_mode["none"], by_mode["cuda"], cuda_again):
        assert torch.equal(a, b), f"{name}: none vs cuda maxima differ on the shared kernel"
        assert torch.equal(b, c), f"{name}: repeated cuda maxima call differs"

    by_mode["int"] = run("int")
    torch.cuda.synchronize()
    # The forward reads lengths on device and does not specialize on the maxima; the hd256
    # backward still selects a separate specialization for explicit int maxima.
    assert (len(fwd_cache.cache), len(bwd_cache.cache)) == (1, 2), (
        "explicit int maxima should select a distinct backward specialization only"
    )
    assert all(
        not torch.is_tensor(value)
        for cache in (fwd_cache, bwd_cache)
        for key in cache.cache
        for value in key
    ), "tensor leaked into a compile key"

    for result in by_mode.values():
        check_against_reference(result, q, k, v, dout, q_lens, k_lens, causal, dtype)


# ---------------------------------------------------------------------------
# CUDA graph: fixed capacity and cu_seqlens shape, varying device values per replay.
# ---------------------------------------------------------------------------

GRAPH_CAPACITY = 640  # physical packed rows; every replay below fits inside it
GRAPH_SLOTS = 3  # fixed cu_seqlens shape (GRAPH_SLOTS + 1)
GRAPH_REPLAYS = [
    (256, 128, 256),  # all slots active, capacity fully used
    (129, 0, 257),  # ragged tiles with an empty middle slot
    (64, 0, 0),  # single short active sequence
    (0, 0, 0),  # no active work; the capacity grid must terminate safely
]


@pytest.mark.parametrize("max_mode", ["none", "cuda"], ids=["max_none", "max_cuda"])
@CAUSAL
def test_hd256_varlen_cuda_graph_replay(max_mode, causal):
    dtype = torch.bfloat16
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    # Keep leaf creation, warmup and capture on one stream so autograd does not
    # introduce a dependency on a stale non-capturing stream during capture.
    with torch.cuda.stream(side):
        q, k, v, dout = make_inputs((GRAPH_CAPACITY,), (GRAPH_CAPACITY,), dtype)
        # Static cu_seqlens buffers; only their device values change between replays.
        cu_q = make_cu_seqlens(GRAPH_REPLAYS[0])
        cu_k = make_cu_seqlens(GRAPH_REPLAYS[0])
    assert cu_q.shape == (GRAPH_SLOTS + 1,)

    def step():
        if max_mode == "cuda":
            # Captured device op: replays recompute the maxima from the new cu values.
            max_q = (cu_q[1:] - cu_q[:-1]).max()
            max_k = (cu_k[1:] - cu_k[:-1]).max()
        else:
            max_q = max_k = None
        return run_varlen(q, k, v, dout, cu_q, cu_k, max_q, max_k, causal)

    with torch.cuda.stream(side):
        for _ in range(2):  # compile + warm outside capture
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        static_out = step()

    for lengths in GRAPH_REPLAYS:
        cu_q.copy_(make_cu_seqlens(lengths))
        cu_k.copy_(make_cu_seqlens(lengths))
        graph.replay()
        torch.cuda.synchronize()
        total = sum(lengths)
        if total == 0:
            continue  # No output/gradient rows are defined for an all-empty replay.
        # Rows beyond cu_seqlens[-1] are outside every sequence and undefined.
        active = tuple(t[:total].clone() for t in static_out)
        check_against_reference(active, q, k, v, dout, lengths, lengths, causal, dtype)


# ---------------------------------------------------------------------------
# Generic (non-HD256) FA4 varlen: no-max must pick the flat scheduler, not the
# tiny-work static-persistent special case, so it shares the CUDA-scalar key.
# ---------------------------------------------------------------------------


def test_generic_varlen_no_max_shares_flat_scheduler_key(monkeypatch):
    q_lens = k_lens = (24, 40)  # total 64 rows fits one 128-row tile ("tiny work")
    head_dim, dtype = 128, torch.bfloat16
    torch.manual_seed(SEED)
    q = torch.randn(sum(q_lens), 2, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cu_q, cu_k = make_cu_seqlens(q_lens), make_cu_seqlens(k_lens)

    def forward(max_mode):
        max_q, max_k = make_maxima(max_mode, q_lens, k_lens, cu_q, cu_k)
        out, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k
        )
        return out

    fwd_cache = JITCache()
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", fwd_cache)
    constructor = Mock(wraps=interface.FlashAttentionForwardSm100)
    monkeypatch.setattr(interface, "FlashAttentionForwardSm100", constructor)
    out_none = forward("none")
    assert constructor.call_args.kwargs["config"].is_static_persistent is False
    assert len(fwd_cache.cache) == 1
    out_cuda = forward("cuda")
    with RejectHostScalarExtraction():
        out_cuda_again = forward("cuda")
    torch.cuda.synchronize()
    assert len(fwd_cache.cache) == 1, (
        "max_seqlen=None on tiny varlen work must use the flat varlen scheduler "
        "(same compile key as a CUDA scalar max), not the static-persistent special case"
    )
    assert torch.equal(out_none, out_cuda)
    assert torch.equal(out_cuda, out_cuda_again)

    check_against_reference((out_none,), q, k, v, None, q_lens, k_lens, False, dtype)


MASKS = {
    "causal": dict(causal=True),
    "noncausal": dict(causal=False),
    "local": dict(causal=False, window_size=(1024, 0)),
}


# hd256 2CTA: 24/4 without PackGQA, 16/4 with it; 6/1 covers one KV head.
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize(
    "head_dim, num_heads_q, num_heads_kv", [(128, 8, 2), (256, 24, 4), (256, 16, 4), (256, 6, 1)]
)
def test_padded_varlen_slots_use_persistent_scheduler(
    head_dim, num_heads_q, num_heads_kv, mask, monkeypatch
):
    """A prompt and 3 decodes in a CUDA graph's 512 request slots, max_seqlen_q = all tokens.
    The persistent route must match the unpadded (single-tile) call bitwise."""
    q_real, k_real = (1500, 1, 1, 1), (1500, 700, 4096, 9)
    q_lens, k_lens = q_real + (0,) * 508, k_real + (0,) * 508
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    q = torch.randn(sum(q_lens), num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    # A fresh compile cache: the mock only sees kernels it constructs.
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", JITCache())
    constructor = Mock(wraps=interface.FlashAttentionForwardSm100)
    monkeypatch.setattr(interface, "FlashAttentionForwardSm100", constructor)

    def forward(lens_q, lens_k, max_q):
        out, lse = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=make_cu_seqlens(lens_q), cu_seqlens_k=make_cu_seqlens(lens_k),
            max_seqlen_q=max_q, max_seqlen_k=max(lens_k), return_lse=True, **MASKS[mask],
        )
        return out, lse, constructor.call_args.kwargs["has_tile_count_semaphore"]

    out_padded, lse_padded, persistent = forward(q_lens, k_lens, sum(q_lens))
    assert persistent
    out_real, lse_real, persistent = forward(q_real, k_real, max(q_real))
    assert not persistent
    assert torch.equal(out_padded, out_real)
    assert torch.equal(lse_padded, lse_real)
    if mask != "local":  # the eager reference has no window
        check_against_reference(
            (out_padded,), q, k, v, None, q_lens, k_lens, mask == "causal", dtype
        )


@pytest.mark.parametrize("mask", ["causal", "noncausal"])
def test_persistent_varlen_cuda_graph_replay(mask, monkeypatch):
    """Replay a persistent-scheduler graph with changing lengths across 64 request slots."""
    capacity, dtype = 640, torch.bfloat16
    replays = [
        (10,) * 64,  # every slot active
        (300, 1, 1, 1) + (0,) * 60,  # a prompt and 3 decodes
        (0,) * 40 + (128,) + (0,) * 23,  # one request in the middle
        (0,) * 64,  # no work: the grid must terminate safely
    ]
    torch.manual_seed(SEED)
    q = torch.randn(capacity, 8, 128, device="cuda", dtype=dtype)
    k = torch.randn(capacity, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cu = make_cu_seqlens(replays[0])

    def step():
        out, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=capacity,
            max_seqlen_k=capacity, causal=mask == "causal",
        )
        return out

    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", JITCache())
    constructor = Mock(wraps=interface.FlashAttentionForwardSm100)
    monkeypatch.setattr(interface, "FlashAttentionForwardSm100", constructor)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # compile outside capture
    assert constructor.call_args.kwargs["has_tile_count_semaphore"]
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        static_out = step()
    for lengths in replays:
        cu.copy_(make_cu_seqlens(lengths))
        graph.replay()
        torch.cuda.synchronize()
        total = sum(lengths)
        if total:
            check_against_reference(
                (static_out[:total].clone(),), q, k, v, None, lengths, lengths, mask == "causal", dtype
            )


def test_persistent_varlen_one_split_ignores_seqlen_k_per_split(monkeypatch):
    """Persistent num_splits=1 ignores seqlen_k_per_split, which only sizes split-KV."""
    q_lens, k_lens = (300,) + (1,) * 15, (4096,) * 16
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    q = torch.randn(sum(q_lens), 8, 128, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), 2, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", JITCache())
    constructor = Mock(wraps=interface.FlashAttentionForwardSm100)
    monkeypatch.setattr(interface, "FlashAttentionForwardSm100", constructor)
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=make_cu_seqlens(q_lens), cu_seqlens_k=make_cu_seqlens(k_lens),
        max_seqlen_q=300, max_seqlen_k=4096, causal=True, num_splits=1, seqlen_k_per_split=1024,
    )
    assert constructor.call_args.kwargs["has_tile_count_semaphore"]
    check_against_reference((out,), q, k, v, None, q_lens, k_lens, True, dtype)


@pytest.mark.parametrize(
    "q_real, k_real, slots, num_splits, persistent",
    [
        ((1500, 1, 1, 1), (1500, 700, 4096, 9), 512, 1, True),  # padded: persistent route
        ((1500, 1, 1, 1), (1500, 700, 4096, 9), 512, 2, False),  # explicit SplitKV
        ((150,), (8192,), 4, 0, False),  # heuristic SplitKV (long K, few tiles)
    ],
)
def test_hd256_pack_gqa_changes_only_on_persistent_route(
    q_real, k_real, slots, num_splits, persistent, monkeypatch
):
    """HD256 6/1 keeps the SplitKV call unpacked and the persistent call packed."""
    q_lens = q_real + (0,) * (slots - len(q_real))
    k_lens = k_real + (0,) * (slots - len(k_real))
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    q = torch.randn(sum(q_lens), 6, 256, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), 1, 256, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", JITCache())
    constructor = Mock(wraps=interface.FlashAttentionForwardSm100)
    monkeypatch.setattr(interface, "FlashAttentionForwardSm100", constructor)
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=make_cu_seqlens(q_lens), cu_seqlens_k=make_cu_seqlens(k_lens),
        max_seqlen_q=max(q_real), max_seqlen_k=max(k_real), causal=True, num_splits=num_splits,
    )
    kwargs = constructor.call_args.kwargs
    assert (kwargs["config"].num_splits > 1) == (not persistent)
    assert kwargs["config"].use_dynamic_varlen == persistent
    assert kwargs["config"].pack_gqa == persistent
    check_against_reference((out,), q, k, v, None, q_lens, k_lens, True, dtype)


@pytest.mark.parametrize(
    "head_dim, num_heads_q, num_heads_kv, batch", [(128, 8, 2, 512), (256, 24, 4, 2)]
)
def test_decode_varlen_keeps_main_scheduler(
    head_dim, num_heads_q, num_heads_kv, batch, monkeypatch
):
    """Single-token decode stays static-persistent, with 2CTA for hd256."""
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    q = torch.randn(batch, num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch * 256, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", JITCache())
    constructor = Mock(wraps=interface.FlashAttentionForwardSm100)
    monkeypatch.setattr(interface, "FlashAttentionForwardSm100", constructor)
    cu_q, cu_k = make_cu_seqlens((1,) * batch), make_cu_seqlens((256,) * batch)
    flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1, max_seqlen_k=256,
        causal=True, num_splits=1,
    )
    kwargs = constructor.call_args.kwargs
    assert not kwargs["has_tile_count_semaphore"]
    assert kwargs["config"].is_static_persistent
    assert kwargs["config"].use_2cta_instrs == (head_dim == 256)


def test_generic_paged_seqused_cuda_maxima_use_host_bounds():
    """Paged SplitKV scheduler bounds must not require host reads of CUDA maxima."""
    torch.manual_seed(SEED)
    dtype, head_dim = torch.bfloat16, 128
    q = torch.randn(2, 128, 4, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(6, 128, 1, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    pages = torch.tensor([[3, 0], [5, 1]], device="cuda", dtype=torch.int32)
    seqused_q = torch.full((2,), 128, device="cuda", dtype=torch.int32)

    def forward():
        return flash_attn_varlen_func(
            q, k, v, page_table=pages, seqused_q=seqused_q, num_splits=3,
            max_seqlen_q=seqused_q.max(),
            max_seqlen_k=torch.tensor(256, device="cuda", dtype=torch.int32),
        )[0]

    forward()
    with RejectHostScalarExtraction():
        out = forward()
    packed_k = k[pages].reshape(512, 1, head_dim)
    packed_v = v[pages].reshape_as(packed_k)
    check_against_reference(
        (out.flatten(0, 1),), q.flatten(0, 1), packed_k, packed_v, None, (128, 128), (256, 256),
        False, dtype,
    )


@pytest.mark.parametrize("num_splits", [1, 3])
@MAX_MODES
def test_hd256_paged_maximum_uses_table_extent(max_mode, num_splits):
    torch.manual_seed(SEED)
    dtype = torch.bfloat16
    q = torch.randn(2, 128, 4, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn(6, 128, 1, HEAD_DIM, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    pages = torch.tensor([[3, 0], [5, 1]], device="cuda", dtype=torch.int32)
    maximum = 256 if max_mode == "int" else None
    if max_mode == "cuda":
        maximum = torch.tensor(256, device="cuda", dtype=torch.int32)

    # Exercise used-length hints with explicit splits.
    seqused_q = torch.full((2,), 128, device="cuda", dtype=torch.int32) if num_splits > 1 else None

    def forward():
        max_q = seqused_q.max() if seqused_q is not None and max_mode == "cuda" else None
        return flash_attn_varlen_func(
            q, k, v, page_table=pages, max_seqlen_q=max_q, max_seqlen_k=maximum,
            seqused_q=seqused_q, num_splits=num_splits,
        )[0]

    forward()
    with RejectHostScalarExtraction():
        out = forward()
    packed_k = k[pages].reshape(512, 1, HEAD_DIM)
    packed_v = v[pages].reshape_as(packed_k)
    check_against_reference(
        (out.flatten(0, 1),),
        q.flatten(0, 1),
        packed_k,
        packed_v,
        None,
        (128, 128),
        (256, 256),
        False,
        dtype,
    )


def test_hd256_clc_requires_host_maximum(monkeypatch):
    monkeypatch.setattr(_flash_attn_bwd, "compile_cache", JITCache())
    monkeypatch.setattr(
        interface,
        "BlackwellFusedMultiHeadAttentionBackward",
        partial(interface.BlackwellFusedMultiHeadAttentionBackward, use_clc_scheduler=True),
    )
    q, k, v, dout = make_inputs((128, 256), (128, 256), torch.bfloat16)
    cu = make_cu_seqlens((128, 256))
    with pytest.raises(
        Exception, match="SM100 hd256 varlen dQ requires max_seqlen_q for grid sizing"
    ):
        run_varlen(q, k, v, dout, cu, cu, None, None, False)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Opt-in stress: random ragged batches with empty slots, MQA and GQA.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_STRESS, reason="set FLASH_ATTENTION_HD256_STRESS=1 to run")
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("heads", [(4, 1), (4, 2)], ids=["mqa", "gqa"])
@MAX_MODES
@CAUSAL
def test_hd256_varlen_stress(seed, heads, max_mode, causal):
    dtype = torch.bfloat16
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, 600, (8,), generator=generator)
    lengths[torch.randint(0, 8, (1,), generator=generator)] = 0  # guarantee an empty slot
    lengths = tuple(int(n) for n in lengths)
    if sum(lengths) == 0:
        pytest.skip("degenerate all-empty batch")
    heads_q, heads_kv = heads
    q, k, v, dout = make_inputs(lengths, lengths, dtype, num_heads_q=heads_q, num_heads_kv=heads_kv)
    cu = make_cu_seqlens(lengths)
    max_q, max_k = make_maxima(max_mode, lengths, lengths, cu, cu)
    result = run_varlen(q, k, v, dout, cu, cu, max_q, max_k, causal)
    torch.cuda.synchronize()
    check_against_reference(result, q, k, v, dout, lengths, lengths, causal, dtype)
