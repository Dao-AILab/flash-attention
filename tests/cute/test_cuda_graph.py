# Copyright (c) 2026, Tri Dao.
"""CUDA graph capture and replay for the FA4 (CuTe) entry points.

A captured call may not synchronize with the host (`.item()`, `.tolist()`, a host read of
a device maximum), must launch on the capturing stream, and must read every per-sequence
length from device memory, so that a replay with new device values computes the right
thing. Each test warms a call outside capture, asserts that the warmed call performs no
host scalar extraction, captures it once, then replays with new inputs and compares the
static outputs against the FP64 eager reference in eager_reference.py.

The hd256 varlen path has its own replay test in test_hd256_varlen_scheduler.py; these
cover the generic (hd128) dense, varlen and paged-decode paths, forward and backward.
"""

import os

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens
from host_sync import RejectHostScalarExtraction

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU"),
    pytest.mark.skipif(USE_FAKE_TENSOR, reason="CUDA graphs need real tensors"),
]

HEAD_DIM = 128
NHEADS, NHEADS_KV = 4, 1
DTYPE = torch.bfloat16
SEED = 20261015
REPLAYS = 3
CAUSAL = pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])


def capture(step, warmup=2):
    """Warm `step` on a side stream, capture it once, return (graph, static outputs)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):  # JIT compile and warm outside capture
            step()
        with RejectHostScalarExtraction():  # a warmed call must not sync with the host
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        static = step()
    return graph, static


def refill(*tensors):
    """New random values in place; the graph only ever sees these static buffers."""
    with torch.no_grad():
        for t in tensors:
            t.copy_(torch.randn_like(t))


def new_tensor(*shape, requires_grad=False):
    return torch.randn(*shape, device="cuda", dtype=DTYPE, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Dense (batch, seqlen, heads, hdim): forward and backward in one graph.
# ---------------------------------------------------------------------------


@CAUSAL
def test_dense_cuda_graph_replay(causal):
    torch.manual_seed(SEED)
    batch, seqlen_q, seqlen_k = 2, 256, 384
    q = new_tensor(batch, seqlen_q, NHEADS, HEAD_DIM, requires_grad=True)
    k = new_tensor(batch, seqlen_k, NHEADS_KV, HEAD_DIM, requires_grad=True)
    v = new_tensor(batch, seqlen_k, NHEADS_KV, HEAD_DIM, requires_grad=True)
    dout = new_tensor(batch, seqlen_q, NHEADS, HEAD_DIM)

    def step():
        out, _ = flash_attn_func(q, k, v, causal=causal)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out.detach(), dq, dk, dv

    graph, static = capture(step)
    for _ in range(REPLAYS):
        refill(q, k, v, dout)
        graph.replay()
        torch.cuda.synchronize()
        check_against_reference(
            tuple(t.flatten(0, 1).clone() for t in static),
            q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), dout.flatten(0, 1),
            (seqlen_q,) * batch, (seqlen_k,) * batch, causal, DTYPE,
        )


# ---------------------------------------------------------------------------
# Varlen: fixed packed capacity and cu_seqlens shape, varying device values per replay.
# ---------------------------------------------------------------------------

GRAPH_CAPACITY = 640  # physical packed rows; every replay below fits inside it
GRAPH_SLOTS = 3  # fixed cu_seqlens shape (GRAPH_SLOTS + 1)
GRAPH_REPLAYS = [
    (256, 128, 256),  # all slots active, capacity fully used
    (129, 0, 257),  # ragged tiles with an empty middle slot
    (64, 0, 0),  # single short active sequence
    (0, 0, 0),  # no active work; the schedule must terminate safely
]


@pytest.mark.parametrize("max_mode", ["none", "int"], ids=["max_none", "max_int"])
@CAUSAL
def test_varlen_cuda_graph_replay(max_mode, causal):
    torch.manual_seed(SEED)
    q = new_tensor(GRAPH_CAPACITY, NHEADS, HEAD_DIM, requires_grad=True)
    k = new_tensor(GRAPH_CAPACITY, NHEADS_KV, HEAD_DIM, requires_grad=True)
    v = new_tensor(GRAPH_CAPACITY, NHEADS_KV, HEAD_DIM, requires_grad=True)
    dout = new_tensor(GRAPH_CAPACITY, NHEADS, HEAD_DIM)
    # Static cu_seqlens buffers; only their device values change between replays.
    cu_q = make_cu_seqlens(GRAPH_REPLAYS[0])
    cu_k = make_cu_seqlens(GRAPH_REPLAYS[0])
    assert cu_q.shape == (GRAPH_SLOTS + 1,)
    # A host maximum, when given, is a capacity bound that holds for every replay.
    max_seqlen = GRAPH_CAPACITY if max_mode == "int" else None

    def step():
        out, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, causal=causal,
        )
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out.detach(), dq, dk, dv

    graph, static = capture(step)
    for lengths in GRAPH_REPLAYS:
        cu_q.copy_(make_cu_seqlens(lengths))
        cu_k.copy_(make_cu_seqlens(lengths))
        refill(q, k, v, dout)
        graph.replay()
        torch.cuda.synchronize()
        total = sum(lengths)
        if total == 0:
            continue  # No output/gradient rows are defined for an all-empty replay.
        # Rows beyond cu_seqlens[-1] are outside every sequence and undefined.
        active = tuple(t[:total].clone() for t in static)
        check_against_reference(active, q, k, v, dout, lengths, lengths, causal, DTYPE)


# ---------------------------------------------------------------------------
# Paged decode: static page_table and seqused_k buffers, varying cache lengths per replay.
# This is the FA4 side of the paged KV-cache capture regression fixed for FA2 in #2984:
# the per-sequence cache length must be read on device, and any capacity bound must come
# from the page_table extent (host shape), never from a host read of seqused_k.
# ---------------------------------------------------------------------------

PAGE_SIZE, PAGES_PER_SEQ = 128, 8
# Eight 128-wide K tiles per sequence: the split heuristic picks several splits for this
# small decode batch, so the auto case captures the split-KV forward plus the combine.
CAPACITY = PAGE_SIZE * PAGES_PER_SEQ
DECODE_REPLAYS = [
    (CAPACITY, 1),  # full capacity next to a single key
    (129, 257),  # partial pages
    (1, CAPACITY - 1),  # one key next to a partial last page
    (PAGE_SIZE, 2 * PAGE_SIZE),  # exact page boundaries
]


@pytest.mark.parametrize(
    "num_splits",
    [
        pytest.param(1, id="splits1"),
        pytest.param(
            0,
            id="splits_auto",
            marks=pytest.mark.skipif(
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
                reason="SM120 forward only supports num_splits=1",
            ),
        ),
    ],
)
def test_paged_decode_cuda_graph_replay(num_splits):
    torch.manual_seed(SEED)
    batch, num_pages = 2, 2 * PAGES_PER_SEQ + 3
    q = new_tensor(batch, 1, NHEADS, HEAD_DIM)
    k = new_tensor(num_pages, PAGE_SIZE, NHEADS_KV, HEAD_DIM)
    v = new_tensor(num_pages, PAGE_SIZE, NHEADS_KV, HEAD_DIM)
    # Distinct pages per sequence in a shuffled pool, so a wrong page is a wrong answer.
    page_table = (
        torch.randperm(num_pages, device="cuda")[: batch * PAGES_PER_SEQ]
        .view(batch, PAGES_PER_SEQ)
        .to(torch.int32)
    )
    seqused_k = torch.tensor(DECODE_REPLAYS[0], device="cuda", dtype=torch.int32)

    def step():
        return flash_attn_varlen_func(
            q, k, v, page_table=page_table, seqused_k=seqused_k, num_splits=num_splits
        )[0]

    graph, out = capture(step)
    for lengths in DECODE_REPLAYS:
        seqused_k.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        refill(q, k, v)
        graph.replay()
        torch.cuda.synchronize()
        packed_k = torch.cat(
            [k[page_table[b]].reshape(-1, NHEADS_KV, HEAD_DIM)[:n] for b, n in enumerate(lengths)]
        )
        packed_v = torch.cat(
            [v[page_table[b]].reshape(-1, NHEADS_KV, HEAD_DIM)[:n] for b, n in enumerate(lengths)]
        )
        check_against_reference(
            (out.flatten(0, 1).clone(),), q.flatten(0, 1), packed_k, packed_v, None,
            (1,) * batch, lengths, False, DTYPE,
        )
