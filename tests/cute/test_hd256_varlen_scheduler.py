"""HD256 varlen API, cache, and CUDA-graph regression tests on SM100/SM110.

None and fresh CUDA scalar maxima must share a device-driven specialization
without host reads; Python integers retain rectangular scheduling.
Set FLASH_ATTENTION_HD256_STRESS=1 for additional randomized cases.
"""

import os
from functools import partial
from itertools import accumulate
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from flash_attn.cute import flash_attn_varlen_func, interface
from flash_attn.cute.cache_utils import JITCache
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
RUN_STRESS = int(os.getenv("FLASH_ATTENTION_HD256_STRESS", 0)) == 1

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


class RejectHostScalarExtraction(TorchDispatchMode):
    """Fail on aten._local_scalar_dense, the op behind .item()/int()/bool() on a tensor."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.name() == "aten::_local_scalar_dense":
            raise AssertionError(
                "host scalar extraction (aten::_local_scalar_dense) during a warmed "
                "varlen call; this is a device-to-host synchronization"
            )
        return func(*args, **(kwargs or {}))


def make_cu_seqlens(lengths):
    return torch.tensor((0, *accumulate(lengths)), dtype=torch.int32, device="cuda")


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


def varlen_reference(q, k, v, dout, q_lens, k_lens, causal, ref_dtype):
    """Return eager O/dQ/dK/dV in ref_dtype for active rows only.

    Ignore extra input capacity; return only O when dout is None.
    """
    scale = q.shape[-1] ** -0.5
    heads_q, heads_kv = q.shape[1], k.shape[1]
    qr, kr, vr = [t.detach().to(ref_dtype).requires_grad_(dout is not None) for t in (q, k, v)]
    outputs = []
    q_off = k_off = 0
    for nq, nk in zip(q_lens, k_lens):
        if nq > 0:
            assert nk > 0, "a query row with no keys has undefined softmax output"
            qi = qr[q_off : q_off + nq].transpose(0, 1)
            ki = (
                kr[k_off : k_off + nk].transpose(0, 1).repeat_interleave(heads_q // heads_kv, dim=0)
            )
            vi = (
                vr[k_off : k_off + nk].transpose(0, 1).repeat_interleave(heads_q // heads_kv, dim=0)
            )
            scores = (qi @ ki.transpose(-1, -2)) * scale
            if causal:
                # Bottom-right aligned causal mask, also correct when nq != nk.
                rows = torch.arange(nq, device=q.device)[:, None]
                cols = torch.arange(nk, device=q.device)[None, :]
                scores = scores.masked_fill(cols > rows + nk - nq, float("-inf"))
            outputs.append((scores.softmax(-1) @ vi).transpose(0, 1))
        q_off += nq
        k_off += nk
    out = torch.cat(outputs)
    if dout is None:
        return (out,)
    dq, dk, dv = torch.autograd.grad(out, (qr, kr, vr), dout[:q_off].to(ref_dtype))
    return out, dq[:q_off], dk[:k_off], dv[:k_off]


def rms(t):
    return t.square().mean().sqrt()


def check_against_reference(actual, q, k, v, dout, q_lens, k_lens, causal, dtype):
    """Compare O/dQ/dK/dV against FP64 eager, allowing the measured low-precision eager error.

    The fused kernel orders its softmax/reductions differently from eager and
    rounds P and the accumulator to the output dtype, so on top of the measured
    unfused low-precision eager error allow two output ULPs (pointwise/max and RMS).
    """
    truth = varlen_reference(q, k, v, dout, q_lens, k_lens, causal, torch.float64)
    eager = varlen_reference(q, k, v, dout, q_lens, k_lens, causal, dtype)
    assert len(actual) == len(truth) == len(eager) == (1 if dout is None else 4)
    ulp = torch.finfo(dtype).eps
    for name, got, want, low in zip(("O", "dQ", "dK", "dV"), actual, truth, eager):
        assert got.shape == want.shape, (name, got.shape, want.shape)
        assert torch.isfinite(got).all(), name
        error = (got.double() - want).abs()
        eager_error = (low.double() - want).abs()
        allowance = 2 * ulp * want.abs()
        assert error.max() <= eager_error.max() + allowance.max(), (
            name,
            error.max().item(),
            eager_error.max().item(),
            allowance.max().item(),
        )
        assert rms(error) <= rms(eager_error) + rms(allowance), (
            name,
            rms(error).item(),
            rms(eager_error).item(),
            rms(allowance).item(),
        )


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


# ---------------------------------------------------------------------------
# Compile-key behaviour: none/cuda share one specialization, int selects another.
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
    assert (len(fwd_cache.cache), len(bwd_cache.cache)) == (2, 2), (
        "explicit int maxima should select a distinct specialization"
    )
    assert all(
        not torch.is_tensor(value)
        for cache in (fwd_cache, bwd_cache)
        for key in cache.cache
        for value in key
    ), "tensor leaked into a compile key"

    for mode, result in by_mode.items():
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
    assert constructor.call_args.kwargs["is_static_persistent"] is False
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


@MAX_MODES
def test_hd256_paged_maximum_uses_table_extent(max_mode):
    torch.manual_seed(SEED)
    dtype = torch.bfloat16
    q = torch.randn(2, 128, 4, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn(6, 128, 1, HEAD_DIM, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    pages = torch.tensor([[3, 0], [5, 1]], device="cuda", dtype=torch.int32)
    maximum = 256 if max_mode == "int" else None
    if max_mode == "cuda":
        maximum = torch.tensor(256, device="cuda", dtype=torch.int32)

    def forward():
        return flash_attn_varlen_func(q, k, v, page_table=pages, max_seqlen_k=maximum)[0]

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
