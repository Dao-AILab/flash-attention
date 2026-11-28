# Copyright (c) 2026, Dao-AILab.

"""Runtime regressions for the opt-in packed-HND SM103 decode adapter."""

import math
import os

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3),
    reason="packed-HND adapter runtime tests require an SM103 GPU",
)


@pytest.fixture(autouse=True)
def _fp32_reference_precision(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def _make_decode_case(batch, capacity=8192, workspace_splits=None):
    from flash_attn.cute.sm103_hd512_splitkv_forward import (
        run_sm103_hd512_splitkv_decode,
        sm103_hd512_splitkv_num_splits,
    )

    torch.manual_seed(20261126 + batch)
    heads, kv_heads, dim, page_size = 32, 4, 512, 64
    splits = sm103_hd512_splitkv_num_splits(batch) if workspace_splits is None else workspace_splits
    table_width = math.ceil(capacity / page_size) + 2
    page_table = (
        torch.arange(batch * table_width, device="cuda", dtype=torch.int32)
        .reshape(batch, table_width)
        .roll(7, dims=1)
    )
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(
        batch * table_width,
        kv_heads,
        page_size,
        2 * dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k, v = packed.split(dim, dim=-1)
    out = torch.empty_like(q)
    partial = torch.empty(splits, batch, heads, dim, device="cuda", dtype=torch.float32)
    lse = torch.empty(
        splits, heads, batch, device="cuda", dtype=torch.float32
    ).transpose(1, 2)
    lengths = torch.empty(batch, device="cuda", dtype=torch.int32)
    valid_splits = torch.empty_like(lengths)

    def invoke(scale):
        run_sm103_hd512_splitkv_decode(
            q, k, v, out, partial, lse, scale, lengths, page_table, valid_splits
        )

    return {
        "q": q,
        "k": k,
        "v": v,
        "packed": packed,
        "page_table": page_table,
        "lengths": lengths,
        "out": out,
        "partial": partial,
        "lse": lse,
        "valid_splits": valid_splits,
        "invoke": invoke,
    }


def _set_lengths_and_poison(case, lengths, poison):
    case["lengths"].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
    if poison is not None:
        for batch_idx, length in enumerate(lengths):
            if length % 64:
                page_id = case["page_table"][batch_idx, length // 64].item()
                case["v"][page_id, :, length % 64 :, :] = poison
            unused_pages = case["page_table"][
                batch_idx, math.ceil(length / 64) :
            ].long()
            case["v"][unused_pages] = poison


def _reference(case, lengths, scale):
    reference = torch.zeros_like(case["q"], dtype=torch.float32)
    # Force the independent FP32 math path instead of another attention kernel.
    with sdpa_kernel(SDPBackend.MATH):
        for batch_idx, length in enumerate(lengths):
            if length == 0:
                continue
            pages = case["page_table"][batch_idx, : math.ceil(length / 64)].long()
            k = case["k"][pages].permute(0, 2, 1, 3).reshape(-1, 4, 512)[:length]
            v = case["v"][pages].permute(0, 2, 1, 3).reshape(-1, 4, 512)[:length]
            reference[batch_idx] = torch.nn.functional.scaled_dot_product_attention(
                case["q"][batch_idx].view(1, 32, 1, 512).float(),
                k.permute(1, 0, 2)[None].float(),
                v.permute(1, 0, 2)[None].float(),
                scale=scale,
                enable_gqa=True,
            ).view(32, 512)
    return reference


def _poison_outputs(case):
    case["out"].fill_(math.nan)
    case["partial"].fill_(math.nan)
    case["lse"].fill_(math.nan)
    case["valid_splits"].fill_(-1)


def _assert_result(case, lengths, reference):
    # Count independent (sequence, KV head) CTAs before splitting the token axis.
    split_cap = math.ceil(128 / (len(lengths) * 4))
    expected_splits = torch.tensor(
        [min(split_cap, math.ceil(length / 128)) for length in lengths],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(case["valid_splits"], expected_splits)
    assert torch.isfinite(case["out"]).all()
    # Preserve the existing adapter regression's BF16-rounded reference gate.
    torch.testing.assert_close(
        case["out"].float(),
        reference.to(torch.bfloat16).float(),
        atol=0.015625,
        rtol=0.0,
    )
    for batch_idx, length in enumerate(lengths):
        if length == 0:
            assert torch.count_nonzero(case["out"][batch_idx]) == 0
        else:
            cosine = torch.nn.functional.cosine_similarity(
                case["out"][batch_idx].double(), reference[batch_idx].double(), dim=-1
            )
            assert torch.all(cosine >= 0.99999), cosine.min().item()


@pytest.mark.parametrize("batch,splits", [(2, 16), (3, 11), (4, 8), (8, 4), (31, 2), (32, 1), (33, 1)])
@pytest.mark.parametrize("legacy_workspace", [False, True])
def test_sm103_hd512_batch_split_workspace(batch, splits, legacy_workspace):
    from flash_attn.cute.sm103_hd512_splitkv_forward import sm103_hd512_splitkv_num_splits

    assert sm103_hd512_splitkv_num_splits(batch) == splits
    edge = 128 * splits  # Crossing this edge revisits split zero.
    lengths = tuple((edge + 1, 0, edge - 1, edge, 65)[b % 5] for b in range(batch))
    case = _make_decode_case(batch, capacity=edge + 1, workspace_splits=32 if legacy_workspace else splits)
    _set_lengths_and_poison(case, lengths, math.inf)
    reference = _reference(case, lengths, 0.5)
    _poison_outputs(case)
    case["invoke"](0.5)
    _assert_result(case, lengths, reference)
    assert torch.isnan(case["partial"][splits:]).all()
    assert torch.isnan(case["lse"][splits:]).all()


@pytest.mark.parametrize(
    "lengths",
    [
        (0,),
        (1,),
        (11,),
        (63,),
        (64,),
        (65,),
        (127,),
        (128,),
        (129,),
        (191,),
        (255,),
        (256,),
        (257,),
        (1024,),
        (4096,),
        (8192,),
        (32767,),
        (32768,),
        (32769,),
        (0, 32767, 32768, 32769),
        (0, 63, 128, 129),
        (0, 1, 63, 64, 65, 127, 128, 129),
        (0, 11, 63, 64, 65, 127, 128, 8192) * 4,
    ],
)
@pytest.mark.parametrize("scale", [1.0 / math.sqrt(512), 0.5])
@pytest.mark.parametrize("poison", [None, math.nan, math.inf, -math.inf])
def test_sm103_hd512_splitkv_tail_and_empty(lengths, scale, poison):
    case = _make_decode_case(len(lengths), capacity=max(8192, max(lengths)))
    _set_lengths_and_poison(case, lengths, poison)
    reference = _reference(case, lengths, scale)
    for _ in range(10):
        _poison_outputs(case)
        case["invoke"](scale)
        _assert_result(case, lengths, reference)


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_sm103_hd512_splitkv_graph_empty_tail_mutation(batch):
    case = _make_decode_case(batch)
    clean_packed = case["packed"].clone()
    initial_table = case["page_table"].clone()
    scale = 1.0 / math.sqrt(512)
    _set_lengths_and_poison(case, (8192,) * batch, None)
    case["invoke"](scale)  # Warm both adapter specializations before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case["invoke"](scale)

    for step, length in enumerate((8192, 129, 0, 65, 0, 8192)):
        lengths = tuple(0 if length == 0 else max(0, length - b) for b in range(batch))
        case["packed"].copy_(clean_packed)
        case["page_table"].copy_(initial_table.roll(step, dims=1))
        _set_lengths_and_poison(case, lengths, math.nan)
        reference = _reference(case, lengths, scale)
        for _ in range(10):
            _poison_outputs(case)
            graph.replay()
            _assert_result(case, lengths, reference)


@pytest.mark.parametrize("batch", [32, 64])
def test_sm103_hd512_single_split_combine_graph(batch):
    from flash_attn.cute.sm103_hd512_splitkv_combine import run_sm103_hd512_splitkv_combine

    # Include BF16 rounding ties, signed zero, and nonfinite live data. Inactive
    # rows have both scratch tensors poisoned, even when the graph was captured live.
    vals = torch.tensor(
        [-0.0, 0.0, 1 + 2**-8, 1 + 3 * 2**-8, -0.25, math.inf, -math.inf, math.nan],
        dtype=torch.float32, device="cuda",
    )
    partial = vals.repeat(1, batch, 32, 64)
    lse = torch.empty(1, 32, batch, device="cuda", dtype=torch.float32).transpose(1, 2)
    valid = torch.ones(batch, device="cuda", dtype=torch.int32)
    out = torch.empty_like(partial[0], dtype=torch.bfloat16)
    lse.zero_()
    run_sm103_hd512_splitkv_combine(partial, lse, out, valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_sm103_hd512_splitkv_combine(partial, lse, out, valid)
    rows = torch.arange(batch, device="cuda")
    for live in (rows >= 0, rows % 2 != 0, rows < 0, rows % 2 == 0):
        valid.copy_(live)
        partial.view(-1, 8).copy_(vals)
        lse.fill_(-32.0)
        for head, value in enumerate([math.nan, math.inf, -math.inf, 0.0]):
            lse[:, :, head] = value
        partial[0, ~live] = math.nan
        lse[0, ~live] = math.nan
        expected = (partial[0] + 0.0).to(torch.bfloat16)
        expected.masked_fill_(~torch.isfinite(lse[0, :, :, None]), math.nan)
        expected[~live] = 0.0
        out.fill_(-123.0)
        graph.replay()
        finite_or_inf = ~expected.isnan()
        assert torch.equal(out.isnan(), expected.isnan())
        assert torch.equal(out.view(torch.int16)[finite_or_inf], expected.view(torch.int16)[finite_or_inf])
