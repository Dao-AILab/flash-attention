# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SM12x FlashAttention-4 forward tests."""

import math
import os

import cutlass.cute as cute
import pytest
import torch

from flash_attn.cute import interface
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode
from flash_attn.cute.sm120.policy import (
    low_hd_paged_decode_tile_m,
    visible_decode_seqlen_k,
)
from flash_attn.cute.sm120.runtime import Sm120ForwardHost

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
    pytest.skip(
        "SM12x FlashAttention-4 test requires CUDA SM 12.x.",
        allow_module_level=True,
    )

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1


def _varlen_fwd(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    softcap=None,
    window_size=(None, None),
    learnable_sink=None,
    sinks=None,
    num_splits=1,
    pack_gqa=None,
    score_mod=None,
    aux_tensors=None,
    return_softmax_lse=False,
    out=None,
):
    """Forward-only call with a preallocated output, as inference engines use it."""
    if learnable_sink is None:
        learnable_sink = sinks
    if window_size == (-1, -1):
        window_size = (None, None)
    out, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_lse=return_softmax_lse,
        out=out,
    )
    return (out, lse) if return_softmax_lse else out


def _paged_decode(
    q,
    k_cache,
    v_cache,
    cache_seqlens=None,
    page_table=None,
    cu_seqlens_q=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    num_splits=0,
    pack_gqa=None,
    sinks=None,
    out=None,
):
    """Paged decode through cu_seqlens_q + seqused_k + page_table."""
    return _varlen_fwd(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sinks=sinks,
        out=out,
    )


@pytest.mark.parametrize(
    (
        "head_dim",
        "head_dim_v",
        "visible_seqlen_k",
        "qhead_per_kvhead",
        "expected_tile_m",
    ),
    [
        pytest.param(64, 64, 256, 1, None, id="minimum-exclusive"),
        pytest.param(64, 64, 512, 8, None, id="hd64-short-mqa-fallback"),
        pytest.param(64, 64, 513, 8, 16, id="hd64-long-mqa"),
        pytest.param(128, 128, 512, 8, 32, id="hd128-short-mqa"),
        pytest.param(128, 128, 2048, 1, 16, id="hd128-mha"),
        pytest.param(64, 128, 2048, 1, None, id="asymmetric-fallback"),
        pytest.param(96, 96, 2048, 1, None, id="unqualified-fallback"),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_low_hd_decode_tile_qualification(
    head_dim,
    head_dim_v,
    visible_seqlen_k,
    qhead_per_kvhead,
    expected_tile_m,
):
    assert (
        low_hd_paged_decode_tile_m(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            paged_kv=True,
            seqlen_q=1,
            visible_seqlen_k=visible_seqlen_k,
            qhead_per_kvhead=qhead_per_kvhead,
        )
        == expected_tile_m
    )


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_decode_visible_k_uses_exact_local_window():
    assert (
        visible_decode_seqlen_k(
            8192,
            is_local=False,
            window_size_left=256,
            window_size_right=0,
        )
        == 8192
    )
    assert (
        visible_decode_seqlen_k(
            8192,
            is_local=True,
            window_size_left=256,
            window_size_right=0,
        )
        == 257
    )
    assert (
        visible_decode_seqlen_k(
            128,
            is_local=True,
            window_size_left=256,
            window_size_right=0,
        )
        == 128
    )


@cute.jit
def _coordinate_score_mod(
    score,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    """Exercise logical batch/head/token indices, including packed GQA."""
    return (
        score
        + batch_idx * 0.00390625
        + head_idx * 0.0078125
        + q_idx * 0.015625
        - kv_idx * 0.0078125
    )


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    batch_idx: int = 0,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    score_mod: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 attention output and head-major LSE for one sequence."""
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_kv_heads, _ = k.shape
    assert num_q_heads % num_kv_heads == 0
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    k = k.float().repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v = v.float().repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scale

    q_idx = torch.arange(q_len, device=q.device)
    kv_idx = torch.arange(k_len, device=q.device)
    if score_mod == "coordinate":
        scores += (
            batch_idx * 0.00390625
            + torch.arange(num_q_heads, device=q.device)[:, None, None] * 0.0078125
            + q_idx[None, :, None] * 0.015625
            - kv_idx[None, None, :] * 0.0078125
        )
    elif score_mod is not None:
        raise ValueError(f"unknown score_mod reference: {score_mod}")

    relative_position = q_idx[:, None] + k_len - q_len - kv_idx[None, :]
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap

    visible = torch.ones_like(relative_position, dtype=torch.bool)
    if causal:
        visible &= relative_position >= 0
    if window_size[0] is not None:
        visible &= relative_position <= window_size[0]
    if window_size[1] is not None:
        visible &= relative_position >= -window_size[1]
    scores.masked_fill_(~visible[None], -torch.inf)

    if sinks is None:
        probabilities = torch.softmax(scores, dim=-1)
        lse = torch.logsumexp(scores, dim=-1)
    else:
        row_max = torch.maximum(scores.amax(dim=-1), sinks.float()[:, None])
        weights = torch.exp(scores - row_max[:, :, None])
        denominator = weights.sum(dim=-1) + torch.exp(sinks.float()[:, None] - row_max)
        probabilities = weights / denominator[:, :, None]
        lse = torch.log(denominator) + row_max
    output = torch.einsum("hqk,khd->qhd", probabilities, v)
    return output, lse


def _assert_close(actual, expected, **kwargs) -> None:
    """torch.testing.assert_close, skipped in the compilation pass (fake tensors)."""
    if not is_fake_mode():
        torch.testing.assert_close(actual, expected, **kwargs)


def _assert_attention_close(
    output: torch.Tensor,
    reference: torch.Tensor,
    lse: torch.Tensor | None = None,
    lse_reference: torch.Tensor | None = None,
) -> None:
    if is_fake_mode():
        return
    error = (output.float() - reference.float()).abs()
    assert is_fake_mode() or error.max().item() < 1e-2
    assert is_fake_mode() or error.mean().item() < 5e-4
    if lse is not None:
        assert lse_reference is not None
        lse_error = (lse.float() - lse_reference.float()).abs()
        assert is_fake_mode() or lse_error.max().item() < 2e-4


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "head_dim_v",
        "causal",
        "window_size",
        "softcap",
        "has_sink",
        "pack_gqa",
        "preallocate_out",
    ),
    [
        pytest.param(
            torch.bfloat16,
            4,
            4,
            32,
            32,
            False,
            (None, None),
            0.0,
            False,
            False,
            False,
            id="bf16-mha-hd32-global",
        ),
        pytest.param(
            torch.float16,
            6,
            1,
            64,
            96,
            True,
            (None, None),
            0.0,
            True,
            True,
            True,
            id="fp16-mqa-hd64-hdv96-causal-sink",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            96,
            64,
            False,
            (47, 11),
            4.0,
            False,
            True,
            False,
            id="bf16-gqa-hd96-hdv64-local-softcap",
        ),
        pytest.param(
            torch.float16,
            4,
            4,
            128,
            64,
            False,
            (63, 7),
            0.0,
            False,
            False,
            True,
            id="fp16-mha-hd128-hdv64-local-out",
        ),
        pytest.param(
            torch.float16,
            8,
            8,
            256,
            256,
            True,
            (None, None),
            0.0,
            False,
            False,
            False,
            id="fp16-mha-hd256-causal",
        ),
        pytest.param(
            torch.float16,
            6,
            1,
            192,
            192,
            True,
            (None, None),
            0.0,
            True,
            None,
            False,
            id="fp16-mqa-hd192-causal-sink-auto-pack",
        ),
        pytest.param(
            torch.float16,
            8,
            2,
            128,
            128,
            False,
            (None, None),
            8.0,
            False,
            True,
            False,
            id="fp16-gqa-hd128-softcap-packed",
        ),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_dense_feature_matrix_matches_reference(
    dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    causal,
    window_size,
    softcap,
    has_sink,
    pack_gqa,
    preallocate_out,
):
    """Cover dense MHA/GQA/MQA, dtype, shape, mask, sink, and output modes."""
    torch.manual_seed(20260801 + head_dim + head_dim_v)
    batch_size, q_len, k_len = 2, 37, 141
    q = torch.randn(
        batch_size,
        q_len,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        k_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        batch_size,
        k_len,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    sinks = (
        torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
        if has_sink
        else None
    )
    out_buffer = torch.empty(
        batch_size,
        q_len,
        num_q_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    output, lse = _varlen_fwd(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        sinks=sinks,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
        out=out_buffer if preallocate_out else None,
    )
    references = [
        _attention_reference(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            batch_idx=batch_idx,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            sinks=sinks,
        )
        for batch_idx in range(batch_size)
    ]
    output_reference = torch.stack([reference[0] for reference in references])
    lse_reference = torch.stack([reference[1] for reference in references])
    _assert_attention_close(output, output_reference, lse, lse_reference)
    if preallocate_out:
        assert is_fake_mode() or output.data_ptr() == out_buffer.data_ptr()


@pytest.mark.parametrize("has_sink", [True], ids=["sink"])
@pytest.mark.parametrize("window_size", [(None, 0), (96, 0)], ids=["causal", "local"])
@pytest.mark.parametrize("head_dim", [64, 192])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_split_kv_empty_splits_match_reference(head_dim, window_size, has_sink):
    """Early causal / local M blocks span fewer KV blocks than splits, so some splits are
    empty and only write the empty-split partial (every tile family, folded DMA included)."""
    torch.manual_seed(20261101 + head_dim)
    batch_size, seqlen, num_q_heads, num_kv_heads = 2, 512, 4, 2
    q = torch.randn(
        batch_size, seqlen, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    sinks = (
        torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16) if has_sink else None
    )
    output, lse = _varlen_fwd(
        q,
        k,
        v,
        window_size=window_size,
        sinks=sinks,
        num_splits=3,
        return_softmax_lse=True,
    )
    if is_fake_mode():
        return
    references = [
        _attention_reference(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            batch_idx=batch_idx,
            window_size=window_size,
            sinks=sinks,
        )
        for batch_idx in range(batch_size)
    ]
    output_reference = torch.stack([reference[0] for reference in references])
    lse_reference = torch.stack([reference[1] for reference in references])
    _assert_attention_close(output, output_reference, lse, lse_reference)


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "head_dim_v",
        "causal",
        "window_size",
        "pack_gqa",
        "num_splits",
    ),
    [
        pytest.param(
            torch.float16,
            8,
            2,
            80,
            48,
            True,
            (None, None),
            True,
            2,
            id="fp16-gqa-hd80-hdv48-causal-splitkv",
        ),
        pytest.param(
            torch.bfloat16,
            6,
            1,
            128,
            96,
            False,
            (63, 9),
            None,
            1,
            id="bf16-mqa-hd128-hdv96-local-auto-pack",
        ),
        pytest.param(
            torch.float16,
            8,
            8,
            256,
            256,
            True,
            (None, None),
            False,
            1,
            id="fp16-mha-hd256-causal",
        ),
        pytest.param(
            torch.float16,
            8,
            2,
            64,
            64,
            False,
            (None, None),
            False,
            3,
            id="fp16-gqa-hd64-global-splitkv",
        ),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_varlen_feature_matrix_matches_reference(
    dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    causal,
    window_size,
    pack_gqa,
    num_splits,
):
    """Cover ragged batches across topology, asymmetric dimensions, and SplitKV."""
    torch.manual_seed(20260802 + head_dim + head_dim_v)
    q_lengths = (19, 53)
    k_lengths = (79, 151)
    q_parts = [
        torch.randn(
            length,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        for length in q_lengths
    ]
    k_parts = [
        torch.randn(
            length,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        for length in k_lengths
    ]
    v_parts = [
        torch.randn(
            length,
            num_kv_heads,
            head_dim_v,
            device="cuda",
            dtype=dtype,
        )
        for length in k_lengths
    ]
    cu_seqlens_q = torch.tensor(
        [0, q_lengths[0], sum(q_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, k_lengths[0], sum(k_lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    output, lse = _varlen_fwd(
        torch.cat(q_parts),
        torch.cat(k_parts),
        torch.cat(v_parts),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        window_size=window_size,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
    )
    references = [
        _attention_reference(
            q,
            k,
            v,
            batch_idx=batch_idx,
            causal=causal,
            window_size=window_size,
        )
        for batch_idx, (q, k, v) in enumerate(zip(q_parts, k_parts, v_parts))
    ]
    output_reference = torch.cat([reference[0] for reference in references])
    lse_reference = torch.cat([reference[1] for reference in references], dim=1)
    _assert_attention_close(output, output_reference, lse, lse_reference)


@pytest.mark.parametrize(
    ("head_dim", "num_kv_heads", "pack_gqa", "causal", "num_splits"),
    # Every pair of factors appears.
    [
        (64, 8, False, True, 1),
        (64, 2, True, False, 3),
        (128, 8, False, False, 3),
        (128, 2, True, True, 1),
        (64, 8, False, False, 1),
        (128, 2, True, True, 3),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_varlen_tile_to_batch_lookup_matches_reference(
    monkeypatch, head_dim, num_kv_heads, pack_gqa, causal, num_splits
):
    """Many-batch varlen maps tiles to batches through the precomputed lookup table
    (forced below its batch threshold), including empty sequences and padding tiles."""
    monkeypatch.setattr(interface, "BIN_BATCH_SEARCH_THRESH", 0)
    lookups = []
    compute_blocks_to_batch = interface._compute_blocks_to_batch

    def spy(*args):
        lookups.append(args)
        return compute_blocks_to_batch(*args)

    # The helper reads its compile cache through the module-level name.
    spy.compile_cache = compute_blocks_to_batch.compile_cache
    monkeypatch.setattr(interface, "_compute_blocks_to_batch", spy)
    torch.manual_seed(20261102 + head_dim)
    num_q_heads, batch_size = 8, 40
    q_lengths = torch.randint(0, 150, (batch_size,)).tolist()
    q_lengths[3] = q_lengths[17] = 0
    k_lengths = [q + int(torch.randint(0, 100, ())) for q in q_lengths]
    q_parts = [
        torch.randn(n, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        for n in q_lengths
    ]
    k_parts = [
        torch.randn(n, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        for n in k_lengths
    ]
    v_parts = [torch.randn_like(k) for k in k_parts]

    def cu(lengths):
        return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda",
                            dtype=torch.int32)

    output, lse = _varlen_fwd(
        torch.cat(q_parts),
        torch.cat(k_parts),
        torch.cat(v_parts),
        cu_seqlens_q=cu(q_lengths),
        cu_seqlens_k=cu(k_lengths),
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
    )
    assert lookups
    if is_fake_mode():
        return
    references = [
        _attention_reference(q, k, v, batch_idx=batch_idx, causal=causal)
        for batch_idx, (q, k, v) in enumerate(zip(q_parts, k_parts, v_parts))
        if q.shape[0] > 0
    ]
    output_reference = torch.cat([reference[0] for reference in references])
    lse_reference = torch.cat([reference[1] for reference in references], dim=1)
    _assert_attention_close(output, output_reference, lse, lse_reference)


@pytest.mark.parametrize(
    "max_k, static_splits, cap, dynamic",
    [
        (512, 1, 8, False),  # the longest request's chain costs less than the split path
        (512, 3, 8, False),
        (4096, 1, 22, True),
        (4096, 8, 22, True),
        (32768, 8, 8, False),  # the static plan already splits as far as the cap
    ],
)
def test_sm120_dynamic_split_gate(monkeypatch, max_k, static_splits, cap, dynamic):
    """Per-batch splits only where the static plan's worst case (one request of the longest
    length on its static splits) costs more than the split path's fixed cost."""
    from flash_attn.cute.sm120.runtime import Sm120ForwardPlan

    def resolve_plan(*, requested_num_splits, **kwargs):
        return Sm120ForwardPlan(
            num_splits=static_splits if requested_num_splits < 1 else cap,
            split_num_n_blocks=max_k // 64,
            direct_uniform_batch=True,
            split_qk_n=False,
            split_kv_blocks_per_cta=0,
            transpose_qk_pv=False,
        )

    monkeypatch.setattr(Sm120ForwardHost, "dynamic_splits", True)
    monkeypatch.setattr(Sm120ForwardHost, "resolve_plan", staticmethod(resolve_plan))
    k = torch.empty(64, 64, 2, 128, device="cuda", dtype=torch.bfloat16)
    _, plan = Sm120ForwardHost.plan_forward(
        arch=interface._get_device_arch(), k=k, v=k, head_dim=128, head_dim_v=128, tile_mn=None,
        total_q=32, num_head=8, num_head_kv=2, qhead_per_kvhead=4, batch_size=32, seqlen_q=None,
        max_seqlen_q=1, max_seqlen_k=max_k, causal=False, local=False, window_size_left=None,
        window_size_right=None, pack_gqa=True, page_table=k, page_size=64, num_splits=0,
        has_cu_seqlens_q=True, has_seqused_q=False, has_cu_seqlens_k=False,
        has_seqused_k=True, has_score_or_mask_mod=False, device=k.device, fake_mode=False,
    )
    assert plan.dynamic_splits == dynamic
    assert plan.num_splits == (cap if dynamic else static_splits)


def _paged_cache(lengths, page_size, num_kv_heads, head_dim):
    """bf16 K / V caches with each request's pages scattered, its page table, and the page
    count of each request."""
    pages = [(n + page_size - 1) // page_size for n in lengths]
    k_cache = torch.randn(
        sum(pages), page_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.zeros(len(lengths), max(pages), device="cuda", dtype=torch.int32)
    order = torch.randperm(sum(pages), device="cuda", dtype=torch.int32)
    first = 0
    for batch_idx, n in enumerate(pages):
        page_table[batch_idx, :n] = order[first : first + n]
        first += n
    return k_cache, v_cache, page_table, pages


def _scheduler_metadata_case(kind, head_dim):
    """(q, k, v, forward kwargs, get_scheduler_metadata kwargs) of a varlen call: paged decode
    with one long request (per-batch splits) or even lengths (static plan), or a dense varlen
    prefill above the tile-lookup batch threshold."""
    num_q_heads, num_kv_heads = 8, 2
    if kind == "prefill":
        batch_size, seqlen = interface.BIN_BATCH_SEARCH_THRESH + 44, 64
        q = torch.randn(batch_size * seqlen, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch_size * seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cu = torch.arange(0, (batch_size + 1) * seqlen, seqlen, device="cuda", dtype=torch.int32)
        common = dict(
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=seqlen, max_seqlen_k=seqlen, causal=True
        )
        return q, k, v, common, dict(total_q=q.shape[0])
    page_size = 64
    lengths = [16384] + [300] * 15 if kind == "one_long" else [512] * 16
    k, v, page_table, _ = _paged_cache(lengths, page_size, num_kv_heads, head_dim)
    q = torch.randn(len(lengths), num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    common = dict(
        cu_seqlens_q=torch.arange(len(lengths) + 1, device="cuda", dtype=torch.int32),
        seqused_k=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=max(lengths),
        causal=True,
    )
    return q, k, v, dict(common, page_table=page_table), dict(page_size=page_size)


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="Compares outputs and counts prepare launches")
@pytest.mark.parametrize(
    ("kind", "head_dim"), [(kind, d) for kind in ("one_long", "even", "prefill") for d in (64, 128)]
)
def test_sm120_forward_reuses_scheduler_metadata(monkeypatch, kind, head_dim):
    """get_scheduler_metadata plans as the forward does: given its result, the forward builds
    no scheduler tensors of its own and computes the same output."""
    torch.manual_seed(0)
    q, k, v, kwargs, plan_kwargs = _scheduler_metadata_case(kind, head_dim)
    kwargs = dict(kwargs, num_splits=0, return_lse=True)
    out_ref, lse_ref, *_ = _flash_attn_fwd(q, k, v, **kwargs)
    metadata = interface.get_scheduler_metadata(
        max_seqlen_q=kwargs["max_seqlen_q"],
        max_seqlen_k=kwargs["max_seqlen_k"],
        nheads=q.shape[1],
        nheads_kv=k.shape[-2],
        headdim=head_dim,
        num_splits=0,
        causal=True,
        cu_seqlens_q=kwargs["cu_seqlens_q"],
        cu_seqlens_k=kwargs.get("cu_seqlens_k"),
        seqused_k=kwargs.get("seqused_k"),
        dtype=q.dtype,
        **plan_kwargs,
    )
    assert metadata.plan_key is not None
    if kind == "one_long":
        assert metadata.num_splits_dynamic_ptr is not None
    if kind == "prefill":
        assert metadata.cu_total_m_blocks is not None

    def unexpected(*args, **kwargs):
        raise AssertionError("the forward rebuilt scheduler metadata it was given")

    monkeypatch.setattr(interface, "_arch_forward_scheduler_metadata", unexpected)
    out, lse, *_ = _flash_attn_fwd(q, k, v, scheduler_metadata=metadata, **kwargs)
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="Runtime metadata check")
def test_sm120_forward_skips_scheduler_metadata_of_another_plan():
    """Metadata planned for a different split budget is not read: the forward warns, builds
    its own, and computes the same output as without metadata."""
    torch.manual_seed(0)
    q, k, v, kwargs, plan_kwargs = _scheduler_metadata_case("one_long", 128)
    kwargs = dict(kwargs, num_splits=0, return_lse=True)
    out_ref, lse_ref, *_ = _flash_attn_fwd(q, k, v, **kwargs)
    metadata = interface.get_scheduler_metadata(
        max_seqlen_q=1,
        max_seqlen_k=kwargs["max_seqlen_k"],
        nheads=q.shape[1],
        nheads_kv=k.shape[-2],
        headdim=128,
        num_splits=1,
        causal=True,
        cu_seqlens_q=kwargs["cu_seqlens_q"],
        seqused_k=kwargs["seqused_k"],
        dtype=q.dtype,
        **plan_kwargs,
    )
    with pytest.warns(UserWarning, match="different forward configuration"):
        out, lse, *_ = _flash_attn_fwd(q, k, v, scheduler_metadata=metadata, **kwargs)
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


def test_sm120_forward_skips_scheduler_metadata_of_another_head_ratio():
    """Metadata built for 4 query heads per KV head has the same tiles and splits as for 8,
    but its tile lookup covers half the packed m blocks: the forward must not read it."""
    torch.manual_seed(0)
    batch_size, seqlen, head_dim, num_kv_heads = interface.BIN_BATCH_SEARCH_THRESH + 44, 64, 128, 2
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, seqlen, device="cuda", dtype=torch.int32)
    q = torch.randn(batch_size * seqlen, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size * seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        window_size_left=32,
        window_size_right=0,
        pack_gqa=True,
        num_splits=1,
        return_lse=True,
    )
    out_ref, lse_ref, *_ = _flash_attn_fwd(q, k, v, **kwargs)
    plan = dict(
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        nheads_kv=num_kv_heads,
        headdim=head_dim,
        num_splits=1,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        window_size_left=32,
        window_size_right=0,
        pack_gqa=True,
        dtype=q.dtype,
        total_q=q.shape[0],
    )
    other = interface.get_scheduler_metadata(nheads=8, **plan)
    own = interface.get_scheduler_metadata(nheads=16, **plan)
    assert other.plan_key[0] == own.plan_key[0] and other.plan_key != own.plan_key
    with pytest.warns(UserWarning, match="different forward configuration"):
        out, lse, *_ = _flash_attn_fwd(q, k, v, scheduler_metadata=other, **kwargs)
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


@pytest.mark.parametrize(
    ("head_dim", "q_len", "window_size", "has_sink"),
    # Every pair of factors appears.
    [
        (64, 1, (None, 0), True),
        (128, 1, (300, 0), False),
        (192, 1, (None, 0), False),
        (64, 4, (300, 0), True),
        (128, 4, (None, 0), True),
        (192, 4, (300, 0), False),
        (64, 1, (300, 0), False),
        (192, 4, (None, 0), True),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_dynamic_splits_match_reference(monkeypatch, head_dim, q_len, window_size, has_sink):
    """Per-batch split counts from the device lengths: a long request splits more than the
    short ones, and splits past a request's own count are skipped (paged; q=1 takes the
    uniform-batch scheduler, q=4 the generic varlen one)."""
    monkeypatch.setattr(Sm120ForwardHost, "dynamic_splits", True)
    # Correctness of the per-batch split path, also where the gate keeps the static plan (a
    # local window bounds the tail, so splitting gains nothing): negative costs force it.
    monkeypatch.setattr(Sm120ForwardHost, "dynamic_split_cost_bytes", -1)
    monkeypatch.setattr(Sm120ForwardHost, "dynamic_split_combine_cost_bytes", -1)
    counts, layouts = [], []
    get_metadata = interface._get_scheduler_metadata

    def spy(*args, **kwargs):
        metadata = get_metadata(*args, **kwargs)
        counts.append(metadata.num_splits_dynamic_ptr)
        layouts.append(metadata.split_slot_batch)
        return metadata

    spy.compile_cache = get_metadata.compile_cache
    monkeypatch.setattr(interface, "_get_scheduler_metadata", spy)
    torch.manual_seed(20261103 + head_dim + q_len)
    num_q_heads, num_kv_heads, page_size = 8, 2, 64
    k_lengths = [8192, 70, 300, 4, 1500, 33, 640, 5]
    batch_size = len(k_lengths)
    k_cache, v_cache, page_table, pages = _paged_cache(
        k_lengths, page_size, num_kv_heads, head_dim
    )
    q = torch.randn(
        batch_size * q_len, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * q_len, q_len, device="cuda", dtype=torch.int32
    )
    seqused_k = torch.tensor(k_lengths, device="cuda", dtype=torch.int32)
    sinks = (
        torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16) if has_sink else None
    )
    output, lse = _varlen_fwd(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=q_len,
        max_seqlen_k=max(k_lengths),
        page_table=page_table,
        window_size=window_size,
        sinks=sinks,
        num_splits=0,
        return_softmax_lse=True,
    )
    assert counts
    # The uniform-batch scheduler launches the compact split layout.
    assert (layouts[-1] is not None) == (q_len == 1)
    if is_fake_mode():
        return
    assert counts[-1].min() < counts[-1].max()
    references = []
    for batch_idx, n in enumerate(k_lengths):
        rows = page_table[batch_idx, : pages[batch_idx]]
        k = k_cache[rows].flatten(0, 1)[:n]
        v = v_cache[rows].flatten(0, 1)[:n]
        references.append(
            _attention_reference(
                q[batch_idx * q_len : (batch_idx + 1) * q_len],
                k,
                v,
                batch_idx=batch_idx,
                window_size=window_size,
                sinks=sinks,
            )
        )
    output_reference = torch.cat([reference[0] for reference in references])
    lse_reference = torch.cat([reference[1] for reference in references], dim=1)
    _assert_attention_close(output, output_reference, lse, lse_reference)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_dynamic_splits_dense_cache_match_reference(monkeypatch, head_dim):
    """Per-batch splits over a dense (batch, seqlen) KV cache: one query row per request
    takes the uniform-batch scheduler and the compact split layout, as with a paged cache."""
    monkeypatch.setattr(Sm120ForwardHost, "dynamic_splits", True)
    monkeypatch.setattr(Sm120ForwardHost, "dynamic_split_cost_bytes", -1)
    monkeypatch.setattr(Sm120ForwardHost, "dynamic_split_combine_cost_bytes", -1)
    layouts = []
    get_metadata = interface._get_scheduler_metadata

    def spy(*args, **kwargs):
        metadata = get_metadata(*args, **kwargs)
        layouts.append(metadata.split_slot_batch)
        return metadata

    spy.compile_cache = get_metadata.compile_cache
    monkeypatch.setattr(interface, "_get_scheduler_metadata", spy)
    torch.manual_seed(20261102 + head_dim)
    num_q_heads, num_kv_heads = 8, 2
    k_lengths = [8192, 70, 300, 4, 1500, 33, 640, 5]
    batch_size, max_k = len(k_lengths), max(k_lengths)
    q = torch.randn(batch_size, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        batch_size, max_k, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    cu_seqlens_q = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor(k_lengths, device="cuda", dtype=torch.int32)
    output, lse = _varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=max_k,
        causal=True,
        num_splits=0,
        return_softmax_lse=True,
    )
    assert layouts and layouts[-1] is not None
    if is_fake_mode():
        return
    references = [
        _attention_reference(q[i : i + 1], k[i, :n], v[i, :n], batch_idx=i)
        for i, n in enumerate(k_lengths)
    ]
    output_reference = torch.cat([reference[0] for reference in references])
    lse_reference = torch.cat([reference[1] for reference in references], dim=1)
    _assert_attention_close(output, output_reference, lse, lse_reference)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_packed_gqa_score_mod_matches_reference():
    """A score_mod sees the logical Q head and Q index of each packed-GQA row."""
    torch.manual_seed(20260804)
    q_lengths = (17, 35)
    k_lengths = (79, 143)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 64
    q_parts = [
        torch.randn(n, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        for n in q_lengths
    ]
    k_parts = [
        torch.randn(n, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        for n in k_lengths
    ]
    v_parts = [torch.randn_like(k) for k in k_parts]
    cu_seqlens_q = torch.tensor([0, q_lengths[0], sum(q_lengths)], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, k_lengths[0], sum(k_lengths)], device="cuda", dtype=torch.int32)
    output, lse = _varlen_fwd(
        torch.cat(q_parts),
        torch.cat(k_parts),
        torch.cat(v_parts),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=True,
        pack_gqa=True,
        score_mod=_coordinate_score_mod,
        return_softmax_lse=True,
    )
    if is_fake_mode():
        return
    references = [
        _attention_reference(q, k, v, batch_idx=batch_idx, causal=True, score_mod="coordinate")
        for batch_idx, (q, k, v) in enumerate(zip(q_parts, k_parts, v_parts))
    ]
    output_reference = torch.cat([reference[0] for reference in references])
    lse_reference = torch.cat([reference[1] for reference in references], dim=1)
    _assert_attention_close(output, output_reference, lse, lse_reference)


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "head_dim_v",
        "q_len",
        "page_size",
        "causal",
        "window_size",
        "pack_gqa",
        "num_splits",
    ),
    [
        pytest.param(
            torch.float16,
            4,
            4,
            32,
            32,
            7,
            64,
            True,
            (None, None),
            False,
            1,
            id="fp16-mha-hd32-page64-causal",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            64,
            96,
            33,
            128,
            False,
            (127, 15),
            True,
            2,
            id="bf16-gqa-hd64-hdv96-page128-local-splitkv",
        ),
        pytest.param(
            torch.bfloat16,
            6,
            1,
            128,
            64,
            5,
            64,
            False,
            (None, None),
            None,
            2,
            id="bf16-mqa-hd128-hdv64-page64-global-splitkv",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            96,
            128,
            17,
            32,
            True,
            (None, None),
            False,
            1,
            id="bf16-gqa-hd96-hdv128-page32-causal-unpacked",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            2,
            64,
            64,
            1,
            64,
            True,
            (None, None),
            True,
            0,
            id="bf16-gqa-hd64-page64-decode-auto-split",
        ),
        pytest.param(
            torch.bfloat16,
            8,
            1,
            128,
            128,
            1,
            64,
            True,
            (None, None),
            True,
            0,
            id="bf16-mqa-hd128-page64-decode-auto-split",
        ),
        pytest.param(
            torch.float16,
            8,
            1,
            256,
            256,
            1,
            64,
            True,
            (None, None),
            True,
            0,
            id="fp16-mqa-hd256-page64-decode-auto-split",
        ),
        pytest.param(
            torch.float16,
            8,
            2,
            128,
            128,
            7,
            16,
            True,
            (63, 0),
            False,
            2,
            id="fp16-gqa-hd128-page16-local-split2",
        ),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_paged_kv_feature_matrix_matches_reference(
    dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    q_len,
    page_size,
    causal,
    window_size,
    pack_gqa,
    num_splits,
):
    """Cover paged MHA/GQA/MQA outside the HD256 decode specialization."""
    torch.manual_seed(20260807 + head_dim + head_dim_v + page_size)
    batch_size, pages_per_sequence = 2, 6
    max_seqlen_k = pages_per_sequence * page_size
    k_lengths = (max_seqlen_k - page_size - 11, max_seqlen_k - 1)
    q = torch.randn(
        batch_size,
        q_len,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    k_cache = torch.randn(
        batch_size * pages_per_sequence,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.randn(
        batch_size * pages_per_sequence,
        page_size,
        num_kv_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )
    page_table = (
        torch.randperm(
            batch_size * pages_per_sequence,
            device="cuda",
            dtype=torch.int64,
        )
        .to(torch.int32)
        .view(batch_size, pages_per_sequence)
    )
    seqused_k = torch.tensor(k_lengths, device="cuda", dtype=torch.int32)
    output, lse = _varlen_fwd(
        q,
        k_cache,
        v_cache,
        seqused_k=seqused_k,
        max_seqlen_q=q_len,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        causal=causal,
        window_size=window_size,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_softmax_lse=True,
    )
    references = []
    for batch_idx, k_len in enumerate(k_lengths):
        pages = page_table[batch_idx]
        k = k_cache.index_select(0, pages).flatten(0, 1)[:k_len]
        v = v_cache.index_select(0, pages).flatten(0, 1)[:k_len]
        references.append(
            _attention_reference(
                q[batch_idx],
                k,
                v,
                batch_idx=batch_idx,
                causal=causal,
                window_size=window_size,
            )
        )
    output_reference = torch.stack([reference[0] for reference in references])
    lse_reference = torch.stack([reference[1] for reference in references])
    _assert_attention_close(output, output_reference, lse, lse_reference)

    if num_splits > 1:
        unsplit_output = _varlen_fwd(
            q,
            k_cache,
            v_cache,
            seqused_k=seqused_k,
            max_seqlen_q=q_len,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            causal=causal,
            window_size=window_size,
            num_splits=1,
            pack_gqa=pack_gqa,
        )
        _assert_close(output, unsplit_output, atol=4e-3, rtol=0.0)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_kv_heads", [8])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_low_hd_paged_decode_is_reproducible(
    head_dim,
    num_kv_heads,
):
    """Low-HD MHA/GQA/MQA decode with a preallocated output is run-to-run exact."""
    torch.manual_seed(20260731 + head_dim + num_kv_heads)
    batch_size, num_q_heads = 2, 8
    max_seqlen, page_size = 512, 64
    pages_per_request = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        batch_size * pages_per_request,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(
        batch_size * pages_per_request,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_request)
    cache_seqlens = torch.full(
        (batch_size,),
        max_seqlen,
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.arange(
        batch_size + 1,
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty_like(q)

    def run():
        return _paged_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=0,
            pack_gqa=None,
            out=out,
        )

    first = run().clone()
    second = run().clone()
    torch.cuda.synchronize()

    _assert_close(first, second, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("window_left", [None, 250])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_varlen_mqa_hd256_learnable_sink(window_left):
    """Cover Q6/KV1 head-dim-256 global and local prefill shapes."""
    torch.manual_seed(1234)
    seq, num_q_heads, head_dim = 512, 6, 256
    q = torch.randn(seq, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    window_size = (None, None) if window_left is None else (window_left, 0)
    out_ref, lse_ref = _attention_reference(
        q, k, v, causal=True, window_size=(window_left, None), sinks=sinks
    )

    outputs = []
    for pack_gqa in (False, True, None):
        out, lse = _varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq,
            max_seqlen_k=seq,
            softmax_scale=head_dim**-0.5,
            causal=True,
            window_size=window_size,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert is_fake_mode() or output_error.max().item() < 1e-2
        assert is_fake_mode() or output_error.mean().item() < 5e-4
        assert is_fake_mode() or lse_error.max().item() < 5e-5
        outputs.append(out)

    _assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
    _assert_close(outputs[1], outputs[2], atol=0.0, rtol=0.0)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_varlen_padding_ctas_are_inert_across_tile_specializations():
    """Padding CTAs must not consume stale SMEM or write another batch's output."""
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_q_heads, head_dim = 6, 256

    def make_inputs(lengths, seed):
        torch.manual_seed(seed)
        total = sum(lengths)
        q = torch.randn(
            total,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn(total, 1, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
        cuts = [0]
        for length in lengths:
            cuts.append(cuts[-1] + length)
        cu_seqlens = torch.tensor(cuts, device="cuda", dtype=torch.int32)
        return q, k, v, sinks, cu_seqlens

    def run(inputs, lengths, pack_gqa):
        q, k, v, sinks, cu_seqlens = inputs
        return _varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            softmax_scale=head_dim**-0.5,
            causal=True,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )

    # Run the three SM-normalized selector regions before the padding case, so stale shared
    # memory left by earlier tiles would show.
    rows_per_sm = (58, 84, 100)
    for seed, ratio in enumerate(rows_per_sm):
        seq = math.ceil(ratio * num_sms / num_q_heads)
        inputs = make_inputs([seq], seed)
        for pack_gqa in (False, True, False):
            run(inputs, [seq], pack_gqa)
        torch.cuda.synchronize()

    # At 64 query rows/SM, two batches select M64 while the conservative
    # varlen grid contains padding CTAs.
    seq = math.ceil(32 * num_sms / num_q_heads)
    lengths = [seq, seq]
    inputs = make_inputs(lengths, 10)
    q, k, v, sinks, _ = inputs
    references = [
        _attention_reference(
            q[start : start + seq],
            k[start : start + seq],
            v[start : start + seq],
            causal=True,
            window_size=(None, None),
            sinks=sinks,
        )
        for start in (0, seq)
    ]
    out_ref = torch.cat([reference[0] for reference in references], dim=0)
    lse_ref = torch.cat([reference[1] for reference in references], dim=1)

    for pack_gqa in (False, True, None):
        out, lse = run(inputs, lengths, pack_gqa)
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert is_fake_mode() or output_error.max().item() < 1e-2
        assert is_fake_mode() or output_error.mean().item() < 5e-4
        assert is_fake_mode() or lse_error.max().item() < 5e-5


@pytest.mark.parametrize("window_left", [None, 192])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_paged_decode_ragged_splits_match_reference(
    window_left,
):
    """Ragged lengths at tile edges through the split-QK-N decode, with a sink and a local
    window; uniform and ragged calls share one compiled kernel."""
    torch.manual_seed(1234)
    batch_size, num_q_heads, head_dim = 4, 6, 256
    max_seqlen, page_size = 1024, 64
    pages_per_request = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        batch_size * pages_per_request,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(
        batch_size * pages_per_request,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_request)
    cu_seqlens_q = torch.arange(
        batch_size + 1,
        device="cuda",
        dtype=torch.int32,
    )
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run(lengths):
        return _paged_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=torch.tensor(
                lengths,
                device="cuda",
                dtype=torch.int32,
            ),
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=((None, None) if window_left is None else (window_left, 0)),
            num_splits=4,
            pack_gqa=True,
            sinks=sinks,
        )

    uniform_lengths = [max_seqlen] * batch_size
    # Exercise both sides of every 64-token tile boundary. N-distributed QK
    # has a different accumulator-column layout, so its mask must not use the
    # ordinary QK path's R2P column mapping.
    ragged_lengths = [1000, 513, 257, 63]
    uniform_first = run(uniform_lengths)
    ragged_first = run(ragged_lengths)
    ragged_second = run(ragged_lengths)
    uniform_second = run(uniform_lengths)

    _assert_close(ragged_first, ragged_second, atol=0.0, rtol=0.0)
    _assert_close(uniform_first, uniform_second, atol=0.0, rtol=0.0)

    reference = torch.empty_like(ragged_first, dtype=torch.float32)
    scale = head_dim**-0.5
    for batch_idx, length in enumerate(ragged_lengths):
        pages = page_table[batch_idx]
        start = 0 if window_left is None else max(0, length - 1 - window_left)
        k = k_cache.index_select(0, pages).flatten(0, 1)[start:length, 0].float()
        v = v_cache.index_select(0, pages).flatten(0, 1)[start:length, 0].float()
        scores = q[batch_idx].float() @ k.T * scale
        row_max = torch.maximum(scores.amax(dim=-1), sinks.float())
        weights = torch.exp(scores - row_max[:, None])
        denominator = weights.sum(dim=-1) + torch.exp(sinks.float() - row_max)
        reference[batch_idx] = weights @ v / denominator[:, None]

    error = (ragged_first.float() - reference).abs()
    assert is_fake_mode() or error.max().item() < 1e-2
    assert is_fake_mode() or error.mean().item() < 5e-4


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_paged_decode_transpose_matches_reference():
    """Gather and page-TMA transpose must not share a compiled specialization."""
    torch.manual_seed(20260729)
    batch_size, num_q_heads, head_dim = 1, 6, 256
    max_seqlen, page_size = 2048, 64
    num_pages = max_seqlen // page_size
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(
        num_pages,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)[None]
    cache_seqlens = torch.tensor(
        [max_seqlen],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run(num_splits):
        return _paged_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=num_splits,
            pack_gqa=True,
            sinks=sinks,
        )

    # S32 owns one KV tile per CTA and selects gather; S16 owns two and selects the
    # transposed page-TMA kernel. Alternate both compile-cache entries.
    normal_first = run(32)
    transpose_after = run(16)
    transpose_repeat = run(16)
    normal_after = run(32)
    torch.cuda.synchronize()

    _assert_close(normal_first, normal_after, atol=0.0, rtol=0.0)
    _assert_close(
        transpose_after,
        transpose_repeat,
        atol=0.0,
        rtol=0.0,
    )
    _assert_close(
        normal_first,
        transpose_after,
        atol=2e-3,
        rtol=0.0,
    )

    pages = page_table[0]
    k = k_cache.index_select(0, pages).flatten(0, 1)[:, 0].float()
    v = v_cache.index_select(0, pages).flatten(0, 1)[:, 0].float()
    scores = q[0].float() @ k.T * head_dim**-0.5
    row_max = torch.maximum(scores.amax(dim=-1), sinks.float())
    weights = torch.exp(scores - row_max[:, None])
    denominator = weights.sum(dim=-1) + torch.exp(sinks.float() - row_max)
    reference = weights @ v / denominator[:, None]
    error = (transpose_after[0].float() - reference).abs()
    assert is_fake_mode() or error.max().item() < 1e-2
    assert is_fake_mode() or error.mean().item() < 5e-4


@pytest.mark.parametrize(
    ("num_q_heads", "pack_gqa", "expected_transpose", "expected_split_qk"),
    [
        pytest.param(6, True, True, False, id="transpose"),
        pytest.param(16, True, False, True, id="split-qk"),
        pytest.param(1, None, False, True, id="auto-mha-split-qk"),
        pytest.param(6, False, False, False, id="single-qk"),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_paged_decode_graph_is_correct_and_eager_reusable(
    monkeypatch,
    num_q_heads,
    pack_gqa,
    expected_transpose,
    expected_split_qk,
):
    """Every SplitKV decode dataflow replays correctly from a CUDA graph and leaves later
    eager calls unchanged."""
    torch.manual_seed(20260730)
    batch_size, head_dim = 1, 256
    max_seqlen, page_size = 2048, 64
    num_pages = max_seqlen // page_size
    captured_plans = []
    original_resolve_plan = Sm120ForwardHost.resolve_plan

    def recording_resolve_plan(**kwargs):
        plan = original_resolve_plan(**kwargs)
        if torch.cuda.is_current_stream_capturing():
            captured_plans.append(plan)
        return plan

    monkeypatch.setattr(
        Sm120ForwardHost,
        "resolve_plan",
        staticmethod(recording_resolve_plan),
    )
    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.randperm(
        num_pages,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)[None]
    cache_seqlens = torch.tensor(
        [max_seqlen],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)

    def run():
        return _paged_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen,
            causal=True,
            num_splits=16,
            pack_gqa=pack_gqa,
            sinks=sinks,
        )

    eager_before = run()
    if is_fake_mode():
        # Compilation pass: plan and compile the captured launch without a real capture.
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        run()
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run()
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()
        eager_after = run()
        torch.cuda.synchronize()

        _assert_close(eager_before, eager_after, atol=0.0, rtol=0.0)
        _assert_close(graph_output, eager_before, atol=2e-3, rtol=0.0)
    assert captured_plans
    assert all(plan.num_splits > 1 for plan in captured_plans)
    assert all(plan.transpose_qk_pv is expected_transpose for plan in captured_plans)
    assert all(plan.split_qk_n is expected_split_qk for plan in captured_plans)


def _block_sparse_reference(q, k, v, keep):
    qhead_per_kvhead = q.shape[2] // k.shape[2]
    k, v = (t.repeat_interleave(qhead_per_kvhead, dim=2).float() for t in (k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(scores, -1)
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, -1).nan_to_num(0.0), v)
    return out, lse


_SPARSE_BLOCKS = {"b128": (128, 128), "b64x128": (64, 128), "b128x64": (128, 64)}


@pytest.mark.parametrize(
    ("mask_name", "head_dim", "heads", "block"),
    # Every head dim, head layout and block shape on one mask (with partial and full blocks),
    # then the other masks once.
    [
        ("sliding_window", d, h, b)
        for d in (64, 128, 192, 256)
        for h in ((8, 8), (8, 2))
        for b in _SPARSE_BLOCKS
    ]
    + [
        (m, 128, (8, 2), "b128")
        for m in ("block_diagonal", "mini_causal", "prefix_lm", "dilated_sliding_window")
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fwd_block_sparse(mask_name, head_dim, heads, block):
    """The first Q block sees no KV block; the others take the mask's partial and full
    blocks."""
    _check_fwd_block_sparse(mask_name, head_dim, heads, _SPARSE_BLOCKS[block], True)


@pytest.mark.parametrize(
    ("layout", "head_dim", "mask_name"),
    [
        (layout, head_dim, ("block_diagonal", "sliding_window", "prefix_lm")[(i + j) % 3])
        for i, layout in enumerate(("split3", "paged64", "paged16", "paged16_split3"))
        for j, head_dim in enumerate((64, 128, 256))
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fwd_block_sparse_split_paged(layout, head_dim, mask_name):
    """Block-sparse forward with SplitKV (each split runs a contiguous share of the lists)
    and paged KV (TMA pages of tile_n rows, cp.async pages of 16)."""
    page = int(layout.split("paged")[1].split("_")[0]) if "paged" in layout else 0
    num_splits = 3 if "split3" in layout else 1
    _check_fwd_block_sparse(
        mask_name, head_dim, (8, 2), (128, 128), True, page_size=page, num_splits=num_splits
    )


@pytest.mark.parametrize("layout", ["dense", "paged16_split3"])
@pytest.mark.parametrize(("head_dim", "mask_name"), [(64, "mini_causal"), (256, "sliding_window")])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm120_fwd_block_sparse_fp16(head_dim, mask_name, layout):
    paged = layout != "dense"
    _check_fwd_block_sparse(
        mask_name, head_dim, (8, 2), (128, 128), True, page_size=16 if paged else 0,
        num_splits=3 if paged else 1, dtype=torch.float16,
    )


def _check_fwd_block_sparse(
    mask_name, head_dim, heads, block_size, empty_q_block, page_size=0, num_splits=1,
    dtype=torch.bfloat16,
):
    """Block-sparse forward (KV lists from create_block_mask, partial and full blocks) against
    the reference. Q 1000 / K 1100: the last KV block is cut at seqlen_k; with empty_q_block
    the first Q block sees no KV block (out 0, lse -inf)."""
    from mask_mod_definitions import get_mask_pair
    from torch.nn.attention.flex_attention import create_block_mask

    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    batch, seqlen_q, seqlen_k = 2, 1000, 1100
    cute_mask, flex_mask = get_mask_pair(mask_name, seqlen_q, seqlen_k, window_size=200)
    block_mask = create_block_mask(
        flex_mask, batch, heads[0], seqlen_q, seqlen_k, device="cuda", BLOCK_SIZE=block_size
    )
    cnt, full_cnt = block_mask.kv_num_blocks.clone(), block_mask.full_kv_num_blocks.clone()
    if empty_q_block:
        cnt[:, :, 0] = 0
        full_cnt[:, :, 0] = 0
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=cnt,
        mask_block_idx=block_mask.kv_indices,
        full_block_cnt=full_cnt,
        full_block_idx=block_mask.full_kv_indices,
        block_size=block_size,
    )
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, heads[0], head_dim, device="cuda", dtype=dtype)
    k, v = (
        torch.randn(batch, seqlen_k, heads[1], head_dim, device="cuda", dtype=dtype)
        for _ in range(2)
    )
    kv_kwargs = {}
    k_in, v_in = k, v
    if page_size:
        num_pages = -(-seqlen_k // page_size)
        perm = torch.randperm(batch * num_pages, device="cuda")
        pad = num_pages * page_size - seqlen_k
        k_in, v_in = (
            torch.empty(batch * num_pages, page_size, heads[1], head_dim, device="cuda", dtype=t.dtype)
            for t in (k, v)
        )
        for dst, src in ((k_in, k), (v_in, v)):
            dst[perm] = torch.nn.functional.pad(src, (0, 0, 0, 0, 0, pad)).reshape(dst.shape)
        kv_kwargs = dict(
            page_table=perm.view(batch, num_pages).to(torch.int32),
            seqused_k=torch.full((batch,), seqlen_k, dtype=torch.int32, device="cuda"),
        )
    out, lse = _flash_attn_fwd(
        q,
        k_in,
        v_in,
        mask_mod=cute_mask,
        block_sparse_tensors=sparse,
        return_lse=True,
        num_splits=num_splits,
        **kv_kwargs,
    )[:2]
    if is_fake_mode():
        return
    rows = torch.arange(seqlen_q, device="cuda")[:, None]
    keep = flex_mask(0, 0, rows, torch.arange(seqlen_k, device="cuda")[None, :])
    keep = keep.expand(seqlen_q, seqlen_k).clone()
    if empty_q_block:
        keep[: block_size[0]] = False
    ref_out, ref_lse = _block_sparse_reference(q, k, v, keep)
    assert (out.float() - ref_out).abs().max() <= 0.02 * ref_out.abs().max()
    visible = torch.isfinite(ref_lse)
    torch.testing.assert_close(lse[visible], ref_lse[visible], atol=0.05, rtol=0)
    assert torch.isneginf(lse[~visible]).all()
    assert (out.permute(0, 2, 1, 3)[~visible] == 0).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
