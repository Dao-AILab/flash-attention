"""Paged SwapAB score coordinates, checked against an independent FP32 oracle.

Single-pass GPU tests (not fake-tensor compilation tests). Supported cases force
SwapAB; only the explicit unsupported-page tests permit automatic fallback.
"""

import math

import cutlass
from cutlass import cute
import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="requires SM100 for paged SwapAB score modifiers",
)


@cute.jit
def _token_head_bias(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    bi = cute.make_rmem_tensor(1, cutlass.Int32)
    hi = cute.make_rmem_tensor(1, cutlass.Int32)
    ki = cute.make_rmem_tensor(1, cutlass.Int32)
    bi.store(b)
    hi.store(h)
    ki.store(kv_idx)
    bias = cute.make_rmem_tensor(1, cutlass.Float32)
    bias[0] = aux_tensors[0][bi[0], hi[0], ki[0]]
    return score + bias.load()


@cute.jit
def _head_temperature(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    hi = cute.make_rmem_tensor(1, cutlass.Int32)
    hi.store(h)
    temperature = cute.make_rmem_tensor(1, cutlass.Float32)
    temperature[0] = aux_tensors[1][hi[0]]
    return score * temperature.load()


@cute.jit
def _end_relative_alibi(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    bi = cute.make_rmem_tensor(1, cutlass.Int32)
    hi = cute.make_rmem_tensor(1, cutlass.Int32)
    bi.store(b)
    hi.store(h)
    slope = cute.make_rmem_tensor(1, cutlass.Float32)
    end = cute.make_rmem_tensor(1, cutlass.Float32)
    slope[0] = aux_tensors[2][hi[0]]
    end[0] = cutlass.Float32(aux_tensors[3][bi[0]] - 1)
    # Decode q_idx is zero, NOT the absolute query position at used_length-1.
    return score + slope.load() * (kv_idx.to(cutlass.Float32) - end.load())


@cute.jit
def _selected_heads_mask(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
    # All tokens removed from heads 1 and 4, partial token mask on other heads.
    return (h % 3 != 1) & (kv_idx % 5 != 2)


_MODIFIERS = {
    "token_head_bias": _token_head_bias,
    "head_temperature": _head_temperature,
    "end_relative_alibi": _end_relative_alibi,
}


@pytest.fixture(autouse=True)
def _force_swapab(monkeypatch):
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "swapab")


def _inputs(page_size, seed, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.bfloat16
        )

    q = randn(2, 1, 6, 96)
    # Non-contiguous head strides; nine physical pages, shared page 7, ratio 3.
    k = randn(9, page_size, 4, 96)[:, :, ::2]
    v = randn(9, page_size, 4, 64)[:, :, ::2]
    table = torch.tensor([[7, 2, 5], [1, 7, 3]], device=device, dtype=torch.int32)
    used = torch.tensor(
        [page_size + 17, 3 * page_size - 9], device=device, dtype=torch.int32
    )
    # Extra allocation covers padded token lanes and the deliberately wrong
    # physical-coordinate oracle used by the sensitivity controls.
    bias = torch.randn(2, 6, 9 * page_size, generator=generator, device=device) * 0.65
    temperatures = torch.tensor([0.55, 0.8, 1.1, 1.35, 1.65, 1.95], device=device)
    slopes = torch.tensor([0.001, 0.002, 0.004, 0.006, 0.009, 0.013], device=device)
    return q, k, v, table, used, (bias, temperatures, slopes, used)


def _oracle(q, k, v, table, used, aux, kind, *, masked=False, wrong=None):
    """Gather logical K/V first; all subsequent math is independent FP32 torch."""
    batch, _, hq, dim = q.shape
    page_size, hkv = k.shape[1:3]
    logical_k = k[table.long()].reshape(batch, -1, hkv, dim).float()
    logical_v = v[table.long()].reshape(batch, -1, hkv, v.shape[-1]).float()
    kv_heads = torch.arange(hq, device=q.device) // (hq // hkv)
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), logical_k[:, :, kv_heads])
    logits = logits * (1.0 / math.sqrt(dim))
    tokens = torch.arange(logical_k.shape[1], device=q.device)
    heads = torch.arange(hq, device=q.device)
    mod_heads = kv_heads if wrong == "kv_head" else heads
    mod_tokens = tokens[None, :].expand(batch, -1)
    if wrong == "physical_token":
        mod_tokens = (
            table.long()[:, tokens // page_size] * page_size + tokens % page_size
        )
    if kind == "token_head_bias":
        bias = aux[0][
            torch.arange(batch, device=q.device)[:, None, None],
            mod_heads[None, :, None],
            mod_tokens[:, None, :],
        ]
        if wrong == "constant_token":
            bias = bias[:, :, :1].expand_as(bias)
        logits = logits + bias[:, :, None, :]
    elif kind == "head_temperature":
        logits = logits * aux[1][mod_heads][None, :, None, None]
    elif kind == "end_relative_alibi":
        end = used.float() - 1
        if wrong == "decode_query_zero":
            end = torch.zeros_like(end)
        relative = mod_tokens.float() - end[:, None]
        logits = (
            logits + aux[2][mod_heads][None, :, None, None] * relative[:, None, None, :]
        )
    else:
        raise AssertionError(kind)
    keep = tokens[None, None, None, :] < used[:, None, None, None]
    if masked:
        keep = (
            keep
            & (heads[None, :, None, None] % 3 != 1)
            & (tokens[None, None, None, :] % 5 != 2)
        )
    logits = logits.masked_fill(~keep, -torch.inf)
    lse = torch.logsumexp(logits, dim=-1)  # Natural log, not log2.
    probabilities = torch.softmax(logits, dim=-1).nan_to_num()
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, logical_v[:, :, kv_heads])
    return output, lse


def _assert_result(result, expected):
    output, lse = result[:2]
    ref_output, ref_lse = expected
    torch.testing.assert_close(output.float(), ref_output, atol=0.004, rtol=0.025)
    torch.testing.assert_close(lse, ref_lse, atol=0.004, rtol=0.003)
    empty = torch.isneginf(ref_lse)
    assert torch.equal(torch.isneginf(lse), empty)
    # Empty batches / fully masked heads must be exactly zero, not merely close.
    assert torch.count_nonzero(output.transpose(1, 2)[empty]) == 0


def _assert_sensitive(args, kind):
    expected = _oracle(*args, kind)
    controls = ["kv_head"]
    if kind != "head_temperature":
        controls.append("physical_token")
    controls.append(
        "constant_token"
        if kind == "token_head_bias"
        else "decode_query_zero"
        if kind == "end_relative_alibi"
        else "kv_head"
    )
    for wrong in set(controls):
        alternate = _oracle(*args, kind, wrong=wrong)
        assert any(
            not torch.allclose(a, b, atol=0.01, rtol=0.01)
            for a, b in zip(expected, alternate)
        ), f"insensitive oracle: {kind}/{wrong}"
    return expected


def _run(args, kind, num_splits, *, masked=False):
    q, k, v, table, used, aux = args
    return _flash_attn_fwd(
        q,
        k,
        v,
        page_table=table,
        seqused_k=used,
        aux_tensors=list(aux),
        score_mod=_MODIFIERS[kind],
        mask_mod=_selected_heads_mask if masked else None,
        num_splits=num_splits,
        return_lse=True,
    )[:2]


_STATIC_CASES = [
    pytest.param(kind, page, splits, id=f"{kind}-page{page}-split{splits}")
    for kind in _MODIFIERS
    for page in (256, 512)
    for splits in (1, 3)
]


@pytest.mark.parametrize("kind,page_size,num_splits", _STATIC_CASES)
def test_paged_score_mod(kind, page_size, num_splits):
    seed = 1100 + page_size + num_splits + 101 * tuple(_MODIFIERS).index(kind)
    args = _inputs(page_size, seed)
    expected = _assert_sensitive(args, kind)
    _assert_result(_run(args, kind, num_splits), expected)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "num_splits", [pytest.param(1, id="split1"), pytest.param(3, id="split3")]
)
def test_paged_score_mod_graph(num_splits):
    args = _inputs(256, 2200 + num_splits)
    _, k, v, table, used, aux = args

    # Combined token/head bias and temperature in one callback, so replay must
    # observe both auxiliary tensors rather than a baked-in host snapshot.
    @cute.jit
    def combined(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
        scaled = _head_temperature(score, b, h, q_idx, kv_idx, seqlen_info, aux_tensors)
        return _token_head_bias(scaled, b, h, q_idx, kv_idx, seqlen_info, aux_tensors)

    def run():
        q, k, v, table, used, aux = args
        return _flash_attn_fwd(
            q,
            k,
            v,
            page_table=table,
            seqused_k=used,
            aux_tensors=list(aux),
            score_mod=combined,
            num_splits=num_splits,
            return_lse=True,
        )[:2]

    # Initialize JIT/cache on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for iteration, (mapping, lengths) in enumerate(
        [
            ([[7, 2, 5], [1, 7, 3]], [273, 759]),
            ([[4, 0, 8], [8, 6, 4]], [0, 513]),
            ([[8, 5, 0], [2, 8, 1]], [529, 17]),
        ]
    ):
        table.copy_(torch.tensor(mapping, device=table.device, dtype=table.dtype))
        used.copy_(torch.tensor(lengths, device=used.device, dtype=used.dtype))
        aux[0].mul_(-0.7).add_(0.13 * iteration)
        aux[1].mul_(0.85).add_(0.11)
        k.mul_(-0.8).add_(0.03)
        v.mul_(-0.6).add_(0.07)
        graph.replay()
        # Scale Q in the independent oracle to express score*temperature+bias.
        q_scaled = args[0].float() * aux[1][None, None, :, None]
        expected = _oracle(q_scaled, *args[1:], "token_head_bias")
        _assert_result(result, expected)
    torch.cuda.synchronize()


def test_paged_score_mod_oversplit_empty_and_masked():
    args = _inputs(256, 3308)
    args[4].copy_(torch.tensor([0, 17], device=args[4].device, dtype=torch.int32))
    expected = _oracle(*args, "token_head_bias", masked=True)
    _assert_result(_run(args, "token_head_bias", 8, masked=True), expected)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "page_size", [pytest.param(128, id="page128"), pytest.param(384, id="page384")]
)
def test_paged_score_mod_unsupported_page(page_size, monkeypatch):
    args = _inputs(page_size, 4400 + page_size)
    with pytest.raises(ValueError, match="SwapAB unsupported"):
        _run(args, "token_head_bias", 3)
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "auto")
    auto = _run(args, "token_head_bias", 3)
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "standard")
    standard = _run(args, "token_head_bias", 3)
    expected = _assert_sensitive(args, "token_head_bias")
    _assert_result(auto, expected)
    _assert_result(standard, expected)
    for actual, reference in zip(auto, standard):
        torch.testing.assert_close(actual, reference)
    torch.cuda.synchronize()
