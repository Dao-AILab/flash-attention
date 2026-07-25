import math

import pytest
import torch
from score_mod_definitions import score_mod_times_two

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.interface import _flash_attn_fwd

COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]
pytestmark = pytest.mark.skipif(COMPUTE_CAPABILITY != 10, reason="SM100-only tests")


def reference_attention(q, k, v, *, score_transform=None, seqused_k=None):
    """Compute non-causal GQA attention in FP32."""
    qhead_per_kvhead = q.shape[2] // k.shape[2]
    if qhead_per_kvhead > 1:
        k = k.repeat_interleave(qhead_per_kvhead, dim=2)
        v = v.repeat_interleave(qhead_per_kvhead, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(
        q.shape[-1]
    )
    if score_transform is not None:
        scores = score_transform(scores)
    if seqused_k is not None:
        kv_idx = torch.arange(scores.shape[-1], device=scores.device)
        scores = scores.masked_fill(
            kv_idx >= seqused_k[:, None, None, None], float("-inf")
        )
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v.float())
    return out, lse


def assert_matches_reference(out, lse, out_ref, lse_ref):
    """Check tile_m=64 output and LSE against an FP32 reference."""
    torch.testing.assert_close(out.float(), out_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(lse.float(), lse_ref, rtol=0, atol=2e-2)


@pytest.mark.parametrize(
    "dtype,head_dim,head_dim_v,num_heads,num_kv_heads,tile_n,pack_gqa",
    [
        pytest.param(torch.bfloat16, 8, 8, 2, 2, 64, False, id="bf16-min-d8"),
        pytest.param(torch.bfloat16, 128, 128, 2, 2, 64, False, id="bf16-tail-n64"),
        pytest.param(torch.float16, 96, 96, 2, 2, 128, False, id="fp16-d96-n128"),
        pytest.param(torch.bfloat16, 24, 40, 2, 2, 64, False, id="padded-d24-dv40"),
        pytest.param(torch.bfloat16, 64, 128, 2, 2, 128, False, id="d64-dv128"),
        pytest.param(torch.bfloat16, 128, 64, 2, 2, 64, False, id="d128-dv64"),
        pytest.param(torch.bfloat16, 192, 128, 2, 2, 128, False, id="deepseek"),
        pytest.param(torch.bfloat16, 128, 128, 4, 2, 64, True, id="packed-gqa"),
        pytest.param(torch.bfloat16, 64, 64, 4, 1, 64, False, id="unpacked-mqa"),
    ],
)
def test_sm100_forward_tile_m64_dense(
    dtype,
    head_dim,
    head_dim_v,
    num_heads,
    num_kv_heads,
    tile_n,
    pack_gqa,
):
    torch.manual_seed(0)
    q = torch.randn((1, 129, num_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((1, 193, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((1, 193, num_kv_heads, head_dim_v), device="cuda", dtype=dtype)

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        tile_mn=(64, tile_n),
        pack_gqa=pack_gqa,
        return_lse=True,
    )
    out_ref, lse_ref = reference_attention(q, k, v)

    assert_matches_reference(out, lse, out_ref, lse_ref)


def test_sm100_forward_tile_m64_block_sparse_empty_tiles():
    """Keep empty sparse tiles on the CTA-wide softmax-stat barrier."""
    torch.manual_seed(6)
    q = torch.randn((1, 65, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    block_count = torch.zeros((1, 1, 2), device="cuda", dtype=torch.int32)
    block_index = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=block_count,
        mask_block_idx=block_index,
        full_block_cnt=block_count,
        full_block_idx=block_index,
        block_size=(64, 64),
    )

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        tile_mn=(64, 64),
        block_sparse_tensors=sparse,
        return_lse=True,
    )

    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()


def test_sm100_forward_tile_m64_block_sparse_nonempty_to_empty():
    """Do not read a NaN-poisoned accumulator when the next sparse tile is empty."""
    torch.manual_seed(8)
    num_q_tiles = torch.cuda.get_device_properties(0).multi_processor_count + 1
    q = torch.randn((1, num_q_tiles * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    v[:, 0] = torch.nan
    mask_count = torch.zeros((1, 1, num_q_tiles), device="cuda", dtype=torch.int32)
    block_index = torch.zeros((1, 1, num_q_tiles, 1), device="cuda", dtype=torch.int32)
    full_count = mask_count.clone()
    full_count[..., 0] = 1
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=mask_count,
        mask_block_idx=block_index,
        full_block_cnt=full_count,
        full_block_idx=block_index,
        block_size=(64, 64),
    )

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        tile_mn=(64, 64),
        block_sparse_tensors=sparse,
        return_lse=True,
    )

    assert torch.isnan(out[:, :64]).all()
    assert torch.count_nonzero(out[:, -64:]) == 0
    assert torch.isneginf(lse[..., -64:]).all()


def test_sm100_forward_tile_m64_block_sparse_full_tail():
    """Mask K-tail columns when a sparse block is classified as full."""
    torch.manual_seed(7)
    q = torch.randn((1, 65, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask_count = torch.zeros((1, 1, 2), device="cuda", dtype=torch.int32)
    block_index = torch.ones((1, 1, 2, 1), device="cuda", dtype=torch.int32)
    sparse = BlockSparseTensorsTorch(
        mask_block_cnt=mask_count,
        mask_block_idx=block_index,
        full_block_cnt=torch.ones_like(mask_count),
        full_block_idx=block_index,
        block_size=(64, 64),
    )

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        tile_mn=(64, 64),
        block_sparse_tensors=sparse,
        return_lse=True,
    )
    out_ref = v[:, 64:65].float().expand_as(out)
    lse_ref = (
        torch.einsum("bqhd,bkhd->bhqk", q.float(), k[:, 64:65].float())
        / math.sqrt(q.shape[-1])
    ).squeeze(-1)

    assert_matches_reference(out, lse, out_ref, lse_ref)


@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize("mask", ["causal", "local", "seqused_k"])
def test_sm100_forward_tile_m64_fully_masked_row(tile_n, mask):
    """A query row with no visible keys must produce zeros, not 0/0 = NaN."""
    torch.manual_seed(11)
    # seqlen_q > seqlen_k makes the first query row see no keys under causal/local masking.
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = {
        "causal": {"causal": True},
        "local": {"window_size_left": 64, "window_size_right": 0},
        "seqused_k": {"causal": True, "seqused_k": torch.tensor([64], dtype=torch.int32, device="cuda")},
    }[mask]

    out, lse, *_ = _flash_attn_fwd(
        q=q, k=k, v=v, tile_mn=(64, tile_n), return_lse=True, **kwargs
    )

    empty_rows = torch.isneginf(lse).transpose(1, 2)  # (b, h, s) lse -> (b, s, h) out layout
    assert empty_rows.any(), "test setup must produce at least one fully masked row"
    assert not torch.isnan(out).any()
    assert (out[empty_rows] == 0).all()


@pytest.mark.parametrize("tile_n", [64, 128])
def test_sm100_forward_tile_m64_split_kv(tile_n):
    torch.manual_seed(1)
    q = torch.randn((1, 65, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 513, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        tile_mn=(64, tile_n),
        num_splits=2,
        pack_gqa=True,
        return_lse=True,
    )
    out_ref, lse_ref = reference_attention(q, k, v)

    assert_matches_reference(out, lse, out_ref, lse_ref)


@pytest.mark.parametrize("feature", ["causal", "score_mod"])
def test_sm100_forward_tile_m64_dense_features(feature):
    """Cover dense causal masking and a custom score transformation at tile_n=64."""
    torch.manual_seed(8)
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 193, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kwargs = {"causal": True} if feature == "causal" else {"score_mod": score_mod_times_two}

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        tile_mn=(64, 64),
        return_lse=True,
        **kwargs,
    )
    if feature == "causal":
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(128)
        q_idx = torch.arange(q.shape[1], device="cuda")[:, None]
        k_idx = torch.arange(k.shape[1], device="cuda")[None, :]
        causal_mask = k_idx <= q_idx + k.shape[1] - q.shape[1]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        lse_ref = torch.logsumexp(scores, dim=-1)
        out_ref = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v.float())
    else:
        out_ref, lse_ref = reference_attention(
            q,
            k,
            v,
            score_transform=lambda scores: scores * 2.0,
        )

    assert_matches_reference(out, lse, out_ref, lse_ref)


@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize(
    "sink_vals,softcap",
    [
        pytest.param((-0.5, 0.75), 7.0, id="softcap-small-sink"),
        # A sink far above the row max dominates the denominator, so it catches an O that was
        # normalized without the sink term even though LSE looks right.
        pytest.param((8.0, -8.0), 0.0, id="dominant-sink"),
    ],
)
def test_sm100_forward_tile_m64_softcap_and_sink(tile_n, sink_vals, softcap):
    torch.manual_seed(2)
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 193, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sink = torch.tensor(sink_vals, device="cuda", dtype=torch.bfloat16)

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        softcap=softcap,
        learnable_sink=sink,
        tile_mn=(64, tile_n),
        return_lse=True,
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(128)
    if softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)
    lse_ref = torch.logsumexp(
        torch.cat(
            (scores, sink.float()[None, :, None, None].expand(1, 2, 129, 1)), dim=-1
        ),
        dim=-1,
    )
    out_ref = torch.einsum(
        "bhqk,bkhd->bqhd", torch.exp(scores - lse_ref[..., None]), v.float()
    )

    assert_matches_reference(out, lse, out_ref, lse_ref)


def make_paged_kv(batch, seqlen_k, page_size, heads, head_dim, head_dim_v=None):
    """Create a shuffled physical page pool and its dense logical view."""
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    pages_per_sequence = seqlen_k // page_size
    num_pages = batch * pages_per_sequence
    k_pages = torch.randn(
        (num_pages, page_size, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_pages = torch.randn(
        (num_pages, page_size, heads, head_dim_v), device="cuda", dtype=torch.bfloat16
    )
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, pages_per_sequence
    )
    k_dense = k_pages[page_table.long()].reshape(batch, seqlen_k, heads, head_dim)
    v_dense = v_pages[page_table.long()].reshape(batch, seqlen_k, heads, head_dim_v)
    return k_pages, v_pages, page_table, k_dense, v_dense


@pytest.mark.parametrize("tile_n", [64, 128])
def test_sm100_forward_tile_m64_paged_kv(tile_n):
    torch.manual_seed(3)
    batch, seqlen_q, seqlen_k, heads, head_dim = 2, 65, 256, 2, 128
    q = torch.randn(
        (batch, seqlen_q, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k_pages, v_pages, page_table, k_dense, v_dense = make_paged_kv(
        batch, seqlen_k, tile_n, heads, head_dim
    )
    seqused_k = torch.tensor([193, 256], device="cuda", dtype=torch.int32)

    out, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k_pages,
        v=v_pages,
        page_table=page_table,
        seqused_k=seqused_k,
        max_seqlen_k=seqlen_k,
        tile_mn=(64, tile_n),
        return_lse=True,
    )
    out_ref, lse_ref = reference_attention(q, k_dense, v_dense, seqused_k=seqused_k)

    assert_matches_reference(out, lse, out_ref, lse_ref)


def test_sm100_forward_tile_m64_rejects_diff_head_dim_paged_splitkv():
    """Fail closed before diff-head-dim SplitKV can retile paged KV."""
    torch.manual_seed(5)
    batch, seqlen_q, seqlen_k, heads = 1, 65, 8192, 2
    q = torch.randn((batch, seqlen_q, heads, 64), device="cuda", dtype=torch.bfloat16)
    k_pages, v_pages, page_table, _, _ = make_paged_kv(
        batch, seqlen_k, 128, heads, 64, 128
    )

    with pytest.raises(NotImplementedError, match="matching K and V head dimensions"):
        _flash_attn_fwd(
            q=q,
            k=k_pages,
            v=v_pages,
            page_table=page_table,
            tile_mn=(64, 128),
            num_splits=2,
            return_lse=True,
        )


def make_cu_seqlens(lengths):
    """Construct device cumulative sequence lengths from Python lengths."""
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )


@pytest.mark.parametrize(
    "dtype,lengths_q,lengths_k,num_heads,num_kv_heads,causal,tile_n,pack_gqa",
    [
        pytest.param(
            torch.bfloat16,
            [63, 129, 257],
            [65, 193, 256],
            2,
            2,
            False,
            64,
            False,
            id="bf16-tail-n64",
        ),
        pytest.param(
            torch.float16,
            [63, 129],
            [65, 193],
            4,
            2,
            True,
            128,
            True,
            id="fp16-packed-gqa-causal-n128",
        ),
    ],
)
def test_sm100_forward_tile_m64_varlen(
    dtype,
    lengths_q,
    lengths_k,
    num_heads,
    num_kv_heads,
    causal,
    tile_n,
    pack_gqa,
):
    torch.manual_seed(4)
    head_dim = 128
    q = torch.randn((sum(lengths_q), num_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn(
        (sum(lengths_k), num_kv_heads, head_dim), device="cuda", dtype=dtype
    )
    v = torch.randn_like(k)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": make_cu_seqlens(lengths_q),
        "cu_seqlens_k": make_cu_seqlens(lengths_k),
        "max_seqlen_q": max(lengths_q),
        "max_seqlen_k": max(lengths_k),
        "causal": causal,
        "pack_gqa": pack_gqa,
        "return_lse": True,
    }

    out, lse, *_ = _flash_attn_fwd(**kwargs, tile_mn=(64, tile_n))
    out_ref, lse_ref, *_ = _flash_attn_fwd(**kwargs, tile_mn=(128, 128))

    torch.testing.assert_close(out, out_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=0, atol=2e-2)


def test_sm100_forward_tile_m64_rejects_mismatched_paged_kv_page_size():
    q = torch.zeros((1, 16, 1, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((4, 32, 1, 64), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    page_table = torch.arange(4, device="cuda", dtype=torch.int32).reshape(1, 4)

    with pytest.raises(NotImplementedError, match="page_size to equal tile_n"):
        _flash_attn_fwd(q=q, k=k, v=v, page_table=page_table, tile_mn=(64, 64))


def test_sm100_forward_tile_m64_rejects_fp8():
    q = torch.zeros((1, 16, 1, 64), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)

    with pytest.raises(NotImplementedError, match="only FP16 and BF16"):
        _flash_attn_fwd(q=q, k=k, v=v, tile_mn=(64, 64))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"use_2cta_instrs": True}, id="two-cta"),
        pytest.param({"use_clc_scheduler": True}, id="clc"),
        pytest.param({"paged_kv_non_tma": True}, id="non-tma-paged-kv"),
        pytest.param({"head_dim": 256, "head_dim_v": 256}, id="hd256"),
        pytest.param({"head_dim": 10, "head_dim_v": 16}, id="misaligned-head-dim"),
    ],
)
def test_sm100_forward_tile_m64_constructor_rejects_outside_envelope(kwargs):
    """Keep unsupported direct constructor configurations fail-closed."""
    constructor_kwargs = {"head_dim": 128, **kwargs}
    with pytest.raises(NotImplementedError, match="tile_m=64 forward"):
        FlashAttentionForwardSm100(
            m_block_size=64,
            n_block_size=128,
            **constructor_kwargs,
        )


@pytest.mark.parametrize(
    "head_dim,head_dim_v,tile_n,arch,match",
    [
        pytest.param(256, 256, 64, 100, "head dimensions", id="hd256"),
        pytest.param(128, 128, 32, 100, "tile_n in", id="tile-n32"),
        pytest.param(128, 128, 64, 103, "only on SM100", id="sm103"),
        pytest.param(128, 128, 64, 110, "only on SM100", id="sm110"),
    ],
)
def test_sm100_forward_tile_m64_rejects_outside_envelope(
    head_dim, head_dim_v, tile_n, arch, match
):
    q = torch.zeros((1, 16, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros((1, 16, 1, head_dim_v), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match=match):
        _flash_attn_fwd(q=q, k=k, v=v, tile_mn=(64, tile_n), _arch=arch)
