import pytest
import torch

from score_mod_definitions import score_mod_times_two
from mask_mod_definitions import cute_mini_causal_mask

from flash_attn.cute.arch_policy import get_forward_arch_policy
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.interface import (
    _validate_sm120_fwd_support,
    flash_attn_func,
    flash_attn_varlen_func,
)
from flash_attn.cute.testing import attention_ref


def _valid_sm120_kwargs(**overrides):
    kwargs = dict(
        dtype=torch.float16,
        head_dim=64,
        head_dim_v=64,
        requires_grad=False,
        is_varlen=False,
        is_local=False,
        softcap=None,
        score_mod=None,
        mask_mod=None,
        aux_tensors=None,
        learnable_sink=None,
        qv=None,
        qhead_per_kvhead=1,
        pack_gqa=False,
        pack_gqa_was_explicit=False,
        num_splits=1,
        page_table=None,
        has_seqused_k=False,
        block_sparse_tensors=None,
        tile_mn=None,
        num_stages=1,
        q_in_regs=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_sm120_forward_policy_keeps_native_path_separate():
    policy = get_forward_arch_policy(120)
    assert policy.name == "sm120"
    assert policy.smem_capacity_arch == "sm_120"
    assert policy.native_tensor_only
    assert policy.supports_warp_mma_f16bf16
    assert not policy.supports_tma_o
    assert not policy.supports_tmem
    assert not policy.supports_tcgen05
    assert not policy.supports_wgmma
    assert not policy.supports_stmatrix_acc_store
    assert policy.supports_pack_gqa
    assert not issubclass(FlashAttentionForwardSm120, FlashAttentionForwardSm80)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dtype": torch.float32}, "only supports fp16 and bf16"),
        ({"requires_grad": True}, "forward-only"),
        ({"head_dim": 192, "head_dim_v": 192}, "head_dim"),
        ({"head_dim": 64, "head_dim_v": 128}, "head_dim == head_dim_v"),
        ({"num_splits": 0}, "explicit num_splits"),
        ({"page_table": object()}, "seqused_k"),
        ({"block_sparse_tensors": object()}, "block sparsity"),
        ({"aux_tensors": [object()]}, "aux_tensors"),
        ({"learnable_sink": object()}, "learnable_sink"),
        ({"qv": object()}, "qv/MLA"),
        ({"tile_mn": (128, 192)}, "tile_mn"),
        ({"num_stages": 3}, "num_stages"),
        ({"num_stages": 2, "q_in_regs": True}, "q_in_regs"),
    ],
)
def test_sm120_forward_validation_rejects_out_of_scope_cases(overrides, message):
    with pytest.raises(NotImplementedError, match=message):
        _validate_sm120_fwd_support(**_valid_sm120_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_splits": 2, "pack_gqa": True}, "pack_gqa=False"),
        ({"num_splits": 2, "pack_gqa": True, "page_table": object(), "has_seqused_k": True}, "pack_gqa=False"),
        ({"num_splits": 2, "block_sparse_tensors": object()}, "block sparsity"),
        (
            {"num_splits": 2, "page_table": object(), "has_seqused_k": True, "block_sparse_tensors": object()},
            "block sparsity",
        ),
    ],
)
def test_sm120_forward_validation_rejects_splitkv_combinations(overrides, message):
    with pytest.raises(NotImplementedError, match=message):
        _validate_sm120_fwd_support(**_valid_sm120_kwargs(**overrides))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_dense_mha_smoke(dtype, head_dim, causal):
    torch.manual_seed(0)
    q = torch.randn(1, 64, 2, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 64, 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(1, 64, 2, head_dim, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(q, k, v, causal=causal, pack_gqa=False, num_splits=1)
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, dtype, head_dim, causal",
    [
        (1, 127, 129, torch.float16, 64, False),
        (1, 129, 127, torch.bfloat16, 64, True),
        (2, 130, 257, torch.float16, 96, False),
        (1, 257, 130, torch.bfloat16, 128, True),
    ],
)
def test_sm120_forward_dense_mha_tile_boundary_shapes(
    batch_size, seqlen_q, seqlen_k, dtype, head_dim, causal
):
    torch.manual_seed(0)
    q = torch.randn(batch_size, seqlen_q, 2, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, 2, head_dim, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(q, k, v, causal=causal, pack_gqa=False, num_splits=1)
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_splits", [2, 3])
def test_sm120_forward_dense_mha_splitkv_smoke(dtype, head_dim, causal, num_splits):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 2, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 257, 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(1, 257, 2, head_dim, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(q, k, v, causal=causal, pack_gqa=False, num_splits=num_splits)
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_kv", [1, 2])
def test_sm120_forward_dense_gqa_splitkv_smoke(dtype, head_dim, causal, num_heads_kv):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 257, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(1, 257, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(q, k, v, causal=causal, pack_gqa=False, num_splits=2)
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
def test_sm120_forward_dense_gqa_splitkv_three_splits_smoke():
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 257, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 257, 2, 64, device="cuda", dtype=torch.float16)
    out, _ = flash_attn_func(q, k, v, causal=False, pack_gqa=False, num_splits=3)
    out_ref, _ = attention_ref(q, k, v, causal=False)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize(
    "window_size",
    [
        (32, 0),
        (48, 16),
        (None, 32),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_dense_local_window_smoke(window_size, causal):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 127, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 127, 2, 64, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_func(
        q, k, v, causal=causal, window_size=window_size, pack_gqa=False, num_splits=1
    )
    out_ref, _ = attention_ref(q, k, v, causal=causal, window_size=window_size)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_dense_softcap_smoke(dtype, causal):
    torch.manual_seed(0)
    softcap = 15.0
    q = torch.randn(1, 129, 2, 64, device="cuda", dtype=dtype) * softcap / 4
    k = torch.randn(1, 127, 2, 64, device="cuda", dtype=dtype)
    v = torch.randn(1, 127, 2, 64, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(
        q, k, v, causal=causal, softcap=softcap, pack_gqa=False, num_splits=1
    )
    out_ref, _ = attention_ref(q, k, v, causal=causal, softcap=softcap)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_dense_score_mod_smoke(dtype, causal):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 2, 64, device="cuda", dtype=dtype)
    k = torch.randn(1, 127, 2, 64, device="cuda", dtype=dtype)
    v = torch.randn(1, 127, 2, 64, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(
        q, k, v, causal=causal, score_mod=score_mod_times_two, pack_gqa=False, num_splits=1
    )
    out_ref, _ = attention_ref(q * 2, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


def _mini_causal_mask_ref(q, k, v):
    if q.shape[2] != k.shape[2]:
        repeats = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q.float() / (q.shape[-1] ** 0.5), k.float())
    q_idx = torch.arange(q.shape[1], device=q.device)[:, None]
    kv_idx = torch.arange(k.shape[1], device=k.device)[None, :]
    mask = (q_idx % 128) >= (kv_idx % 128)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhts,bshd->bthd", attn, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
def test_sm120_forward_dense_mask_mod_smoke():
    torch.manual_seed(0)
    q = torch.randn(1, 129, 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 127, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 127, 2, 64, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_func(
        q, k, v, causal=False, mask_mod=cute_mini_causal_mask, pack_gqa=False, num_splits=1
    )

    out_ref = _mini_causal_mask_ref(q, k, v)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


def _attention_ref_varlen(
    q, k, v, cu_seqlens_q, cu_seqlens_k, causal, window_size=(None, None), softcap=0.0
):
    outs = []
    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = cu_seqlens_q[batch_idx : batch_idx + 2].tolist()
        k_start, k_end = cu_seqlens_k[batch_idx : batch_idx + 2].tolist()
        out, _ = attention_ref(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        outs.append(out.squeeze(0))
    return torch.cat(outs, dim=0)


def _make_paged_kv(k, v, page_size):
    batch_size, seqlen_k, num_heads_kv, head_dim = k.shape
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_seq
    k_paged = torch.zeros(
        num_pages, page_size, num_heads_kv, head_dim, device=k.device, dtype=k.dtype
    )
    v_paged = torch.zeros_like(k_paged)
    page_table = torch.empty(
        batch_size, num_pages_per_seq, device=k.device, dtype=torch.int32
    )
    for batch_idx in range(batch_size):
        for page_idx in range(num_pages_per_seq):
            global_page_idx = batch_idx * num_pages_per_seq + page_idx
            page_table[batch_idx, page_idx] = global_page_idx
            start = page_idx * page_size
            end = min(start + page_size, seqlen_k)
            if start < end:
                k_paged[global_page_idx, : end - start] = k[batch_idx, start:end]
                v_paged[global_page_idx, : end - start] = v[batch_idx, start:end]
    return k_paged, v_paged, page_table


def _attention_ref_paged(q, k, v, cache_seqlens, causal, window_size=(None, None), softcap=0.0):
    outs = []
    for batch_idx, seqlen_k in enumerate(cache_seqlens.tolist()):
        out, _ = attention_ref(
            q[batch_idx : batch_idx + 1],
            k[batch_idx : batch_idx + 1, :seqlen_k],
            v[batch_idx : batch_idx + 1, :seqlen_k],
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        outs.append(out)
    return torch.cat(outs, dim=0)


def _attention_ref_paged_varlen(
    q, k, v, cu_seqlens_q, cache_seqlens, causal, window_size=(None, None), softcap=0.0
):
    outs = []
    for batch_idx, seqlen_k in enumerate(cache_seqlens.tolist()):
        q_start, q_end = cu_seqlens_q[batch_idx : batch_idx + 2].tolist()
        out, _ = attention_ref(
            q[q_start:q_end].unsqueeze(0),
            k[batch_idx : batch_idx + 1, :seqlen_k],
            v[batch_idx : batch_idx + 1, :seqlen_k],
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )
        outs.append(out.squeeze(0))
    return torch.cat(outs, dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(2, 2), (4, 2), (4, 1)])
def test_sm120_forward_varlen_dense_smoke(dtype, head_dim, causal, num_heads_q, num_heads_kv):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    q = torch.randn(sum(q_lens), num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lens), num_heads_kv, head_dim, device="cuda", dtype=dtype)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=causal,
        pack_gqa=False,
    )
    out_ref = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_varlen_mha_splitkv_smoke(dtype, head_dim, causal):
    torch.manual_seed(0)
    q_lens = [33, 97, 129]
    k_lens = [65, 131, 257]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), 2, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lens), 2, head_dim, device="cuda", dtype=dtype)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=causal,
        pack_gqa=False,
        num_splits=2,
    )
    out_ref = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_kv", [1, 2])
def test_sm120_forward_varlen_gqa_splitkv_smoke(causal, num_heads_kv):
    torch.manual_seed(0)
    q_lens = [33, 97, 129]
    k_lens = [65, 131, 257]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=torch.float16)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=causal,
        pack_gqa=False,
        num_splits=2,
    )
    out_ref = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_paged_kv_varlen_q_smoke(causal):
    torch.manual_seed(0)
    batch_size, max_seqlen_k, num_heads_q, num_heads_kv, head_dim = 3, 127, 4, 1, 64
    q_lens = [17, 64, 129]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cache_seqlens = torch.tensor([127, 95, 63], device="cuda", dtype=torch.int32)
    q = torch.randn(sum(q_lens), num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        batch_size, max_seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, page_size=64)
    out, _ = flash_attn_varlen_func(
        q,
        k_paged,
        v_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=None,
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=causal,
        pack_gqa=False,
    )
    out_ref = _attention_ref_paged_varlen(q, k, v, cu_seqlens_q, cache_seqlens, causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("page_size", [16, 64, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(2, 2), (4, 1)])
def test_sm120_forward_paged_kv_smoke(page_size, causal, num_heads_q, num_heads_kv):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.float16)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, page_size)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    out, _ = flash_attn_varlen_func(
        q.reshape(batch_size * seqlen_q, num_heads_q, head_dim),
        k_paged,
        v_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=None,
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=causal,
        pack_gqa=False,
    )
    out_ref = _attention_ref_paged(q, k, v, cache_seqlens, causal).reshape(
        batch_size * seqlen_q, num_heads_q, head_dim
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(2, 2), (4, 2), (4, 1)])
def test_sm120_forward_paged_kv_splitkv_smoke(
    dtype, head_dim, causal, num_heads_q, num_heads_kv
):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k = 2, 129, 257
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, page_size=64)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    out, _ = flash_attn_varlen_func(
        q.reshape(batch_size * seqlen_q, num_heads_q, head_dim),
        k_paged,
        v_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=None,
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=causal,
        pack_gqa=False,
        num_splits=2,
    )
    out_ref = _attention_ref_paged(q, k, v, cache_seqlens, causal).reshape(
        batch_size * seqlen_q, num_heads_q, head_dim
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
def test_sm120_forward_paged_kv_splitkv_three_splits_smoke():
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, 4, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seqlen_k, 2, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seqlen_k, 2, head_dim, device="cuda", dtype=torch.float16)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, page_size=64)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    out, _ = flash_attn_varlen_func(
        q.reshape(batch_size * seqlen_q, 4, head_dim),
        k_paged,
        v_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=None,
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=False,
        pack_gqa=False,
        num_splits=3,
    )
    out_ref = _attention_ref_paged(q, k, v, cache_seqlens, False).reshape(
        batch_size * seqlen_q, 4, head_dim
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("feature", ["local", "softcap"])
def test_sm120_forward_paged_kv_extensions_smoke(feature):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen_k, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen_k, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, 64)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    kwargs = {}
    ref_kwargs = {}
    if feature == "local":
        kwargs["window_size"] = (48, 16)
        ref_kwargs["window_size"] = (48, 16)
    elif feature == "softcap":
        kwargs["softcap"] = 15.0
        ref_kwargs["softcap"] = 15.0
        q = q * kwargs["softcap"] / 4
    else:
        raise AssertionError(f"Unexpected feature: {feature}")
    out, _ = flash_attn_varlen_func(
        q.reshape(batch_size * seqlen_q, 2, head_dim),
        k_paged,
        v_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=None,
        seqused_k=cache_seqlens,
        page_table=page_table,
        causal=False,
        pack_gqa=False,
        **kwargs,
    )
    out_ref = _attention_ref_paged(q, k, v, cache_seqlens, False, **ref_kwargs).reshape(
        batch_size * seqlen_q, 2, head_dim
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize(
    "window_size, score_mod",
    [
        ((32, 0), None),
        ((48, 16), None),
        ((None, None), score_mod_times_two),
    ],
)
def test_sm120_forward_varlen_core_extensions_smoke(window_size, score_mod):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    q = torch.randn(sum(q_lens), 2, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lens), 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(sum(k_lens), 2, 64, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=False,
        window_size=window_size,
        score_mod=score_mod,
        pack_gqa=False,
    )
    q_ref = q * 2 if score_mod is score_mod_times_two else q
    out_ref = _attention_ref_varlen(
        q_ref, k, v, cu_seqlens_q, cu_seqlens_k, False, window_size=window_size
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
def test_sm120_forward_varlen_softcap_smoke():
    torch.manual_seed(0)
    softcap = 15.0
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    q = torch.randn(sum(q_lens), 2, 64, device="cuda", dtype=torch.bfloat16) * softcap / 4
    k = torch.randn(sum(k_lens), 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(sum(k_lens), 2, 64, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=True,
        softcap=softcap,
        pack_gqa=False,
    )
    out_ref_no_softcap = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, True)
    out_ref = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, True, softcap=softcap)
    assert not torch.allclose(out_ref, out_ref_no_softcap)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize(
    "num_heads_kv, causal, pack_gqa",
    [
        (1, False, False),
        (1, False, True),
        (2, False, False),
        (2, False, True),
        (1, True, None),
        (1, True, True),
        (2, True, None),
        (2, True, True),
    ],
)
def test_sm120_forward_dense_gqa_smoke(num_heads_kv, causal, pack_gqa):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 127, num_heads_kv, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 127, num_heads_kv, 64, device="cuda", dtype=torch.float16)
    kwargs = {"num_splits": 1}
    if pack_gqa is not None:
        kwargs["pack_gqa"] = pack_gqa
    out, _ = flash_attn_func(q, k, v, causal=causal, **kwargs)
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("num_heads_kv", [1, 2])
@pytest.mark.parametrize("feature", ["local", "score_mod", "mask_mod"])
def test_sm120_forward_dense_gqa_extensions_smoke(num_heads_kv, feature):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 127, num_heads_kv, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 127, num_heads_kv, 64, device="cuda", dtype=torch.bfloat16)
    kwargs = {"pack_gqa": True, "num_splits": 1}
    ref_q = q
    ref_kwargs = {}
    if feature == "local":
        kwargs["window_size"] = (48, 16)
        ref_kwargs["window_size"] = (48, 16)
    elif feature == "score_mod":
        kwargs["score_mod"] = score_mod_times_two
        ref_q = q * 2
    elif feature == "mask_mod":
        kwargs["mask_mod"] = cute_mini_causal_mask
    else:
        raise AssertionError(f"Unexpected feature: {feature}")

    out, _ = flash_attn_func(q, k, v, causal=False, **kwargs)
    if feature == "mask_mod":
        out_ref = _mini_causal_mask_ref(q, k, v)
    else:
        out_ref, _ = attention_ref(ref_q, k, v, causal=False, **ref_kwargs)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(4, 2), (4, 1)])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_varlen_packed_gqa_smoke(num_heads_q, num_heads_kv, causal):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), num_heads_q, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=torch.float16)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=causal,
        pack_gqa=True,
    )
    out_ref = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
