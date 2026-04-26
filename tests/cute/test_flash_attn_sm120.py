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
        pack_gqa=False,
        pack_gqa_was_explicit=False,
        num_splits=1,
        page_table=None,
        block_sparse_tensors=None,
        tile_mn=None,
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
    assert not policy.supports_pack_gqa
    assert not issubclass(FlashAttentionForwardSm120, FlashAttentionForwardSm80)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"dtype": torch.float32}, "only supports fp16 and bf16"),
        ({"requires_grad": True}, "forward-only"),
        ({"head_dim": 192, "head_dim_v": 192}, "head_dim"),
        ({"head_dim": 64, "head_dim_v": 128}, "head_dim == head_dim_v"),
        ({"pack_gqa": True, "pack_gqa_was_explicit": True}, "packed GQA"),
        ({"num_splits": 2}, "SplitKV"),
        ({"page_table": object()}, "paged KV"),
        ({"block_sparse_tensors": object()}, "block sparsity"),
        ({"aux_tensors": [object()]}, "aux_tensors"),
        ({"learnable_sink": object()}, "learnable_sink"),
        ({"qv": object()}, "qv/MLA"),
        ({"tile_mn": (128, 192)}, "tile_mn"),
    ],
)
def test_sm120_forward_validation_rejects_out_of_scope_cases(overrides, message):
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

    scores = torch.einsum("bthd,bshd->bhts", q.float() / (64**0.5), k.float())
    q_idx = torch.arange(q.shape[1], device="cuda")[:, None]
    kv_idx = torch.arange(k.shape[1], device="cuda")[None, :]
    mask = (q_idx % 128) >= (kv_idx % 128)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    out_ref = torch.einsum("bhts,bshd->bthd", attn, v)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_varlen_dense_smoke(dtype, head_dim, causal):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
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
    )
    out_ref = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
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
        (2, False, False),
        (1, True, None),
        (2, True, None),
    ],
)
def test_sm120_forward_dense_nonpacked_gqa_smoke(num_heads_kv, causal, pack_gqa):
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
