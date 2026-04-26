import pytest
import torch

from score_mod_definitions import (
    score_mod_batch_bias,
    score_mod_dual_buffer,
    score_mod_global_kv_bias,
    score_mod_times_two,
)
from mask_mod_definitions import (
    cute_document_mask,
    cute_global_offset_mask,
    cute_ima_mask,
    cute_mini_causal_mask,
)

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
        ({"head_dim": 64, "head_dim_v": 192}, "head_dim_v"),
        ({"num_splits": 0}, "explicit num_splits"),
        ({"page_table": object()}, "seqused_k"),
        ({"block_sparse_tensors": object()}, "block sparsity"),
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


@pytest.mark.parametrize("head_dim,head_dim_v", [(64, 128), (128, 64), (96, 64)])
def test_sm120_forward_validation_allows_unequal_head_dim_v(head_dim, head_dim_v):
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(head_dim=head_dim, head_dim_v=head_dim_v)
    )


def test_sm120_forward_validation_allows_dense_aux_tensors():
    _validate_sm120_fwd_support(**_valid_sm120_kwargs(aux_tensors=[object()]))


def test_sm120_forward_validation_allows_varlen_score_mod_aux_tensors():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(is_varlen=True, score_mod=object(), aux_tensors=[object()])
    )


def test_sm120_forward_validation_allows_splitkv_score_mod_aux_tensors():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(num_splits=2, score_mod=object(), aux_tensors=[object()])
    )


def test_sm120_forward_validation_allows_paged_kv_score_mod_aux_tensors():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            page_table=object(),
            has_seqused_k=True,
            score_mod=object(),
            aux_tensors=[object()],
        )
    )


def test_sm120_forward_validation_allows_paged_kv_pack_gqa():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            page_table=object(),
            has_seqused_k=True,
            qhead_per_kvhead=4,
            pack_gqa=True,
        )
    )


def test_sm120_forward_validation_allows_paged_kv_splitkv_score_mod_aux_tensors():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            num_splits=2,
            page_table=object(),
            has_seqused_k=True,
            score_mod=object(),
            aux_tensors=[object()],
        )
    )


def test_sm120_forward_validation_allows_paged_kv_pack_gqa_splitkv():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            num_splits=2,
            page_table=object(),
            has_seqused_k=True,
            qhead_per_kvhead=4,
            pack_gqa=True,
        )
    )


def test_sm120_forward_validation_allows_paged_kv_pack_gqa_splitkv_score_mod_aux_tensors():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            num_splits=2,
            page_table=object(),
            has_seqused_k=True,
            qhead_per_kvhead=4,
            pack_gqa=True,
            score_mod=object(),
            aux_tensors=[object()],
        )
    )


def test_sm120_forward_validation_allows_varlen_pack_gqa_splitkv():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            is_varlen=True,
            num_splits=2,
            qhead_per_kvhead=4,
            pack_gqa=True,
        )
    )


def test_sm120_forward_validation_allows_varlen_pack_gqa_splitkv_score_mod_aux_tensors():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            is_varlen=True,
            num_splits=2,
            qhead_per_kvhead=4,
            pack_gqa=True,
            score_mod=object(),
            aux_tensors=[object()],
        )
    )


def test_sm120_forward_validation_allows_dense_pack_gqa_splitkv():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            is_varlen=False,
            num_splits=2,
            qhead_per_kvhead=4,
            pack_gqa=True,
        )
    )


def test_sm120_forward_validation_allows_dense_pack_gqa_splitkv_score_mod_aux_tensors():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            is_varlen=False,
            num_splits=2,
            qhead_per_kvhead=4,
            pack_gqa=True,
            score_mod=object(),
            aux_tensors=[object()],
        )
    )


def test_sm120_forward_validation_allows_varlen_mask_mod_without_aux_tensors():
    _validate_sm120_fwd_support(**_valid_sm120_kwargs(is_varlen=True, mask_mod=object()))


def test_sm120_forward_validation_allows_dense_learnable_sink():
    _validate_sm120_fwd_support(**_valid_sm120_kwargs(learnable_sink=object()))


def test_sm120_forward_validation_allows_varlen_learnable_sink():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(is_varlen=True, learnable_sink=object())
    )


def test_sm120_forward_validation_allows_paged_kv_learnable_sink():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            page_table=object(),
            has_seqused_k=True,
            learnable_sink=object(),
            pack_gqa=False,
        )
    )


def test_sm120_forward_validation_allows_paged_kv_pack_gqa_learnable_sink():
    _validate_sm120_fwd_support(
        **_valid_sm120_kwargs(
            page_table=object(),
            has_seqused_k=True,
            qhead_per_kvhead=4,
            pack_gqa=True,
            learnable_sink=object(),
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_splits": 2, "learnable_sink": object()},
        {"num_splits": 2, "is_varlen": True, "learnable_sink": object()},
        {
            "num_splits": 2,
            "page_table": object(),
            "has_seqused_k": True,
            "learnable_sink": object(),
        },
        {"num_splits": 2, "qhead_per_kvhead": 4, "pack_gqa": True, "learnable_sink": object()},
        {
            "num_splits": 2,
            "is_varlen": True,
            "qhead_per_kvhead": 4,
            "pack_gqa": True,
            "learnable_sink": object(),
        },
        {
            "num_splits": 2,
            "page_table": object(),
            "has_seqused_k": True,
            "qhead_per_kvhead": 4,
            "pack_gqa": True,
            "learnable_sink": object(),
        },
    ],
)
def test_sm120_forward_validation_allows_splitkv_learnable_sink(overrides):
    _validate_sm120_fwd_support(**_valid_sm120_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"aux_tensors": [object()], "is_varlen": True, "mask_mod": object()},
            "aux_tensors with varlen mask_mod",
        ),
        ({"aux_tensors": [object()], "num_splits": 2}, "aux_tensors with SplitKV"),
        (
            {
                "aux_tensors": [object()],
                "num_splits": 2,
                "score_mod": object(),
                "mask_mod": object(),
            },
            "aux_tensors with SplitKV",
        ),
        (
            {"aux_tensors": [object()], "page_table": object(), "has_seqused_k": True},
            "aux_tensors with paged KV",
        ),
        (
            {
                "aux_tensors": [object()],
                "page_table": object(),
                "has_seqused_k": True,
                "score_mod": object(),
                "mask_mod": object(),
            },
            "aux_tensors with paged KV",
        ),
        ({"aux_tensors": [object()], "block_sparse_tensors": object()}, "block sparsity"),
    ],
)
def test_sm120_forward_validation_rejects_aux_tensor_combinations(overrides, message):
    with pytest.raises(NotImplementedError, match=message):
        _validate_sm120_fwd_support(**_valid_sm120_kwargs(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"score_mod": object()},
        {"mask_mod": object()},
        {"aux_tensors": [object()]},
    ],
)
def test_sm120_forward_validation_rejects_varlen_pack_gqa_splitkv_extensions(overrides):
    kwargs = dict(is_varlen=True, num_splits=2, qhead_per_kvhead=4, pack_gqa=True)
    kwargs.update(overrides)
    with pytest.raises(NotImplementedError, match="pack_gqa=False"):
        _validate_sm120_fwd_support(**_valid_sm120_kwargs(**kwargs))


@pytest.mark.parametrize(
    "overrides",
    [
        {"score_mod": object()},
        {"mask_mod": object()},
        {"aux_tensors": [object()]},
    ],
)
def test_sm120_forward_validation_rejects_dense_pack_gqa_splitkv_extensions(overrides):
    kwargs = dict(is_varlen=False, num_splits=2, qhead_per_kvhead=4, pack_gqa=True)
    kwargs.update(overrides)
    with pytest.raises(NotImplementedError, match="pack_gqa=False"):
        _validate_sm120_fwd_support(**_valid_sm120_kwargs(**kwargs))


@pytest.mark.parametrize(
    "overrides",
    [
        {"score_mod": object()},
        {"mask_mod": object()},
        {"aux_tensors": [object()]},
        {"block_sparse_tensors": object()},
    ],
)
def test_sm120_forward_validation_rejects_paged_pack_gqa_splitkv_extensions(overrides):
    kwargs = dict(
        num_splits=2,
        page_table=object(),
        has_seqused_k=True,
        qhead_per_kvhead=4,
        pack_gqa=True,
    )
    kwargs.update(overrides)
    with pytest.raises(NotImplementedError, match="pack_gqa=False|block sparsity"):
        _validate_sm120_fwd_support(**_valid_sm120_kwargs(**kwargs))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"learnable_sink": object(), "num_splits": 2, "score_mod": object()},
            "learnable_sink with SplitKV modifiers",
        ),
        (
            {"learnable_sink": object(), "num_splits": 2, "aux_tensors": [object()]},
            "aux_tensors with SplitKV",
        ),
        (
            {"learnable_sink": object(), "num_splits": 2, "mask_mod": object()},
            "learnable_sink with SplitKV modifiers",
        ),
        ({"learnable_sink": object(), "block_sparse_tensors": object()}, "block sparsity"),
    ],
)
def test_sm120_forward_validation_rejects_learnable_sink_combinations(overrides, message):
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
@pytest.mark.parametrize("head_dim,head_dim_v", [(64, 128), (128, 64), (96, 64)])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_dense_mha_unequal_head_dim_v_smoke(dtype, head_dim, head_dim_v, causal):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 2, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 127, 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(1, 127, 2, head_dim_v, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(q, k, v, causal=causal, pack_gqa=False, num_splits=1)
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("pack_gqa", [False, True])
def test_sm120_forward_dense_gqa_unequal_head_dim_v_smoke(pack_gqa):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 127, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 127, 1, 64, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_func(q, k, v, causal=False, pack_gqa=pack_gqa, num_splits=1)
    out_ref, _ = attention_ref(q, k, v, causal=False)
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_dense_packed_gqa_splitkv_smoke(dtype, causal):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(1, 257, 1, 64, device="cuda", dtype=dtype)
    v = torch.randn(1, 257, 1, 64, device="cuda", dtype=dtype)
    out, _ = flash_attn_func(q, k, v, causal=causal, pack_gqa=True, num_splits=2)
    out_ref, _ = attention_ref(q, k, v, causal=causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
def test_sm120_forward_dense_packed_gqa_splitkv_three_splits_smoke():
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 257, 1, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 257, 1, 64, device="cuda", dtype=torch.float16)
    out, _ = flash_attn_func(q, k, v, causal=False, pack_gqa=True, num_splits=3)
    out_ref, _ = attention_ref(q, k, v, causal=False)
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(2, 2), (4, 2)])
def test_sm120_forward_dense_splitkv_score_mod_aux_tensors_smoke(dtype, num_heads_q, num_heads_kv):
    torch.manual_seed(0)
    q = torch.randn(1, 129, num_heads_q, 64, device="cuda", dtype=dtype)
    k = torch.randn(1, 257, num_heads_kv, 64, device="cuda", dtype=dtype)
    v = torch.randn(1, 257, num_heads_kv, 64, device="cuda", dtype=dtype)
    kv_bias = torch.randn(k.shape[1], device="cuda", dtype=dtype)
    out, _ = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        score_mod=score_mod_global_kv_bias,
        aux_tensors=[kv_bias],
        pack_gqa=False,
        num_splits=2,
    )
    out_ref = _dense_kv_bias_ref(q, k, v, kv_bias)
    out_no_bias, _ = attention_ref(q, k, v, causal=False)
    assert not torch.allclose(out_ref, out_no_bias)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm120_forward_dense_packed_gqa_splitkv_score_mod_aux_tensors_smoke(dtype):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(1, 257, 1, 64, device="cuda", dtype=dtype)
    v = torch.randn(1, 257, 1, 64, device="cuda", dtype=dtype)
    kv_bias = torch.randn(k.shape[1], device="cuda", dtype=dtype)
    out, _ = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        score_mod=score_mod_global_kv_bias,
        aux_tensors=[kv_bias],
        pack_gqa=True,
        num_splits=2,
    )
    out_ref = _dense_kv_bias_ref(q, k, v, kv_bias)
    out_no_bias, _ = attention_ref(q, k, v, causal=False)
    assert not torch.allclose(out_ref, out_no_bias)
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
@pytest.mark.parametrize("num_heads_kv,pack_gqa", [(4, False), (1, False), (1, True)])
def test_sm120_forward_dense_learnable_sink_smoke(dtype, causal, num_heads_kv, pack_gqa):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, num_heads_q, head_dim = 2, 129, 127, 4, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=1,
        return_lse=True,
    )
    out_ref, _, lse_ref = attention_ref(
        q, k, v, causal=causal, learnable_sink=learnable_sink, return_lse=True
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv,pack_gqa", [(2, 2, False), (4, 1, True)])
def test_sm120_forward_dense_splitkv_learnable_sink_smoke(causal, num_heads_q, num_heads_kv, pack_gqa):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)
    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=2,
        return_lse=True,
    )
    out_ref, _, lse_ref = attention_ref(
        q, k, v, causal=causal, learnable_sink=learnable_sink, return_lse=True
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=5e-2, rtol=5e-2)


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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads_kv", [1, 2])
@pytest.mark.parametrize("score_mod_name", ["batch_bias", "dual_buffer"])
def test_sm120_forward_dense_score_mod_aux_tensors_smoke(dtype, num_heads_kv, score_mod_name):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, num_heads_q, head_dim = 2, 129, 127, 4, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    if score_mod_name == "batch_bias":
        score_mod = score_mod_batch_bias
        aux_tensors = [torch.randn(batch_size, device="cuda", dtype=dtype)]
    elif score_mod_name == "dual_buffer":
        score_mod = score_mod_dual_buffer
        aux_tensors = [
            torch.randn(num_heads_q, device="cuda", dtype=dtype),
            torch.randn(seqlen_q, device="cuda", dtype=dtype),
        ]
    else:
        raise AssertionError(f"Unexpected score_mod_name: {score_mod_name}")

    out, _ = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        pack_gqa=False,
        num_splits=1,
    )
    out_ref, _ = attention_ref(q, k, v, causal=False)
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


def _dense_mask_ref(q, k, v, mask):
    if q.shape[2] != k.shape[2]:
        repeats = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q.float() / (q.shape[-1] ** 0.5), k.float())
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhts,bshd->bthd", attn, v)


def _dense_kv_bias_ref(q, k, v, kv_bias):
    if q.shape[2] != k.shape[2]:
        repeats = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q.float() / (q.shape[-1] ** 0.5), k.float())
    scores = scores + kv_bias.float().view(1, 1, 1, -1)
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhts,bshd->bthd", attn, v)


def _document_mask_ref(q, k, v, doc_ids):
    q_doc = doc_ids[:, :, : q.shape[1]].unsqueeze(-1)
    k_doc = doc_ids[:, :, : k.shape[1]].unsqueeze(-2)
    return _dense_mask_ref(q, k, v, q_doc == k_doc)


def _ima_mask_ref(q, k, v, threshold):
    kv_idx = torch.arange(k.shape[1], device=k.device)
    mask = (kv_idx >= threshold).view(1, 1, 1, k.shape[1])
    return _dense_mask_ref(q, k, v, mask)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("num_heads_kv", [1, 2])
@pytest.mark.parametrize("mask_mod_name", ["document", "ima"])
def test_sm120_forward_dense_mask_mod_aux_tensors_smoke(num_heads_kv, mask_mod_name):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, num_heads_q, head_dim = 2, 129, 127, 4, 64
    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    if mask_mod_name == "document":
        mask_mod = cute_document_mask
        doc_pattern = torch.arange(max(seqlen_q, seqlen_k), device="cuda", dtype=torch.int32) % 7
        doc_ids = doc_pattern.view(1, 1, -1).expand(batch_size, num_heads_q, -1).contiguous()
        aux_tensors = [doc_ids]
        out_ref = _document_mask_ref(q, k, v, doc_ids)
    elif mask_mod_name == "ima":
        mask_mod = cute_ima_mask
        threshold = torch.arange(seqlen_k, device="cuda", dtype=torch.int32) // 2
        aux_tensors = [threshold]
        out_ref = _ima_mask_ref(q, k, v, threshold)
    else:
        raise AssertionError(f"Unexpected mask_mod_name: {mask_mod_name}")

    out, _ = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        pack_gqa=False,
        num_splits=1,
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("pack_gqa", [False, True])
def test_sm120_forward_dense_learnable_sink_all_masked(pack_gqa):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, num_heads_q, head_dim = 2, 129, 127, 4, 64
    num_heads_kv = 1 if pack_gqa else num_heads_q
    q = torch.randn(
        batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)
    threshold = torch.arange(seqlen_k, device="cuda", dtype=torch.int32) + 1

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        mask_mod=cute_ima_mask,
        aux_tensors=[threshold],
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=1,
        return_lse=True,
    )
    expected_out = torch.zeros_like(out)
    expected_lse = learnable_sink.float().view(1, num_heads_q, 1).expand_as(lse)
    torch.testing.assert_close(out, expected_out, atol=0.0, rtol=0.0)
    torch.testing.assert_close(lse.float(), expected_lse, atol=5e-3, rtol=5e-3)


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


def _attention_ref_varlen_with_lse(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    causal,
    window_size=(None, None),
    softcap=0.0,
    learnable_sink=None,
):
    outs = []
    lses = []
    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = cu_seqlens_q[batch_idx : batch_idx + 2].tolist()
        k_start, k_end = cu_seqlens_k[batch_idx : batch_idx + 2].tolist()
        out, _, lse = attention_ref(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            learnable_sink=learnable_sink,
            return_lse=True,
        )
        outs.append(out.squeeze(0))
        lses.append(lse.squeeze(0))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def _attention_ref_varlen_kv_bias(q, k, v, cu_seqlens_q, cu_seqlens_k, kv_bias):
    outs = []
    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = cu_seqlens_q[batch_idx : batch_idx + 2].tolist()
        k_start, k_end = cu_seqlens_k[batch_idx : batch_idx + 2].tolist()
        q_slice = q[q_start:q_end]
        k_slice = k[k_start:k_end]
        v_slice = v[k_start:k_end]
        if q_slice.shape[1] != k_slice.shape[1]:
            repeats = q_slice.shape[1] // k_slice.shape[1]
            k_slice = k_slice.repeat_interleave(repeats, dim=1)
            v_slice = v_slice.repeat_interleave(repeats, dim=1)
        scores = torch.einsum(
            "thd,shd->hts", q_slice.float() / (q_slice.shape[-1] ** 0.5), k_slice.float()
        )
        scores = scores + kv_bias[k_start:k_end].float().view(1, 1, -1)
        attn = torch.softmax(scores, dim=-1).to(v_slice.dtype)
        outs.append(torch.einsum("hts,shd->thd", attn, v_slice))
    return torch.cat(outs, dim=0)


def _attention_ref_varlen_global_offset_mask(q, k, v, cu_seqlens_q, cu_seqlens_k):
    outs = []
    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = cu_seqlens_q[batch_idx : batch_idx + 2].tolist()
        k_start, k_end = cu_seqlens_k[batch_idx : batch_idx + 2].tolist()
        q_slice = q[q_start:q_end]
        k_slice = k[k_start:k_end]
        v_slice = v[k_start:k_end]
        if q_slice.shape[1] != k_slice.shape[1]:
            repeats = q_slice.shape[1] // k_slice.shape[1]
            k_slice = k_slice.repeat_interleave(repeats, dim=1)
            v_slice = v_slice.repeat_interleave(repeats, dim=1)
        scores = torch.einsum(
            "thd,shd->hts", q_slice.float() / (q_slice.shape[-1] ** 0.5), k_slice.float()
        )
        q_global = torch.arange(q_start, q_end, device=q.device)[:, None]
        kv_global = torch.arange(k_start, k_end, device=k.device)[None, :]
        mask = (kv_global % 3) != ((q_global + 1) % 3)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v_slice.dtype)
        outs.append(torch.einsum("hts,shd->thd", attn, v_slice))
    return torch.cat(outs, dim=0)


def _make_paged_kv(k, v, page_size):
    batch_size, seqlen_k, num_heads_kv, head_dim = k.shape
    head_dim_v = v.shape[-1]
    num_pages_per_seq = (seqlen_k + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_seq
    k_paged = torch.zeros(
        num_pages, page_size, num_heads_kv, head_dim, device=k.device, dtype=k.dtype
    )
    v_paged = torch.zeros(
        num_pages, page_size, num_heads_kv, head_dim_v, device=v.device, dtype=v.dtype
    )
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


def _attention_ref_paged(
    q,
    k,
    v,
    cache_seqlens,
    causal,
    window_size=(None, None),
    softcap=0.0,
    learnable_sink=None,
    return_lse=False,
):
    outs = []
    lses = []
    for batch_idx, seqlen_k in enumerate(cache_seqlens.tolist()):
        result = attention_ref(
            q[batch_idx : batch_idx + 1],
            k[batch_idx : batch_idx + 1, :seqlen_k],
            v[batch_idx : batch_idx + 1, :seqlen_k],
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            learnable_sink=learnable_sink,
            return_lse=return_lse,
        )
        out = result[0]
        outs.append(out)
        if return_lse:
            lses.append(result[2].squeeze(0))
    out = torch.cat(outs, dim=0)
    if return_lse:
        return out, torch.cat(lses, dim=1)
    return out


def _attention_ref_paged_kv_bias(q, k, v, cache_seqlens, kv_bias, causal=False):
    outs = []
    for batch_idx, seqlen_k in enumerate(cache_seqlens.tolist()):
        q_slice = q[batch_idx]
        k_slice = k[batch_idx, :seqlen_k]
        v_slice = v[batch_idx, :seqlen_k]
        if q_slice.shape[1] != k_slice.shape[1]:
            repeats = q_slice.shape[1] // k_slice.shape[1]
            k_slice = k_slice.repeat_interleave(repeats, dim=1)
            v_slice = v_slice.repeat_interleave(repeats, dim=1)
        scores = torch.einsum(
            "thd,shd->hts", q_slice.float() / (q_slice.shape[-1] ** 0.5), k_slice.float()
        )
        scores = scores + kv_bias[:seqlen_k].float().view(1, 1, -1)
        if causal:
            q_idx = torch.arange(q.shape[1], device=q.device)[:, None]
            kv_idx = torch.arange(seqlen_k, device=k.device)[None, :]
            offset = seqlen_k - q.shape[1]
            scores = scores.masked_fill((kv_idx > q_idx + offset).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1).to(v_slice.dtype)
        outs.append(torch.einsum("hts,shd->thd", attn, v_slice).unsqueeze(0))
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
@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 64), (64, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_varlen_unequal_head_dim_v_smoke(head_dim, head_dim_v, causal):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), 2, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lens), 2, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(sum(k_lens), 2, head_dim_v, device="cuda", dtype=torch.bfloat16)
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
@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 64), (64, 128)])
def test_sm120_forward_dense_mha_splitkv_unequal_head_dim_v_smoke(head_dim, head_dim_v):
    torch.manual_seed(0)
    q = torch.randn(1, 129, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 257, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 257, 2, head_dim_v, device="cuda", dtype=torch.bfloat16)
    out, _ = flash_attn_func(q, k, v, causal=False, pack_gqa=False, num_splits=2)
    out_ref, _ = attention_ref(q, k, v, causal=False)
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
def test_sm120_forward_paged_kv_unequal_head_dim_v_smoke():
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim, head_dim_v = 2, 129, 257, 128, 64
    q = torch.randn(batch_size, seqlen_q, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen_k, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen_k, 1, head_dim_v, device="cuda", dtype=torch.bfloat16)
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
    )
    out_ref = _attention_ref_paged(q, k, v, cache_seqlens, False).reshape(
        batch_size * seqlen_q, 4, head_dim_v
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(2, 2), (4, 1)])
def test_sm120_forward_paged_kv_learnable_sink_smoke(dtype, causal, num_heads_q, num_heads_kv):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, page_size=64)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    out, lse = flash_attn_varlen_func(
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
        learnable_sink=learnable_sink,
        pack_gqa=False,
        return_lse=True,
    )
    out_ref, lse_ref = _attention_ref_paged(
        q,
        k,
        v,
        cache_seqlens,
        causal,
        learnable_sink=learnable_sink,
        return_lse=True,
    )
    out_ref = out_ref.reshape(batch_size * seqlen_q, num_heads_q, head_dim)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv,pack_gqa", [(2, 2, False), (4, 1, True)])
def test_sm120_forward_paged_kv_splitkv_learnable_sink_smoke(causal, num_heads_q, num_heads_kv, pack_gqa):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, page_size=64)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    out, lse = flash_attn_varlen_func(
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
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=2,
        return_lse=True,
    )
    out_ref, lse_ref = _attention_ref_paged(
        q,
        k,
        v,
        cache_seqlens,
        causal,
        learnable_sink=learnable_sink,
        return_lse=True,
    )
    out_ref = out_ref.reshape(batch_size * seqlen_q, num_heads_q, head_dim)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("feature", ["baseline", "learnable_sink", "score_mod_aux"])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_paged_kv_pack_gqa_smoke(feature, causal):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim = 2, 129, 257, 4, 1, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    k_paged, v_paged, page_table = _make_paged_kv(k, v, page_size=64)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    kwargs = {}
    if feature == "learnable_sink":
        learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)
        kwargs["learnable_sink"] = learnable_sink
        kwargs["return_lse"] = True
        out_ref, lse_ref = _attention_ref_paged(
            q,
            k,
            v,
            cache_seqlens,
            causal,
            learnable_sink=learnable_sink,
            return_lse=True,
        )
    elif feature == "score_mod_aux":
        kv_bias = torch.randn(seqlen_k, device="cuda", dtype=torch.bfloat16)
        kwargs["score_mod"] = score_mod_global_kv_bias
        kwargs["aux_tensors"] = [kv_bias]
        out_ref = _attention_ref_paged_kv_bias(q, k, v, cache_seqlens, kv_bias, causal)
        out_no_bias = _attention_ref_paged(q, k, v, cache_seqlens, causal)
        assert not torch.allclose(out_ref, out_no_bias)
    elif feature == "baseline":
        out_ref = _attention_ref_paged(q, k, v, cache_seqlens, causal)
    else:
        raise AssertionError(f"Unexpected feature: {feature}")

    out, lse = flash_attn_varlen_func(
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
        pack_gqa=True,
        **kwargs,
    )
    out_ref = out_ref.reshape(batch_size * seqlen_q, num_heads_q, head_dim)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    if feature == "learnable_sink":
        torch.testing.assert_close(lse.float(), lse_ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits", [1, 2])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(2, 2), (4, 1)])
def test_sm120_forward_paged_kv_score_mod_aux_tensors_smoke(
    dtype, num_splits, num_heads_q, num_heads_kv
):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    kv_bias = torch.randn(seqlen_k, device="cuda", dtype=dtype)
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
        causal=False,
        score_mod=score_mod_global_kv_bias,
        aux_tensors=[kv_bias],
        pack_gqa=False,
        num_splits=num_splits,
    )
    out_ref = _attention_ref_paged_kv_bias(q, k, v, cache_seqlens, kv_bias).reshape(
        batch_size * seqlen_q, num_heads_q, head_dim
    )
    out_no_bias = _attention_ref_paged(q, k, v, cache_seqlens, False).reshape(
        batch_size * seqlen_q, num_heads_q, head_dim
    )
    assert not torch.allclose(out_ref, out_no_bias)
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_paged_kv_packed_gqa_splitkv_smoke(dtype, causal):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim = 2, 129, 257, 4, 1, 64
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
        pack_gqa=True,
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
def test_sm120_forward_paged_kv_packed_gqa_splitkv_three_splits_smoke():
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, head_dim = 2, 129, 257, 64
    q = torch.randn(batch_size, seqlen_q, 4, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seqlen_k, 1, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seqlen_k, 1, head_dim, device="cuda", dtype=torch.float16)
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
        pack_gqa=True,
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_paged_kv_packed_gqa_splitkv_score_mod_aux_tensors_smoke(
    dtype, causal
):
    torch.manual_seed(0)
    batch_size, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim = 2, 129, 257, 4, 1, 64
    q = torch.randn(batch_size, seqlen_q, num_heads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    cache_seqlens = torch.tensor([257, 193], device="cuda", dtype=torch.int32)
    kv_bias = torch.randn(seqlen_k, device="cuda", dtype=dtype)
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
        score_mod=score_mod_global_kv_bias,
        aux_tensors=[kv_bias],
        pack_gqa=True,
        num_splits=2,
    )
    out_ref = _attention_ref_paged_kv_bias(q, k, v, cache_seqlens, kv_bias, causal).reshape(
        batch_size * seqlen_q, num_heads_q, head_dim
    )
    out_no_bias = _attention_ref_paged(q, k, v, cache_seqlens, causal).reshape(
        batch_size * seqlen_q, num_heads_q, head_dim
    )
    assert not torch.allclose(out_ref, out_no_bias)
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv,pack_gqa", [(2, 2, False), (4, 2, False), (4, 1, True)])
def test_sm120_forward_varlen_learnable_sink_smoke(dtype, causal, num_heads_q, num_heads_kv, pack_gqa):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), num_heads_q, 64, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=causal,
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        return_lse=True,
    )
    out_ref, lse_ref = _attention_ref_varlen_with_lse(
        q, k, v, cu_seqlens_q, cu_seqlens_k, causal, learnable_sink=learnable_sink
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads_q,num_heads_kv,pack_gqa", [(2, 2, False), (4, 1, True)])
def test_sm120_forward_varlen_splitkv_learnable_sink_smoke(causal, num_heads_q, num_heads_kv, pack_gqa):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 257]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), num_heads_q, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=torch.bfloat16)
    learnable_sink = torch.randn(num_heads_q, device="cuda", dtype=torch.bfloat16)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=causal,
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=2,
        return_lse=True,
    )
    out_ref, lse_ref = _attention_ref_varlen_with_lse(
        q, k, v, cu_seqlens_q, cu_seqlens_k, causal, learnable_sink=learnable_sink
    )
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads_q,num_heads_kv,pack_gqa", [(2, 2, False), (4, 2, False), (4, 1, True)])
def test_sm120_forward_varlen_score_mod_aux_tensors_smoke(dtype, num_heads_q, num_heads_kv, pack_gqa):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), num_heads_q, 64, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    kv_bias = torch.randn(sum(k_lens), device="cuda", dtype=dtype)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=False,
        score_mod=score_mod_global_kv_bias,
        aux_tensors=[kv_bias],
        pack_gqa=pack_gqa,
    )
    out_ref = _attention_ref_varlen_kv_bias(q, k, v, cu_seqlens_q, cu_seqlens_k, kv_bias)
    out_no_bias = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, False)
    assert not torch.allclose(out_ref, out_no_bias)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads_q,num_heads_kv", [(2, 2), (4, 2)])
def test_sm120_forward_varlen_splitkv_score_mod_aux_tensors_smoke(dtype, num_heads_q, num_heads_kv):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 257]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), num_heads_q, 64, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    kv_bias = torch.randn(sum(k_lens), device="cuda", dtype=dtype)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=False,
        score_mod=score_mod_global_kv_bias,
        aux_tensors=[kv_bias],
        pack_gqa=False,
        num_splits=2,
    )
    out_ref = _attention_ref_varlen_kv_bias(q, k, v, cu_seqlens_q, cu_seqlens_k, kv_bias)
    out_no_bias = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, False)
    assert not torch.allclose(out_ref, out_no_bias)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm120_forward_varlen_packed_gqa_splitkv_score_mod_aux_tensors_smoke(dtype):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 257]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), 1, 64, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lens), 1, 64, device="cuda", dtype=dtype)
    kv_bias = torch.randn(sum(k_lens), device="cuda", dtype=dtype)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=False,
        score_mod=score_mod_global_kv_bias,
        aux_tensors=[kv_bias],
        pack_gqa=True,
        num_splits=2,
    )
    out_ref = _attention_ref_varlen_kv_bias(q, k, v, cu_seqlens_q, cu_seqlens_k, kv_bias)
    out_no_bias = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, False)
    assert not torch.allclose(out_ref, out_no_bias)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads_q,num_heads_kv,pack_gqa", [(2, 2, False), (4, 2, False), (4, 1, True)])
def test_sm120_forward_varlen_mask_mod_smoke(dtype, num_heads_q, num_heads_kv, pack_gqa):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 127]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), num_heads_q, 64, device="cuda", dtype=dtype)
    k = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    v = torch.randn(sum(k_lens), num_heads_kv, 64, device="cuda", dtype=dtype)
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=False,
        mask_mod=cute_global_offset_mask,
        pack_gqa=pack_gqa,
    )
    out_ref = _attention_ref_varlen_global_offset_mask(q, k, v, cu_seqlens_q, cu_seqlens_k)
    out_no_mask = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, False)
    assert not torch.allclose(out_ref, out_no_mask)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 hardware",
)
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_varlen_packed_gqa_splitkv_smoke(causal):
    torch.manual_seed(0)
    q_lens = [17, 64, 129]
    k_lens = [19, 63, 257]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(sum(k_lens), 1, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(sum(k_lens), 1, 64, device="cuda", dtype=torch.float16)
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
        num_splits=2,
    )
    out_ref = _attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
