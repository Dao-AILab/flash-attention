import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flash_attn.cute import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)


def _leaf_copy(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _assert_outputs_and_grads_match(packed_result, unpacked_result, packed, unpacked):
    packed_out, packed_lse = packed_result
    unpacked_out, unpacked_lse = unpacked_result
    torch.testing.assert_close(packed_out, unpacked_out, atol=0, rtol=0)
    torch.testing.assert_close(packed_lse, unpacked_lse, atol=0, rtol=0)

    torch.manual_seed(1)
    dout = torch.randn_like(packed_out)
    dlse = torch.randn_like(packed_lse)
    packed_grads = torch.autograd.grad(packed_result, packed, (dout, dlse))
    unpacked_grads = torch.autograd.grad(unpacked_result, unpacked, (dout, dlse))
    for packed_grad, unpacked_grad in zip(packed_grads, unpacked_grads):
        torch.testing.assert_close(packed_grad, unpacked_grad, atol=0.02, rtol=0.02)
        assert packed_grad.is_contiguous()


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_qkvpacked_matches_unpacked(causal):
    torch.manual_seed(0)
    packed = torch.randn(2, 48, 3, 4, 64, device="cuda", dtype=torch.bfloat16)
    packed_input = _leaf_copy(packed)
    unpacked_input = _leaf_copy(packed)

    packed_result = flash_attn_qkvpacked_func(
        packed_input, causal=causal, return_lse=True
    )
    unpacked_result = flash_attn_func(
        *unpacked_input.unbind(dim=-3), causal=causal, return_lse=True
    )

    _assert_outputs_and_grads_match(
        packed_result, unpacked_result, (packed_input,), (unpacked_input,)
    )


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_kvpacked_gqa_matches_unpacked(causal):
    torch.manual_seed(0)
    q = torch.randn(2, 40, 4, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 56, 2, 2, 64, device="cuda", dtype=torch.bfloat16)
    packed_q, packed_kv = _leaf_copy(q), _leaf_copy(kv)
    unpacked_q, unpacked_kv = _leaf_copy(q), _leaf_copy(kv)

    packed_result = flash_attn_kvpacked_func(
        packed_q, packed_kv, causal=causal, return_lse=True
    )
    unpacked_result = flash_attn_func(
        unpacked_q,
        *unpacked_kv.unbind(dim=-3),
        causal=causal,
        return_lse=True,
    )

    _assert_outputs_and_grads_match(
        packed_result,
        unpacked_result,
        (packed_q, packed_kv),
        (unpacked_q, unpacked_kv),
    )


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_qkvpacked_matches_unpacked(causal):
    torch.manual_seed(0)
    cu_seqlens = torch.tensor([0, 17, 48], device="cuda", dtype=torch.int32)
    packed = torch.randn(48, 3, 4, 64, device="cuda", dtype=torch.bfloat16)
    packed_input = _leaf_copy(packed)
    unpacked_input = _leaf_copy(packed)

    packed_result = flash_attn_varlen_qkvpacked_func(
        packed_input,
        cu_seqlens,
        31,
        causal=causal,
        return_lse=True,
    )
    unpacked_result = flash_attn_varlen_func(
        *unpacked_input.unbind(dim=-3),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=31,
        max_seqlen_k=31,
        causal=causal,
        return_lse=True,
    )

    _assert_outputs_and_grads_match(
        packed_result, unpacked_result, (packed_input,), (unpacked_input,)
    )


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_kvpacked_gqa_matches_unpacked(causal):
    torch.manual_seed(0)
    cu_seqlens_q = torch.tensor([0, 17, 48], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 23, 60], device="cuda", dtype=torch.int32)
    q = torch.randn(48, 4, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(60, 2, 2, 64, device="cuda", dtype=torch.bfloat16)
    packed_q, packed_kv = _leaf_copy(q), _leaf_copy(kv)
    unpacked_q, unpacked_kv = _leaf_copy(q), _leaf_copy(kv)

    packed_result = flash_attn_varlen_kvpacked_func(
        packed_q,
        packed_kv,
        cu_seqlens_q,
        cu_seqlens_k,
        31,
        37,
        causal=causal,
        return_lse=True,
    )
    unpacked_result = flash_attn_varlen_func(
        unpacked_q,
        *unpacked_kv.unbind(dim=-3),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=31,
        max_seqlen_k=37,
        causal=causal,
        return_lse=True,
    )

    _assert_outputs_and_grads_match(
        packed_result,
        unpacked_result,
        (packed_q, packed_kv),
        (unpacked_q, unpacked_kv),
    )


@pytest.mark.parametrize("api", ["qkv", "kv", "varlen_qkv", "varlen_kv"])
def test_packed_apis_work_with_torch_compile(api):
    torch.manual_seed(0)

    if api == "qkv":
        inputs = (torch.randn(1, 32, 3, 4, 64, device="cuda", dtype=torch.bfloat16),)

        def fn(qkv):
            return flash_attn_qkvpacked_func(qkv, causal=True)[0]

    elif api == "kv":
        inputs = (
            torch.randn(1, 24, 4, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 32, 2, 2, 64, device="cuda", dtype=torch.bfloat16),
        )

        def fn(q, kv):
            return flash_attn_kvpacked_func(q, kv, causal=True)[0]

    elif api == "varlen_qkv":
        inputs = (
            torch.randn(48, 3, 4, 64, device="cuda", dtype=torch.bfloat16),
            torch.tensor([0, 17, 48], device="cuda", dtype=torch.int32),
        )

        def fn(qkv, cu_seqlens):
            return flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, 31, causal=True)[0]

    else:
        inputs = (
            torch.randn(48, 4, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(60, 2, 2, 64, device="cuda", dtype=torch.bfloat16),
            torch.tensor([0, 17, 48], device="cuda", dtype=torch.int32),
            torch.tensor([0, 23, 60], device="cuda", dtype=torch.int32),
        )

        def fn(q, kv, cu_seqlens_q, cu_seqlens_k):
            return flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens_q,
                cu_seqlens_k,
                31,
                37,
                causal=True,
            )[0]

    expected = fn(*inputs)
    compiled_fn = torch.compile(fn, backend="eager")
    actual = compiled_fn(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_packed_apis_support_fake_tensor_forward_and_backward():
    with FakeTensorMode():
        qkv = torch.empty(
            1, 32, 3, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out, lse = flash_attn_qkvpacked_func(qkv, return_lse=True)
        (dqkv,) = torch.autograd.grad(out.sum(), qkv)
        assert isinstance(out, FakeTensor)
        assert out.shape == (1, 32, 4, 64)
        assert lse.shape == (1, 4, 32)
        assert dqkv.shape == qkv.shape

        q = torch.empty(
            1, 24, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        kv = torch.empty(
            1, 32, 2, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out, _ = flash_attn_kvpacked_func(q, kv)
        dq, dkv = torch.autograd.grad(out.sum(), (q, kv))
        assert out.shape == q.shape
        assert dq.shape == q.shape
        assert dkv.shape == kv.shape

        cu_seqlens = torch.tensor([0, 17, 48], device="cuda", dtype=torch.int32)
        qkv = torch.empty(
            48, 3, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out, _ = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, 31)
        (dqkv,) = torch.autograd.grad(out.sum(), qkv)
        assert out.shape == (48, 4, 64)
        assert dqkv.shape == qkv.shape

        q = torch.empty(
            48, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        kv = torch.empty(
            48, 2, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out, _ = flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens, cu_seqlens, 31, 31)
        dq, dkv = torch.autograd.grad(out.sum(), (q, kv))
        assert out.shape == q.shape
        assert dq.shape == q.shape
        assert dkv.shape == kv.shape


@pytest.mark.parametrize(
    "function,args,match",
    [
        (flash_attn_qkvpacked_func, (torch.empty(1, 2, 4, 3, 8),), "size 3"),
        (
            flash_attn_kvpacked_func,
            (torch.empty(1, 2, 3, 8), torch.empty(1, 2, 3, 3, 8)),
            "size 2",
        ),
        (
            flash_attn_varlen_qkvpacked_func,
            (torch.empty(1, 2, 3, 4, 8),),
            "4 dimensions",
        ),
        (
            flash_attn_varlen_kvpacked_func,
            (torch.empty(1, 3, 8), torch.empty(1, 2, 3, 4, 8)),
            "4 dimensions",
        ),
    ],
)
def test_packed_apis_reject_invalid_layouts(function, args, match):
    with pytest.raises(ValueError, match=match):
        function(*args)
