"""Native Windows SM120 forward and gradient checks against FP32 attention."""

import sys

import pytest
import torch

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    sys.platform != "win32"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="Native Windows SM120 acceptance",
)


def reference_attention(q, k, v, causal):
    repetitions = q.shape[-2] // k.shape[-2]
    k = k.repeat_interleave(repetitions, dim=-2)
    v = v.repeat_interleave(repetitions, dim=-2)
    scores = torch.einsum("shd,thd->hst", q, k) / q.shape[-1] ** 0.5
    if causal:
        mask = torch.ones(q.shape[0], k.shape[0], device="cuda", dtype=torch.bool).triu(
            1
        )
        scores = scores.masked_fill(mask, -float("inf"))
    return torch.einsum("hst,thd->shd", scores.softmax(-1), v)


def check_output_and_gradients(actual, expected, inputs, references):
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient.float())
    for value, reference in zip(inputs, references):
        torch.testing.assert_close(
            value.grad.float(), reference.grad, atol=3e-2, rtol=3e-2
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads,dimension", [(4, 64), (2, 128)])
def test_dense_attention(dtype, causal, kv_heads, dimension):
    torch.manual_seed(1234)
    q = torch.randn(128, 4, dimension, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [
        torch.randn(
            128, kv_heads, dimension, device="cuda", dtype=dtype, requires_grad=True
        )
        for _ in range(2)
    ]
    references = [value.detach().float().requires_grad_() for value in (q, k, v)]
    expected = reference_attention(*references, causal)
    actual, _ = flash_attn_func(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=causal
    )
    check_output_and_gradients(actual.squeeze(0), expected, (q, k, v), references)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ragged_attention(dtype):
    torch.manual_seed(1234)
    q, k, v = [
        torch.randn(82, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    references = [value.detach().float().requires_grad_() for value in (q, k, v)]
    expected = torch.cat(
        [
            reference_attention(*(value[start:end] for value in references), True)
            for start, end in [(0, 17), (17, 82)]
        ]
    )
    offsets = torch.tensor([0, 17, 82], device="cuda", dtype=torch.int32)
    actual, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=offsets,
        cu_seqlens_k=offsets,
        max_seqlen_q=65,
        max_seqlen_k=65,
        causal=True,
    )
    check_output_and_gradients(actual, expected, (q, k, v), references)
