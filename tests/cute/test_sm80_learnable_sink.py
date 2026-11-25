"""Forward attention-sink coverage for SM80-family and SM100-family kernels."""

import math
import os

import pytest
import torch

from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.testing import attention_ref, is_fake_mode, maybe_fake_tensor_mode

SM_MAJOR = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None
USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1
requires_sm80_family = pytest.mark.skipif(
    SM_MAJOR not in (8, 12), reason="Requires an SM80-family kernel"
)


@requires_sm80_family
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,nheads,nheads_kv,d",
    [
        pytest.param(64, 64, 64, 8, 64, id="gqa"),
        pytest.param(64, 512, 64, 8, 64, id="gqa-multiblock"),
        pytest.param(128, 256, 16, 16, 128, id="mha"),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm80_learnable_sink_matches_reference(
    seqlen_q, seqlen_k, nheads, nheads_kv, d, causal
):
    device = "cuda"
    dtype = torch.bfloat16
    torch.random.manual_seed(0)
    batch_size = 2

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype) * 0.1
    k = (
        torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
        * 0.1
    )
    v = torch.randn_like(k) * 0.1
    sink = torch.randn(nheads, dtype=dtype, device=device) * 2.0

    out, lse = flash_attn_func(
        q, k, v, causal=causal, learnable_sink=sink, return_lse=True
    )
    if is_fake_mode():
        return

    out_ref, _, lse_ref = attention_ref(
        q.float(),
        k.float(),
        v.float(),
        causal=causal,
        learnable_sink=sink.float(),
        return_lse=True,
    )
    torch.testing.assert_close(out.float(), out_ref, rtol=0.01, atol=5e-4)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)


def _assert_sink_normalization(
    out: torch.Tensor,
    lse: torch.Tensor,
    log_real_mass: torch.Tensor,
    sink: torch.Tensor,
) -> None:
    # Each visible value is one, so the output is the probability mass on real tokens.
    expected_lse = torch.logaddexp(log_real_mass, sink.float()[:, None])
    expected_out = (log_real_mass - expected_lse).exp().to(out.dtype)
    torch.testing.assert_close(
        lse, expected_lse[None].expand_as(lse), atol=1e-5, rtol=1e-6
    )
    torch.testing.assert_close(
        out,
        expected_out.T[None, :, :, None].expand_as(out),
        atol=0,
        rtol=torch.finfo(out.dtype).eps,
    )


@requires_sm80_family
@pytest.mark.parametrize("sink_logit", [6.0, 100.0], ids=["moderate", "large"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm80_packed_gqa_sink_scales_output_per_head(sink_logit: float):
    q = torch.zeros(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 64, 2, 64, device=q.device, dtype=q.dtype)
    v = torch.ones_like(k)
    sink = torch.tensor([0.0, sink_logit, 3.0, 5.0], device=q.device, dtype=q.dtype)

    out, lse = flash_attn_func(
        q, k, v, pack_gqa=True, learnable_sink=sink, return_lse=True
    )
    if is_fake_mode():
        return

    log_real_mass = torch.full((), math.log(k.shape[1]), device=q.device)
    _assert_sink_normalization(out, lse, log_real_mass, sink)


@pytest.mark.skipif(SM_MAJOR not in (10, 11), reason="Requires an SM100-family kernel")
@pytest.mark.parametrize("num_splits", [1, 3], ids=["unsplit", "split-kv"])
@pytest.mark.parametrize("sink_logit", [6.0, 100.0], ids=["moderate", "large"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm100_packed_gqa_sink_scales_output_per_head(
    sink_logit: float, num_splits: int
):
    seqlen_k = 768
    q = torch.full((1, seqlen_k + 1, 4, 64), 0.5, device="cuda", dtype=torch.bfloat16)
    k = torch.full((1, seqlen_k, 2, 64), -0.5, device=q.device, dtype=q.dtype)
    v = torch.ones_like(k)
    sink = torch.tensor([0.0, sink_logit, 3.0, 5.0], device=q.device, dtype=q.dtype)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        causal=True,
        pack_gqa=True,
        learnable_sink=sink,
        num_splits=num_splits,
        return_lse=True,
    )
    if is_fake_mode():
        return

    # Bottom-right causal alignment gives row i exactly i visible keys, each with logit -2.
    # The first row has no visible keys but shares a tile with nonempty rows in split 0.
    num_visible_keys = torch.arange(seqlen_k + 1, device=q.device, dtype=torch.float32)
    log_real_mass = num_visible_keys.log() - 2.0
    _assert_sink_normalization(out, lse, log_real_mass, sink)


@requires_sm80_family
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm80_without_sink_matches_reference():
    device = "cuda"
    dtype = torch.bfloat16
    torch.random.manual_seed(0)
    q = torch.randn(2, 64, 16, 64, device=device, dtype=dtype) * 0.1
    k = torch.randn(2, 64, 16, 64, device=device, dtype=dtype) * 0.1
    v = torch.randn(2, 64, 16, 64, device=device, dtype=dtype) * 0.1

    out, _ = flash_attn_func(q, k, v, causal=True)
    if is_fake_mode():
        return

    out_ref, _ = attention_ref(q.float(), k.float(), v.float(), causal=True)
    torch.testing.assert_close(out.float(), out_ref, rtol=0.01, atol=5e-4)
