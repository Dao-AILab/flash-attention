# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import os
import random

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import flash_attn.cute.interface as interface
from flash_attn.cute.cache_utils import JITCache
from flash_attn.cute.testing import (
    maybe_fake_tensor_mode,
    is_fake_mode,
)
from flash_attn.cute.interface import (
    _flash_attn_fwd,
    _flash_attn_fwd_combine,
    flash_attn_combine,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1


def attention_combine_ref(out_partial, lse_partial):
    """
    out_partial: (num_splits, batch_size, seqlen, nheads, d)
    lse_partial: (num_splits, batch_size, seqlen, nheads)
    """
    lse = torch.logsumexp(lse_partial, dim=0)
    scale = torch.exp(lse_partial - lse)
    scale = torch.where(
        torch.isinf(scale) | torch.isnan(scale), torch.zeros_like(scale), scale
    )
    out = (scale.unsqueeze(-1) * out_partial).sum(0)
    return out, lse


def check_combine_results(out, lse, out_ref, lse_ref, dtype):
    """Check combine kernel output against reference for a single (seqlen, nheads, d) chunk."""
    out_pt = out_ref.to(dtype)
    print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}, "
          f"Output max diff: {(out - out_ref).abs().max().item()}, "
          f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    assert torch.allclose(lse, lse_ref, atol=1e-5, rtol=1e-5)
    assert (
        (out - out_ref).abs().max().item()
        <= 2 * (out_pt - out_ref).abs().max().item()
    ) or torch.allclose(out, out_pt, atol=1e-5, rtol=1e-5)


def test_splitkv_forwards_explicit_arch(monkeypatch):
    class HitCache:
        def __contains__(self, _key):
            return True

    def unexpected_arch_query():
        raise AssertionError("explicit architecture was not forwarded to combine")

    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", HitCache())
    monkeypatch.setattr(_flash_attn_fwd_combine, "compile_cache", HitCache())
    monkeypatch.setattr(interface, "_get_device_arch", unexpected_arch_query)
    with FakeTensorMode():
        q = torch.empty(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.empty(1, 1024, 8, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.empty_like(k)
        _flash_attn_fwd(q, k, v, num_splits=2, _arch=100)


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="Runtime combine optional-input test")
def test_flash_attn_combine_dynamic_splits_and_semaphore(monkeypatch):
    torch.manual_seed(0)
    num_splits, batch, seqlen, heads, head_dim = 3, 2, 17, 4, 64
    out_partial = torch.randn(
        num_splits,
        batch,
        seqlen,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    lse_partial = torch.randn(
        num_splits, batch, heads, seqlen, device="cuda", dtype=torch.float32
    ).transpose(-1, -2)
    out = torch.empty(
        batch, seqlen, heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    lse = torch.empty(
        batch, heads, seqlen, device="cuda", dtype=torch.float32
    ).transpose(-1, -2)
    dynamic_splits = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    semaphore = torch.ones(1, device="cuda", dtype=torch.int32)

    full_reference = attention_combine_ref(out_partial, lse_partial)
    dynamic_references = [
        attention_combine_ref(
            out_partial[:split_count, batch_idx : batch_idx + 1],
            lse_partial[:split_count, batch_idx : batch_idx + 1],
        )
        for batch_idx, split_count in enumerate((2, 3))
    ]
    dynamic_reference = (
        torch.cat([reference[0] for reference in dynamic_references]),
        torch.cat([reference[1] for reference in dynamic_references]),
    )
    cache = JITCache()
    monkeypatch.setattr(_flash_attn_fwd_combine, "compile_cache", cache)

    cases = (
        (None, None, full_reference),
        (dynamic_splits, None, dynamic_reference),
        (None, semaphore, full_reference),
        (dynamic_splits, semaphore, dynamic_reference),
    )
    for expected_cache_size, (dynamic_ptr, semaphore_ptr, reference) in enumerate(cases, 1):
        if semaphore_ptr is not None:
            semaphore_ptr.fill_(1)
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial,
            out,
            lse,
            num_splits_dynamic_ptr=dynamic_ptr,
            semaphore_to_reset=semaphore_ptr,
        )
        check_combine_results(out, lse, *reference, torch.bfloat16)
        assert len(cache.cache) == expected_cache_size
        if semaphore_ptr is not None:
            assert semaphore_ptr.item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("d", [64, 96, 128, 192, 256, 512])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen", [1, 2, 3, 32, 64, 256, 113, 108, 640, 1024])
# @pytest.mark.parametrize("seqlen", [12, 32, 64, 256, 112, 108, 640, 1024, 2048, 8192])
# @pytest.mark.parametrize("seqlen", [15])
@pytest.mark.parametrize("num_splits", [1, 2, 3, 5, 17, 32, 55, 97, 133])
# @pytest.mark.parametrize("num_splits", [1, 2, 3, 5, 11])
# @pytest.mark.parametrize("num_splits", [11])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_combine(num_splits, seqlen, d, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(1)
    batch_size = 5
    nheads = 16
    # batch_size = 1
    # nheads = 1
    # Create tensors in the expected format: (num_splits, batch_size, seqlen, nheads, d) and (num_splits, batch_size, seqlen, nheads)
    out_partial = torch.randn(
        num_splits * 2,
        batch_size,
        nheads,
        seqlen,
        d,
        device=device,
        dtype=torch.float32,
    ).transpose(2, 3)[:num_splits]  # To test non-contiguous tensor
    lse_partial = torch.randn(
        num_splits, batch_size, nheads * 2, seqlen, device=device, dtype=torch.float32
    ).transpose(-1, -2)[:, :, :, :nheads]  # To test non-contiguous tensor
    # To test short-circuiting based on num_splits
    lse_partial[num_splits // 2 :, : batch_size // 3] = -float("inf")

    # Test with LSE returned (default behavior)
    out, lse = flash_attn_combine(
        out_partial, lse_partial, out_dtype=dtype, return_lse=True
    )
    if is_fake_mode():
        return
    out_ref, lse_ref = attention_combine_ref(out_partial, lse_partial)
    check_combine_results(out, lse, out_ref, lse_ref, dtype)

    # Test with LSE not returned
    out_no_lse, lse_no_lse = flash_attn_combine(
        out_partial, lse_partial, out_dtype=dtype, return_lse=False
    )
    assert lse_no_lse is None, "LSE should be None when return_lse=False"
    assert torch.allclose(out_no_lse, out, atol=1e-5, rtol=1e-5), (
        "Output should be the same regardless of return_lse"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [64, 96, 128, 256])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen", [1, 32, 113, 256, 1024])
# @pytest.mark.parametrize("seqlen", [113])
@pytest.mark.parametrize("num_splits", [2, 5, 17, 55])
# @pytest.mark.parametrize("num_splits", [5])
@pytest.mark.parametrize(
    "varlen_mode",
    ["cu_seqlens", "seqused", "cu_seqlens_seqused"],
)
# @pytest.mark.parametrize("varlen_mode", ["cu_seqlens"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_combine_varlen(varlen_mode, num_splits, seqlen, d, dtype):
    device = "cuda"
    torch.random.manual_seed(1)
    random.seed(1)
    batch_size = 3
    nheads = 8
    use_cu_seqlens = "cu_seqlens" in varlen_mode
    use_seqused = "seqused" in varlen_mode

    # Generate variable-length sequences
    seqlens_list = [random.randint(1, seqlen) for _ in range(batch_size)]
    seqlens = torch.tensor(seqlens_list, device=device, dtype=torch.int32)
    # For cu_seqlens+seqused mode, seqused < seqlen (kernel processes fewer tokens)
    if use_cu_seqlens and use_seqused:
        seqused_list = [
            max(1, s - random.randrange(0, max(1, seqlen // 4))) for s in seqlens_list
        ]
    else:
        seqused_list = seqlens_list
    seqused_vals = torch.tensor(seqused_list, device=device, dtype=torch.int32)

    if use_cu_seqlens:
        # Packed varlen layout: (num_splits, total_q, nheads, d)
        total_q = sum(seqlens_list)
        cu_seqlens_q = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens, dim=0)

        out_partial = torch.randn(
            num_splits * 2, total_q, nheads, d, device=device, dtype=torch.float32,
        )[:num_splits]  # Non-contiguous in splits dim
        # lse_partial needs stride(-2)==1 (seqlen dim contiguous)
        lse_partial = torch.randn(
            num_splits, nheads, total_q, device=device, dtype=torch.float32
        ).transpose(-1, -2)
        lse_partial[num_splits // 2:, :total_q // 3] = -float("inf")

        out, lse = flash_attn_combine(
            out_partial, lse_partial, out_dtype=dtype,
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_vals if use_seqused else None,
            return_lse=True,
        )
        if is_fake_mode():
            return

        # Reference on full packed tensor
        out_ref, lse_ref = attention_combine_ref(
            out_partial.unsqueeze(1), lse_partial.unsqueeze(1)
        )
        out_ref = out_ref.squeeze(0)
        lse_ref = lse_ref.squeeze(0)

        # Validate per-batch (only seqused_vals tokens are guaranteed correct)
        for i in range(batch_size):
            start = cu_seqlens_q[i].item()
            sl = seqused_vals[i].item()
            check_combine_results(
                out[start:start + sl], lse[start:start + sl],
                out_ref[start:start + sl], lse_ref[start:start + sl], dtype,
            )

        # Also test return_lse=False
        out_no_lse, lse_no_lse = flash_attn_combine(
            out_partial, lse_partial, out_dtype=dtype,
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_vals if use_seqused else None,
            return_lse=False,
        )
        assert lse_no_lse is None
        # Only compare valid positions (beyond seqused, output is undefined)
        for i in range(batch_size):
            start = cu_seqlens_q[i].item()
            sl = seqused_vals[i].item()
            assert torch.allclose(out_no_lse[start:start + sl], out[start:start + sl], atol=1e-5, rtol=1e-5)

    else:
        # seqused only — batched layout: (num_splits, batch, max_seqlen, nheads, d)
        max_seqlen = max(seqlens_list)
        out_partial = torch.randn(
            num_splits, batch_size, max_seqlen, nheads, d, device=device, dtype=torch.float32,
        )
        # lse_partial needs stride(-2)==1 (seqlen dim contiguous)
        lse_partial = torch.randn(
            num_splits, batch_size, nheads, max_seqlen, device=device, dtype=torch.float32,
        ).transpose(-1, -2)
        lse_partial[num_splits // 2:, :batch_size // 2] = -float("inf")
        # Zero out / -inf beyond seqused so reference matches kernel
        for i, sl in enumerate(seqlens_list):
            out_partial[:, i, sl:] = 0
            lse_partial[:, i, sl:] = -float("inf")

        out, lse = flash_attn_combine(
            out_partial, lse_partial, out_dtype=dtype, seqused=seqlens, return_lse=True,
        )
        if is_fake_mode():
            return

        out_ref, lse_ref = attention_combine_ref(out_partial, lse_partial)

        # Validate per-batch (only seqused tokens)
        for i in range(batch_size):
            sl = seqlens[i].item()
            check_combine_results(
                out[i, :sl], lse[i, :sl],
                out_ref[i, :sl], lse_ref[i, :sl], dtype,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen", [32, 113, 256])
# @pytest.mark.parametrize("seqlen", [113])
@pytest.mark.parametrize("num_splits", [2, 5, 17])
# @pytest.mark.parametrize("num_splits", [5])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_combine_virtual_batch_idx(num_splits, seqlen, d, dtype):
    """Test that virtual_batch_idx correctly remaps virtual batch indices to real batch indices.

    virtual_batch_idx maps blockIdx.z (virtual batch) -> real batch index. The kernel
    reads AND writes using the remapped batch_idx, so with a permutation the output
    should match running without virtual_batch_idx (each real batch is processed once).

    We also test with seqused to verify interaction with variable-length sequences.
    """
    device = "cuda"
    torch.random.manual_seed(42)
    random.seed(42)
    batch_size = 4
    nheads = 8

    # Create batched input data
    out_partial = torch.randn(
        num_splits, batch_size, seqlen, nheads, d, device=device, dtype=torch.float32,
    )
    lse_partial = torch.randn(
        num_splits, batch_size, nheads, seqlen, device=device, dtype=torch.float32,
    ).transpose(-1, -2)  # stride(-2)==1
    lse_partial[num_splits // 2:, :batch_size // 2] = -float("inf")

    # Create a permuted batch index mapping: virtual batch -> real batch
    perm = torch.tensor([2, 0, 3, 1], device=device, dtype=torch.int32)
    assert perm.shape[0] == batch_size

    # Also test with seqused to verify interaction with virtual_batch_idx
    seqused_list = [random.randint(1, seqlen) for _ in range(batch_size)]
    seqused = torch.tensor(seqused_list, device=device, dtype=torch.int32)
    # Zero out / -inf beyond seqused so reference matches kernel
    for i, sl in enumerate(seqused_list):
        out_partial[:, i, sl:] = 0
        lse_partial[:, i, sl:] = -float("inf")

    # Run with virtual_batch_idx and seqused via public API
    out, lse = flash_attn_combine(
        out_partial, lse_partial, out_dtype=dtype,
        seqused=seqused,
        virtual_batch_idx=perm,
        return_lse=True,
    )
    if is_fake_mode():
        return

    # Reference: standard combine (no remapping needed since perm is a bijection
    # and both reads and writes use the remapped batch_idx)
    out_ref, lse_ref = attention_combine_ref(out_partial, lse_partial)

    # The kernel reads from input[perm[v]] and writes to output[perm[v]],
    # so the net result is output[b] = combine(input[b]) for all b.
    for b in range(batch_size):
        sl = seqused[b].item()
        check_combine_results(
            out[b, :sl], lse[b, :sl],
            out_ref[b, :sl], lse_ref[b, :sl], dtype,
        )
