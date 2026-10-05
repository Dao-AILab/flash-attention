import math

import pytest
import torch

from flash_attn.cute.testing import attention_ref

IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

if IS_SM100:
    from flash_attn.cute.flash_fwd_decode_sm100 import flash_attn_decode_func

pytestmark = pytest.mark.skipif(not IS_SM100, reason="SM100 swap-AB decode kernel")


def _ref_lse(q, k, seqused_k=None):
    # fp32 log-sum-exp of the scaled scores, (batch, num_heads, 1)
    g = q.shape[2] // k.shape[2]
    kr = k.float().repeat_interleave(g, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q.float(), kr) / math.sqrt(q.shape[-1])
    if seqused_k is not None:
        mask = torch.arange(k.shape[1], device=k.device)[None, :] >= seqused_k[:, None]
        scores.masked_fill_(mask[:, None, None, :], float("-inf"))
    return torch.logsumexp(scores, dim=-1)


def _check(out, q, k, v, key_padding_mask=None):
    out_ref, _ = attention_ref(q, k, v, key_padding_mask=key_padding_mask)
    out_pt, _ = attention_ref(
        q, k, v, key_padding_mask=key_padding_mask, upcast=False, reorder_ops=True
    )
    err = (out.float() - out_ref.float()).abs().max().item()
    pt_err = (out_pt.float() - out_ref.float()).abs().max().item()
    # Same criterion as the other FA4 forward tests: at most ~2x the error of a bf16/fp16
    # PyTorch implementation.
    assert err <= 2 * pt_err + 1e-5, f"max err {err} vs pytorch {pt_err}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d", [64, 128, 192, 256])
@pytest.mark.parametrize(
    "nheads,nheads_kv",
    [(16, 16), (16, 2), (16, 1), (32, 8), (40, 8), (64, 8), (128, 4), (256, 4)],
)
@pytest.mark.parametrize("seqlen_k", [1, 113, 256, 1000, 4096, 32768])
@pytest.mark.parametrize("batch", [1, 3])
def test_flash_attn_decode_output(batch, seqlen_k, nheads, nheads_kv, d, dtype):
    if d >= 192 and nheads // nheads_kv > 16:
        pytest.skip(
            "hdim >= 192 with > 16 query heads per KV head does not fit in SMEM"
        )
    torch.random.manual_seed(0)
    q = torch.randn(batch, 1, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    out, lse = flash_attn_decode_func(q, k, v, return_lse=True)
    _check(out, q, k, v)
    assert (lse - _ref_lse(q, k)).abs().max().item() < 1e-3
    # Workspaces are uninitialized and reused: a second call must give the same answer.
    out2 = flash_attn_decode_func(q, k, v)
    assert torch.equal(out, out2)


@pytest.mark.parametrize("num_splits", [1, 3, 8, 32])
@pytest.mark.parametrize("seqlen_k", [700, 5000])
def test_flash_attn_decode_num_splits(seqlen_k, num_splits):
    torch.random.manual_seed(0)
    q = torch.randn(2, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, seqlen_k, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, seqlen_k, 8, 128, device="cuda", dtype=torch.bfloat16)
    _check(flash_attn_decode_func(q, k, v, num_splits=num_splits), q, k, v)


@pytest.mark.parametrize("nheads,nheads_kv", [(32, 8), (16, 16)])
@pytest.mark.parametrize("d", [64, 128])
def test_flash_attn_decode_seqused_k(nheads, nheads_kv, d):
    torch.random.manual_seed(0)
    batch, seqlen_k = 6, 9000
    q = torch.randn(batch, 1, nheads, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=torch.bfloat16)
    seqused_k = torch.tensor(
        [9000, 1, 255, 257, 4321, 0], device="cuda", dtype=torch.int32
    )
    out, lse = flash_attn_decode_func(q, k, v, seqused_k=seqused_k, return_lse=True)
    key_padding_mask = (
        torch.arange(seqlen_k, device="cuda")[None, :] < seqused_k[:, None]
    )
    nonempty = seqused_k > 0
    _check(
        out[nonempty], q[nonempty], k[nonempty], v[nonempty], key_padding_mask[nonempty]
    )
    assert torch.all(out[~nonempty] == 0)
    lse_ref = _ref_lse(q, k, seqused_k)
    assert (lse[nonempty] - lse_ref[nonempty]).abs().max().item() < 1e-3
    assert torch.all(torch.isneginf(lse[~nonempty]))


def test_flash_attn_decode_strided_kv_cache():
    # K/V as a slice of a larger, preallocated cache (non-contiguous batch and seqlen strides).
    torch.random.manual_seed(0)
    cache_k = torch.randn(4, 8192, 8, 128, device="cuda", dtype=torch.bfloat16)
    cache_v = torch.randn(4, 8192, 8, 128, device="cuda", dtype=torch.bfloat16)
    k, v = cache_k[1:3, :3000], cache_v[1:3, :3000]
    q = torch.randn(2, 1, 64, 128, device="cuda", dtype=torch.bfloat16)
    _check(flash_attn_decode_func(q, k, v), q, k, v)


def test_flash_attn_decode_softmax_scale():
    torch.random.manual_seed(0)
    q = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2048, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2048, 4, 128, device="cuda", dtype=torch.bfloat16)
    scale = 0.3
    out = flash_attn_decode_func(q, k, v, softmax_scale=scale)
    # attention_ref uses 1/sqrt(d); fold the custom scale into q
    _check(out, q * (scale * math.sqrt(128)), k, v)


@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize("reduction_mode", ["kernel", "atomic", "auto"])
@pytest.mark.parametrize(
    "batch,seqlen_k,nheads,nheads_kv,d,num_splits",
    [
        (1, 32768, 64, 8, 128, 0),
        (1, 32768, 64, 8, 128, 8),
        (1, 16384, 16, 2, 64, 16),
        (2, 3000, 32, 8, 128, 0),
        (3, 5000, 16, 16, 128, 4),
        (1, 131072, 16, 1, 128, 0),
        (2, 1000, 16, 1, 256, 2),
        (4, 113, 40, 8, 64, 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flash_attn_decode_reduction_mode(
    batch, seqlen_k, nheads, nheads_kv, d, num_splits, reduction_mode, use_pdl, dtype
):
    torch.random.manual_seed(0)
    q = torch.randn(batch, 1, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    # A garbage-filled `out`: atomic mode must zero it before accumulating.
    out = torch.full_like(q, 7.0)
    out, lse = flash_attn_decode_func(
        q,
        k,
        v,
        num_splits=num_splits,
        return_lse=True,
        out=out,
        reduction_mode=reduction_mode,
        use_pdl=use_pdl,
    )
    _check(out, q, k, v)
    assert (lse - _ref_lse(q, k)).abs().max().item() < 1e-3


@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize("reduction_mode", ["atomic", "auto"])
def test_flash_attn_decode_reduction_mode_seqused_k(reduction_mode, use_pdl):
    torch.random.manual_seed(0)
    batch, seqlen_k, nheads, nheads_kv, d = 6, 9000, 32, 8, 128
    q = torch.randn(batch, 1, nheads, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen_k, nheads_kv, d, device="cuda", dtype=torch.bfloat16)
    seqused_k = torch.tensor(
        [9000, 1, 255, 257, 4321, 0], device="cuda", dtype=torch.int32
    )
    out, lse = flash_attn_decode_func(
        q,
        k,
        v,
        seqused_k=seqused_k,
        return_lse=True,
        num_splits=8,
        reduction_mode=reduction_mode,
        use_pdl=use_pdl,
    )
    key_padding_mask = (
        torch.arange(seqlen_k, device="cuda")[None, :] < seqused_k[:, None]
    )
    nonempty = seqused_k > 0
    _check(
        out[nonempty], q[nonempty], k[nonempty], v[nonempty], key_padding_mask[nonempty]
    )
    assert torch.all(out[~nonempty] == 0)
    lse_ref = _ref_lse(q, k, seqused_k)
    assert (lse[nonempty] - lse_ref[nonempty]).abs().max().item() < 1e-3
    assert torch.all(torch.isneginf(lse[~nonempty]))
