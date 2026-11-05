"""Shared inputs, oracle, tolerances, and backend selection checks for the SM100 SwapAB decode
tests."""

import math
from typing import NamedTuple

import pytest
import torch

from flash_attn.cute import interface
from flash_attn.cute.config import FwdConfig
from flash_attn.cute.testing import attention_ref, check_tensor_vs_ref

SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))

# Finite value for the padding contract's "finite garbage" slots, far outside the data range.
FINITE_GARBAGE = 3e4

# LSE is fp32 in both the kernel and the oracle, so only exp2 approximation and summation
# order differ. Empirical: the decode suite shows <= 1.5e-5 on B200; 1e-4 keeps >= 6x headroom.
LSE_ATOL = 1e-4


def record_fwd_configs(monkeypatch: pytest.MonkeyPatch) -> list[FwdConfig]:
    """Spy on the forward's config selection: returns the list of configs it selects from now
    on. Calls with an explicit config skip selection."""
    configs = []
    select = interface.select_fwd_config

    def record(inputs):
        configs.append(select(inputs))
        return configs[-1]

    monkeypatch.setattr(interface, "select_fwd_config", record)
    return configs


def require_swapab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request SwapAB through FLASH_ATTENTION_SM100_BACKEND=swapab and fail every selection that
    falls back to the standard kernel, which the request alone does silently. The check stays
    installed for the whole test; a later auto or standard setting passes it trivially."""
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "swapab")
    select = interface.select_fwd_config

    def select_swapab(inputs):
        config = select(inputs)
        assert config.use_swapab or not inputs.requested_use_swapab, (
            f"SwapAB request fell back to the standard kernel: {inputs}"
        )
        return config

    monkeypatch.setattr(interface, "select_fwd_config", select_swapab)


def decode_inputs(
    dq=128,
    dv=128,
    *,
    ratio=3,
    length=1025,
    dtype=torch.bfloat16,
    batch=2,
    heads_kv=2,
    seed=42,
):
    """Single-query GQA inputs; K/V have non-contiguous head strides."""
    torch.manual_seed(seed)
    q = torch.randn(batch, 1, ratio * heads_kv, dq, device="cuda", dtype=dtype)
    k = torch.randn(batch, length, 2 * heads_kv, dq, device="cuda", dtype=dtype)
    v = torch.randn(batch, length, 2 * heads_kv, dv, device="cuda", dtype=dtype)
    return q, k[:, :, ::2], v[:, :, ::2]


def paged_inputs(page_size, dq=128, dv=128, *, dtype=torch.bfloat16, seed=17):
    """Nine strided physical pages; page 7 is shared and the mappings are permuted.

    Returns (q, k, v, page_table, seqused) with used lengths [page + 17, 3 * page - 9].
    """
    torch.manual_seed(seed)
    q = torch.randn(2, 1, 6, dq, device="cuda", dtype=dtype)
    k = torch.randn(9, page_size, 4, dq, device="cuda", dtype=dtype)[:, :, ::2]
    v = torch.randn(9, page_size, 4, dv, device="cuda", dtype=dtype)[:, :, ::2]
    table = torch.tensor([[7, 2, 5], [1, 7, 3]], device="cuda", dtype=torch.int32)
    used = torch.tensor(
        [page_size + 17, 3 * page_size - 9], device="cuda", dtype=torch.int32
    )
    return q, k, v, table, used


def poison_unused_kv(k, v, used, page_table=None):
    """Fill logically unused K/V storage in place per the forward's padding contract.

    Dense KV: every token past ``used`` holds NaN (test_flash_attn_sm100_unused_v_cache).
    Paged KV: tokens past ``used`` inside the last partially used page hold finite garbage, and
    pages no batch entry needs hold NaN (test_flash_attn_paged_padding_contract). ``0 * NaN``
    through the PV product is the classic decode bug this guards against.
    """
    if page_table is None:
        token = torch.arange(k.shape[1], device=k.device)[None, :]
        unused = token >= used.clamp(min=0)[:, None]
        garbage = torch.zeros_like(unused)
    else:
        page_size = k.shape[1]
        pages = torch.arange(page_table.shape[1], device=k.device)[None, :]
        tokens_in_page = (used[:, None] - pages * page_size).clamp(0, page_size)
        extent = torch.zeros(k.shape[0], dtype=torch.long, device=k.device)
        extent.scatter_reduce_(
            0, page_table.long().flatten(), tokens_in_page.flatten().long(), "amax"
        )
        offset = torch.arange(page_size, device=k.device)[None, :]
        garbage = (offset >= extent[:, None]) & (extent[:, None] > 0)
        unused = (extent == 0)[:, None].expand_as(garbage)
    for t in (k, v):
        t.masked_fill_(garbage[:, :, None, None], FINITE_GARBAGE)
        t.masked_fill_(unused[:, :, None, None], float("nan"))


class DecodeReference(NamedTuple):
    out: torch.Tensor  # fp32 (batch, 1, heads, head_dim_v)
    lse: torch.Tensor  # fp32 natural-log (batch, heads, 1)
    out_pt: torch.Tensor  # low-precision eager result, the error baseline for ``out``


def decode_reference(
    q,
    k,
    v,
    *,
    page_table=None,
    seqused=None,
    keep=None,
    bias=None,
    causal=False,
    window=None,
    softcap=0.0,
    softmax_scale=None,
):
    """Independent oracle for single-query attention built on ``attention_ref``.

    ``q`` may already be fp32 (e.g. pre-scaled per head); ``page_table`` gathers logical K/V;
    ``seqused`` truncates each batch entry; ``keep`` is a bool key mask of shape (batch, seqlen_k) or (batch, heads, seqlen_k); ``bias`` is an fp32
    additive score term of shape (batch, heads, seqlen_k); ``window`` is the left window
    relative to the last used token. K/V past ``seqused`` are zeroed so NaN poison in logically
    unused storage cannot reach the oracle. Fully masked rows have zero output and -inf LSE.
    """
    if page_table is not None:
        k = k[page_table.long()].flatten(1, 2)
        v = v[page_table.long()].flatten(1, 2)
    key_mask = None
    if seqused is not None:
        key_mask = torch.arange(k.shape[1], device=k.device)[None, :] < seqused[:, None]
        k = k.masked_fill(~key_mask[:, :, None, None], 0)
        v = v.masked_fill(~key_mask[:, :, None, None], 0)
    if keep is not None and keep.dim() == 2:
        key_mask = keep if key_mask is None else key_mask & keep
    attn_bias = None if bias is None else bias.float()[:, :, None, :]
    if keep is not None and keep.dim() == 3:
        head_mask = torch.zeros(keep.shape, device=k.device).masked_fill(~keep, -math.inf)
        head_mask = head_mask[:, :, None, :]
        attn_bias = head_mask if attn_bias is None else attn_bias + head_mask
    q = q.float()
    if softmax_scale is not None:
        # attention_ref scales by 1/sqrt(head_dim); fold the requested scale into Q.
        q = q * (softmax_scale * math.sqrt(q.shape[-1]))
    kwargs = dict(
        key_padding_mask=key_mask,
        attn_bias=attn_bias,
        causal=causal,
        window_size=(window, None),
        softcap=softcap or 0.0,
    )
    out, _, lse = attention_ref(q, k.float(), v.float(), return_lse=True, **kwargs)
    out_pt, _ = attention_ref(q.to(v.dtype), k, v, upcast=False, **kwargs)
    empty = torch.isneginf(lse).transpose(1, 2)[..., None]  # (batch, 1, heads, 1)
    return DecodeReference(
        out.masked_fill(empty, 0.0), lse, out_pt.masked_fill(empty, 0.0)
    )


def assert_decode_close(out, lse, ref: DecodeReference):
    """Kernel O within 2x the low-precision eager error (repo convention); LSE within LSE_ATOL.

    Empty rows (batch entries or heads with no kept key) must be exactly zero with -inf LSE.
    """
    check_tensor_vs_ref("out", out, ref.out.to(out.dtype), ref.out_pt)
    empty = torch.isneginf(ref.lse)
    assert torch.equal(torch.isneginf(lse), empty), "LSE -inf rows differ from the oracle"
    torch.testing.assert_close(lse, ref.lse, atol=LSE_ATOL, rtol=0)
    assert torch.count_nonzero(out.transpose(1, 2)[empty]) == 0, "empty row output not zero"
