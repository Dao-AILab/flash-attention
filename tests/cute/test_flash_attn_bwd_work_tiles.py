"""SM100 backward work tiles cover every K/V row once and every Q row (NOTE [SM100 backward
work coordinates] in tile_scheduler.py, NOTE [SM100 backward coordinate spaces] in
flash_bwd_sm100.py).

dQ / dK / dV are NaN-filled before the backward: a KV row that no CTA writes, or that two
CTAs of a cluster map to the same block, shows up as NaN or as a wrong value against the FP64
eager reference. The cases force the scheduler and tile choices through the config inputs:
1CTA / 2CTA (both CTA ranks of a work tile), the 64-row KV tile, pack_gqa (scheduler head =
KV head) and unpacked GQA (fp32 dK/dV accumulate per query head), the deterministic LPT
scheduler, and dense / packed varlen / seqused_k layouts with partial last work tiles.
"""

from unittest import mock

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens

from flash_attn.cute import interface
from flash_attn.cute.config import select_bwd_config
from flash_attn.cute.interface import (
    _flash_attn_bwd,
    flash_attn_func,
    flash_attn_varlen_func,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110 backward work tiles",
)

HEAD_DIM = 128
NHEADS = 8
DTYPE = torch.bfloat16


def _cases():
    # (layout, heads, cta, tile_n, deterministic, causal)
    cases = []
    for causal in (False, True):
        for cta in (1, 2):
            for heads in ("mha", "gqa_pack", "gqa"):
                for layout in ("dense", "varlen", "seqused"):
                    cases.append((layout, heads, cta, 128, False, causal))
            for heads in ("mha", "gqa"):
                cases.append(("dense", heads, cta, 128, True, causal))
        # The 64-row KV tile is a request, which turns pack_gqa off (select_bwd_pack_gqa).
        for layout in ("dense", "seqused"):
            cases.append((layout, "mha", 2, 64, False, causal))
    return cases


def _case_id(case):
    layout, heads, cta, tile_n, deterministic, causal = case
    return "-".join(
        [
            layout,
            heads,
            f"{cta}cta",
            f"n{tile_n}",
            "det" if deterministic else "nondet",
            "causal" if causal else "noncausal",
        ]
    )


def _forced_select(cta, tile_n, pack):
    """Select the backward config as the interface does, with the work-tile choices forced."""

    def select(inputs):
        return select_bwd_config(
            inputs._replace(
                disable_2cta=cta == 1,
                # A tile_n request disables pack_gqa, so only request the non-default tile.
                requested_tile_n=None if tile_n == 128 else tile_n,
                requested_pack_gqa=pack,
            )
        )

    return select


def _problem(layout, nheads_kv):
    """q, k, v and the kwargs of one layout, plus the per-sequence lengths for the reference."""
    device = "cuda"
    if layout == "dense":
        # 1000 rows leave partial last work tiles for every tile size.
        batch, seqlen_q, seqlen_k = 2, 1000, 1000
        q = torch.randn(batch, seqlen_q, NHEADS, HEAD_DIM, device=device, dtype=DTYPE)
        k = torch.randn(
            batch, seqlen_k, nheads_kv, HEAD_DIM, device=device, dtype=DTYPE
        )
        v = torch.randn_like(k)
        q_lens, k_lens = [seqlen_q] * batch, [seqlen_k] * batch
        kwargs = {}
    elif layout == "varlen":
        q_lens, k_lens = [1000, 333, 65], [1000, 333, 65]
        q = torch.randn(sum(q_lens), NHEADS, HEAD_DIM, device=device, dtype=DTYPE)
        k = torch.randn(sum(k_lens), nheads_kv, HEAD_DIM, device=device, dtype=DTYPE)
        v = torch.randn_like(k)
        kwargs = dict(
            cu_seqlens_q=make_cu_seqlens(q_lens),
            cu_seqlens_k=make_cu_seqlens(k_lens),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
        )
    else:  # seqused: dense K/V capacity with a used prefix per batch (partial, odd tails)
        batch, seqlen_q, capacity = 4, 128, 1024
        k_lens = [1000, 129, 130, 333]
        q_lens = [seqlen_q] * batch
        q = torch.randn(batch, seqlen_q, NHEADS, HEAD_DIM, device=device, dtype=DTYPE)
        k = torch.randn(
            batch, capacity, nheads_kv, HEAD_DIM, device=device, dtype=DTYPE
        )
        v = torch.randn_like(k)
        kwargs = dict(
            seqused_k=torch.tensor(k_lens, device=device, dtype=torch.int32),
            max_seqlen_q=seqlen_q,
            max_seqlen_k=capacity,
        )
    return q, k, v, kwargs, q_lens, k_lens


def _active_rows(t, lens, layout):
    """The rows the reference covers, packed (total, heads, d)."""
    if layout == "varlen":
        return t
    return torch.cat([t[b, :n] for b, n in enumerate(lens)])


@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_bwd_work_tiles_cover_kv(case):
    layout, heads, cta, tile_n, deterministic, causal = case
    nheads_kv = NHEADS if heads == "mha" else 2
    torch.manual_seed(0)
    q, k, v, kwargs, q_lens, k_lens = _problem(layout, nheads_kv)
    fwd = flash_attn_varlen_func if kwargs else flash_attn_func
    out, lse = fwd(q, k, v, return_lse=True, causal=causal, **kwargs)
    dout = torch.randn_like(out)
    # NaN sentinels expose rows that no CTA writes.
    dq, dk, dv = [torch.full_like(t, float("nan")) for t in (q, k, v)]
    selected = {}

    def select(inputs):
        selected["config"] = _forced_select(cta, tile_n, heads == "gqa_pack")(inputs)
        return selected["config"]

    with mock.patch.object(interface, "select_bwd_config", side_effect=select):
        _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            causal=causal,
            deterministic=deterministic,
            dq=dq,
            dk=dk,
            dv=dv,
            **kwargs,
        )
    torch.cuda.synchronize()
    config = selected["config"]
    assert config.use_2cta_instrs == (cta == 2) and config.cluster_size == cta
    assert config.tile_n == tile_n and config.pack_gqa == (heads == "gqa_pack")
    actual = [
        _active_rows(t, lens, layout)
        for t, lens in ((out, q_lens), (dq, q_lens), (dk, k_lens), (dv, k_lens))
    ]
    packed = [
        _active_rows(t, lens, layout)
        for t, lens in ((q, q_lens), (k, k_lens), (v, k_lens), (dout, q_lens))
    ]
    # Every active row written (no NaN) and correct against FP64 eager.
    check_against_reference(actual, *packed, q_lens, k_lens, causal, DTYPE)
