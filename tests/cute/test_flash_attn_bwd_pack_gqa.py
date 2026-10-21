"""SM100 backward with the query heads of a KV head folded into one tile, against the
per-head path (FLASH_ATTENTION_BWD_PACK_GQA=0, FLASH_ATTENTION_BWD_HEAD_LOOP=0) and against
FP64 eager. The fold modes:

- pack: FLASH_ATTENTION_BWD_PACK_FACTOR=2 packs 2 query heads into the Q tile rows (NOTE
  [bwd pack_gqa] in pack_gqa.py); the remaining qhead_per_kvhead // 2 sub-groups of a KV
  head are separate tiles whose dK / dV accumulate in fp32 (+ postprocess). pack4: the same
  with 4 (hdim 256: what the deterministic causal fold picks when the dQ chains are few).
  pack_full: the one full pack (largest power of two dividing qhead_per_kvhead) that the
  other head dims take.
- pack_loop: pack 2, and the sub-groups looped inside one tile per KV head
  (FLASH_ATTENTION_BWD_HEAD_LOOP=1, hdim 256): K / V loaded once, dK / dV stay in TMEM and are
  stored directly.
- loop: head loop only (no packing), every query head of the KV head a pass of the loop.
"""

import itertools
import os

import pytest
import torch

from eager_reference import check_against_reference, make_cu_seqlens

from flash_attn.cute import utils
from flash_attn.cute.interface import (
    _flash_attn_bwd,
    flash_attn_func,
    flash_attn_varlen_func,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (10, 11),
    reason="SM100/SM110-only backward test",
)

SEED = 20261004
HEAD_DIMS = [64, 128, 256]
# qhead_per_kvhead 4 / 4 / 8 / 8 (pack 2 or 4 + sub-groups, full pack 4 / 8), 6 / 6 / 3 (pack
# 2 + 3 sub-groups, no pack 4) and 5 (no packing: head loop only)
HEADS = {
    "gqa8_2": (8, 2),
    "gqa16_4": (16, 4),
    "gqa32_4": (32, 4),
    "mqa8_1": (8, 1),
    "gqa24_4": (24, 4),
    "gqa12_2": (12, 2),
    "gqa6_2": (6, 2),
    "gqa40_8": (40, 8),
}
# fold -> (pack_gqa, head_loop, pack_factor) overrides
FOLDS = {
    "pack": (True, False, 2),  # pack 2; the other sub-groups through the dK / dV postprocess
    "pack4": (True, False, 4),  # pack 4; the other sub-groups through the dK / dV postprocess
    "pack_full": (True, False, None),  # other head dims: the one full pack
    "pack_loop": (True, True, 2),  # pack 2 and the other sub-groups looped in the CTA
    "loop": (False, True, None),  # unpacked, every query head a pass of the loop
}
ENV_KEYS = (
    "FLASH_ATTENTION_BWD_PACK_GQA",
    "FLASH_ATTENTION_BWD_HEAD_LOOP",
    "FLASH_ATTENTION_BWD_PACK_FACTOR",
)
FEATURE_KWARGS = {
    "none": {},
    "local": {"window_size": (64, 32)},
    "softcap": {"softcap": 15.0},
    "deterministic": {"deterministic": True},
}
# (q_lens, k_lens) per layout; dense uses equal lengths per batch and the dense API.
SHAPES = [
    pytest.param((256, 256), (192, 192), id="2x256x192"),
    pytest.param((1000,), (1000,), id="1x1000"),
    pytest.param((16, 16), (640, 640), id="2x16x640"),
]


@pytest.fixture
def pack_gqa_env():
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def hd256_on_general_kernel():
    getter = getattr(utils, "_get_hd256_generic_bwd", None)
    return getter is None or getter()


def run_fwd_bwd_layout(
    layout, q, k, v, dout, q_lens, k_lens, causal, pack, head_loop, pack_factor=None, **kwargs
):
    os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = "1" if pack else "0"
    os.environ["FLASH_ATTENTION_BWD_HEAD_LOOP"] = "1" if head_loop else "0"
    if pack_factor is None:
        os.environ.pop("FLASH_ATTENTION_BWD_PACK_FACTOR", None)
    else:
        os.environ["FLASH_ATTENTION_BWD_PACK_FACTOR"] = str(pack_factor)
    head_dim = q.shape[-1]
    if layout == "dense":
        b = len(q_lens)
        heads_q, heads_kv = q.shape[1], k.shape[1]
        out, _ = flash_attn_func(
            q.view(b, q_lens[0], heads_q, head_dim),
            k.view(b, k_lens[0], heads_kv, head_dim),
            v.view(b, k_lens[0], heads_kv, head_dim),
            causal=causal,
            **kwargs,
        )
        out = out.flatten(0, 1)
    else:
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=make_cu_seqlens(q_lens),
            cu_seqlens_k=make_cu_seqlens(k_lens),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            causal=causal,
            **kwargs,
        )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    torch.cuda.synchronize()
    return out.detach(), dq, dk, dv


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("feature", list(FEATURE_KWARGS))
@pytest.mark.parametrize("layout", ["dense", "varlen"])
@pytest.mark.parametrize("fold", list(FOLDS))
@pytest.mark.parametrize("mha_type", list(HEADS))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("q_lens,k_lens", SHAPES)
def test_bwd_pack_gqa_matches_unpacked(
    q_lens, k_lens, head_dim, mha_type, fold, layout, feature, causal, pack_gqa_env
):
    if head_dim == 256 and not hd256_on_general_kernel():
        pytest.skip("hdim 256 backward runs on the dedicated kernels (no packing)")
    pack, head_loop, pack_factor = FOLDS[fold]
    heads_q, heads_kv = HEADS[mha_type]
    qhead_per_kvhead = heads_q // heads_kv
    full_pack = qhead_per_kvhead & -qhead_per_kvhead
    if head_loop and head_dim != 256:
        pytest.skip("the head loop is an hdim 256 (serial schedule) path")
    if fold == "pack_full" and (head_dim == 256 or full_pack <= 2):
        pytest.skip("hdim 256 never takes the full pack; a full pack of 2 is the pack cell")
    if pack and qhead_per_kvhead % (pack_factor or full_pack):
        pytest.skip("qhead_per_kvhead does not divide into the packed heads")
    if head_loop and qhead_per_kvhead // (pack_factor if pack else 1) == 1:
        pytest.skip("one sub-group: nothing to loop over")
    if layout == "varlen":
        # Ragged version of the same budget: tile edges, a short slot, an empty slot.
        q_lens, k_lens = (
            (q_lens[0] - 3, 0, 65) if len(q_lens) == 2 else (q_lens[0] - 7,),
            ((k_lens[0] + 5, 0, 129) if len(k_lens) == 2 else (k_lens[0] + 1,)),
        )
    if causal and any(nq > nk for nq, nk in zip(q_lens, k_lens)):
        pytest.skip("bottom-right causal leaves query rows without keys")
    torch.manual_seed(SEED)
    dtype = torch.bfloat16
    q = torch.randn(
        sum(q_lens), heads_q, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        sum(k_lens), heads_kv, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    kwargs = FEATURE_KWARGS[feature]
    args = (layout, q, k, v, dout, q_lens, k_lens, causal)
    ref = run_fwd_bwd_layout(*args, False, False, **kwargs)
    got = run_fwd_bwd_layout(*args, pack, head_loop, pack_factor, **kwargs)
    assert torch.equal(got[0], ref[0]), (
        "forward does not depend on the backward packing"
    )
    for name, a, b_ in zip(("dQ", "dK", "dV"), ref[1:], got[1:]):
        # dQ: same fp32 reduction per scratch head; dK / dV: one TMEM accumulation over the
        # group (or fp32 atomics in another order) -> at most bf16 rounding of the result.
        assert torch.isfinite(b_.float()).all(), f"{name} has non-finite values"
        tol = 2 * torch.finfo(dtype).eps * a.float().abs().max()
        assert (a.float() - b_.float()).abs().max() <= tol, (
            f"{name} differs from the unpacked path"
        )
    if feature == "deterministic":
        again = run_fwd_bwd_layout(*args, pack, head_loop, pack_factor, **kwargs)
        assert all(torch.equal(x, y) for x, y in zip(got, again)), (
            "deterministic run differs"
        )
    if feature in ("none", "deterministic"):
        check_against_reference(got, q, k, v, dout, q_lens, k_lens, causal, dtype)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("layout", ["dense", "varlen"])
@pytest.mark.parametrize("mha_type", ["gqa8_2", "mqa8_1"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_bwd_pack_gqa_unused_kv_tail(head_dim, mha_type, layout, causal, pack_gqa_env):
    """K/V rows past seqused_k must not reach dQ / dK / dV on the packed kernel either.

    The packed kernel bounds its K/V TMA loads like the per-head one (per-CTA descriptor copies
    rewritten to the sequence's row extent), but it runs one tile per KV head over qh times the
    rows: the descriptor slots, the packed LSE / dPsum / dQaccum offsets and the seqused_k masking
    are exercised with a different grid. Same layouts as test_flash_attn_sm100_bwd_unused_kv_tail:
    packed K/V with cu_seqlens_k + seqused_k, and dense (batch, seqlen, heads, d) with seqused_k.
    """
    if head_dim == 256 and not hd256_on_general_kernel():
        pytest.skip("hdim 256 backward runs on the dedicated kernels (no packing)")
    torch.manual_seed(SEED)
    heads_q, heads_kv = HEADS[mha_type]
    dtype = torch.bfloat16
    lengths = [100, 256, 37, 200, 129]
    batch, capacity = len(lengths), 256
    seqused_k = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    if layout == "varlen":
        q = torch.randn(sum(lengths), heads_q, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(
            batch * capacity, heads_kv, head_dim, device="cuda", dtype=dtype
        )
        v = torch.randn_like(k)
        kwargs = dict(
            causal=causal,
            cu_seqlens_q=torch.tensor(
                [0] + list(itertools.accumulate(lengths)),
                device="cuda",
                dtype=torch.int32,
            ),
            cu_seqlens_k=torch.arange(
                0, (batch + 1) * capacity, capacity, device="cuda", dtype=torch.int32
            ),
            seqused_k=seqused_k,
            max_seqlen_q=max(lengths),
            max_seqlen_k=capacity,
        )
    else:
        seqlen_q = 128
        q = torch.randn(batch, seqlen_q, heads_q, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, capacity, heads_kv, head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        kwargs = dict(
            causal=causal,
            seqused_k=seqused_k,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=capacity,
        )
    out, lse = flash_attn_varlen_func(q, k, v, return_lse=True, **kwargs)
    dout = torch.randn_like(out)

    def backward(k_in, v_in, pack):
        os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = "1" if pack else "0"
        dq, dk, dv = _flash_attn_bwd(q, k_in, v_in, out, dout, lse, **kwargs)[:3]
        torch.cuda.synchronize()
        return dq, dk, dv

    valid = torch.zeros(batch, capacity, dtype=torch.bool, device="cuda")
    for b, length in enumerate(lengths):
        valid[b, :length] = True
    if layout == "varlen":
        valid = valid.flatten()
    eps = torch.finfo(dtype).eps
    unpacked = backward(k, v, False)
    ref = backward(k, v, True)
    for name, a, b_ in zip(("dQ", "dK", "dV"), unpacked, ref):
        if name != "dQ":
            # dK / dV rows past seqused_k are never written; compare the valid rows only.
            a, b_ = a[valid], b_[valid]
        assert torch.isfinite(b_.float()).all(), f"{name} has non-finite values"
        tol = 2 * eps * a.float().abs().max()
        assert (a.float() - b_.float()).abs().max() <= tol, (
            f"{name} differs from the unpacked path"
        )
    # Rows past seqused_k hold NaN: a K/V load that is not bounded by the sequence end leaks it
    # through 0 * NaN in dP = dO @ V^T / dQ = dS @ K. The clean and poisoned runs may still
    # differ by an ulp (fp32 reduce-add order).
    for poison in ("k", "v", "kv"):
        k_nan, v_nan = k.clone(), v.clone()
        if "k" in poison:
            k_nan[~valid] = float("nan")
        if "v" in poison:
            v_nan[~valid] = float("nan")
        got = backward(k_nan, v_nan, True)
        for name, a, b_ in zip(("dQ", "dK", "dV"), ref, got):
            if name != "dQ":
                a, b_ = a[valid], b_[valid]
            assert torch.isfinite(b_.float()).all(), f"{poison}: {name} has NaN / Inf"
            torch.testing.assert_close(
                b_, a, atol=4 * eps * a.abs().max().item(), rtol=4 * eps
            )


def test_bwd_pack_gqa_heuristic():
    """Pack when the grid still hides the qh-times longer packed tiles; keep ragged / long causal
    MQA unpacked (measured on B200: MQA 16/1 hd256, one 8k causal sequence plus 16 x 256 in a
    varlen pack: packed 3946 us, unpacked 2297 us)."""
    from flash_attn.cute.interface import _bwd_pack_gqa_heuristic

    def decide(
        batch,
        heads,
        heads_kv,
        seqlen_q,
        seqlen_k,
        causal,
        varlen=False,
        total_q=None,
        total_k=None,
    ):
        return _bwd_pack_gqa_heuristic(
            num_sms=148,
            batch_size=batch,
            num_head=heads,
            num_head_kv=heads_kv,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            total_q=total_q if total_q is not None else batch * seqlen_q,
            total_k=total_k if total_k is not None else batch * seqlen_k,
            m_block_size=128,
            n_block_size=128,
            cluster_size=2,
            causal=causal,
            local=False,
            varlen=varlen,
        )

    # dense 2k / 4k GQA and MQA batches: enough packed tiles, pack
    assert decide(8, 16, 4, 2048, 2048, False)
    assert decide(4, 16, 1, 4096, 4096, False)
    # one long causal MQA sequence: the first packed KV tile runs 16 x 64 iterations, serial
    assert not decide(1, 16, 1, 8192, 8192, True)
    # ragged varlen pack, one 8k sequence + 16 x 256 (PR 2993's workload), causal and not
    lens = (8192,) + (256,) * 16
    assert not decide(
        17, 16, 1, 8192, 8192, True, varlen=True, total_q=sum(lens), total_k=sum(lens)
    )
    assert not decide(
        17,
        16,
        1,
        8192,
        4096,
        False,
        varlen=True,
        total_q=sum(lens),
        total_k=sum(lens) // 2,
    )


def test_bwd_choose_fold():
    """Fold choice on B200 (148 SMs, 74 clusters), hdim 256, 32k tokens per shape: the cells of
    the chain probes (deterministic causal) and of the MQA probe (non-deterministic)."""
    from flash_attn.cute.interface import _bwd_choose_fold

    def decide(batch, heads, heads_kv, seqlen, causal, deterministic, head_loop_possible=True, varlen=False,
               cluster_size=2, m_block_size=128, n_block_size=128, head_dim=256, **kw):
        return _bwd_choose_fold(
            num_sms=148,
            cluster_size=cluster_size,
            batch_size=batch,
            num_head=heads,
            num_head_kv=heads_kv,
            seqlen_q=seqlen,
            seqlen_k=seqlen,
            total_q=batch * seqlen,
            total_k=batch * seqlen,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            head_dim=head_dim,
            causal=causal,
            local=False,
            varlen=varlen,
            deterministic=deterministic,
            fold_allowed=True,
            head_loop_possible=head_loop_possible,
            **kw,
        )

    # non-deterministic: pack 2 + loop whenever the heuristic takes the full fold; where it
    # refuses (long causal MQA: critical path / waves) a smaller fold instead of none
    assert decide(16, 16, 1, 2048, True, False) == (2, True)
    assert decide(16, 40, 8, 2048, True, False) == (1, True)
    assert decide(8, 16, 1, 4096, True, False) == (4, False)
    assert decide(4, 16, 1, 8192, True, False)[0] > 1
    # deterministic causal: MQA 16/1 1k b32 and 2k b16 pack 4 (+4% / +13% over pack 2 + loop),
    # 4k b8 and 8k b4 pack 2 (+14%); GQA 32/2 2k b16 pack 4 (+9%), 8k b4 pack 2 (+13%);
    # GQA 24/4 (qh 6) 2k b16 keeps the loop (+5%), 8k b4 pack 2 (+10%); GQA 40/8 (odd qh) loops
    assert decide(32, 16, 1, 1024, True, True) == (4, False)
    assert decide(16, 16, 1, 2048, True, True) == (4, False)
    assert decide(8, 16, 1, 4096, True, True) == (2, False)
    assert decide(4, 16, 1, 8192, True, True) == (2, False)
    assert decide(16, 32, 2, 2048, True, True) == (4, False)
    assert decide(4, 32, 2, 8192, True, True) == (2, False)
    assert decide(16, 24, 4, 2048, True, True) == (2, True)
    assert decide(4, 24, 4, 8192, True, True) == (2, False)
    assert decide(16, 40, 8, 2048, True, True) == (1, True)  # odd qh: no unlooped fold, the loop stays
    # no head loop (hdim 192, 1 CTA, 64 x 96 tile), GQA 32/8: pack 4 while its chain waste is under
    # 3% (2k b16), pack 2 at 4k b8 / 8k b4 (+3% / +7% over pack 4 on B200), the heuristic's pack 1
    # at 16k; non-deterministic keeps the largest accepted pack
    for seqlen, expect in (
        (2048, (4, False)),
        (4096, (2, False)),
        (8192, (2, False)),
        (16384, (1, False)),
    ):
        assert decide(
            32768 // seqlen, 32, 8, seqlen, True, True, head_loop_possible=False,
            cluster_size=1, m_block_size=64, n_block_size=96, head_dim=192,
        ) == expect, seqlen
    assert decide(
        4, 32, 8, 8192, True, False, head_loop_possible=False,
        cluster_size=1, m_block_size=64, n_block_size=96, head_dim=192,
    ) == (4, False)
    # the loop stays while trips x m_blocks <= 48 (dense) / m_blocks <= 16 (varlen)
    assert decide(32, 12, 2, 1024, True, True) == (2, True)
    assert decide(16, 12, 2, 2048, True, True) == (2, True)
    assert decide(8, 12, 2, 4096, True, True) == (2, False)
    assert decide(32, 64, 8, 1024, True, True) == (2, True)
    assert decide(16, 64, 8, 2048, True, True) == (4, False)
    assert decide(16, 48, 4, 2048, True, True) == (4, False)
    assert decide(16, 32, 2, 2048, True, True, varlen=True) == (2, True)
    assert decide(8, 32, 2, 4096, True, True, varlen=True) == (2, False)
    assert decide(32, 16, 1, 1024, True, True, varlen=True) == (2, True)
    assert decide(16, 16, 8, 2048, True, True) == (2, False)
    # 2-CTA kernels drop pack 4 for pack 2 under deterministic causal once m_blocks * head_dim
    # > 4096 (hdim 192 past 2k, hdim 128 past 4k), also where the heuristic refused pack 2 for
    # its halved partial-pack gain (hdim 192 GQA 48/4 and 64/8 8k b4: pack 4 is 6-9% below the
    # unpacked kernel, pack 2 +0..+2%; hdim 128 pack 2 +7 / +8%); the 1-CTA hdim 64 kernel keeps
    # pack 4 (GQA 64/8 4k b8 +5% over pack 2, 8k b4 level)
    for head_dim, seqlen, expect in (
        (192, 2048, (4, False)),
        (192, 4096, (2, False)),
        (192, 8192, (2, False)),
        (192, 16384, (1, False)),
        (128, 4096, (4, False)),
        (128, 8192, (2, False)),
    ):
        for heads, heads_kv in ((48, 4), (64, 8)):
            for varlen in (False, True):
                assert decide(
                    32768 // seqlen, heads, heads_kv, seqlen, True, True, head_loop_possible=False,
                    head_dim=head_dim, varlen=varlen,
                ) == expect, (head_dim, seqlen, heads, varlen)
    assert decide(8, 32, 8, 4096, True, True, head_loop_possible=False, head_dim=192) == (2, False)
    assert decide(8, 32, 8, 4096, True, True, head_loop_possible=False, head_dim=128) == (4, False)
    assert decide(4, 48, 4, 8192, True, True) == (2, False)  # hdim 256: +8% over pack 4
    assert decide(8, 64, 8, 4096, True, True, head_loop_possible=False, cluster_size=1, head_dim=64) == (4, False)
    assert decide(4, 64, 8, 8192, True, True, head_loop_possible=False, cluster_size=1, head_dim=64) == (4, False)
    # off the deterministic path the hdim 192 kernel also steps a partial pack 4 down to pack 2
    # from 4k on (pack 2 / pack 4 / unpacked on B200: MQA 16/1 4k b8 805 / 770 / 787, GQA 64/8
    # 825 / 803 / 781, 48/4 8k b4 879 / 874 / 864); the full pack 4 of GQA 32/8 stays (818 /
    # 835 / 788), hdim 128 keeps pack 4 (48/4 8k b4 1167 vs 1138), and 2k is below the rule
    for heads, heads_kv, seqlen, expect in (
        (48, 4, 4096, (2, False)),
        (48, 4, 8192, (2, False)),
        (64, 8, 4096, (2, False)),
        (64, 8, 8192, (2, False)),
        (32, 2, 4096, (2, False)),
        (16, 1, 4096, (2, False)),
        (32, 8, 4096, (4, False)),
        (48, 4, 2048, (4, False)),
        (16, 1, 2048, (4, False)),
    ):
        for varlen in (False, True):
            assert decide(
                32768 // seqlen, heads, heads_kv, seqlen, True, False, head_loop_possible=False,
                head_dim=192, varlen=varlen,
            ) == expect, (heads, seqlen, varlen)
    assert decide(4, 48, 4, 8192, True, False, head_loop_possible=False, head_dim=128) == (4, False)
    assert decide(8, 64, 8, 4096, True, False, head_loop_possible=False, head_dim=128) == (4, False)
    assert decide(8, 48, 4, 4096, False, False, head_loop_possible=False, head_dim=192) == (4, False)  # non-causal unchanged
    # deterministic non-causal: no pace difference, the default fold
    assert decide(16, 16, 1, 2048, False, True) == (2, True)
    # overrides
    assert decide(16, 24, 4, 2048, True, False, pack_gqa_override=False) == (1, True)
    assert decide(16, 24, 4, 2048, True, False, head_loop_override=False) == (2, False)
    assert decide(1, 16, 1, 8192, True, False, pack_gqa_override=True) == (2, True)
    assert decide(2, 16, 4, 256, True, True, pack_gqa_override=True, pack_factor_override=4) == (4, False)
    assert decide(16, 16, 1, 2048, True, False, pack_factor_override=4) == (4, False)
    assert decide(16, 16, 1, 2048, True, False, pack_factor_override=2, head_loop_override=True) == (2, True)
    assert decide(16, 8, 1, 2048, True, False, head_loop_possible=False, pack_factor_override=2) == (2, False)
    assert decide(16, 40, 8, 2048, True, False, pack_factor_override=2) == (1, False)
    # without the loop (not hdim 256): pack 4 / pack 2, largest accepted first; the full pack
    # 16 / 8 of hdim 128 MQA 16/1 loses 17% / 6% at 2k b16 causal where pack 4 gains 7%
    assert decide(16, 16, 1, 2048, True, False, head_loop_possible=False, cluster_size=1) == (4, False)
    assert decide(16, 16, 1, 2048, False, False, head_loop_possible=False, cluster_size=1) == (4, False)
    assert decide(16, 24, 4, 2048, True, False, head_loop_possible=False, cluster_size=1) == (2, False)
    assert decide(8, 16, 1, 4096, True, False, head_loop_possible=False, cluster_size=1) == (4, False)
    # a partial pack of four or more sub-groups keeps half the packing gain: MQA 16/1 8k b4
    # causal stays unpacked (pack 4 -2..-3%), 8k non-causal still packs (+3%); the full pack 4
    # of GQA 32/8 is not partial; two sub-groups keep the gain (hdim 192 GQA 32/8 det causal
    # 8k pack 2 above, GQA 64/8 pack 4)
    assert decide(4, 16, 1, 8192, True, False, head_loop_possible=False, cluster_size=1) == (1, False)
    assert decide(4, 16, 1, 8192, False, False, head_loop_possible=False, cluster_size=1) == (4, False)
    assert decide(4, 32, 8, 8192, True, False, head_loop_possible=False, cluster_size=1) == (4, False)
    assert decide(4, 64, 8, 8192, True, False, head_loop_possible=False, cluster_size=1) == (4, False)
    assert decide(4, 16, 1, 8192, True, True, head_loop_possible=False, cluster_size=1) == (1, False)
