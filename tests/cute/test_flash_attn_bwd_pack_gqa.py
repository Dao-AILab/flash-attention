"""SM100 backward with the query heads of a KV head folded into one tile, against the
per-head path (FLASH_ATTENTION_BWD_PACK_GQA=0, FLASH_ATTENTION_BWD_HEAD_LOOP=0) and against
FP64 eager. The fold modes:

- pack: FLASH_ATTENTION_BWD_PACK_GQA=2 packs 2 query heads into the Q tile rows (NOTE
  [bwd pack_gqa] in pack_gqa.py); the remaining qhead_per_kvhead // 2 sub-groups of a KV
  head are separate tiles whose dK / dV accumulate in fp32 (+ postprocess). pack4: the same
  with 4 (hdim 256: what the deterministic causal fold picks when the dQ chains are few).
  pack_full: the one full pack (largest power of two dividing qhead_per_kvhead) that the
  other head dims take (FLASH_ATTENTION_BWD_PACK_GQA=1, the chooser's pick).
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
# fold -> (pack_gqa, head_loop, pack_factor) overrides; FLASH_ATTENTION_BWD_PACK_GQA takes
# 0 (off), 1 (on, the chooser picks the pack) or 2 / 4 (that pack)
FOLDS = {
    "pack": (
        True,
        False,
        2,
    ),  # pack 2; the other sub-groups through the dK / dV postprocess
    "pack4": (
        True,
        False,
        4,
    ),  # pack 4; the other sub-groups through the dK / dV postprocess
    "pack_full": (True, False, None),  # other head dims: the one full pack
    "pack_loop": (True, True, 2),  # pack 2 and the other sub-groups looped in the CTA
    "loop": (False, True, None),  # unpacked, every query head a pass of the loop
}
ENV_KEYS = ("FLASH_ATTENTION_BWD_PACK_GQA", "FLASH_ATTENTION_BWD_HEAD_LOOP")
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
    layout,
    q,
    k,
    v,
    dout,
    q_lens,
    k_lens,
    causal,
    pack,
    head_loop,
    pack_factor=None,
    **kwargs,
):
    if pack and pack_factor is not None:
        os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = str(pack_factor)
    else:
        os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = "1" if pack else "0"
    os.environ["FLASH_ATTENTION_BWD_HEAD_LOOP"] = "1" if head_loop else "0"
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
        pytest.skip(
            "hdim 256 never takes the full pack; a full pack of 2 is the pack cell"
        )
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


def choose_fold(
    batch,
    heads,
    heads_kv,
    seqlen,
    causal,
    deterministic,
    head_loop_possible=True,
    varlen=False,
    cluster_size=2,
    m_block_size=128,
    n_block_size=128,
    head_dim=256,
    **kw,
):
    """_bwd_choose_fold on B200 (148 SMs, 74 clusters) for a dense shape of batch x
    seqlen tokens; hdim 256 with the head loop unless given."""
    from flash_attn.cute.interface import _bwd_choose_fold

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


def fold_case(expect, batch, heads, heads_kv, seqlen, *, causal, deterministic, **kw):
    """One row of CHOOSE_FOLD_CASES: the choose_fold() inputs and the expected
    (pack_factor, head_loop), with an id like det-causal-mqa16/1-2k-b16-hd192-noloop."""
    inputs = dict(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seqlen=seqlen,
        causal=causal,
        deterministic=deterministic,
        **kw,
    )
    parts = [
        "det" if deterministic else "nondet",
        "causal" if causal else "noncausal",
        f"{'mqa' if heads_kv == 1 else 'gqa'}{heads}/{heads_kv}",
        f"{seqlen // 1024}k" if seqlen >= 1024 else str(seqlen),
        f"b{batch}",
    ]
    if "head_dim" in kw:
        parts.append(f"hd{kw['head_dim']}")
    if not kw.get("head_loop_possible", True):
        parts.append("noloop")
    if "cluster_size" in kw:
        parts.append(f"cs{kw['cluster_size']}")
    if "m_block_size" in kw or "n_block_size" in kw:
        parts.append(f"{kw.get('m_block_size', 128)}x{kw.get('n_block_size', 128)}")
    if kw.get("varlen"):
        parts.append("varlen")
    if "pack_gqa_override" in kw:
        parts.append(f"pack{kw['pack_gqa_override']}")
    if "head_loop_override" in kw:
        parts.append(f"loop{int(kw['head_loop_override'])}")
    return pytest.param(inputs, expect, id="-".join(parts))


# Fold choice on B200 (148 SMs, 74 clusters), 32k tokens per shape: the cells of the
# chain probes (deterministic causal) and of the MQA probe (non-deterministic). Expected
# = (pack_factor, head_loop); the measurements behind each rule are in the PR body.
# fmt: off
CHOOSE_FOLD_CASES = [
    # non-deterministic: pack 2 + loop whenever the heuristic takes the full fold; where
    # it refuses (long causal MQA: critical path / waves) a smaller fold instead of none
    fold_case((2, True), 16, 16, 1, 2048, causal=True, deterministic=False),
    fold_case((1, True), 16, 40, 8, 2048, causal=True, deterministic=False),
    fold_case((4, False), 8, 16, 1, 4096, causal=True, deterministic=False),
    # deterministic causal, hdim 256: the dQ chains pace the looped tiles, so few chains
    # take an unlooped pack 4 at 2k and pack 2 from 4k (at 1k the loop's 8 trips x 8
    # m_blocks = 64 still fit); GQA 24/4 (qh 6) keeps the loop at 2k; GQA 40/8 (odd qh)
    # has no unlooped fold, the loop stays
    fold_case((2, True), 32, 16, 1, 1024, causal=True, deterministic=True),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=True),
    fold_case((2, False), 8, 16, 1, 4096, causal=True, deterministic=True),
    fold_case((2, False), 4, 16, 1, 8192, causal=True, deterministic=True),
    fold_case((4, False), 16, 32, 2, 2048, causal=True, deterministic=True),
    fold_case((2, False), 4, 32, 2, 8192, causal=True, deterministic=True),
    fold_case((2, True), 16, 24, 4, 2048, causal=True, deterministic=True),
    fold_case((2, False), 4, 24, 4, 8192, causal=True, deterministic=True),
    fold_case((1, True), 16, 40, 8, 2048, causal=True, deterministic=True),
    # no head loop (hdim 192, 1 CTA, 64 x 96 tile), GQA 32/8: pack 4 while its chain
    # waste is small (2k), pack 2 at 4k / 8k, the heuristic's pack 1 at 16k;
    # non-deterministic keeps the largest accepted pack
    fold_case((4, False), 16, 32, 8, 2048, causal=True, deterministic=True,
              head_loop_possible=False, cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((2, False), 8, 32, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((2, False), 4, 32, 8, 8192, causal=True, deterministic=True,
              head_loop_possible=False, cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((1, False), 2, 32, 8, 16384, causal=True, deterministic=True,
              head_loop_possible=False, cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    fold_case((4, False), 4, 32, 8, 8192, causal=True, deterministic=False,
              head_loop_possible=False, cluster_size=1, m_block_size=64,
              n_block_size=96, head_dim=192),
    # the loop stays while trips x m_blocks <= 64 (dense) / <= 96 (varlen): dense 32/2 and
    # MQA 16/1 at 1k and 64/8 at 2k (64) loop, 48/4 at 2k (96) does not; varlen 32/2 and
    # MQA 16/1 at 2k (128) do not, 48/4 at 2k and 24/4 at 4k (96) do, 64/8 at 4k (128)
    # does not
    fold_case((2, True), 32, 12, 2, 1024, causal=True, deterministic=True),
    fold_case((2, True), 16, 12, 2, 2048, causal=True, deterministic=True),
    fold_case((2, False), 8, 12, 2, 4096, causal=True, deterministic=True),
    fold_case((2, True), 32, 64, 8, 1024, causal=True, deterministic=True),
    fold_case((2, True), 16, 64, 8, 2048, causal=True, deterministic=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True),
    fold_case((2, True), 32, 32, 2, 1024, causal=True, deterministic=True),
    fold_case((4, False), 16, 32, 2, 2048, causal=True, deterministic=True),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=True),
    fold_case((2, True), 32, 32, 2, 1024, causal=True, deterministic=True,
              varlen=True),
    fold_case((4, False), 16, 32, 2, 2048, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, False), 8, 32, 2, 4096, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, True), 32, 16, 1, 1024, causal=True, deterministic=True,
              varlen=True),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, True), 16, 48, 4, 2048, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, True), 8, 24, 4, 4096, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              varlen=True),
    fold_case((2, False), 16, 16, 8, 2048, causal=True, deterministic=True),
    # 2-CTA kernels drop pack 4 for pack 2 under deterministic causal once m_blocks *
    # head_dim > 4096 (hdim 192 past 2k, hdim 128 past 4k), also where the heuristic
    # refused pack 2 for its halved partial-pack gain; hdim 256 (GQA 48/4 8k) likewise;
    # the 1-CTA hdim 64 kernel keeps pack 4
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((4, False), 16, 64, 8, 2048, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((4, False), 16, 64, 8, 2048, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((1, False), 2, 48, 4, 16384, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((1, False), 2, 48, 4, 16384, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((1, False), 2, 64, 8, 16384, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((1, False), 2, 64, 8, 16384, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((4, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128),
    fold_case((4, False), 8, 48, 4, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128, varlen=True),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128, varlen=True),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128, varlen=True),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128, varlen=True),
    fold_case((2, False), 8, 32, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=192),
    fold_case((4, False), 8, 32, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, head_dim=128),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=True),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=True,
              head_loop_possible=False, cluster_size=1, head_dim=64),
    fold_case((4, False), 4, 64, 8, 8192, causal=True, deterministic=True,
              head_loop_possible=False, cluster_size=1, head_dim=64),
    # off the deterministic path the hdim 192 kernel also steps a partial pack 4 down to
    # pack 2 from 4k on; the full pack 4 of GQA 32/8 stays, hdim 128 keeps pack 4, 2k is
    # below the rule, and non-causal is unchanged
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 8, 48, 4, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 4, 48, 4, 8192, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 8, 64, 8, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 4, 64, 8, 8192, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 8, 32, 2, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 8, 32, 2, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((2, False), 8, 16, 1, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((2, False), 8, 16, 1, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((4, False), 8, 32, 8, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((4, False), 8, 32, 8, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((4, False), 16, 48, 4, 2048, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=192, varlen=True),
    fold_case((4, False), 4, 48, 4, 8192, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=128),
    fold_case((4, False), 8, 64, 8, 4096, causal=True, deterministic=False,
              head_loop_possible=False, head_dim=128),
    fold_case((4, False), 8, 48, 4, 4096, causal=False, deterministic=False,
              head_loop_possible=False, head_dim=192),
    # deterministic non-causal: no pace difference, the default fold
    fold_case((2, True), 16, 16, 1, 2048, causal=False, deterministic=True),
    # overrides: pack_gqa_override 0 off / 1 on / 2, 4 that pack; head_loop_override
    # forces or excludes the loop
    fold_case((1, True), 16, 24, 4, 2048, causal=True, deterministic=False,
              pack_gqa_override=0),
    fold_case((2, False), 16, 24, 4, 2048, causal=True, deterministic=False,
              head_loop_override=False),
    fold_case((2, True), 1, 16, 1, 8192, causal=True, deterministic=False,
              pack_gqa_override=1),
    fold_case((4, False), 2, 16, 4, 256, causal=True, deterministic=True,
              pack_gqa_override=4),
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              pack_gqa_override=4),
    fold_case((2, True), 16, 16, 1, 2048, causal=True, deterministic=False,
              pack_gqa_override=2, head_loop_override=True),
    fold_case((2, False), 16, 8, 1, 2048, causal=True, deterministic=False,
              head_loop_possible=False, pack_gqa_override=2),
    fold_case((1, False), 16, 40, 8, 2048, causal=True, deterministic=False,
              pack_gqa_override=2),
    # without the loop (not hdim 256): pack 4 / pack 2, largest accepted first; the full
    # pack 16 / 8 of MQA 16/1 loses to pack 4
    fold_case((4, False), 16, 16, 1, 2048, causal=True, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    fold_case((4, False), 16, 16, 1, 2048, causal=False, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    fold_case((2, False), 16, 24, 4, 2048, causal=True, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    fold_case((4, False), 8, 16, 1, 4096, causal=True, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    # a partial pack of four or more sub-groups keeps half the packing gain: MQA 16/1 8k
    # b4 causal stays unpacked, non-causal still packs; the full pack 4 of GQA 32/8 is
    # not partial; two sub-groups (GQA 64/8) keep the gain
    fold_case((1, False), 4, 16, 1, 8192, causal=True, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    fold_case((4, False), 4, 16, 1, 8192, causal=False, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    fold_case((4, False), 4, 32, 8, 8192, causal=True, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    fold_case((4, False), 4, 64, 8, 8192, causal=True, deterministic=False,
              head_loop_possible=False, cluster_size=1),
    fold_case((1, False), 4, 16, 1, 8192, causal=True, deterministic=True,
              head_loop_possible=False, cluster_size=1),
]
# fmt: on


@pytest.mark.parametrize("inputs,expected", CHOOSE_FOLD_CASES)
def test_bwd_choose_fold(inputs, expected):
    assert choose_fold(**inputs) == expected


def test_bwd_choose_fold_long_causal_mqa_partial_fold():
    """Where the heuristic refuses the full fold of a long causal MQA batch the chooser
    still takes a smaller fold instead of none."""
    assert choose_fold(4, 16, 1, 8192, causal=True, deterministic=False)[0] > 1
