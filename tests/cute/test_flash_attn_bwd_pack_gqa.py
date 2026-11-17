"""SM100 backward with the query heads of a KV head folded into one tile, against the
per-head path (FLASH_ATTENTION_BWD_PACK_GQA=0, FLASH_ATTENTION_BWD_HEAD_LOOP=0) and against
FP64 eager. The fold modes:

- pack: FLASH_ATTENTION_BWD_PACK_GQA=2 packs 2 query heads into the Q tile rows (NOTE
  [bwd pack_gqa] in pack_gqa.py); the remaining qhead_per_kvhead // 2 sub-groups of a KV
  head are separate tiles whose dK / dV accumulate in fp32 (+ postprocess). pack4: the same
  with 4 (hdim 256: what the deterministic causal fold picks when the dQ chains are few).
  pack_auto: FLASH_ATTENTION_BWD_PACK_GQA=1, the chooser's pick: the largest pack that
  divides qhead_per_kvhead, at most _BWD_MAX_PACK (asserted on the selected config).
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

from flash_attn.cute import interface, utils
from flash_attn.cute.config import _BWD_MAX_PACK
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
# qhead_per_kvhead 4 / 4 / 8 / 8 (pack 2 or 4 + sub-groups), 6 / 6 / 3 (pack 2 + 3
# sub-groups, no pack 4) and 5 (no packing: head loop only)
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
    "pack_auto": (True, False, None),  # the chooser's pack (largest dividing, <= _BWD_MAX_PACK)
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
def selected_bwd_configs(monkeypatch):
    """The BwdConfig of every backward call, in call order (the interface selects one per
    call, cached kernels included)."""
    configs = []
    select = interface.select_bwd_config

    def record(inputs):
        config = select(inputs)
        configs.append(config)
        return config

    monkeypatch.setattr(interface, "select_bwd_config", record)
    return configs


@pytest.fixture
def pack_gqa_env():
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    clear_fold_env_caches()


def clear_fold_env_caches():
    """The backward reads the fold env knobs once per process; clear them to re-read."""
    utils._get_bwd_pack_gqa_override.cache_clear()
    utils._get_bwd_head_loop_override.cache_clear()


def set_fold_env(pack, head_loop=False, pack_factor=None):
    """FLASH_ATTENTION_BWD_PACK_GQA = 0 (off), 1 (on, the chooser picks the pack) or the pack
    (2 / 4); FLASH_ATTENTION_BWD_HEAD_LOOP = 0 / 1."""
    if pack and pack_factor is not None:
        os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = str(pack_factor)
    else:
        os.environ["FLASH_ATTENTION_BWD_PACK_GQA"] = "1" if pack else "0"
    os.environ["FLASH_ATTENTION_BWD_HEAD_LOOP"] = "1" if head_loop else "0"
    clear_fold_env_caches()


def set_pack_gqa_env(pack):
    set_fold_env(pack)


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
    set_fold_env(pack, head_loop, pack_factor)
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
    q_lens,
    k_lens,
    head_dim,
    mha_type,
    fold,
    layout,
    feature,
    causal,
    pack_gqa_env,
    selected_bwd_configs,
):
    pack, head_loop, pack_factor = FOLDS[fold]
    heads_q, heads_kv = HEADS[mha_type]
    qhead_per_kvhead = heads_q // heads_kv
    # FLASH_ATTENTION_BWD_PACK_GQA=1: the largest pack dividing qhead_per_kvhead, capped
    auto_pack = min(qhead_per_kvhead & -qhead_per_kvhead, _BWD_MAX_PACK)
    if head_loop and head_dim != 256:
        pytest.skip("the head loop is an hdim 256 (serial schedule) path")
    if fold == "pack_auto" and (head_dim == 256 or auto_pack <= 2):
        pytest.skip(
            "the chooser's pack is the pack cell (2), or the pack4 cell with the head loop "
            "off (hdim 256)"
        )
    if pack and qhead_per_kvhead % (pack_factor or auto_pack):
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
    # the cell ran the fold it is named after
    assert selected_bwd_configs[-2].pack_factor == 1
    assert not selected_bwd_configs[-2].head_loop
    assert selected_bwd_configs[-1].pack_factor == (
        (pack_factor or auto_pack) if pack else 1
    )
    assert selected_bwd_configs[-1].head_loop == head_loop
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
        set_pack_gqa_env(pack)
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
    from flash_attn.cute.config import BwdHeuristicInputs, bwd_pack_gqa_heuristic

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
        inputs = BwdHeuristicInputs(
            device_arch=100,
            head_dim=128,
            head_dim_v=128,
            num_heads=heads,
            num_heads_kv=heads_kv,
            dtype="torch.bfloat16",
            num_sms=148,
            batch_size=batch,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
            total_q=total_q if total_q is not None else batch * seqlen_q,
            total_k=total_k if total_k is not None else batch * seqlen_k,
            has_cu_seqlens=varlen,
            causal=causal,
        )
        return bwd_pack_gqa_heuristic(inputs, tile_m=128, tile_n=128, cluster_size=2)

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
