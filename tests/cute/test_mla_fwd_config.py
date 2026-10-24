"""MLA (qv) forward config selection: parity with the routing that lived in interface.py.

legacy_mla_routing is a frozen copy of the inline policy (_mla_1cta_route, _mla_fwd_plan, the
sparse head padding, the CLC block, use_s_ahead and the config-dependent asserts) as of
996fc6b2, written over the scalars the interface computed. Do not update it with policy changes.
"""

import math
import random
from collections.abc import Iterator
from dataclasses import replace
from typing import NamedTuple

import pytest

from flash_attn.cute.config import (
    FwdHeuristicInputs,
    MlaFwdConfig,
    MlaFwdHeuristicInputs,
    mla_1cta_route,
    num_splits_heuristic,
    select_fwd_config,
    select_mla_fwd_config,
    validate_mla_fwd_config,
)


class Case(NamedTuple):
    num_sms: int
    fp8: bool
    descale: bool
    batch: int
    s_q: int
    s_k: int
    # dense, dense_maxq (caller int max_seqlen_q), cu_q_hint, cu_q_nohint, seqused_q, cu_k,
    # cu_k_nohint (no caller max_seqlen_k), seqused_k, paged
    layout: str
    page_size: int | None
    heads: int
    heads_kv: int
    causal: bool
    tile_mn: tuple[int, int] | None
    num_splits: int
    pack_gqa: bool | None
    topk: bool
    grad: bool
    recompute_p: bool
    env: str | None  # FLASH_ATTENTION_MLA_1CTA


def legacy_mla_routing(c: Case):
    """Return (mla_1cta, kb64, tile_m, tile_n, q_stage, num_splits, pack_gqa, use_clc, s_ahead)."""
    sc = mla_inputs(c).fwd
    nheads_per_kv = qhead_per_kvhead = c.heads // c.heads_kv
    seqlen_q_hint = sc.max_seqlen_q if sc.has_cu_seqlens_q else c.s_q
    seqlen_q_known = None if c.layout == "cu_q_nohint" else seqlen_q_hint
    needs_1cta = c.fp8 or c.descale or c.num_splits > 1
    ctas_2cta = 2 * c.heads_kv * (sc.total_q if c.topk else c.batch)
    batch_size, total_q, num_head_kv, num_sms = c.batch, sc.total_q, c.heads_kv, c.num_sms
    max_seqlen_q, max_seqlen_k, tile_mn = sc.max_seqlen_q, sc.max_seqlen_k, c.tile_mn

    def route(split_kv):
        supported = not c.topk or (nheads_per_kv <= 64 and (not c.grad or c.recompute_p))
        if c.env is not None:
            return c.env == "1" and supported
        if not supported:
            return False
        if needs_1cta:
            return True
        past_one_wave = ctas_2cta > num_sms
        if c.topk:
            return past_one_wave
        if split_kv:
            return True
        if seqlen_q_hint is None or seqlen_q_hint * nheads_per_kv > 64:
            return False
        return past_one_wave

    mla_1cta = route(c.num_splits < 1)
    pack_gqa = c.pack_gqa
    if c.topk and qhead_per_kvhead < 128:
        assert num_head_kv == 1, "sparse MLA requires a single KV head"
        min_tile = 64 if mla_1cta else 128
        qhead_per_kvhead = min_tile if qhead_per_kvhead <= min_tile else 128
        pack_gqa = True
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    tile_m, tile_n = (128, 128) if tile_mn is None else tile_mn
    cfg = (tile_m, tile_n, 2 if max_seqlen_q * (qhead_per_kvhead if pack_gqa else 1) > tile_m
           else 1, 1)
    rows_per_token = qhead_per_kvhead if pack_gqa else 1
    kb64_ok = not (c.fp8 or nheads_per_kv > 64) and (
        c.topk or nheads_per_kv == 64 or (num_head_kv == 1 and seqlen_q_hint == 1)
    )
    if c.topk and c.num_splits > 1:
        raise ValueError(
            f"num_splits={c.num_splits} is not supported with gather_kv_indices "
            "(sparse MLA runs without split-KV)"
        )

    def plan(mla_1cta):
        if not mla_1cta:
            return False, cfg
        if kb64_ok:
            tile_m, tile_n = (64, 64) if c.topk or tile_mn is None else tile_mn
            q_stage, token_rows, min_blocks = 1, 64, 4
        else:
            tile_m, tile_n, q_stage = cfg[:3]
            token_rows, min_blocks = rows_per_token, 1
        splits = 1 if c.topk else c.num_splits
        if splits < 1:
            num_n_blocks = (max_seqlen_k + tile_n - 1) // tile_n
            m_rows = q_stage * tile_m
            partial = 0 if token_rows % m_rows == 0 else batch_size - 1
            num_tiles = num_head_kv * min(
                batch_size * ((max_seqlen_q * token_rows + m_rows - 1) // m_rows),
                (total_q * token_rows + m_rows - 1) // m_rows + partial,
            )
            max_splits = min(128, max(1, num_n_blocks // min_blocks))
            splits = num_splits_heuristic(num_tiles, num_sms, num_n_blocks, max_splits)
        return kb64_ok, (tile_m, tile_n, q_stage, splits)

    kb64, tiles = plan(mla_1cta)
    if mla_1cta and c.num_splits < 1 and not c.topk and tiles[3] < (2 if kb64 else 5):
        mla_1cta = route(False)
        if not mla_1cta:
            kb64, tiles = plan(False)
    use_clc = False  # the legacy dense CLC request, unset by default
    if mla_1cta:
        use_clc = (
            c.topk or (seqlen_q_known is not None and seqlen_q_known > 1)
            if kb64 else c.topk or use_clc
        )
    if not mla_1cta:
        assert not c.descale, "q_descale/k_descale/v_descale with qv are only supported by 1CTA"
    assert tiles[1] == (64 if kb64 else 128)
    assert pack_gqa or not c.topk  # the kernel constructors' check
    s_ahead = (
        mla_1cta and not kb64 and c.fp8
        and seqlen_q_known is not None and seqlen_q_known * nheads_per_kv >= 512
    )
    return (mla_1cta, kb64, *tiles, pack_gqa, use_clc, s_ahead)


def mla_inputs(c: Case) -> MlaFwdHeuristicInputs:
    """The inputs _get_fwd_config builds for the same call."""
    is_cu_q = c.layout in ("cu_q_hint", "cu_q_nohint")
    total_q, total_k = c.batch * c.s_q, c.batch * c.s_k
    if c.layout == "paged":
        total_k = c.batch * math.ceil(c.s_k / c.page_size) * c.page_size
    caller_q = {"dense_maxq": max(1, c.s_q // 2), "cu_q_hint": c.s_q}.get(c.layout)
    q_capacity = total_q if is_cu_q else c.s_q
    fwd = FwdHeuristicInputs(
        device_arch=100, num_sms=c.num_sms,
        dtype="torch.float8_e4m3fn" if c.fp8 else "torch.bfloat16",
        head_dim=64, head_dim_v=512, num_heads=c.heads, num_heads_kv=c.heads_kv,
        batch_size=c.batch, total_q=total_q, total_k=total_k,
        max_seqlen_q=q_capacity if caller_q is None else caller_q,
        max_seqlen_k=total_k if c.layout == "cu_k_nohint" else c.s_k,
        has_max_seqlen_q_hint=caller_q is not None,
        causal=c.causal, has_cu_seqlens_q=is_cu_q,
        has_cu_seqlens_k=c.layout in ("cu_k", "cu_k_nohint"),
        has_seqused_q=c.layout == "seqused_q", has_seqused_k=c.layout == "seqused_k",
        pack_gqa=c.pack_gqa, page_size=c.page_size, has_lse=c.grad,
        requested_tile_m=None if c.tile_mn is None else c.tile_mn[0],
        requested_tile_n=None if c.tile_mn is None else c.tile_mn[1],
        requested_num_splits=None if c.num_splits < 1 else c.num_splits,
    )
    return MlaFwdHeuristicInputs(
        fwd,
        is_topk_gather=c.topk,
        requires_grad=c.grad,
        gather_bwd_recompute_p=c.recompute_p,
        has_descale=c.descale,
        mla_1cta_override=None if c.env is None else c.env == "1",
    )


def selected_routing(c: Case):
    config = select_mla_fwd_config(mla_inputs(c))
    return (
        config.kernel != "2cta",
        config.kernel == "1cta_kb64",
        config.tile_m,
        config.tile_n,
        config.q_stage,
        config.num_splits,
        config.pack_gqa,
        config.use_clc_scheduler,
        config.s_ahead,
    )


_DENSE = Case(num_sms=148, fp8=False, descale=False, batch=2, s_q=1, s_k=2048, layout="dense",
              page_size=None, heads=64, heads_kv=1, causal=False, tile_mn=None, num_splits=1,
              pack_gqa=None, topk=False, grad=False, recompute_p=False, env=None)


def routing_cases() -> Iterator[Case]:
    # Pin policy boundaries independently of the small, deterministic interaction sample.
    yield _DENSE
    for changes in [
        dict(num_splits=0, heads=128, batch=32),  # four splits: re-route to 2CTA
        dict(num_splits=0, heads=128, batch=28),  # five splits: keep 1CTA
        dict(num_splits=0, s_k=200),
        dict(num_splits=0, s_k=1024),
        dict(heads=16, s_q=1, env="1"),
        dict(heads=16, s_q=2, env="1"),
        dict(heads=64, s_q=2, env="1"),
        dict(heads=128, s_q=2, env="1"),
        dict(topk=True, heads=24, s_q=2, pack_gqa=False, env="0"),
        dict(topk=True, heads=24, s_q=2, pack_gqa=False, env="1"),
        dict(fp8=True, s_q=7),
        dict(fp8=True, s_q=8),
        dict(fp8=True, s_q=8, layout="cu_q_nohint"),
        dict(s_q=2, layout="cu_q_hint", env="1"),
        dict(s_q=2, layout="cu_q_nohint", env="1"),
    ]:
        yield _DENSE._replace(**changes)
    rng = random.Random(0)
    for _ in range(256):
        c = _DENSE._replace(
            fp8=rng.choice([False, True]), batch=rng.choice([1, 2, 32, 64, 128]),
            s_q=rng.choice([1, 2, 4, 256, 4096]), s_k=rng.choice([200, 1024, 2048, 8192]),
            layout=rng.choice(["dense", "dense_maxq", "cu_q_hint", "cu_q_nohint",
                               "seqused_q", "cu_k", "cu_k_nohint", "seqused_k", "paged"]),
            page_size=rng.choice([16, 64, 128]), heads=rng.choice([1, 16, 24, 48, 64, 128]),
            heads_kv=rng.choice([1, 2]), causal=rng.choice([False, True]),
            tile_mn=rng.choice([None, (64, 64), (128, 128)]), num_splits=rng.choice([0, 1, 3]),
            pack_gqa=rng.choice([None, True, False]), env=rng.choice([None, "0", "1"]),
        )
        if c.heads % c.heads_kv == 0:
            yield c._replace(page_size=c.page_size if c.layout == "paged" else None)


def test_dense_1cta_clc_is_an_explicit_choice():
    """Dense 1CTA (128-key mainloop) runs without CLC by default; an explicit config may
    turn it on."""
    inputs = mla_inputs(_DENSE._replace(heads=16, s_q=2, causal=True, env="1"))
    selected = select_mla_fwd_config(inputs)
    assert selected.kernel == "1cta" and not selected.use_clc_scheduler

    validate_mla_fwd_config(replace(selected, use_clc_scheduler=True), inputs)


def outcome(fn, c: Case):
    try:
        return fn(c)
    except (ValueError, AssertionError) as exc:
        return type(exc), str(exc)


def test_selection_matches_the_legacy_interface_routing():
    """Same kernel, tiles, splits, pack_gqa, CLC and s_ahead as the legacy inline routing.

    The config records the kernels' implicit choices: 2CTA always ran CLC and kb64 always
    packed GQA. Legacy asserts become ValueErrors, including the dense 2CTA packed-head
    divisibility check that previously happened only in the constructor.
    """
    mismatches = []
    for c in routing_cases():
        legacy, new = outcome(legacy_mla_routing, c), outcome(selected_routing, c)
        if isinstance(legacy[0], type):
            # The split-KV rejection keeps its type and message; legacy asserts become ValueError.
            ok = new == legacy if legacy[0] is ValueError else (
                isinstance(new[0], type) and new[0] is ValueError
            )
        elif not legacy[0]:  # 2CTA
            if not c.topk and legacy[6] and 128 % (c.heads // c.heads_kv) != 0:
                ok = new[0] is ValueError
            else:
                ok = new == (*legacy[:7], True, legacy[8])
        elif legacy[1]:  # kb64 packs even when the legacy routing recorded False
            ok = new == (*legacy[:6], True, *legacy[7:])
        else:
            ok = new == legacy
        if not ok:
            mismatches.append((c, legacy, new))
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:3]}"


@pytest.mark.parametrize(
    ("case", "changes", "match"),
    [
        (_DENSE, dict(kernel="2cta", num_splits=3), "split kv"),
        (_DENSE._replace(fp8=True, descale=True), dict(kernel="2cta"), "descale"),
        (_DENSE, dict(tile_n=128), "tile_n=64"),
        (_DENSE, dict(q_stage=2), "q_stage=1"),
        (_DENSE._replace(s_q=256, heads=16), dict(kernel="1cta_kb64"), "does not serve"),
        (_DENSE._replace(s_q=256), dict(kernel="1cta", tile_n=64), "tile_n=128"),
        (_DENSE, dict(kernel="2cta", tile_m=128, tile_n=128, use_clc_scheduler=False), "CLC"),
        (_DENSE, dict(s_ahead=True), "s_ahead"),
        (_DENSE, dict(num_splits=0), r"\[1, 256\]"),
        (_DENSE._replace(pack_gqa=True), dict(pack_gqa=False), "pack_gqa=True"),
        (_DENSE._replace(topk=True, heads=128, s_q=256), dict(kernel="1cta", tile_m=128,
                                                              tile_n=128), "64 Q heads"),
        (_DENSE._replace(topk=True, grad=True), dict(kernel="1cta", tile_m=128, tile_n=128),
         "recompute_p"),
        (_DENSE._replace(topk=True), dict(num_splits=2), "without split-KV"),
        (_DENSE._replace(topk=True, heads=128, s_q=256, pack_gqa=False),
         dict(kernel="2cta", tile_m=128, tile_n=128, use_clc_scheduler=True), "pack_gqa"),
        (_DENSE._replace(topk=True, heads=32, heads_kv=2), dict(), "single KV head"),
    ],
)
def test_validation_rejects_unsupported_mla_configs(case, changes, match):
    inputs = mla_inputs(case)
    base = MlaFwdConfig(kernel="1cta_kb64", tile_m=64, tile_n=64, q_stage=1, num_splits=1,
                        pack_gqa=True, use_clc_scheduler=False, s_ahead=False)
    with pytest.raises(ValueError, match=match):
        validate_mla_fwd_config(replace(base, **changes), inputs)


# Ported from test_flash_attn_mla.py's interface-level route test; the expectations are unchanged.
@pytest.mark.parametrize(
    "case",
    [
        # (description, route kwargs, expected 1CTA)
        # split: the split heuristic (num_splits < 1). Dense goes 1CTA provisionally at any shape.
        ("dense decode 64 heads, split heuristic", dict(topk=False, h=64, s_q=1, split=True, ctas=16), True),
        ("dense 64 heads x 2 tokens, split heuristic", dict(topk=False, h=64, s_q=2, split=True, ctas=16), True),
        ("dense decode 128 heads, split heuristic", dict(topk=False, h=128, s_q=1, split=True, ctas=16), True),
        ("dense prefill, split heuristic", dict(topk=False, h=64, s_q=4096, split=True), True),
        # unsplit: decode shapes (seqlen_q x heads per KV head <= 64) go 1CTA only once the 2CTA
        # kernel (2 CTAs per batch x KV head) exceeds one wave of num_sms
        ("dense decode 64 heads, 2CTA one wave", dict(topk=False, h=64, s_q=1, ctas=144), False),
        ("dense decode 64 heads, 2CTA two waves", dict(topk=False, h=64, s_q=1, ctas=160), True),
        ("dense decode 16 heads x 4 tokens (64 rows), two waves", dict(topk=False, h=16, s_q=4, ctas=160), True),
        ("dense decode 16 heads, one wave", dict(topk=False, h=16, s_q=1, ctas=128), False),
        ("dense 64 heads x 2 tokens (128 rows), two waves", dict(topk=False, h=64, s_q=2, ctas=160), False),
        ("dense decode 128 heads, two waves", dict(topk=False, h=128, s_q=1, ctas=160), False),
        ("dense prefill, two waves", dict(topk=False, h=64, s_q=4096, ctas=160), False),
        ("dense varlen without a host max_seqlen_q", dict(topk=False, h=64, s_q=None, ctas=160), False),
        ("dense prefill fp8 (2CTA has no fp8)", dict(topk=False, h=64, s_q=4096, needs=True), True),
        # sparse (never split): 2 CTAs per token on 2CTA; 1CTA only past one 2CTA wave
        ("sparse 64 heads", dict(topk=True, h=64, s_q=4096), True),
        ("sparse 24 heads decode", dict(topk=True, h=24, s_q=1), True),
        ("sparse decode, 2CTA one wave", dict(topk=True, h=64, s_q=1, ctas=152), False),
        ("sparse decode, 2CTA past one wave", dict(topk=True, h=64, s_q=1, ctas=160), True),
        ("sparse decode 16 heads, one wave", dict(topk=True, h=16, s_q=1, ctas=64), False),
        ("sparse fp8 below one wave (2CTA has no fp8)", dict(topk=True, h=64, s_q=1, ctas=64, needs=True), True),
        ("sparse 128 heads (unsupported on 1CTA)", dict(topk=True, h=128, s_q=1), False),
        ("sparse training, load-P (unsupported)", dict(topk=True, h=64, s_q=4096, grad=True), False),
        ("sparse training, recompute-P", dict(topk=True, h=64, s_q=4096, grad=True, rp=True), True),
    ],
    ids=lambda c: c[0].replace(" ", "_"),
)
def test_mla_dispatch_heuristic(case):
    """The 1CTA / 2CTA MLA dispatch: sparse -> 1CTA up to 64 heads once 2CTA exceeds one wave
    (2 CTAs per token); dense with the split heuristic -> 1CTA (provisionally), unsplit -> 1CTA
    on decode shapes (seqlen_q x heads per KV head <= 64) once 2CTA exceeds one wave, 2CTA
    otherwise; fp8 / descales / explicit split-KV -> 1CTA. The override wins where supported."""
    _, kw, expected = case
    ctas = kw.get("ctas", 1024)  # absent: past any one-wave bound
    batch = 1 if kw["topk"] else ctas // 2
    s_q = kw["s_q"]
    total_q = ctas // 2 if kw["topk"] else batch * (s_q or 4096)
    fwd = FwdHeuristicInputs(
        device_arch=100, num_sms=152, dtype="torch.bfloat16", head_dim=64, head_dim_v=512,
        num_heads=kw["h"], num_heads_kv=1, batch_size=batch, total_q=total_q, total_k=total_q,
        max_seqlen_q=total_q if s_q is None else s_q, max_seqlen_k=4096,
        has_max_seqlen_q_hint=s_q is not None, has_cu_seqlens_q=s_q is None or kw["topk"],
        requested_num_splits=None if kw.get("split") else 1,
    )
    inputs = MlaFwdHeuristicInputs(
        fwd, is_topk_gather=kw["topk"], requires_grad=kw.get("grad", False),
        gather_bwd_recompute_p=kw.get("rp", False), has_descale=kw.get("needs", False),
    )
    split_kv = kw.get("split", False)
    assert mla_1cta_route(inputs, split_kv=split_kv) == expected
    supported = not (kw["topk"] and (kw["h"] > 64 or (kw.get("grad") and not kw.get("rp"))))
    assert mla_1cta_route(inputs._replace(mla_1cta_override=True), split_kv=split_kv) == supported
    assert not mla_1cta_route(inputs._replace(mla_1cta_override=False), split_kv=split_kv)


# Interface marshaling: these run on CPU with fake tensors and stop at the compile-cache lookup.
torch = pytest.importorskip("torch")


class _KeyCaptured(Exception):
    pass


@pytest.fixture
def capture_key(monkeypatch):
    """Call _flash_attn_fwd under FakeTensorMode and return its compile key without compiling."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from flash_attn.cute import interface

    class Spy:
        def __contains__(self, key):
            raise _KeyCaptured(key)

    monkeypatch.setenv("FLASH_ATTENTION_NUM_SMS", "148")
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", Spy())

    def run(*, b=2, s_q=1, s_k=2048, h=64, **kwargs):
        with FakeTensorMode():
            q = torch.empty(b, s_q, h, 64, dtype=torch.bfloat16)
            qv = torch.empty(b, s_q, h, 512, dtype=torch.bfloat16)
            k = torch.empty(b, s_k, 1, 64, dtype=torch.bfloat16)
            v = torch.empty(b, s_k, 1, 512, dtype=torch.bfloat16)
            tensor_kwargs = {
                name: torch.empty((), dtype=torch.int32) if value == "tensor" else value
                for name, value in kwargs.items()
            }
            with pytest.raises(_KeyCaptured) as exc:
                interface._flash_attn_fwd(q, k, v, qv=qv, _arch=100, **tensor_kwargs)
        return exc.value.args[0]

    return run


@pytest.mark.parametrize("heads", [1, 64])
def test_kb64_selection_resolves_packing(heads):
    inputs = mla_inputs(_DENSE._replace(heads=heads, pack_gqa=False, env="1"))
    config = select_mla_fwd_config(inputs)
    assert config.kernel == "1cta_kb64"
    assert config.pack_gqa is True
    with pytest.raises(ValueError, match="pack_gqa"):
        validate_mla_fwd_config(replace(config, pack_gqa=False), inputs)


@pytest.mark.parametrize("heads", [0, -1, 129, 24])
def test_2cta_rejects_unsupported_head_counts(heads):
    inputs = mla_inputs(_DENSE._replace(heads=heads))
    config = MlaFwdConfig(kernel="2cta", tile_m=128, tile_n=128, q_stage=1, num_splits=1,
                          pack_gqa=True, use_clc_scheduler=True, s_ahead=False)
    with pytest.raises(ValueError, match="Q heads per KV head"):
        validate_mla_fwd_config(config, inputs)


@pytest.mark.parametrize(
    ("heads", "pack_gqa", "topk"), [(1, True, False), (128, True, False),
                                     (24, False, False), (24, True, True)]
)
def test_2cta_accepts_supported_head_counts(heads, pack_gqa, topk):
    inputs = mla_inputs(_DENSE._replace(heads=heads, pack_gqa=pack_gqa, topk=topk, env="0"))
    config = select_mla_fwd_config(inputs)
    assert config.kernel == "2cta"
    validate_mla_fwd_config(config, inputs)


def test_kb64_fallback_preserves_caller_packing_preference():
    inputs = mla_inputs(_DENSE._replace(heads=24, s_k=200, num_splits=0, pack_gqa=False))
    config = select_mla_fwd_config(inputs)
    assert config.kernel == "2cta"
    assert config.pack_gqa is False


@pytest.mark.parametrize("kernel", ["2cta", "1cta", "1cta_kb64"])
def test_mla_compile_key_ignores_heuristic_only_tiles(capture_key, kernel):
    config = MlaFwdConfig(kernel=kernel, tile_m=64, tile_n=64 if kernel == "1cta_kb64" else 128,
                          q_stage=1, num_splits=1, pack_gqa=True,
                          use_clc_scheduler=True, s_ahead=False)
    key = capture_key(config=config)
    assert capture_key(config=replace(config, tile_m=128)) == key
    if kernel != "1cta_kb64":
        assert capture_key(config=replace(config, q_stage=2)) == key


def test_tensor_max_seqlen_on_dense_mla_selects_without_reading_it(capture_key, monkeypatch):
    """Dense MLA sizes its tiles and splits from the host capacity bounds; reading a fake
    tensor's value would raise. Legacy fed the tensor into the split math."""
    monkeypatch.delenv("FLASH_ATTENTION_MLA_1CTA", raising=False)
    key = capture_key(max_seqlen_q="tensor", max_seqlen_k="tensor", num_splits=0)
    assert key == capture_key(max_seqlen_q=1, max_seqlen_k=2048, num_splits=0)


@pytest.mark.parametrize(("env", "expect_1cta"), [(None, True), ("1", True), ("0", False),
                                                  ("yes", False)])
def test_mla_1cta_environment_override(capture_key, monkeypatch, env, expect_1cta):
    """FLASH_ATTENTION_MLA_1CTA is read per call: "1" wherever supported, any other value never."""
    if env is None:
        monkeypatch.delenv("FLASH_ATTENTION_MLA_1CTA", raising=False)
    else:
        monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", env)
    assert capture_key(num_splits=0)[0] is expect_1cta


def test_explicit_mla_config_round_trip_and_wrong_type_rejection(capture_key, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from flash_attn.cute import interface

    monkeypatch.setenv("FLASH_ATTENTION_MLA_1CTA", "1")
    config = select_mla_fwd_config(mla_inputs(_DENSE._replace(num_splits=0, env="1")))
    key = capture_key(num_splits=0)
    assert capture_key(num_splits=0, config=config) == key
    # The explicit config, not the num_splits argument, decides split-KV.
    assert config.num_splits > 1
    assert capture_key(num_splits=0, config=replace(config, num_splits=1)) != key

    generic = select_fwd_config(mla_inputs(_DENSE).fwd._replace(head_dim_v=64))
    with pytest.raises(ValueError, match="MlaFwdConfig with qv"):
        capture_key(config=generic)
    with FakeTensorMode():
        q = torch.empty(2, 128, 16, 64, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="FwdConfig without qv"):
            interface._flash_attn_fwd(q, q, q, _arch=100, config=config)
