"""P quant-dequant on the SM100 forward: device codec, forward, LSE gate, gradients, SplitKV
and context-parallel patterns.

Needs sm_100 to execute. The compile pass (FLASH_ATTENTION_FAKE_TENSOR=1) also runs when the
kernels are cross-compiled (FLASH_ATTENTION_ARCH=sm_100 CUTE_DSL_ARCH=sm_100a
FLASH_ATTENTION_NUM_SMS=148). References: tests/cute/p_quant_attention_oracle.py and
docs/p-quant-contract.md.
"""

import itertools
import math
import os
from dataclasses import replace

import pytest
import torch

from flash_attn.cute import interface
from flash_attn.cute.config import needs_split_metadata
from flash_attn.cute.interface import (
    _flash_attn_bwd,
    _flash_attn_fwd,
    flash_attn_func,
    flash_attn_varlen_func,
)
from flash_attn.cute.p_quant import NVFP4_BLOCK, resolve_p_quant
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode
from p_quant_attention_oracle import (
    Schedule,
    merge_partials,
    textbook_attention,
    tiled_attention,
)
from p_quant_codec_probe import run_codec_probe
from p_quant_oracle import E2M1_MAGNITUDES, qdq
from p_quant_test_utils import (
    assert_identity_lse_gate,
    assert_matches_oracle,
    capture_fwd_config,
    check_allowance_budgets,
    default_scale,
    device_arch,
    oracle_batch,
    per_sequence_split_counts,
    rounding_noise,
    schedule_of,
    settle_single_row,
    surrogate_grads,
)
from test_p_quant_codec import E4M3_VALUES, ties_to_even_expected

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
FORMATS = ["fp8_e4m3", "nvfp4"]
# Smallest effect quantization must have on off-grid inputs (see test_p_quant_oracle.py).
MIN_EFFECT = {"fp8_e4m3": 1e-4, "nvfp4": 1e-3}

pytestmark = pytest.mark.skipif(
    device_arch() != 100, reason="P quant-dequant runs on sm_100 only"
)


def rand_bf16(*shape, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )


def fwd(q, k, v, p_quant, **kwargs):
    out, lse, *_ = _flash_attn_fwd(q, k, v, return_lse=True, p_quant=p_quant, **kwargs)
    return out, lse


def fwd_with_config(q, k, v, p_quant, **kwargs):
    """Forward plus the config it resolved; later calls pass that config explicitly."""
    with capture_fwd_config() as captured:
        out, lse = fwd(q, k, v, p_quant, **kwargs)
    config, inputs = captured[0]
    return out, lse, config, inputs


def call_kwargs(case):
    kw = {}
    if case.get("causal"):
        kw["causal"] = True
    if "window" in case:
        kw["window_size_left"], kw["window_size_right"] = case["window"]
    for name in ("num_splits", "seqlen_k_per_split", "pack_gqa"):
        if name in case:
            kw[name] = case[name]
    return kw


def case_schedule(case, config, inputs, qpk):
    wl, wr = case.get("window", (None, None))
    return schedule_of(
        config,
        inputs,
        qhead_per_kvhead=qpk,
        causal=case.get("causal", False),
        window_size_left=wl,
        window_size_right=wr,
        seqlen_k_per_split=case.get("seqlen_k_per_split"),
    )


# --------------------------------------------------------------- device codec


def codec_inputs(fmt):
    """Exact FP32 codec inputs: ties, their neighbours, subnormals, saturation, overshoot,
    zero-rounded scales and partial blocks."""
    f32 = lambda vals: torch.tensor(vals, dtype=torch.float32)  # noqa: E731
    chunks = []
    if fmt == "fp8_e4m3":
        mids = f32([m for m, _ in ties_to_even_expected(E4M3_VALUES)])
        chunks += [
            f32(E4M3_VALUES),
            mids,
            torch.nextafter(mids, torch.zeros_like(mids)),
            torch.nextafter(mids, torch.full_like(mids, 1e9)),
            f32([2.0**-10, 2.0**-11, 0.0, 448.0, 448.0001, 464.0, 480.0, 1e6]),
            torch.nextafter(f32([256.0]), f32([1e9])),  # overshoot of the ideal maximum
        ]
        gen = torch.Generator().manual_seed(0)
        chunks.append(torch.exp2(torch.empty(4096).uniform_(-12, 8, generator=gen)))
    else:
        blocks = []
        # Block max 0.75 gives scale code 336 and effective scale exactly 1/8, so code ties
        # are exact: x = tie / 8.
        ties = [m for m, _ in ties_to_even_expected(E2M1_MAGNITUDES)]
        blk = torch.zeros(NVFP4_BLOCK)
        blk[0] = 0.75
        blk[1 : 1 + len(ties)] = f32(ties) / 8
        blk[8:15] = torch.nextafter(f32(ties) / 8, torch.zeros(7))
        blocks.append(blk)
        blk2 = blk.clone()
        blk2[8:15] = torch.nextafter(f32(ties) / 8, torch.ones(7))
        blocks.append(blk2)
        # Every code magnitude under an exact scale, and saturation of the block maximum.
        blk3 = torch.zeros(NVFP4_BLOCK)
        blk3[: len(E2M1_MAGNITUDES)] = f32(E2M1_MAGNITUDES) / 8
        blk3[0] = 0.75
        blocks.append(blk3)
        # Scale decision ties: 448 * b on E4M3 midpoints.
        for mid, _ in ties_to_even_expected(E4M3_VALUES)[40::9]:
            b = torch.full((NVFP4_BLOCK,), 0.0)
            b[0] = mid / 448.0
            b[1:9] = b[0] * f32([0.9, 0.7, 0.55, 0.4, 0.3, 0.2, 0.1, 0.03])
            blocks.append(b)
        # Zero block, zero-rounded scale, the first scale above it, a partial block, overshoot.
        blocks.append(torch.zeros(NVFP4_BLOCK))
        tiny = f32([float.fromhex("0x1.24924ap-19")]).item()
        blocks.append(torch.full((NVFP4_BLOCK,), tiny))
        above = torch.nextafter(f32([tiny]), f32([1.0])).item()
        blocks.append(torch.full((NVFP4_BLOCK,), above))
        partial = torch.zeros(NVFP4_BLOCK)
        partial[:5] = f32([1.0, 0.61, 0.33, 0.07, 0.004])
        blocks.append(partial)
        over = torch.zeros(NVFP4_BLOCK)
        over[0] = torch.nextafter(f32([1.0]), f32([2.0])).item()
        over[1:4] = f32([0.5, 0.26, 0.9])
        blocks.append(over)
        gen = torch.Generator().manual_seed(1)
        blocks.append(torch.rand(64 * NVFP4_BLOCK, generator=gen))
        peaked = torch.rand(64, NVFP4_BLOCK, generator=gen) * torch.exp2(
            -torch.randint(0, 20, (64, 1), generator=gen).float()
        )
        blocks.append(peaked.flatten())
        chunks = blocks
    return torch.cat([c.flatten() for c in chunks])


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_device_codec_is_bit_identical_to_the_cpu_oracle(fmt):
    x = codec_inputs(fmt)
    out = run_codec_probe(x.to("cuda"), fmt)
    if is_fake_mode():
        return
    expected = qdq(torch.nn.functional.pad(x, (0, (-x.numel()) % 32)), fmt)[: x.numel()]
    got = out.cpu()
    mismatch = got != expected
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} of {x.numel()} codec outputs differ; first: "
        f"x={x[mismatch][0].item()!r} device={got[mismatch][0].item()!r} "
        f"oracle={expected[mismatch][0].item()!r}"
    )
    # A wrong tie rule is told apart on these inputs.
    wrong = qdq(
        torch.nn.functional.pad(x, (0, (-x.numel()) % 32)), fmt, rounding="half_away"
    )
    assert not torch.equal(got, wrong[: x.numel()])


# -------------------------------------------------------------------- forward

CASES = [
    dict(id="hd128_gqa_causal", b=2, sq=300, sk=300, h=4, hk=2, d=128, causal=True),
    dict(id="hd64_mha", b=2, sq=200, sk=333, h=2, hk=2, d=64),
    dict(id="hd96_window", b=1, sq=257, sk=257, h=4, hk=2, d=96, window=(70, 0)),
    dict(
        id="hd128_pack_gqa",
        b=1,
        sq=290,
        sk=290,
        h=4,
        hk=1,
        d=128,
        causal=True,
        pack_gqa=True,
    ),
    dict(
        id="hd128_no_pack_gqa",
        b=1,
        sq=290,
        sk=290,
        h=4,
        hk=1,
        d=128,
        causal=True,
        pack_gqa=False,
    ),
    dict(id="hd64_unequal_causal", b=1, sq=150, sk=401, h=2, hk=2, d=64, causal=True),
    dict(id="hd64_empty_rows", b=1, sq=400, sk=150, h=2, hk=2, d=64, causal=True),
    dict(id="hd128_split3", b=1, sq=130, sk=1500, h=2, hk=2, d=128, num_splits=3),
    dict(
        id="hd64_split2_causal",
        b=1,
        sq=700,
        sk=700,
        h=2,
        hk=1,
        d=64,
        causal=True,
        num_splits=2,
    ),
    dict(
        id="hd128_split2_window",
        b=1,
        sq=600,
        sk=600,
        h=2,
        hk=2,
        d=128,
        window=(300, 0),
        num_splits=2,
    ),
    dict(
        id="hd128_fixed_split",
        b=1,
        sq=100,
        sk=1400,
        h=2,
        hk=2,
        d=128,
        # A fixed extent only applies under SplitKV: 1400 keys in 512-key splits need 3.
        num_splits=3,
        seqlen_k_per_split=512,
    ),
    # Automatic SplitKV (num_splits=0): the selector picks the count. Few query tiles (32, one
    # per batch and head) against 12 KV tiles on 148 SMs resolve to 4 splits of 3 tiles.
    dict(id="hd64_auto_split", b=4, sq=60, sk=1500, h=8, hk=8, d=64, num_splits=0),
    # The selector gives 2-CTA kernels only to non-causal, unsplit head-dim-128 calls; every
    # causal, windowed or SplitKV call above is 1-CTA.
    dict(id="hd128_noncausal_2cta", b=2, sq=260, sk=391, h=2, hk=2, d=128, cta2=True),
]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_forward_matches_the_tiled_oracle(fmt, case):
    b, sq, sk, h, hk, d = (case[n] for n in ("b", "sq", "sk", "h", "hk", "d"))
    q = rand_bf16(b, sq, h, d, seed=0)
    k = rand_bf16(b, sk, hk, d, seed=1)
    v = rand_bf16(b, sk, hk, d, seed=2)
    kw = call_kwargs(case)
    out, lse, config, inputs = fwd_with_config(q, k, v, fmt, **kw)
    out_ctl, lse_ctl = fwd(q, k, v, f"identity_{fmt}", config=config, **kw)
    if is_fake_mode():
        return
    sch = case_schedule(case, config, inputs, h // hk)
    scale = default_scale(d)
    # The case runs the kernel variant it is meant to cover.
    assert config.use_2cta_instrs == case.get("cta2", False)
    if "num_splits" in case or "seqlen_k_per_split" in case:
        assert config.num_splits > 1

    # No NaN anywhere; LSE is -inf exactly on empty rows.
    assert not torch.isnan(out).any() and not torch.isnan(lse).any()
    assert torch.isfinite(out).all()
    # N9 stop gate: the row sum never sees quantized P, so LSE equals its control bit for bit.
    assert torch.equal(lse, lse_ctl), "LSE differs from the identity control"

    math_ctl, lse_math, _, _ = oracle_batch(q, k, v, scale, sch, f"identity_{fmt}")
    impl_ctl, lse_impl, _, _ = oracle_batch(
        q, k, v, scale, sch, f"identity_{fmt}", path="impl"
    )
    impl_fmt, lse_fmt, prob_allow, out_allow = oracle_batch(
        q, k, v, scale, sch, fmt, path="impl", with_allowance=True
    )
    noise = rounding_noise(impl_ctl, math_ctl)

    empty = torch.isinf(lse_impl)
    assert torch.equal(torch.isinf(lse.cpu()), empty)
    assert (lse.cpu()[empty] < 0).all()
    assert (out.cpu().transpose(1, 2)[empty] == 0).all()  # empty rows give exactly zero
    lse_tol = 2 * (lse_impl - lse_math)[~empty].abs().max().item() + 1e-4
    assert ((lse.cpu().double() - lse_impl)[~empty].abs() <= lse_tol).all()

    assert_matches_oracle("identity control", out_ctl, impl_ctl, noise)
    assert_matches_oracle(fmt, out, impl_fmt, noise, allowance=out_allow)
    check_allowance_budgets(prob_allow, ~empty)
    # The mode is really on: quantization changes the output well beyond rounding noise.
    assert (out.float() - out_ctl.float()).abs().max().item() > MIN_EFFECT[fmt]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    "split_kw",
    [
        dict(),
        dict(num_splits=2),
        dict(num_splits=2, disable_scheduler_metadata=True),
        dict(
            num_splits=0
        ),  # automatic: the selector picks the upper bound of the counts
    ],
    ids=["unsplit", "per_sequence_splits", "static_splits", "auto_splits"],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_varlen_forward_matches_the_tiled_oracle(fmt, split_kw):
    h, hk, d = 4, 2, 64
    lens_q = [130, 0, 77, 301]  # a zero-length sequence in the batch
    lens_k = [700, 64, 0, 301]  # ... and one with no keys
    cu_q = torch.tensor(
        [0, *itertools.accumulate(lens_q)], dtype=torch.int32, device="cuda"
    )
    cu_k = torch.tensor(
        [0, *itertools.accumulate(lens_k)], dtype=torch.int32, device="cuda"
    )
    q = rand_bf16(sum(lens_q), h, d, seed=3)
    k = rand_bf16(sum(lens_k), hk, d, seed=4)
    v = rand_bf16(sum(lens_k), hk, d, seed=5)
    # The split variants are non-causal so every row sees many tiles.
    causal = not split_kw
    kw = dict(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(lens_q),
        max_seqlen_k=max(lens_k),
        causal=causal,
        **split_kw,
    )
    with capture_fwd_config() as captured:
        out, lse = fwd(q, k, v, fmt, **kw)
    config, inputs = captured[0]
    out_ctl, lse_ctl = fwd(q, k, v, f"identity_{fmt}", config=config, **kw)
    if is_fake_mode():
        return
    assert torch.equal(lse, lse_ctl)
    assert lse.shape == (h, sum(lens_q))  # varlen layout: heads first, tokens flat
    counts = per_sequence_split_counts(captured)
    if "num_splits" in split_kw:
        # The case runs SplitKV, also when the selector chose the count (num_splits=0).
        assert config.num_splits > 1
    if "num_splits" in split_kw and not split_kw.get("disable_scheduler_metadata"):
        # The oracle uses the split count the kernel's scheduler metadata holds per sequence.
        assert needs_split_metadata(config, inputs)
        assert counts is not None and len(counts) == len(lens_q)
        assert all(1 <= c <= config.num_splits for c in counts)
    else:
        assert counts is None
    scale = default_scale(d)
    oq, ok = 0, 0
    for b, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        qs, ks, vs = q[oq : oq + lq][None], k[ok : ok + lk][None], v[ok : ok + lk][None]
        o_k, l_k = out[oq : oq + lq][None], lse[:, oq : oq + lq][None]
        o_c = out_ctl[oq : oq + lq][None]
        oq, ok = oq + lq, ok + lk
        if lq == 0:
            continue
        if lk == 0:
            assert (o_k == 0).all() and torch.isinf(l_k).all()
            continue
        sch = schedule_of(
            config,
            inputs,
            qhead_per_kvhead=h // hk,
            causal=causal,
            num_splits=None if counts is None else counts[b],
        )
        math_ctl, lse_math, _, _ = oracle_batch(
            qs, ks, vs, scale, sch, f"identity_{fmt}"
        )
        impl_ctl, lse_impl, _, _ = oracle_batch(
            qs, ks, vs, scale, sch, f"identity_{fmt}", path="impl"
        )
        impl_fmt, lse_fmt, prob_allow, out_allow = oracle_batch(
            qs, ks, vs, scale, sch, fmt, path="impl", with_allowance=True
        )
        noise = rounding_noise(impl_ctl, math_ctl)
        assert_matches_oracle("identity control", o_c, impl_ctl, noise)
        assert_matches_oracle(fmt, o_k, impl_fmt, noise, allowance=out_allow)
        empty = torch.isinf(lse_fmt)
        assert torch.equal(torch.isinf(l_k.cpu()), empty)
        check_allowance_budgets(prob_allow, ~empty)
        # The LSE does not depend on the format. The dense test's rule: twice the distance
        # between the oracle's implementation path and exact math, plus float32 slack.
        assert torch.equal(lse_fmt, lse_impl)
        lse_tol = 2 * (lse_impl - lse_math)[~empty].abs().max().item() + 1e-4
        assert lse_tol < 1e-3  # far below the size of an LSE; rows differ by much more

        def lse_agrees(got):
            return ((got - lse_impl)[~empty].abs() <= lse_tol).all().item()

        got = l_k.cpu().double()
        assert lse_agrees(got)
        # An LSE off by a few times the bound, in either direction, does not pass.
        if not empty.all():
            assert not lse_agrees(got + 4 * lse_tol)
            assert not lse_agrees(got - 4 * lse_tol)
        # A dense-layout reading of the varlen LSE (tokens first) is told apart.
        if lq > h and not empty.any():
            assert not lse_agrees(got[0].t().reshape(1, h, lq))


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_keys_past_the_used_length_do_not_change_the_output(fmt):
    b, s, h, d = 2, 300, 2, 64
    q, k, v = (
        rand_bf16(b, s, h, d, seed=6),
        rand_bf16(b, s, h, d, seed=7),
        rand_bf16(b, s, h, d, seed=8),
    )
    used = torch.tensor([180, 257], dtype=torch.int32, device="cuda")
    v_poison = v.clone()
    k_poison = k.clone()
    if not is_fake_mode():
        for bi, n in enumerate(used.tolist()):
            v_poison[bi, n:] = 3e4  # masked keys carry huge values ...
            k_poison[bi, n:] = 30.0  # ... and scores that would dominate if they leaked
    out, lse = fwd(q, k, v, fmt, seqused_k=used, causal=True)
    out_p, lse_p = fwd(q, k_poison, v_poison, fmt, seqused_k=used, causal=True)
    if is_fake_mode():
        return
    assert torch.equal(out, out_p) and torch.equal(lse, lse_p)


# -------------------------------------------------------- constructed witnesses


def kernel_witness(sk=384, d=64):
    """One query row over three 128-key tiles with scattered, off-grid probabilities.

    The row max sits on key 5 (the tile visited last) and a smaller maximum on key 300 (the
    tile visited first), so the middle tile is quantized against 0.83**3 in one pass, against
    its own maximum when a split restarts there, and against 1 in a textbook reference.
    """
    order = 8 + ((torch.arange(sk) * 37) % sk) % 40
    order[5], order[300] = 0, 3
    q = torch.zeros(1, 1, 1, d)
    q[0, 0, 0, 0] = 1.0
    k = torch.zeros(1, sk, 1, d)
    k[0, :, 0, 0] = math.log(0.83) * order.float()
    gen = torch.Generator().manual_seed(11)
    v = torch.randn(1, sk, 1, d, generator=gen)
    return tuple(t.to(torch.bfloat16).to("cuda") for t in (q, k, v))


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_witness_matches_only_the_schedule_aware_oracle(fmt):
    q, k, v = kernel_witness()
    out, lse, config, inputs = fwd_with_config(q, k, v, fmt, softmax_scale=1.0)
    out_ctl, _ = fwd(q, k, v, f"identity_{fmt}", softmax_scale=1.0, config=config)
    out_split, lse_split, config_s, inputs_s = fwd_with_config(
        q, k, v, fmt, softmax_scale=1.0, num_splits=2
    )
    if is_fake_mode():
        return
    sch = schedule_of(config, inputs, qhead_per_kvhead=1)
    sch_split = schedule_of(config_s, inputs_s, qhead_per_kvhead=1)
    assert sch.tile_n == 128 and sch_split.num_splits == 2
    math_ctl, _, _, _ = oracle_batch(q, k, v, 1.0, sch, f"identity_{fmt}")
    impl_ctl, _, _, _ = oracle_batch(q, k, v, 1.0, sch, f"identity_{fmt}", path="impl")
    noise = rounding_noise(impl_ctl, math_ctl)
    for name, got, s in (("one pass", out, sch), ("two splits", out_split, sch_split)):
        impl, _, prob_allow, _ = oracle_batch(
            q, k, v, 1.0, s, fmt, path="impl", with_allowance=True
        )
        assert prob_allow.max() == 0, (
            "the witness must sit away from every decision boundary"
        )
        assert_matches_oracle(f"{fmt} {name}", got, impl, noise)
    # The schedule matters, and a reference that ignores it is wrong by far more than noise.
    qc, kc, vc = (t[0, :, 0].cpu().double() for t in (q, k, v))
    textbook = textbook_attention(qc[:, None], kc, vc, 1.0, sch, fmt).out
    bound = 2 * noise + 2 * (out.float() + 0.3 - 0.3 - out.float()).abs().max().item()
    assert (out.cpu().double()[0] - textbook).abs().max().item() > max(
        MIN_EFFECT[fmt], 4 * bound
    )
    assert (out.float() - out_split.float()).abs().max().item() > MIN_EFFECT[fmt]
    # A format that leaves the control's output unchanged would be a silent no-op.
    assert (out.float() - out_ctl.float()).abs().max().item() > MIN_EFFECT[fmt]


def readout_inputs(fmt, sch, slice_start, d=64, sk=128, seed=0):
    """P-readout probe: one query, V rows are basis vectors, so output column c is the
    quantized probability of the active key that owns column c divided by the row sum.

    QK arithmetic is exact (q is a unit vector, so each score is one BF16 value). Keys whose
    codec decision the kernel arithmetic could flip are deactivated, so the probe needs no
    allowance.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.zeros(1, 1, 1, d)
    q[0, 0, 0, 0] = 1.0
    k = torch.zeros(1, sk, 1, d)
    k[0, :, 0, 0] = -200.0  # inactive: probability underflows to zero
    active = torch.arange(slice_start, slice_start + d)
    # Scores spread over 12 octaves below the maximum, on a 1/16 grid (exact in BF16).
    scores = -torch.randint(0, 12 * 16, (d,), generator=gen).float() / 16
    scores[0] = 0.0  # the row max
    k[0, active, 0, 0] = scores
    v = torch.zeros(1, sk, 1, d)
    v[0, active, 0, torch.arange(d)] = 1.0
    q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
    scale = math.log(2.0)  # scores are then (almost exactly) log2 probabilities
    for _ in range(8):
        res = tiled_attention(
            q[0].double(),
            k[0, :, 0].double(),
            v[0, :, 0].double(),
            scale,
            sch,
            fmt,
            path="impl",
            with_allowance=True,
        )
        flagged = res.eligible_keys[0, 0]
        if not flagged.any():
            break
        k[0, flagged, 0, 0] = -200.0
    else:
        raise AssertionError("could not place the probe away from decision boundaries")
    return q.to("cuda"), k.to("cuda"), v.to("cuda"), scale


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("slice_start", [0, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_p_readout_probe_matches_the_codec(fmt, slice_start, seed):
    d, sk = 64, 128
    placeholder = tuple(
        torch.zeros(1, n, 1, d, dtype=torch.bfloat16, device="cuda")
        for n in (1, sk, sk)
    )
    # Resolve the config first (the schedule decides which columns use the emulated exp2).
    _, _, config, inputs = fwd_with_config(
        *placeholder, fmt, softmax_scale=math.log(2.0)
    )
    if is_fake_mode():
        # The call the execution pass makes: same shapes, with the resolved config.
        fwd(*placeholder, fmt, softmax_scale=math.log(2.0), config=config)
        return
    sch = schedule_of(config, inputs, qhead_per_kvhead=1)
    q, k, v, scale = readout_inputs(fmt, sch, slice_start, seed=seed)
    out, lse = fwd(q, k, v, fmt, softmax_scale=scale, config=config)
    impl, _, prob_allow, _ = oracle_batch(
        q, k, v, scale, sch, fmt, path="impl", with_allowance=True
    )
    assert prob_allow.max() == 0
    # Each column is one probability: only BF16 output rounding and the reciprocal separate
    # the kernel from the oracle.
    got, want = out.cpu().double(), impl
    assert ((got - want).abs() <= 2.0**-7 * want.abs() + 1e-9).all()
    # The readout really is quantized: it differs from the unquantized probabilities.
    ctl, _, _, _ = oracle_batch(q, k, v, scale, sch, f"identity_{fmt}", path="impl")
    assert (want - ctl).abs().max() > MIN_EFFECT[fmt]


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_overshoot_of_the_ideal_range_matches_the_oracle(fmt):
    # Head dim 96 with a raw score of 109 on the row max: the separately rounded bias leaves
    # a positive residual, so the codec input lands just above its ideal maximum.
    d, sk = 96, 160
    q = torch.zeros(1, 1, 1, d)
    q[0, 0, 0, 0] = 1.0
    k = torch.zeros(1, sk, 1, d)
    k[0, :, 0, 0] = 109.0 - torch.arange(sk).float() / 4
    gen = torch.Generator().manual_seed(5)
    v = torch.randn(1, sk, 1, d, generator=gen)
    q, k, v = (t.to(torch.bfloat16).to("cuda") for t in (q, k, v))
    _, _, config, inputs = fwd_with_config(q, k, v, fmt)
    if is_fake_mode():
        fwd(q, k, v, fmt, config=config)  # the call the execution pass makes
        return
    sch = schedule_of(config, inputs, qhead_per_kvhead=1)
    scale = default_scale(d)
    # A constructed witness needs no allowance: keys near a decision boundary are deactivated.
    k = settle_single_row(q, k, v, scale, sch, fmt)
    out, lse = fwd(q, k, v, fmt, config=config)
    math_ctl, _, _, _ = oracle_batch(q, k, v, scale, sch, f"identity_{fmt}")
    impl_ctl, _, _, _ = oracle_batch(
        q, k, v, scale, sch, f"identity_{fmt}", path="impl"
    )
    impl, _, prob_allow, _ = oracle_batch(
        q, k, v, scale, sch, fmt, path="impl", with_allowance=True
    )
    assert prob_allow.max() == 0
    # The row max really overshoots its ideal value in the kernel's arithmetic.
    a32 = torch.tensor(scale * math.log2(math.e), dtype=torch.float32)
    m32 = torch.tensor(109.0, dtype=torch.float32)
    residual = (m32.double() * a32.double() - (m32 * a32).double()).item()
    assert residual > 0, "this witness needs a positive bias residual"
    assert torch.isfinite(out).all()
    assert_matches_oracle(fmt, out, impl, rounding_noise(impl_ctl, math_ctl))


# ------------------------------------------------------------------ gradients


def bwd_cache_keys():
    return {
        name: set(getattr(interface, fn).compile_cache.cache)
        for name, fn in (
            ("bwd_pre", "_bwd_preprocess"),
            ("bwd", "_flash_attn_bwd"),
            ("bwd_post", "_bwd_postprocess_convert"),
        )
    }


GRAD_CASES = [
    dict(id="hd128_gqa_causal", b=1, sq=200, sk=200, h=4, hk=2, d=128, causal=True),
    dict(id="hd64_mha", b=2, sq=130, sk=190, h=2, hk=2, d=64),
    dict(id="hd96_window", b=1, sq=150, sk=150, h=2, hk=2, d=96, window=(40, 0)),
    dict(id="hd64_empty_rows", b=1, sq=200, sk=90, h=2, hk=2, d=64, causal=True),
    dict(
        id="hd128_deterministic",
        b=1,
        sq=160,
        sk=160,
        h=2,
        hk=2,
        d=128,
        causal=True,
        deterministic=True,
    ),
]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("with_dlse", [False, True], ids=["no_dlse", "dlse"])
@pytest.mark.parametrize("case", GRAD_CASES, ids=[c["id"] for c in GRAD_CASES])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_backward_is_the_contract_surrogate(fmt, case, with_dlse):
    b, sq, sk, h, hk, d = (case[n] for n in ("b", "sq", "sk", "h", "hk", "d"))
    q = rand_bf16(b, sq, h, d, seed=10).requires_grad_()
    k = rand_bf16(b, sk, hk, d, seed=11).requires_grad_()
    v = rand_bf16(b, sk, hk, d, seed=12).requires_grad_()
    g_out = rand_bf16(b, sq, h, d, seed=13)
    g_lse = rand_bf16(b, h, sq, seed=14).float()
    kw = dict(
        causal=case.get("causal", False), deterministic=case.get("deterministic", False)
    )
    if "window" in case:
        kw["window_size"] = case["window"]

    def run(p_quant):
        for t in (q, k, v):
            t.grad = None
        out, lse = flash_attn_func(q, k, v, return_lse=True, p_quant=p_quant, **kw)
        # Explicit output gradients, the same call in the compile pass and the execution
        # pass: the backward preprocess kernel is specialized on whether an LSE gradient
        # is present.
        if with_dlse:
            # Empty rows have LSE = -inf; their LSE gradient is dropped.
            lse_grad = (
                g_lse
                if is_fake_mode()
                else torch.where(torch.isinf(lse), torch.zeros_like(g_lse), g_lse)
            )
            torch.autograd.backward([out, lse], [g_out, lse_grad])
        else:
            out.backward(g_out)
        if is_fake_mode():
            return None
        return (
            out.detach(),
            lse.detach(),
            q.grad.clone(),
            k.grad.clone(),
            v.grad.clone(),
        )

    baseline = run(None)  # warms the backward kernels of the unquantized call
    keys_before = bwd_cache_keys()
    result = run(fmt)
    control = run(f"identity_{fmt}")
    if is_fake_mode():
        return
    # The mode is forward-only: no backward kernel is added for it.
    assert bwd_cache_keys() == keys_before
    out, lse, dq, dk, dv = result
    wl, wr = case.get("window", (None, None))
    causal, _, wl, wr = interface._resolve_causal_local_window(
        kw["causal"], wl, wr, None
    )
    masks = Schedule(
        128,
        128,
        causal=causal,
        window_left=None if causal else wl,
        window_right=None if causal else wr,
    )
    scale = default_scale(d)
    g_lse_eff = (
        torch.where(torch.isinf(lse), torch.zeros_like(g_lse), g_lse)
        if with_dlse
        else None
    )
    ref = surrogate_grads(q, k, v, out, lse, g_out, scale, masks, dlse=g_lse_eff)
    low = surrogate_grads(
        q,
        k,
        v,
        out,
        lse,
        g_out,
        scale,
        masks,
        dlse=g_lse_eff,
        operand_dtype=torch.bfloat16,
    )
    for name, got, r, p in zip(("dq", "dk", "dv"), (dq, dk, dv), ref, low):
        assert torch.isfinite(got).all()
        assert_matches_oracle(f"{fmt} {name}", got, r, (p - r).abs().max().item())
    # Empty rows contribute no query gradient.
    empty_rows = torch.isinf(lse).any(dim=1)  # [B, Sq]
    assert (dq[empty_rows] == 0).all()
    # The correction term uses the quantized output: the same formula evaluated with the
    # control's unquantized output is told apart from the kernel's gradient.
    out_ctl = control[0]
    wrong = surrogate_grads(q, k, v, out_ctl, lse, g_out, scale, masks, dlse=g_lse_eff)
    gap = (wrong[0] - ref[0]).abs().max().item()
    bound = 2 * (low[0] - ref[0]).abs().max().item()
    if gap > 4 * bound:  # only meaningful when the difference clears the rounding bound
        assert (dq.cpu().double() - wrong[0]).abs().max().item() > bound
    del baseline


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_surrogate_uses_the_quantized_output_on_a_witness(fmt):
    """dO is aligned with the quantization error of O, so rowsum(dO * O_q) and
    rowsum(dO * O) differ by far more than rounding, and only the former matches."""
    b, s, h, d = 1, 192, 2, 64
    q = rand_bf16(b, s, h, d, seed=20).requires_grad_()
    k = rand_bf16(b, s, h, d, seed=21).requires_grad_()
    v = (rand_bf16(b, s, h, d, seed=22).float() * 4).to(torch.bfloat16).requires_grad_()
    out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True, p_quant=fmt)
    out_ctl, _ = flash_attn_func(
        q, k, v, causal=True, return_lse=True, p_quant=f"identity_{fmt}"
    )
    if is_fake_mode():
        out.backward(torch.zeros_like(out))
        return
    g_out = (torch.sign(out.detach().float() - out_ctl.detach().float()) * 8).to(
        torch.bfloat16
    )
    (out * g_out).sum().backward()
    masks = Schedule(128, 128, causal=True)
    scale = default_scale(d)
    args = (q, k, v)
    ref = surrogate_grads(*args, out.detach(), lse.detach(), g_out, scale, masks)
    low = surrogate_grads(
        *args,
        out.detach(),
        lse.detach(),
        g_out,
        scale,
        masks,
        operand_dtype=torch.bfloat16,
    )
    wrong = surrogate_grads(*args, out_ctl.detach(), lse.detach(), g_out, scale, masks)
    noise = (low[0] - ref[0]).abs().max().item()
    assert_matches_oracle(f"{fmt} dq", q.grad, ref[0], noise)
    gap = (wrong[0] - ref[0]).abs().max().item()
    assert gap > 8 * noise, "witness too weak to tell the two correction terms apart"
    with pytest.raises(AssertionError):
        assert_matches_oracle("unquantized-output reference", q.grad, wrong[0], noise)


# ------------------------------------------------- varlen and used-length gradients


def check_surrogate(
    name, grads, q, k, v, out, lse, dout, scale, masks, dlse=None, seqlens=None
):
    """Kernel gradients against the FP64 surrogate of the contract (section 4)."""
    args = (q, k, v, out, lse, dout, scale, masks)
    ref = surrogate_grads(*args, dlse=dlse, seqlens=seqlens)
    low = surrogate_grads(
        *args, dlse=dlse, seqlens=seqlens, operand_dtype=torch.bfloat16
    )
    for part, got, r, lo in zip(("dq", "dk", "dv"), grads, ref, low):
        assert torch.isfinite(got).all()
        assert_matches_oracle(f"{name} {part}", got, r, (lo - r).abs().max().item())


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("with_dlse", [False, True], ids=["no_dlse", "dlse"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_varlen_backward_is_the_contract_surrogate(fmt, with_dlse):
    h, hk, d = 4, 2, 64
    lens_q, lens_k = [90, 0, 150], [120, 40, 150]
    cu_q = torch.tensor(
        [0, *itertools.accumulate(lens_q)], dtype=torch.int32, device="cuda"
    )
    cu_k = torch.tensor(
        [0, *itertools.accumulate(lens_k)], dtype=torch.int32, device="cuda"
    )
    q = rand_bf16(sum(lens_q), h, d, seed=70).requires_grad_()
    k = rand_bf16(sum(lens_k), hk, d, seed=71).requires_grad_()
    v = rand_bf16(sum(lens_k), hk, d, seed=72).requires_grad_()
    g_out = rand_bf16(sum(lens_q), h, d, seed=73)
    g_lse = rand_bf16(h, sum(lens_q), seed=74).float()
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(lens_q),
        max_seqlen_k=max(lens_k),
        causal=True,
        return_lse=True,
        p_quant=fmt,
    )
    if with_dlse:
        lse_grad = (
            g_lse
            if is_fake_mode()
            else torch.where(torch.isinf(lse), torch.zeros_like(g_lse), g_lse)
        )
        torch.autograd.backward([out, lse], [g_out, lse_grad])
    else:
        out.backward(g_out)
    if is_fake_mode():
        return
    masks = Schedule(128, 128, causal=True)
    scale = default_scale(d)
    oq, ok = 0, 0
    for lq, lk in zip(lens_q, lens_k):
        sq_, sk_ = slice(oq, oq + lq), slice(ok, ok + lk)
        oq, ok = oq + lq, ok + lk
        if lq == 0:
            # No query: the keys of this sequence receive no gradient.
            assert (k.grad[sk_] == 0).all() and (v.grad[sk_] == 0).all()
            continue
        lse_b = lse.detach()[:, sq_][None]
        dlse_b = None
        if with_dlse:
            dlse_b = torch.where(
                torch.isinf(lse_b), torch.zeros_like(lse_b), g_lse[:, sq_][None]
            )
        check_surrogate(
            fmt,
            (q.grad[sq_][None], k.grad[sk_][None], v.grad[sk_][None]),
            q[sq_][None],
            k[sk_][None],
            v[sk_][None],
            out.detach()[sq_][None],
            lse_b,
            g_out[sq_][None],
            scale,
            masks,
            dlse=dlse_b,
        )


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_used_length_forward_and_backward(fmt):
    """Dense storage with per-sequence used lengths on both sides (seqused_q, seqused_k)."""
    b, s, h, d = 2, 220, 2, 64
    used_q = torch.tensor([150, 220], dtype=torch.int32, device="cuda")
    used_k = torch.tensor([201, 97], dtype=torch.int32, device="cuda")
    q = rand_bf16(b, s, h, d, seed=80).requires_grad_()
    k = rand_bf16(b, s, h, d, seed=81).requires_grad_()
    v = rand_bf16(b, s, h, d, seed=82).requires_grad_()
    g_out = rand_bf16(b, s, h, d, seed=83)
    with capture_fwd_config() as captured:
        out, lse = flash_attn_varlen_func(
            q, k, v, seqused_q=used_q, seqused_k=used_k, return_lse=True, p_quant=fmt
        )
    config, inputs = captured[0]
    # The identity control of the same call, with the config the format resolved.
    _, lse_ctl = flash_attn_varlen_func(
        q,
        k,
        v,
        seqused_q=used_q,
        seqused_k=used_k,
        return_lse=True,
        p_quant=f"identity_{fmt}",
        config=config,
    )
    out.backward(g_out)
    if is_fake_mode():
        return
    seqlens = list(zip(used_q.tolist(), used_k.tolist()))
    sch = schedule_of(config, inputs, qhead_per_kvhead=1)
    scale = default_scale(d)
    math_ctl, lse_math, _, _ = oracle_batch(
        q, k, v, scale, sch, f"identity_{fmt}", seqlens=seqlens
    )
    impl_ctl, lse_impl, _, _ = oracle_batch(
        q, k, v, scale, sch, f"identity_{fmt}", path="impl", seqlens=seqlens
    )
    impl, lse_ref, prob_allow, out_allow = oracle_batch(
        q, k, v, scale, sch, fmt, path="impl", with_allowance=True, seqlens=seqlens
    )
    valid = torch.zeros(b, s, dtype=torch.bool)
    for bi, (lq, _) in enumerate(seqlens):
        valid[bi, :lq] = True
    noise = rounding_noise(impl_ctl, math_ctl)
    # Rows past the used query length are padding and are not compared.
    assert_matches_oracle(
        fmt, out.detach().cpu()[valid], impl[valid], noise, allowance=out_allow[valid]
    )
    check_allowance_budgets(prob_allow.transpose(1, 2)[valid])
    # N9 stop gate on the rows inside the used query length; padding rows are not compared.
    assert lse.shape == (b, h, s)
    assert_identity_lse_gate(
        f"used length {fmt}",
        lse,
        lse_ctl,
        lse_impl,
        lse_math,
        valid=valid[:, None, :].expand(b, h, s),
    )
    masks = Schedule(128, 128)
    out_v = torch.where(
        valid[..., None, None].to(out.device), out.detach(), torch.zeros_like(out)
    )
    g_v = torch.where(
        valid[..., None, None].to(out.device), g_out, torch.zeros_like(g_out)
    )
    # Gradients are compared inside the used lengths only. The backward does not define
    # entries at padding positions: it leaves them unwritten (its buffers come from
    # torch.empty_like) or writes zero, the same with and without p_quant (measured on B200
    # with sentinel-filled buffers).
    valid_k = torch.zeros(b, s, dtype=torch.bool)
    for bi, (_, lk) in enumerate(seqlens):
        valid_k[bi, :lk] = True

    def inside(grad, used):
        return torch.where(
            used[..., None, None].to(grad.device), grad, torch.zeros_like(grad)
        )

    check_surrogate(
        fmt,
        (inside(q.grad, valid), inside(k.grad, valid_k), inside(v.grad, valid_k)),
        q,
        k,
        v,
        out_v,
        lse.detach(),
        g_v,
        scale,
        masks,
        seqlens=seqlens,
    )


# ------------------------------------------------- context-parallel patterns


def log_add_exp_merge(outs, lses):
    """Merge per-chunk results the way context parallel does: weights exp(lse_c - lse)."""
    out, lse = merge_partials(
        [o.float().cpu().double() for o in outs],
        [ls.cpu().double().transpose(1, 2) for ls in lses],
    )
    return out, lse.transpose(1, 2)


def cp_inputs(seed, s=256, h=2, d=64):
    return (
        rand_bf16(1, s, h, d, seed=seed),
        rand_bf16(1, s, h, d, seed=seed + 1),
        rand_bf16(1, s, h, d, seed=seed + 2),
    )


def oracle_with_noise(qs, ks, vs, scale, sch, fmt):
    """Implementation-path oracle with its allowances, the rounding-noise scale, and the
    identity control's oracle LSEs (implementation path, mathematical path) for the LSE gate."""
    math_ctl, lse_math, _, _ = oracle_batch(qs, ks, vs, scale, sch, f"identity_{fmt}")
    impl_ctl, lse_impl, _, _ = oracle_batch(
        qs, ks, vs, scale, sch, f"identity_{fmt}", path="impl"
    )
    impl, lse, prob_allow, out_allow = oracle_batch(
        qs, ks, vs, scale, sch, fmt, path="impl", with_allowance=True
    )
    noise = rounding_noise(impl_ctl, math_ctl)
    return impl, lse, prob_allow, out_allow, noise, (lse_impl, lse_math)


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_p2p_chunk_loop_matches_the_same_partition_oracle(fmt):
    """Point-to-point pattern for the second of two ranks, causal: its queries see the first
    rank's keys in full ("all" step) and its own keys causally ("diagonal" step); the two
    partial results are merged by LSE, and the backward runs per chunk on the merged output."""
    q, k, v = cp_inputs(30)
    half = q.shape[1] // 2
    q1 = q[:, half:].contiguous()
    chunks = [
        (k[:, :half].contiguous(), v[:, :half].contiguous(), False),
        (k[:, half:].contiguous(), v[:, half:].contiguous(), True),
    ]
    parts, scheds, lse_ctls = [], [], []
    for kc, vc, causal in chunks:
        o, ls, config, inputs = fwd_with_config(q1, kc, vc, fmt, causal=causal)
        parts.append((o, ls))
        scheds.append((config, inputs, causal))
        # The identity control of each chunk call, with the config the format resolved.
        _, ls_ctl = fwd(q1, kc, vc, f"identity_{fmt}", causal=causal, config=config)
        lse_ctls.append(ls_ctl)
    g_out = rand_bf16(*q1.shape, seed=33)
    if is_fake_mode():
        for (kc, vc, causal), (o, ls) in zip(chunks, parts):
            _flash_attn_bwd(q1, kc, vc, o, g_out, ls, causal=causal)
        return
    scale = default_scale(q.shape[-1])
    # Dense LSE layout: (batch, heads, seqlen).
    assert parts[0][1].shape == (1, q.shape[2], half)
    out, lse = log_add_exp_merge([p[0] for p in parts], [p[1] for p in parts])
    # Same-partition oracle: each chunk is its own kernel call with its own running max. The
    # chunk outputs and their allowances are merged with the chunks' LSE weights.
    o_parts, l_parts, a_parts, pa_parts, noise = [], [], [], [], 0.0
    for c, ((kc, vc, causal), (config, inputs, _)) in enumerate(zip(chunks, scheds)):
        sch = schedule_of(config, inputs, qhead_per_kvhead=1, causal=causal)
        o_i, l_i, pa_i, a_i, n_i, lse_oracles = oracle_with_noise(
            q1, kc, vc, scale, sch, fmt
        )
        # N9 stop gate per partition, before anything is merged.
        assert_identity_lse_gate(
            f"p2p {fmt} chunk {c}", parts[c][1], lse_ctls[c], *lse_oracles
        )
        noise = max(noise, n_i)
        o_parts.append(o_i)
        l_parts.append(l_i.transpose(1, 2))
        a_parts.append(a_i)
        pa_parts.append(pa_i.transpose(1, 2))
    o_ref, l_ref = merge_partials(o_parts, l_parts)
    weights = [torch.exp(l_c - l_ref) for l_c in l_parts]
    allow = sum(w.unsqueeze(-1) * a for w, a in zip(weights, a_parts))
    check_allowance_budgets(sum(w * pa for w, pa in zip(weights, pa_parts)))
    # The merge itself adds one BF16 rounding per chunk output.
    merge_noise = 2.0**-8 * o_ref.abs().max().item()
    assert_matches_oracle(
        f"p2p {fmt}", out, o_ref, noise + merge_noise, allowance=allow
    )
    assert torch.allclose(lse.transpose(1, 2), l_ref, rtol=0, atol=1e-3)
    # Backward per chunk on the merged output and LSE; dq sums over chunks.
    out_t = out.to(torch.bfloat16).to("cuda")
    lse_t = lse.float().to("cuda")
    dq_total = torch.zeros_like(q1, dtype=torch.float32)
    grads = []
    for kc, vc, causal in chunks:
        dq_c, dk_c, dv_c = _flash_attn_bwd(
            q1, kc, vc, out_t, g_out, lse_t, causal=causal
        )[:3]
        dq_total += dq_c.float()
        grads.append((dk_c, dv_c))
    # Rank 1's queries are the last `half` rows of a causal problem over all keys.
    dk = torch.cat([g[0] for g in grads], dim=1)
    dv = torch.cat([g[1] for g in grads], dim=1)
    check_surrogate(
        f"p2p {fmt}",
        (dq_total, dk, dv),
        q1,
        k,
        v,
        out_t,
        lse_t,
        g_out,
        scale,
        Schedule(128, 128, causal=True),
    )


def te_kv_seq_info_after_all_gather(
    local_chunk_id, cp_size, max_seqlen_q, max_seqlen_kv, window_size, causal
):
    """Transcription of Transformer Engine 2.20.2 ``get_kv_seq_info_after_all_gather``
    (attention/dot_product_attention/context_parallel.py, lines 3202-3237): the gathered K/V
    range one all-gather step slices and the window it passes, in TE's convention (-1 means
    unbounded). ``max_seqlen_q`` / ``max_seqlen_kv`` are the chunk lengths, i.e. the global
    lengths divided by ``2 * cp_size``."""
    local_chunk_end_idx = (local_chunk_id + 1) * max_seqlen_kv
    full_seq_end_idx = max_seqlen_kv * cp_size * 2

    if window_size is None:
        window_size = (-1, 0) if causal else (-1, -1)

    # Right boundary: how far past chunk_end the kernel can see.
    if window_size[1] == -1:
        seq_end_idx = full_seq_end_idx
        window_size_right = -1
    else:
        seq_end_idx = min(full_seq_end_idx, local_chunk_end_idx + window_size[1])
        window_size_right = local_chunk_end_idx + window_size[1] - seq_end_idx

    # Left boundary: how far before chunk_end - q_len the kernel can see.
    if window_size[0] == -1:
        seq_start_idx = 0
        window_size_left = -1
    else:
        seq_start_idx = max(0, local_chunk_end_idx - max_seqlen_q - window_size[0])
        window_size_left = window_size[0] + seq_end_idx - local_chunk_end_idx

    return (seq_start_idx, seq_end_idx), (window_size_left, window_size_right)


def fa4_window(te_window):
    """TE's per-step window with its -1 "unbounded" marker spelled as FA4's None.

    This is the one place the test departs from TE's literal call. TE passes
    ``window_size_left=-1, window_size_right=0`` next to ``causal=True``; this tree's
    ``_resolve_causal_local_window`` only reads -1 as "unbounded" when both sides are negative,
    so the literal pair is a local window with no visible key (every row empty). A caller that
    forwards TE's arguments has to do the same translation; see also
    test_te_causal_window_spelling_is_an_empty_local_window_in_this_tree in test_p_quant_api.py.
    """
    return tuple(None if w == -1 else w for w in te_window)


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_all_gather_chunks_match_the_same_partition_oracle(fmt):
    """All-gather pattern, as Transformer Engine 2.20.2 issues it for dense (bshd) causal
    attention on FA4 (AttnFuncWithCPAndKVAllGather): the sequence is cut into 2 * cp_size
    chunks, rank r holds chunks r and 2 * cp_size - 1 - r, and each step attends one query
    chunk to the gathered K/V sliced to the range ``get_kv_seq_info_after_all_gather`` returns
    (here the prefix up to the chunk's end), bottom-right aligned. With cp_size 2 that is the
    first and the last chunk on rank 0 and the two interior chunks on rank 1. The forward gets
    causal, softmax_scale, return_lse and the per-step window; the backward gets the same
    window, causal, softmax_scale, deterministic and preallocated dq / dk / dv."""
    cp_size, chunk = 2, 128
    q, k, v = cp_inputs(40, s=2 * cp_size * chunk)
    g_all = rand_bf16(*q.shape, seed=43)
    scale = default_scale(q.shape[-1])  # TE: q.shape[-1] ** (-0.5), passed explicitly
    runs = []
    for rank in range(cp_size):
        for chunk_id in (rank, 2 * cp_size - rank - 1):  # TE: local_seq_chunk_ids
            (start, end), te_window = te_kv_seq_info_after_all_gather(
                chunk_id, cp_size, chunk, chunk, (-1, 0), True
            )
            # Causal without a sliding window: the whole prefix up to the chunk's end.
            assert (start, end) == (0, (chunk_id + 1) * chunk)
            assert te_window == (-1, 0)
            wl, wr = fa4_window(te_window)
            rows = slice(chunk_id * chunk, (chunk_id + 1) * chunk)
            qs, gs = q[:, rows].contiguous(), g_all[:, rows].contiguous()
            ks, vs = k[:, start:end].contiguous(), v[:, start:end].contiguous()
            kw = dict(
                causal=True,
                softmax_scale=scale,
                window_size_left=wl,
                window_size_right=wr,
            )
            o, ls, cfg, inp = fwd_with_config(qs, ks, vs, fmt, **kw)
            # The identity control of each step, with the config the format resolved.
            _, ls_ctl = fwd(qs, ks, vs, f"identity_{fmt}", config=cfg, **kw)
            grads = _flash_attn_bwd(
                qs,
                ks,
                vs,
                o,
                gs,
                ls,
                deterministic=False,
                dq=torch.empty_like(qs),
                dk=torch.empty_like(ks),
                dv=torch.empty_like(vs),
                **kw,
            )[:3]
            runs.append(
                (chunk_id, rows, qs, ks, vs, gs, kw, o, ls, ls_ctl, cfg, inp, grads)
            )
    out_full, _ = fwd(q, k, v, fmt, causal=True)
    if is_fake_mode():
        return
    # First, interior and last chunks are all covered.
    assert sorted(run[0] for run in runs) == list(range(2 * cp_size))
    for chunk_id, rows, qs, ks, vs, gs, kw, o, ls, ls_ctl, cfg, inp, grads in runs:
        # One schedule, window bounds included, for the kernel call, the oracle and the
        # backward reference.
        sch = schedule_of(
            cfg,
            inp,
            qhead_per_kvhead=1,
            causal=kw["causal"],
            window_size_left=kw["window_size_left"],
            window_size_right=kw["window_size_right"],
        )
        # The step is the causal, bottom-right aligned call TE means: no row is empty.
        assert sch.causal and not sch.is_local
        impl, lse_ref, prob_allow, allow, noise, lse_oracles = oracle_with_noise(
            qs, ks, vs, scale, sch, fmt
        )
        assert torch.isfinite(lse_ref).all()
        name = f"all-gather chunk {chunk_id}"
        assert_matches_oracle(name, o, impl, noise, allowance=allow)
        check_allowance_budgets(prob_allow)
        assert torch.allclose(ls.cpu().double(), lse_ref, rtol=0, atol=1e-3)
        # N9 stop gate per partition.
        assert_identity_lse_gate(f"{name} {fmt}", ls, ls_ctl, *lse_oracles)
        check_surrogate(name, grads, qs, ks, vs, o, ls, gs, scale, sch)
        # The chunked result is a valid quantized attention, but it need not equal the
        # unchunked call: a query chunk is its own query tile, so its KV tiles can differ.
        drift = (o.float() - out_full[:, rows].float()).abs().max().item()
        print(f"all-gather drift vs unchunked ({fmt}, chunk {chunk_id}): {drift:.3e}")


@pytest.mark.parametrize("fmt", FORMATS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_a2a_head_scatter_equals_the_full_call_when_configs_match(fmt):
    """All-to-all pattern: each rank runs complete sequences for a subset of heads."""
    b, s, h, d = 1, 300, 4, 64
    q, k, v = (rand_bf16(b, s, h, d, seed=n) for n in (50, 51, 52))
    g_all = rand_bf16(b, s, h, d, seed=53)
    out_full, lse_full, cfg_full, in_full = fwd_with_config(q, k, v, fmt, causal=True)
    parts = []
    for heads in (slice(0, 2), slice(2, 4)):
        qs, ks, vs, gs = (t[:, :, heads].contiguous() for t in (q, k, v, g_all))
        o, ls, cfg, inp = fwd_with_config(qs, ks, vs, fmt, causal=True)
        # The identity control of each rank's call, with the config the format resolved.
        _, ls_ctl = fwd(qs, ks, vs, f"identity_{fmt}", causal=True, config=cfg)
        grads = _flash_attn_bwd(qs, ks, vs, o, gs, ls, causal=True)[:3]
        parts.append((heads, qs, ks, vs, gs, o, ls, ls_ctl, cfg, inp, grads))
    if is_fake_mode():
        return
    sch_full = schedule_of(cfg_full, in_full, qhead_per_kvhead=1, causal=True)
    scale = default_scale(d)
    for heads, qs, ks, vs, gs, o, ls, ls_ctl, cfg, inp, grads in parts:
        sch = schedule_of(cfg, inp, qhead_per_kvhead=1, causal=True)
        impl, _, prob_allow, allow, noise, lse_oracles = oracle_with_noise(
            qs, ks, vs, scale, sch, fmt
        )
        # N9 stop gate per partition (head subset).
        assert_identity_lse_gate(
            f"a2a {fmt} heads {heads.start}:{heads.stop}", ls, ls_ctl, *lse_oracles
        )
        if sch == sch_full:
            # Same schedule, same per-row arithmetic: the head subset reproduces the full call.
            assert torch.equal(o, out_full[:, :, heads]) and torch.equal(
                ls, lse_full[:, heads]
            )
        else:
            assert_matches_oracle("a2a head subset", o, impl, noise, allowance=allow)
            check_allowance_budgets(prob_allow)
        check_surrogate(
            f"a2a {fmt}",
            grads,
            qs,
            ks,
            vs,
            o,
            ls,
            gs,
            scale,
            Schedule(128, 128, causal=True),
        )


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_default_call_is_unchanged_by_the_keyword():
    q, k, v = cp_inputs(60)
    a, la = fwd(q, k, v, None, causal=True)
    b_, lb = fwd(q, k, v, "none", causal=True)
    c, lc, *_ = _flash_attn_fwd(q, k, v, return_lse=True, causal=True)
    if is_fake_mode():
        return
    assert (
        torch.equal(a, b_)
        and torch.equal(a, c)
        and torch.equal(la, lb)
        and torch.equal(la, lc)
    )
    assert resolve_p_quant(None) is None
    assert replace(Schedule(128, 128), causal=True).causal
