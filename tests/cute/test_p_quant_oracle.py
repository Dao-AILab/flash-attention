"""Tiled attention oracle for P quant-dequant: schedule, paths, witnesses (CPU only)."""

import math

import pytest
import torch

from flash_attn.cute.p_quant import NVFP4_BLOCK
from flash_attn.cute.testing import attention_ref
from p_quant_attention_oracle import (
    Schedule,
    key_visible,
    merge_partials,
    n_block_range,
    row_tiles,
    textbook_attention,
    tiled_attention,
)

FORMATS = ["fp8_e4m3", "nvfp4"]
# Smallest output change a schedule change must cause on the witness. FP8 has 8 codes
# per binade against NVFP4's 2, so its effects are an order smaller.
WITNESS_MIN_EFFECT = {"fp8_e4m3": 1e-4, "nvfp4": 1e-3}


def rand_qkv(seed, sq, sk, g=1, d=16, dv=None, bf16=True, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    dv = dv or d
    q = torch.randn(sq, g, d, generator=gen) * scale
    k = torch.randn(sk, d, generator=gen) * scale
    v = torch.randn(sk, dv, generator=gen)
    if bf16:
        q, k, v = (t.to(torch.bfloat16).to(torch.float64) for t in (q, k, v))
    return q.double(), k.double(), v.double()


def dense_reference(q, k, v, scale, sch):
    """Plain FP64 softmax attention with the schedule's masks (no tiling)."""
    sq, g, _ = q.shape
    out = torch.zeros(sq, g, v.shape[1], dtype=torch.float64)
    lse = torch.full((sq, g), float("-inf"), dtype=torch.float64)
    for i in range(sq):
        vis = key_visible(sch, sq, k.shape[0], i)
        if not vis.any():
            continue
        for h in range(g):
            s = (k @ q[i, h]) * scale
            s = torch.where(vis, s, torch.full_like(s, float("-inf")))
            lse[i, h] = torch.logsumexp(s, 0)
            out[i, h] = torch.softmax(s, 0) @ v
    return out, lse


SCHEDULES = [
    Schedule(tile_n=32, tile_m=64),
    Schedule(tile_n=16, tile_m=32, causal=True),
    Schedule(tile_n=32, tile_m=16, window_left=40, window_right=8),
    Schedule(tile_n=16, tile_m=64, num_splits=3),
    Schedule(tile_n=16, tile_m=32, causal=True, num_splits=2),
    Schedule(tile_n=16, tile_m=32, causal=True, n_blocks_per_split=2),
    Schedule(tile_n=32, tile_m=32, pack_factor=4, causal=True),
    Schedule(tile_n=16, tile_m=48, pack_factor=2, window_left=30, num_splits=2),
]


# ------------------------------------------------------------ schedule builder


def test_n_block_range_matches_hand_computed_cases():
    # Non-causal: every query tile sees all KV tiles.
    assert n_block_range(Schedule(32, 64), 100, 100, 0) == (0, 4)
    # Causal, Sq == Sk: query tile 0 (rows 0..63) needs keys 0..63 -> tiles 0..1.
    sch = Schedule(32, 64, causal=True)
    assert n_block_range(sch, 100, 100, 0) == (0, 2)
    assert n_block_range(sch, 100, 100, 1) == (0, 4)
    # Causal with Sk > Sq: the diagonal is shifted right by Sk - Sq.
    assert n_block_range(sch, 64, 100, 0) == (0, 4)
    # Local window: left bound drops leading tiles.
    sch = Schedule(16, 32, window_left=10, window_right=0)
    assert n_block_range(sch, 128, 128, 2) == ((64 - 10) // 16, 96 // 16)
    # Pack-GQA: tile_m counts rows = positions * g.
    sch = Schedule(32, 64, pack_factor=4, causal=True)
    assert n_block_range(sch, 100, 100, 0) == (0, 1)  # rows 0..63 -> positions 0..15


def test_split_ranges_partition_the_unsplit_range_in_order():
    for sch in SCHEDULES:
        if not sch.is_split_kv:
            continue
        base = Schedule(
            sch.tile_n,
            sch.tile_m,
            sch.pack_factor,
            sch.causal,
            sch.window_left,
            sch.window_right,
        )
        for q_pos in range(0, 100, 7):
            for h in range(sch.pack_factor):
                unsplit = row_tiles(base, 100, 117, q_pos, h)[0]
                splits = row_tiles(sch, 100, 117, q_pos, h)
                merged = sorted(t for s in splits for t in s)
                assert merged == sorted(unsplit)
                for s in splits:
                    assert s == sorted(s, reverse=True)  # last tile first


def test_ceil_divided_splits_can_be_empty():
    sch = Schedule(16, 32, causal=True, num_splits=4)
    # Query tile 0 of a causal problem owns 2 KV tiles: 4 splits -> 1 block each, 2 empty.
    assert row_tiles(sch, 64, 64, 0) == [[0], [1], [], []]


# ------------------------------------------------- mathematical path = attention


@pytest.mark.parametrize("sch", SCHEDULES)
@pytest.mark.parametrize("sq,sk", [(50, 50), (37, 101), (70, 45)])
def test_identity_math_path_equals_plain_attention(sch, sq, sk):
    g = sch.pack_factor
    q, k, v = rand_qkv(0, sq, sk, g=g)
    ref_out, ref_lse = dense_reference(q, k, v, 0.25, sch)
    for fmt in (None, "identity_fp8_e4m3", "identity_nvfp4"):
        res = tiled_attention(q, k, v, 0.25, sch, fmt)
        assert torch.allclose(res.out, ref_out, rtol=1e-9, atol=1e-11)
        finite = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(res.lse), finite)
        assert torch.allclose(res.lse[finite], ref_lse[finite], rtol=1e-10, atol=1e-10)


def test_identity_math_path_equals_attention_ref():
    sq, sk, d = 48, 80, 16
    q, k, v = rand_qkv(1, sq, sk, d=d)
    for causal in (False, True):
        sch = Schedule(16, 32, causal=causal, num_splits=2)
        res = tiled_attention(q, k, v, 1 / math.sqrt(d), sch)
        out_ref, _, lse_ref = attention_ref(
            q.permute(1, 0, 2)[None].transpose(1, 2).float(),
            k[None, :, None].float(),
            v[None, :, None].float(),
            causal=causal,
            return_lse=True,
        )
        assert torch.allclose(
            res.out[:, 0].float(), out_ref[0, :, 0], rtol=1e-4, atol=1e-5
        )
        assert torch.allclose(
            res.lse[:, 0].float(), lse_ref[0, 0], rtol=1e-4, atol=1e-5
        )


def test_empty_rows_give_zero_output_and_minus_inf_lse():
    # Causal with Sq > Sk: the first Sq - Sk queries see no key.
    q, k, v = rand_qkv(2, 40, 24)
    sch = Schedule(16, 32, causal=True)
    for fmt in (None, *FORMATS):
        for path in ("math", "impl"):
            res = tiled_attention(q, k, v, 0.3, sch, fmt, path=path)
            assert torch.equal(res.out[:16], torch.zeros_like(res.out[:16]))
            assert torch.isinf(res.lse[:16]).all() and (res.lse[:16] < 0).all()
            assert torch.isfinite(res.out).all() and torch.isfinite(res.lse[16:]).all()


# ------------------------------------------------------- implementation path


@pytest.mark.parametrize("sch", SCHEDULES[:6])
def test_impl_path_tracks_math_path_within_rounding_tolerance(sch):
    q, k, v = rand_qkv(3, 60, 90)
    scale = 0.25
    for ctl in ("identity_fp8_e4m3", "identity_nvfp4"):
        m = tiled_attention(q, k, v, scale, sch, ctl)
        i = tiled_attention(q, k, v, scale, sch, ctl, path="impl")
        # BF16 operand rounding (2**-9 relative per probability) and BF16 output rounding
        # (2**-9 relative), FP32 accumulation and FP32 scores are orders smaller.
        tol = 2.0**-8 * v.abs().max() + 2.0**-9 * m.out.abs()
        assert ((i.out - m.out).abs() <= tol).all()
        finite = torch.isfinite(m.lse)
        assert torch.allclose(i.lse[finite], m.lse[finite], rtol=0, atol=2e-5)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("sch", SCHEDULES[:6])
def test_impl_path_with_codec_within_tolerance_plus_allowance(fmt, sch):
    q, k, v = rand_qkv(4, 60, 90)
    scale = 0.25
    m = tiled_attention(q, k, v, scale, sch, fmt, with_allowance=True)
    i = tiled_attention(q, k, v, scale, sch, fmt, path="impl")
    tol = 2.0**-8 * v.abs().max() + 2.0**-9 * m.out.abs() + m.out_allowance
    assert ((i.out - m.out).abs() <= tol).all()


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("path", ["math", "impl"])
def test_codec_leaves_lse_identical_to_its_identity_control(fmt, path):
    q, k, v = rand_qkv(5, 50, 77)
    for sch in SCHEDULES[:6]:
        res = tiled_attention(q, k, v, 0.25, sch, fmt, path=path)
        ctl = tiled_attention(q, k, v, 0.25, sch, f"identity_{fmt}", path=path)
        assert torch.equal(res.lse, ctl.lse)  # bit-identical
        assert not torch.equal(res.out, ctl.out)


@pytest.mark.parametrize("fmt", FORMATS)
def test_oracle_that_sums_quantized_p_fails_the_lse_check(fmt):
    q, k, v = rand_qkv(5, 50, 77)
    sch = SCHEDULES[1]
    ctl = tiled_attention(q, k, v, 0.25, sch, f"identity_{fmt}")
    wrong = tiled_attention(q, k, v, 0.25, sch, fmt, quantized_row_sum=True)
    assert not torch.equal(wrong.lse, ctl.lse)
    assert (wrong.lse - ctl.lse).abs().max() > 1e-4


# ------------------------------------------------------------------ witnesses


def rising_max_witness(sk=96, d=8, dv=8):
    """One query whose scores are scattered over the keys.

    Tiles are visited last to first, so the row max rises several times and most
    tiles are quantized against a maximum smaller than the final one; a split
    restarts the maximum. Probabilities are off the code grids (powers of 0.83).
    """
    q = torch.zeros(1, 1, d, dtype=torch.float64)
    q[0, 0, 0] = 1.0
    k = torch.zeros(sk, d, dtype=torch.float64)
    # Exponents scattered over 8..47, then two planted maxima: the row max on key 5 (the
    # tile visited last) and a smaller maximum on key 70 (the tile visited first). Tiles in
    # between are quantized against 0.83**3 when visited in one pass, but against their own
    # maximum when a split restarts there; the textbook quantizes everything against 1.
    order = 8 + ((torch.arange(sk) * 37) % sk) % 40
    order[5], order[70] = 0, 3
    k[:, 0] = math.log(0.83) * order.double()  # p_j = 0.83**order[j]
    gen = torch.Generator().manual_seed(11)
    v = torch.randn(sk, dv, generator=gen).double()
    return q, k, v


@pytest.mark.parametrize("fmt", FORMATS)
def test_witness_output_depends_on_the_schedule(fmt):
    q, k, v = rising_max_witness()
    outs = {}
    for name, sch in {
        "tile16": Schedule(16, 64),
        "tile32": Schedule(32, 64),
        "tile32_split2": Schedule(32, 64, num_splits=2),
        "one_tile": Schedule(128, 64),
    }.items():
        outs[name] = tiled_attention(q, k, v, 1.0, sch, fmt).out
    names = list(outs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert (outs[a] - outs[b]).abs().max() > WITNESS_MIN_EFFECT[fmt], (a, b)
    # Without a codec the schedule does not matter.
    ctl = [
        tiled_attention(q, k, v, 1.0, s, f"identity_{fmt}").out
        for s in (Schedule(16, 64), Schedule(32, 64, num_splits=2))
    ]
    assert torch.allclose(ctl[0], ctl[1], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("fmt", FORMATS)
def test_textbook_oracle_does_not_match_the_tiled_oracle_on_the_witness(fmt):
    q, k, v = rising_max_witness()
    sch = Schedule(16, 64)
    tiled = tiled_attention(q, k, v, 1.0, sch, fmt).out
    textbook = textbook_attention(q, k, v, 1.0, sch, fmt).out
    assert (tiled - textbook).abs().max() > WITNESS_MIN_EFFECT[fmt]


def test_attention_ref_intermediate_dtype_is_not_a_kernel_reference():
    # The repository's textbook FP8 reference quantizes the normalized row.
    q, k, v = rising_max_witness()
    tiled = tiled_attention(q, k, v, 1.0, Schedule(16, 64), "fp8_e4m3").out
    out_ref, _ = attention_ref(
        q.float()[None],
        k.float()[None, :, None],
        v.float()[None, :, None],
        intermediate_dtype=torch.float8_e4m3fn,
    )
    assert (tiled[:, 0].float() - out_ref[0, :, 0]).abs().max() > 1e-3


def test_masked_keys_with_large_values_do_not_change_the_output():
    q, k, v = rand_qkv(6, 30, 64)
    sch = Schedule(16, 32, causal=True)
    for fmt in FORMATS:
        base = tiled_attention(q, k, v, 0.3, sch, fmt)
        # Keys no row can see (beyond the last row's causal bound) carry huge values.
        v2 = v.clone()
        v2[key_visible(sch, 30, 64, 29).logical_not()] = 1e6
        poisoned = tiled_attention(q, k, v2, 0.3, sch, fmt)
        assert torch.equal(poisoned.out, base.out)


# -------------------------------------------------------- allowance / flips


def scale_boundary_witness():
    """One row, two NVFP4 blocks in one tile.

    Block 0 holds the row max (p = 1). Block 1's maximum sits on the tie between
    the E4M3 scale codes 416 and 448 (448 * b = 432), and its other keys sit in
    the middle of E2M1 cells, far from their own code boundaries.
    """
    d = 4
    p = torch.full((2 * NVFP4_BLOCK,), 0.0, dtype=torch.float64)
    p[0] = 1.0
    p[1:NVFP4_BLOCK] = 0.3
    b = 432.0 / 448.0
    p[NVFP4_BLOCK] = b
    # Codes relative to the block scale: p / (sigma / 2688); with sigma ~ 432 the
    # effective scale is ~0.1607. Pick values near code centres 1, 2, 3, 4 -> not on ties.
    centres = torch.tensor([1.02, 2.05, 3.05, 4.1], dtype=torch.float64) * (
        432.0 / 2688.0
    )
    p[NVFP4_BLOCK + 1 :] = centres.repeat(4)[: NVFP4_BLOCK - 1]
    q = torch.zeros(1, 1, d, dtype=torch.float64)
    q[0, 0, 0] = 1.0
    k = torch.zeros(2 * NVFP4_BLOCK, d, dtype=torch.float64)
    k[:, 0] = torch.log(p)
    v = torch.ones(2 * NVFP4_BLOCK, 1, dtype=torch.float64)
    return q, k, v


def test_allowance_flags_every_key_of_a_block_whose_scale_is_on_a_boundary():
    q, k, v = scale_boundary_witness()
    sch = Schedule(32, 64)
    res = tiled_attention(q, k, v, 1.0, sch, "nvfp4", with_allowance=True)
    flagged = res.eligible_keys[0, 0]
    assert flagged[NVFP4_BLOCK:].all()  # all 16 keys of the boundary block
    assert not flagged[:NVFP4_BLOCK].any()  # the other block is untouched
    # The allowance covers the output change between the two reachable scale codes.
    up = k.clone()
    up[NVFP4_BLOCK, 0] += 1e-6
    down = k.clone()
    down[NVFP4_BLOCK, 0] -= 1e-6
    o_up = tiled_attention(q, up, v, 1.0, sch, "nvfp4").out
    o_down = tiled_attention(q, down, v, 1.0, sch, "nvfp4").out
    spread = (o_up - o_down).abs().max()
    assert spread > 1e-3  # the scale flip is a real, block-wide effect
    assert res.out_allowance.max() >= spread * 0.99


def test_allowance_that_ignores_neighbouring_keys_fails_the_scale_boundary_witness():
    q, k, v = scale_boundary_witness()
    sch = Schedule(32, 64)
    wrong = tiled_attention(
        q, k, v, 1.0, sch, "nvfp4", with_allowance=True, block_aware=False
    )
    flagged = wrong.eligible_keys[0, 0]
    assert not flagged[NVFP4_BLOCK:].all()  # neighbours of the block max are missed
    up, down = k.clone(), k.clone()
    up[NVFP4_BLOCK, 0] += 1e-6
    down[NVFP4_BLOCK, 0] -= 1e-6
    spread = (
        (
            tiled_attention(q, up, v, 1.0, sch, "nvfp4").out
            - tiled_attention(q, down, v, 1.0, sch, "nvfp4").out
        )
        .abs()
        .max()
    )
    assert (
        wrong.out_allowance.max() < spread * 0.5
    )  # the allowance does not cover the flip


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("emu_freq", [0, 16])
def test_random_inputs_stay_within_the_contract_budgets(fmt, emu_freq):
    d = 64
    q, k, v = rand_qkv(7, 96, 384, d=d)
    sch = Schedule(128, 128, ex2_emu_freq=emu_freq)
    res = tiled_attention(
        q, k, v, 1 / math.sqrt(d), sch, fmt, path="impl", with_allowance=True
    )
    pa = res.prob_allowance
    assert pa.max() <= 2.0**-3  # no row's check becomes vacuous
    assert (pa > 2.0**-5).double().mean() <= 0.05
    assert (
        pa > 2.0**-8
    ).double().mean() <= 0.30  # most rows are checked at BF16 precision
    # The output allowance is the probability allowance weighted by |V|.
    assert (res.out_allowance <= pa.unsqueeze(-1) * v.abs().max() + 1e-15).all()


def test_exp2_emulation_columns_and_accuracy():
    from p_quant_attention_oracle import emulated_exp2_mask, exp2_emulated

    # freq 16, res 4: pairs 12..15 of each 16 are emulated, except in the last fragment
    # and before start_frg.
    mask = emulated_exp2_mask(
        Schedule(128, 128, ex2_emu_freq=16, ex2_emu_start_frg=1), 128
    )
    assert mask.sum() == 2 * 8  # fragments 1 and 2, 8 columns each
    assert not mask[:32].any() and not mask[96:].any()
    assert mask[32 + 12 : 32 + 16].all() and mask[32 + 28 : 64].all()
    assert not emulated_exp2_mask(Schedule(128, 128), 128).any()
    z = torch.linspace(-30, 8, 4001, dtype=torch.float64).float().double()
    rel = (exp2_emulated(z) / torch.exp2(z) - 1).abs().max()
    assert rel < 9e-5  # the bound the contract quotes for the polynomial
    assert (
        exp2_emulated(torch.tensor([float("-inf"), -200.0], dtype=torch.float64))
        .eq(0)
        .all()
    )


# -------------------------------------------------------------- chunk merging


def test_merging_chunks_by_lse_reproduces_whole_sequence_attention_without_codec():
    q, k, v = rand_qkv(8, 20, 96)
    sch = Schedule(16, 32)
    whole = tiled_attention(q, k, v, 0.2, sch)
    parts = [
        tiled_attention(q, k[s], v[s], 0.2, sch) for s in (slice(0, 40), slice(40, 96))
    ]
    out, lse = merge_partials([p.out for p in parts], [p.lse for p in parts])
    assert torch.allclose(out, whole.out, rtol=1e-10, atol=1e-12)
    assert torch.allclose(lse, whole.lse, rtol=1e-12, atol=1e-12)
    # With a codec, chunking changes the result (each chunk has its own running max).
    wq = tiled_attention(q, k, v, 0.2, sch, "nvfp4")
    pq = [
        tiled_attention(q, k[s], v[s], 0.2, sch, "nvfp4")
        for s in (slice(0, 40), slice(40, 96))
    ]
    out_q, lse_q = merge_partials([p.out for p in pq], [p.lse for p in pq])
    assert torch.allclose(lse_q, whole.lse, rtol=1e-12, atol=1e-12)
    assert (out_q - wq.out).abs().max() > 1e-4


# ------------------------------------------- running-max uncertainty across tiles


def cancellation_witness():
    """Two 16-key tiles. The tile visited first holds the row max, 3.0, computed as
    65536 - 65536 + 3: exact in real arithmetic but with a large accumulation envelope.
    The tile visited second has small, well-conditioned scores below it, so the running
    max stays 3.0 and carries the first tile's uncertainty into the second tile."""
    d = 4
    q = torch.zeros(1, 1, d, dtype=torch.float64)
    q[0, 0, :3] = 1.0
    k = torch.zeros(32, d, dtype=torch.float64)
    k[16:, 2] = -200.0  # rest of the first-visited tile: negligible
    k[16] = torch.tensor([65536.0, -65536.0, 3.0, 0.0])
    k[:16, 2] = (
        2.75 - torch.arange(16, dtype=torch.float64) * 0.21
    )  # second-visited tile
    v = torch.ones(32, 1, dtype=torch.float64)
    return q, k, v


@pytest.mark.parametrize("fmt", FORMATS)
def test_running_max_uncertainty_is_carried_into_later_tiles(fmt):
    q, k, v = cancellation_witness()
    sch = Schedule(16, 64)
    good = tiled_attention(q, k, v, 1.0, sch, fmt, with_allowance=True)
    # The max is uncertain by ~0.06 in score units (about 6% in probability), so most keys of
    # the second tile can flip a code or the block scale.
    assert good.eligible_keys[0, 0, :16].sum() >= 8
    assert good.prob_allowance.max() > 1e-3
    # With one tile there is no "later tile": the same keys are flagged as well.
    one_tile = tiled_attention(q, k, v, 1.0, Schedule(32, 64), fmt, with_allowance=True)
    assert one_tile.eligible_keys[0, 0, :16].sum() >= 8


@pytest.mark.parametrize("fmt", FORMATS)
def test_oracle_that_forgets_the_previous_maximums_interval_misses_the_flips(fmt):
    q, k, v = cancellation_witness()
    sch = Schedule(16, 64)
    good = tiled_attention(q, k, v, 1.0, sch, fmt, with_allowance=True)
    wrong = tiled_attention(
        q, k, v, 1.0, sch, fmt, with_allowance=True, forget_running_max=True
    )
    missed = good.eligible_keys[0, 0, :16] & ~wrong.eligible_keys[0, 0, :16]
    assert missed.sum() >= 6  # flips the forgetful envelope does not see
    # Look only at the second-visited tile: V is zero on the first one, so output allowances
    # and output changes come from keys 0..15.
    v2 = v.clone()
    v2[16:] = 0.0
    good2 = tiled_attention(q, k, v2, 1.0, sch, fmt, with_allowance=True)
    wrong2 = tiled_attention(
        q, k, v2, 1.0, sch, fmt, with_allowance=True, forget_running_max=True
    )
    assert wrong2.out_allowance.max() < 0.25 * good2.out_allowance.max()
    # Moving the max inside its interval really changes those keys' codes: the effect exceeds
    # the forgetful allowance and stays within the correct one.
    k_shift = k.clone()
    k_shift[16, 2] += 0.05  # inside the 0.0625 envelope of that score
    moved = tiled_attention(q, k_shift, v2, 1.0, sch, fmt)
    # Undo the smooth change of the row sum; what is left is codec decisions that flipped.
    rescaled = moved.out * torch.exp(moved.lse - good2.lse).unsqueeze(-1)
    flip_effect = (rescaled - good2.out).abs().max()
    assert flip_effect > wrong2.out_allowance.max()
