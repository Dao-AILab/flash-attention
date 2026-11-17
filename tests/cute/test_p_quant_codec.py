"""P quant-dequant format table and CPU codec oracle (no GPU, no kernel launch)."""

import itertools

import pytest
import torch

from flash_attn.cute.p_quant import (
    E2M1_MAX,
    E4M3_MAX,
    NVFP4_BLOCK,
    NVFP4_GLOBAL_SCALE,
    PQuantFormat,
    p_quant_format_names,
    register_p_quant_format,
    resolve_p_quant,
    unregister_p_quant_format,
)
from p_quant_oracle import (
    E2M1_MAGNITUDES,
    e2m1_round,
    e4m3_round,
    nvfp4_block_scale,
    qdq,
    qdq_e4m3,
    qdq_nvfp4,
)


def f32(values):
    return torch.tensor(values, dtype=torch.float32)


def e4m3_values():
    """Every non-negative finite E4M3 value, from the bit layout."""
    vals = {0.0}
    for exp_field, mant in itertools.product(range(16), range(8)):
        if exp_field == 0:
            vals.add(mant * 2.0**-9)
        else:
            vals.add((1 + mant / 8) * 2.0 ** (exp_field - 7))
    vals = sorted(v for v in vals if v <= E4M3_MAX)  # 0x7F (480) is NaN in e4m3fn
    return vals


E4M3_VALUES = e4m3_values()


def ties_to_even_expected(grid):
    """(midpoint, expected) pairs: on a minifloat grid, RNE picks the even mantissa."""
    out = []
    for i, (lo, hi) in enumerate(zip(grid[:-1], grid[1:])):
        out.append(((lo + hi) / 2, lo if i % 2 == 0 else hi))
    return out


# ---------------------------------------------------------------- format table


def test_builtin_formats_resolve_with_distinct_policies():
    fp8, nvfp4 = resolve_p_quant("fp8_e4m3"), resolve_p_quant("nvfp4")
    assert (fp8.codec, fp8.block, fp8.max_offset, fp8.rescale_threshold) == (
        "e4m3",
        1,
        8,
        0.0,
    )
    assert (nvfp4.codec, nvfp4.block, nvfp4.max_offset, nvfp4.rescale_threshold) == (
        "nvfp4",
        NVFP4_BLOCK,
        0,
        0.0,
    )
    assert 2**fp8.max_offset == 256 and 2**nvfp4.max_offset == 1
    assert resolve_p_quant(None) is None and resolve_p_quant("none") is None
    assert resolve_p_quant(fp8) is fp8


def test_identity_controls_share_the_policy_of_their_format():
    for name in ("fp8_e4m3", "nvfp4"):
        fmt, ctl = resolve_p_quant(name), resolve_p_quant(f"identity_{name}")
        assert ctl.codec == "identity" and ctl.diagnostic and not fmt.diagnostic
        assert (ctl.max_offset, ctl.rescale_threshold, ctl.block) == (
            fmt.max_offset,
            fmt.rescale_threshold,
            fmt.block,
        )
    assert p_quant_format_names() == ["fp8_e4m3", "nvfp4"]
    assert "identity_nvfp4" in p_quant_format_names(include_diagnostic=True)


@pytest.mark.parametrize("bad", ["int4", "FP8_E4M3", "", 8, "fp8"])
def test_unknown_format_is_rejected_listing_supported(bad):
    with pytest.raises(ValueError) as exc:
        resolve_p_quant(bad)
    assert "fp8_e4m3" in str(exc.value) and "nvfp4" in str(exc.value)


def test_format_beyond_its_input_range_is_rejected():
    # 2**9 exceeds E4M3's 448.
    with pytest.raises(ValueError, match="input range"):
        PQuantFormat("fp8_too_big", "e4m3", 1, 9, 0.0, E4M3_MAX)
    # A lagging row max (threshold 8) lets the scaled probability reach 2**16.
    with pytest.raises(ValueError, match="input range"):
        PQuantFormat("fp8_lagging", "e4m3", 1, 8, 8.0, E4M3_MAX)
    with pytest.raises(ValueError, match="input range"):
        PQuantFormat("nvfp4_lagging", "nvfp4", NVFP4_BLOCK, 0, 8.0, 1.0)
    with pytest.raises(ValueError, match="integer"):
        PQuantFormat("fp8_448", "e4m3", 1, 8.807, 0.0, E4M3_MAX)
    with pytest.raises(ValueError, match="codec"):
        PQuantFormat("int8", "int8", 1, 0, 0.0, 1.0)


def test_registering_a_format_is_one_table_entry():
    fmt = PQuantFormat("fp8_e4m3_x128", "e4m3", 1, 7, 0.0, E4M3_MAX)
    try:
        register_p_quant_format(fmt)
        assert resolve_p_quant("fp8_e4m3_x128") is fmt
        x = f32([0.3, 100.3, 128.0])
        assert torch.equal(qdq(x, "fp8_e4m3_x128"), qdq_e4m3(x))
        with pytest.raises(ValueError, match="already taken"):
            register_p_quant_format(fmt)
    finally:
        unregister_p_quant_format("fp8_e4m3_x128")
    with pytest.raises(ValueError):
        resolve_p_quant("fp8_e4m3_x128")
    with pytest.raises(ValueError, match="built-in"):
        unregister_p_quant_format("nvfp4")


# ----------------------------------------------------------------------- E2M1


def test_e2m1_magnitudes_round_trip():
    x = f32(E2M1_MAGNITUDES)
    assert torch.equal(e2m1_round(x), x)
    assert torch.equal(e2m1_round(-x), -x)


def test_e2m1_midpoints_round_to_even():
    mids, expected = zip(*ties_to_even_expected(E2M1_MAGNITUDES))
    # 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4
    assert expected == (0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0)
    assert torch.equal(e2m1_round(f32(mids)), f32(expected))


def test_e2m1_neighbours_of_midpoints_round_to_nearest():
    for mid, _ in ties_to_even_expected(E2M1_MAGNITUDES):
        lo = max(v for v in E2M1_MAGNITUDES if v < mid)
        hi = min(v for v in E2M1_MAGNITUDES if v > mid)
        below = torch.nextafter(f32([mid]), f32([0.0]))
        above = torch.nextafter(f32([mid]), f32([10.0]))
        assert e2m1_round(below).item() == lo and e2m1_round(above).item() == hi


def test_e2m1_saturates_above_six():
    x = f32([6.0, 6.0000005, 7.0, 448.0, 1e30])
    assert torch.equal(e2m1_round(x), torch.full_like(x, E2M1_MAX))
    assert torch.equal(e2m1_round(-x), torch.full_like(x, -E2M1_MAX))


# ----------------------------------------------------------------------- E4M3


def test_e4m3_values_round_trip_and_match_torch_cast():
    x = f32(E4M3_VALUES)
    assert len(E4M3_VALUES) == 127  # 0x00 .. 0x7E
    assert torch.equal(e4m3_round(x), x)
    assert torch.equal(x.to(torch.float8_e4m3fn).to(torch.float32), x)


def test_e4m3_midpoints_round_to_even():
    mids, expected = zip(*ties_to_even_expected(E4M3_VALUES))
    assert torch.equal(e4m3_round(f32(mids)), f32(expected))


def test_e4m3_subnormals_zero_and_floor():
    step = 2.0**-9
    assert e4m3_round(f32([step])).item() == step
    # Half the smallest subnormal ties to zero; anything above it rounds up.
    assert e4m3_round(f32([step / 2])).item() == 0.0
    assert e4m3_round(torch.nextafter(f32([step / 2]), f32([1.0]))).item() == step
    assert e4m3_round(f32([1.5 * step])).item() == 2 * step  # tie to even mantissa
    z = e4m3_round(f32([0.0, -0.0]))
    assert z[0].item() == 0.0 and torch.signbit(z[1]).item()


def test_e4m3_saturates_at_448():
    x = f32([448.0, 448.0001, 464.0, 480.0, 1e6, float("inf")])
    assert torch.equal(e4m3_round(x), torch.full_like(x, E4M3_MAX))


def test_e4m3_matches_torch_cast_on_random_in_range_inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.exp2(torch.empty(20000).uniform_(-12, 8.8, generator=g)).float()
    x = x[x <= E4M3_MAX]
    ours = e4m3_round(x)
    ref = x.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(ours, ref)


def test_overshoot_by_a_rounding_residual_saturates():
    # FP8 policy: ideal maximum 256; one ulp above still gives 256.
    over = torch.nextafter(f32([256.0]), f32([1e9]))
    assert qdq_e4m3(over).item() == 256.0
    # NVFP4 policy: ideal maximum 1; one ulp above still gives scale 448 and code 6.
    x = torch.zeros(NVFP4_BLOCK)
    x[3] = torch.nextafter(f32([1.0]), f32([2.0])).item()
    assert nvfp4_block_scale(x).item() == E4M3_MAX
    assert qdq_nvfp4(x)[3].item() == 1.0


# ---------------------------------------------------------------------- NVFP4


def test_nvfp4_block_of_zeros_is_zeros():
    x = torch.zeros(2, 3 * NVFP4_BLOCK)
    out = qdq_nvfp4(x)
    assert torch.equal(out, x) and torch.isfinite(out).all()


def test_nvfp4_zero_rounded_scale_gives_zeros():
    # 448 * b below half of E4M3's smallest subnormal rounds the scale to zero.
    tiny = (2.0**-10) / 448.0 * 0.99
    x = torch.full((NVFP4_BLOCK,), tiny)
    assert nvfp4_block_scale(x).item() == 0.0
    out = qdq_nvfp4(x)
    assert torch.equal(out, torch.zeros_like(x)) and torch.isfinite(out).all()
    # Just above the tie the scale is the smallest subnormal and the result is finite.
    x2 = torch.full((NVFP4_BLOCK,), (2.0**-10) / 448.0 * 1.01)
    assert nvfp4_block_scale(x2).item() == 2.0**-9
    assert torch.isfinite(qdq_nvfp4(x2)).all()


def test_nvfp4_partial_block_equals_block_padded_with_zeros():
    g = torch.Generator().manual_seed(1)
    valid = torch.rand(11, generator=g)
    padded = torch.cat([valid, torch.zeros(NVFP4_BLOCK - 11)])
    out = qdq_nvfp4(padded)
    assert torch.equal(out[11:], torch.zeros(NVFP4_BLOCK - 11))
    # Masked keys are zero and never raise the block max.
    assert nvfp4_block_scale(padded).item() == e4m3_round(valid.max() * 448.0).item()


def test_nvfp4_values_are_code_times_effective_scale():
    g = torch.Generator().manual_seed(2)
    x = torch.rand(4, 8 * NVFP4_BLOCK, generator=g)
    x[:, 0] = 1.0  # row max present
    out = qdq_nvfp4(x)
    sigma = nvfp4_block_scale(x).repeat_interleave(NVFP4_BLOCK, dim=-1)
    eff = sigma / NVFP4_GLOBAL_SCALE
    code = out / eff
    grid = f32(E2M1_MAGNITUDES)
    assert (code.unsqueeze(-1) - grid).abs().amin(dim=-1).max().item() < 1e-6
    # The block max always takes the top code, so each block's largest output is 6 * eff.
    blocks = out.reshape(4, -1, NVFP4_BLOCK)
    assert torch.allclose(blocks.amax(-1), 6 * eff.reshape(4, -1, NVFP4_BLOCK)[..., 0])
    # Quantization error is bounded by half the coarsest E2M1 step times the scale.
    assert ((out - x).abs() <= eff + 1e-7).all()


def test_nvfp4_blocks_are_independent_and_aligned_from_the_first_key():
    g = torch.Generator().manual_seed(3)
    x = torch.rand(3 * NVFP4_BLOCK, generator=g)
    whole = qdq_nvfp4(x)
    for b in range(3):
        sl = slice(b * NVFP4_BLOCK, (b + 1) * NVFP4_BLOCK)
        assert torch.equal(whole[sl], qdq_nvfp4(x[sl]))


def test_identity_codec_and_none_leave_inputs_unchanged():
    x = torch.rand(2 * NVFP4_BLOCK)
    for name in (None, "none", "identity_fp8_e4m3", "identity_nvfp4"):
        assert qdq(x, name) is x
    assert not torch.equal(qdq(x, "nvfp4"), x)
    assert not torch.equal(qdq(x * 256, "fp8_e4m3"), x * 256)


# ------------------------------------------------- wrong tie rules are detected


@pytest.mark.parametrize("rounding", ["half_away", "trunc"])
def test_wrong_rounding_rule_fails_the_tie_tests(rounding):
    mids, expected = zip(*ties_to_even_expected(E2M1_MAGNITUDES))
    assert not torch.equal(e2m1_round(f32(mids), rounding), f32(expected))
    mids, expected = zip(*ties_to_even_expected(E4M3_VALUES))
    assert not torch.equal(e4m3_round(f32(mids), rounding), f32(expected))
    # ... while the default rule passes the same checks.
    assert torch.equal(e4m3_round(f32(mids)), f32(expected))


def test_wrong_rounding_rule_changes_the_block_codec():
    x = torch.zeros(NVFP4_BLOCK)
    x[0] = 1.0  # scale 448, effective scale 1/6
    x[1] = 1.25 / 6  # exactly on the 1 / 1.5 code tie
    x[2] = 0.25 / 6  # exactly on the 0 / 0.5 code tie
    rne = qdq_nvfp4(x)
    assert rne[1].item() == pytest.approx(1.0 / 6) and rne[2].item() == 0.0
    assert not torch.equal(qdq_nvfp4(x, "half_away"), rne)


# ------------------------------------------------ registration and validation


def test_format_objects_must_be_the_registered_entry():
    fp8 = resolve_p_quant("fp8_e4m3")
    assert resolve_p_quant(fp8) is fp8
    # A format object that is not in the table cannot reach a kernel.
    with pytest.raises(ValueError, match="unknown p_quant"):
        resolve_p_quant(PQuantFormat("not_in_table", "e4m3", 1, 8, 0.0, E4M3_MAX))
    # Nor can an altered recipe that reuses a registered name.
    with pytest.raises(ValueError, match="does not match"):
        resolve_p_quant(PQuantFormat("fp8_e4m3", "e4m3", 1, 7, 0.0, E4M3_MAX))
    with pytest.raises(ValueError, match="does not match"):
        qdq(f32([1.0]), PQuantFormat("nvfp4", "identity", NVFP4_BLOCK, 0, 0.0, 1.0))


def test_a_new_codec_is_a_name_a_helper_and_a_table_entry():
    """An E5M2 codec added through the registration points only."""
    import p_quant_oracle
    from flash_attn.cute import p_quant

    def qdq_e5m2(x, rounding="rne"):
        return x.clamp(max=57344.0).to(torch.float8_e5m2).to(torch.float32)

    with pytest.raises(ValueError, match="codec"):
        PQuantFormat(
            "fp8_e5m2_test", "e5m2", 1, 8, 0.0, 57344.0
        )  # codec not registered yet
    try:
        p_quant.register_p_quant_codec("e5m2")
        p_quant_oracle.ORACLE_CODECS["e5m2"] = qdq_e5m2
        fmt = register_p_quant_format(
            PQuantFormat("fp8_e5m2_test", "e5m2", 1, 8, 0.0, 57344.0)
        )
        assert resolve_p_quant("fp8_e5m2_test") is fmt
        x = f32([0.3, 100.3, 255.9, 256.0])
        got = qdq(x, "fp8_e5m2_test")
        assert torch.equal(got, qdq_e5m2(x))
        assert not torch.equal(got, qdq(x, "fp8_e4m3"))  # 2 mantissa bits against 3
    finally:
        unregister_p_quant_format("fp8_e5m2_test")
        p_quant._CODECS.discard("e5m2")
        p_quant_oracle.ORACLE_CODECS.pop("e5m2", None)
    with pytest.raises(ValueError):
        resolve_p_quant("fp8_e5m2_test")
