from dataclasses import asdict, replace
from unittest import mock

import pytest

from flash_attn.cute import config as config_module
from flash_attn.cute.config import (
    FwdConfig,
    FwdHeuristicInputs,
    TunedSm100Overrides,
    num_splits_heuristic,
    select_fwd_config,
    validate_fwd_config,
)

_BASE_INPUTS = FwdHeuristicInputs(
    device_arch=100,
    num_sms=132,
    dtype="torch.bfloat16",
    head_dim=64,
    head_dim_v=64,
    num_heads=16,
    num_heads_kv=16,
    batch_size=2,
    total_q=8192,
    total_k=8192,
    max_seqlen_q=4096,
    max_seqlen_k=4096,
    requested_num_splits=1,
)


def make_inputs(**changes) -> FwdHeuristicInputs:
    return _BASE_INPUTS._replace(**changes)


@pytest.mark.parametrize("device_arch", [80, 90, 100, 103, 110, 120])
def test_every_architecture_selects_a_valid_default(device_arch):
    select_fwd_config(make_inputs(device_arch=device_arch))


def test_2cta_boundary_follows_packed_q_tiles():
    one_cta = make_inputs(dtype="torch.float16", head_dim=128, head_dim_v=128, max_seqlen_q=256)

    assert not select_fwd_config(one_cta).use_2cta_instrs
    assert select_fwd_config(one_cta._replace(max_seqlen_q=257)).use_2cta_instrs


def test_explicit_2cta_is_valid_when_the_environment_disables_the_default():
    inputs = make_inputs(head_dim=128, head_dim_v=128, disable_2cta=True)
    config = select_fwd_config(inputs)

    assert not config.use_2cta_instrs
    validate_fwd_config(replace(config, use_2cta_instrs=True), inputs)


def test_varlen_q_disables_tma_output_and_static_persistence():
    inputs = make_inputs(has_cu_seqlens_q=True, batch_size=4, max_seqlen_q=512)
    config = select_fwd_config(inputs)

    assert not config.is_static_persistent
    assert not config.use_tma_o
    with pytest.raises(ValueError, match="TMA O"):
        validate_fwd_config(replace(config, use_tma_o=True), inputs)


def test_clc_excludes_static_persistence():
    inputs = make_inputs(
        head_dim=128,
        head_dim_v=128,
        num_heads_kv=4,
        pack_gqa=True,
        has_cu_seqlens_k=True,
        requested_use_clc_scheduler=True,
    )
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler
    assert not config.is_static_persistent
    with pytest.raises(ValueError, match="Static persistent scheduling"):
        validate_fwd_config(replace(config, is_static_persistent=True), inputs)


def test_explicit_split_count_is_kept_exactly():
    config = select_fwd_config(make_inputs(head_dim=128, head_dim_v=128, requested_num_splits=4))

    assert config.num_splits == 4
    assert not config.use_2cta_instrs


def test_diff_head_split_request_keeps_the_legacy_reselection():
    decode = make_inputs(
        head_dim=192,
        head_dim_v=128,
        num_heads=1,
        num_heads_kv=1,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=16384,
        requested_num_splits=4,
    )
    config = select_fwd_config(decode)

    assert (config.tile_n, config.num_splits) == (64, 128)
    validate_fwd_config(replace(config, num_splits=4), decode)
    with pytest.raises(ValueError, match="Diff-head SplitKV requires tile_n=64"):
        validate_fwd_config(replace(config, tile_n=128), decode)
    # Short KV or a split-incompatible value head falls back to one split.
    short = decode._replace(max_seqlen_k=1024)
    assert select_fwd_config(short).num_splits == 1
    assert select_fwd_config(short._replace(head_dim=128, head_dim_v=192)).num_splits == 1


def test_auto_split_resolves_from_shape_metadata():
    decode = make_inputs(
        num_heads=1,
        num_heads_kv=1,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=16384,
        requested_num_splits=None,
    )

    assert select_fwd_config(decode).num_splits == 128
    assert select_fwd_config(decode._replace(max_seqlen_k=8192)).num_splits == 64
    assert num_splits_heuristic(1024, 132, 128, 128) == 1


def test_seqlen_k_per_split_bounds_only_splitkv_configs():
    decode = make_inputs(
        num_heads=1,
        num_heads_kv=1,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=4224,
        seqlen_k_per_split=1024,
        requested_num_splits=None,
    )
    config = select_fwd_config(decode)

    # The fixed extent does not change selection; it only bounds split counts.
    assert config.num_splits == 33
    validate_fwd_config(replace(config, num_splits=1), decode)
    with pytest.raises(ValueError, match="requires num_splits >= 5"):
        validate_fwd_config(replace(config, num_splits=4), decode)
    with pytest.raises(ValueError, match="divisible by tile_n"):
        validate_fwd_config(replace(config, tile_n=96), decode._replace(seqlen_k_per_split=1000))


def test_invalid_geometry_and_features_are_rejected():
    config = select_fwd_config(_BASE_INPUTS)

    with pytest.raises(ValueError, match="multiples of 16"):
        validate_fwd_config(replace(config, tile_n=127), _BASE_INPUTS)
    with pytest.raises(ValueError, match="targets SM9"):
        validate_fwd_config(replace(config, device_capacity=9), _BASE_INPUTS)
    with pytest.raises(ValueError, match="Unsupported forward architecture family SM7"):
        select_fwd_config(make_inputs(device_arch=70))
    with pytest.raises(ValueError, match="SM120 forward does not support"):
        select_fwd_config(make_inputs(device_arch=120, page_size=128))
    with pytest.raises(ValueError, match="head dimensions divisible by 16"):
        select_fwd_config(make_inputs(head_dim=72, head_dim_v=72, page_size=64))
    with pytest.raises(AssertionError, match="SM120 forward only supports num_splits=1"):
        select_fwd_config(make_inputs(device_arch=120, requested_num_splits=3))


def test_sm90_flags_follow_tile_overrides_and_stay_sm90_only():
    sm90 = make_inputs(device_arch=90, head_dim=96, head_dim_v=96, requested_tile_m=128)
    config = select_fwd_config(sm90)

    assert (config.tile_m, config.mma_pv_is_rs, config.intra_wg_overlap) == (128, True, True)
    with pytest.raises(ValueError, match="SM90 tile_m must be"):
        validate_fwd_config(replace(config, tile_m=256), sm90)
    # Other architectures ignore SM90 flag requests but reject them in explicit configs.
    sm120 = make_inputs(device_arch=120, requested_tile_m=64, requested_mma_pv_is_rs=True)
    config = select_fwd_config(sm120)
    assert not config.mma_pv_is_rs
    with pytest.raises(ValueError, match="does not expose SM90 MMA flags"):
        validate_fwd_config(replace(config, mma_pv_is_rs=True), sm120)


def test_hd256_uses_the_generic_sm100_contract():
    inputs = make_inputs(head_dim=256, head_dim_v=256, causal=True)
    config = select_fwd_config(inputs)

    assert config.q_stage == 1
    assert config.use_2cta_instrs
    assert not config.use_clc_scheduler
    with pytest.raises(ValueError, match="TMEM"):
        validate_fwd_config(replace(config, q_stage=2), inputs)
    with pytest.raises(ValueError, match="CLC"):
        validate_fwd_config(replace(config, use_clc_scheduler=True), inputs)
    split = select_fwd_config(inputs._replace(requested_num_splits=2))
    assert (split.num_splits, split.use_tma_o, split.use_s_ping_pong) == (2, False, False)


def test_hd256_decode_uses_2cta_only_while_the_grid_fits_one_wave():
    decode = make_inputs(head_dim=256, head_dim_v=256, num_heads=8, num_heads_kv=8, max_seqlen_q=1)

    assert select_fwd_config(decode._replace(batch_size=8)).use_2cta_instrs
    assert not select_fwd_config(decode._replace(batch_size=16)).use_2cta_instrs


def test_decode_uses_s_ping_pong_only_without_score_modifiers():
    decode = make_inputs(head_dim=128, head_dim_v=128, max_seqlen_q=1)
    config = select_fwd_config(decode)

    assert config.q_stage == 1
    assert config.use_s_ping_pong
    assert not select_fwd_config(decode._replace(disable_s_ping_pong=True)).use_s_ping_pong
    masked = decode._replace(has_mask_mod=True)
    assert not select_fwd_config(masked).use_s_ping_pong
    with pytest.raises(ValueError, match="S ping-pong"):
        validate_fwd_config(config, masked)


_B200_LONG_PREFILL = make_inputs(
    num_heads=32,
    num_heads_kv=32,
    batch_size=1,
    total_q=257,
    total_k=2048,
    max_seqlen_q=257,
    max_seqlen_k=2048,
)


def select_untuned(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Return the config with the tuned overrides patched out."""
    with mock.patch.object(
        config_module, "select_tuned_sm100_overrides", return_value=TunedSm100Overrides()
    ):
        return select_fwd_config.__wrapped__(inputs)


def tuned_fields(inputs: FwdHeuristicInputs) -> set[str]:
    """Return the config fields a measured rule changed for this problem."""
    tuned, untuned = asdict(select_fwd_config(inputs)), asdict(select_untuned(inputs))
    return {name for name in tuned if tuned[name] != untuned[name]}


_TUNED_RULES = {
    # rule -> (inputs where it fires, config fields it must change)
    "b200_d64_direct_o": (_B200_LONG_PREFILL, {"use_tma_o"}),
    "b200_d128_1cta": (
        _B200_LONG_PREFILL._replace(head_dim=128, head_dim_v=128),
        {"use_2cta_instrs"},
    ),
    "sm103_d64_nonpersistent": (
        make_inputs(device_arch=103, max_seqlen_k=8192),
        {"is_static_persistent"},
    ),
    "sm103_balanced_varlen_mha_clc": (
        make_inputs(
            device_arch=103,
            batch_size=4,
            total_q=10240,
            max_seqlen_q=6400,
            has_cu_seqlens_q=True,
            has_cu_seqlens_k=True,
        ),
        {"use_clc_scheduler"},
    ),
    "sm103_high_head_varlen_clc": (
        make_inputs(
            device_arch=103,
            num_heads=28,
            num_heads_kv=4,
            pack_gqa=True,
            batch_size=3,
            total_q=4096,
            total_k=4096,
            max_seqlen_q=2048,
            max_seqlen_k=2048,
            causal=True,
            has_cu_seqlens_q=True,
            has_cu_seqlens_k=True,
        ),
        {"use_clc_scheduler"},
    ),
    "sm103_dense_short_k_clc": (
        make_inputs(
            device_arch=103,
            num_heads=32,
            num_heads_kv=8,
            pack_gqa=True,
            batch_size=32,
            total_q=32 * 64,
            total_k=32 * 1024,
            max_seqlen_q=64,
            max_seqlen_k=1024,
            causal=True,
        ),
        {"use_clc_scheduler"},
    ),
}

_NOT_PLAIN_BF16 = [
    {"dtype": "torch.float16"},
    {"page_size": 128},
    {"use_block_sparsity": True},
    {"has_score_mod": True},
    {"has_mask_mod": True},
    {"has_learnable_sink": True},
    {"has_lse": True},
    {"requested_tile_n": 64},
    {"requested_num_splits": 2},
    {"local": True, "window_size_left": 256, "window_size_right": 0},
]


@pytest.mark.parametrize("rule", _TUNED_RULES)
@pytest.mark.parametrize("changes", [{}, *_NOT_PLAIN_BF16])
def test_tuned_rules_fire_only_for_plain_bf16(rule, changes):
    inputs, fields = _TUNED_RULES[rule]

    changed = tuned_fields(inputs._replace(**changes))

    assert not changed if changes else fields <= changed


@pytest.mark.parametrize(
    ("rule", "changes", "fires"),
    [
        ("b200_d64_direct_o", {"max_seqlen_q": 256}, False),
        ("b200_d64_direct_o", {"max_seqlen_k": 2047}, False),
        ("b200_d64_direct_o", {"num_heads": 31, "num_heads_kv": 31}, False),
        ("b200_d64_direct_o", {"causal": True}, False),
        ("b200_d64_direct_o", {"num_heads": 64, "num_heads_kv": 16, "pack_gqa": True}, True),
        ("b200_d64_direct_o", {"num_heads": 64, "num_heads_kv": 32, "pack_gqa": True}, False),
        ("b200_d64_direct_o", {"num_heads_kv": 8, "pack_gqa": False}, False),
        ("b200_d64_direct_o", {"device_arch": 103}, False),
        ("b200_d128_1cta", {"max_seqlen_k": 2047}, False),
        ("b200_d128_1cta", {"num_heads": 31, "num_heads_kv": 31}, False),
        ("b200_d128_1cta", {"num_heads": 64, "num_heads_kv": 4, "pack_gqa": True}, True),
        ("sm103_d64_nonpersistent", {"max_seqlen_k": 4096}, True),
        ("sm103_d64_nonpersistent", {"max_seqlen_k": 3072}, False),
        ("sm103_d64_nonpersistent", {"head_dim": 96, "head_dim_v": 96}, False),
        ("sm103_d64_nonpersistent", {"num_heads_kv": 2}, False),
        ("sm103_d64_nonpersistent", {"device_arch": 110}, False),
        ("sm103_balanced_varlen_mha_clc", {"total_q": 10239}, False),
        ("sm103_balanced_varlen_mha_clc", {"max_seqlen_q": 6401}, False),
        ("sm103_balanced_varlen_mha_clc", {"total_k": 6553}, False),
        ("sm103_balanced_varlen_mha_clc", {"batch_size": 3}, False),
        ("sm103_balanced_varlen_mha_clc", {"batch_size": 25}, False),
        ("sm103_balanced_varlen_mha_clc", {"num_heads": 4, "num_heads_kv": 4}, False),
        ("sm103_balanced_varlen_mha_clc", {"has_cu_seqlens_k": False}, False),
        ("sm103_balanced_varlen_mha_clc", {"has_seqused_k": True}, False),
        ("sm103_balanced_varlen_mha_clc", {"has_caller_scheduler_metadata": True}, False),
        ("sm103_high_head_varlen_clc", {"causal": False}, True),
        ("sm103_high_head_varlen_clc", {"total_q": 4095}, False),
        ("sm103_high_head_varlen_clc", {"batch_size": 2}, False),
        ("sm103_high_head_varlen_clc", {"batch_size": 65}, False),
        ("sm103_high_head_varlen_clc", {"num_heads": 23, "num_heads_kv": 1}, False),
        ("sm103_high_head_varlen_clc", {"head_dim": 80, "head_dim_v": 80}, False),
        ("sm103_high_head_varlen_clc", {"pack_gqa": False}, False),
        ("sm103_dense_short_k_clc", {"max_seqlen_k": 640}, True),
        ("sm103_dense_short_k_clc", {"max_seqlen_k": 2048}, True),
        ("sm103_dense_short_k_clc", {"max_seqlen_k": 639}, False),
        ("sm103_dense_short_k_clc", {"max_seqlen_k": 2049}, False),
        ("sm103_dense_short_k_clc", {"batch_size": 31}, False),
        ("sm103_dense_short_k_clc", {"causal": False}, False),
        ("sm103_dense_short_k_clc", {"num_heads": 28, "num_heads_kv": 4}, False),
        ("sm103_dense_short_k_clc", {"num_heads": 71, "num_heads_kv": 1}, True),
        ("sm103_dense_short_k_clc", {"num_heads": 71, "num_heads_kv": 1, "max_seqlen_q": 1}, False),
        ("sm103_dense_short_k_clc", {"head_dim": 80, "head_dim_v": 80}, False),
        ("sm103_dense_short_k_clc", {"pack_gqa": False}, False),
    ],
)
def test_tuned_rule_boundaries(rule, changes, fires):
    inputs, fields = _TUNED_RULES[rule]

    changed = tuned_fields(inputs._replace(**changes))

    assert fields <= changed if fires else not changed
