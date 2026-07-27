import inspect
import json
from dataclasses import asdict, replace

import pytest

from flash_attn.cute.config import (
    FwdConfig,
    FwdHeuristicInputs,
    FwdSm90RegisterAllocation,
    FwdSm100RegisterAllocation,
    combine_log_max_splits,
    fwd_combine_tile,
    fwd_config_compile_bucket,
    num_splits_heuristic,
    select_fwd_config,
    select_sm100_register_allocation,
    validate_fwd_config,
)
from flash_attn.cute.interface import _flash_attn_fwd

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
    seqlen_k_per_split=None,
    causal=False,
    local=False,
    window_size_left=None,
    window_size_right=None,
    is_varlen_q=False,
    has_cu_seqlens_q=False,
    has_cu_seqlens_k=False,
    has_seqused=False,
    pack_gqa=False,
    page_size=None,
    use_block_sparsity=False,
    sparse_q_block_size=None,
    has_qv=False,
    has_gather_kv=False,
    has_score_mod=False,
    has_mask_mod=False,
    has_learnable_sink=False,
    has_lse=False,
    requested_tile_m=None,
    requested_tile_n=None,
    requested_mma_pv_is_rs=None,
    requested_intra_wg_overlap=None,
    requested_num_splits=1,
    requested_use_clc_scheduler=True,
    disable_2cta=False,
)


def make_inputs(**changes) -> FwdHeuristicInputs:
    return _BASE_INPUTS._replace(**changes)


def test_forward_config_is_optional_and_keyword_only():
    parameter = inspect.signature(_flash_attn_fwd).parameters["config"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"device_arch": 90, "requested_use_clc_scheduler": False},
    ],
)
def test_register_allocation_round_trips_through_json_and_yaml_shapes(changes):
    config = select_fwd_config(make_inputs(**changes))
    serialized = json.loads(json.dumps(asdict(config)))
    reconstructed = FwdConfig(**serialized)

    assert reconstructed == config
    assert hash(reconstructed) == hash(config)

    serialized["registers"] = config.registers._asdict()
    assert FwdConfig(**serialized) == config


@pytest.mark.parametrize(
    ("changes", "use_2cta_instrs", "expected"),
    [
        ({"head_dim": 64, "head_dim_v": 64}, False, (200, 64, 48)),
        ({"head_dim": 80, "head_dim_v": 80, "page_size": 16}, False, (184, 64, 80)),
        ({"head_dim": 96, "head_dim_v": 96}, False, (192, 80, 48)),
        ({"head_dim": 128, "head_dim_v": 128}, True, (176, 88, 72)),
        ({"head_dim": 128, "head_dim_v": 128, "causal": True}, False, (192, 72, 56)),
        ({"head_dim": 128, "head_dim_v": 128, "device_arch": 103}, True, (176, 80, 80)),
        (
            {"head_dim": 128, "head_dim_v": 128, "device_arch": 103, "causal": True},
            False,
            (176, 64, 96),
        ),
        ({"head_dim": 192, "head_dim_v": 128}, True, (184, 80, 64)),
        ({"head_dim": 192, "head_dim_v": 128, "device_arch": 103}, True, (176, 64, 96)),
        (
            {"dtype": "torch.float8_e4m3fn", "head_dim": 64, "head_dim_v": 64},
            False,
            (168, 96, 80),
        ),
        (
            {
                "dtype": "torch.float8_e5m2",
                "head_dim": 64,
                "head_dim_v": 64,
                "page_size": 16,
            },
            False,
            (152, 96, 112),
        ),
        (
            {"dtype": "torch.float8_e4m3fn", "head_dim": 128, "head_dim_v": 128},
            True,
            (160, 72, 120),
        ),
        (
            {
                "dtype": "torch.float8_e5m2",
                "head_dim": 128,
                "head_dim_v": 128,
                "causal": True,
            },
            False,
            (192, 72, 56),
        ),
    ],
)
def test_sm100_register_allocation_matches_existing_kernel_policy(
    changes, use_2cta_instrs, expected
):
    inputs = make_inputs(**changes)

    assert select_sm100_register_allocation(
        inputs, tile_n=128, use_2cta_instrs=use_2cta_instrs
    ) == FwdSm100RegisterAllocation(*expected)


@pytest.mark.parametrize(
    ("changes", "expected_static_persistent"),
    [
        ({}, False),
        ({"max_seqlen_k": 4096}, False),
        ({"max_seqlen_k": 3072}, True),
        ({"head_dim": 32, "head_dim_v": 32}, True),
        ({"head_dim": 96, "head_dim_v": 96}, True),
        ({"dtype": "torch.float16"}, True),
        ({"device_arch": 100}, True),
        ({"page_size": 128}, True),
        ({"use_block_sparsity": True}, True),
        ({"has_gather_kv": True}, True),
        ({"has_score_mod": True}, True),
        ({"has_mask_mod": True}, True),
        ({"has_learnable_sink": True}, True),
        ({"has_lse": True}, True),
        ({"requested_tile_n": 64}, True),
        ({"num_heads_kv": 2, "pack_gqa": False}, True),
    ],
)
def test_sm103_long_d64_nonpersistent_scheduler_scope(
    changes, expected_static_persistent
):
    inputs = make_inputs(device_arch=103, max_seqlen_k=8192)._replace(**changes)

    assert select_fwd_config(inputs).is_static_persistent is expected_static_persistent


@pytest.mark.parametrize(
    ("changes", "expected_clc"),
    [
        ({}, True),
        ({"total_q": 10239}, False),
        ({"max_seqlen_q": 6401}, False),
        ({"total_k": 6553}, False),
        ({"batch_size": 3}, False),
        ({"batch_size": 25}, False),
        ({"head_dim": 192, "head_dim_v": 192}, False),
        ({"num_heads": 4, "num_heads_kv": 4}, False),
        ({"dtype": "torch.float16"}, False),
        ({"device_arch": 100}, False),
        ({"is_varlen_q": False}, False),
        ({"has_cu_seqlens_q": False}, False),
        ({"has_cu_seqlens_k": False}, False),
        ({"has_seqused": True}, False),
        ({"num_heads_kv": 4}, False),
        ({"requested_num_splits": 2}, False),
        ({"page_size": 128}, False),
        ({"use_block_sparsity": True}, False),
        ({"has_gather_kv": True}, False),
        ({"has_score_mod": True}, False),
        ({"has_mask_mod": True}, False),
        ({"has_learnable_sink": True}, False),
        ({"has_lse": True}, False),
        ({"requested_tile_n": 64}, False),
    ],
)
def test_sm103_balanced_varlen_mha_clc_scope(changes, expected_clc):
    inputs = make_inputs(
        device_arch=103,
        batch_size=4,
        total_q=10240,
        max_seqlen_q=6400,
        is_varlen_q=True,
        has_cu_seqlens_q=True,
        has_cu_seqlens_k=True,
        requested_use_clc_scheduler=False,
    )._replace(**changes)
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler is expected_clc
    if expected_clc:
        assert not config.is_static_persistent


@pytest.mark.parametrize(
    ("changes", "expected_clc"),
    [
        ({}, True),
        ({"causal": False}, True),
        ({"total_q": 4095}, False),
        ({"batch_size": 2}, False),
        ({"batch_size": 65}, False),
        ({"num_heads": 23, "num_heads_kv": 1}, False),
        ({"head_dim": 80, "head_dim_v": 80}, False),
        ({"pack_gqa": False}, False),
        ({"has_score_mod": True}, False),
        ({"has_mask_mod": True}, False),
        ({"has_learnable_sink": True}, False),
        ({"has_lse": True}, False),
    ],
)
def test_sm103_high_head_varlen_clc_scope(changes, expected_clc):
    inputs = make_inputs(
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
        is_varlen_q=True,
        has_cu_seqlens_q=True,
        has_cu_seqlens_k=True,
        requested_use_clc_scheduler=False,
    )._replace(**changes)

    assert select_fwd_config(inputs).use_clc_scheduler is expected_clc


@pytest.mark.parametrize(
    ("changes", "expected_clc"),
    [
        ({}, True),
        ({"max_seqlen_k": 640}, True),
        ({"max_seqlen_k": 2048}, True),
        ({"max_seqlen_k": 639}, False),
        ({"max_seqlen_k": 2049}, False),
        ({"batch_size": 31}, False),
        ({"causal": False}, False),
        ({"num_heads": 23, "num_heads_kv": 1}, False),
        ({"num_heads": 28, "num_heads_kv": 4}, False),
        ({"num_heads": 71, "num_heads_kv": 1}, True),
        ({"num_heads": 71, "num_heads_kv": 1, "max_seqlen_q": 1}, False),
        ({"num_heads": 64, "num_heads_kv": 1, "max_seqlen_q": 1}, False),
        ({"head_dim": 80, "head_dim_v": 80}, False),
        ({"pack_gqa": False}, False),
        ({"requested_num_splits": 2}, False),
        ({"has_score_mod": True}, False),
        ({"has_mask_mod": True}, False),
        ({"has_learnable_sink": True}, False),
        ({"has_lse": True}, False),
    ],
)
def test_sm103_dense_short_k_clc_scope(changes, expected_clc):
    inputs = make_inputs(
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
        requested_use_clc_scheduler=False,
    )._replace(**changes)

    assert select_fwd_config(inputs).use_clc_scheduler is expected_clc


def test_selector_preserves_a_codegen_changing_2cta_boundary():
    one_cta = make_inputs(
        dtype="torch.float16",
        head_dim=128,
        head_dim_v=128,
        max_seqlen_q=129,
    )
    two_cta = one_cta._replace(max_seqlen_q=257)

    assert not select_fwd_config(one_cta).use_2cta_instrs
    assert select_fwd_config(two_cta).use_2cta_instrs


def test_varlen_mha_default_is_nonpersistent_without_tma_output():
    inputs = make_inputs(
        is_varlen_q=True,
        has_cu_seqlens_q=True,
        batch_size=4,
        max_seqlen_q=512,
        max_seqlen_k=768,
    )
    config = select_fwd_config(inputs)

    assert not config.use_clc_scheduler
    assert not config.is_static_persistent
    assert not config.use_tma_o
    with pytest.raises(ValueError, match="TMA O"):
        validate_fwd_config(replace(config, use_tma_o=True), inputs)


def test_padded_d192_overlap_disables_static_persistent_scheduling():
    inputs = make_inputs(head_dim=192, head_dim_v=128)

    assert not select_fwd_config(inputs).is_static_persistent


def test_varlen_k_dense_q_clc_rejects_static_persistent_codegen_flag():
    inputs = make_inputs(
        head_dim=128,
        head_dim_v=128,
        num_heads=16,
        num_heads_kv=4,
        pack_gqa=True,
        is_varlen_q=False,
        has_cu_seqlens_k=True,
    )
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler
    assert not config.is_static_persistent
    with pytest.raises(
        ValueError, match="Static persistent scheduling is not effective"
    ):
        validate_fwd_config(replace(config, is_static_persistent=True), inputs)


_B200_OUTPUT_BOUNDARY_INPUTS = make_inputs(
    num_heads=32,
    num_heads_kv=32,
    batch_size=1,
    total_q=257,
    total_k=2048,
    max_seqlen_q=257,
    max_seqlen_k=2048,
)
_B200_OUTPUT_EXCLUSION_INPUTS = _B200_OUTPUT_BOUNDARY_INPUTS._replace(
    batch_size=2,
    total_q=2 * 257,
    total_k=2 * 2048,
)
_B200_OUTPUT_EXCLUSIONS = (
    {"dtype": "torch.float16"},
    {"device_arch": 103},
    {"page_size": 128},
    {"has_qv": True},
    {"has_gather_kv": True},
    {"has_score_mod": True},
    {"has_mask_mod": True},
    {"has_learnable_sink": True},
    {"has_lse": True},
    {"pack_gqa": True},
    {"requested_tile_m": 64},
    {"requested_tile_n": 64},
    {"num_heads": 32, "num_heads_kv": 16, "pack_gqa": True},
    {"num_heads": 32, "num_heads_kv": 8, "pack_gqa": False},
)


@pytest.mark.parametrize(
    ("head_dim", "config_field"),
    [(64, "use_tma_o"), (128, "use_2cta_instrs")],
)
@pytest.mark.parametrize("qhead_per_kvhead", [1, 4, 8, 16])
def test_b200_bf16_output_policy_covers_retained_mha_and_gqa(
    head_dim, config_field, qhead_per_kvhead
):
    inputs = make_inputs(
        head_dim=head_dim,
        head_dim_v=head_dim,
        num_heads=16,
        num_heads_kv=16 // qhead_per_kvhead,
        pack_gqa=qhead_per_kvhead != 1,
        batch_size=4,
        total_q=4 * 257,
        total_k=4 * 2048,
        max_seqlen_q=257,
        max_seqlen_k=2048,
    )

    assert not getattr(select_fwd_config(inputs), config_field)


@pytest.mark.parametrize(
    ("head_dim", "config_field"),
    [(64, "use_tma_o"), (128, "use_2cta_instrs")],
)
def test_b200_bf16_output_policy_requires_measured_m_block_work(head_dim, config_field):
    underfilled = make_inputs(
        head_dim=head_dim,
        head_dim_v=head_dim,
        num_heads=31,
        num_heads_kv=31,
        batch_size=1,
        total_q=257,
        total_k=2048,
        max_seqlen_q=257,
        max_seqlen_k=2048,
    )
    measured_boundary = underfilled._replace(num_heads=32, num_heads_kv=32)

    assert getattr(select_fwd_config(underfilled), config_field)
    assert not getattr(select_fwd_config(measured_boundary), config_field)


@pytest.mark.parametrize(
    ("changes", "expected_tma_o"),
    [
        ({}, False),
        ({"max_seqlen_q": 256}, True),
        ({"max_seqlen_k": 2047}, True),
    ],
)
def test_b200_d64_output_policy_boundaries(changes, expected_tma_o):
    inputs = _B200_OUTPUT_BOUNDARY_INPUTS._replace(**changes)

    assert select_fwd_config(inputs).use_tma_o is expected_tma_o


def test_b200_d128_output_policy_keeps_short_k_on_2cta():
    inputs = _B200_OUTPUT_BOUNDARY_INPUTS._replace(head_dim=128, head_dim_v=128)

    assert not select_fwd_config(inputs).use_2cta_instrs
    assert select_fwd_config(inputs._replace(max_seqlen_k=2047)).use_2cta_instrs


@pytest.mark.parametrize(
    "changes",
    (
        *_B200_OUTPUT_EXCLUSIONS,
        {"causal": True},
        {"local": True},
        {"requested_num_splits": 2},
        {"use_block_sparsity": True},
    ),
)
def test_b200_d64_output_policy_exclusions(changes):
    inputs = _B200_OUTPUT_EXCLUSION_INPUTS._replace(**changes)

    assert select_fwd_config(inputs).use_tma_o


@pytest.mark.parametrize("changes", _B200_OUTPUT_EXCLUSIONS)
def test_b200_d128_output_policy_exclusions(changes):
    inputs = _B200_OUTPUT_EXCLUSION_INPUTS._replace(
        head_dim=128, head_dim_v=128, **changes
    )

    assert select_fwd_config(inputs).use_2cta_instrs


def test_fixed_split_does_not_require_sm_count():
    inputs = make_inputs(num_sms=0, requested_num_splits=1)

    assert select_fwd_config(inputs).num_splits == 1


def test_split_count_is_resolved_but_not_silently_changed_for_explicit_config():
    inputs = make_inputs(
        head_dim=128,
        head_dim_v=128,
        requested_num_splits=4,
    )
    config = select_fwd_config(inputs)

    assert config.num_splits == 4
    assert not config.use_2cta_instrs


def test_diff_head_split_uses_required_tile():
    inputs = make_inputs(
        head_dim=192,
        head_dim_v=128,
        max_seqlen_k=16384,
        requested_num_splits=4,
    )
    config = select_fwd_config(inputs)

    assert config.tile_n == 64
    assert config.num_splits == 4


def test_hd256_has_one_canonical_config():
    inputs = make_inputs(head_dim=256, head_dim_v=256)
    config = select_fwd_config(inputs)
    auto_inputs = inputs._replace(
        batch_size=1,
        num_heads=8,
        num_heads_kv=8,
        total_q=128,
        max_seqlen_q=128,
        max_seqlen_k=8192,
        requested_num_splits=None,
    )

    assert config.use_2cta_instrs
    assert config.registers == FwdSm100RegisterAllocation(256, 160, 32)
    assert select_fwd_config(auto_inputs) == config
    with pytest.raises(ValueError, match="does not support SplitKV"):
        select_fwd_config(inputs._replace(requested_num_splits=2))
    with pytest.raises(ValueError, match="one fixed configuration"):
        select_fwd_config(inputs._replace(requested_tile_m=64))
    with pytest.raises(ValueError, match="head-dim-256 kernel"):
        validate_fwd_config(replace(config, use_tma_o=True), inputs)


def test_mla_has_one_canonical_config():
    inputs = make_inputs(head_dim=64, head_dim_v=512, has_qv=True)
    config = select_fwd_config(inputs)

    assert config.num_threads == 384
    assert config.registers is None
    assert select_fwd_config(inputs._replace(has_gather_kv=True)).num_threads == 512
    with pytest.raises(ValueError, match="dedicated MLA kernel"):
        validate_fwd_config(replace(config, tile_m=128), inputs)
    with pytest.raises(ValueError, match="does not support SplitKV"):
        select_fwd_config(inputs._replace(requested_num_splits=2))


def test_unsupported_diff_head_split_is_rejected_not_normalized():
    inputs = make_inputs(
        head_dim=64,
        head_dim_v=512,
        requested_num_splits=4,
    )

    with pytest.raises(ValueError, match="does not support padded value"):
        select_fwd_config(inputs)


def test_basic_launch_geometry_is_rejected_before_compilation():
    inputs = make_inputs()
    config = select_fwd_config(inputs)

    with pytest.raises(ValueError, match="Tile dimensions must be multiples of 16"):
        validate_fwd_config(replace(config, tile_n=127), inputs)
    with pytest.raises(ValueError, match="num_threads must be a positive multiple"):
        validate_fwd_config(replace(config, num_threads=511), inputs)


def test_auto_split_resolves_from_exact_shape_metadata():
    inputs = make_inputs(
        num_heads=1,
        num_heads_kv=1,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=16384,
        requested_num_splits=None,
    )
    changed_shape = inputs._replace(max_seqlen_k=8192)

    assert select_fwd_config(inputs).num_splits == 128
    assert select_fwd_config(changed_shape).num_splits == 64


def test_auto_split_normalizes_nonsplit_to_one():
    assert (
        num_splits_heuristic(
            total_mblocks=1024,
            num_sms=132,
            num_n_blocks=128,
            max_splits=128,
        )
        == 1
    )


def test_seqlen_k_per_split_sets_minimum_split_count():
    inputs = make_inputs(
        max_seqlen_k=4224,
        seqlen_k_per_split=1024,
        requested_num_splits=None,
    )
    config = select_fwd_config(inputs)

    assert config.num_splits == 5
    with pytest.raises(ValueError, match="requires num_splits >= 5"):
        validate_fwd_config(replace(config, num_splits=4), inputs)


@pytest.mark.parametrize("device_arch", [80, 90, 100, 110, 120])
def test_architecture_defaults_validate(device_arch):
    inputs = make_inputs(
        device_arch=device_arch,
        requested_use_clc_scheduler=device_arch // 10 in (10, 11),
    )
    config = select_fwd_config(inputs)

    assert config.device_capacity == device_arch // 10


def test_architecture_specific_unsupported_features_are_rejected():
    unsupported_arch = make_inputs(device_arch=70, requested_use_clc_scheduler=False)
    sm120_paged = make_inputs(
        device_arch=120,
        page_size=128,
        requested_use_clc_scheduler=False,
    )
    irregular_paged = make_inputs(head_dim=72, head_dim_v=72, page_size=64)

    with pytest.raises(ValueError, match="Unsupported forward architecture family SM7"):
        select_fwd_config(unsupported_arch)
    with pytest.raises(ValueError, match="SM120 forward does not support"):
        select_fwd_config(sm120_paged)
    with pytest.raises(ValueError, match="head dimensions divisible by 16"):
        select_fwd_config(irregular_paged)


def test_non_sm90_tile_override_does_not_claim_ignored_mma_flags():
    inputs = make_inputs(
        device_arch=120,
        requested_tile_m=64,
        requested_use_clc_scheduler=False,
    )
    config = select_fwd_config(inputs)

    assert not config.mma_pv_is_rs
    assert not config.intra_wg_overlap
    with pytest.raises(ValueError, match="does not expose SM90 MMA flags"):
        validate_fwd_config(replace(config, mma_pv_is_rs=True), inputs)


def test_sm90_tile_override_preserves_legacy_rs_defaults():
    inputs = make_inputs(
        device_arch=90,
        head_dim=96,
        head_dim_v=96,
        requested_tile_m=128,
        requested_tile_n=128,
        requested_use_clc_scheduler=False,
    )
    config = select_fwd_config(inputs)

    assert config.tile_m == 128
    assert config.tile_n == 128
    assert config.num_threads == 384
    assert config.mma_pv_is_rs
    assert config.intra_wg_overlap
    with pytest.raises(ValueError, match="requires num_threads=384"):
        validate_fwd_config(replace(config, num_threads=512), inputs)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"requested_tile_m": 64}, (256, 56)),
        ({"requested_tile_m": 128}, (240, 24)),
        ({}, (160, 32)),
        ({"requested_tile_m": 128, "page_size": 16}, (224, 40)),
        (
            {
                "requested_tile_m": 128,
                "num_heads": 24,
                "num_heads_kv": 8,
                "pack_gqa": True,
            },
            (224, 40),
        ),
    ],
)
def test_sm90_register_allocation_matches_existing_kernel_policy(changes, expected):
    inputs = make_inputs(
        device_arch=90,
        requested_use_clc_scheduler=False,
        **changes,
    )

    config = select_fwd_config(inputs)

    assert config.num_threads == 128 * (config.tile_m // 64 + 1)
    assert config.registers == FwdSm90RegisterAllocation(*expected)


def test_explicit_2cta_config_is_valid_when_environment_default_disables_it():
    inputs = make_inputs(head_dim=128, head_dim_v=128, disable_2cta=True)
    config = select_fwd_config(inputs)

    assert not config.use_2cta_instrs
    validate_fwd_config(replace(config, use_2cta_instrs=True), inputs)


@pytest.mark.parametrize(
    ("device_arch", "registers"),
    [
        (90, FwdSm90RegisterAllocation(152, 56)),
        (100, FwdSm100RegisterAllocation(192, 80, 48)),
    ],
)
def test_explicit_register_allocation_accepts_valid_values(device_arch, registers):
    inputs = make_inputs(device_arch=device_arch, requested_use_clc_scheduler=False)
    config = select_fwd_config(inputs)

    validate_fwd_config(replace(config, registers=registers), inputs)
    assert config.registers != registers


@pytest.mark.parametrize(
    ("device_arch", "registers", "error_type", "match"),
    [
        (100, None, TypeError, "requires an SM100 register allocation"),
        (100, FwdSm100RegisterAllocation(199, 64, 48), ValueError, "multiples of 8"),
        (100, FwdSm100RegisterAllocation(120, 120, 120), ValueError, "at least 128"),
        (100, FwdSm100RegisterAllocation(184, 136, 24), ValueError, "at most 128"),
        (100, FwdSm100RegisterAllocation(200, 64, 56), ValueError, "512-register budget"),
        (90, None, TypeError, "requires an SM90 register allocation"),
        (
            90,
            FwdSm100RegisterAllocation(192, 80, 48),
            TypeError,
            "requires an SM90 register allocation",
        ),
        (90, FwdSm90RegisterAllocation(159, 32), ValueError, "multiples of 8"),
        (90, FwdSm90RegisterAllocation(120, 128), ValueError, "at least 128"),
        (90, FwdSm90RegisterAllocation(128, 136), ValueError, "at most 128"),
        (90, FwdSm90RegisterAllocation(168, 24), ValueError, "512-register budget"),
    ],
)
def test_explicit_register_allocation_rejects_invalid_values(
    device_arch, registers, error_type, match
):
    inputs = make_inputs(device_arch=device_arch, requested_use_clc_scheduler=False)
    config = select_fwd_config(inputs)

    with pytest.raises(error_type, match=match):
        validate_fwd_config(replace(config, registers=registers), inputs)


def test_specialization_projects_exact_split_counts_per_kernel():
    split_2 = select_fwd_config(make_inputs(requested_num_splits=2))
    split_4 = select_fwd_config(make_inputs(requested_num_splits=4))

    assert fwd_config_compile_bucket(split_2, 64) == fwd_config_compile_bucket(
        split_4, 64
    )

    assert fwd_combine_tile(64) == (16, 64)
    assert fwd_combine_tile(128) == (8, 128)
    assert combine_log_max_splits(2, tile_m=16) == 4
    assert combine_log_max_splits(16, tile_m=16) == 4
    assert combine_log_max_splits(17, tile_m=16) == 5
    assert combine_log_max_splits(2, tile_m=8) == 5


def test_sm120_legacy_split_request_preserves_public_error():
    inputs = make_inputs(
        device_arch=120,
        requested_num_splits=3,
        requested_use_clc_scheduler=False,
    )

    with pytest.raises(
        AssertionError, match="SM120 forward only supports num_splits=1"
    ):
        select_fwd_config(inputs)


def test_validator_rejects_invalid_explicit_architecture():
    inputs = make_inputs()
    config = replace(select_fwd_config(inputs), device_capacity=9)

    with pytest.raises(ValueError, match="targets SM9"):
        validate_fwd_config(config, inputs)
