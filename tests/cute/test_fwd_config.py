import inspect
from dataclasses import replace

import pytest

from flash_attn.cute.config import (
    FwdHeuristicInputs,
    FwdMainKernelConfigSpec,
    combine_log_max_splits,
    fwd_combine_tile,
    num_splits_heuristic,
    select_fwd_config,
    validate_fwd_config,
)
from flash_attn.cute.interface import _flash_attn_fwd


def make_inputs(**changes) -> FwdHeuristicInputs:
    values = {
        "device_capacity": 10,
        "device_arch": 100,
        "num_sms": 132,
        "dtype": "torch.bfloat16",
        "head_dim": 64,
        "head_dim_v": 64,
        "num_heads": 16,
        "num_heads_kv": 16,
        "batch_size": 2,
        "total_q": 8192,
        "total_k": 8192,
        "max_seqlen_q": 4096,
        "max_seqlen_k": 4096,
        "causal": False,
        "local": False,
        "window_size_left": None,
        "window_size_right": None,
        "is_varlen": False,
        "is_varlen_q": False,
        "has_cu_seqlens_q": False,
        "has_cu_seqlens_k": False,
        "has_seqused": False,
        "pack_gqa": False,
        "page_size": None,
        "use_block_sparsity": False,
        "sparse_q_block_size": None,
        "has_qv": False,
        "has_gather_kv": False,
        "has_score_mod": False,
        "has_mask_mod": False,
        "has_learnable_sink": False,
        "has_lse": False,
        "requested_tile_m": None,
        "requested_tile_n": None,
        "requested_mma_pv_is_rs": None,
        "requested_intra_wg_overlap": None,
        "requested_num_splits": 1,
        "requested_use_clc_scheduler": True,
        "disable_2cta": False,
    }
    values.update(changes)
    return FwdHeuristicInputs(**values)


def test_forward_config_is_optional_and_keyword_only():
    parameter = inspect.signature(_flash_attn_fwd).parameters["config"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_selector_inputs_and_config_are_hashable():
    inputs = make_inputs()

    assert hash(inputs)
    assert hash(select_fwd_config(inputs))


@pytest.mark.parametrize(
    ("changes", "expected_persistent"),
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
def test_sm103_long_d64_nonpersistent_scheduler_scope(changes, expected_persistent):
    inputs = replace(make_inputs(device_arch=103, max_seqlen_k=8192), **changes)

    assert select_fwd_config(inputs).is_persistent is expected_persistent


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
    inputs = replace(
        make_inputs(
            device_arch=103,
            batch_size=4,
            total_q=10240,
            max_seqlen_q=6400,
            is_varlen=True,
            is_varlen_q=True,
            has_cu_seqlens_q=True,
            has_cu_seqlens_k=True,
            requested_use_clc_scheduler=False,
        ),
        **changes,
    )
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler is expected_clc
    if expected_clc:
        assert not config.is_persistent


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
    inputs = replace(
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
            is_varlen=True,
            is_varlen_q=True,
            has_cu_seqlens_q=True,
            has_cu_seqlens_k=True,
            requested_use_clc_scheduler=False,
        ),
        **changes,
    )

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
    inputs = replace(
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
            requested_use_clc_scheduler=False,
        ),
        **changes,
    )

    assert select_fwd_config(inputs).use_clc_scheduler is expected_clc


def test_selector_preserves_a_codegen_changing_2cta_boundary():
    one_cta = make_inputs(
        dtype="torch.float16",
        head_dim=128,
        head_dim_v=128,
        max_seqlen_q=129,
    )
    two_cta = replace(one_cta, max_seqlen_q=257)

    assert not select_fwd_config(one_cta).use_2cta_instrs
    assert select_fwd_config(two_cta).use_2cta_instrs


def test_varlen_mha_default_is_nonpersistent_without_tma_output():
    inputs = make_inputs(
        is_varlen=True,
        is_varlen_q=True,
        batch_size=4,
        max_seqlen_q=512,
        max_seqlen_k=768,
    )
    config = select_fwd_config(inputs)

    assert not config.use_clc_scheduler
    assert not config.is_persistent
    assert not config.use_tma_o


def test_padded_d192_overlap_disables_persistent_scheduling():
    inputs = make_inputs(head_dim=192, head_dim_v=128)

    assert not select_fwd_config(inputs).is_persistent


def test_varlen_k_dense_q_clc_rejects_persistent_codegen_flag():
    inputs = make_inputs(
        head_dim=128,
        head_dim_v=128,
        num_heads=16,
        num_heads_kv=4,
        pack_gqa=True,
        is_varlen=True,
        is_varlen_q=False,
    )
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler
    assert not config.is_persistent
    validate_fwd_config(config, inputs)
    with pytest.raises(ValueError, match="Persistent scheduling is not effective"):
        validate_fwd_config(replace(config, is_persistent=True), inputs)


@pytest.mark.parametrize(
    ("head_dim", "config_field"),
    [(64, "use_tma_o"), (128, "use_2cta_instrs")],
)
@pytest.mark.parametrize("qhead_per_kvhead", [1, 4, 8, 16])
def test_b200_bf16_output_policy_covers_measured_mha_and_gqa(
    head_dim, config_field, qhead_per_kvhead
):
    inputs = make_inputs(
        head_dim=head_dim,
        head_dim_v=head_dim,
        num_heads=16,
        num_heads_kv=16 // qhead_per_kvhead,
        pack_gqa=qhead_per_kvhead != 1,
        batch_size=1,
        total_q=257,
        total_k=2048,
        max_seqlen_q=257,
        max_seqlen_k=2048,
    )

    assert not getattr(select_fwd_config(inputs), config_field)


@pytest.mark.parametrize(
    ("changes", "expected_tma_o"),
    [
        ({}, False),
        ({"max_seqlen_q": 256}, True),
        ({"max_seqlen_k": 2047}, True),
    ],
)
def test_b200_d64_output_policy_boundaries(changes, expected_tma_o):
    inputs = replace(
        make_inputs(
            batch_size=1,
            total_q=257,
            total_k=2048,
            max_seqlen_q=257,
            max_seqlen_k=2048,
        ),
        **changes,
    )

    assert select_fwd_config(inputs).use_tma_o is expected_tma_o


def test_b200_d128_output_policy_keeps_short_k_on_2cta():
    inputs = make_inputs(
        head_dim=128,
        head_dim_v=128,
        batch_size=1,
        total_q=257,
        total_k=2048,
        max_seqlen_q=257,
        max_seqlen_k=2048,
    )

    assert not select_fwd_config(inputs).use_2cta_instrs
    assert select_fwd_config(replace(inputs, max_seqlen_k=2047)).use_2cta_instrs


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "torch.float16"},
        {"device_arch": 103},
        {"causal": True},
        {"local": True},
        {"requested_num_splits": 2},
        {"page_size": 128},
        {"use_block_sparsity": True},
        {"has_qv": True},
        {"has_gather_kv": True},
        {"has_score_mod": True},
        {"has_mask_mod": True},
        {"has_learnable_sink": True},
        {"has_lse": True},
        {"requested_tile_m": 64},
        {"requested_tile_n": 64},
        {"num_heads": 16, "num_heads_kv": 8, "pack_gqa": True},
        {"num_heads": 16, "num_heads_kv": 4, "pack_gqa": False},
    ],
)
def test_b200_d64_output_policy_exclusions(changes):
    inputs = replace(
        make_inputs(
            batch_size=1,
            total_q=257,
            total_k=2048,
            max_seqlen_q=257,
            max_seqlen_k=2048,
        ),
        **changes,
    )

    assert select_fwd_config(inputs).use_tma_o


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "torch.float16"},
        {"device_arch": 103},
        {"page_size": 128},
        {"has_qv": True},
        {"has_gather_kv": True},
        {"has_score_mod": True},
        {"has_mask_mod": True},
        {"has_learnable_sink": True},
        {"has_lse": True},
        {"requested_tile_m": 64},
        {"requested_tile_n": 64},
        {"num_heads": 16, "num_heads_kv": 8, "pack_gqa": True},
        {"num_heads": 16, "num_heads_kv": 4, "pack_gqa": False},
    ],
)
def test_b200_d128_output_policy_exclusions(changes):
    inputs = replace(
        make_inputs(
            head_dim=128,
            head_dim_v=128,
            batch_size=1,
            total_q=257,
            total_k=2048,
            max_seqlen_q=257,
            max_seqlen_k=2048,
        ),
        **changes,
    )

    assert select_fwd_config(inputs).use_2cta_instrs


def test_split_count_is_resolved_but_not_silently_changed_for_explicit_config():
    inputs = make_inputs(
        head_dim=128,
        head_dim_v=128,
        requested_num_splits=4,
    )
    config = select_fwd_config(inputs)

    assert config.num_splits == 4
    assert not config.use_2cta_instrs
    validate_fwd_config(config, inputs)


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

    assert config.use_2cta_instrs
    with pytest.raises(ValueError, match="head-dim-256 kernel"):
        validate_fwd_config(replace(config, use_tma_o=True), inputs)


def test_mla_has_one_canonical_config():
    inputs = make_inputs(head_dim=64, head_dim_v=512, has_qv=True)
    config = select_fwd_config(inputs)

    assert config.num_threads == 384
    assert select_fwd_config(replace(inputs, has_gather_kv=True)).num_threads == 512
    with pytest.raises(ValueError, match="dedicated MLA kernel"):
        validate_fwd_config(replace(config, tile_m=128), inputs)
    with pytest.raises(ValueError, match="does not support SplitKV"):
        select_fwd_config(replace(inputs, requested_num_splits=2))


def test_sm100_rejects_ignored_sm90_mma_flags():
    inputs = make_inputs()
    config = replace(select_fwd_config(inputs), mma_pv_is_rs=True)

    with pytest.raises(ValueError, match="does not expose SM90 MMA flags"):
        validate_fwd_config(config, inputs)


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


def test_invalid_tma_o_varlen_config_is_rejected():
    inputs = make_inputs(is_varlen=True, is_varlen_q=True)
    config = replace(select_fwd_config(inputs), use_tma_o=True)

    with pytest.raises(ValueError, match="TMA O"):
        validate_fwd_config(config, inputs)


def test_auto_split_resolves_from_exact_shape_metadata():
    inputs = make_inputs(
        num_heads=1,
        num_heads_kv=1,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=16384,
        requested_num_splits=None,
    )
    changed_shape = replace(inputs, max_seqlen_k=8192)

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


@pytest.mark.parametrize("device_capacity", [8, 9, 10, 11, 12])
def test_architecture_defaults_validate(device_capacity):
    inputs = make_inputs(
        device_capacity=device_capacity,
        device_arch=device_capacity * 10,
        requested_use_clc_scheduler=device_capacity in (10, 11),
    )
    config = select_fwd_config(inputs)

    validate_fwd_config(config, inputs)
    assert config.device_capacity == device_capacity


def test_architecture_specific_unsupported_features_are_rejected():
    sm120_paged = make_inputs(
        device_capacity=12,
        device_arch=120,
        page_size=128,
        requested_use_clc_scheduler=False,
    )
    irregular_paged = make_inputs(head_dim=72, head_dim_v=72, page_size=64)

    with pytest.raises(ValueError, match="SM120 forward does not support"):
        select_fwd_config(sm120_paged)
    with pytest.raises(ValueError, match="head dimensions divisible by 16"):
        select_fwd_config(irregular_paged)


def test_non_sm90_tile_override_does_not_claim_ignored_mma_flags():
    inputs = make_inputs(
        device_capacity=12,
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
        device_capacity=9,
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


def test_sm90_thread_count_is_derived_from_tile_m():
    inputs = make_inputs(
        device_capacity=9,
        device_arch=90,
        requested_use_clc_scheduler=False,
    )

    config = select_fwd_config(inputs)

    assert config.tile_m == 192
    assert config.num_threads == 512


def test_explicit_2cta_config_is_valid_when_environment_default_disables_it():
    inputs = make_inputs(head_dim=128, head_dim_v=128, disable_2cta=True)
    config = select_fwd_config(inputs)

    assert not config.use_2cta_instrs
    validate_fwd_config(replace(config, use_2cta_instrs=True), inputs)


def test_specialization_projects_exact_split_counts_per_kernel():
    split_2 = select_fwd_config(make_inputs(requested_num_splits=2))
    split_4 = select_fwd_config(make_inputs(requested_num_splits=4))

    def projection(config):
        return FwdMainKernelConfigSpec.project(
            config,
            kernel_family="sm10",
            arch=100,
            pack_gqa=False,
            page_size=None,
            q_subtile_factor=1,
            kv_subtile_factor=1,
        )

    assert projection(split_2) == projection(split_4)

    assert fwd_combine_tile(64) == (16, 64)
    assert fwd_combine_tile(128) == (8, 128)
    assert combine_log_max_splits(2, tile_m=16) == 4
    assert combine_log_max_splits(16, tile_m=16) == 4
    assert combine_log_max_splits(17, tile_m=16) == 5
    assert combine_log_max_splits(2, tile_m=8) == 5


def test_sm120_legacy_split_request_preserves_public_error():
    inputs = make_inputs(
        device_capacity=12,
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
