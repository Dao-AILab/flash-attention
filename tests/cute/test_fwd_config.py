import inspect
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from flash_attn.cute.config import (
    FwdCombineKernelSpec,
    FwdHeuristicInputs,
    FwdMainKernelSpec,
    combine_log_max_splits,
    fwd_combine_tile,
    get_fwd_config_bucket,
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
        "max_seqlen_q": 4096,
        "max_seqlen_k": 4096,
        "causal": False,
        "local": False,
        "window_size_left": None,
        "window_size_right": None,
        "is_varlen": False,
        "is_varlen_q": False,
        "pack_gqa": False,
        "page_size": None,
        "use_block_sparsity": False,
        "sparse_q_block_size": None,
        "has_qv": False,
        "has_gather_kv": False,
        "requested_tile_m": None,
        "requested_tile_n": None,
        "requested_num_threads": 384,
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


def test_selector_inputs_and_config_are_frozen_and_hashable():
    inputs = make_inputs()
    config = select_fwd_config(inputs)

    assert hash(inputs)
    assert hash(config)
    with pytest.raises(FrozenInstanceError):
        inputs.head_dim = 128
    with pytest.raises(FrozenInstanceError):
        config.tile_m = 64


def test_dense_sm100_default_and_candidates_share_named_bucket():
    inputs = make_inputs()
    bucket = get_fwd_config_bucket(inputs)

    assert bucket.default == select_fwd_config(inputs)
    assert bucket.default in bucket.candidates
    assert len(bucket.candidates) == len(set(bucket.candidates))
    assert bucket.name == (
        "sm100.bfloat16.dense.noncausal.mha.d64.standard.tile128x128.q2.nosplit."
        "1cta.persistent.tma-o"
    )
    assert replace(bucket.default, q_stage=1) in bucket.candidates
    assert replace(bucket.default, is_persistent=False) in bucket.candidates
    assert (
        replace(bucket.default, use_clc_scheduler=True, is_persistent=False)
        in bucket.candidates
    )
    assert replace(bucket.default, use_tma_o=False) in bucket.candidates


def test_same_named_region_has_stable_default_and_candidates():
    first = make_inputs(max_seqlen_q=4096, max_seqlen_k=4096)
    second = make_inputs(max_seqlen_q=8192, max_seqlen_k=8192)

    first_bucket = get_fwd_config_bucket(first)
    second_bucket = get_fwd_config_bucket(second)

    assert first_bucket.name == second_bucket.name
    assert first_bucket.default == second_bucket.default
    assert first_bucket.candidates == second_bucket.candidates


def test_default_selector_does_not_construct_tuning_candidates():
    inputs = make_inputs()
    select_fwd_config.cache_clear()
    get_fwd_config_bucket.cache_clear()

    select_fwd_config(inputs)
    select_fwd_config(replace(inputs))

    assert select_fwd_config.cache_info().misses == 1
    assert select_fwd_config.cache_info().hits == 1
    assert get_fwd_config_bucket.cache_info().misses == 0


def test_bucket_identity_includes_exact_architecture_and_dtype():
    inputs = make_inputs(device_arch=103)
    bucket = get_fwd_config_bucket(inputs)
    other_arch = get_fwd_config_bucket(replace(inputs, device_arch=100))
    other_dtype = get_fwd_config_bucket(replace(inputs, dtype="torch.float16"))

    assert bucket.name.startswith("sm103.bfloat16.")
    assert bucket.name != other_arch.name
    assert bucket.name != other_dtype.name


def test_selector_preserves_a_codegen_changing_2cta_boundary():
    one_cta = make_inputs(
        head_dim=128,
        head_dim_v=128,
        max_seqlen_q=129,
    )
    two_cta = replace(one_cta, max_seqlen_q=257)

    assert not select_fwd_config(one_cta).use_2cta_instrs
    assert select_fwd_config(two_cta).use_2cta_instrs


def test_varlen_mha_uses_offline_default_with_valid_clc_candidate():
    inputs = make_inputs(
        is_varlen=True,
        is_varlen_q=True,
        batch_size=4,
        max_seqlen_q=512,
        max_seqlen_k=768,
    )
    bucket = get_fwd_config_bucket(inputs)

    assert not bucket.default.use_clc_scheduler
    assert not bucket.default.is_persistent
    assert not bucket.default.use_tma_o
    assert replace(bucket.default, use_clc_scheduler=True) in bucket.candidates


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


def test_long_dense_d128_selects_2cta_and_keeps_1cta_candidate():
    inputs = make_inputs(head_dim=128, head_dim_v=128)
    bucket = get_fwd_config_bucket(inputs)

    assert bucket.default.use_2cta_instrs
    assert replace(bucket.default, use_2cta_instrs=False) in bucket.candidates


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


def test_hd256_has_one_canonical_non_tunable_config():
    inputs = make_inputs(head_dim=256, head_dim_v=256)
    bucket = get_fwd_config_bucket(inputs)

    assert bucket.candidates == (bucket.default,)
    assert bucket.default.use_2cta_instrs
    with pytest.raises(ValueError, match="head-dim-256 kernel"):
        validate_fwd_config(replace(bucket.default, use_tma_o=True), inputs)


def test_mla_has_one_canonical_non_tunable_config():
    inputs = make_inputs(
        head_dim=64,
        head_dim_v=512,
        has_qv=True,
    )
    bucket = get_fwd_config_bucket(inputs)

    assert bucket.candidates == (bucket.default,)
    assert bucket.default.num_threads == 384
    topk = replace(inputs, has_gather_kv=True)
    topk_bucket = get_fwd_config_bucket(topk)
    assert topk_bucket.default.num_threads == 512
    assert ".mla-topk." in topk_bucket.name
    with pytest.raises(ValueError, match="dedicated MLA kernel"):
        validate_fwd_config(replace(bucket.default, tile_m=128), inputs)
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


def test_sm80_rejects_unimplementable_tile_thread_geometry():
    inputs = make_inputs(
        device_capacity=8,
        device_arch=80,
        requested_use_clc_scheduler=False,
    )
    config = select_fwd_config(inputs)

    with pytest.raises(ValueError, match="cannot implement"):
        validate_fwd_config(replace(config, tile_m=80, num_threads=128), inputs)


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
        requested_num_threads=384,
        requested_use_clc_scheduler=False,
    )
    config = select_fwd_config(inputs)

    assert config.tile_m == 128
    assert config.tile_n == 128
    assert config.mma_pv_is_rs
    assert config.intra_wg_overlap


def test_environment_policy_does_not_remove_valid_2cta_candidate():
    inputs = make_inputs(
        head_dim=128,
        head_dim_v=128,
        disable_2cta=True,
    )
    bucket = get_fwd_config_bucket(inputs)

    assert not bucket.default.use_2cta_instrs
    assert replace(bucket.default, use_2cta_instrs=True) in bucket.candidates


def test_specialization_projects_exact_split_counts_per_kernel():
    main_fields = {field.name for field in fields(FwdMainKernelSpec)}
    combine_fields = {field.name for field in fields(FwdCombineKernelSpec)}

    assert "is_split_kv" in main_fields
    assert "num_heads_kv" in main_fields
    assert "num_splits" not in main_fields
    assert "log_max_splits" in combine_fields
    assert "num_splits" not in combine_fields
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


def test_bucket_rejects_invalid_explicit_architecture():
    inputs = make_inputs()
    config = replace(select_fwd_config(inputs), device_capacity=9)

    with pytest.raises(ValueError, match="targets SM9"):
        validate_fwd_config(config, inputs)
