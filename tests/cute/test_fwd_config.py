import itertools
from dataclasses import replace

import pytest

from flash_attn.cute.config import (
    FwdHeuristicInputs,
    Sm90Registers,
    Sm100Registers,
    num_splits_heuristic,
    select_fwd_config,
    select_registers,
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


def test_varlen_tile_count_semaphore_selects_dynamic_scheduling():
    # Single-M-block varlen decode permits static persistence without the semaphore.
    inputs = make_inputs(
        has_cu_seqlens_q=True,
        has_caller_scheduler_metadata=True,
        has_tile_count_semaphore=True,
        requested_use_clc_scheduler=True,
        causal=True,
        max_seqlen_q=1,
        num_heads_kv=4,
        pack_gqa=True,
    )
    config = select_fwd_config(inputs)

    assert not config.use_clc_scheduler
    assert not config.is_static_persistent
    with pytest.raises(ValueError, match="CLC"):
        validate_fwd_config(replace(config, use_clc_scheduler=True), inputs)
    with pytest.raises(ValueError, match="Static persistent"):
        validate_fwd_config(replace(config, is_static_persistent=True), inputs)
    without_semaphore = select_fwd_config(inputs._replace(has_tile_count_semaphore=False))
    assert without_semaphore.is_static_persistent or without_semaphore.use_clc_scheduler


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

    # Fixed KV extents impose a minimum split count, not an exact count.
    assert config.num_splits == 33
    validate_fwd_config(replace(config, num_splits=1), decode)
    with pytest.raises(ValueError, match="requires num_splits >= 5"):
        validate_fwd_config(replace(config, num_splits=4), decode)
    with pytest.raises(ValueError, match="divisible by tile_n"):
        validate_fwd_config(replace(config, tile_n=96), decode._replace(seqlen_k_per_split=1000))


def test_auto_splits_cover_the_fixed_kv_extent():
    # The heuristic picks 132 // 7 = 18 splits; 8192 / 256 needs 32.
    decode = make_inputs(
        num_heads=7,
        num_heads_kv=7,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=8192,
        seqlen_k_per_split=256,
        requested_num_splits=None,
    )
    config = select_fwd_config(decode)

    assert config.num_splits == 32
    validate_fwd_config(config, decode)
    assert select_fwd_config(decode._replace(seqlen_k_per_split=None)).num_splits == 18


def test_hd192_shared_memory_bounds_explicit_tiles():
    inputs = make_inputs(
        head_dim=192, head_dim_v=128, max_seqlen_q=128, max_seqlen_k=4096
    )
    config = select_fwd_config(inputs)

    validate_fwd_config(config, inputs)
    from flash_attn.cute.config import sm100_hd192_uneven_kv_smem_bytes

    too_wide = replace(config, tile_n=192, q_stage=1, use_2cta_instrs=False)
    # Observed SMEM allocation at launch.
    assert sm100_hd192_uneven_kv_smem_bytes(too_wide, inputs) == 247808
    with pytest.raises(ValueError, match="needs 247808 bytes of shared memory"):
        validate_fwd_config(too_wide, inputs)


def test_automatic_splitkv_stays_unsplit_without_sm100():
    decode = make_inputs(
        batch_size=1,
        num_heads=1,
        num_heads_kv=1,
        total_q=1,
        max_seqlen_q=1,
        total_k=16384,
        max_seqlen_k=16384,
        requested_num_splits=None,
    )

    assert select_fwd_config(decode).num_splits > 1
    # The selector must not pick a split count that validation rejects.
    for device_arch in (80, 90, 120):
        assert select_fwd_config(decode._replace(device_arch=device_arch)).num_splits == 1


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
    with pytest.raises(ValueError, match="SM120 forward only supports num_splits=1"):
        select_fwd_config(make_inputs(device_arch=120, requested_num_splits=3))


def test_sm90_flags_follow_tile_overrides_and_stay_sm90_only():
    sm90 = make_inputs(device_arch=90, head_dim=96, head_dim_v=96, requested_tile_m=128)
    config = select_fwd_config(sm90)

    assert (config.tile_m, config.mma_pv_is_rs, config.intra_wg_overlap) == (128, True, True)
    with pytest.raises(ValueError, match="SM90 tile_m must be"):
        validate_fwd_config(replace(config, tile_m=256), sm90)
    # SM90 flag requests are ignored on SM120; explicit configs must reject them.
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


# Frozen copy of the register policy the forward kernels applied before it moved into the
# selector (flash_fwd_sm100.py __init__ and the FP8 override in __call__; flash_fwd_sm90.py).
_LEGACY_SM100_TUNING = {
    (True, False, 128, False): (184, 80),
    (False, True, 128, False): (192, 72),
    (True, False, 192, False): (184, 80),
    (False, True, 192, False): (192, 72),
    (True, False, 128, True): (176, 80),
    (False, True, 128, True): (176, 64),
    (True, False, 192, True): (176, 64),
    (False, True, 192, True): (176, 72),
    (True, False, 256, False): (256, 160),
    (True, True, 256, False): (256, 160),
}


def legacy_sm100_registers(fp8, hd_padded, causal, use_2cta, sm103, paged_non_tma, q_stage):
    key = (use_2cta, causal, hd_padded, sm103)
    if hd_padded < 96:
        regs = (184, 64, 80) if paged_non_tma else (200, 64, 48)
    else:
        if not paged_non_tma and key in _LEGACY_SM100_TUNING:
            softmax, correction = _LEGACY_SM100_TUNING[key]
        elif not paged_non_tma:
            softmax, correction = 192, 80
        else:
            softmax, correction = 184, 64
        other = min((512 - softmax * q_stage - correction) // (3 - q_stage) // 8 * 8, 128)
        regs = (softmax, correction, other)
    if fp8:
        if hd_padded < 96:
            regs = (152, 96, 112) if paged_non_tma else (168, 96, 80)
        elif not paged_non_tma and key == (True, False, 128, False):
            regs = (160, 72, 512 - 160 * 2 - 72)
    return Sm100Registers(*regs)


def legacy_sm90_registers(tile_m, pack_gqa, qhead_per_kvhead, paged_non_tma):
    num_wg_mma = tile_m // 64
    regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[num_wg_mma]
    use_tma_q = not (pack_gqa and tile_m % qhead_per_kvhead != 0)
    if num_wg_mma == 2 and (not use_tma_q or paged_non_tma):
        regs = (224, 40)
    return Sm90Registers(*regs)


@pytest.mark.parametrize("device_arch", [100, 103, 107, 110])
def test_sm100_registers_match_the_legacy_kernel_policy(device_arch):
    sm103 = device_arch // 10 == 10 and device_arch % 10 >= 3
    dtypes = ["torch.bfloat16", "torch.float16", "torch.float8_e4m3fn"]
    head_dims = [32, 64, 72, 80, 96, 128, 192, 256]
    for dtype, d, causal, use_2cta, page, q_stage in itertools.product(
        dtypes, head_dims, [False, True], [False, True], [None, 64, 128], [1, 2]
    ):
        inputs = make_inputs(device_arch=device_arch, dtype=dtype, head_dim=d, head_dim_v=d,
                             causal=causal, page_size=page)
        got = select_registers(inputs, tile_m=128, tile_n=128, q_stage=q_stage,
                               use_2cta_instrs=use_2cta)
        expected = legacy_sm100_registers(
            "float8" in dtype, inputs.head_dim_padded, causal, use_2cta, sm103,
            page not in (None, 128), q_stage,
        )
        assert got == expected, (dtype, d, causal, use_2cta, page, q_stage)


def test_sm90_registers_match_the_legacy_kernel_policy():
    for tile_m, pack_gqa, qhead_per_kvhead, page in itertools.product(
        [64, 128, 192], [False, True], [1, 3, 4, 8], [None, 64, 128]
    ):
        inputs = make_inputs(device_arch=90, num_heads=8 * qhead_per_kvhead, num_heads_kv=8,
                             pack_gqa=pack_gqa, page_size=page)
        got = select_registers(inputs, tile_m=tile_m, tile_n=128, q_stage=1, use_2cta_instrs=False)
        expected = legacy_sm90_registers(tile_m, pack_gqa, qhead_per_kvhead, page not in (None, 128))
        assert got == expected, (tile_m, pack_gqa, qhead_per_kvhead, page)


def test_selected_configs_carry_the_legacy_register_split():
    # Wiring: select_fwd_config must hand select_registers its final tile, stage, and 2CTA choice.
    hd256_decode = make_inputs(device_arch=100, head_dim=256, head_dim_v=256, num_heads=8,
                               num_heads_kv=8, batch_size=16, max_seqlen_q=1, total_q=16)
    fp8_2cta = make_inputs(device_arch=100, dtype="torch.float8_e4m3fn", head_dim=128,
                           head_dim_v=128, max_seqlen_q=4096)
    for inputs in (hd256_decode, fp8_2cta, make_inputs(device_arch=103, head_dim=192, head_dim_v=128)):
        config = select_fwd_config(inputs)
        expected = legacy_sm100_registers(
            "float8" in inputs.dtype, inputs.head_dim_padded, inputs.causal, config.use_2cta_instrs,
            inputs.device_arch % 10 >= 3, False, config.q_stage,
        )
        assert config.registers == expected, (inputs, config)
    sm90 = make_inputs(device_arch=90, head_dim=128, head_dim_v=128)
    config = select_fwd_config(sm90)
    assert config.registers == legacy_sm90_registers(config.tile_m, False, 1, False)


def test_invalid_sm90_register_splits_are_rejected():
    sm90 = make_inputs(device_arch=90, head_dim=128, head_dim_v=128)
    config = select_fwd_config(sm90)
    assert config.tile_m == 128  # two MMA warp groups: 170 registers per thread at launch
    with pytest.raises(ValueError, match="MMA registers must be"):
        validate_fwd_config(replace(config, registers=Sm90Registers(160, 24)), sm90)
    with pytest.raises(ValueError, match="producer <="):
        validate_fwd_config(replace(config, registers=Sm90Registers(240, 176)), sm90)
    with pytest.raises(ValueError, match="exceeds 512"):
        validate_fwd_config(replace(config, registers=Sm90Registers(248, 24)), sm90)


def test_sm100_rejects_tile_m_other_than_128():
    inputs = make_inputs(device_arch=100, head_dim=64, head_dim_v=64)
    config = select_fwd_config(inputs)
    with pytest.raises(ValueError, match="tile_m=128"):
        validate_fwd_config(replace(config, tile_m=64), inputs)


def test_invalid_register_splits_are_rejected():
    config = select_fwd_config(_BASE_INPUTS)
    with pytest.raises(ValueError, match="needs registers of type Sm100Registers"):
        validate_fwd_config(replace(config, registers=None), _BASE_INPUTS)
    with pytest.raises(ValueError, match="needs registers of type Sm100Registers"):
        validate_fwd_config(replace(config, registers=Sm90Registers(240, 24)), _BASE_INPUTS)
    with pytest.raises(ValueError, match="multiples of 8"):
        validate_fwd_config(replace(config, registers=Sm100Registers(196, 64, 48)), _BASE_INPUTS)
    # Lowering "other" above the launch register count is not a valid setmaxnreg.dec.
    with pytest.raises(ValueError, match="other <= 128"):
        validate_fwd_config(replace(config, registers=Sm100Registers(160, 64, 136)), _BASE_INPUTS)
    with pytest.raises(ValueError, match="exceeds 512"):
        validate_fwd_config(replace(config, registers=Sm100Registers(232, 64, 48)), _BASE_INPUTS)
    # Correction may raise above the launch count (hd256 uses 160).
    assert config.q_stage == 2
    validate_fwd_config(replace(config, registers=Sm100Registers(176, 136, 24)), _BASE_INPUTS)
    sm80 = make_inputs(device_arch=80)
    with pytest.raises(ValueError, match="needs registers of type NoneType"):
        validate_fwd_config(
            replace(select_fwd_config(sm80), registers=Sm100Registers(200, 64, 48)), sm80
        )
