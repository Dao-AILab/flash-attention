import itertools
from contextlib import nullcontext
from dataclasses import replace

import pytest

from flash_attn.cute.config import (
    FwdHeuristicInputs,
    Sm90Registers,
    Sm100Registers,
    select_fwd_config,
    select_registers,
    validate_fwd_config,
)

_BASE_INPUTS = FwdHeuristicInputs(
    device_arch=100, num_sms=132, dtype="torch.bfloat16", head_dim=64, head_dim_v=64,
    num_heads=16, num_heads_kv=16, batch_size=2, total_q=8192, total_k=8192,
    max_seqlen_q=4096, max_seqlen_k=4096, requested_num_splits=1,
)
_HD128 = dict(head_dim=128, head_dim_v=128)
_HD192 = dict(head_dim=192, head_dim_v=128)
_HD256 = dict(head_dim=256, head_dim_v=256)
_DECODE = dict(num_heads=1, num_heads_kv=1, batch_size=1, max_seqlen_q=1, max_seqlen_k=16384,
               requested_num_splits=None)
_SEMAPHORE = dict(has_cu_seqlens_q=True, has_caller_scheduler_metadata=True,
                  has_tile_count_semaphore=True)


def make_inputs(*, pack_gqa: bool | None = None, **changes) -> FwdHeuristicInputs:
    return _BASE_INPUTS._replace(pack_gqa=pack_gqa, **changes)


@pytest.mark.parametrize("device_arch", [80, 90, 100, 103, 110, 120])
def test_every_architecture_selects_a_valid_default(device_arch):
    select_fwd_config(make_inputs(device_arch=device_arch))


@pytest.mark.parametrize(
    ("changes", "pack_gqa", "expected"),
    [
        ({}, None, False),
        ({"num_heads_kv": 4}, None, True),
        ({"num_heads_kv": 4}, False, False),
        ({}, True, True),
        ({"device_arch": 90, "num_heads": 24, "num_heads_kv": 8, **_HD128}, None, True),
        ({"num_heads": 24, "num_heads_kv": 8, **_HD256}, None, False),
        # Per-head sparse masks force pack_gqa off, even over an explicit request.
        ({"num_heads_kv": 4, "use_block_sparsity": True, "sparse_mask_per_head": True},
         True, False),
    ],
)
def test_pack_gqa_resolves_before_dependent_policy(changes, pack_gqa, expected):
    inputs = make_inputs(pack_gqa=pack_gqa, **changes)

    config = select_fwd_config(inputs)

    assert config.pack_gqa is expected
    assert config == select_fwd_config(inputs._replace(pack_gqa=expected))
    if inputs.sparse_mask_per_head:
        assert select_fwd_config(inputs._replace(sparse_mask_per_head=False)).pack_gqa
        with pytest.raises(ValueError, match="head-broadcast sparse masks"):
            validate_fwd_config(replace(config, pack_gqa=True), inputs)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        # 2CTA needs more than two packed Q tiles.
        ({"dtype": "torch.float16", "max_seqlen_q": 256, **_HD128}, {"use_2cta_instrs": False}),
        ({"dtype": "torch.float16", "max_seqlen_q": 257, **_HD128}, {"use_2cta_instrs": True}),
        ({"requested_num_splits": 4, **_HD128}, {"num_splits": 4, "use_2cta_instrs": False}),
        # hd256 decode uses 2CTA only while the grid fits one wave.
        ({**_HD256, "num_heads": 8, "num_heads_kv": 8, "max_seqlen_q": 1, "batch_size": 8},
         {"use_2cta_instrs": True}),
        ({**_HD256, "num_heads": 8, "num_heads_kv": 8, "max_seqlen_q": 1, "batch_size": 16},
         {"use_2cta_instrs": False}),
        # Decode uses S ping-pong only without score modifiers.
        ({"max_seqlen_q": 1, **_HD128}, {"q_stage": 1, "use_s_ping_pong": True}),
        ({"max_seqlen_q": 1, "disable_s_ping_pong": True, **_HD128}, {"use_s_ping_pong": False}),
        ({"device_arch": 90, "head_dim": 96, "head_dim_v": 96, "requested_tile_m": 128},
         {"tile_m": 128, "mma_pv_is_rs": True, "intra_wg_overlap": True}),
    ],
)
def test_selection_policy(changes, expected):
    config = select_fwd_config(make_inputs(**changes))

    assert {name: getattr(config, name) for name in expected} == expected


@pytest.mark.parametrize(
    ("changes", "expected_splits", "expected_tile_n"),
    [
        (_DECODE, 128, 128),
        ({**_DECODE, "max_seqlen_k": 8192}, 64, 128),
        ({"requested_num_splits": None}, 1, 128),
        # Only SM100/SM110 implement SplitKV.
        ({**_DECODE, "device_arch": 90}, 1, 128),
        # Diff-head SplitKV reselects: tile_n=64 for long KV, no split for short KV.
        ({**_DECODE, **_HD192, "requested_num_splits": 4}, 128, 64),
        ({**_DECODE, **_HD192, "requested_num_splits": 4, "max_seqlen_k": 1024}, 1, 128),
        ({**_DECODE, "head_dim": 128, "head_dim_v": 192, "requested_num_splits": 4,
          "max_seqlen_k": 1024}, 1, 128),
        # Fixed KV extents impose a minimum split count, not an exact count.
        ({**_DECODE, "max_seqlen_k": 4224, "seqlen_k_per_split": 1024}, 33, 128),
        # The heuristic picks 132 // 7 = 18 splits; 8192 / 256 needs 32.
        ({**_DECODE, "num_heads": 7, "num_heads_kv": 7, "max_seqlen_k": 8192}, 18, 128),
        ({**_DECODE, "num_heads": 7, "num_heads_kv": 7, "max_seqlen_k": 8192,
          "seqlen_k_per_split": 256}, 32, 128),
    ],
)
def test_split_selection(changes, expected_splits, expected_tile_n):
    config = select_fwd_config(make_inputs(**changes))

    assert (config.num_splits, config.tile_n) == (expected_splits, expected_tile_n)


@pytest.mark.parametrize(
    ("changes", "config_changes", "match"),
    [
        # Selection-time rejections.
        ({"device_arch": 70}, {}, "Unsupported forward architecture family SM7"),
        ({"device_arch": 120, "page_size": 128}, {}, "SM120 forward does not support"),
        ({"device_arch": 120, "requested_num_splits": 3}, {}, "SM120 forward only supports num_s"),
        ({"head_dim": 72, "head_dim_v": 72, "page_size": 64}, {}, "head dimensions divisible"),
        # Geometry and architecture.
        ({}, dict(tile_m=64), "tile_m=128"),
        ({}, dict(tile_n=127), "multiples of 16"),
        ({}, dict(device_capacity=9), "targets SM9"),
        ({"device_arch": 90, **_HD128}, dict(tile_m=256), "SM90 tile_m must be"),
        ({"device_arch": 120}, dict(mma_pv_is_rs=True), "does not expose SM90 MMA flags"),
        ({**_HD256}, dict(q_stage=2), "TMEM"),
        # pack_gqa: an explicit request must match; an unspecified one lets the config decide.
        ({"num_heads_kv": 4, "pack_gqa": True}, dict(pack_gqa=False), "disagrees"),
        ({"num_heads": 24, "num_heads_kv": 8, **_HD128},
         dict(pack_gqa=False, use_tma_o=True, use_2cta_instrs=True), None),
        ({"num_heads": 24, "num_heads_kv": 8, **_HD128},
         dict(pack_gqa=True, use_tma_o=True, use_2cta_instrs=True), "TMA O"),
        # Features.
        ({"disable_2cta": True, **_HD128}, dict(use_2cta_instrs=True), None),
        ({"has_cu_seqlens_q": True, "batch_size": 4, "max_seqlen_q": 512},
         dict(use_tma_o=True), "TMA O"),
        ({"num_heads_kv": 4, "pack_gqa": True, "has_cu_seqlens_k": True,
          "requested_use_clc_scheduler": True, **_HD128},
         dict(is_static_persistent=True), "Static persistent scheduling"),
        ({**_SEMAPHORE, "causal": True, "max_seqlen_q": 1, "num_heads_kv": 4, "pack_gqa": True},
         dict(use_clc_scheduler=True), "CLC"),
        ({**_HD256, "causal": True}, dict(use_clc_scheduler=True), "CLC"),
        ({"max_seqlen_q": 1, "has_mask_mod": True, **_HD128},
         dict(use_s_ping_pong=True), "S ping-pong"),
        # SplitKV.
        ({**_DECODE, **_HD192, "requested_num_splits": 4}, dict(num_splits=4), None),
        ({**_DECODE, **_HD192, "requested_num_splits": 4}, dict(tile_n=128),
         "Diff-head SplitKV requires tile_n=64"),
        ({**_DECODE, "max_seqlen_k": 4224, "seqlen_k_per_split": 1024}, dict(num_splits=1), None),
        ({**_DECODE, "max_seqlen_k": 4224, "seqlen_k_per_split": 1024}, dict(num_splits=4),
         "requires num_splits >= 5"),
        ({**_DECODE, "max_seqlen_k": 4224, "seqlen_k_per_split": 1000}, dict(tile_n=96),
         "divisible by tile_n"),
        # Registers.
        ({**_HD128}, dict(registers=Sm90Registers(240, 24)), "type Sm100Registers"),
        ({**_HD128}, dict(registers=Sm100Registers(196, 64, 48)), "multiples of 8"),
        # Lowering "other" above the launch register count is not a valid setmaxnreg.dec.
        ({**_HD128}, dict(registers=Sm100Registers(160, 64, 136)), "other <= 128"),
        ({**_HD128}, dict(registers=Sm100Registers(232, 64, 48)), "exceeds 512"),
        # Correction may exceed the launch count (hd256 uses 160).
        ({}, dict(registers=Sm100Registers(176, 136, 24)), None),
        # SM90 hd128 uses two MMA warp groups: 170 registers per thread at launch.
        ({"device_arch": 90, **_HD128}, dict(registers=Sm90Registers(160, 24)), "MMA registers"),
        ({"device_arch": 80, **_HD128}, dict(registers=Sm100Registers(200, 64, 48)),
         "needs registers of type NoneType"),
    ],
)
def test_explicit_config_validation(changes, config_changes, match):
    inputs = make_inputs(**changes)

    with pytest.raises(ValueError, match=match) if match else nullcontext():
        validate_fwd_config(replace(select_fwd_config(inputs), **config_changes), inputs)


def test_smem_model_matches_measured_kernel_allocations():
    # Observed hd192 SMEM allocation at launch.
    hd192 = make_inputs(**_HD192, max_seqlen_q=128)
    too_wide = replace(select_fwd_config(hd192), tile_n=192, q_stage=1, use_2cta_instrs=False)
    with pytest.raises(ValueError, match="needs 247808 bytes of shared memory"):
        validate_fwd_config(too_wide, hd192)


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


_DTYPES = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float8_e4m3fn": 1}


def test_sm100_registers_match_the_legacy_kernel_policy():
    for device_arch, dtype, d, causal, use_2cta, page, q_stage in itertools.product(
        [100, 103, 107, 110], _DTYPES, [64, 80, 96, 128, 192, 256], [False, True],
        [False, True], [None, 64, 128], [1, 2],
    ):
        inputs = make_inputs(device_arch=device_arch, dtype=dtype, head_dim=d, head_dim_v=d,
                             causal=causal, page_size=page)
        got = select_registers(inputs, tile_m=128, tile_n=128, q_stage=q_stage,
                               use_2cta_instrs=use_2cta)
        expected = legacy_sm100_registers(
            "float8" in dtype, inputs.head_dim_padded, causal, use_2cta,
            device_arch in (103, 107), page not in (None, 128), q_stage,
        )
        assert got == expected, (device_arch, dtype, d, causal, use_2cta, page, q_stage)


def test_sm90_registers_match_the_legacy_kernel_policy():
    for tile_m, pack_gqa, qhead_per_kvhead, page in itertools.product(
        [64, 128, 192], [False, True], [1, 3, 4, 8], [None, 64, 128]
    ):
        inputs = make_inputs(device_arch=90, num_heads=8 * qhead_per_kvhead, num_heads_kv=8,
                             pack_gqa=pack_gqa, page_size=page)
        got = select_registers(inputs, tile_m=tile_m, tile_n=128, q_stage=1,
                               use_2cta_instrs=False)
        expected = legacy_sm90_registers(
            tile_m, pack_gqa, qhead_per_kvhead, page not in (None, 128)
        )
        assert got == expected, (tile_m, pack_gqa, qhead_per_kvhead, page)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["torch.bfloat16", "torch.float8_e4m3fn"])
def test_selected_configs_carry_the_legacy_policy(dtype, causal):
    # Wiring: select_fwd_config must pass its final tile, stage, and 2CTA choice.
    shapes = [
        dict(max_seqlen_q=4096),
        dict(max_seqlen_q=1, max_seqlen_k=16384, num_heads=8, num_heads_kv=1,
             requested_num_splits=None),
        dict(**_SEMAPHORE, max_seqlen_q=512),
        dict(requested_tile_n=64),
        dict(num_heads=8, num_heads_kv=8, batch_size=16, max_seqlen_q=1, total_q=16),
    ]
    for (hd, hdv), shape, device_arch in itertools.product(
        [(64, 64), (128, 128), (192, 128), (256, 256)], shapes, [100, 103]
    ):
        inputs = make_inputs(device_arch=device_arch, dtype=dtype, head_dim=hd, head_dim_v=hdv,
                             causal=causal, **shape)
        config = select_fwd_config(inputs)
        expected_registers = legacy_sm100_registers(
            "float8" in dtype, inputs.head_dim_padded, causal, config.use_2cta_instrs,
            device_arch == 103, False, config.q_stage,
        )
        assert config.registers == expected_registers, (inputs, config)
    sm90 = make_inputs(device_arch=90, **_HD128)
    config = select_fwd_config(sm90)
    assert config.registers == legacy_sm90_registers(config.tile_m, False, 1, False)

