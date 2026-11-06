import itertools
from contextlib import nullcontext
from dataclasses import asdict, astuple, replace
from unittest import mock

import pytest

from flash_attn.cute import config as config_module
from flash_attn.cute.config import (
    SM100_SMEM_BYTES,
    FwdConfig,
    FwdHeuristicInputs,
    Sm90Registers,
    Sm100Registers,
    TunedSm100Overrides,
    derive_fwd_config,
    select_ex2_emu,
    select_fwd_config,
    select_registers,
    select_sm100_smem_layout,
    sm100_fwd_smem_bytes,
    sm100_smem_capacity,
    sm100_uneven_kv_smem,
    validate_fwd_config,
)

# Standard-selector cases; test_swapab_config.py covers SwapAB selection.
_BASE_INPUTS = FwdHeuristicInputs(
    device_arch=100, num_sms=132, dtype="torch.bfloat16", head_dim=64, head_dim_v=64,
    num_heads=16, num_heads_kv=16, batch_size=2, total_q=8192, total_k=8192,
    max_seqlen_q=4096, max_seqlen_k=4096, requested_num_splits=1, requested_use_swapab=False,
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
@pytest.mark.parametrize("causal", [False, True])
def test_every_architecture_selects_a_valid_default(device_arch, causal):
    inputs = make_inputs(device_arch=device_arch, causal=causal)
    validate_fwd_config(select_fwd_config(inputs), inputs)


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
        # Explicit CLC may run outside the measured defaults, but not with static persistence.
        ({"num_heads_kv": 4, "pack_gqa": True, "has_cu_seqlens_k": True, **_HD128},
         dict(use_clc_scheduler=True, is_static_persistent=False), None),
        ({"num_heads_kv": 4, "pack_gqa": True, "has_cu_seqlens_k": True, **_HD128},
         dict(use_clc_scheduler=True, is_static_persistent=True), "Static persistent scheduling"),
        ({**_SEMAPHORE, "causal": True, "max_seqlen_q": 1, "num_heads_kv": 4, "pack_gqa": True},
         dict(use_clc_scheduler=True), "CLC"),
        # hd256 2CTA CLC supports dense, not varlen, cluster mapping.
        ({**_HD256, "causal": True},
         dict(use_clc_scheduler=True, is_static_persistent=False), None),
        ({**_HD256, "causal": True, "has_cu_seqlens_q": True},
         dict(use_clc_scheduler=True, is_static_persistent=False), "CLC"),
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
        # ex2 emulation: any non-negative SM100 setting; other architectures carry (0, 0).
        ({**_HD128}, dict(ex2_emu_freq=0, ex2_emu_start_frg=0), None),
        ({**_HD128}, dict(ex2_emu_freq=0, ex2_emu_start_frg=1), "must be 0 when ex2_emu_freq is 0"),
        ({**_HD128, "device_arch": 103}, dict(ex2_emu_freq=8, ex2_emu_start_frg=2), None),
        ({**_HD128}, dict(ex2_emu_freq=-1), "must be >= 0"),
        ({**_HD128}, dict(ex2_emu_start_frg=-1), "must be >= 0"),
        ({"device_arch": 90, **_HD128}, dict(ex2_emu_freq=16), "SM100-only options"),
        ({"device_arch": 80}, dict(ex2_emu_start_frg=1), "SM100-only options"),
        # SMEM layout: any layout that fits is the caller's choice.
        ({**_HD128}, dict(kv_stage=2), None),
        ({**_HD128}, dict(kv_stage=1), r"kv_stage must be in \[2, 32\], got 1"),
        ({**_HD128}, dict(kv_stage=33), r"kv_stage must be in \[2, 32\], got 33"),
        ({**_HD128}, dict(kv_stage=7), "needs 248832 bytes of shared memory"),
        # A layout chosen for 2CTA K/V halves does not fit 1CTA stages.
        ({**_HD128}, dict(use_2cta_instrs=False), r"kv_stage=6\) needs"),
        ({"device_arch": 90, **_HD128}, dict(kv_stage=2), "SM100-only options"),
        # hd192 Q tiles leave no room for a separate sO.
        ({**_HD192}, dict(overlap_sO_sQ=False), "needs 289792 bytes"),
        # SplitKV fp32 partials double sO for hdim_v 128.
        ({**_HD128, "requested_num_splits": 3}, dict(overlap_sO_sQ=False), "needs 297984 bytes"),
        # The uneven [K, V, K] offset, tile_n * 32 elements, must be 1024-element aligned.
        ({**_HD192, "max_seqlen_q": 128}, dict(tile_n=112), "tile_n divisible by 32"),
        # 2CTA never uses the uneven layout, so it has no offset to align.
        ({**_HD192}, dict(tile_n=112, kv_stage=3), None),
    ],
)
def test_explicit_config_validation(changes, config_changes, match):
    inputs = make_inputs(**changes)

    with pytest.raises(ValueError, match=match) if match else nullcontext():
        validate_fwd_config(replace(select_fwd_config(inputs), **config_changes), inputs)


def test_derive_fwd_config_rederives_unset_ex2_emu():
    # hd192 emulates exp2 only with 2CTA; PackGQA then changes the non-causal frequency.
    inputs = make_inputs(**_HD192)
    selected = select_fwd_config(inputs)
    assert selected.use_2cta_instrs
    assert (selected.ex2_emu_freq, selected.ex2_emu_start_frg) == (16, 0)

    one_cta = derive_fwd_config(selected, inputs, use_2cta_instrs=False)
    assert (one_cta.ex2_emu_freq, one_cta.ex2_emu_start_frg) == (0, 0)
    gqa_varlen = make_inputs(num_heads_kv=4, has_seqused_q=True, **_HD128)
    unpacked = select_fwd_config(gqa_varlen._replace(pack_gqa=False))
    packed = derive_fwd_config(unpacked, gqa_varlen, pack_gqa=True)
    assert (unpacked.ex2_emu_freq, packed.ex2_emu_freq) == (16, 32)
    # Explicit fields are kept, and the result is validated.
    assert derive_fwd_config(selected, inputs, ex2_emu_freq=8).ex2_emu_freq == 8
    with pytest.raises(ValueError, match="must be >= 0"):
        derive_fwd_config(selected, inputs, ex2_emu_start_frg=-1)


def test_smem_model_matches_measured_kernel_allocations():
    inputs = make_inputs(**_HD128)
    config = select_fwd_config(inputs)
    # SharedStorage.size_in_bytes() of this kernel: the selected layout fills SMEM exactly.
    assert (config.q_stage, config.use_2cta_instrs, config.overlap_sO_sQ, config.kv_stage) == (
        2, True, False, 6
    )
    assert sm100_fwd_smem_bytes(config, inputs) == SM100_SMEM_BYTES == 232448
    # Observed hd192 SMEM allocation at launch.
    hd192 = make_inputs(**_HD192, max_seqlen_q=128)
    too_wide = replace(select_fwd_config(hd192), tile_n=192, q_stage=1, use_2cta_instrs=False)
    with pytest.raises(ValueError, match="needs 247808 bytes of shared memory"):
        validate_fwd_config(too_wide, hd192)


def test_sm107_layouts_may_exceed_the_sm100_smem_capacity():
    # hd256 with 8 K/V stages places operands above 256 KiB: sm_107 only.
    inputs = make_inputs(**_HD256, causal=True, device_arch=107)
    selected = select_fwd_config(inputs)
    assert selected.kv_stage < 8  # selection keeps the 224 KiB budget
    deep = derive_fwd_config(selected, inputs, kv_stage=8)
    assert SM100_SMEM_BYTES < sm100_fwd_smem_bytes(deep, inputs) <= 327 * 1024
    sm100 = inputs._replace(device_arch=100)
    with pytest.raises(ValueError, match="SM100 provides 232448"):
        validate_fwd_config(replace(deep, registers=select_fwd_config(sm100).registers), sm100)


def test_2cta_hd192_three_kv_stages_use_the_even_layout():
    # The [K, V, K] stage-1 offset assumes full K/V tiles; 2CTA halves them per CTA, so V at
    # stage 1 would overlap K at stage 2.
    inputs = make_inputs(**_HD192)
    selected = select_fwd_config(inputs)
    assert selected.use_2cta_instrs
    config = replace(selected, kv_stage=3)
    validate_fwd_config(config, inputs)

    assert not sm100_uneven_kv_smem(192, 128, 3, use_2cta_instrs=True)
    assert sm100_uneven_kv_smem(192, 128, 3, use_2cta_instrs=False)
    # Each 2CTA stage holds a full per-CTA K tile: 128 x 192 x 2 bytes / 2.
    stage_bytes = sm100_fwd_smem_bytes(config, inputs) - sm100_fwd_smem_bytes(
        replace(config, kv_stage=2), inputs
    )
    assert stage_bytes == 128 * 192
    assert sm100_fwd_smem_bytes(config, inputs) <= SM100_SMEM_BYTES


def test_derive_fwd_config_rederives_unset_layout_and_registers():
    inputs = make_inputs(**_HD192)
    selected = select_fwd_config(inputs)
    assert selected.q_stage == 2

    derived = derive_fwd_config(selected, inputs, q_stage=1)

    layout = select_sm100_smem_layout(
        inputs, tile_m=128, tile_n=selected.tile_n, q_stage=1, num_splits=1,
        use_2cta_instrs=selected.use_2cta_instrs,
        is_static_persistent=selected.is_static_persistent,
        use_clc_scheduler=selected.use_clc_scheduler,
    )
    registers = select_registers(
        inputs, tile_m=128, tile_n=selected.tile_n, q_stage=1, num_splits=1,
        use_2cta_instrs=selected.use_2cta_instrs, use_clc_scheduler=selected.use_clc_scheduler,
    )
    assert (derived.overlap_sO_sQ, derived.kv_stage) == layout
    assert derived.registers == registers
    assert (derived.kv_stage, derived.registers) != (selected.kv_stage, selected.registers)
    assert derived == replace(selected, q_stage=1, registers=registers, kv_stage=layout[1])
    # Explicit fields are kept, and the result is validated.
    assert derive_fwd_config(selected, inputs, q_stage=1, kv_stage=3).kv_stage == 3
    with pytest.raises(ValueError, match="kv_stage must be in"):
        derive_fwd_config(selected, inputs, kv_stage=1)


def test_derive_fwd_config_sizes_kv_stage_for_a_requested_overlap():
    # hd128 2CTA keeps Q and O apart; aliasing them frees room for more KV stages.
    inputs = make_inputs(**_HD128)
    selected = select_fwd_config(inputs)
    assert selected.use_2cta_instrs and not selected.overlap_sO_sQ

    derived = derive_fwd_config(selected, inputs, overlap_sO_sQ=True)

    assert derived.overlap_sO_sQ
    assert derived.kv_stage > selected.kv_stage
    assert derived.kv_stage == select_sm100_smem_layout(
        inputs, tile_m=128, tile_n=selected.tile_n, q_stage=selected.q_stage, num_splits=1,
        use_2cta_instrs=True, is_static_persistent=selected.is_static_persistent,
        use_clc_scheduler=selected.use_clc_scheduler, overlap_sO_sQ=True,
    )[1]


_DENSE_CAUSAL_103 = dict(device_arch=103, causal=True, **_HD128)
_HD256_CAUSAL_100 = dict(device_arch=100, causal=True, **_HD256)
_DEFAULT_HD64 = Sm100Registers(softmax=200, correction=64, other=48)


@pytest.mark.parametrize(
    ("changes", "clc", "registers"),
    [
        (_DENSE_CAUSAL_103, True, None),
        ({**_DENSE_CAUSAL_103, "causal": False}, False, None),
        ({**_DENSE_CAUSAL_103, "has_cu_seqlens_k": True}, False, None),
        ({**_DENSE_CAUSAL_103, "device_arch": 90}, False, None),
        # BF16 hd64 measured slower on SM103 but faster on SM100.
        ({"device_arch": 103, "causal": True}, False, None),
        ({"device_arch": 100, "causal": True}, True, None),
        # The default split spills once persistent; this CLC cell names its own.
        ({"device_arch": 103, "causal": True, "dtype": "torch.float16"}, True,
         Sm100Registers(softmax=184, correction=64, other=80)),
        # Measured defaults only: varlen GQA stays off CLC by default.
        ({"device_arch": 103, "causal": True, "dtype": "torch.float16", "num_heads_kv": 4,
          "pack_gqa": True, "has_cu_seqlens_k": True}, False, _DEFAULT_HD64),
        # B200 FP16 hd192 measured no better with a CLC split, so it keeps the default.
        ({"device_arch": 100, "causal": True, "dtype": "torch.float16", **_HD192}, True,
         Sm100Registers(softmax=192, correction=72, other=56)),
        # hd256 2CTA needs 32 (batch, launched head) slices, on sm_100 only.
        (_HD256_CAUSAL_100, True, None),
        ({**_HD256_CAUSAL_100, "dtype": "torch.float16"}, True, None),
        ({**_HD256_CAUSAL_100, "num_heads": 12, "num_heads_kv": 12}, False, None),
        ({**_HD256_CAUSAL_100, "batch_size": 1, "num_heads": 32, "num_heads_kv": 32}, True, None),
        # PackGQA launches KV heads: 2 x 4 slices.
        ({**_HD256_CAUSAL_100, "num_heads_kv": 4, "pack_gqa": True}, False, None),
        ({**_HD256_CAUSAL_100, "device_arch": 103}, False, None),
        ({**_HD256_CAUSAL_100, "disable_2cta": True}, False, None),
    ],
)
def test_causal_clc_cells(changes, clc, registers):
    config = select_fwd_config(make_inputs(**changes))

    assert config.use_clc_scheduler is clc
    if registers is not None:
        assert config.registers == registers


@pytest.mark.parametrize(
    ("changes", "use_clc_scheduler", "registers"),
    [
        # Turning CLC off on a cell with a CLC split restores the default split.
        ({"device_arch": 103, "causal": True, "dtype": "torch.float16"}, False, _DEFAULT_HD64),
        # CLC outside the measured dense causal class keeps the default split.
        ({"device_arch": 103, "causal": True, "dtype": "torch.float16", "num_heads_kv": 4,
          "pack_gqa": True, "has_cu_seqlens_k": True}, True, _DEFAULT_HD64),
        ({"device_arch": 103}, True, _DEFAULT_HD64),
    ],
)
def test_explicit_clc_configs(changes, use_clc_scheduler, registers):
    inputs = make_inputs(**changes)
    selected = select_fwd_config(inputs)
    assert selected.use_clc_scheduler is not use_clc_scheduler

    # derive_fwd_config re-derives the register split for the scheduler.
    config = derive_fwd_config(
        selected, inputs, use_clc_scheduler=use_clc_scheduler, is_static_persistent=False
    )
    assert config.registers == registers


# 3 x 12 heads x 8 two-stage tiles = 288 tiles, within 2.25 x 132 SMs.
_FEW_TILES_CAUSAL_103 = dict(
    _DENSE_CAUSAL_103, batch_size=3, num_heads=12, num_heads_kv=12, max_seqlen_q=2048,
    max_seqlen_k=2048, total_q=6144, total_k=6144,
)


@pytest.mark.parametrize(
    ("changes", "lpt_global"),
    [
        (_FEW_TILES_CAUSAL_103, True),
        # 19 x 16 = 304 tiles exceed 2.25 x 132 = 297.
        ({**_FEW_TILES_CAUSAL_103, "batch_size": 1, "num_heads": 19, "num_heads_kv": 19,
          "max_seqlen_q": 4096, "max_seqlen_k": 4096}, False),
        ({**_FEW_TILES_CAUSAL_103, "num_sms": 148, "batch_size": 1, "num_heads": 19,
          "num_heads_kv": 19, "max_seqlen_q": 4096, "max_seqlen_k": 4096}, True),
        # Only on CLC: BF16 hd64 stays on the grid on SM103.
        ({**_FEW_TILES_CAUSAL_103, "head_dim": 64, "head_dim_v": 64}, False),
        ({**_FEW_TILES_CAUSAL_103, "causal": False}, False),
        # Measured on dense K; an explicit CLC config over varlen K keeps the per-section order.
        ({**_FEW_TILES_CAUSAL_103, "has_cu_seqlens_k": True}, False),
    ],
)
def test_lpt_global_region(changes, lpt_global):
    inputs = make_inputs(**changes)
    config = select_fwd_config(inputs)

    assert config.lpt_global is lpt_global
    if lpt_global:
        assert config.use_clc_scheduler
        # derive_fwd_config re-derives it with the scheduler.
        assert not derive_fwd_config(
            config, inputs, use_clc_scheduler=False, is_static_persistent=False
        ).lpt_global
        with pytest.raises(ValueError, match="lpt_global needs"):
            validate_fwd_config(replace(config, use_clc_scheduler=False), inputs)


# 65536 batches put the CLC grid's z extent past CUDA's 65535 limit.
_CLC_GRID_Z_OVERFLOW = dict(
    device_arch=100, causal=True, batch_size=65536, total_q=65536 * 129, total_k=65536 * 129,
    max_seqlen_q=129, max_seqlen_k=129,
)


@pytest.mark.parametrize(
    ("changes", "clc"),
    [
        (_CLC_GRID_Z_OVERFLOW, False),
        ({**_CLC_GRID_Z_OVERFLOW, "batch_size": 65535}, True),
        # Grid y holds the launched heads: KV heads when Q heads are packed.
        ({**_CLC_GRID_Z_OVERFLOW, "batch_size": 1, "num_heads": 65536, "num_heads_kv": 65536},
         False),
    ],
)
def test_clc_respects_cuda_grid_limits(changes, clc):
    inputs = make_inputs(**changes)
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler is clc
    if not clc:
        with pytest.raises(ValueError, match="CLC is not supported"):
            derive_fwd_config(config, inputs, use_clc_scheduler=True, is_static_persistent=False)


def test_softmax_stats_release_order_follows_the_measured_table():
    # BF16 hd64 non-causal measured faster with cross-stage on SM100, same-stage on SM103.
    inputs = make_inputs(device_arch=100)
    config = select_fwd_config(inputs)
    assert config.sm_stats_cross_stage
    assert not select_fwd_config(inputs._replace(device_arch=103)).sm_stats_cross_stage
    assert not select_fwd_config(inputs._replace(dtype="torch.float8_e4m3fn",
                                                 head_dim=128, head_dim_v=128)).sm_stats_cross_stage
    # Both orders are correct, so an explicit config may pick either.
    validate_fwd_config(replace(config, sm_stats_cross_stage=False), inputs)
    # S ping-pong keeps its own order; other architectures have no such choice.
    decode = inputs._replace(max_seqlen_q=1)
    decode_config = select_fwd_config(decode)
    assert decode_config.use_s_ping_pong and not decode_config.sm_stats_cross_stage
    with pytest.raises(ValueError, match="S ping-pong"):
        validate_fwd_config(replace(decode_config, sm_stats_cross_stage=True), decode)
    # Deriving a config re-derives the order for the changed S ping-pong choice.
    assert derive_fwd_config(decode_config, decode, use_s_ping_pong=False).sm_stats_cross_stage
    sm90 = inputs._replace(device_arch=90)
    with pytest.raises(ValueError, match="SM100-only options"):
        validate_fwd_config(replace(select_fwd_config(sm90), sm_stats_cross_stage=True), sm90)


# Frozen copies of the register and SMEM policy the forward kernels applied before it moved
# into the selector (flash_fwd_sm100.py __init__ and the FP8 override in __call__;
# flash_fwd_sm90.py; FlashAttentionForwardSm100's SharedStorage layout).
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


def legacy_sm100_registers(dtype, hd_padded, causal, use_2cta, sm103, paged_non_tma, q_stage):
    fp8 = "float8" in dtype
    key = (use_2cta, causal, hd_padded, sm103)
    if hd_padded < 96:
        regs = (184, 64, 80) if paged_non_tma else (200, 64, 48)
    else:
        if not paged_non_tma and dtype == "torch.bfloat16" and key == (True, False, 128, False):
            softmax, correction = 192, 64  # BF16 2CTA hd128 override
        elif not paged_non_tma and key in _LEGACY_SM100_TUNING:
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


# Frozen copy of the exp2 emulation policy the SM100 forward kernel applied before it moved
# into the selector: __init__ default and SM100_TUNING, the FP8 override in __call__, and the
# packed-GQA adjustment after _setup_attributes. Keys are (use_2cta, causal, hd_padded, sm103).
_LEGACY_SM100_EX2_TUNING = {
    (True, False, 128, False): (10, 1),
    (False, True, 128, False): (16, 1),
    (True, False, 192, False): (16, 0),
    (False, True, 192, False): (32, 1),
    (True, False, 128, True): (0, 0),
    (False, True, 128, True): (0, 0),
    (True, False, 192, True): (0, 0),
    (False, True, 192, True): (0, 0),
    (True, False, 256, False): (14, 0),
    (True, True, 256, False): (14, 0),
}
_LEGACY_SM100_FP8_EX2_TUNING = {
    (True, False, 128, False): (10, 1),
    (False, True, 128, False): (8, 1),
}


def legacy_sm100_ex2_emu(fp8, hd_padded, causal, local, use_2cta, sm103, pack_gqa, varlen_q):
    key = (use_2cta, causal, hd_padded, sm103)
    enable = (
        hd_padded <= 128 or (hd_padded == 192 and use_2cta and not causal and not local)
    ) and not sm103
    tune = _LEGACY_SM100_EX2_TUNING.get(key)
    if tune is not None:
        enable = tune[0] > 0
    if fp8 and key in _LEGACY_SM100_FP8_EX2_TUNING:
        tune = _LEGACY_SM100_FP8_EX2_TUNING[key]
        enable = tune[0] > 0
    freq, start_frg = 0, 1 if tune is None else tune[1]
    if enable:
        freq = 16 if tune is None else tune[0]
        if pack_gqa and hd_padded > 64 and not causal and not local:
            freq = 32 if varlen_q else (10 if tune is None else tune[0])
    # The kernel ignored start_frg without emulation; the config encodes "off" as (0, 0).
    return (freq, start_frg) if freq else (0, 0)


def legacy_sm90_registers(tile_m, pack_gqa, qhead_per_kvhead, paged_non_tma):
    num_wg_mma = tile_m // 64
    regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[num_wg_mma]
    use_tma_q = not (pack_gqa and tile_m % qhead_per_kvhead != 0)
    if num_wg_mma == 2 and (not use_tma_q or paged_non_tma):
        regs = (224, 40)
    return Sm90Registers(*regs)


def legacy_sm100_smem_layout(
    element_bytes, hd, hdv, q_stage, tile_n, use_2cta, is_persistent, is_split_kv
):
    hd_padded, hdv_padded = -(-hd // 16) * 16, -(-hdv // 16) * 16
    overlap = (
        (hd_padded == 192 and hdv_padded >= 64)
        or (hd_padded == 256 and not (is_persistent and use_2cta))
        or (hdv_padded >= 128 and is_split_kv)
    )
    epi_hdv = 128 if is_split_kv and hdv == 256 else hdv_padded
    smem_q = q_stage * 128 * hd_padded * element_bytes
    smem_o = q_stage * 128 * epi_hdv * (4 if is_split_kv else 2)
    smem_q_o = max(smem_q, smem_o) if overlap else smem_q + smem_o
    smem_kv = max(tile_n * hd_padded, tile_n * hdv_padded) * element_bytes // (2 if use_2cta else 1)
    kv_stage = min((224 * 1024 - smem_q_o) // smem_kv, 32)
    # Deliberate fix: the uneven [K, V, K] layout behind the third stage is 1CTA-only.
    if hd_padded == 192 and hdv_padded == 128 and kv_stage == 2 and not use_2cta:
        kv_stage = 3
    return overlap, kv_stage


_ELEMENT_BYTES = {"torch.bfloat16": 2, "torch.float8_e4m3fn": 1}


def test_sm100_registers_match_the_legacy_kernel_policy():
    for device_arch, dtype, d, causal, use_2cta, page, q_stage in itertools.product(
        [100, 103, 110], ["torch.bfloat16", "torch.float8_e4m3fn"], [64, 80, 96, 128, 192, 256],
        [False, True], [False, True], [None, 64], [1, 2],
    ):
        inputs = make_inputs(device_arch=device_arch, dtype=dtype, head_dim=d, head_dim_v=d,
                             causal=causal, page_size=page)
        got = select_registers(inputs, tile_m=128, tile_n=128, q_stage=q_stage,
                               num_splits=1, use_2cta_instrs=use_2cta, use_clc_scheduler=False)
        expected = legacy_sm100_registers(
            dtype, inputs.head_dim_padded, causal, use_2cta,
            device_arch == 103, page is not None, q_stage,
        )
        assert got == expected, (device_arch, dtype, d, causal, use_2cta, page, q_stage)


def test_ex2_emu_matches_the_legacy_kernel_policy():
    masks = {"none": {}, "causal": {"causal": True}, "local": {"local": True}}
    varlen_q = {"dense": {}, "cu_seqlens_q": {"has_cu_seqlens_q": True}}
    for device_arch, dtype, (d, dv), mask, use_2cta, (qhead, pack_gqa), layout in itertools.product(
        [90, 100, 103], ["torch.bfloat16", "torch.float8_e4m3fn"],
        [(64, 64), (96, 96), (128, 128), (192, 128), (256, 256)],
        masks, [False, True], [(1, False), (3, False), (3, True)], varlen_q,
    ):
        inputs = make_inputs(
            device_arch=device_arch, dtype=dtype, head_dim=d, head_dim_v=dv,
            num_heads=4 * qhead, num_heads_kv=4, pack_gqa=pack_gqa, **masks[mask],
            **varlen_q[layout],
        )
        got = select_ex2_emu(inputs, use_2cta_instrs=use_2cta)
        expected = (0, 0) if device_arch == 90 else legacy_sm100_ex2_emu(
            "float8" in dtype, inputs.head_dim_padded, mask == "causal", mask == "local",
            use_2cta, device_arch == 103, pack_gqa, layout != "dense",
        )
        assert got == expected, (device_arch, dtype, d, mask, use_2cta, qhead, pack_gqa, layout)


def test_sm90_registers_match_the_legacy_kernel_policy():
    for tile_m, pack_gqa, qhead_per_kvhead, page in itertools.product(
        [64, 128, 192], [False, True], [1, 3, 4, 8], [None, 64]
    ):
        inputs = make_inputs(device_arch=90, num_heads=8 * qhead_per_kvhead, num_heads_kv=8,
                             pack_gqa=pack_gqa, page_size=page)
        got = select_registers(inputs, tile_m=tile_m, tile_n=128, q_stage=1,
                               num_splits=1, use_2cta_instrs=False, use_clc_scheduler=False)
        expected = legacy_sm90_registers(
            tile_m, pack_gqa, qhead_per_kvhead, page is not None
        )
        assert got == expected, (tile_m, pack_gqa, qhead_per_kvhead, page)


def test_sm100_smem_layout_matches_the_legacy_kernel_formula():
    head_dims = [(64, 64), (128, 128), (192, 128), (256, 256)]
    for (dtype, element_bytes), (hd, hdv), q_stage, tile_n, use_2cta, persistent, splits in (
        itertools.product(
            _ELEMENT_BYTES.items(), head_dims, [1, 2], [64, 128, 192], [False, True],
            [False, True], [1, 3],
        )
    ):
        inputs = make_inputs(dtype=dtype, head_dim=hd, head_dim_v=hdv)
        got = select_sm100_smem_layout(
            inputs, tile_m=128, tile_n=tile_n, q_stage=q_stage, num_splits=splits,
            use_2cta_instrs=use_2cta, is_static_persistent=persistent, use_clc_scheduler=False,
        )
        expected = legacy_sm100_smem_layout(
            element_bytes, hd, hdv, q_stage, tile_n, use_2cta, persistent, splits > 1
        )
        assert got == expected, (dtype, hd, hdv, q_stage, tile_n, use_2cta, persistent, splits)


_WIRING_CASES = {
    "sm100_bf16_2cta": dict(**_HD128),
    "sm103_fp8_causal": dict(device_arch=103, dtype="torch.float8_e4m3fn", causal=True, **_HD128),
    "sm100_hd256_persistent": dict(**_HD256),
    "sm100_diff_head_split": {**_DECODE, **_HD192, "requested_num_splits": 4},
    "sm90": dict(device_arch=90, **_HD128),
}


@pytest.mark.parametrize("case", _WIRING_CASES)
def test_selected_configs_carry_the_legacy_policy(case):
    # Wiring: select_fwd_config must pass its final tile, stage, split, 2CTA, and scheduler.
    inputs = make_inputs(**_WIRING_CASES[case])
    config = select_fwd_config(inputs)
    if inputs.device_capacity == 9:
        assert config.registers == legacy_sm90_registers(config.tile_m, False, 1, False)
        assert (config.ex2_emu_freq, config.ex2_emu_start_frg) == (0, 0)
        assert (config.overlap_sO_sQ, config.kv_stage) == (False, 0)
        return
    fp8, sm103 = "float8" in inputs.dtype, inputs.device_arch == 103
    is_persistent = (
        config.is_static_persistent or config.use_clc_scheduler or inputs.has_varlen_q_semaphore
    )
    expected_layout = legacy_sm100_smem_layout(
        _ELEMENT_BYTES[inputs.dtype], inputs.head_dim, inputs.head_dim_v, config.q_stage,
        config.tile_n, config.use_2cta_instrs, is_persistent, config.num_splits > 1,
    )
    assert (config.overlap_sO_sQ, config.kv_stage) == expected_layout, config
    expected_registers = legacy_sm100_registers(
        inputs.dtype, inputs.head_dim_padded, inputs.causal, config.use_2cta_instrs, sm103, False,
        config.q_stage,
    )
    assert config.registers == expected_registers, config
    expected_ex2 = legacy_sm100_ex2_emu(
        fp8, inputs.head_dim_padded, inputs.causal, False, config.use_2cta_instrs, sm103,
        config.pack_gqa, inputs.is_varlen_q,
    )
    assert (config.ex2_emu_freq, config.ex2_emu_start_frg) == expected_ex2, config


def select_untuned(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Select without measured overrides or cached results."""
    with mock.patch.object(
        config_module, "select_tuned_sm100_overrides", return_value=TunedSm100Overrides()
    ):
        return select_fwd_config.__wrapped__(inputs)


_TUNED_RULES = {
    # rule -> (matching inputs, required field changes)
    "sm103_d64_nonpersistent": (
        make_inputs(device_arch=103, max_seqlen_k=8192),
        {"is_static_persistent"},
    ),
    "sm103_balanced_varlen_mha_clc": (
        make_inputs(device_arch=103, batch_size=4, total_q=10240, max_seqlen_q=6400,
                    has_cu_seqlens_q=True, has_cu_seqlens_k=True),
        {"use_clc_scheduler"},
    ),
    "sm103_high_head_varlen_clc": (
        make_inputs(device_arch=103, num_heads=28, num_heads_kv=4, pack_gqa=True, batch_size=3,
                    total_q=4096, total_k=4096, max_seqlen_q=2048, max_seqlen_k=2048,
                    causal=True, has_cu_seqlens_q=True, has_cu_seqlens_k=True),
        {"use_clc_scheduler"},
    ),
    "sm103_dense_short_k_clc": (
        make_inputs(device_arch=103, num_heads=32, num_heads_kv=8, pack_gqa=True, batch_size=32,
                    total_q=32 * 64, total_k=32 * 1024, max_seqlen_q=64, max_seqlen_k=1024,
                    causal=True),
        {"use_clc_scheduler"},
    ),
}
# Tuned rules cover only plain output-only BF16 attention on 128x128 nonsplit tiles.
_NOT_PLAIN_BF16 = [
    {"dtype": "torch.float16"}, {"page_size": 128}, {"use_block_sparsity": True},
    {"has_score_mod": True}, {"has_mask_mod": True}, {"has_learnable_sink": True},
    {"has_lse": True}, {"requested_tile_n": 64}, {"requested_num_splits": 2},
    {"local": True, "window_size_left": 256, "window_size_right": 0},
]


@pytest.mark.parametrize(
    ("rule", "changes", "fires"),
    [
        *[(rule, {}, True) for rule in _TUNED_RULES],
        # One plain-BF16 gate precedes every rule.
        *[("sm103_d64_nonpersistent", changes, False) for changes in _NOT_PLAIN_BF16],
        ("sm103_d64_nonpersistent", {"max_seqlen_k": 4096}, True),
        ("sm103_d64_nonpersistent", {"max_seqlen_k": 3072}, False),
        ("sm103_d64_nonpersistent", {"head_dim": 96, "head_dim_v": 96}, False),
        ("sm103_d64_nonpersistent", {"num_heads_kv": 2, "pack_gqa": False}, False),
        ("sm103_balanced_varlen_mha_clc", {"total_q": 10239}, False),
        ("sm103_balanced_varlen_mha_clc", {"max_seqlen_q": 6401}, False),
        ("sm103_balanced_varlen_mha_clc", {"total_k": 6553}, False),
        ("sm103_balanced_varlen_mha_clc", {"batch_size": 3}, False),
        ("sm103_balanced_varlen_mha_clc", {"batch_size": 25}, False),
        ("sm103_balanced_varlen_mha_clc", {"has_cu_seqlens_k": False}, False),
        ("sm103_balanced_varlen_mha_clc", {"has_seqused_k": True}, False),
        ("sm103_balanced_varlen_mha_clc", {"has_caller_scheduler_metadata": True}, False),
        ("sm103_high_head_varlen_clc", {"causal": False}, True),
        ("sm103_high_head_varlen_clc", {"total_q": 4095}, False),
        ("sm103_high_head_varlen_clc", {"batch_size": 2}, False),
        ("sm103_high_head_varlen_clc", {"batch_size": 65}, False),
        ("sm103_high_head_varlen_clc", {"num_heads": 23, "num_heads_kv": 1}, False),
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
    inputs, tuned_fields = _TUNED_RULES[rule]
    inputs = inputs._replace(**changes)

    tuned, untuned = asdict(select_fwd_config(inputs)), asdict(select_untuned(inputs))
    changed = {name for name in tuned if tuned[name] != untuned[name]}

    assert tuned_fields <= changed if fires else not changed

def test_mxfp8_qk_selects_a_1cta_unpacked_config():
    inputs = make_inputs(dtype="torch.float8_e4m3fn", head_dim=128, head_dim_v=128,
                         num_heads=16, num_heads_kv=4, mxfp8_path="qk")
    config = select_fwd_config(inputs)
    assert not (config.pack_gqa or config.use_2cta_instrs or config.use_s_ping_pong)
    # V and P stay BF16, so the BF16 register split and exp2 tuning apply rather than the FP8 ones.
    hd64 = inputs._replace(head_dim=64, head_dim_v=64)
    assert select_fwd_config(hd64).registers == Sm100Registers(softmax=200, correction=64, other=48)
    causal = select_fwd_config(inputs._replace(causal=True))
    assert (causal.ex2_emu_freq, causal.ex2_emu_start_frg) == (16, 1)
    with pytest.raises(ValueError, match="MXFP8 forward requires 128 x 128 tiles"):
        validate_fwd_config(replace(config, tile_n=64), inputs)
    decode = inputs._replace(max_seqlen_q=1, total_q=2)
    assert not select_fwd_config(decode).use_s_ping_pong
    # SplitKV would raise for long-KV decode, so automatic selection keeps one split.
    assert select_fwd_config(decode._replace(requested_num_splits=None)).num_splits == 1
    with pytest.raises(ValueError, match="MXFP8 forward requires SM10x"):
        select_fwd_config(inputs._replace(device_arch=90))


def test_mxfp8_hd256_runs_on_the_2cta_s_ping_pong_schedule():
    qk = make_inputs(dtype="torch.float8_e4m3fn", **_HD256, num_heads_kv=4, mxfp8_path="qk")
    # The 2CTA / S ping-pong knobs (FA_DISABLE_2CTA, the CUDA 12 workaround, FA_DISABLE_S_PING_PONG)
    # have no other hd256 MXFP8 schedule to fall back to.
    knobs = {"disable_2cta": True, "disable_s_ping_pong": True}
    for changes in ({}, {"causal": True}, {"device_arch": 103}, {"max_seqlen_q": 1, "total_q": 2},
                    knobs):
        config = select_fwd_config(qk._replace(**changes))
        assert config.use_2cta_instrs and config.use_s_ping_pong and config.q_stage == 1, changes
        assert not config.pack_gqa, changes
    # P stays BF16, so the BF16 hd256 registers and exp2 tuning apply.
    config = select_fwd_config(qk)
    bf16 = select_fwd_config(qk._replace(dtype="torch.bfloat16", mxfp8_path=None))
    assert config.registers == bf16.registers
    assert (config.ex2_emu_freq, config.ex2_emu_start_frg) == (
        bf16.ex2_emu_freq, bf16.ex2_emu_start_frg,
    )
    for changes in ({"use_2cta_instrs": False}, {"use_s_ping_pong": False}):
        with pytest.raises(ValueError, match="hd256 MXFP8 forward requires 2CTA instructions"):
            validate_fwd_config(replace(config, **changes), qk)


def test_mxfp8_qk_policy_follows_its_own_rows():
    causal_hd64 = make_inputs(dtype="torch.float8_e4m3fn", causal=True, mxfp8_path="qk")
    # Causal CLC: the "mxfp8_qk" rows, not the FP8 ones. hd64 measured faster on SM100 only.
    assert select_fwd_config(causal_hd64).use_clc_scheduler
    assert not select_fwd_config(causal_hd64._replace(device_arch=103)).use_clc_scheduler
    assert select_fwd_config(causal_hd64._replace(device_arch=103, **_HD128)).use_clc_scheduler
    # Softmax stats release: hd64 causal keeps cross-stage on SM103.
    assert not select_fwd_config(causal_hd64).sm_stats_cross_stage
    assert select_fwd_config(causal_hd64._replace(device_arch=103)).sm_stats_cross_stage


@pytest.mark.parametrize("mxfp8_path", ["qk", "pv", "pv_fast"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("device_arch", [100, 103, 107])
def test_mxfp8_kv_stages_fit_the_scales(mxfp8_path, head_dim, causal, device_arch):
    inputs = make_inputs(dtype="torch.float8_e4m3fn", head_dim=head_dim, head_dim_v=head_dim,
                         causal=causal, device_arch=device_arch, mxfp8_path=mxfp8_path)
    config = select_fwd_config(inputs)
    # The kernel's previous K/V stage formula: e4m3 Q plus one 512-byte scale chunk per Q stage,
    # BF16 O, native PV's constant P-scale chunk, and per stage the larger of e4m3 K and V (BF16
    # for QK MXFP8) plus K's scale chunk.
    rows = config.q_stage * 128
    q_o = rows * head_dim + config.q_stage * 512 + rows * head_dim * 2
    q_o += 0 if mxfp8_path == "qk" else 512
    if mxfp8_path == "pv":  # all-ones tiles of the P row-sum MMAs
        if device_arch == 107:
            q_o += 2048 + (4096 if causal else 0) + 2048
        elif config.q_stage == 2:
            q_o += 2048 + 1024
    bf16_v = mxfp8_path == "qk" or causal  # causal native PV: BF16 diagonal V blocks
    kv_stage_bytes = 128 * head_dim * (2 if bf16_v else 1) + 512
    # sm_107 causal hd128 native PV takes a 288 KiB budget (SM107_MXFP8_PV_TUNING).
    sm107_causal_hd128_pv = (device_arch, mxfp8_path, causal, head_dim) in (
        (107, "pv", True, 128), (107, "pv_fast", True, 128)
    )
    budget_kib = 288 if sm107_causal_hd128_pv else 224
    assert config.kv_stage == min((budget_kib * 1024 - q_o) // kv_stage_bytes, 32)
    for has_lse in (False, True):
        smem_bytes = sm100_fwd_smem_bytes(config, inputs._replace(has_lse=has_lse))
        assert smem_bytes <= sm100_smem_capacity(device_arch)


def test_mxfp8_pv_register_splits():
    pv = make_inputs(dtype="torch.float8_e4m3fn", head_dim=128, head_dim_v=128, mxfp8_path="pv")

    def registers(**changes):
        config = select_fwd_config(pv._replace(**changes))
        assert not (config.use_2cta_instrs or config.use_s_ping_pong)
        return astuple(config.registers)

    # sm_100/sm_103 with two Q stages: the ones MMA adds MMA-warp pressure at hd128.
    assert registers() == (176, 64, 96)
    # Otherwise the MMA warp needs more than the hd128 FP8 default leaves it.
    assert registers(max_seqlen_q=128, total_q=256) == (184, 80, 64)
    assert registers(head_dim=64, head_dim_v=64) == (192, 64, 64)
    assert registers(device_arch=103, max_seqlen_q=128, total_q=256) == (168, 96, 80)
    # sm_107 has dedicated scale columns and its own "exact" table.
    assert registers(device_arch=107) == (192, 80, 48)
    assert registers(device_arch=107, head_dim=64, head_dim_v=64) == (176, 96, 64)
    assert registers(device_arch=107, head_dim=64, head_dim_v=64, causal=True) == (184, 72, 72)
    # Causal native PV without CLC runs a single-tile scheduler, so sm_103 keeps the sm_100
    # split.
    causal_103 = pv._replace(device_arch=103, causal=True)
    no_clc = derive_fwd_config(select_fwd_config(causal_103), causal_103, use_clc_scheduler=False)
    assert astuple(no_clc.registers) == (176, 80, 80)


_MXFP8_CAUSAL = dict(dtype="torch.float8_e4m3fn", causal=True)
_MXFP8_CLC_HD128 = Sm100Registers(softmax=168, correction=88, other=88)
_MXFP8_CLC_HD64 = Sm100Registers(softmax=184, correction=64, other=80)


@pytest.mark.parametrize(
    ("changes", "clc", "registers"),
    [
        # QK-only MXFP8 keeps its own rows, all with the default split.
        ({"mxfp8_path": "qk", "device_arch": 100}, True, None),
        ({"mxfp8_path": "qk", "device_arch": 103}, False, None),
        ({"mxfp8_path": "qk", "device_arch": 103, **_HD128}, True, None),
        # Native PV's default splits spill once persistent; those cells name their own.
        ({"mxfp8_path": "pv", "device_arch": 100, **_HD128}, True, _MXFP8_CLC_HD128),
        ({"mxfp8_path": "pv", "device_arch": 103, **_HD128}, True, _MXFP8_CLC_HD128),
        ({"mxfp8_path": "pv", "device_arch": 107, **_HD128}, True, None),
        ({"mxfp8_path": "pv", "device_arch": 100}, True, None),
        ({"mxfp8_path": "pv", "device_arch": 103}, True, _MXFP8_CLC_HD64),
        ({"mxfp8_path": "pv", "device_arch": 107}, False, None),
        ({"mxfp8_path": "pv_fast", "device_arch": 103, **_HD128}, True, _MXFP8_CLC_HD64),
        ({"mxfp8_path": "pv_fast", "device_arch": 107, **_HD128}, True, None),
        ({"mxfp8_path": "pv_fast", "device_arch": 103}, True, None),
    ],
)
def test_causal_mxfp8_clc_cells(changes, clc, registers):
    inputs = make_inputs(**_MXFP8_CAUSAL, **changes)
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler is clc
    if clc:
        # A cell without its own split keeps the one the nonpersistent grid uses.
        grid = derive_fwd_config(config, inputs, use_clc_scheduler=False)
        assert config.registers == (grid.registers if registers is None else registers)


def test_mxfp8_pv_keeps_hardware_exp2():
    # FP8 hd128 emulates exp2 on SM100; native PV must not, so PV consumes the reference P.
    pv = make_inputs(dtype="torch.float8_e4m3fn", head_dim=128, head_dim_v=128, mxfp8_path="pv")
    fp8 = select_fwd_config(pv._replace(mxfp8_path=None, pack_gqa=False, disable_2cta=True))
    assert fp8.ex2_emu_freq > 0
    config = select_fwd_config(pv)
    assert (config.ex2_emu_freq, config.ex2_emu_start_frg) == (0, 0)


def test_mxfp8_pv_fast_policy():
    fast = make_inputs(dtype="torch.float8_e4m3fn", head_dim=128, head_dim_v=128,
                       mxfp8_path="pv_fast")
    # sm_100 "fast" has its own spill-free splits; sm_103 keeps the native-PV one.
    assert astuple(select_fwd_config(fast).registers) == (176, 64, 96)
    assert astuple(select_fwd_config(fast._replace(device_arch=103)).registers) == (168, 96, 80)
    assert astuple(select_fwd_config(fast._replace(device_arch=107)).registers) == (168, 96, 80)
    # Causal "fast" hd128 runs CLC with its own split on sm_100 and sm_103.
    causal_inputs = fast._replace(causal=True)
    causal = select_fwd_config(causal_inputs)
    assert causal.use_clc_scheduler and astuple(causal.registers) == (184, 64, 80)
    no_clc = derive_fwd_config(causal, causal_inputs, use_clc_scheduler=False)
    assert astuple(no_clc.registers) == (192, 64, 64)
    # "fast" has no quantized-P contract: sm_100 emulates exp2 for half the columns.
    assert (causal.ex2_emu_freq, causal.ex2_emu_start_frg) == (8, 1)
    assert select_fwd_config(fast._replace(device_arch=103)).ex2_emu_freq == 0
    assert select_fwd_config(fast).sm_stats_cross_stage
    assert not select_fwd_config(fast._replace(mxfp8_path="pv")).sm_stats_cross_stage


@pytest.mark.parametrize("mxfp8_path", ["pv", "pv_fast"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sm107_native_pv_schedule_follows_its_table(mxfp8_path, causal, head_dim):
    inputs = make_inputs(dtype="torch.float8_e4m3fn", head_dim=head_dim, head_dim_v=head_dim,
                         causal=causal, device_arch=107, mxfp8_path=mxfp8_path)
    config = select_fwd_config(inputs)
    tuning = config_module.SM107_MXFP8_PV_TUNING.get(
        ("fast" if mxfp8_path == "pv_fast" else "exact", head_dim, causal),
        config_module.Sm107PvTuning(),
    )
    schedule = (config.split_p_k64, config.sm_stats_mbarrier, config.sm_stats_early)
    assert schedule == (tuning.split_p, tuning.stats_mbarrier, tuning.early_stats)
    # Each field is an explicit choice on sm_107 native PV; derive keeps a set one.
    flipped = derive_fwd_config(config, inputs, split_p_k64=not config.split_p_k64)
    assert flipped.split_p_k64 != config.split_p_k64
    # Off that path the fields stay False and validation rejects them.
    for changes in ({"device_arch": 100}, {"mxfp8_path": "qk"}):
        other = inputs._replace(**changes)
        other_config = select_fwd_config(other)
        assert not (other_config.split_p_k64 or other_config.sm_stats_mbarrier
                    or other_config.sm_stats_early)
        with pytest.raises(ValueError, match="need sm_107 native MXFP8 PV"):
            validate_fwd_config(replace(other_config, sm_stats_early=True), other)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("device_arch", [100, 103, 107])
def test_mxfp8_hd256_kv_stages_fit_the_scales(causal, device_arch):
    inputs = make_inputs(dtype="torch.float8_e4m3fn", **_HD256, causal=causal,
                         device_arch=device_arch, mxfp8_path="qk")
    config = select_fwd_config(inputs)
    assert config.use_2cta_instrs and config.use_s_ping_pong and config.q_stage == 1
    # Persistent 2CTA hd256 keeps a separate sO; causal (single-tile LPT) aliases it with sQ.
    assert config.overlap_sO_sQ == (not config.is_static_persistent)
    # The kernel's previous K/V stage formula: e4m3 Q plus two 512-byte scale chunks, BF16 O, and
    # per stage half of a BF16 V tile plus all of K's scale chunks.
    q, o = 128 * 256 + 1024, 128 * 256 * 2
    q_o = max(q, o) if config.overlap_sO_sQ else q + o
    assert config.kv_stage == min((224 * 1024 - q_o) // (128 * 256 + 1024), 32)
    assert sm100_fwd_smem_bytes(config, inputs) <= sm100_smem_capacity(device_arch)
