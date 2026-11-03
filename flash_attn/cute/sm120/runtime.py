# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host-side planning of the SM12x forward: tile and warp configuration, SplitKV sizing
(static and per-batch), the decode specializations, and the kernel arguments the generic
FA4 interface passes through without interpreting them."""

import math
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from typing import Callable, Optional

import cutlass
import cutlass.utils as utils_basic
import torch

from flash_attn.cute.sm120.flash_fwd import (
    FlashAttentionForwardSm120,
)
from flash_attn.cute.sm120.policy import (
    LOW_HD_DECODE_LONG_MIN_TILES_PER_CTA,
    LOW_HD_DECODE_MAX_SPLITS,
    is_low_hd_paged_decode_tile,
    visible_decode_seqlen_k,
)
from flash_attn.cute.config import (
    _FWD_SM90_ONLY_NEUTRAL_FIELDS,
    _FWD_SM100_ONLY_NEUTRAL_FIELDS,
    FwdConfig,
    num_splits_heuristic,
)
from flash_attn.cute.cute_dsl_utils import get_num_sms_for_selection, to_cute_tensor
from flash_attn.cute.prepare_scheduler import FlashPrepareScheduler, SchedulerMetadataTensorsTorch

_DECODE_REFERENCE_CORE_GHZ = 2.4
_DECODE_POWER_OF_TWO_SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)

# Latency / bandwidth model of the hd 256 M16N64 paged-decode gather kernel: fitted constants
# at a reference clock, scaled by the device's SM count, clock and memory bus.
_DECODE_LATENCY_FIXED_REF_US = 4.31046205978645
_DECODE_LATENCY_PER_KV_TILE_REF_US = 1.3115656095100847
_DECODE_MEMORY_FIXED_REF_US = 4.532105547934724
_DECODE_MEMORY_SOL = 0.9415559971890943
_DECODE_UNDERFILL_TAU_CTAS_PER_CHANNEL = 0.741265436766481
_DECODE_COMBINE_FIXED_US = 3.654860520719416
_DECODE_COMBINE_SPLIT_TO_8_US = 0.015181182195831644
_DECODE_COMBINE_SPLIT_ABOVE_8_US = 0.12157753908563328
_DECODE_COMBINE_OUTPUT_CTA_US = 0.019185022429169515


@lru_cache(maxsize=None)
def l2_group_bytes(device_index: int) -> int:
    """K / V bytes per LPT head group: half the L2, as in the backward. The generic fixed
    50 MB budget exceeds the L2 of the smaller SM12x parts and thrashes it."""
    return torch.cuda.get_device_properties(device_index).L2_cache_size // 2


@lru_cache(maxsize=None)
def schedule_rates(device_index: int) -> tuple[float, float]:
    """(SM clock in Hz, streaming DRAM bytes / s) of the forward schedule model: the boost
    clock, and the memory transfer rate x bus width at the measured streaming efficiency.
    Discrete boards report half their GDDR transfer rate as the memory clock; integrated
    LPDDR (GB10) reports the transfer rate itself."""
    props = torch.cuda.get_device_properties(device_index)
    transfers = props.memory_clock_rate * 1e3 * (1 if props.is_integrated else 2)
    bandwidth = transfers * props.memory_bus_width / 8
    return props.clock_rate * 1e3, bandwidth * FlashAttentionForwardSm120._dram_efficiency


@dataclass(frozen=True)
class Sm120ForwardConfig:
    tile_m: int
    tile_n: int
    num_stages: int
    num_threads: int
    l2_group_bytes: int = 50 * 1024 * 1024
    lpt_order: bool = False

    @property
    def compile_key(self) -> tuple:
        """Return SM120 configuration state baked into generated code."""
        return (
            self.tile_m,
            self.tile_n,
            self.num_stages,
            self.num_threads,
            self.l2_group_bytes,
            self.lpt_order,
        )


@dataclass(frozen=True)
class Sm120ForwardPlan:
    """Compile- and launch-time decisions for one normalized forward call."""

    num_splits: int
    split_num_n_blocks: int
    direct_uniform_batch: bool
    split_qk_n: bool
    split_kv_blocks_per_cta: int
    transpose_qk_pv: bool
    # Each batch's split count comes from its device-side length (split_metadata); batches
    # left with one split write the output directly, so the combine skips them.
    dynamic_splits: bool = False
    # With dynamic_splits: the most KV blocks one split takes.
    split_grain: int = 0
    # With dynamic_splits: the grid holds one split per batch plus this many (compact split
    # layout), or num_splits per batch when None.
    extra_splits: Optional[int] = None

    @property
    def compile_key(self) -> tuple:
        """Return only specialization state baked into the generated kernel."""
        return (
            self.direct_uniform_batch,
            self.split_qk_n,
            self.split_kv_blocks_per_cta,
            self.transpose_qk_pv,
            self.extra_splits is not None,
        )


@dataclass(frozen=True)
class _DecodeHardware:
    num_sms: int
    memory_channels: int
    peak_memory_gbps: float
    core_clock_ghz: float
    is_integrated: bool


def _get_decode_hardware(device: torch.device, num_sms: int) -> _DecodeHardware:
    """Device memory / clock model with the SMs available to the launch context."""
    return replace(_get_device_decode_hardware(device), num_sms=num_sms)


@lru_cache(maxsize=None)
def _get_device_decode_hardware(device: torch.device) -> _DecodeHardware:
    properties = torch.cuda.get_device_properties(device)
    memory_bus_width = properties.memory_bus_width
    is_integrated = bool(getattr(properties, "is_integrated", False))
    # Discrete GPUs report the physical memory clock, while CUDA reports the
    # effective LPDDR data rate on integrated devices. Convert kHz and bus bits
    # to GB/s without doubling the already-effective integrated rate.
    data_rate_multiplier = 1 if is_integrated else 2
    peak_memory_gbps = (
        properties.memory_clock_rate * memory_bus_width * data_rate_multiplier / 8_000_000
    )
    return _DecodeHardware(
        num_sms=properties.multi_processor_count,
        memory_channels=max(1, memory_bus_width // 32),
        peak_memory_gbps=peak_memory_gbps,
        core_clock_ghz=properties.clock_rate / 1_000_000,
        is_integrated=is_integrated,
    )


def _normalize_num_splits(num_splits: int, num_n_blocks: int) -> int:
    if num_splits <= 1 or num_n_blocks <= 1:
        return 1
    requested_splits = min(num_splits, num_n_blocks)
    blocks_per_split = (num_n_blocks + requested_splits - 1) // requested_splits
    return (num_n_blocks + blocks_per_split - 1) // blocks_per_split


def _predict_decode_split_us(
    *,
    num_splits: int,
    num_n_blocks: int,
    total_mblocks: int,
    packed_q_rows: int,
    tile_m: int,
    tile_n: int,
    head_dim: int,
    head_dim_v: int,
    element_size: int,
    hardware: _DecodeHardware,
) -> float:
    num_m_blocks = (packed_q_rows + tile_m - 1) // tile_m
    batch_head_groups = total_mblocks // num_m_blocks
    main_ctas = total_mblocks * num_splits
    kv_tiles_per_cta = (num_n_blocks + num_splits - 1) // num_splits
    clock_scale = _DECODE_REFERENCE_CORE_GHZ / hardware.core_clock_ghz
    latency_main_us = (
        _DECODE_LATENCY_FIXED_REF_US + _DECODE_LATENCY_PER_KV_TILE_REF_US * kv_tiles_per_cta
    ) * clock_scale
    ctas_per_channel = main_ctas / hardware.memory_channels
    memory_fill = max(
        1e-6,
        1.0 - math.exp(-ctas_per_channel / _DECODE_UNDERFILL_TAU_CTAS_PER_CHANNEL),
    )
    logical_kv_bytes = (
        batch_head_groups * num_n_blocks * tile_n * (head_dim + head_dim_v) * element_size
    )
    theoretical_transfer_us = logical_kv_bytes / hardware.peak_memory_gbps / 1000.0
    memory_main_us = (
        _DECODE_MEMORY_FIXED_REF_US * clock_scale
        + theoretical_transfer_us / _DECODE_MEMORY_SOL / memory_fill
    )
    main_us = max(latency_main_us, memory_main_us)
    if num_splits == 1:
        return main_us
    return main_us + _decode_combine_us(num_splits, batch_head_groups * packed_q_rows, head_dim_v)


def _decode_combine_us(num_splits: int, output_rows: int, head_dim_v: int) -> float:
    combine_ctas = ((output_rows + 7) // 8) * ((head_dim_v + 127) // 128)
    return (
        _DECODE_COMBINE_FIXED_US
        + _DECODE_COMBINE_SPLIT_TO_8_US * min(num_splits, 8)
        + _DECODE_COMBINE_SPLIT_ABOVE_8_US * max(num_splits - 8, 0)
        + _DECODE_COMBINE_OUTPUT_CTA_US * combine_ctas
    )


@lru_cache(maxsize=1024)
def _split_past_partial_wave(
    *,
    num_splits: int,
    total_mblocks: int,
    packed_q_rows: int,
    num_n_blocks: int,
    tile_n: int,
    head_dim: int,
    head_dim_v: int,
    element_size: int,
    resident_ctas_per_sm: int,
    hardware: _DecodeHardware,
) -> int:
    """Split a one-m-block grid that spills past its resident waves.

    With one m block per batch-head, every CTA streams a whole KV range. A wave of CTAs takes
    the longer of one CTA's latency-bound KV loop and the wave's bytes at the memory
    bandwidth, so a tail wave of a few CTAs costs a whole CTA's KV loop while moving little.
    Splitting KV shortens every CTA's loop, but every split CTA writes an FP32 partial O and
    LSE that the combine reads back, and the combine pays per split.
    """
    slots = hardware.num_sms * resident_ctas_per_sm
    if total_mblocks * num_splits <= slots or num_n_blocks <= 4:
        return num_splits
    clock_scale = _DECODE_REFERENCE_CORE_GHZ / hardware.core_clock_ghz
    tile_bytes = tile_n * (head_dim + head_dim_v) * element_size
    partial_bytes = packed_q_rows * (head_dim_v + 1) * 4
    bytes_per_us = hardware.peak_memory_gbps * _DECODE_MEMORY_SOL * 1000.0
    output_rows = total_mblocks * packed_q_rows

    def predict_us(splits: int) -> float:
        tiles = -(-num_n_blocks // splits)
        cta_us = (
            _DECODE_LATENCY_FIXED_REF_US + _DECODE_LATENCY_PER_KV_TILE_REF_US * tiles
        ) * clock_scale
        cta_bytes = tiles * tile_bytes + (partial_bytes if splits > 1 else 0)
        full_waves, tail_ctas = divmod(total_mblocks * splits, slots)
        wave_us = max(cta_us, slots * cta_bytes / bytes_per_us)
        tail_us = max(cta_us, tail_ctas * cta_bytes / bytes_per_us) if tail_ctas else 0.0
        combine_us = (
            _decode_combine_us(splits, output_rows, head_dim_v)
            + total_mblocks * splits * partial_bytes / bytes_per_us
            if splits > 1
            else 0.0
        )
        return full_waves * wave_us + tail_us + combine_us

    candidates = {
        _normalize_num_splits(splits, num_n_blocks)
        for splits in range(num_splits, min(128, num_n_blocks) + 1)
    }
    return min(sorted(candidates), key=predict_us)


def _select_decode_num_splits(
    *,
    head_dim: int,
    head_dim_v: int,
    element_size: int,
    packed_q_rows: int,
    tile_m: int,
    tile_n: int,
    total_mblocks: int,
    num_n_blocks: int,
    hardware: _DecodeHardware,
) -> int:
    # At four or fewer N tiles, SplitKV's fixed combine launch costs more than
    # the remaining serialized work.
    if num_n_blocks <= 4:
        return 1

    max_one_wave_splits = min(
        128,
        num_n_blocks,
        max(1, hardware.num_sms // total_mblocks),
    )
    requested_candidates = (
        *_DECODE_POWER_OF_TWO_SPLITS,
        max_one_wave_splits,
    )
    candidates = {
        _normalize_num_splits(requested, num_n_blocks) for requested in requested_candidates
    }
    candidates = {
        splits for splits in candidates if splits == 1 or total_mblocks * splits <= hardware.num_sms
    }
    if hardware.is_integrated:
        if total_mblocks >= 3 * hardware.memory_channels:
            # Three unsplit CTAs per LPDDR channel already saturate the LPDDR path.
            return 1
        if num_n_blocks >= 64:
            if total_mblocks >= 2 * hardware.memory_channels:
                return 1
            target_main_ctas = (2 * hardware.num_sms + 2) // 3
            if total_mblocks == hardware.memory_channels:
                # One base CTA per channel needs a second wave to cover UMA
                # latency; retain the next balanced power-of-two split.
                max_integrated_splits = 1 << ((hardware.num_sms // total_mblocks - 1).bit_length())
                candidates.add(_normalize_num_splits(max_integrated_splits, num_n_blocks))
            elif total_mblocks < hardware.memory_channels:
                max_integrated_splits = max(
                    1,
                    (target_main_ctas + total_mblocks - 1) // total_mblocks,
                )
            else:
                max_integrated_splits = None
            if max_integrated_splits is not None:
                bounded_candidates = {
                    splits for splits in candidates if splits <= max_integrated_splits
                }
                if bounded_candidates:
                    candidates = bounded_candidates
    # A 160-thread hd 256 CTA has enough independent QK / PV work that two thirds of an SM
    # wave saturates the memory system; the model below underestimates sparser grids at
    # larger batch-head counts, so it only chooses above that floor.
    target_main_ctas = (2 * hardware.num_sms + 2) // 3
    fill_splits = min(
        max_one_wave_splits,
        max(1, (target_main_ctas + total_mblocks - 1) // total_mblocks),
    )
    min_fill_splits = _normalize_num_splits(fill_splits, num_n_blocks)
    filled_candidates = {splits for splits in candidates if splits >= min_fill_splits}
    if filled_candidates:
        candidates = filled_candidates
    return min(
        candidates,
        key=lambda splits: _predict_decode_split_us(
            num_splits=splits,
            num_n_blocks=num_n_blocks,
            total_mblocks=total_mblocks,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            element_size=element_size,
            hardware=hardware,
        ),
    )


def _select_low_hd_decode_num_splits(
    *,
    max_head_dim: int,
    packed_q_rows: int,
    total_mblocks: int,
    num_n_blocks: int,
    hardware: _DecodeHardware,
) -> int:
    """Overschedule the short-M low-HD decode kernel by bounded waves.

    One hardware wave leaves each CTA with too long a serial K loop for this
    one-warp kernel. For short KV, use the fewest splits that approximately
    fill one device-specific SM wave; more waves only add combine work. For
    longer KV, bound the launch by hardware waves and a minimum per-CTA KV
    grain. Once eight-way splitting leaves a long grain, prefer balanced
    power-of-two partitions over device-specific odd split counts.
    """
    if num_n_blocks <= 4 or total_mblocks <= 0:
        return 1
    num_sms = hardware.num_sms
    if total_mblocks >= num_sms and (num_n_blocks <= 8 or num_n_blocks >= 64):
        return 1
    if hardware.is_integrated and total_mblocks <= hardware.memory_channels // 2:
        # Small batches need enough independent CTAs to cover UMA latency.
        # Two thirds of the compact SM array is sufficient; larger grids only
        # add combine work.
        target_main_ctas = (2 * num_sms + 2) // 3
        fill_splits = min(
            32,
            num_n_blocks,
            max(
                1,
                (target_main_ctas + total_mblocks - 1) // total_mblocks,
            ),
        )
        return _normalize_num_splits(fill_splits, num_n_blocks)
    wave_split_budget = max(1, (3 * num_sms) // total_mblocks)
    if num_n_blocks >= LOW_HD_DECODE_MAX_SPLITS * LOW_HD_DECODE_LONG_MIN_TILES_PER_CTA:
        # Long KV can afford the next power-of-two split without making the
        # per-CTA grain too small. Balanced partitions avoid partial waves and
        # sustain memory bandwidth better than device-specific odd counts.
        wave_split_budget = 1 << (wave_split_budget - 1).bit_length()
    if num_n_blocks <= 8:
        one_wave_splits = max(1, (num_sms + total_mblocks - 1) // total_mblocks)
        if hardware.is_integrated:
            one_wave_splits = 1 << (one_wave_splits - 1).bit_length()
        kv_grain_splits = min(LOW_HD_DECODE_MAX_SPLITS, one_wave_splits)
    else:
        blocks_per_cta_floor = (
            8
            if (max_head_dim >= 128 and (hardware.is_integrated or 2 <= packed_q_rows <= 4))
            else 4
        )
        kv_grain_splits = min(
            LOW_HD_DECODE_MAX_SPLITS,
            num_n_blocks // blocks_per_cta_floor,
        )
    return _normalize_num_splits(
        min(wave_split_budget, kv_grain_splits),
        num_n_blocks,
    )


class Sm120ForwardHost:
    """SM12x forward planning and kernel construction for the FA4 interface."""

    # Varlen SplitKV with device-side lengths: per-batch split counts from the prepare kernel,
    # where the gate below finds them worth their cost.
    dynamic_splits: bool = True
    # Costs of per-batch splits, in K / V bytes one chain (a KV head of one request, unsplit)
    # streams in the same time; a chain's rate is latency-bound and about the same per byte
    # whatever the head dim. The prepare kernel and its graph node, and, when the static plan
    # has one split, also the combine and the split-KV kernel build it does without.
    dynamic_split_cost_bytes: int = 128 * 1024
    dynamic_split_combine_cost_bytes: int = 128 * 1024

    @classmethod
    def resolve_plan(
        cls,
        *,
        requested_num_splits: int,
        generic_num_n_blocks: int,
        head_dim: int,
        head_dim_v: int,
        batch_size: int,
        num_head_kv: int,
        paged_kv: bool,
        page_size: Optional[int],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        pack_gqa: bool,
        element_size: int,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        num_m_blocks: int,
        total_mblocks: int,
        num_sms: int,
        total_q: int,
        has_cu_seqlens_q: bool,
        has_seqused_q: bool,
        has_seqused_k: bool,
        is_causal: bool,
        is_local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        has_score_or_mask_mod: bool,
        device: torch.device,
        fake_mode: bool,
        generic_heuristic: Callable[[int, int, int, int], int],
        block_sparse: bool = False,
        resident_ctas_per_sm: int,
    ) -> Sm120ForwardPlan:
        """Resolve all SM120 dataflow decisions behind one host-side boundary."""
        has_compact_q_groups = pack_gqa or packed_q_rows == max_seqlen_q
        split_num_n_blocks = cls.split_num_n_blocks(
            generic_num_n_blocks=generic_num_n_blocks,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_local=is_local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            tile_n=tile_n,
        )
        num_splits = cls.select_num_splits(
            requested_num_splits=requested_num_splits,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            has_compact_q_groups=has_compact_q_groups,
            element_size=element_size,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            total_mblocks=total_mblocks,
            num_sms=num_sms,
            num_n_blocks=split_num_n_blocks,
            device=device,
            fake_mode=fake_mode,
            generic_heuristic=generic_heuristic,
            resident_ctas_per_sm=resident_ctas_per_sm,
        )
        is_split_kv = num_splits > 1
        direct_uniform_batch = cls.use_direct_uniform_batch(
            batch_size=batch_size,
            paged_kv=paged_kv,
            has_cu_seqlens_q=has_cu_seqlens_q,
            has_seqused_q=has_seqused_q,
            total_q=total_q,
            max_seqlen_q=max_seqlen_q,
            num_m_blocks=num_m_blocks,
        )
        split_qk_n = cls.use_paged_decode_split_qk_n(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            has_compact_q_groups=has_compact_q_groups,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            num_n_blocks=generic_num_n_blocks,
            is_causal=is_causal,
            is_local=is_local,
            has_score_or_mask_mod=has_score_or_mask_mod,
        )
        dense_paged_kv = (
            page_size is not None
            and k is not None
            and k.stride(-1) == 1
            and v.stride(-1) == 1
            and k.stride(-2) == head_dim
            and v.stride(-2) == head_dim_v
            and k.stride(1) == num_head_kv * head_dim
            and v.stride(1) == num_head_kv * head_dim_v
            and k.stride(0) == page_size * num_head_kv * head_dim
            and v.stride(0) == page_size * num_head_kv * head_dim_v
        )
        transpose_qk_pv = cls.use_paged_decode_transpose_qk_pv(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            batch_size=batch_size,
            num_head_kv=num_head_kv,
            paged_kv=paged_kv,
            page_size=page_size,
            dense_paged_kv=dense_paged_kv,
            max_seqlen_q=max_seqlen_q,
            pack_gqa=pack_gqa,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            num_splits=num_splits,
            num_n_blocks=split_num_n_blocks,
            is_causal=is_causal,
            is_local=is_local,
            has_score_or_mask_mod=has_score_or_mask_mod,
        )
        if transpose_qk_pv:
            split_qk_n = False
        if block_sparse:
            # The block-sparse lists run in the generic consumers.
            split_qk_n = transpose_qk_pv = False
        split_kv_blocks_per_cta = cls.select_paged_decode_split_kv_blocks_per_cta(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            batch_size=batch_size,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            has_compact_q_groups=has_compact_q_groups,
            packed_q_rows=packed_q_rows,
            is_split_kv=is_split_kv,
            direct_uniform_batch=direct_uniform_batch,
            has_seqused_k=has_seqused_k,
            split_qk_n=split_qk_n,
            num_splits=num_splits,
            num_n_blocks=split_num_n_blocks,
        )
        if block_sparse:
            split_kv_blocks_per_cta = 0
        return Sm120ForwardPlan(
            num_splits=num_splits,
            split_num_n_blocks=split_num_n_blocks,
            direct_uniform_batch=direct_uniform_batch,
            split_qk_n=split_qk_n,
            split_kv_blocks_per_cta=split_kv_blocks_per_cta,
            transpose_qk_pv=transpose_qk_pv,
        )

    @classmethod
    def plan_forward(
        cls,
        *,
        arch: int,
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        head_dim: int,
        head_dim_v: int,
        tile_mn: Optional[tuple[int, int]],
        total_q: int,
        num_head: int,
        num_head_kv: int,
        qhead_per_kvhead: int,
        batch_size: int,
        seqlen_q: Optional[int],
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        pack_gqa: bool,
        page_table: Optional[torch.Tensor],
        page_size: Optional[int],
        num_splits: int,
        has_cu_seqlens_q: bool,
        has_seqused_q: bool,
        has_cu_seqlens_k: bool,
        has_seqused_k: bool,
        has_score_or_mask_mod: bool,
        device,
        fake_mode: bool,
        sparse_block_size: Optional[tuple[int, int]] = None,
        block_sparse: bool = False,
        sparse_varlen: bool = False,
    ) -> tuple[Sm120ForwardConfig, Sm120ForwardPlan]:
        """Tile configuration and launch plan of one forward call."""
        num_sms = get_num_sms_for_selection(device.index, arch)
        config = cls.select_config(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            tile_mn=tile_mn,
            total_q_rows=total_q * num_head,
            num_sms=num_sms,
            num_batch=batch_size,
            seqlen_q=max_seqlen_q if max_seqlen_q is not None else seqlen_q,
            seqlen_k=max_seqlen_k,
            num_head_kv=num_head_kv,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            pack_gqa=pack_gqa,
            paged_kv=page_table is not None,
            l2_group_bytes=l2_group_bytes(device.index),
            schedule_rates=schedule_rates(device.index),
            sparse_block_size=sparse_block_size,
            sparse_varlen=sparse_varlen,
        )
        tile_m, tile_n = config.tile_m, config.tile_n
        # The policy counts packed GQA rows per KV head whether or not pack_gqa is set.
        seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
        seqlen_k_loaded = (
            max_seqlen_k
            if not local
            else max(
                0,
                min(
                    max_seqlen_k,
                    (max_seqlen_k if window_size_right is None else window_size_right)
                    + (max_seqlen_k if window_size_left is None else window_size_left)
                    + 1
                    + tile_m,
                ),
            )
        )
        num_m_blocks = (seqlen_q_packgqa + tile_m - 1) // tile_m
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
        dynamic_splits = (
            cls.dynamic_splits
            and num_splits < 1
            and (has_cu_seqlens_q or has_seqused_q)
            and (has_cu_seqlens_k or has_seqused_k)
            and not block_sparse
            # hd 256 paged decode splits each request by its own length in the kernel.
            and not (head_dim == 256 and page_table is not None)
        )
        resolve = partial(
            cls.resolve_plan,
            generic_num_n_blocks=num_n_blocks,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            batch_size=batch_size,
            num_head_kv=num_head_kv,
            paged_kv=page_table is not None,
            page_size=page_size,
            k=k,
            v=v,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            pack_gqa=pack_gqa,
            element_size=v.element_size(),
            packed_q_rows=seqlen_q_packgqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_m_blocks=num_m_blocks,
            total_mblocks=total_mblocks,
            num_sms=num_sms,
            total_q=total_q,
            has_cu_seqlens_q=has_cu_seqlens_q,
            has_seqused_q=has_seqused_q,
            has_seqused_k=has_seqused_k,
            is_causal=causal,
            is_local=local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            has_score_or_mask_mod=has_score_or_mask_mod,
            device=device,
            fake_mode=fake_mode,
            generic_heuristic=num_splits_heuristic,
            block_sparse=block_sparse,
            resident_ctas_per_sm=cls.resident_ctas_per_sm(
                cutlass.BFloat16,
                head_dim,
                head_dim_v,
                tile_m,
                tile_n,
                config.num_stages,
                config.num_threads,
                page_table is not None,
            ),
        )
        # The split heuristic counts dense KV blocks; block-sparse calls split only on request.
        static_plan = resolve(
            requested_num_splits=max(num_splits, 1) if block_sparse else num_splits
        )
        if not dynamic_splits:
            return config, static_plan
        # The grid and workspace fit the most splits a longest request can use alone; no
        # request takes a coarser share per split than the static plan gives every request.
        plan = resolve(
            requested_num_splits=num_splits_heuristic(
                num_head_kv * num_m_blocks, num_sms, num_n_blocks, 128
            )
        )
        # The lengths are on the device, so the worst case of each plan decides: the static
        # plan leaves one request of the longest length on num_splits chains, per-batch splits
        # pay their fixed cost on every call.
        tail_bytes = (
            seqlen_k_loaded
            * (head_dim + head_dim_v)
            * v.element_size()
            * (1 / static_plan.num_splits - 1 / plan.num_splits)
        )
        cost_bytes = cls.dynamic_split_cost_bytes + (
            cls.dynamic_split_combine_cost_bytes if static_plan.num_splits == 1 else 0
        )
        if tail_bytes <= cost_bytes:
            return config, static_plan
        if plan.num_splits > 1:
            plan = replace(
                plan,
                dynamic_splits=True,
                split_grain=-(-static_plan.split_num_n_blocks // static_plan.num_splits),
            )
            if (
                plan.direct_uniform_batch
                and FlashPrepareScheduler.get_grid_shape(batch_size)[0] == 1
            ):
                # Room for the static plan's splits, the longest request alone at the cap,
                # and the extra splits the prepare heuristic spreads over the SMs (one per
                # blocks_per_sm share of the total).
                margin = max(num_sms / 128 + 0.001, 1.1)
                heads = num_head_kv if pack_gqa else num_head
                plan = replace(
                    plan,
                    extra_splits=max(
                        batch_size * (static_plan.num_splits - 1),
                        plan.num_splits - 1,
                        math.ceil(num_sms / (margin * heads)),
                    ),
                )
        return config, plan

    @staticmethod
    def split_metadata(
        plan: Sm120ForwardPlan, prepare: Callable
    ) -> Optional[SchedulerMetadataTensorsTorch]:
        """Per-batch split counts (and the compact split layout), or None for the plan's
        uniform count. prepare runs the scheduler-metadata kernel."""
        if not plan.dynamic_splits:
            return None
        return prepare(
            num_splits=plan.num_splits,
            max_n_blocks_per_split=plan.split_grain,
            max_extra_splits=plan.extra_splits,
        )

    @staticmethod
    def fwd_config(
        arch: int, config: Sm120ForwardConfig, plan: Sm120ForwardPlan, pack_gqa: bool
    ) -> FwdConfig:
        """The FwdConfig the shared forward code reads: the plan's tile and split count, and the
        neutral values of the SM90 / SM100 options (one Q stage, warp MMA with no register
        split, no cluster, CLC or persistent scheduling)."""
        return FwdConfig(
            device_capacity=arch // 10,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            pack_gqa=pack_gqa,
            registers=None,
            **{
                **_FWD_SM90_ONLY_NEUTRAL_FIELDS,
                **_FWD_SM100_ONLY_NEUTRAL_FIELDS,
                "num_splits": plan.num_splits,
            },
        )

    @staticmethod
    def metadata_key(config: Sm120ForwardConfig, plan: Sm120ForwardPlan) -> tuple:
        """Every plan decision the scheduler metadata depends on: the tile (m blocks per
        batch, KV blocks per split), the split budget and layout, and the tile lookup."""
        return (
            "sm120",
            config.tile_m,
            config.tile_n,
            plan.num_splits,
            plan.dynamic_splits,
            plan.split_grain,
            plan.extra_splits,
            plan.direct_uniform_batch,
        )

    @staticmethod
    def uses_tile_lookup(plan: Sm120ForwardPlan) -> bool:
        """Whether varlen tiles find their batch through the precomputed lookup; the
        uniform-batch scheduler maps them arithmetically."""
        return not plan.direct_uniform_batch

    @staticmethod
    def writes_single_split_output(plan: Sm120ForwardPlan) -> bool:
        """Whether batches with one split hold their final output after the kernel."""
        return plan.dynamic_splits

    @staticmethod
    def _final_outputs(
        plan: Sm120ForwardPlan, out: torch.Tensor, lse: Optional[torch.Tensor]
    ) -> tuple:
        """out / lse as views shaped like the split partials (a leading split mode), for
        the one-split batches of a dynamic plan."""
        if not plan.dynamic_splits:
            return (None, None)
        return (out.unsqueeze(0), lse.unsqueeze(0) if lse is not None else None)

    @classmethod
    def compile_arguments(
        cls,
        plan: Sm120ForwardPlan,
        *,
        cu_total_m_blocks: Optional[torch.Tensor],
        blocks_to_batch_idx: Optional[torch.Tensor],
        num_splits_dynamic: Optional[torch.Tensor],
        split_slot_batch: Optional[torch.Tensor],
        out: torch.Tensor,
        lse: Optional[torch.Tensor],
    ) -> tuple:
        """Extra CuTe arguments required by the selected SM120 kernel ABI."""
        o_final, lse_final = cls._final_outputs(plan, out, lse)
        return (
            *(
                to_cute_tensor(t, assumed_align=4, leading_dim=0)
                for t in (cu_total_m_blocks, blocks_to_batch_idx, num_splits_dynamic)
            ),
            to_cute_tensor(o_final, assumed_align=16),
            to_cute_tensor(lse_final, assumed_align=4),
            to_cute_tensor(split_slot_batch, assumed_align=4, leading_dim=0),
        )

    @classmethod
    def runtime_arguments(
        cls,
        plan: Sm120ForwardPlan,
        *,
        cu_total_m_blocks: Optional[torch.Tensor],
        blocks_to_batch_idx: Optional[torch.Tensor],
        num_splits_dynamic: Optional[torch.Tensor],
        split_slot_batch: Optional[torch.Tensor],
        out: torch.Tensor,
        lse: Optional[torch.Tensor],
    ) -> tuple:
        """Extra runtime arguments required by the selected SM120 kernel ABI."""
        return (
            cu_total_m_blocks,
            blocks_to_batch_idx,
            num_splits_dynamic,
            *cls._final_outputs(plan, out, lse),
            split_slot_batch,
        )

    @staticmethod
    def _fit_tile_to_sparse_block(
        head_dim: int,
        head_dim_v: int,
        tile_m: int,
        tile_n: int,
        sparse_block_size: tuple[int, int],
        exact_q: bool = False,
    ) -> tuple[int, int]:
        """A sparse block spans whole tiles (q_subtile_factor = Bq / tile_m, kv_subtile_factor =
        Bk / tile_n): keep the selected tile when it divides the block, else take the largest
        divisor the kernel runs (the block itself first, e.g. Bk 112), within the SMEM. Varlen
        lists count Q blocks per batch, so there tile_m is Bq itself (exact_q)."""
        wide = max(head_dim, head_dim_v) > 128
        limit_m, limit_n = (64, 64) if wide else (128, 128)
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")

        def fits(m: int, n: int) -> bool:
            return (
                FlashAttentionForwardSm120.smem_bytes(
                    cutlass.BFloat16,
                    head_dim,
                    head_dim_v,
                    m,
                    n,
                    1,
                    FlashAttentionForwardSm120.get_fwd_num_threads(head_dim, head_dim_v, m, n),
                    False,
                )
                <= smem_capacity
            )

        def fit(size: int, tile: int, limit: int, other: int, is_m: bool) -> int:
            if size % tile == 0:
                return tile
            for t in (size, 128, 112, 96, 80, 64, 48, 32, 16):
                if t <= limit and size % t == 0 and fits(*((t, other) if is_m else (other, t))):
                    return t
            raise ValueError(f"SM120 block sparsity: no forward tile divides {size}")

        tile_n = fit(sparse_block_size[1], tile_n, limit_n, tile_m, False)
        if exact_q:
            tile_m = sparse_block_size[0]
            if tile_m > limit_m or not fits(tile_m, tile_n):
                raise ValueError(
                    f"SM120 varlen block sparsity runs Q tiles of the sparse block ({tile_m})"
                )
        else:
            tile_m = fit(sparse_block_size[0], tile_m, limit_m, tile_n, True)
        return tile_m, tile_n

    @staticmethod
    @lru_cache(maxsize=None)
    def resident_ctas_per_sm(
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        tile_m: int,
        tile_n: int,
        num_stages: int,
        num_threads: int,
        paged_kv: bool,
    ) -> int:
        """CTAs of a forward configuration that one SM holds at once. The SM's shared memory
        is the per-CTA maximum plus one 1 KB reservation, and every CTA reserves 1 KB."""
        if num_threads >= 256:
            return 1
        smem = FlashAttentionForwardSm120.smem_bytes(
            dtype, head_dim, head_dim_v, tile_m, tile_n, num_stages, num_threads, paged_kv
        )
        sm_smem = utils_basic.get_smem_capacity_in_bytes("sm_120") + 1024
        return max(1, min(2, sm_smem // (smem + 1024)))

    @staticmethod
    def select_config(
        *,
        head_dim: int,
        head_dim_v: int,
        tile_mn: Optional[tuple[int, int]],
        total_q_rows: int,
        num_sms: Optional[int],
        num_batch: int,
        seqlen_q: Optional[int],
        seqlen_k: Optional[int],
        num_head_kv: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        pack_gqa: bool,
        paged_kv: bool,
        l2_group_bytes: int = 50 * 1024 * 1024,
        sparse_block_size: Optional[tuple[int, int]] = None,
        sparse_varlen: bool = False,
        schedule_rates: Optional[tuple[float, float]] = None,
    ) -> Sm120ForwardConfig:
        if tile_mn is None:
            tile_m, tile_n = FlashAttentionForwardSm120.get_fwd_tile_size(
                head_dim,
                head_dim_v,
                total_q_rows=total_q_rows,
                num_sms=num_sms,
                num_batch=num_batch,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                num_head_kv=num_head_kv,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_local=is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                pack_gqa=pack_gqa,
                paged_kv=paged_kv,
                l2_group_bytes=l2_group_bytes,
                schedule_rates=schedule_rates,
            )
            if sparse_block_size is not None:
                tile_m, tile_n = Sm120ForwardHost._fit_tile_to_sparse_block(
                    head_dim, head_dim_v, tile_m, tile_n, sparse_block_size, exact_q=sparse_varlen
                )
        else:
            tile_m, tile_n = tile_mn
        num_threads = FlashAttentionForwardSm120.get_fwd_num_threads(
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            paged_kv=paged_kv,
        )
        lpt_order = False
        if (
            (is_causal or is_local)
            and not paged_kv
            and num_sms is not None
            and seqlen_q is not None
            and seqlen_k is not None
            and total_q_rows == num_batch * seqlen_q * num_head_kv * qhead_per_kvhead
        ):
            ctas_per_sm = Sm120ForwardHost.resident_ctas_per_sm(
                cutlass.BFloat16,
                head_dim,
                head_dim_v,
                tile_m,
                tile_n,
                FlashAttentionForwardSm120.get_fwd_num_stages(head_dim, head_dim_v, tile_m, tile_n),
                num_threads,
                False,
            )
            lpt_order = FlashAttentionForwardSm120.use_lpt_order(
                num_sms,
                head_dim,
                tile_m,
                ctas_per_sm,
                num_batch * (num_head_kv if pack_gqa else num_head_kv * qhead_per_kvhead),
                1 if pack_gqa else qhead_per_kvhead,
                seqlen_q,
                seqlen_k,
                qhead_per_kvhead if pack_gqa else 1,
                is_causal,
                window_size_left,
                window_size_right,
                l2_group_bytes,
                schedule_rates,
            )
        return Sm120ForwardConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=FlashAttentionForwardSm120.get_fwd_num_stages(
                head_dim, head_dim_v, tile_m, tile_n
            ),
            num_threads=num_threads,
            l2_group_bytes=l2_group_bytes,
            lpt_order=lpt_order,
        )

    @staticmethod
    def select_num_splits(
        *,
        requested_num_splits: int,
        head_dim: int,
        head_dim_v: int,
        paged_kv: bool,
        max_seqlen_q: int,
        has_compact_q_groups: bool,
        element_size: int,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        total_mblocks: int,
        num_sms: int,
        num_n_blocks: int,
        device: torch.device,
        fake_mode: bool,
        generic_heuristic: Callable[[int, int, int, int], int],
        resident_ctas_per_sm: int,
    ) -> int:
        num_splits = requested_num_splits
        hardware = None
        if num_splits < 1:
            is_hd256_paged_decode = (
                head_dim == 256
                and head_dim_v == 256
                and paged_kv
                and max_seqlen_q == 1
                and has_compact_q_groups
            )
            is_low_hd_paged_decode = is_low_hd_paged_decode_tile(
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                paged_kv=paged_kv,
                seqlen_q=max_seqlen_q,
                tile_m=tile_m,
                tile_n=tile_n,
            )
            if is_low_hd_paged_decode and tile_m == 16:
                num_splits = _select_low_hd_decode_num_splits(
                    max_head_dim=max(head_dim, head_dim_v),
                    packed_q_rows=packed_q_rows,
                    total_mblocks=total_mblocks,
                    num_n_blocks=num_n_blocks,
                    hardware=_get_decode_hardware(device, num_sms),
                )
            elif is_hd256_paged_decode:
                if fake_mode:
                    max_splits = min(128, 1 << (num_sms.bit_length() - 1))
                    num_splits = generic_heuristic(
                        total_mblocks,
                        num_sms,
                        num_n_blocks,
                        max_splits,
                    )
                else:
                    hardware = _get_decode_hardware(device, num_sms)
                    num_splits = _select_decode_num_splits(
                        head_dim=head_dim,
                        head_dim_v=head_dim_v,
                        element_size=element_size,
                        packed_q_rows=packed_q_rows,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        total_mblocks=total_mblocks,
                        num_n_blocks=num_n_blocks,
                        hardware=hardware,
                    )
            else:
                num_splits = generic_heuristic(
                    total_mblocks,
                    num_sms,
                    num_n_blocks,
                    128,
                )
            if (
                not fake_mode
                and packed_q_rows <= tile_m
                and not (is_low_hd_paged_decode and tile_m == 16)
            ):
                num_splits = _split_past_partial_wave(
                    num_splits=num_splits,
                    total_mblocks=total_mblocks,
                    packed_q_rows=packed_q_rows,
                    num_n_blocks=num_n_blocks,
                    tile_n=tile_n,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    element_size=element_size,
                    resident_ctas_per_sm=resident_ctas_per_sm,
                    hardware=hardware or _get_decode_hardware(device, num_sms),
                )

        if num_splits <= 1 or num_n_blocks == 0:
            return 1

        # BlockInfo assigns a uniform ceil-div chunk to every split. Normalize
        # the count so no SM120 CTA owns an empty tail chunk.
        return _normalize_num_splits(num_splits, num_n_blocks)

    @staticmethod
    def split_num_n_blocks(
        *,
        generic_num_n_blocks: int,
        paged_kv: bool,
        max_seqlen_q: int,
        max_seqlen_k: int,
        is_local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        tile_n: int,
    ) -> int:
        """Return the KV-block count used by the decode SplitKV policy.

        The generic local-attention bound includes a full query tile because
        prefill tiles can span multiple query positions. Packed decode has one
        query position regardless of how many GQA heads occupy the M tile, so
        its exact visible span is just left + current + right.
        """
        if not (paged_kv and max_seqlen_q == 1 and is_local):
            return generic_num_n_blocks
        visible_seqlen_k = visible_decode_seqlen_k(
            max_seqlen_k,
            is_local=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        return max(0, (visible_seqlen_k + tile_n - 1) // tile_n)

    @staticmethod
    def use_direct_uniform_batch(
        *,
        batch_size: int,
        paged_kv: bool,
        has_cu_seqlens_q: bool,
        has_seqused_q: bool,
        total_q: int,
        max_seqlen_q: int,
        num_m_blocks: int,
    ) -> bool:
        """Use arithmetic scheduling for one uniform query row per request.

        This invariant is independent of whether GQA heads are folded into the
        query-row mode, and of the KV cache layout. In particular, unpacked MHA
        should not pay the generic varlen scheduler's prefix-sum walk merely to
        preserve its ordinary Q loader.
        """
        return (
            has_cu_seqlens_q
            and not has_seqused_q
            and max_seqlen_q == 1
            and total_q == batch_size * max_seqlen_q
            and num_m_blocks == 1
        )

    @staticmethod
    def use_paged_decode_split_qk_n(
        *,
        head_dim: int,
        head_dim_v: int,
        paged_kv: bool,
        max_seqlen_q: int,
        has_compact_q_groups: bool,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        num_n_blocks: int,
        is_causal: bool,
        is_local: bool,
        has_score_or_mask_mod: bool,
    ) -> bool:
        """Four-way N-distributed QK for hd 256 paged decode: it removes the single-QK-warp
        critical path and lowers the register footprint, with the same threads and
        one-CTA-per-SM residency."""
        return (
            (head_dim, head_dim_v) == (256, 256)
            and paged_kv
            and max_seqlen_q == 1
            and has_compact_q_groups
            and packed_q_rows <= 16
            and (tile_m, tile_n) == (16, 64)
            and num_n_blocks >= 1
            and (is_causal or is_local)
            and not has_score_or_mask_mod
        )

    @staticmethod
    def select_paged_decode_split_kv_blocks_per_cta(
        *,
        head_dim: int,
        head_dim_v: int,
        batch_size: int,
        paged_kv: bool,
        max_seqlen_q: int,
        has_compact_q_groups: bool,
        packed_q_rows: int,
        is_split_kv: bool,
        direct_uniform_batch: bool,
        has_seqused_k: bool,
        split_qk_n: bool,
        num_splits: int,
        num_n_blocks: int,
    ) -> int:
        """Preserve the longest request's split grain for ragged decode."""
        if not (
            (head_dim, head_dim_v) == (256, 256)
            and batch_size > 1
            and paged_kv
            and max_seqlen_q == 1
            and has_compact_q_groups
            and packed_q_rows <= 16
            and is_split_kv
            and direct_uniform_batch
            and has_seqused_k
            and split_qk_n
            and num_splits > 1
            and num_n_blocks > 0
        ):
            return 0
        return (num_n_blocks + num_splits - 1) // num_splits

    @staticmethod
    def use_paged_decode_transpose_qk_pv(
        *,
        head_dim: int,
        head_dim_v: int,
        batch_size: int,
        num_head_kv: int,
        paged_kv: bool,
        page_size: Optional[int],
        dense_paged_kv: bool,
        max_seqlen_q: int,
        pack_gqa: bool,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        num_splits: int,
        num_n_blocks: int,
        is_causal: bool,
        is_local: bool,
        has_score_or_mask_mod: bool,
    ) -> bool:
        """The page-TMA M64N8 decode dataflow: transposed QK and PV put two to eight packed
        query rows on the MMA N = 8 axis instead of padding them to M = 16. The transpose is
        amortized over at least two KV tiles per CTA (four above two batch-head groups), for
        up to eight groups."""
        effective_splits = max(1, num_splits)
        kv_tiles_per_cta = (
            (num_n_blocks + effective_splits - 1) // effective_splits if num_n_blocks > 0 else 0
        )
        batch_head_groups = batch_size * num_head_kv
        min_kv_tiles_per_cta = 2 if batch_head_groups <= 2 else 4
        return (
            (head_dim, head_dim_v) == (256, 256)
            and 1 <= batch_head_groups <= 8
            and paged_kv
            and page_size == tile_n
            and dense_paged_kv
            and max_seqlen_q == 1
            and pack_gqa
            and 2 <= packed_q_rows <= 8
            and (tile_m, tile_n) == (16, 64)
            and kv_tiles_per_cta >= min_kv_tiles_per_cta
            and (is_causal or is_local)
            and not has_score_or_mask_mod
        )

    @staticmethod
    def make_kernel(
        *,
        dtype,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        pack_gqa: bool,
        config: Sm120ForwardConfig,
        paged_kv: bool,
        score_mod: Optional[Callable],
        mask_mod: Optional[Callable],
        has_aux_tensors: bool,
        is_split_kv: bool,
        plan: Sm120ForwardPlan,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
    ) -> FlashAttentionForwardSm120:
        if not FlashAttentionForwardSm120.can_implement(
            dtype,
            head_dim,
            head_dim_v,
            config.tile_m,
            config.tile_n,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            is_causal=is_causal,
            paged_kv=paged_kv,
        ):
            raise ValueError(
                "The requested FlashAttention forward configuration exceeds "
                "SM120 kernel constraints or shared-memory capacity"
            )
        Kernel = FlashAttentionForwardSm120
        if plan.transpose_qk_pv:
            from flash_attn.cute.sm120.flash_fwd_decode import (
                FlashAttentionForwardSm120DecodeTranspose,
            )

            Kernel = FlashAttentionForwardSm120DecodeTranspose
        return Kernel(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            pack_gqa=pack_gqa,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            Q_in_regs=False,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=has_aux_tensors,
            is_split_kv=is_split_kv,
            direct_uniform_batch=plan.direct_uniform_batch,
            paged_kv=paged_kv,
            split_qk_n=plan.split_qk_n,
            split_kv_blocks_per_cta=plan.split_kv_blocks_per_cta,
            l2_group_bytes=config.l2_group_bytes,
            q_subtile_factor=q_subtile_factor,
            kv_subtile_factor=kv_subtile_factor,
            lpt_order=config.lpt_order,
        )


sm120_forward_host = Sm120ForwardHost()
