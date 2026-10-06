# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Host-only forward config selection and validation.

FwdConfig uses exact split counts: 1 disables SplitKV. Explicit configs are
validated, not normalized. Selection must not read tensor contents, synchronize
CUDA, or allocate device memory. MLA (qv) uses a separate policy in interface.py.
"""

import math
from dataclasses import astuple, dataclass
from functools import lru_cache
from typing import NamedTuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Sm90Registers:
    """Per-thread setmaxnreg targets: each MMA warp group raises, the producer lowers."""

    mma: int
    producer: int


@dataclass(frozen=True)
class Sm100Registers:
    """Per-thread setmaxnreg targets for the SM100/SM110 forward warp-group roles.

    One softmax warp group per Q stage raises to ``softmax``; the remaining
    ``3 - q_stage`` load/MMA/epilogue groups lower to ``other``; correction raises or
    lowers depending on whether ``correction`` exceeds the launch register count.
    """

    softmax: int
    correction: int
    other: int


# None for architectures whose forward kernels do not split registers by role.
Registers = Sm90Registers | Sm100Registers | None


@dataclass(frozen=True)
class FwdConfig:
    """Fully resolved forward launch configuration.

    Dynamic persistence is derived from scheduler metadata, not is_static_persistent.
    num_splits and tile_n affect reduction order. With causal/local SplitKV, the
    per-split KV range also depends on the M tile extent, q_stage * tile_m.
    """

    device_capacity: int
    tile_m: int
    tile_n: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    q_stage: int
    use_clc_scheduler: bool
    is_static_persistent: bool
    use_tma_o: bool
    use_s_ping_pong: bool
    num_splits: int
    use_2cta_instrs: bool
    registers: Registers


class FwdHeuristicInputs(NamedTuple):
    """Host metadata for config selection and validation.

    has_max_seqlen_q_hint distinguishes a caller's varlen Q bound from total_q.
    """

    device_arch: int
    num_sms: int
    dtype: str
    head_dim: int
    head_dim_v: int
    num_heads: int
    num_heads_kv: int
    batch_size: int
    total_q: int
    total_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    has_max_seqlen_q_hint: bool = True
    seqlen_k_per_split: int | None = None
    causal: bool = False
    local: bool = False
    window_size_left: int | None = None
    window_size_right: int | None = None
    has_cu_seqlens_q: bool = False
    has_cu_seqlens_k: bool = False
    has_seqused_q: bool = False
    has_seqused_k: bool = False
    has_caller_scheduler_metadata: bool = False
    has_tile_count_semaphore: bool = False
    pack_gqa: bool = False
    page_size: int | None = None
    use_block_sparsity: bool = False
    sparse_q_block_size: int | None = None
    sparse_kv_block_size: int | None = None
    has_score_mod: bool = False
    has_mask_mod: bool = False
    has_learnable_sink: bool = False
    has_lse: bool = False
    requested_tile_m: int | None = None
    requested_tile_n: int | None = None
    requested_mma_pv_is_rs: bool | None = None
    requested_intra_wg_overlap: bool | None = None
    requested_num_splits: int | None = None
    requested_use_clc_scheduler: bool = False
    disable_2cta: bool = False
    disable_s_ping_pong: bool = False

    @property
    def device_capacity(self) -> int:
        return self.device_arch // 10

    @property
    def is_varlen_q(self) -> bool:
        return self.has_cu_seqlens_q or self.has_seqused_q

    @property
    def is_varlen(self) -> bool:
        return self.is_varlen_q or self.has_cu_seqlens_k or self.has_seqused_k

    @property
    def qhead_per_kvhead(self) -> int:
        return self.num_heads // self.num_heads_kv

    @property
    def packed_max_seqlen_q(self) -> int:
        return self.max_seqlen_q * (self.qhead_per_kvhead if self.pack_gqa else 1)

    @property
    def head_dim_padded(self) -> int:
        return ceil_div(self.head_dim, 16) * 16

    @property
    def head_dim_v_padded(self) -> int:
        return ceil_div(self.head_dim_v, 16) * 16

    @property
    def is_hdim256(self) -> bool:
        return self.head_dim == 256 and self.head_dim_v == 256


def num_splits_heuristic(
    total_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """Resolve automatic SplitKV to a concrete positive split count."""
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1
    # Avoid ZeroDivisionError when batch_size or seqlen_q is 0.
    if total_mblocks == 0:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    # More tiles than SMs means no split, not zero splits.
    return max(1, min(num_sms // total_mblocks, max_splits, num_n_blocks))


def fixed_seqlen_k_num_splits(inputs: FwdHeuristicInputs) -> int:
    """Return the minimum split count required by a fixed KV extent."""
    seqlen_k_per_split = inputs.seqlen_k_per_split
    if seqlen_k_per_split is None:
        return 1
    if seqlen_k_per_split <= 0:
        raise ValueError("seqlen_k_per_split must be positive")
    return max(1, ceil_div(inputs.max_seqlen_k, seqlen_k_per_split))


def loaded_seqlen_k(inputs: FwdHeuristicInputs, tile_m: int) -> int:
    """Return the KV extent one M tile can load."""
    if not inputs.local:
        return inputs.max_seqlen_k
    # Only None is unbounded; preserve 0, e.g. the right bound of a causal window.
    window_left = (
        inputs.max_seqlen_k
        if inputs.window_size_left is None
        else inputs.window_size_left
    )
    window_right = (
        inputs.max_seqlen_k
        if inputs.window_size_right is None
        else inputs.window_size_right
    )
    return max(0, min(inputs.max_seqlen_k, window_left + window_right + 1 + tile_m))


def select_sm90_tile(
    head_dim: int,
    head_dim_v: int,
    is_causal: bool,
    is_local: bool,
    sparse_block_size_q: int | None,
) -> tuple[int, int, bool, bool]:
    """Return (tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap) for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).

    When sparse_block_size_q is set, tile_m must divide it. For head_dim <= 96 the
    optimal tile_m=192 is used when compatible, otherwise we fall back to 128.
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return 128, 128, True, True
        return 192, 128, True, True
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return 128, 128, False, True
        if is_causal or is_local:
            return 192, 128, False, True
        else:
            return 192, 144, False, True
    elif head_dim <= 128:
        return 128, 128, True, True
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return 128, tile_n, True, True
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return 128, tile_n, True, True


def fit_sm90_tile_to_block_sparsity(
    tile_m: int, tile_n: int, sparse_block_size_q: int, sparse_block_size_kv: int
) -> tuple[int, int]:
    """Shrink SM90 forward tiles to fit explicit sparse blocks.

    Sparse Q blocks may span multiple M tiles, but SM90 requires tile_n to match the
    sparse KV block. Unsupported sizes are rejected by normalize_block_sparse_config.
    """
    if sparse_block_size_q % tile_m != 0 and sparse_block_size_q % 64 == 0:
        tile_m = 64
    if sparse_block_size_kv < tile_n and sparse_block_size_kv % 16 == 0:
        tile_n = sparse_block_size_kv
    return tile_m, tile_n


# === SM100 TUNING KNOBS (agent-editable) ===
# Keys: (use_2cta_instrs: bool, is_causal: bool, head_dim_padded: int, is_sm103: bool)
# Values:
#   ex2_emu_freq: int — how often to use emulated exp2 (0=all hardware exp2, higher=more emulation).
#                        SM103 has fast native exp2, so set freq=0 there.
#   ex2_emu_start_frg: int — fragment index to start emulation from
#   num_regs_softmax: int — register count for softmax warps (multiple of 8)
#   num_regs_correction: int — register count for correction warps (multiple of 8)
# The forward kernel reads the ex2 fields; select_sm100_registers reads the register fields and
# gives the remaining budget to the other warp groups.
SM100_TUNING = {
    (True, False, 128, False): {"ex2_emu_freq": 10, "ex2_emu_start_frg": 1, "num_regs_softmax": 184, "num_regs_correction": 80},
    (False, True, 128, False): {"ex2_emu_freq": 16, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 72},
    (True, False, 192, False): {"ex2_emu_freq": 16, "ex2_emu_start_frg": 0, "num_regs_softmax": 184, "num_regs_correction": 80},
    (False, True, 192, False): {"ex2_emu_freq": 32, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 72},
    (True, False, 128, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 80},
    (False, True, 128, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 64},
    (True, False, 192, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 64},
    (False, True, 192, True): {"ex2_emu_freq": 0, "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 72},
    (True, False, 256, False): {"ex2_emu_freq": 14, "ex2_emu_start_frg": 0, "num_regs_softmax": 256, "num_regs_correction": 160},
    (True, True, 256, False): {"ex2_emu_freq": 14, "ex2_emu_start_frg": 0, "num_regs_softmax": 256, "num_regs_correction": 160},
}
SM100_FP8_TUNING = {
    (True, False, 128, False): {'ex2_emu_freq': 10, 'ex2_emu_start_frg': 1, 'num_regs_softmax': 160, 'num_regs_correction': 72},
    # Causal hd128 FP8 previously inherited bf16's freq=16. FP8 fwd is MUFU/ex2-bound, so a more
    # aggressive emulation freq=8 offloads more exp from MUFU: +3.4%(4k)..+5.5%(16k) on B200,
    # MHA & GQA, accuracy-neutral (3-run validated, locked clocks). freq=8 would regress non-causal
    # (0.94x), hence keyed on is_causal=True only.
    (False, True, 128, False): {'ex2_emu_freq': 8, 'ex2_emu_start_frg': 1},
}
# head_dim_padded < 96, keyed by paged non-TMA KV.
SM100_SMALL_HDIM_REGISTERS = {
    False: Sm100Registers(softmax=200, correction=64, other=48),
    True: Sm100Registers(softmax=184, correction=64, other=80),
}
SM100_FP8_SMALL_HDIM_REGISTERS = {
    False: Sm100Registers(softmax=168, correction=96, other=80),
    True: Sm100Registers(softmax=152, correction=96, other=112),
}
# === END SM100 TUNING KNOBS ===

# 16 warps: two softmax, correction, MMA, load, epilogue, and an empty warp group.
SM100_THREADS_PER_CTA = 512
# Per-thread registers at launch, the setmaxnreg pivot between raising and lowering.
SM100_LAUNCH_REGISTERS = 65536 // SM100_THREADS_PER_CTA


def sm100_is_sm103_family(device_arch: int) -> bool:
    """Match the kernel's ``Arch.is_family_of(sm_103f)``: SM10x with minor version >= 3."""
    return device_arch // 10 == 10 and device_arch % 10 >= 3


def select_sm100_registers(
    inputs: FwdHeuristicInputs, *, tile_n: int, q_stage: int, use_2cta_instrs: bool
) -> Sm100Registers:
    head_dim_padded = inputs.head_dim_padded
    paged_kv_non_tma = inputs.page_size not in (None, tile_n)
    key = (use_2cta_instrs, inputs.causal, head_dim_padded, sm100_is_sm103_family(inputs.device_arch))
    is_fp8 = "float8" in inputs.dtype
    if head_dim_padded < 96:
        small = SM100_FP8_SMALL_HDIM_REGISTERS if is_fp8 else SM100_SMALL_HDIM_REGISTERS
        return small[paged_kv_non_tma]
    fp8_tune = SM100_FP8_TUNING.get(key, {}) if is_fp8 else {}
    if not paged_kv_non_tma and "num_regs_softmax" in fp8_tune:
        softmax, correction = fp8_tune["num_regs_softmax"], fp8_tune["num_regs_correction"]
        return Sm100Registers(softmax, correction, 512 - softmax * 2 - correction)
    tune = SM100_TUNING.get(key, {})
    if paged_kv_non_tma:
        softmax, correction = 184, 64
    elif "num_regs_softmax" in tune:
        softmax, correction = tune["num_regs_softmax"], tune["num_regs_correction"]
    else:
        softmax, correction = 192, 80
    # The other warp groups share the remaining budget; setmaxnreg.dec cannot exceed launch.
    other = min(
        (512 - softmax * q_stage - correction) // (3 - q_stage) // 8 * 8,
        SM100_LAUNCH_REGISTERS,
    )
    return Sm100Registers(softmax, correction, other)


# SM90 splits with TMA Q and KV loads, keyed by MMA warp groups (tile_m // 64).
SM90_TMA_REGISTERS = {
    1: Sm90Registers(mma=256, producer=56),
    2: Sm90Registers(mma=240, producer=24),
    3: Sm90Registers(mma=160, producer=32),
}


def select_sm90_registers(inputs: FwdHeuristicInputs, *, tile_m: int, tile_n: int) -> Sm90Registers:
    num_mma_warp_groups = tile_m // 64
    tma_q = not (inputs.pack_gqa and tile_m % inputs.qhead_per_kvhead != 0)
    tma_kv = inputs.page_size in (None, tile_n)
    # A cp.async Q or KV producer needs more registers.
    if num_mma_warp_groups == 2 and not (tma_q and tma_kv):
        return Sm90Registers(mma=224, producer=40)
    return SM90_TMA_REGISTERS[num_mma_warp_groups]


def select_registers(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    tile_n: int,
    q_stage: int,
    use_2cta_instrs: bool,
) -> Registers:
    """Resolve the register split after the final tile, stage, and 2CTA choices."""
    match inputs.device_capacity:
        case 9:
            return select_sm90_registers(inputs, tile_m=tile_m, tile_n=tile_n)
        case 10 | 11:
            return select_sm100_registers(
                inputs, tile_n=tile_n, q_stage=q_stage, use_2cta_instrs=use_2cta_instrs
            )
        case _:
            return None


def validate_registers(config: "FwdConfig") -> None:
    """Reject register splits the kernels cannot apply with setmaxnreg."""
    registers = config.registers
    expected = {9: Sm90Registers, 10: Sm100Registers, 11: Sm100Registers}.get(
        config.device_capacity, type(None)
    )
    if type(registers) is not expected:
        raise ValueError(
            f"SM{config.device_capacity} needs registers of type {expected.__name__}, "
            f"got {type(registers).__name__}"
        )
    if registers is None:
        return
    if any(v % 8 != 0 or not 24 <= v <= 256 for v in astuple(registers)):
        raise ValueError(f"Register counts must be multiples of 8 in [24, 256], got {registers}")
    if isinstance(registers, Sm90Registers):
        num_mma_warp_groups = config.tile_m // 64
        launch = 65536 // (128 * (num_mma_warp_groups + 1))
        budget = num_mma_warp_groups * registers.mma + registers.producer
        if not registers.producer <= launch <= registers.mma:
            raise ValueError(
                f"SM90 MMA registers must be >= {launch} and producer <= {launch}, got {registers}"
            )
    else:
        launch = SM100_LAUNCH_REGISTERS
        q_stage = config.q_stage
        budget = q_stage * registers.softmax + registers.correction + (3 - q_stage) * registers.other
        if registers.softmax < launch or registers.other > launch:
            raise ValueError(
                f"SM100 softmax registers must be >= {launch} and other <= {launch}, got {registers}"
            )
    if budget > 512:
        raise ValueError(f"Register split {registers} exceeds 512 per thread (uses {budget})")


def select_initial_fwd_tile(inputs: FwdHeuristicInputs) -> tuple[int, int, bool, bool]:
    """Resolve tile and SM90-specific flags before shape-dependent policies."""
    tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap = 128, 128, False, False
    match inputs.device_capacity:
        case 8:
            tile_n = 64
        case 9:
            tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap = select_sm90_tile(
                inputs.head_dim,
                inputs.head_dim_v,
                inputs.causal,
                inputs.local,
                inputs.sparse_q_block_size,
            )
            if (
                inputs.sparse_q_block_size is not None
                and inputs.sparse_kv_block_size is not None
            ):
                tile_m, tile_n = fit_sm90_tile_to_block_sparsity(
                    tile_m,
                    tile_n,
                    inputs.sparse_q_block_size,
                    inputs.sparse_kv_block_size,
                )
        case 12 if inputs.head_dim > 64:
            # SM120 has 99 KB SMEM: 128×128 is preferred for D<=64, and 128×64
            # avoids the occupancy loss from a 96 KB tile for larger head dimensions.
            tile_n = 64

    if inputs.requested_tile_m is not None:
        tile_m = inputs.requested_tile_m
    if inputs.requested_tile_n is not None:
        tile_n = inputs.requested_tile_n
    # SM90-only flags; other architectures ignore the requests.
    if inputs.device_capacity == 9:
        if inputs.requested_tile_m is not None or inputs.requested_tile_n is not None:
            mma_pv_is_rs = True
            intra_wg_overlap = True
        if inputs.requested_mma_pv_is_rs is not None:
            mma_pv_is_rs = inputs.requested_mma_pv_is_rs
        if inputs.requested_intra_wg_overlap is not None:
            intra_wg_overlap = inputs.requested_intra_wg_overlap
    return tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap


def default_pack_gqa(inputs: FwdHeuristicInputs) -> bool:
    """Resolve pack_gqa=None; ``inputs.pack_gqa`` is ignored."""
    if inputs.qhead_per_kvhead == 1:
        return False
    # hd256 favors 2CTA over cp.async-Q PackGQA.
    prefer_hd256_2cta = (
        inputs.device_capacity in (10, 11)
        and inputs.is_hdim256
        and 128 % inputs.qhead_per_kvhead != 0
        and not inputs.has_seqused_q
        and hd256_2cta_varlen_ok(inputs)
        and (
            inputs.max_seqlen_q > 128
            or (
                inputs.requested_num_splits == 1
                and 2 * inputs.batch_size * inputs.num_heads <= inputs.num_sms
            )
        )
    )
    return not prefer_hd256_2cta


def hd256_2cta_varlen_ok(inputs: FwdHeuristicInputs) -> bool:
    """Caller-built varlen scheduler metadata assumes 1CTA tiles."""
    return not inputs.has_cu_seqlens_q or not inputs.has_caller_scheduler_metadata


def can_use_tma_o(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    num_splits: int,
) -> bool:
    """Whether the SM100 output layout supports TMA."""
    return (
        # hd256 SplitKV uses a chunked half-width epilogue without TMA.
        not (num_splits > 1 and inputs.head_dim_v == 256)
        and not (
            inputs.pack_gqa
            and (tile_m % inputs.qhead_per_kvhead != 0 or num_splits > 1)
        )
        and not inputs.is_varlen_q
    )


# Minimum KV blocks per split for S ping-pong; hd256 SplitKV stays disabled.
_S_PING_PONG_MIN_N_BLOCKS_PER_SPLIT = {64: 16, 128: 64}


def can_use_s_ping_pong(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    tile_n: int,
    q_stage: int,
) -> bool:
    """Whether the SM100 kernel supports S/P ping-pong."""
    return (
        q_stage == 1
        and inputs.head_dim in (64, 128, 256)
        and inputs.head_dim_v == inputs.head_dim
        and tile_m == 128
        and tile_n == 128
        and inputs.page_size in (None, tile_n)
        and not inputs.has_score_mod
        and not inputs.has_mask_mod
        and not inputs.use_block_sparsity
        and not inputs.has_learnable_sink
    )


def is_2cta_eligible(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    q_stage: int,
    num_splits: int,
) -> bool:
    """Whether the SM100 tile geometry supports 2CTA instructions."""
    # Single-M-block hd256 2CTA halves K/V loads per CTA; favor it when the grid fits one wave.
    num_q_tiles_per_batch = inputs.num_heads_kv if inputs.pack_gqa else inputs.num_heads
    hd256_decode_2cta = (
        inputs.is_hdim256
        and 2 * inputs.batch_size * num_q_tiles_per_batch <= inputs.num_sms
    )
    return (
        num_splits == 1
        and (
            not inputs.has_cu_seqlens_q
            or (inputs.is_hdim256 and hd256_2cta_varlen_ok(inputs))
        )
        and not inputs.has_seqused_q
        and not inputs.use_block_sparsity
        and inputs.page_size in (None, 128)
        and (inputs.packed_max_seqlen_q > q_stage * tile_m or hd256_decode_2cta)
        and (tile_m % inputs.qhead_per_kvhead == 0 or not inputs.pack_gqa)
        and (
            inputs.is_hdim256
            or (
                not inputs.causal
                and not inputs.local
                and inputs.head_dim_padded in (128, 192)
                and inputs.head_dim_v_padded == 128
            )
        )
    )


def can_use_static_persistent(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    q_stage: int,
    num_splits: int,
    use_2cta_instrs: bool,
    use_clc_scheduler: bool,
) -> bool:
    """Whether static persistence supports this tile mapping."""
    # A varlen-Q tile-count semaphore selects the dynamic persistent scheduler.
    if (
        num_splits > 1
        or use_clc_scheduler
        or (inputs.is_varlen_q and inputs.has_tile_count_semaphore)
    ):
        return False
    dense_noncausal = not inputs.causal and not inputs.local and not inputs.is_varlen_q
    single_m_block = (
        inputs.packed_max_seqlen_q <= q_stage * tile_m
        and (not inputs.has_cu_seqlens_q or inputs.has_max_seqlen_q_hint)
        # Dense causal/local LPT supports 2CTA clusters only without static persistence.
        and not (
            use_2cta_instrs
            and not inputs.has_cu_seqlens_q
            and (inputs.causal or inputs.local)
        )
    )
    return dense_noncausal or single_m_block


def can_use_clc(
    inputs: FwdHeuristicInputs, *, tile_n: int, use_2cta_instrs: bool
) -> bool:
    """Whether the SM100 kernel honors CLC for this layout."""
    return (
        inputs.page_size in (None, tile_n)
        # A varlen-Q tile-count semaphore selects the dynamic persistent scheduler.
        and not (inputs.is_varlen_q and inputs.has_tile_count_semaphore)
        # CLC does not map hd256 2CTA tiles correctly.
        and not (inputs.is_hdim256 and use_2cta_instrs)
    )


@lru_cache(maxsize=1024)
def select_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Select one fully resolved config from host-visible metadata."""
    device_capacity = inputs.device_capacity
    if device_capacity == 12 and inputs.requested_num_splits not in (None, 1):
        raise ValueError("SM120 forward only supports num_splits=1")

    tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap = select_initial_fwd_tile(inputs)
    is_sm100_family = device_capacity in (10, 11)
    packed_seqlen_q = inputs.packed_max_seqlen_q
    # Two Q tiles exceed TMEM capacity at hdim_v 256.
    q_stage = (
        2
        if is_sm100_family and packed_seqlen_q > tile_m and inputs.head_dim_v != 256
        else 1
    )
    seqlen_k_loaded = loaded_seqlen_k(inputs, tile_m)

    num_m_blocks = ceil_div(packed_seqlen_q, q_stage * tile_m)
    heads_per_m_block = 1 if inputs.pack_gqa else inputs.qhead_per_kvhead
    total_mblocks = (
        inputs.batch_size * inputs.num_heads_kv * heads_per_m_block * num_m_blocks
    )
    num_n_blocks = ceil_div(seqlen_k_loaded, tile_n)
    if inputs.requested_num_splits is not None:
        num_splits = max(1, inputs.requested_num_splits)
    elif not is_sm100_family:
        # Only SM100/SM110 implement SplitKV.
        num_splits = 1
    else:
        num_splits = num_splits_heuristic(
            total_mblocks, inputs.num_sms, num_n_blocks, 128
        )

    # Diff-head SplitKV needs extra SMEM for fp32 partial O: use tile_n=64 for long KV,
    # no split for short KV or DV512. A requested split count may change here.
    if is_sm100_family and inputs.head_dim != inputs.head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and inputs.head_dim_v != 512:
            tile_n = 64
            num_n_blocks = ceil_div(seqlen_k_loaded, tile_n)
            num_splits = num_splits_heuristic(
                total_mblocks, inputs.num_sms, num_n_blocks, 128
            )
        else:
            num_splits = 1
    if inputs.requested_num_splits is None and num_splits > 1:
        num_splits = max(num_splits, fixed_seqlen_k_num_splits(inputs))

    is_split_kv = num_splits > 1
    use_2cta_instrs = (
        is_sm100_family
        and not inputs.disable_2cta
        and is_2cta_eligible(
            inputs, tile_m=tile_m, q_stage=q_stage, num_splits=num_splits
        )
    )

    # A CLC request alone skips varlen MHA (K/V traffic under imbalance) and dense
    # noncausal (work-stealing overhead).
    is_varlen_mha = inputs.is_varlen and inputs.qhead_per_kvhead == 1
    is_dense_noncausal = not inputs.is_varlen and not inputs.causal and not inputs.local
    use_clc_scheduler = (
        is_sm100_family
        and can_use_clc(inputs, tile_n=tile_n, use_2cta_instrs=use_2cta_instrs)
        and inputs.requested_use_clc_scheduler
        and not is_varlen_mha
        and not is_dense_noncausal
    )
    is_static_persistent = is_sm100_family and can_use_static_persistent(
        inputs,
        tile_m=tile_m,
        q_stage=q_stage,
        num_splits=num_splits,
        use_2cta_instrs=use_2cta_instrs,
        use_clc_scheduler=use_clc_scheduler,
    )
    use_tma_o = is_sm100_family and can_use_tma_o(
        inputs, tile_m=tile_m, num_splits=num_splits
    )
    n_blocks_per_split = ceil_div(num_n_blocks, num_splits)
    use_s_ping_pong = (
        is_sm100_family
        and not inputs.disable_s_ping_pong
        and can_use_s_ping_pong(inputs, tile_m=tile_m, tile_n=tile_n, q_stage=q_stage)
        and (
            not is_split_kv
            or n_blocks_per_split
            >= _S_PING_PONG_MIN_N_BLOCKS_PER_SPLIT.get(inputs.head_dim, math.inf)
        )
    )

    config = FwdConfig(
        device_capacity=device_capacity,
        tile_m=tile_m,
        tile_n=tile_n,
        mma_pv_is_rs=mma_pv_is_rs,
        intra_wg_overlap=intra_wg_overlap,
        q_stage=q_stage,
        use_clc_scheduler=use_clc_scheduler,
        is_static_persistent=is_static_persistent,
        use_tma_o=use_tma_o,
        use_s_ping_pong=use_s_ping_pong,
        num_splits=num_splits,
        use_2cta_instrs=use_2cta_instrs,
        registers=select_registers(
            inputs,
            tile_m=tile_m,
            tile_n=tile_n,
            q_stage=q_stage,
            use_2cta_instrs=use_2cta_instrs,
        ),
    )
    validate_fwd_config(config, inputs)
    return config


_SM100_SMEM_BYTES = 227 * 1024
# Barrier/scheduler overhead calibrated to hd192/128, tile_n=192: 247808 bytes at launch.
_SM100_SMEM_OVERHEAD_BYTES = 2048


def sm100_hd192_uneven_kv_smem_bytes(
    config: FwdConfig, inputs: FwdHeuristicInputs
) -> int | None:
    """SM100 hd192/128 SMEM footprint when a two-stage budget becomes uneven [K, V, K].

    Return None outside this case; mirrors FlashAttentionForwardSm100.
    """
    if inputs.head_dim_padded != 192 or inputs.head_dim_v_padded != 128:
        return None
    element_bytes = 1 if "float8" in inputs.dtype else 2
    o_bytes = 4 if config.num_splits > 1 else 2  # SplitKV stores fp32 partials
    rows = config.q_stage * config.tile_m
    # sO overlaps sQ for head-dim 192.
    q_o_bytes = max(rows * 192 * element_bytes, rows * 128 * o_bytes)
    cta_group_size = 2 if config.use_2cta_instrs else 1
    k_stage_bytes = config.tile_n * 192 * element_bytes // cta_group_size
    v_stage_bytes = config.tile_n * 128 * element_bytes // cta_group_size
    if (224 * 1024 - q_o_bytes) // k_stage_bytes != 2:
        return None
    return q_o_bytes + 2 * k_stage_bytes + v_stage_bytes + _SM100_SMEM_OVERHEAD_BYTES


@lru_cache(maxsize=1024)
def validate_fwd_config(config: FwdConfig, inputs: FwdHeuristicInputs) -> None:
    """Reject known unsupported or silently normalized config combinations."""
    if inputs.device_capacity not in (8, 9, 10, 11, 12):
        raise ValueError(
            f"Unsupported forward architecture family SM{inputs.device_capacity}"
        )
    if config.device_capacity != inputs.device_capacity:
        raise ValueError(
            f"Config targets SM{config.device_capacity}, but the problem targets SM{inputs.device_capacity}"
        )
    if config.tile_m <= 0 or config.tile_n <= 0:
        raise ValueError(
            f"Tile dimensions must be positive, got {(config.tile_m, config.tile_n)}"
        )
    if config.tile_m % 16 != 0 or config.tile_n % 16 != 0:
        raise ValueError(
            f"Tile dimensions must be multiples of 16, got {(config.tile_m, config.tile_n)}"
        )
    if config.num_splits < 1 or config.num_splits > 256:
        raise ValueError(f"num_splits must be in [1, 256], got {config.num_splits}")
    # Fixed KV extents constrain split counts only under SplitKV.
    fixed_num_splits = fixed_seqlen_k_num_splits(inputs)
    if config.num_splits > 1 and config.num_splits < fixed_num_splits:
        raise ValueError(
            f"seqlen_k_per_split={inputs.seqlen_k_per_split} requires "
            f"num_splits >= {fixed_num_splits}, got {config.num_splits}"
        )
    if (
        inputs.seqlen_k_per_split is not None
        and inputs.seqlen_k_per_split % config.tile_n != 0
    ):
        raise ValueError(
            f"seqlen_k_per_split must be divisible by tile_n={config.tile_n}"
        )

    is_sm90 = inputs.device_capacity == 9
    is_sm100_family = inputs.device_capacity in (10, 11)
    if not is_sm100_family:
        match inputs.device_capacity:
            case 8 if inputs.page_size is not None:
                raise ValueError("SM80 forward does not support paged KV")
            case 12 if inputs.page_size is not None or inputs.use_block_sparsity:
                raise ValueError(
                    "SM120 forward does not support paged KV or block sparsity"
                )
        if is_sm90:
            if config.tile_m not in (64, 128, 192):
                raise ValueError(
                    f"SM90 tile_m must be 64, 128, or 192, got {config.tile_m}"
                )
        elif config.mma_pv_is_rs or config.intra_wg_overlap:
            raise ValueError(
                f"SM{inputs.device_capacity} does not expose SM90 MMA flags"
            )
        if config.q_stage != 1:
            raise ValueError(f"SM{inputs.device_capacity} requires q_stage=1")
        if config.num_splits != 1:
            raise ValueError(f"SM{inputs.device_capacity} does not support SplitKV")
        if (
            config.use_2cta_instrs
            or config.use_clc_scheduler
            or config.is_static_persistent
            or config.use_tma_o
            or config.use_s_ping_pong
        ):
            raise ValueError(
                f"SM{inputs.device_capacity} does not support SM100 scheduler options"
            )
        validate_registers(config)
        return

    if config.mma_pv_is_rs or config.intra_wg_overlap:
        raise ValueError("SM100 forward does not expose SM90 MMA flags")
    # Softmax warp groups own 128 rows each; smaller tiles overrun the row statistics.
    if config.tile_m != 128:
        raise ValueError(f"SM100 forward requires tile_m=128, got {config.tile_m}")
    if config.q_stage not in (1, 2):
        raise ValueError(f"SM100 q_stage must be 1 or 2, got {config.q_stage}")
    if 2 * config.tile_n + config.q_stage * inputs.head_dim_v_padded > 512:
        raise ValueError("S and O accumulators exceed the 512 TMEM columns")
    hd192_smem_bytes = sm100_hd192_uneven_kv_smem_bytes(config, inputs)
    if hd192_smem_bytes is not None and hd192_smem_bytes > _SM100_SMEM_BYTES:
        raise ValueError(
            f"Head-dim 192/128 with tile_n={config.tile_n} needs {hd192_smem_bytes} bytes "
            f"of shared memory; SM100 provides {_SM100_SMEM_BYTES}"
        )

    if config.num_splits > 1:
        if inputs.head_dim_v_padded >= 192 and inputs.head_dim_v != 256:
            raise ValueError(
                "SplitKV does not support padded value head dimensions >= 192 except 256"
            )
        if inputs.head_dim != inputs.head_dim_v and config.tile_n != 64:
            raise ValueError("Diff-head SplitKV requires tile_n=64")

    paged_kv_non_tma = inputs.page_size not in (None, config.tile_n)
    if paged_kv_non_tma and (inputs.head_dim % 16 != 0 or inputs.head_dim_v % 16 != 0):
        raise ValueError("Non-TMA paged KV requires head dimensions divisible by 16")
    if config.use_clc_scheduler and not can_use_clc(
        inputs, tile_n=config.tile_n, use_2cta_instrs=config.use_2cta_instrs
    ):
        raise ValueError(
            "CLC is not supported for this forward problem"
        )
    if config.is_static_persistent and not can_use_static_persistent(
        inputs,
        tile_m=config.tile_m,
        q_stage=config.q_stage,
        num_splits=config.num_splits,
        use_2cta_instrs=config.use_2cta_instrs,
        use_clc_scheduler=config.use_clc_scheduler,
    ):
        raise ValueError(
            "Static persistent scheduling is not supported for this forward problem"
        )

    if config.use_tma_o and not can_use_tma_o(
        inputs, tile_m=config.tile_m, num_splits=config.num_splits
    ):
        raise ValueError(
            "TMA O is not supported for this GQA, SplitKV, or varlen layout"
        )
    if config.use_s_ping_pong and not can_use_s_ping_pong(
        inputs, tile_m=config.tile_m, tile_n=config.tile_n, q_stage=config.q_stage
    ):
        raise ValueError("S ping-pong is not supported for this forward problem")
    if config.use_2cta_instrs and not is_2cta_eligible(
        inputs,
        tile_m=config.tile_m,
        q_stage=config.q_stage,
        num_splits=config.num_splits,
    ):
        raise ValueError("2CTA instructions are not supported for this forward problem")
    validate_registers(config)
