# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Host-only forward, MLA forward, and backward config selection and validation.

FwdConfig, MlaFwdConfig, and BwdConfig are fully resolved launch choices; forward
configs use exact split counts (1 disables SplitKV). Explicit configs are validated,
not normalized. Selection must not read tensor contents, synchronize CUDA, or
allocate device memory.
"""

import math
from dataclasses import astuple, dataclass
from functools import lru_cache
from typing import ClassVar, Literal, NamedTuple


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
    pack_gqa: bool
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
    pack_gqa: bool | None = None  # None = no preference; selection resolves before policy reads.
    page_size: int | None = None
    use_block_sparsity: bool = False
    sparse_mask_per_head: bool = False
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
        inputs.max_seqlen_k if inputs.window_size_left is None else inputs.window_size_left
    )
    window_right = (
        inputs.max_seqlen_k if inputs.window_size_right is None else inputs.window_size_right
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
    (True, False, 128, False): {
        "ex2_emu_freq": 10,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 184,
        "num_regs_correction": 80,
    },
    (False, True, 128, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    (True, False, 192, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 184,
        "num_regs_correction": 80,
    },
    (False, True, 192, False): {
        "ex2_emu_freq": 32,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    (True, False, 128, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 80,
    },
    (False, True, 128, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
    (True, False, 192, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
    (False, True, 192, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 72,
    },
    (True, False, 256, False): {
        "ex2_emu_freq": 14,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 256,
        "num_regs_correction": 160,
    },
    (True, True, 256, False): {
        "ex2_emu_freq": 14,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 256,
        "num_regs_correction": 160,
    },
}
SM100_FP8_TUNING = {
    (True, False, 128, False): {
        "ex2_emu_freq": 10,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 160,
        "num_regs_correction": 72,
    },
    # Causal hd128 FP8 previously inherited bf16's freq=16. FP8 fwd is MUFU/ex2-bound, so a more
    # aggressive emulation freq=8 offloads more exp from MUFU: +3.4%(4k)..+5.5%(16k) on B200,
    # MHA & GQA, accuracy-neutral (3-run validated, locked clocks). freq=8 would regress non-causal
    # (0.94x), hence keyed on is_causal=True only.
    (False, True, 128, False): {"ex2_emu_freq": 8, "ex2_emu_start_frg": 1},
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
    key = (
        use_2cta_instrs,
        inputs.causal,
        head_dim_padded,
        sm100_is_sm103_family(inputs.device_arch),
    )
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
    num_splits: int,
    use_2cta_instrs: bool,
    use_clc_scheduler: bool,
) -> Registers:
    """Resolve the register split after the final tile, stage, 2CTA, and CLC choices."""
    match inputs.device_capacity:
        case 9:
            return select_sm90_registers(inputs, tile_m=tile_m, tile_n=tile_n)
        case 10 | 11:
            # Persistent CLC keeps scheduler state live across tiles; see NOTE [causal CLC].
            if use_clc_scheduler:
                _, clc_registers = causal_clc_policy(inputs, num_splits=num_splits, q_stage=q_stage)
                if clc_registers is not None:
                    return clc_registers
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
        budget = (
            q_stage * registers.softmax + registers.correction + (3 - q_stage) * registers.other
        )
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
            if inputs.sparse_q_block_size is not None and inputs.sparse_kv_block_size is not None:
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
        and not (inputs.pack_gqa and (tile_m % inputs.qhead_per_kvhead != 0 or num_splits > 1))
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
        inputs.is_hdim256 and 2 * inputs.batch_size * num_q_tiles_per_batch <= inputs.num_sms
    )
    return (
        num_splits == 1
        and (not inputs.has_cu_seqlens_q or (inputs.is_hdim256 and hd256_2cta_varlen_ok(inputs)))
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
            use_2cta_instrs and not inputs.has_cu_seqlens_q and (inputs.causal or inputs.local)
        )
    )
    return dense_noncausal or single_m_block


# CUDA caps gridDim.y and gridDim.z at 65535.
_MAX_GRID_YZ = 65535


def can_use_clc(
    inputs: FwdHeuristicInputs, *, tile_n: int, num_splits: int, use_2cta_instrs: bool
) -> bool:
    """Whether the SM100 kernel honors CLC for this layout."""
    return (
        inputs.page_size in (None, tile_n)
        # The CLC grid is (m blocks, heads, batch * splits), without the folding of the
        # nonpersistent grid.
        and (inputs.num_heads_kv if inputs.pack_gqa else inputs.num_heads) <= _MAX_GRID_YZ
        and inputs.batch_size * num_splits <= _MAX_GRID_YZ
        # A varlen-Q tile-count semaphore selects the dynamic persistent scheduler.
        and not (inputs.is_varlen_q and inputs.has_tile_count_semaphore)
        # Varlen CLC maps hd256 2CTA tiles per CTA, not per cluster.
        and not (inputs.is_hdim256 and use_2cta_instrs and inputs.is_varlen)
    )


# NOTE [causal CLC]
# CLC amortizes CTA setup across tiles while preserving L2-swizzled LPT order
# (clc_late_claim). The benefit depends on dtype, head dimension, and architecture, so this
# table enables it by default only for benchmark-validated combinations: <=0.5% loss per
# shape, >1% gain on at least one shape. A cell may carry a measured register split that
# runs faster under CLC; None keeps the default split; missing entries leave CLC off by
# default. Timing candidates: see benchmarks/clc_bench.py --summary.
# (dtype path, head_dim_padded) -> {arch family: register override}.
_CAUSAL_CLC_DEFAULT = {
    ("bf16", 64): {"sm_100": None},
    ("bf16", 96): {"sm_100": None, "sm_103": None, "sm_107": None},
    ("bf16", 128): {"sm_100": None, "sm_103": None, "sm_107": None},
    ("bf16", 192): {"sm_103": None},
    ("fp16", 64): {"sm_103": Sm100Registers(softmax=184, correction=64, other=80)},
    ("fp16", 96): {"sm_103": None, "sm_107": None},
    ("fp16", 128): {"sm_100": None, "sm_103": None, "sm_107": None},
    ("fp16", 192): {"sm_100": None, "sm_103": None},
    ("fp8", 64): {"sm_100": None, "sm_103": None, "sm_107": None},
    ("fp8", 128): {"sm_100": None, "sm_103": None, "sm_107": None},
}


def causal_clc_policy(
    inputs: FwdHeuristicInputs, *, num_splits: int, q_stage: int
) -> tuple[bool, Sm100Registers | None]:
    """Return (enabled, register override) for measured dense causal CLC cells.

    An enabled cell with no override keeps the default register split; see NOTE [causal CLC].
    """
    if (
        inputs.device_capacity != 10
        or not inputs.causal
        or inputs.is_varlen
        or inputs.local
        or num_splits != 1
        or q_stage != 2
        or inputs.page_size is not None
        or inputs.use_block_sparsity
        or inputs.has_score_mod
        or inputs.has_mask_mod
    ):
        return False, None
    minor = inputs.device_arch % 10
    family = "sm_107" if minor >= 7 else "sm_103" if minor >= 3 else "sm_100"
    dtype = inputs.dtype
    dtype_path = "fp8" if "float8" in dtype else "fp16" if dtype == "torch.float16" else "bf16"
    row = _CAUSAL_CLC_DEFAULT.get((dtype_path, inputs.head_dim_padded), {})
    return family in row, row.get(family)


@lru_cache(maxsize=1024)
def select_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Select one fully resolved config from host-visible metadata."""
    pack_gqa = default_pack_gqa(inputs) if inputs.pack_gqa is None else inputs.pack_gqa
    if inputs.sparse_mask_per_head:
        pack_gqa = False
    inputs = inputs._replace(pack_gqa=pack_gqa)
    device_capacity = inputs.device_capacity
    if device_capacity == 12 and inputs.requested_num_splits not in (None, 1):
        raise ValueError("SM120 forward only supports num_splits=1")

    tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap = select_initial_fwd_tile(inputs)
    is_sm100_family = device_capacity in (10, 11)
    packed_seqlen_q = inputs.packed_max_seqlen_q
    # Two Q tiles exceed TMEM capacity at hdim_v 256.
    q_stage = 2 if is_sm100_family and packed_seqlen_q > tile_m and inputs.head_dim_v != 256 else 1
    seqlen_k_loaded = loaded_seqlen_k(inputs, tile_m)

    num_m_blocks = ceil_div(packed_seqlen_q, q_stage * tile_m)
    heads_per_m_block = 1 if inputs.pack_gqa else inputs.qhead_per_kvhead
    total_mblocks = inputs.batch_size * inputs.num_heads_kv * heads_per_m_block * num_m_blocks
    num_n_blocks = ceil_div(seqlen_k_loaded, tile_n)
    if inputs.requested_num_splits is not None:
        num_splits = max(1, inputs.requested_num_splits)
    elif not is_sm100_family:
        # Only SM100/SM110 implement SplitKV.
        num_splits = 1
    else:
        num_splits = num_splits_heuristic(total_mblocks, inputs.num_sms, num_n_blocks, 128)

    # Diff-head SplitKV needs extra SMEM for fp32 partial O: use tile_n=64 for long KV,
    # no split for short KV or DV512. A requested split count may change here.
    if is_sm100_family and inputs.head_dim != inputs.head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and inputs.head_dim_v != 512:
            tile_n = 64
            num_n_blocks = ceil_div(seqlen_k_loaded, tile_n)
            num_splits = num_splits_heuristic(total_mblocks, inputs.num_sms, num_n_blocks, 128)
        else:
            num_splits = 1
    if inputs.requested_num_splits is None and num_splits > 1:
        num_splits = max(num_splits, fixed_seqlen_k_num_splits(inputs))

    is_split_kv = num_splits > 1
    use_2cta_instrs = (
        is_sm100_family
        and not inputs.disable_2cta
        and is_2cta_eligible(inputs, tile_m=tile_m, q_stage=q_stage, num_splits=num_splits)
    )

    # CLC runs by default only on measured cells; explicit configs may choose it elsewhere.
    default_clc, _ = causal_clc_policy(inputs, num_splits=num_splits, q_stage=q_stage)
    use_clc_scheduler = (
        is_sm100_family
        and default_clc
        and can_use_clc(
            inputs, tile_n=tile_n, num_splits=num_splits, use_2cta_instrs=use_2cta_instrs
        )
    )
    is_static_persistent = is_sm100_family and can_use_static_persistent(
        inputs,
        tile_m=tile_m,
        q_stage=q_stage,
        num_splits=num_splits,
        use_2cta_instrs=use_2cta_instrs,
        use_clc_scheduler=use_clc_scheduler,
    )
    use_tma_o = is_sm100_family and can_use_tma_o(inputs, tile_m=tile_m, num_splits=num_splits)
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
        pack_gqa=pack_gqa,
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
            num_splits=num_splits,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
        ),
    )
    validate_fwd_config(config, inputs)
    return config


_SM100_SMEM_BYTES = 227 * 1024
# Barrier/scheduler overhead calibrated to hd192/128, tile_n=192: 247808 bytes at launch.
_SM100_SMEM_OVERHEAD_BYTES = 2048


def sm100_hd192_uneven_kv_smem_bytes(config: FwdConfig, inputs: FwdHeuristicInputs) -> int | None:
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
    if config.pack_gqa and inputs.sparse_mask_per_head:
        raise ValueError("PackGQA requires head-broadcast sparse masks")
    if inputs.pack_gqa is not None and inputs.pack_gqa != config.pack_gqa:
        raise ValueError(
            f"pack_gqa={inputs.pack_gqa} disagrees with config.pack_gqa={config.pack_gqa}"
        )
    inputs = inputs._replace(pack_gqa=config.pack_gqa)
    if inputs.device_capacity not in (8, 9, 10, 11, 12):
        raise ValueError(f"Unsupported forward architecture family SM{inputs.device_capacity}")
    if config.device_capacity != inputs.device_capacity:
        raise ValueError(
            f"Config targets SM{config.device_capacity}, but the problem targets SM{inputs.device_capacity}"
        )
    if config.tile_m <= 0 or config.tile_n <= 0:
        raise ValueError(f"Tile dimensions must be positive, got {(config.tile_m, config.tile_n)}")
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
    if inputs.seqlen_k_per_split is not None and inputs.seqlen_k_per_split % config.tile_n != 0:
        raise ValueError(f"seqlen_k_per_split must be divisible by tile_n={config.tile_n}")

    is_sm90 = inputs.device_capacity == 9
    is_sm100_family = inputs.device_capacity in (10, 11)
    if not is_sm100_family:
        match inputs.device_capacity:
            case 8 if inputs.page_size is not None:
                raise ValueError("SM80 forward does not support paged KV")
            case 12 if inputs.page_size is not None or inputs.use_block_sparsity:
                raise ValueError("SM120 forward does not support paged KV or block sparsity")
        if is_sm90:
            if config.tile_m not in (64, 128, 192):
                raise ValueError(f"SM90 tile_m must be 64, 128, or 192, got {config.tile_m}")
        elif config.mma_pv_is_rs or config.intra_wg_overlap:
            raise ValueError(f"SM{inputs.device_capacity} does not expose SM90 MMA flags")
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
            raise ValueError(f"SM{inputs.device_capacity} does not support SM100 scheduler options")
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
        inputs,
        tile_n=config.tile_n,
        num_splits=config.num_splits,
        use_2cta_instrs=config.use_2cta_instrs,
    ):
        raise ValueError("CLC is not supported for this forward problem")
    if config.is_static_persistent and not can_use_static_persistent(
        inputs,
        tile_m=config.tile_m,
        q_stage=config.q_stage,
        num_splits=config.num_splits,
        use_2cta_instrs=config.use_2cta_instrs,
        use_clc_scheduler=config.use_clc_scheduler,
    ):
        raise ValueError("Static persistent scheduling is not supported for this forward problem")

    if config.use_tma_o and not can_use_tma_o(
        inputs, tile_m=config.tile_m, num_splits=config.num_splits
    ):
        raise ValueError("TMA O is not supported for this GQA, SplitKV, or varlen layout")
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


# === MLA (qv) forward ===
# The absorbed-MLA kernels have their own policy: a kernel choice (2CTA, 1CTA, or 1CTA with
# 64-key blocks), its tiles and split count. Measurements: AI/SPARSE_MLA_1CTA.md.

MlaKernel = Literal["2cta", "1cta", "1cta_kb64"]
# Sparse MLA pads a token's Q heads to one tile (pack_gqa.qheads_first_tma_view).
MLA_2CTA_SPARSE_HEAD_TILE = 128
MLA_1CTA_SPARSE_HEAD_TILE = 64
# kb64: one token per tile (heads padded to 64) x 64 keys, and at least 4 of those key
# blocks per split: each split pays a Q load, a 128 KB fp32 O partial and its share of the
# combine (one block per split measured 30% slower than 32 splits at s_k = 8K, b = 1).
MLA_KB64_TILE_MN = (64, 64)
MLA_KB64_MIN_BLOCKS_PER_SPLIT = 4


def sparse_mla_qhead_tile(qhead_per_kvhead: int, min_tile: int = 128) -> int:
    """Rows one token's Q heads occupy in a sparse-MLA tile: the real head count padded to
    ``min_tile`` (128 in the 2CTA forward and the generic dQ/dQv, 64 in the backward, the
    1CTA forward and the 64-row dQ/dQv) or to 128 when it exceeds it."""
    assert 0 < qhead_per_kvhead <= 128, (
        f"sparse MLA: MQA with 1 to 128 heads, got {qhead_per_kvhead}"
    )
    return min_tile if qhead_per_kvhead <= min_tile else 128


@dataclass(frozen=True)
class MlaFwdConfig:
    """Fully resolved absorbed-MLA (qv) forward launch configuration.

    The kernels take neither tile_m nor q_stage: they size the 1CTA split heuristic only.
    s_ahead applies to the 1CTA 128-key kernel only.
    """

    kernel: MlaKernel
    tile_m: int
    tile_n: int
    q_stage: int
    num_splits: int
    pack_gqa: bool
    use_clc_scheduler: bool
    s_ahead: bool
    # Generic-kernel knobs, fixed for MLA; ClassVars so equality and astuple ignore them.
    mma_pv_is_rs: ClassVar[bool] = False
    intra_wg_overlap: ClassVar[bool] = False
    use_2cta_instrs: ClassVar[bool] = False
    is_static_persistent: ClassVar[bool] = False
    use_tma_o: ClassVar[bool] = False
    use_s_ping_pong: ClassVar[bool] = False
    registers: ClassVar[None] = None  # the MLA kernels own their register split


class MlaFwdHeuristicInputs(NamedTuple):
    """Host metadata for MLA forward selection: the shared forward facts plus the qv-only bits.

    mla_1cta_override is FLASH_ATTENTION_MLA_1CTA: True runs 1CTA wherever supported, False never.
    """

    fwd: FwdHeuristicInputs
    is_topk_gather: bool = False
    requires_grad: bool = False
    gather_bwd_recompute_p: bool = False
    has_descale: bool = False
    mla_1cta_override: bool | None = None

    @property
    def nheads_per_kv(self) -> int:
        return self.fwd.qhead_per_kvhead

    @property
    def is_fp8(self) -> bool:
        return "float8" in self.fwd.dtype

    @property
    def seqlen_q_hint(self) -> int:
        """Dense seqlen_q, else the varlen max_seqlen_q (total_q without a caller hint)."""
        fwd = self.fwd
        return fwd.max_seqlen_q if fwd.has_cu_seqlens_q else fwd.total_q // fwd.batch_size

    @property
    def seqlen_q_known(self) -> int | None:
        """seqlen_q_hint, or None for varlen without a caller max_seqlen_q."""
        fwd = self.fwd
        if fwd.has_cu_seqlens_q and not fwd.has_max_seqlen_q_hint:
            return None
        return self.seqlen_q_hint

    @property
    def auto_split(self) -> bool:
        return self.fwd.requested_num_splits is None

    @property
    def needs_1cta(self) -> bool:
        """fp8, descales and explicit split-KV exist only on 1CTA."""
        num_splits = self.fwd.requested_num_splits
        return self.is_fp8 or self.has_descale or (num_splits is not None and num_splits > 1)

    @property
    def ctas_2cta(self) -> int:
        """2CTA tiles: one per token (sparse) or per batch element (dense decode), 2 CTAs each."""
        fwd = self.fwd
        return 2 * fwd.num_heads_kv * (fwd.total_q if self.is_topk_gather else fwd.batch_size)


def mla_1cta_supported(inputs: MlaFwdHeuristicInputs) -> bool:
    """Dense MLA at any shape; sparse up to 64 Q heads (one 64-row tile per token), training
    only with recompute-P (the 1CTA forward emits no P / row_max)."""
    return not inputs.is_topk_gather or (
        inputs.nheads_per_kv <= 64 and (not inputs.requires_grad or inputs.gather_bwd_recompute_p)
    )


def mla_1cta_route(inputs: MlaFwdHeuristicInputs, *, split_kv: bool) -> bool:
    """Whether an MLA call runs the 1CTA (tcgen05.mma.ws) kernel rather than the 2CTA one.

    Heuristic (measurements: AI/SPARSE_MLA_1CTA.md "Dispatch heuristic"):
      - fp8, descales or explicit split-KV (needs_1cta; 2CTA has none) -> 1CTA.
      - One-wave rule: 2CTA spends 2 CTAs on each token (sparse) or batch element (dense
        decode), ctas_2cta in all; 1CTA beats it only once that exceeds one wave of num_sms.
      - Sparse -> the one-wave rule.
      - Dense with the split heuristic (split_kv) -> 1CTA provisionally: selection sizes the
        split for 1CTA and asks again with split_kv=False if that gives too few splits.
      - Dense unsplit -> the one-wave rule on decode shapes (seqlen_q x heads per KV head
        <= 64), else 2CTA. Varlen without a host max_seqlen_q counts as prefill.
    mla_1cta_override overrides it (True: wherever supported; False: never).
    """
    supported = mla_1cta_supported(inputs)
    if inputs.mla_1cta_override is not None:
        return inputs.mla_1cta_override and supported
    if not supported:
        return False
    if inputs.needs_1cta:
        return True
    past_one_wave = inputs.ctas_2cta > inputs.fwd.num_sms
    if inputs.is_topk_gather:
        return past_one_wave
    if split_kv:
        return True
    if inputs.seqlen_q_hint * inputs.nheads_per_kv > 64:
        return False
    return past_one_wave


def can_use_mla_kb64(inputs: MlaFwdHeuristicInputs) -> bool:
    """1CTA MLA calls the 64-key-block mainloop serves (the rest run the 128-key mainloop):
    16-bit, <= 64 heads; sparse at any such head count; dense at 64 heads, or with fewer on
    decode only -- padding the one-token tile wins there (1.07-1.30x over the 128-key
    mainloop) but wastes 64 / H of the MMA work on prefill, where the 128-key mainloop packs
    several tokens per tile (kb64 0.34-0.81x)."""
    if inputs.is_fp8 or inputs.nheads_per_kv > 64:
        return False
    return (
        inputs.is_topk_gather
        or inputs.nheads_per_kv == 64
        or (inputs.fwd.num_heads_kv == 1 and inputs.seqlen_q_hint == 1)
    )


def mla_tiles_sparse_heads(inputs: MlaFwdHeuristicInputs) -> bool:
    """Sparse MLA below 128 heads tiles one token's heads (pack_gqa) padded to the kernel tile."""
    return inputs.is_topk_gather and inputs.nheads_per_kv < 128


def mla_qhead_tile(inputs: MlaFwdHeuristicInputs, kernel: MlaKernel) -> int:
    """Q rows per token: sparse MLA pads its heads to the kernel's tile."""
    if not mla_tiles_sparse_heads(inputs):
        return inputs.nheads_per_kv
    min_tile = MLA_2CTA_SPARSE_HEAD_TILE if kernel == "2cta" else MLA_1CTA_SPARSE_HEAD_TILE
    return sparse_mla_qhead_tile(inputs.nheads_per_kv, min_tile=min_tile)


def select_mla_tiles(
    inputs: MlaFwdHeuristicInputs, kernel: MlaKernel, *, qhead_tile: int, pack_gqa: bool
) -> tuple[int, int, int, int]:
    """Return (tile_m, tile_n, q_stage, num_splits) for one MLA kernel.

    The 2CTA kernel runs unsplit. 1CTA sizes the split heuristic with its own tiles: one
    token per 64-row tile on kb64, packed GQA rows on the 128-key mainloop.
    """
    fwd = inputs.fwd
    tile_m = 128 if fwd.requested_tile_m is None else fwd.requested_tile_m
    tile_n = 128 if fwd.requested_tile_n is None else fwd.requested_tile_n
    rows_per_token = qhead_tile if pack_gqa else 1
    q_stage = 2 if fwd.max_seqlen_q * rows_per_token > tile_m else 1
    if kernel == "2cta":
        return tile_m, tile_n, q_stage, 1
    min_blocks = 1
    if kernel == "1cta_kb64":
        if inputs.is_topk_gather or fwd.requested_tile_m is None:
            tile_m, tile_n = MLA_KB64_TILE_MN
        q_stage, rows_per_token = 1, MLA_KB64_TILE_MN[0]
        min_blocks = MLA_KB64_MIN_BLOCKS_PER_SPLIT
    if inputs.is_topk_gather:
        return tile_m, tile_n, q_stage, 1
    if not inputs.auto_split:
        return tile_m, tile_n, q_stage, fwd.requested_num_splits
    num_n_blocks = ceil_div(fwd.max_seqlen_k, tile_n)
    # Tiles per KV head: at most batch x ceil(max rows / tile), and at most the token rows plus
    # one partial tile per extra sequence (varlen without a host max_seqlen_q has
    # max_seqlen_q = total_q, which only bounds a single sequence). Whole-token tiles (kb64:
    # one token per tile) have no partial tiles.
    m_rows = q_stage * tile_m
    partial_tiles = 0 if rows_per_token % m_rows == 0 else fwd.batch_size - 1
    num_tiles = fwd.num_heads_kv * min(
        fwd.batch_size * ceil_div(fwd.max_seqlen_q * rows_per_token, m_rows),
        ceil_div(fwd.total_q * rows_per_token, m_rows) + partial_tiles,
    )
    max_splits = min(128, max(1, num_n_blocks // min_blocks))
    num_splits = num_splits_heuristic(num_tiles, fwd.num_sms, num_n_blocks, max_splits)
    return tile_m, tile_n, q_stage, num_splits


@lru_cache(maxsize=1024)
def select_mla_fwd_config(inputs: MlaFwdHeuristicInputs) -> MlaFwdConfig:
    """Select one fully resolved MLA forward config from host-visible metadata."""
    fwd = inputs.fwd
    use_1cta = mla_1cta_route(inputs, split_kv=inputs.auto_split)
    kernel: MlaKernel = (
        "2cta" if not use_1cta else "1cta_kb64" if can_use_mla_kb64(inputs) else "1cta"
    )
    # The provisional route fixes the padding: dense never pads and sparse never re-routes.
    qhead_tile = mla_qhead_tile(inputs, kernel)
    if mla_tiles_sparse_heads(inputs):
        pack_gqa = True
    else:
        pack_gqa = qhead_tile > 1 if fwd.pack_gqa is None else fwd.pack_gqa
    if inputs.is_topk_gather and not inputs.auto_split and fwd.requested_num_splits > 1:
        raise ValueError(
            f"num_splits={fwd.requested_num_splits} is not supported with gather_kv_indices "
            "(sparse MLA runs without split-KV)"
        )
    tiles = select_mla_tiles(inputs, kernel, qhead_tile=qhead_tile, pack_gqa=pack_gqa)
    # Too few auto splits for 1CTA to beat 2CTA (AI/SPARSE_MLA_1CTA.md "Auto split"): about
    # 2 on kb64 and 5 on the 128-key mainloop. Route again unsplit.
    min_auto_splits = 2 if kernel == "1cta_kb64" else 5
    if (
        use_1cta
        and inputs.auto_split
        and not inputs.is_topk_gather
        and tiles[3] < min_auto_splits
        and not mla_1cta_route(inputs, split_kv=False)
    ):
        assert qhead_tile == inputs.nheads_per_kv  # dense: no head padding to redo
        kernel = "2cta"
        tiles = select_mla_tiles(inputs, kernel, qhead_tile=qhead_tile, pack_gqa=pack_gqa)
    tile_m, tile_n, q_stage, num_splits = tiles
    # Normalize only after the final route: kb64 always packs, even if the caller opts out.
    if kernel == "1cta_kb64":
        pack_gqa = True

    seqlen_q_known = inputs.seqlen_q_known
    match kernel:
        case "2cta":
            use_clc_scheduler = True  # the 2CTA kernel always runs the CLC scheduler
        case "1cta":
            # Sparse tiles are one token each with uniform cost: the persistent CLC scheduler
            # overlaps a tile's epilogue with the next tile's gather (~10% at 64 heads).
            # Dense runs without it unless an explicit MlaFwdConfig asks for it.
            use_clc_scheduler = inputs.is_topk_gather
        case "1cta_kb64":
            # Dense: CLC measured +2-10% on prefill (many tiles per KV stream) and neutral on
            # decode; varlen without a max_seqlen_q hint counts as decode.
            use_clc_scheduler = inputs.is_topk_gather or (
                seqlen_q_known is not None and seqlen_q_known > 1
            )
    # fp8 (a two-block V ring): issue S(n) ahead of PVt(n-1) when the tensor core is the
    # bottleneck, i.e. when many tiles share one KV stream, which then stays L2-resident:
    # >= 8 tiles of 64 rows, seqlen_q x heads >= 512 (prefill / extend: +33-36%). Decode
    # streams each tile's KV from DRAM and needs the loads' one-block look-ahead that S-ahead
    # gives up (-5-16%). Varlen without a max_seqlen_q hint stays in order. kb64 always
    # issues S ahead of PV.
    s_ahead = (
        kernel == "1cta"
        and inputs.is_fp8
        and seqlen_q_known is not None
        and seqlen_q_known * inputs.nheads_per_kv >= 512
    )
    config = MlaFwdConfig(
        kernel=kernel,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stage=q_stage,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        use_clc_scheduler=use_clc_scheduler,
        s_ahead=s_ahead,
    )
    validate_mla_fwd_config(config, inputs)
    return config


@lru_cache(maxsize=1024)
def validate_mla_fwd_config(config: MlaFwdConfig, inputs: MlaFwdHeuristicInputs) -> None:
    """Reject MLA configs the kernels do not support."""
    fwd = inputs.fwd
    if fwd.device_capacity not in (10, 11):
        raise ValueError(f"MLA (qv) forward requires SM100/SM110, got SM{fwd.device_capacity}")
    kernel = config.kernel
    if kernel not in ("2cta", "1cta", "1cta_kb64"):
        raise ValueError(f"Unknown MLA kernel {kernel!r}")
    sparse_heads = mla_tiles_sparse_heads(inputs)
    # Sparse head tiling and kb64 force pack_gqa; selection ignores the caller's preference.
    if (
        not sparse_heads
        and kernel != "1cta_kb64"
        and fwd.pack_gqa is not None
        and fwd.pack_gqa != config.pack_gqa
    ):
        raise ValueError(
            f"pack_gqa={fwd.pack_gqa} disagrees with config.pack_gqa={config.pack_gqa}"
        )
    if config.tile_m <= 0 or config.q_stage not in (1, 2):
        raise ValueError(f"MLA needs a positive tile_m and q_stage 1 or 2, got {config}")
    if config.num_splits < 1 or config.num_splits > 256:
        raise ValueError(f"num_splits must be in [1, 256], got {config.num_splits}")
    if kernel == "2cta":
        if not 0 < inputs.nheads_per_kv <= 128:
            raise ValueError(
                f"The 2CTA MLA kernel needs 1 to 128 Q heads per KV head, got {inputs.nheads_per_kv}"
            )
        # Sparse heads are padded to 128; dense packed heads must divide the cluster tile.
        if config.pack_gqa and not inputs.is_topk_gather and 128 % inputs.nheads_per_kv != 0:
            raise ValueError(
                "The 2CTA MLA kernel with pack_gqa=True needs Q heads per KV head to divide "
                f"128, got {inputs.nheads_per_kv}"
            )
        if inputs.has_descale or config.num_splits > 1:
            raise ValueError(
                "q_descale/k_descale/v_descale and split kv with qv are only supported by the "
                "1CTA MLA kernel (FLASH_ATTENTION_MLA_1CTA=1)"
            )
        if not config.use_clc_scheduler:
            raise ValueError("The 2CTA MLA kernel always uses the CLC scheduler")
    elif not mla_1cta_supported(inputs):
        raise ValueError(
            "The 1CTA MLA kernel supports sparse MLA only up to 64 Q heads, and training only "
            "with gather_bwd_recompute_p"
        )
    if kernel == "1cta_kb64":
        if not config.pack_gqa:
            raise ValueError("The 64-key-block MLA mainloop requires pack_gqa=True")
        if not can_use_mla_kb64(inputs):
            raise ValueError("The 64-key-block 1CTA MLA mainloop does not serve this problem")
        if config.tile_n != 64 or config.q_stage != 1:
            raise ValueError(
                f"The 64-key-block MLA mainloop needs tile_n=64 and q_stage=1, got {config}"
            )
    elif config.tile_n != 128:
        raise ValueError(f"The {kernel} MLA kernel needs tile_n=128, got {config.tile_n}")
    if config.s_ahead and (kernel != "1cta" or not inputs.is_fp8):
        raise ValueError("s_ahead is the fp8 schedule of the 1CTA 128-key MLA kernel")
    if inputs.is_topk_gather:
        if config.num_splits != 1:
            raise ValueError("Sparse MLA runs without split-KV")
        if not config.pack_gqa:
            raise ValueError("Sparse MLA packs one token's Q heads (pack_gqa)")
        if sparse_heads and fwd.num_heads_kv != 1:
            raise ValueError("sparse MLA requires a single KV head")


# === Backward ===
# Backward selection mirrors the forward: host-only BwdHeuristicInputs, one fully resolved
# BwdConfig, and validate_bwd_config rejecting (never normalizing) explicit configs. The
# sparse MLA backward uses its own policy in interface.py.


@dataclass(frozen=True)
class Sm90BwdRegisters:
    """Per-thread setmaxnreg targets for the SM90 backward warp-group roles.

    MMA warp groups that compute dQ raise to ``mma``; with dQ_single_wg the other MMA warp
    group raises to ``mma_no_dq``; the producer warp group lowers to ``producer``.
    """

    mma: int
    mma_no_dq: int
    producer: int


@dataclass(frozen=True)
class Sm100BwdRegisters:
    """Per-thread setmaxnreg targets for the SM100/SM110 backward warp roles.

    The dQ reduce warp group and the two compute warp groups raise; the load, MMA, and empty
    warps lower. The relay warp lowers to ``mma`` with 2CTA and to ``empty`` otherwise.
    """

    reduce: int
    compute: int
    load: int
    mma: int
    empty: int


# None for kernels that do not split registers by role (SM120, dedicated SM100 hdim 256).
BwdRegisters = Sm90BwdRegisters | Sm100BwdRegisters | None


@dataclass(frozen=True)
class BwdConfig:
    """Fully resolved backward launch configuration.

    tile_m / tile_n are the main kernel's Q and KV tiles, which also size the preprocess,
    postprocess, and accumulators. The dedicated SM100 hdim-256 kernels run fixed internal
    tiles; its config records the 128x128 tiles the surrounding kernels use. SM90-only MMA
    options (PdS stages, SdP_swapAB / AtomLayoutMSdP, dQ_single_wg) hold neutral values on
    other architectures. cluster_size differs from use_2cta_instrs only for the dedicated
    hdim-256 kernel, which always uses 2CTA instructions.
    """

    device_capacity: int
    tile_m: int
    tile_n: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_threads: int
    dQ_single_wg: bool
    V_in_regs: bool
    use_2cta_instrs: bool
    cluster_size: int
    spt: bool
    use_dedicated_hd256_kernel: bool
    registers: BwdRegisters

    @property
    def num_wg(self) -> int:
        """SM90 MMA warp groups; the producer warp group is the remaining 128 threads."""
        return self.num_threads // 128 - 1


class BwdHeuristicInputs(NamedTuple):
    """Host metadata for backward config selection and validation.

    Holds only the facts the policy reads, so the selection cache key does not vary with
    sequence lengths or batch size. sparse_block_size_q / sparse_block_size_kv are the
    caller's block_size, None when unset.
    """

    device_arch: int
    head_dim: int
    head_dim_v: int
    num_heads: int
    num_heads_kv: int
    causal: bool = False
    local: bool = False
    deterministic: bool = False
    use_block_sparsity: bool = False
    sparse_block_size_q: int | None = None
    sparse_block_size_kv: int | None = None
    sparse_spt: bool | None = None
    requested_V_in_regs: bool = False
    # FLASH_ATTENTION_BWD_TILE_N: 64 requests the SM100 2CTA 64-row KV tile.
    requested_tile_n: int | None = None
    disable_2cta: bool = False
    # FLASH_ATTENTION_HD256_GENERIC_BWD: SM100 hdim 256 on FlashAttentionBackwardSm100.
    hd256_generic: bool = False

    @property
    def device_capacity(self) -> int:
        return self.device_arch // 10

    @property
    def qhead_per_kvhead(self) -> int:
        return self.num_heads // self.num_heads_kv

    @property
    def is_hdim256(self) -> bool:
        return self.head_dim == 256 and self.head_dim_v == 256


def bwd_sparse_q_block_size(inputs: BwdHeuristicInputs) -> int | None:
    """Sparse Q block size the backward tiles must divide; SM90 defaults to 128."""
    if not inputs.use_block_sparsity:
        return None
    if inputs.sparse_block_size_q is not None:
        return inputs.sparse_block_size_q
    return 128 if inputs.device_capacity == 9 else None


def bwd_q_subtile_factor(config: BwdConfig, inputs: BwdHeuristicInputs) -> int:
    """Q tiles per sparse Q block; unset sizes keep the legacy factor of 2."""
    sparse_q = bwd_sparse_q_block_size(inputs)
    return sparse_q // config.tile_m if sparse_q is not None else 2


def bwd_kv_subtile_factor(sparse_block_size_kv: int | None, tile_n: int) -> int:
    """Physical KV tiles covered by one sparse KV block."""
    if sparse_block_size_kv is None:
        return 1
    if sparse_block_size_kv % tile_n != 0:
        raise ValueError(
            "Block sparsity expects sparse_block_size[1] "
            f"to be a multiple of tile_n={tile_n}; got {sparse_block_size_kv}."
        )
    return sparse_block_size_kv // tile_n


def config_kv_subtile_factor(config: BwdConfig, inputs: BwdHeuristicInputs) -> int:
    """KV tiles per sparse KV block; SM90 and SM120 map one sparse block per tile."""
    if config.device_capacity not in (10, 11) or not inputs.use_block_sparsity:
        return 1
    return bwd_kv_subtile_factor(inputs.sparse_block_size_kv, config.tile_n)


def bwd_spt(inputs: BwdHeuristicInputs) -> bool:
    """SM90/SM100 deterministic shortest-processing-time dQ lock order.

    SM100 block sparsity may name it; the SM90 kernel has no deterministic block-sparse path.
    """
    if inputs.device_capacity != 9 and inputs.use_block_sparsity and inputs.sparse_spt is not None:
        return inputs.sparse_spt and inputs.deterministic
    return (inputs.causal or inputs.local) and inputs.deterministic


def bwd_tile_hdim(head_dim: int) -> int:
    """SM100 backward head-dim tile: padded to a multiple of 32."""
    return ceil_div(head_dim, 32) * 32


def can_use_bwd_2cta(inputs: BwdHeuristicInputs, *, tile_n: int) -> bool:
    """Whether the SM100 generic backward supports 2CTA instructions for this problem."""
    # 2CTA tiles pair adjacent KV tiles, so a sparse KV block must cover an even count.
    sparse_ok = (
        not inputs.use_block_sparsity
        or bwd_kv_subtile_factor(inputs.sparse_block_size_kv, tile_n) % 2 == 0
    )
    return inputs.head_dim >= 128 and sparse_ok


def can_use_bwd_tile_n_64(inputs: BwdHeuristicInputs, *, use_2cta_instrs: bool) -> bool:
    """Whether the SM100 generic backward supports a 64-row KV tile per CTA.

    See NOTE [M=64 accumulator layout] in flash_bwd_sm100.py: only with 2CTA, for hdim 256
    (serial schedule) or the pipelined schedule (hdim <= 128) without block sparsity, where
    the dK / dV epilogue needs 32-padded head dims that are multiples of 64.
    """
    if not use_2cta_instrs:
        return False
    if inputs.is_hdim256:
        return True
    return (
        inputs.head_dim <= 128
        and inputs.head_dim_v <= 128
        and all(bwd_tile_hdim(d) % 64 == 0 for d in (inputs.head_dim, inputs.head_dim_v))
        and not inputs.use_block_sparsity
    )


def bwd_needs_2cta(inputs: BwdHeuristicInputs) -> str | None:
    """Name the SM100 generic-backward problem that only the 2CTA serial schedule supports."""
    if inputs.is_hdim256:
        return "SM100 backward with head_dim=256"
    if bwd_tile_hdim(inputs.head_dim) == 192:
        if inputs.use_block_sparsity and inputs.head_dim == 192:
            return "SM100 block-sparse backward with head_dim=192"
        return f"SM100 backward with head_dim={inputs.head_dim}"
    return None


# fmt: off
def select_sm90_bwd_tile(
    head_dim: int,
    head_dim_v: int,
    causal: bool,
    local: bool,
    sparse_block_size_q: int | None,
) -> dict:
    """Return the SM90 backward tile and MMA fields of BwdConfig.

    Configs based on C++ FA3 hopper/flash_bwd_launch_template.h, benchmarked on H100 SXM.
    """
    if head_dim <= 64:
        # C++ FA3: 128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return dict(
            tile_m=128, tile_n=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3: 64, 128, 96, dQ_swapAB=False
        return dict(
            tile_m=64, tile_n=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        tile_m = 64 if causal or local else 80
        if sparse_block_size_q is not None and sparse_block_size_q % tile_m != 0:
            tile_m = 64
        return dict(
            tile_m=tile_m, tile_n=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=tile_m % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        return dict(
            tile_m=64, tile_n=96,
            num_stages_Q=2, num_stages_dO=2 if head_dim_v <= 128 else 1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    # hdim 256. The smem dQaccum staging buffer (64 KB) leaves no room for the C++ FA3
    # tile (64x80, 2-stage Q); 64x48 is the largest tile_n that fits a 2-stage Q pipeline,
    # and dKV_swapAB (hdim on WGMMA's M axis) is what allows tile_n=48. It also rounds
    # tile_hdim to 64, which is what makes head_dim in (192, 256) compile.
    if sparse_block_size_q is not None:
        # Block sparsity derives its KV block size from tile_n; keep it at 64.
        return dict(
            tile_m=64, tile_n=64,
            num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )
    return dict(
        tile_m=64, tile_n=48,
        num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
        SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
        AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
    )
# fmt: on


def select_sm90_bwd_registers(*, num_wg: int, dQ_single_wg: bool) -> Sm90BwdRegisters:
    if num_wg == 2:
        if dQ_single_wg:
            return Sm90BwdRegisters(mma=256, mma_no_dq=224, producer=24)
        return Sm90BwdRegisters(mma=240, mma_no_dq=240, producer=24)
    return Sm90BwdRegisters(mma=160, mma_no_dq=160, producer=32)


def select_sm100_bwd_registers(
    inputs: BwdHeuristicInputs, *, use_2cta_instrs: bool
) -> Sm100BwdRegisters:
    # Both 2CTA schedules use the same split.
    if use_2cta_instrs:
        return Sm100BwdRegisters(reduce=136, compute=136, load=104, mma=104, empty=24)
    if (inputs.causal or inputs.local) and not inputs.deterministic:
        return Sm100BwdRegisters(reduce=136, compute=144, load=88, mma=88, empty=24)
    return Sm100BwdRegisters(reduce=152, compute=136, load=88, mma=88, empty=24)


# Architecture-specific options at their neutral values, for architectures that do not
# expose them: the SM90 MMA options, and the SM100 2CTA options. spt (SM90 and SM100) is
# always bwd_spt(inputs) there and False on SM120.
_SM90_ONLY_NEUTRAL_FIELDS = dict(
    num_stages_PdS=1,
    SdP_swapAB=False,
    dKV_swapAB=False,
    dQ_swapAB=False,
    AtomLayoutMSdP=1,
    AtomLayoutNdKV=1,
    AtomLayoutMdQ=1,
    dQ_single_wg=False,
    V_in_regs=False,
)
_SM100_ONLY_NEUTRAL_FIELDS = dict(
    use_2cta_instrs=False,
    cluster_size=1,
    use_dedicated_hd256_kernel=False,
)
# The SM100 kernel launches 16 warps: dQ reduce, 2x compute, MMA, load, relay, and empty.
SM100_BWD_THREADS_PER_CTA = 512


def sm120_bwd_fields(inputs: BwdHeuristicInputs) -> dict:
    """The one SM120 backward configuration: SM80 MMA, 99 KB SMEM, 128 threads (4 warps)."""
    stages = 2 if inputs.head_dim <= 64 else 1
    return {
        **_SM90_ONLY_NEUTRAL_FIELDS,
        **_SM100_ONLY_NEUTRAL_FIELDS,
        "tile_m": 64,
        "tile_n": 64,
        "num_stages_Q": stages,
        "num_stages_dO": stages,
        "AtomLayoutMSdP": 4,
        "AtomLayoutNdKV": 4,
        "AtomLayoutMdQ": 4,
        "num_threads": 128,
        "spt": False,
    }


@lru_cache(maxsize=1024)
def select_bwd_config(inputs: BwdHeuristicInputs) -> BwdConfig:
    """Select one fully resolved backward config from host-visible metadata."""
    fields = {**_SM90_ONLY_NEUTRAL_FIELDS, **_SM100_ONLY_NEUTRAL_FIELDS}
    match inputs.device_capacity:
        case 9:
            fields.update(
                select_sm90_bwd_tile(
                    inputs.head_dim,
                    inputs.head_dim_v,
                    inputs.causal,
                    inputs.local,
                    bwd_sparse_q_block_size(inputs),
                ),
                num_threads=3 * 128,
                V_in_regs=inputs.requested_V_in_regs,
                spt=bwd_spt(inputs),
            )
            registers = select_sm90_bwd_registers(
                num_wg=fields["num_threads"] // 128 - 1, dQ_single_wg=fields["dQ_single_wg"]
            )
        case 10 | 11:
            hd256_generic = inputs.is_hdim256 and inputs.hd256_generic
            # hdim 256 on the generic kernel runs a 64-row KV tile per CTA (NOTE [M=64
            # accumulator layout] in flash_bwd_sm100.py); the 2CTA decision sees that tile.
            tile_n = 64 if hd256_generic else 128
            use_2cta_instrs = not inputs.disable_2cta and can_use_bwd_2cta(inputs, tile_n=tile_n)
            if (
                inputs.requested_tile_n == 64
                and not inputs.is_hdim256
                and can_use_bwd_tile_n_64(inputs, use_2cta_instrs=use_2cta_instrs)
            ):
                tile_n = 64
            dedicated = inputs.is_hdim256 and not inputs.hd256_generic
            # The dedicated hdim-256 kernel always uses 2CTA instructions; its cluster follows
            # the generic 2CTA decision.
            any_2cta = use_2cta_instrs or dedicated
            fields.update(
                tile_m=128,
                tile_n=tile_n,
                num_stages_Q=1 if any_2cta else 2,
                num_stages_dO=1,
                num_threads=SM100_BWD_THREADS_PER_CTA,
                use_2cta_instrs=any_2cta,
                cluster_size=2 if use_2cta_instrs else 1,
                spt=bwd_spt(inputs),
                use_dedicated_hd256_kernel=dedicated,
            )
            registers = (
                None
                if dedicated
                else select_sm100_bwd_registers(inputs, use_2cta_instrs=use_2cta_instrs)
            )
        case 12:
            fields = sm120_bwd_fields(inputs)
            registers = None
        case _:
            raise ValueError(f"Unsupported backward architecture family SM{inputs.device_capacity}")
    config = BwdConfig(device_capacity=inputs.device_capacity, registers=registers, **fields)
    validate_bwd_config(config, inputs)
    return config


def validate_bwd_registers(config: BwdConfig) -> None:
    """Reject register splits the backward kernels cannot apply with setmaxnreg."""
    registers = config.registers
    if config.device_capacity == 9:
        expected = Sm90BwdRegisters
    elif config.device_capacity in (10, 11) and not config.use_dedicated_hd256_kernel:
        expected = Sm100BwdRegisters
    else:
        expected = type(None)
    if type(registers) is not expected:
        raise ValueError(
            f"This SM{config.device_capacity} backward needs registers of type "
            f"{expected.__name__}, got {type(registers).__name__}"
        )
    if registers is None:
        return
    if any(v % 8 != 0 or not 24 <= v <= 256 for v in astuple(registers)):
        raise ValueError(f"Register counts must be multiples of 8 in [24, 256], got {registers}")
    if isinstance(registers, Sm90BwdRegisters):
        num_wg = config.num_wg
        num_wg_dq = 1 if config.dQ_single_wg else num_wg
        if not config.dQ_single_wg and registers.mma_no_dq != registers.mma:
            raise ValueError(
                f"Without dQ_single_wg every MMA warp group uses mma registers, got {registers}"
            )
        launch = 65536 // config.num_threads
        raised = (registers.mma, registers.mma_no_dq)
        if registers.producer > launch or min(raised) < launch:
            raise ValueError(
                f"SM90 MMA registers must be >= {launch} and producer <= {launch}, got {registers}"
            )
        budget = num_wg_dq * registers.mma + (num_wg - num_wg_dq) * registers.mma_no_dq
        budget += registers.producer
        limit = 504 if num_wg == 2 else 512
    else:
        launch = 65536 // SM100_BWD_THREADS_PER_CTA
        if (
            min(registers.reduce, registers.compute) < launch
            or max(registers.load, registers.mma, registers.empty) > launch
        ):
            raise ValueError(
                f"SM100 reduce/compute registers must be >= {launch} and load/mma/empty "
                f"<= {launch}, got {registers}"
            )
        # One warp group each for reduce and the lowered warps, two for compute.
        budget = registers.reduce + 2 * registers.compute
        budget += max(registers.load, registers.mma, registers.empty)
        limit = 512
    if budget > limit:
        raise ValueError(f"Register split {registers} exceeds {limit} per thread (uses {budget})")


def validate_sm90_bwd_mma(config: BwdConfig, inputs: BwdHeuristicInputs) -> None:
    """Mirror FlashAttentionBackwardSm90's stage, warp-group, and WGMMA tiling requirements."""
    if config.num_threads not in (384, 512):
        raise ValueError(
            f"SM90 backward needs 2 or 3 MMA warp groups (384 or 512 threads), "
            f"got {config.num_threads}"
        )
    num_wg = config.num_wg
    if num_wg != 2 and (config.dQ_single_wg or inputs.qhead_per_kvhead > 1):
        raise ValueError("SM90 dQ_single_wg and GQA backward need 2 MMA warp groups")
    if inputs.qhead_per_kvhead > 1 and inputs.head_dim != inputs.head_dim_v:
        raise ValueError("SM90 GQA backward requires head_dim == head_dim_v")
    Q_stage = config.num_stages_Q
    if config.num_stages_dO not in (1, Q_stage) or config.num_stages_PdS not in (1, Q_stage):
        raise ValueError(
            f"SM90 backward needs dO and PdS stages of 1 or num_stages_Q={Q_stage}, got "
            f"{(config.num_stages_dO, config.num_stages_PdS)}"
        )
    # Swapped dK/dV accumulators put head_dim on WGMMA's 64-row M axis.
    hdim_multiple_of = 64 if config.dKV_swapAB else 16
    tile_hdim = ceil_div(inputs.head_dim, hdim_multiple_of) * hdim_multiple_of
    tile_hdimv = ceil_div(inputs.head_dim_v, hdim_multiple_of) * hdim_multiple_of
    num_wg_dQ = 1 if config.dQ_single_wg else num_wg
    # (GEMM, M, N, warp groups along M, warp groups, swapAB) as in _get_tiled_mma.
    gemms = [
        ("SdP", config.tile_m, config.tile_n, config.AtomLayoutMSdP, num_wg, config.SdP_swapAB),
        ("dK", config.tile_n, tile_hdim, config.AtomLayoutNdKV, num_wg, config.dKV_swapAB),
        ("dV", config.tile_n, tile_hdimv, config.AtomLayoutNdKV, num_wg, config.dKV_swapAB),
        ("dQ", config.tile_m, tile_hdim, config.AtomLayoutMdQ, num_wg_dQ, config.dQ_swapAB),
    ]
    for name, m, n, atom_m, wgs, swap in gemms:
        if atom_m < 1 or wgs % atom_m != 0:
            raise ValueError(
                f"SM90 {name} atom layout {atom_m} must divide its {wgs} MMA warp groups"
            )
        atom_n = wgs // atom_m
        if swap:
            m, n, atom_m, atom_n = n, m, atom_n, atom_m
        # Each warp group issues 64-row WGMMAs with N a multiple of 8.
        if m % (64 * atom_m) != 0 or n % atom_n != 0 or (n // atom_n) % 8 != 0:
            raise ValueError(
                f"SM90 {name} WGMMA (swapAB={swap}) cannot tile {m}x{n} over "
                f"{atom_m}x{atom_n} warp groups of 64-row atoms"
            )


@lru_cache(maxsize=1024)
def validate_bwd_config(config: BwdConfig, inputs: BwdHeuristicInputs) -> None:
    """Reject known unsupported or silently normalized backward config combinations."""
    capacity = inputs.device_capacity
    if capacity not in (9, 10, 11, 12):
        raise ValueError(f"Unsupported backward architecture family SM{capacity}")
    if config.device_capacity != capacity:
        raise ValueError(
            f"Config targets SM{config.device_capacity}, but the problem targets SM{capacity}"
        )
    if config.tile_m <= 0 or config.tile_n <= 0:
        raise ValueError(f"Tile dimensions must be positive, got {(config.tile_m, config.tile_n)}")
    if config.tile_m % 16 != 0 or config.tile_n % 16 != 0:
        raise ValueError(
            f"Tile dimensions must be multiples of 16, got {(config.tile_m, config.tile_n)}"
        )
    if config.num_stages_Q < 1 or config.num_stages_dO < 1 or config.num_stages_PdS < 1:
        raise ValueError("Pipeline stage counts must be positive")
    if capacity not in (10, 11) and any(
        getattr(config, name) != value for name, value in _SM100_ONLY_NEUTRAL_FIELDS.items()
    ):
        raise ValueError(f"SM{capacity} backward does not support SM100 2CTA options")
    if capacity != 12 and config.spt != bwd_spt(inputs):
        raise ValueError(
            f"SM{capacity} backward needs spt={bwd_spt(inputs)} (deterministic dQ lock order)"
        )
    if capacity == 9:
        validate_sm90_bwd_mma(config, inputs)
    elif capacity == 12:
        mismatched = {
            name: value
            for name, value in sm120_bwd_fields(inputs).items()
            if getattr(config, name) != value
        }
        if mismatched:
            raise ValueError(f"SM120 backward supports one configuration; expected {mismatched}")
    else:
        neutral = _SM90_ONLY_NEUTRAL_FIELDS
        if any(getattr(config, name) != value for name, value in neutral.items()):
            raise ValueError(
                "SM100 backward does not expose SM80/SM90 MMA options: "
                + ", ".join(f"{name}={value}" for name, value in neutral.items())
            )
        if config.num_threads != SM100_BWD_THREADS_PER_CTA:
            raise ValueError(f"SM100 backward launches {SM100_BWD_THREADS_PER_CTA} threads")
        if config.num_stages_Q != (1 if config.use_2cta_instrs else 2) or config.num_stages_dO != 1:
            raise ValueError("SM100 backward uses 2 Q stages (1 with 2CTA) and 1 dO stage")
        if config.use_dedicated_hd256_kernel:
            if not inputs.is_hdim256:
                raise ValueError("The dedicated SM100 backward kernel requires head_dim=256")
            if not config.use_2cta_instrs or config.cluster_size not in (1, 2):
                raise ValueError(
                    "The dedicated SM100 hdim-256 backward needs 2CTA instructions "
                    "and cluster_size 1 or 2"
                )
            if (config.tile_m, config.tile_n) != (128, 128):
                raise ValueError("The dedicated SM100 hdim-256 backward uses 128x128 tiles")
            # The dedicated kernel checks the sparse KV size against tile_n like the generic one.
            config_kv_subtile_factor(config, inputs)
        else:
            if config.cluster_size != (2 if config.use_2cta_instrs else 1):
                raise ValueError("SM100 backward uses cluster_size=2 exactly with 2CTA")
            if config.tile_m != 128 or config.tile_n not in (64, 128):
                raise ValueError(
                    f"SM100 backward requires tile_m=128 and tile_n in (64, 128), "
                    f"got {(config.tile_m, config.tile_n)}"
                )
            config_kv_subtile_factor(config, inputs)
            can_2cta = can_use_bwd_2cta(inputs, tile_n=config.tile_n)
            needs_2cta = bwd_needs_2cta(inputs)
            if needs_2cta is not None and not config.use_2cta_instrs:
                reason = (
                    "2CTA was disabled by request"
                    if inputs.disable_2cta or can_2cta
                    else (
                        f"sparse_block_size[1] must cover an even number of "
                        f"tile_n={config.tile_n} tiles; got factor "
                        f"{config_kv_subtile_factor(config, inputs)}"
                    )
                )
                raise ValueError(f"{needs_2cta} requires 2CTA; {reason}.")
            if config.use_2cta_instrs and not can_2cta:
                raise ValueError("2CTA instructions are not supported for this backward problem")
            if config.tile_n == 64 and not can_use_bwd_tile_n_64(
                inputs, use_2cta_instrs=config.use_2cta_instrs
            ):
                raise ValueError("tile_n=64 is not supported for this SM100 backward problem")
            if inputs.is_hdim256 and config.tile_n != 64:
                raise ValueError("SM100 generic backward with head_dim=256 needs tile_n=64")
    sparse_q = bwd_sparse_q_block_size(inputs)
    if sparse_q is not None and (sparse_q <= 0 or sparse_q % config.tile_m != 0):
        raise ValueError(
            "Block sparsity requires sparse_block_size[0] to be a multiple of "
            f"tile_m={config.tile_m}; got {sparse_q}."
        )
    validate_bwd_registers(config)
