# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Host-only forward, MLA forward, and backward config selection and validation.

FwdConfig, MlaFwdConfig, and BwdConfig are fully resolved launch choices; forward
configs use exact split counts (1 disables SplitKV). Explicit configs are validated,
not normalized. Selection must not read tensor contents, synchronize CUDA, or
allocate device memory.
"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal, NamedTuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


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


# None for architectures whose forward kernels do not split registers by role, and for SwapAB,
# whose kernel owns fixed per-role budgets.
Registers = Sm90Registers | Sm100Registers | None


@dataclass(frozen=True)
class FwdConfig:
    """Fully resolved forward launch configuration.

    Dynamic persistence is derived from scheduler metadata, not is_static_persistent.
    num_splits and tile_n affect reduction order. With causal/local SplitKV, the
    per-split KV range also depends on the M tile extent, q_stage * tile_m.
    ex2_emu_freq and ex2_emu_start_frg are the SM100/SM110 softmax exp2 emulation
    (freq 0 keeps hardware exp2 throughout); other architectures carry 0 and 0.
    overlap_sO_sQ and kv_stage are the SM100 forward SMEM layout (see
    sm100_fwd_smem_bytes); other architectures carry False and 0.
    use_swapab selects the SM100/SM103 SwapAB decode kernel (flash_fwd_decode_sm100.py), where
    tile_m is the query-head tile and only num_splits is tunable; see SwapABPolicy.config. The
    SwapAB kernel sizes its own SMEM stages, so its configs carry the neutral layout.
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
    ex2_emu_freq: int
    ex2_emu_start_frg: int
    overlap_sO_sQ: bool
    kv_stage: int
    use_swapab: bool = False


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
    disable_scheduler_metadata: bool = False
    # FLASH_ATTENTION_SM100_BACKEND: None selects SwapAB when admitted and not measured slower,
    # True prefers it wherever admitted (other calls fall back to the standard kernel), False
    # disables it.
    requested_use_swapab: bool | None = None
    # Per-sequence KV capacity, the page-table row when paged. The interface always sets it;
    # None (max_seqlen_k) exists for CPU callers that build inputs by hand.
    seqlen_k_capacity: int | None = None
    requires_grad: bool = False  # any input, aux tensors included
    has_aux_scalars: bool = False

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
    def has_varlen_q_semaphore(self) -> bool:
        """A varlen-Q tile-count semaphore selects the dynamic persistent scheduler."""
        return self.is_varlen_q and self.has_tile_count_semaphore

    @property
    def qhead_per_kvhead(self) -> int:
        return self.num_heads // self.num_heads_kv

    @property
    def packed_max_seqlen_q(self) -> int:
        return self.max_seqlen_q * (self.qhead_per_kvhead if self.pack_gqa else 1)

    @property
    def head_dim_padded(self) -> int:
        return round_up(self.head_dim, 16)

    @property
    def head_dim_v_padded(self) -> int:
        return round_up(self.head_dim_v, 16)

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
# select_ex2_emu reads the ex2 fields; select_sm100_registers reads the register fields and
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


def select_ex2_emu(inputs: FwdHeuristicInputs, *, use_2cta_instrs: bool) -> tuple[int, int]:
    """Return (ex2_emu_freq, ex2_emu_start_frg) for the SM100/SM110 forward softmax.

    Untuned keys emulate up to head dim 128, and at 192 only for 2CTA non-causal non-local;
    the SM103 family keeps hardware exp2. Disabled emulation, and every other architecture,
    is (0, 0).
    """
    if inputs.device_capacity not in (10, 11):
        return 0, 0
    head_dim_padded = inputs.head_dim_padded
    is_sm103 = sm100_is_sm103_family(inputs.device_arch)
    key = (use_2cta_instrs, inputs.causal, head_dim_padded, is_sm103)
    tune = SM100_TUNING.get(key, {})
    fp8_tune = SM100_FP8_TUNING.get(key, {}) if "float8" in inputs.dtype else {}
    if "ex2_emu_freq" in fp8_tune:
        tune = {**tune, **fp8_tune}
    start_frg = tune.get("ex2_emu_start_frg", 1)
    if "ex2_emu_freq" in tune:
        enabled = tune["ex2_emu_freq"] > 0
    else:
        hd192_2cta_noncausal = (
            head_dim_padded == 192 and use_2cta_instrs and not inputs.causal and not inputs.local
        )
        enabled = (head_dim_padded <= 128 or hd192_2cta_noncausal) and not is_sm103
    if not enabled:
        return 0, 0
    freq = tune.get("ex2_emu_freq", 16)
    if inputs.pack_gqa and head_dim_padded > 64 and not inputs.causal and not inputs.local:
        freq = 32 if inputs.is_varlen_q else tune.get("ex2_emu_freq", 10)
    return freq, start_frg


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
    """Resolve the register split after the final tile, stage, split, 2CTA, and CLC choices."""
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


def check_register_record(registers: object, expected: type, kernel: str) -> None:
    """Reject a register record of the wrong type, or counts setmaxnreg cannot apply."""
    if type(registers) is not expected:
        raise ValueError(
            f"{kernel} needs registers of type {expected.__name__}, got {type(registers).__name__}"
        )
    if registers is not None and any(v % 8 != 0 or not 24 <= v <= 256 for v in vars(registers).values()):
        raise ValueError(f"Register counts must be multiples of 8 in [24, 256], got {registers}")


def validate_registers(config: "FwdConfig") -> None:
    """Reject register splits the kernels cannot apply with setmaxnreg."""
    registers = config.registers
    expected = {9: Sm90Registers, 10: Sm100Registers, 11: Sm100Registers}.get(
        config.device_capacity, type(None)
    )
    check_register_record(registers, expected, f"SM{config.device_capacity}")
    if registers is None:
        return
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
    if num_splits > 1 or use_clc_scheduler or inputs.has_varlen_q_semaphore:
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


# CUDA's launch limit on grid Y and Z (grid X allows 2**31 - 1).
CUDA_MAX_GRID_DIM_YZ = 65535


def can_use_clc(
    inputs: FwdHeuristicInputs, *, tile_n: int, num_splits: int, use_2cta_instrs: bool
) -> bool:
    """Whether the SM100 kernel honors CLC for this layout."""
    return (
        inputs.page_size in (None, tile_n)
        # The CLC grid is (m blocks, heads, batch * splits), without the folding of the
        # nonpersistent grid.
        and (inputs.num_heads_kv if inputs.pack_gqa else inputs.num_heads) <= CUDA_MAX_GRID_DIM_YZ
        and inputs.batch_size * num_splits <= CUDA_MAX_GRID_DIM_YZ
        and not inputs.has_varlen_q_semaphore
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


# === SM100 forward SMEM layout ===
# Host model of FlashAttentionForwardSm100's SharedStorage: barriers and row statistics, then
# 1024-byte aligned sO, sQ, and the sK/sV buffer (K and V alias one buffer per stage).
SM100_SMEM_BYTES = 227 * 1024
# kv_stage selection budget for Q/O plus KV stages; the rest holds barriers and sScale.
_SM100_KV_STAGE_BUDGET_BYTES = 224 * 1024
# Small head dims would otherwise pick up to 52 stages and overflow the barrier region.
SM100_MAX_KV_STAGE = 32
_SM100_SMEM_ALIGN_BYTES = 1024


def sm100_uneven_kv_smem(
    head_dim_padded: int, head_dim_v_padded: int, kv_stage: int, use_2cta_instrs: bool
) -> bool:
    """Whether 1CTA hd192/128 stores three KV stages as [K, V, K] instead of three K-sized stages.

    The stage stride is the mean of the K and V stage sizes, so stages 0 and 2 index directly
    and stage 1 is offset by tile_n * (192 - 128) / 2 elements depending on the phase. The
    offset assumes full K/V tiles; 2CTA halves them per CTA, so 2CTA always uses even stages.
    """
    return (
        not use_2cta_instrs
        and head_dim_padded == 192
        and head_dim_v_padded == 128
        and kv_stage == 3
    )


def sm100_fwd_buffer_bytes(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    tile_n: int,
    q_stage: int,
    num_splits: int,
    use_2cta_instrs: bool,
) -> tuple[int, int, int, int]:
    """Per-CTA (sQ, sO, K stage, V stage) bytes of the SM100 forward kernel."""
    element_bytes = 1 if "float8" in inputs.dtype else 2
    is_split_kv = num_splits > 1
    # SplitKV stores fp32 partials; FP8 inputs produce bf16 output.
    o_bytes = 4 if is_split_kv else 2
    # hd256 SplitKV stages fp32 O in half-width chunks.
    epi_head_dim_v = 128 if is_split_kv and inputs.head_dim_v == 256 else inputs.head_dim_v_padded
    rows = q_stage * tile_m
    q = rows * inputs.head_dim_padded * element_bytes
    o = rows * epi_head_dim_v * o_bytes
    # 2CTA splits each K/V tile across the cluster.
    cta_group_size = 2 if use_2cta_instrs else 1
    k = tile_n * inputs.head_dim_padded * element_bytes // cta_group_size
    v = tile_n * inputs.head_dim_v_padded * element_bytes // cta_group_size
    return q, o, k, v


def select_sm100_smem_layout(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    tile_n: int,
    q_stage: int,
    num_splits: int,
    use_2cta_instrs: bool,
    is_static_persistent: bool,
    use_clc_scheduler: bool,
) -> tuple[bool, int]:
    """Return (overlap_sO_sQ, kv_stage): alias O with Q, then fill the budget with KV stages."""
    head_dim_padded, head_dim_v_padded = inputs.head_dim_padded, inputs.head_dim_v_padded
    # FlashAttentionForwardSm100.is_persistent: static, CLC, or varlen-semaphore dynamic.
    is_persistent = is_static_persistent or use_clc_scheduler or inputs.has_varlen_q_semaphore
    overlap_sO_sQ = (
        (head_dim_padded == 192 and head_dim_v_padded >= 64)
        # Persistent hd256 2CTA separates Q/O to overlap the next Q load with O stores.
        # 1CTA aliases Q/O to preserve KV pipeline depth.
        or (head_dim_padded == 256 and not (is_persistent and use_2cta_instrs))
        or (head_dim_v_padded >= 128 and num_splits > 1)
    )
    q, o, k, v = sm100_fwd_buffer_bytes(
        inputs,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stage=q_stage,
        num_splits=num_splits,
        use_2cta_instrs=use_2cta_instrs,
    )
    q_o = max(q, o) if overlap_sO_sQ else q + o
    kv_stage = min((_SM100_KV_STAGE_BUDGET_BYTES - q_o) // max(k, v), SM100_MAX_KV_STAGE)
    # 1CTA hd192/128 fits a third stage with the uneven [K, V, K] layout.
    if sm100_uneven_kv_smem(head_dim_padded, head_dim_v_padded, kv_stage + 1, use_2cta_instrs):
        kv_stage = 3
    return overlap_sO_sQ, kv_stage


def sm100_fwd_smem_bytes(config: FwdConfig, inputs: FwdHeuristicInputs) -> int:
    """Dynamic SMEM bytes of FlashAttentionForwardSm100's SharedStorage for config."""
    q_stage, kv_stage = config.q_stage, config.kv_stage
    s_p_o_stage = 2 if config.use_s_ping_pong else q_stage
    dynamic_persistent = config.use_clc_scheduler or inputs.has_varlen_q_semaphore
    # Int64 mbarriers: load Q and KV, four S/P/O pipelines, O epilogue, load epilogue,
    # S0/S1 sequence, TMEM dealloc.
    num_mbarriers = (
        2 * q_stage
        + 2 * kv_stage
        + 4 * 2 * s_p_o_stage
        + 2 * q_stage
        + (2 if config.overlap_sO_sQ else 0)
        + 2 * 2
        + 1
    )
    header = 8 * num_mbarriers + 4  # Int32 TMEM holding buffer
    header += 4 * s_p_o_stage * config.tile_m * 2  # Float32 sScale: row max and row sum
    if dynamic_persistent:  # one scheduler stage: full/empty mbarriers and a 4 x Int32 response
        header = round_up(header, 8) + 2 * 8 + 4 * 4
    q, o, k, v = sm100_fwd_buffer_bytes(
        inputs,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        q_stage=q_stage,
        num_splits=config.num_splits,
        use_2cta_instrs=config.use_2cta_instrs,
    )
    if config.overlap_sO_sQ:
        q, o = max(q, o), 0
    if sm100_uneven_kv_smem(
        inputs.head_dim_padded, inputs.head_dim_v_padded, kv_stage, config.use_2cta_instrs
    ):
        kv_bytes = 2 * k + v  # [K, V, K]
    else:
        kv_bytes = kv_stage * max(k, v)
    align = _SM100_SMEM_ALIGN_BYTES
    total = round_up(header, align)
    for buffer in (o, q, kv_bytes):
        total = round_up(total + buffer, align)
    return total


@dataclass(frozen=True)
class SwapABPolicy:
    """Shapes, admission, and split policy of the SM100/SM103 SwapAB decode kernel
    (flash_fwd_decode_sm100.py). The selector, validation, and interface use SWAPAB."""

    # Kernel shapes. Each 256-token KV block is two 128-row MMA subtiles, and pages are resolved
    # per subtile, so a page holds a whole number of subtiles. One lane owns one query head in
    # the softmax and correction warpgroups.
    tile_n: int = 256
    page_multiple: int = 128
    max_qhead_per_kvhead: int = 32
    # Split policy. Decode is bandwidth bound and DRAM saturates with roughly this many CTAs
    # (hd64 moves fewer bytes per tile, so it needs more); more splits only add per-CTA fixed
    # cost and combine work. Fitted on B200 within 2% of the best split on average over a
    # 22-shape sweep.
    hd64_target_ctas: int = 128
    default_target_ctas: int = 64
    max_splits: int = 32
    min_tokens_per_split: int = 512
    # With per-sequence splits the prepare-scheduler kernel picks each sequence's split count
    # (split_policy) and the grid's split extent only caps it. CTAs past a sequence's count exit
    # at once but still cost a launch each, so the grid holds about grid_ctas_per_sm * num_sms
    # CTAs.
    split_min_blocks: int = 16
    split_ctas_per_sm: int = 4
    grid_ctas_per_sm: int = 8
    # Below this KV capacity the static split count is kept: a short capacity bounds how far
    # lengths can fall below it, so per-sequence counts cannot gain more than the prepare launch
    # costs (~1.5-2.5 us on B300; 256-token pages with a tight page table measured 1.2-1.7x slower
    # at <= 8K tokens and within ~2% at 32K+).
    per_sequence_min_capacity: int = 32768
    # Automatic selection keeps the standard kernel for many short sequences at head dims <= 128:
    # at most short_kv_max_tokens of KV capacity (two KV blocks, so per-CTA fixed cost dominates)
    # and at least short_kv_min_heads (batch * KV heads) CTAs without splitting.
    # Source: B300 CUDA-graph decode sweep (bf16, num_splits=0, dense KV and paged KV whose page
    # table fits the lengths; head configs 8/1, 32/8, 64/8 hd128, 32/4 hd64, 16/2 and 24/4 hd256;
    # batch 1-256, KV 512-128K). SwapAB was 1.12-1.33x slower than the standard kernel on all 10
    # such shapes (KV 512, batch * KV heads 512-2048), and within 1.05x or faster everywhere else
    # except one long-KV shape; hd256 stayed 0.89-0.93x faster. The capacity bounds the real
    # lengths, so large (vLLM-style) page tables never take this rule.
    short_kv_max_tokens: int = 512
    short_kv_min_heads: int = 512

    def supports_head_dims(self, head_dim: int, head_dim_v: int) -> bool:
        """Tested and benchmarked head dims. Adding a shape needs tests and a decode benchmark."""
        return {head_dim, head_dim_v} <= {64, 96, 128} or (head_dim, head_dim_v) == (256, 256)

    def tile_hq(self, qhead_per_kvhead: int) -> int:
        """Query heads per CTA: the next power of two, so one head block covers the group."""
        return 1 << (qhead_per_kvhead - 1).bit_length()

    def target_ctas(self, head_dim: int) -> int:
        return self.hd64_target_ctas if head_dim == 64 else self.default_target_ctas

    def request_veto(self, inputs: FwdHeuristicInputs) -> str | None:
        """The selection request that keeps the selector off SwapAB, even under a SwapAB
        request, or None.

        These steer only select_fwd_config, like the standard kernel's tile and S ping-pong
        requests; an explicit SwapAB config is forced exactly. The default causal CLC scheduler
        (NOTE [causal CLC]) is chosen after SwapAB is declined, so it vetoes nothing.
        """
        requests = {
            "tile_mn": inputs.requested_tile_m is not None or inputs.requested_tile_n is not None,
            "mma_pv_is_rs": inputs.requested_mma_pv_is_rs is not None,
            "intra_wg_overlap": inputs.requested_intra_wg_overlap is not None,
            "pack_gqa": inputs.pack_gqa is False,
            "disable_s_ping_pong": inputs.disable_s_ping_pong,
        }
        return next((reason for reason, vetoed in requests.items() if vetoed), None)

    def rejection(self, inputs: FwdHeuristicInputs) -> str | None:
        """Why the SwapAB decode kernel cannot run this problem, or None when it can.

        SwapAB is inference with one query token per sequence: dense Q of seqlen 1, or varlen Q
        (cu_seqlens_q) with a host max_seqlen_q of 1. A new forward option needs a
        FwdHeuristicInputs fact that is rejected here, unless SwapAB supports it. The split
        count's grid limit is grid_rejection.
        """
        is_dense_q = not inputs.has_cu_seqlens_q
        unsupported = {
            "cu_seqlens_k": inputs.has_cu_seqlens_k,
            "seqused_q": inputs.has_seqused_q,
            "learnable_sink": inputs.has_learnable_sink,
            "block_sparse_tensors": inputs.use_block_sparsity,
            "aux_scalars": inputs.has_aux_scalars,
            "scheduler_metadata": inputs.has_caller_scheduler_metadata,
            "disable_scheduler_metadata": inputs.disable_scheduler_metadata,
            "seqlen_k_per_split": inputs.seqlen_k_per_split is not None,
            # B200 (SM100) and B300 (SM103) share the tcgen05/TMEM features and SMEM capacity used.
            "architecture must be SM100 or SM103": inputs.device_arch not in (100, 103),
            "training": inputs.requires_grad,
            "varlen Q needs max_seqlen_q == 1": not is_dense_q
            and not (inputs.has_max_seqlen_q_hint and inputs.max_seqlen_q == 1),
            "single query token per sequence": is_dense_q and inputs.total_q != inputs.batch_size,
            "dtype": inputs.dtype not in ("torch.float16", "torch.bfloat16"),
            "head dimensions": not self.supports_head_dims(inputs.head_dim, inputs.head_dim_v),
            "GQA ratio": inputs.qhead_per_kvhead > self.max_qhead_per_kvhead,
            f"page size must be a multiple of {self.page_multiple}": (
                inputs.page_size is not None and inputs.page_size % self.page_multiple != 0
            ),
            # The launch grid holds the batch in grid Z, or grid Y when split-major.
            f"batch_size must be at most {CUDA_MAX_GRID_DIM_YZ}": (
                inputs.batch_size > CUDA_MAX_GRID_DIM_YZ
            ),
        }
        return next((reason for reason, rejected in unsupported.items() if rejected), None)

    def grid_rejection(self, inputs: FwdHeuristicInputs, num_splits: int) -> str | None:
        """Why the launch grid of num_splits splits exceeds CUDA's grid limits, or None.

        The default grid holds the (KV head, split) pairs in grid Y. The split-major grid of
        per-sequence splits holds only the splits (at most 256) in grid Z.
        """
        if (
            not self.needs_split_metadata(inputs, num_splits)
            and inputs.num_heads_kv * num_splits > CUDA_MAX_GRID_DIM_YZ
        ):
            return f"num_heads_kv * num_splits must be at most {CUDA_MAX_GRID_DIM_YZ}"
        return None

    def seqlen_k_capacity(self, inputs: FwdHeuristicInputs) -> int:
        return inputs.max_seqlen_k if inputs.seqlen_k_capacity is None else inputs.seqlen_k_capacity

    def slower_than_standard(self, inputs: FwdHeuristicInputs) -> bool:
        """Measured SwapAB losses that automatic selection avoids; see short_kv_max_tokens."""
        return (
            max(inputs.head_dim, inputs.head_dim_v) <= 128
            and self.seqlen_k_capacity(inputs) <= self.short_kv_max_tokens
            and inputs.batch_size * inputs.num_heads_kv >= self.short_kv_min_heads
        )

    def per_sequence_splits(self, inputs: FwdHeuristicInputs) -> bool:
        """Whether SwapAB takes per-sequence split counts from the prepare-scheduler kernel.

        Lengths vary per sequence with varlen Q or seqused_k (the SwapAB tile scheduler reads the
        counts for dense Q too), but only a capacity of per_sequence_min_capacity tokens leaves
        room for the counts to pay for the prepare launch.
        """
        lengths_vary = inputs.is_varlen_q or inputs.has_seqused_k
        return lengths_vary and self.seqlen_k_capacity(inputs) >= self.per_sequence_min_capacity

    def needs_split_metadata(self, inputs: FwdHeuristicInputs, num_splits: int) -> bool:
        """Whether SwapAB prepares per-sequence split counts and launches split-major."""
        return num_splits > 1 and self.per_sequence_splits(inputs)

    def select_num_splits(self, inputs: FwdHeuristicInputs) -> int:
        """Automatic split count; with per-sequence splits, the cap of the per-sequence counts."""
        seqlen_k = self.seqlen_k_capacity(inputs)
        bh = inputs.batch_size * inputs.num_heads_kv
        # An empty batch has no tiles to split, as in num_splits_heuristic.
        if bh == 0:
            return 1
        if self.per_sequence_splits(inputs):
            grid_splits = ceil_div(self.grid_ctas_per_sm * inputs.num_sms, bh)
            return max(1, min(self.max_splits, ceil_div(seqlen_k, self.tile_n), grid_splits))
        pow2_splits = 1 << (max(1, self.target_ctas(inputs.head_dim) // bh).bit_length() - 1)
        return min(pow2_splits, self.max_splits, max(1, seqlen_k // self.min_tokens_per_split))

    def split_policy(self, head_dim: int) -> dict:
        """Split-rule options of prepare_scheduler.num_splits_for_sequence for SwapAB decode.

        The prepare kernel's one-wave rule, at split_ctas_per_sm waves of one CTA per SM, with
        blocks per split raised to min(ceil(tiles / low_ctas), split_min_blocks): small batches
        aim at the static policy's DRAM-saturating CTA count, and splits keep at least
        split_min_blocks blocks until the batch fills the one-wave target.
        """
        return dict(
            ctas_per_sm=self.split_ctas_per_sm,
            min_blocks_floor=(self.target_ctas(head_dim), self.split_min_blocks),
        )

    def config(self, inputs: FwdHeuristicInputs, num_splits: int) -> FwdConfig:
        """The one SwapAB config for a problem: a power-of-two query-head tile covering the GQA
        group, a 256-token KV tile, and no standard-kernel scheduling features."""
        return FwdConfig(
            device_capacity=inputs.device_capacity,
            tile_m=self.tile_hq(inputs.qhead_per_kvhead),
            tile_n=self.tile_n,
            mma_pv_is_rs=False,
            intra_wg_overlap=False,
            q_stage=1,
            pack_gqa=True,
            use_clc_scheduler=False,
            is_static_persistent=False,
            use_tma_o=False,
            use_s_ping_pong=False,
            num_splits=num_splits,
            use_2cta_instrs=False,
            registers=None,
            ex2_emu_freq=0,
            ex2_emu_start_frg=0,
            overlap_sO_sQ=False,
            kv_stage=0,
            use_swapab=True,
        )


SWAPAB = SwapABPolicy()


def needs_split_metadata(config: FwdConfig, inputs: FwdHeuristicInputs) -> bool:
    """Whether the forward must prepare per-sequence split counts (prepare-scheduler kernel).

    num_splits is then the counts' upper bound. Caller-built metadata needs no preparation.
    MLA (qv) does not use this contract.
    """
    if config.use_swapab:
        return SWAPAB.needs_split_metadata(inputs, config.num_splits)
    return (
        config.num_splits > 1
        and inputs.is_varlen_q
        and not inputs.has_caller_scheduler_metadata
        and not inputs.disable_scheduler_metadata
    )


class TunedSm100Overrides(NamedTuple):
    """Measured scheduling wins applied on top of the default policy."""

    clc: bool = False
    nonpersistent: bool = False


def select_tuned_sm100_overrides(
    inputs: FwdHeuristicInputs,
    *,
    tile_n: int,
    num_n_blocks: int,
    is_split_kv: bool,
) -> TunedSm100Overrides:
    """Return measured BF16 output-only scheduling wins for one problem.

    Rules come from benchmarks/fwd_config_bench.py A/B campaigns on the GPU named in
    each arm, and cover only plain output-only BF16 attention on 128x128 nonsplit tiles.
    They tune the standard kernel only: select_fwd_config returns SwapAB decode configs first.
    """
    plain = (
        inputs.dtype == "torch.bfloat16"
        and inputs.page_size is None
        and not inputs.use_block_sparsity
        and not inputs.has_score_mod
        and not inputs.has_mask_mod
        and not inputs.has_learnable_sink
        and not inputs.has_lse
        and tile_n == 128
        and not is_split_kv
        and inputs.head_dim == inputs.head_dim_v
        and not inputs.local
    )
    if not plain:
        return TunedSm100Overrides()
    head_dim, num_heads, batch = inputs.head_dim, inputs.num_heads, inputs.batch_size
    packs_all_q_heads = inputs.qhead_per_kvhead == 1 or inputs.pack_gqa
    dense_noncausal = not inputs.is_varlen and not inputs.causal
    match inputs.device_arch:
        case 103:
            # GB300: nonpersistent scheduling wins once D64 spans at least 32 K tiles.
            nonpersistent = (
                dense_noncausal and head_dim == 64 and packs_all_q_heads and num_n_blocks >= 32
            )
            # CLC wins on balanced MHA or packed high-head varlen and dense causal short-K.
            packed_varlen = (
                inputs.has_cu_seqlens_q
                and inputs.has_cu_seqlens_k
                and not inputs.has_seqused_q
                and not inputs.has_seqused_k
                # Caller metadata may select dynamic scheduling via a tile semaphore.
                and not inputs.has_caller_scheduler_metadata
            )
            balanced_mha = (
                inputs.qhead_per_kvhead == 1
                and not inputs.causal
                and num_heads in (8, 16)
                and 4 <= batch <= 24
                and inputs.total_q >= 10240
                # Mean sequence length at least 40% of the maximum, for Q and K.
                and inputs.total_q * 5 >= 2 * batch * inputs.max_seqlen_q
                and inputs.total_k * 5 >= 2 * batch * inputs.max_seqlen_k
            )
            high_head = (
                num_heads >= 24
                and packs_all_q_heads
                and 3 <= batch <= 64
                and inputs.total_q >= 4096
            )
            dense_short_k = (
                not inputs.is_varlen
                and inputs.causal
                and num_heads >= 24
                and (num_heads % 8 == 0 or inputs.num_heads_kv == 1)
                and (inputs.num_heads_kv != 1 or inputs.max_seqlen_q > 1)
                and packs_all_q_heads
                and batch >= 32
                and 640 <= inputs.max_seqlen_k <= 2048
            )
            clc = head_dim in (64, 96, 128) and (
                (packed_varlen and (balanced_mha or high_head)) or dense_short_k
            )
            return TunedSm100Overrides(clc=clc, nonpersistent=nonpersistent)
    return TunedSm100Overrides()


@lru_cache(maxsize=1024)
def select_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Select one fully resolved config from host-visible metadata."""
    # A SwapAB request overrides only the measured-loss rule: vetoed or unsupported calls take
    # the standard kernel.
    if (
        inputs.requested_use_swapab is not False
        and SWAPAB.request_veto(inputs) is None
        and SWAPAB.rejection(inputs) is None
        and (inputs.requested_use_swapab or not SWAPAB.slower_than_standard(inputs))
    ):
        num_splits = inputs.requested_num_splits or SWAPAB.select_num_splits(inputs)
        if SWAPAB.grid_rejection(inputs, num_splits) is None:
            config = SWAPAB.config(inputs, num_splits)
            validate_fwd_config(config, inputs)
            return config
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

    tuned = select_tuned_sm100_overrides(
        inputs,
        tile_n=tile_n,
        num_n_blocks=num_n_blocks,
        is_split_kv=is_split_kv,
    )
    # CLC runs by default only on measured cells; explicit configs may choose it elsewhere.
    default_clc, _ = causal_clc_policy(inputs, num_splits=num_splits, q_stage=q_stage)
    use_clc_scheduler = (
        is_sm100_family
        and (tuned.clc or default_clc)
        and can_use_clc(
            inputs, tile_n=tile_n, num_splits=num_splits, use_2cta_instrs=use_2cta_instrs
        )
    )
    is_static_persistent = (
        is_sm100_family
        and not tuned.nonpersistent
        and can_use_static_persistent(
            inputs,
            tile_m=tile_m,
            q_stage=q_stage,
            num_splits=num_splits,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
        )
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
    ex2_emu_freq, ex2_emu_start_frg = select_ex2_emu(inputs, use_2cta_instrs=use_2cta_instrs)
    overlap_sO_sQ, kv_stage = (
        select_sm100_smem_layout(
            inputs,
            tile_m=tile_m,
            tile_n=tile_n,
            q_stage=q_stage,
            num_splits=num_splits,
            use_2cta_instrs=use_2cta_instrs,
            is_static_persistent=is_static_persistent,
            use_clc_scheduler=use_clc_scheduler,
        )
        if is_sm100_family
        else (False, 0)
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
        ex2_emu_freq=ex2_emu_freq,
        ex2_emu_start_frg=ex2_emu_start_frg,
        overlap_sO_sQ=overlap_sO_sQ,
        kv_stage=kv_stage,
    )
    validate_fwd_config(config, inputs)
    return config


def derive_fwd_config(base: FwdConfig, inputs: FwdHeuristicInputs, **changes) -> FwdConfig:
    """Apply changes to base and re-derive the fields the selector computes from them.

    registers, the exp2 emulation, and the SM100 SMEM layout (overlap_sO_sQ, kv_stage) are
    recomputed for the changed tile, stage, split, 2CTA, PackGQA, and scheduler choices unless
    changes sets them. The result is validated.
    """
    config = replace(base, **changes)
    resolved = inputs._replace(pack_gqa=config.pack_gqa)
    if config.use_swapab:
        # registers=None: the SwapAB kernel owns its fixed per-role budgets (64/120/208 at
        # tile_hq=32) and its SMEM stages; it keeps hardware exp2.
        derived = dict(
            registers=None, ex2_emu_freq=0, ex2_emu_start_frg=0, overlap_sO_sQ=False, kv_stage=0
        )
    else:
        ex2_emu_freq, ex2_emu_start_frg = select_ex2_emu(
            resolved, use_2cta_instrs=config.use_2cta_instrs
        )
        overlap_sO_sQ, kv_stage = (
            select_sm100_smem_layout(
                resolved,
                tile_m=config.tile_m,
                tile_n=config.tile_n,
                q_stage=config.q_stage,
                num_splits=config.num_splits,
                use_2cta_instrs=config.use_2cta_instrs,
                is_static_persistent=config.is_static_persistent,
                use_clc_scheduler=config.use_clc_scheduler,
            )
            if config.device_capacity in (10, 11)
            else (False, 0)
        )
        derived = dict(
            ex2_emu_freq=ex2_emu_freq,
            ex2_emu_start_frg=ex2_emu_start_frg,
            overlap_sO_sQ=overlap_sO_sQ,
            kv_stage=kv_stage,
        )
    # The register tables cover only supported tiles and stages; validation rejects the rest.
    if "registers" not in changes and not config.use_swapab:
        derived["registers"] = select_registers(
            resolved,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            q_stage=config.q_stage,
            num_splits=config.num_splits,
            use_2cta_instrs=config.use_2cta_instrs,
            use_clc_scheduler=config.use_clc_scheduler,
        )
    config = replace(config, **{k: v for k, v in derived.items() if k not in changes})
    validate_fwd_config(config, inputs)
    return config


# Architecture-specific fields at their neutral values, for architectures that do not expose
# them: the SM90 MMA flags, and the SM100/SM110 stage, split, scheduler, exp2, and SMEM
# layout options.
_FWD_SM90_ONLY_NEUTRAL_FIELDS = dict(mma_pv_is_rs=False, intra_wg_overlap=False)
_FWD_SM100_ONLY_NEUTRAL_FIELDS = dict(
    q_stage=1,
    num_splits=1,
    use_2cta_instrs=False,
    use_clc_scheduler=False,
    is_static_persistent=False,
    use_tma_o=False,
    use_s_ping_pong=False,
    ex2_emu_freq=0,
    ex2_emu_start_frg=0,
    overlap_sO_sQ=False,
    kv_stage=0,
)


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
    if config.use_swapab:
        reason = SWAPAB.rejection(inputs)
        if reason is not None:
            raise ValueError("SwapAB unsupported: " + reason)
        if not 1 <= config.num_splits <= 256:
            raise ValueError(f"num_splits must be in [1, 256], got {config.num_splits}")
        reason = SWAPAB.grid_rejection(inputs, config.num_splits)
        if reason is not None:
            raise ValueError("SwapAB unsupported: " + reason)
        expected = SWAPAB.config(inputs, config.num_splits)
        if config != expected:
            raise ValueError(f"SwapAB tunes only num_splits: expected {expected}, got {config}")
        return
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

    capacity = inputs.device_capacity
    if capacity != 9 and any(
        getattr(config, name) != value for name, value in _FWD_SM90_ONLY_NEUTRAL_FIELDS.items()
    ):
        raise ValueError(f"SM{capacity} forward does not expose SM90 MMA flags")
    if capacity not in (10, 11):
        match capacity:
            case 8 if inputs.page_size is not None:
                raise ValueError("SM80 forward does not support paged KV")
            case 9 if config.tile_m not in (64, 128, 192):
                raise ValueError(f"SM90 tile_m must be 64, 128, or 192, got {config.tile_m}")
            case 12 if inputs.page_size is not None or inputs.use_block_sparsity:
                raise ValueError("SM120 forward does not support paged KV or block sparsity")
        neutral = _FWD_SM100_ONLY_NEUTRAL_FIELDS
        mismatched = [name for name, value in neutral.items() if getattr(config, name) != value]
        if mismatched:
            raise ValueError(
                f"SM{capacity} forward does not support SM100-only options: "
                + ", ".join(f"{n}={getattr(config, n)} (needs {neutral[n]})" for n in mismatched)
            )
        validate_registers(config)
        return

    if config.ex2_emu_freq < 0 or config.ex2_emu_start_frg < 0:
        raise ValueError(
            "ex2_emu_freq and ex2_emu_start_frg must be >= 0, got "
            f"{(config.ex2_emu_freq, config.ex2_emu_start_frg)}"
        )
    # The softmax reads start_frg only when emulating; one encoding for "off" keeps one kernel.
    if config.ex2_emu_freq == 0 and config.ex2_emu_start_frg != 0:
        raise ValueError(
            f"ex2_emu_start_frg must be 0 when ex2_emu_freq is 0, got {config.ex2_emu_start_frg}"
        )
    # Softmax warp groups own 128 rows each; smaller tiles overrun the row statistics.
    if config.tile_m != 128:
        raise ValueError(f"SM100 forward requires tile_m=128, got {config.tile_m}")
    if config.q_stage not in (1, 2):
        raise ValueError(f"SM100 q_stage must be 1 or 2, got {config.q_stage}")
    if 2 * config.tile_n + config.q_stage * inputs.head_dim_v_padded > 512:
        raise ValueError("S and O accumulators exceed the 512 TMEM columns")

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
    validate_sm100_smem_layout(config, inputs)
    validate_registers(config)


def validate_sm100_smem_layout(config: FwdConfig, inputs: FwdHeuristicInputs) -> None:
    """Reject SM100 forward SMEM layouts the kernel cannot pipeline or launch."""
    # K and V share the stage pipeline: with two Q stages the MMA warp waits for K_{i+1}
    # before releasing V_i, and one stage would serialize every load behind its MMA.
    if not 2 <= config.kv_stage <= SM100_MAX_KV_STAGE:
        raise ValueError(
            f"SM100 kv_stage must be in [2, {SM100_MAX_KV_STAGE}], got {config.kv_stage}"
        )
    # The uneven stage-1 offset, tile_n * 32 elements, must keep 1024-element swizzle alignment.
    if (
        sm100_uneven_kv_smem(
            inputs.head_dim_padded,
            inputs.head_dim_v_padded,
            config.kv_stage,
            config.use_2cta_instrs,
        )
        and config.tile_n % 32 != 0
    ):
        raise ValueError(
            f"1CTA head-dim 192/128 with kv_stage=3 needs tile_n divisible by 32, got {config.tile_n}"
        )
    smem_bytes = sm100_fwd_smem_bytes(config, inputs)
    if smem_bytes > SM100_SMEM_BYTES:
        raise ValueError(
            f"SM100 forward layout (tile_n={config.tile_n}, q_stage={config.q_stage}, "
            f"overlap_sO_sQ={config.overlap_sO_sQ}, kv_stage={config.kv_stage}) needs "
            f"{smem_bytes} bytes of shared memory; SM100 provides {SM100_SMEM_BYTES}"
        )


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


class MlaFwdHeuristicInputs(NamedTuple):
    """Host metadata for MLA forward selection: the shared forward facts plus the qv-only bits.

    mla_1cta_override is FLASH_ATTENTION_MLA_1CTA: True runs 1CTA wherever supported, False never.
    """

    fwd: FwdHeuristicInputs
    is_topk_gather: bool = False
    gather_bwd_recompute_p: bool = False
    has_descale: bool = False
    mla_1cta_override: bool | None = None

    @property
    def requires_grad(self) -> bool:
        return self.fwd.requires_grad

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
    # Q rows per token. The provisional route fixes the padding: dense never pads and sparse
    # never re-routes.
    qhead_tile = inputs.nheads_per_kv
    if mla_tiles_sparse_heads(inputs):
        min_tile = MLA_2CTA_SPARSE_HEAD_TILE if kernel == "2cta" else MLA_1CTA_SPARSE_HEAD_TILE
        qhead_tile = sparse_mla_qhead_tile(qhead_tile, min_tile=min_tile)
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
    hdim-256 kernel, which always uses 2CTA instructions. pack_gqa (SM100 generic kernel only)
    folds the query heads of a KV head into the Q tile (NOTE [bwd pack_gqa] in pack_gqa.py);
    the preprocess then writes LSE_log2 / dPsum / dQaccum per KV head in packed row order and
    the postprocess reads them that way. fold_LSE_dPsum (SM100 generic kernel only) folds the
    LSE / dPsum subtractions into the S / dP MMAs (NOTE [hdim64 LSE / dPsum fold into the S / dP
    MMA] in flash_bwd_sm100.py). dQ_after_dS_xchg (SM100 pipelined 2CTA schedule only) issues Q
    before Qt and starts each dQ reduce-add after the next dS exchange copy (NOTE [dQ reduce
    after the dS exchange] in flash_bwd_sm100.py).
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
    pack_gqa: bool
    fold_LSE_dPsum: bool
    dQ_after_dS_xchg: bool
    registers: BwdRegisters

    @property
    def num_wg(self) -> int:
        """SM90 MMA warp groups; the producer warp group is the remaining 128 threads."""
        return self.num_threads // 128 - 1


class BwdHeuristicInputs(NamedTuple):
    """Host metadata for backward config selection and validation.

    Holds only the facts the policy reads. dtype is str(torch.dtype) of the inputs.
    sparse_block_size_q / sparse_block_size_kv are the caller's block_size, None when unset.
    num_sms, batch_size, the sequence bounds, and the token totals feed only the SM100 GQA
    pack_gqa heuristic (bwd_pack_gqa_heuristic); max_seqlen_q / max_seqlen_k fall back to the
    token totals for varlen calls without a host bound, and num_sms is 0 where the heuristic
    cannot run.
    """

    device_arch: int
    head_dim: int
    head_dim_v: int
    num_heads: int
    num_heads_kv: int
    dtype: str
    num_sms: int = 0
    batch_size: int = 1
    max_seqlen_q: int = 1
    max_seqlen_k: int = 1
    total_q: int = 1
    total_k: int = 1
    has_cu_seqlens: bool = False  # cu_seqlens_q or cu_seqlens_k
    causal: bool = False
    local: bool = False
    deterministic: bool = False
    use_block_sparsity: bool = False
    sparse_block_size_q: int | None = None
    sparse_block_size_kv: int | None = None
    sparse_spt: bool | None = None
    has_score_mod: bool = False
    has_score_mod_bwd: bool = False
    has_softcap: bool = False
    has_mask_mod: bool = False
    has_learnable_sink: bool = False
    requested_V_in_regs: bool = False
    # FLASH_ATTENTION_BWD_TILE_N: 64 requests the SM100 2CTA 64-row KV tile; any value
    # (64 or 128) also leaves pack_gqa off.
    requested_tile_n: int | None = None
    # FLASH_ATTENTION_BWD_PACK_GQA=0/1: replaces the pack_gqa heuristic where packing is
    # supported; None leaves it to the heuristic.
    requested_pack_gqa: bool | None = None
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
    return round_up(head_dim, 32)


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


def can_use_bwd_pack_gqa(
    inputs: BwdHeuristicInputs, *, tile_m: int, use_dedicated_hd256_kernel: bool
) -> bool:
    """Whether the SM100 backward supports packing the query heads of a KV head into the Q tile.

    NOTE [bwd pack_gqa] (pack_gqa.py) scope: the generic SM100 kernel, qhead_per_kvhead > 1
    dividing tile_m, without block sparsity, score_mod, mask_mod, or learnable_sink. softcap
    does not count as a score_mod here.
    """
    qhead_per_kvhead = inputs.qhead_per_kvhead
    return (
        inputs.device_capacity in (10, 11)
        and not use_dedicated_hd256_kernel
        and qhead_per_kvhead > 1
        and tile_m % qhead_per_kvhead == 0
        and not inputs.use_block_sparsity
        and not inputs.has_score_mod
        and not inputs.has_mask_mod
        and not inputs.has_learnable_sink
    )


def bwd_pack_gqa_heuristic(
    inputs: BwdHeuristicInputs, *, tile_m: int, tile_n: int, cluster_size: int
) -> bool:
    """Whether the SM100 backward should pack the query heads of a KV head into the Q tile.

    Packing divides the tile count by qhead_per_kvhead. Its per-tile saving (K/V loaded once per
    KV head, dK/dV kept in TMEM, no fp32 dK/dV accumulate + postprocess) shrinks with seqlen_q,
    while the wave-quantization loss of the smaller grid can grow; measured on B200: ~25% gross
    gain at 2k, ~7% at 8k, and a 3.5-wave packed grid loses ~13%.

    A packed tile also loops over qhead_per_kvhead times more Q rows, so the longest packed CTA
    runs qh * ceil(seqlen_q / tile_m) iterations (every KV tile, or the first one with causal).
    When that exceeds the work a cluster slot gets on average, the grid cannot hide it and the
    long tiles serialize: MQA 16/1 on one 8k causal sequence runs 2x slower packed. Varlen only
    knows the longest sequence and the totals, so the average is taken over mean sequence
    lengths; ragged packs (one long sequence plus short ones) then stay unpacked.
    """
    num_sms, batch_size = inputs.num_sms, inputs.batch_size
    seqlen_q, seqlen_k = inputs.max_seqlen_q, inputs.max_seqlen_k
    if num_sms <= 0:
        raise ValueError("The SM100 backward pack_gqa heuristic needs num_sms")
    kv_tile = tile_n * cluster_size
    n_blocks_kv = (seqlen_k + kv_tile - 1) // kv_tile
    slots = max(1, num_sms // cluster_size)

    # critical path of a packed tile vs the average cluster-slot work
    packed_max_iters = inputs.qhead_per_kvhead * ((seqlen_q + tile_m - 1) // tile_m)
    mean_q = inputs.total_q / max(batch_size, 1) if inputs.has_cu_seqlens else seqlen_q
    mean_k = inputs.total_k / max(batch_size, 1) if inputs.has_cu_seqlens else seqlen_k
    total_iters = (
        batch_size
        * inputs.num_heads
        * math.ceil(mean_k / kv_tile)
        * math.ceil(mean_q / tile_m)
        * (0.5 if inputs.causal else 1.0)
    )
    if packed_max_iters > 1.25 * total_iters / slots:
        return False

    def wave_loss(num_ctas):
        waves = num_ctas / num_sms
        return (math.ceil(waves) - waves) / math.ceil(waves)

    ctas_unpacked = batch_size * inputs.num_heads * n_blocks_kv * cluster_size
    ctas_packed = batch_size * inputs.num_heads_kv * n_blocks_kv * cluster_size
    gain = 0.25 * 2048 / max(seqlen_q, 1)
    if inputs.causal or inputs.local:
        # causal / local tiles are uneven and the qh-times longer packed tiles balance
        # worse at the tail: smaller gain, and only pack with a margin
        gain *= 0.6
        gain -= 0.02
    return gain > wave_loss(ctas_packed) - wave_loss(ctas_unpacked)


def select_bwd_pack_gqa(
    inputs: BwdHeuristicInputs,
    *,
    tile_m: int,
    tile_n: int,
    cluster_size: int,
    use_dedicated_hd256_kernel: bool,
) -> bool:
    """SM100 backward pack_gqa: off when FLASH_ATTENTION_BWD_TILE_N is set or unsupported, else
    FLASH_ATTENTION_BWD_PACK_GQA if given, else bwd_pack_gqa_heuristic."""
    if inputs.requested_tile_n is not None or not can_use_bwd_pack_gqa(
        inputs, tile_m=tile_m, use_dedicated_hd256_kernel=use_dedicated_hd256_kernel
    ):
        return False
    if inputs.requested_pack_gqa is not None:
        return inputs.requested_pack_gqa
    return bwd_pack_gqa_heuristic(inputs, tile_m=tile_m, tile_n=tile_n, cluster_size=cluster_size)


def can_fold_LSE_dPsum(
    inputs: BwdHeuristicInputs, *, tile_m: int, tile_n: int, cluster_size: int
) -> bool:
    """Whether the SM100 backward folds the LSE / dPsum subtractions into the S / dP MMAs.

    NOTE [hdim64 LSE / dPsum fold into the S / dP MMA] (flash_bwd_sm100.py) scope: the generic
    kernel's split_P_dS layout (1-CTA, 128 x 128 tiles, both head dims padded to 64), bf16, and
    no block sparsity, score_mod, score_mod_bwd, or softcap. Selected wherever allowed; also MHA
    and non-deterministic only, since GQA and deterministic mode measured slower.
    """
    return (
        inputs.device_capacity in (10, 11)
        and cluster_size == 1
        and (tile_m, tile_n) == (128, 128)
        and bwd_tile_hdim(inputs.head_dim) == 64
        and bwd_tile_hdim(inputs.head_dim_v) == 64
        and inputs.dtype == "torch.bfloat16"
        and inputs.qhead_per_kvhead == 1
        and not inputs.deterministic
        and not inputs.use_block_sparsity
        and not (inputs.has_score_mod or inputs.has_score_mod_bwd or inputs.has_softcap)
    )


def can_use_dQ_after_dS_xchg(
    inputs: BwdHeuristicInputs, *, use_2cta_instrs: bool, use_dedicated_hd256_kernel: bool
) -> bool:
    """Whether the SM100 backward orders Q before Qt and the dQ reduce-adds after the dS exchange.

    NOTE [dQ reduce after the dS exchange] (flash_bwd_sm100.py) scope: the generic kernel's
    pipelined 2CTA schedule (2CTA with head_dim padded to neither 192 nor 256, i.e. hdim 128).
    Selected wherever allowed; also non-deterministic only, since both orderings measured slower
    with the serialized deterministic dQ adds.
    """
    return (
        inputs.device_capacity in (10, 11)
        and use_2cta_instrs
        and not use_dedicated_hd256_kernel
        and bwd_tile_hdim(inputs.head_dim) not in (192, 256)
        and not inputs.deterministic
    )


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
    if head_dim <= 128:
        # C++ FA3: hdim 64: 128x128 (2, 2, true, false, false, 2, 1, 2, 2); hdim 96: 64x128;
        # hdim 128: 64x128 causal/local, else 80x128 with dQ_swapAB.
        tile_m = 128 if head_dim <= 64 else 64
        if head_dim > 96 and not (causal or local):
            tile_m = 80
            if sparse_block_size_q is not None and sparse_block_size_q % 80 != 0:
                tile_m = 64
        return dict(
            tile_m=tile_m, tile_n=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=tile_m == 80,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2 if head_dim <= 64 else 1,
            dQ_single_wg=64 < head_dim <= 96,
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
# expose them: the SM90 MMA options, and the SM100 2CTA, pack_gqa, fold, and dQ-order options.
# spt (SM90 and SM100) is always bwd_spt(inputs) there and False on SM120.
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
    pack_gqa=False,
    fold_LSE_dPsum=False,
    dQ_after_dS_xchg=False,
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
            # Two MMA warp groups; with dQ_single_wg the one without dQ needs fewer registers.
            registers = (
                Sm90BwdRegisters(mma=256, mma_no_dq=224, producer=24)
                if fields["dQ_single_wg"]
                else Sm90BwdRegisters(mma=240, mma_no_dq=240, producer=24)
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
            cluster_size = 2 if use_2cta_instrs else 1
            fields.update(
                tile_m=128,
                tile_n=tile_n,
                num_stages_Q=1 if any_2cta else 2,
                num_stages_dO=1,
                num_threads=SM100_BWD_THREADS_PER_CTA,
                use_2cta_instrs=any_2cta,
                cluster_size=cluster_size,
                spt=bwd_spt(inputs),
                use_dedicated_hd256_kernel=dedicated,
                pack_gqa=select_bwd_pack_gqa(
                    inputs,
                    tile_m=128,
                    tile_n=tile_n,
                    cluster_size=cluster_size,
                    use_dedicated_hd256_kernel=dedicated,
                ),
                fold_LSE_dPsum=can_fold_LSE_dPsum(
                    inputs, tile_m=128, tile_n=tile_n, cluster_size=cluster_size
                ),
                dQ_after_dS_xchg=can_use_dQ_after_dS_xchg(
                    inputs, use_2cta_instrs=any_2cta, use_dedicated_hd256_kernel=dedicated
                ),
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
    check_register_record(registers, expected, f"This SM{config.device_capacity} backward")
    if registers is None:
        return
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
    tile_hdim = round_up(inputs.head_dim, hdim_multiple_of)
    tile_hdimv = round_up(inputs.head_dim_v, hdim_multiple_of)
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
        if m % (64 * atom_m) != 0 or n % (8 * atom_n) != 0:
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
    sm100_only = [
        name for name, value in _SM100_ONLY_NEUTRAL_FIELDS.items() if getattr(config, name) != value
    ]
    if capacity not in (10, 11) and sm100_only:
        raise ValueError(f"SM{capacity} backward does not support the SM100 options {sm100_only}")
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
        if config.pack_gqa and not can_use_bwd_pack_gqa(
            inputs,
            tile_m=config.tile_m,
            use_dedicated_hd256_kernel=config.use_dedicated_hd256_kernel,
        ):
            raise ValueError(
                "SM100 backward pack_gqa needs the generic kernel, qhead_per_kvhead > 1 dividing "
                f"tile_m={config.tile_m}, and no block sparsity, score_mod, mask_mod, or "
                "learnable_sink"
            )
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
        if config.fold_LSE_dPsum and not can_fold_LSE_dPsum(
            inputs, tile_m=config.tile_m, tile_n=config.tile_n, cluster_size=config.cluster_size
        ):
            raise ValueError(
                "SM100 backward fold_LSE_dPsum needs the 1-CTA 128x128 kernel with head dims "
                "padded to 64, bf16, MHA, non-deterministic, and no block sparsity, score_mod, "
                "score_mod_bwd, or softcap"
            )
        if config.dQ_after_dS_xchg and not can_use_dQ_after_dS_xchg(
            inputs,
            use_2cta_instrs=config.use_2cta_instrs,
            use_dedicated_hd256_kernel=config.use_dedicated_hd256_kernel,
        ):
            raise ValueError(
                "SM100 backward dQ_after_dS_xchg needs the generic kernel's pipelined 2CTA "
                "schedule (hdim 128) and a non-deterministic run"
            )
    sparse_q = bwd_sparse_q_block_size(inputs)
    if sparse_q is not None and (sparse_q <= 0 or sparse_q % config.tile_m != 0):
        raise ValueError(
            "Block sparsity requires sparse_block_size[0] to be a multiple of "
            f"tile_m={config.tile_m}; got {sparse_q}."
        )
    validate_bwd_registers(config)
