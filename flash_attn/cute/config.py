# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Typed forward configuration, selection, validation, and compile keys.

`FwdHeuristicInputs` contains immutable host-visible metadata. `FwdConfig` is
fully resolved: one split means the nonsplit path and larger values are exact
SplitKV counts. Explicit configs are validated without ranking or normalization.
The main and combine kernel specs independently project only codegen-changing
values. Selection must not read tensor contents, synchronize CUDA, or allocate.
"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import NamedTuple


class FwdSm90RegisterAllocation(NamedTuple):
    """SM90 per-thread register allocation by warp-group role."""

    mma: int
    producer: int


class FwdSm100RegisterAllocation(NamedTuple):
    """SM100 per-thread register allocation by warp-group role."""

    softmax: int
    correction: int
    other: int


FwdRegisterAllocation = FwdSm90RegisterAllocation | FwdSm100RegisterAllocation


@dataclass(frozen=True)
class FwdConfig:
    """Fully resolved forward launch and algorithm configuration.

    ``is_static_persistent`` controls static persistence; dynamic persistence is
    derived from scheduler metadata. ``registers`` names the warp-group roles
    exposed by the selected architecture and is ``None`` for other kernels.
    """

    device_capacity: int
    tile_m: int
    tile_n: int
    num_stages: int
    num_threads: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    q_stage: int
    use_clc_scheduler: bool
    is_static_persistent: bool
    use_tma_o: bool
    num_splits: int
    use_2cta_instrs: bool
    registers: FwdRegisterAllocation | None

    def __post_init__(self) -> None:
        # JSON turns NamedTuple into a list; campaign YAML may use a mapping.
        if self.registers is None or isinstance(
            self.registers, (FwdSm90RegisterAllocation, FwdSm100RegisterAllocation)
        ):
            return
        register_type = {
            9: FwdSm90RegisterAllocation,
            10: FwdSm100RegisterAllocation,
            11: FwdSm100RegisterAllocation,
        }.get(self.device_capacity)
        if register_type is None:
            raise ValueError(f"SM{self.device_capacity} does not expose register allocation")
        registers = (
            register_type(**self.registers)
            if isinstance(self.registers, dict)
            else register_type(*self.registers)
        )
        object.__setattr__(self, "registers", registers)


class FwdHeuristicInputs(NamedTuple):
    """Host-visible metadata used to select and validate a forward config."""

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
    seqlen_k_per_split: int | None
    causal: bool
    local: bool
    window_size_left: int | None
    window_size_right: int | None
    is_varlen_q: bool
    has_cu_seqlens_q: bool
    has_cu_seqlens_k: bool
    has_seqused: bool
    pack_gqa: bool
    page_size: int | None
    use_block_sparsity: bool
    sparse_q_block_size: int | None
    has_qv: bool
    has_gather_kv: bool
    has_score_mod: bool
    has_mask_mod: bool
    has_learnable_sink: bool
    has_lse: bool
    requested_tile_m: int | None
    requested_tile_n: int | None
    requested_mma_pv_is_rs: bool | None
    requested_intra_wg_overlap: bool | None
    requested_num_splits: int | None
    requested_use_clc_scheduler: bool
    disable_2cta: bool

    @property
    def device_capacity(self) -> int:
        return self.device_arch // 10

    @property
    def qhead_per_kvhead(self) -> int:
        return self.num_heads // self.num_heads_kv

    @property
    def head_dim_padded(self) -> int:
        return math.ceil(self.head_dim / 16) * 16

    @property
    def head_dim_v_padded(self) -> int:
        return math.ceil(self.head_dim_v / 16) * 16


class FwdMainKernelSpec(NamedTuple):
    """Typed cache key for one generated forward main kernel."""

    kernel_family: str
    dtype: object
    head_dim: int
    head_dim_v: int
    qhead_per_kvhead: int
    num_heads_kv: int | None
    causal: bool
    score_mod_hash: object
    mask_mod_hash: object
    use_block_sparsity: bool
    block_sparse_broadcast_pattern: object
    tensor_broadcast_patterns: object
    aux_tensor_metadata: object
    aux_scalar_metadata: object
    has_lse: bool
    has_cu_seqlens_q: bool
    has_cu_seqlens_k: bool
    has_seqused_q: bool
    has_seqused_k: bool
    has_page_table: bool
    has_window_size_left: bool
    has_window_size_right: bool
    learnable_sink_dtype: object
    has_q_descale: bool
    has_k_descale: bool
    has_v_descale: bool
    has_cu_total_m_blocks: bool
    has_cu_block_idx_offsets: bool
    tile_m: int | None
    tile_n: int | None
    num_stages: int | None
    num_threads: int | None
    q_stage: int | None
    is_split_kv: bool | None
    pack_gqa: bool
    arch: int
    paged_kv_non_tma: bool | None
    use_2cta_instrs: bool | None
    q_subtile_factor: int | None
    kv_subtile_factor: int | None
    mma_pv_is_rs: bool | None
    intra_wg_overlap: bool | None
    use_clc_scheduler: bool | None
    is_static_persistent: bool | None
    use_tma_o: bool | None
    registers: FwdRegisterAllocation | None
    has_num_splits_dynamic: bool
    has_virtual_batch_idx: bool
    has_num_nheads_in_l2: bool
    has_tile_count_semaphore: bool
    has_scheduler_cu_total_m_blocks: bool
    has_scheduler_cu_total_splits_m_blocks: bool
    has_blocks_to_batch_idx: bool
    seqlen_k_per_split: int | None
    has_q: bool
    has_p: bool
    has_row_max: bool
    gather_kv_length: int | None
    sparse_kv: bool | None
    disable_sparse_kv_bitmask: bool | None
    log_level: int

    @classmethod
    def from_config(
        cls,
        config: FwdConfig,
        *,
        kernel_family: str,
        arch: int,
        pack_gqa: bool,
        page_size: int | None,
        q_subtile_factor: int,
        kv_subtile_factor: int,
        **kernel: object,
    ) -> "FwdMainKernelSpec":
        """Build a main-kernel key without irrelevant config fields."""
        is_standard = kernel_family in ("sm8", "sm9", "sm10", "sm11", "sm12")
        is_sm90 = kernel_family == "sm9"
        is_sm100 = kernel_family in ("sm10", "sm11")
        return cls(
            kernel_family=kernel_family,
            tile_m=config.tile_m if is_standard else None,
            tile_n=config.tile_n if is_standard else None,
            num_stages=(config.num_stages if kernel_family in ("sm8", "sm9", "sm12") else None),
            num_threads=(config.num_threads if kernel_family in ("sm8", "sm12") else None),
            q_stage=config.q_stage if is_sm100 else None,
            is_split_kv=config.num_splits > 1 if is_sm100 else None,
            pack_gqa=pack_gqa,
            arch=arch,
            paged_kv_non_tma=(
                page_size not in (None, config.tile_n)
                if is_sm90 or is_sm100 or kernel_family == "mla_sm100"
                else None
            ),
            use_2cta_instrs=config.use_2cta_instrs if is_sm100 else None,
            q_subtile_factor=q_subtile_factor if is_sm90 or is_sm100 else None,
            kv_subtile_factor=kv_subtile_factor if is_sm100 else None,
            mma_pv_is_rs=config.mma_pv_is_rs if is_sm90 else None,
            intra_wg_overlap=config.intra_wg_overlap if is_sm90 else None,
            use_clc_scheduler=config.use_clc_scheduler if is_sm100 else None,
            is_static_persistent=config.is_static_persistent if is_sm100 else None,
            use_tma_o=config.use_tma_o if is_sm100 else None,
            registers=(
                config.registers if is_sm90 or is_sm100 or kernel_family == "sm100_hd256" else None
            ),
            **kernel,
        )


class FwdCombineKernelSpec(NamedTuple):
    """Typed cache key for one generated SplitKV combine kernel."""

    arch: int
    dtype: object
    dtype_partial: object
    head_dim: int
    num_head: int
    log_max_splits: int
    has_cu_seqlens: bool
    has_seqused: bool
    has_lse: bool
    has_num_splits_dynamic_ptr: bool
    has_virtual_batch_idx: bool
    has_semaphore_to_reset: bool


def fwd_combine_tile(head_dim: int) -> tuple[int, int]:
    """Return the combine kernel's tile and padded head-dimension block."""
    # Rounding D96 and D192 up gives a smaller M tile and more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    return (16 if k_block_size == 64 else 8), k_block_size


def combine_log_max_splits(num_splits: int, tile_m: int) -> int:
    """Project an exact split count into the combine kernel's codegen bucket."""
    if num_splits < 1:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    # An 8-row M tile requires at least 32 splits; other tiles require 16.
    return max((num_splits - 1).bit_length(), 5 if tile_m == 8 else 4)


def fwd_config_compile_bucket(config: FwdConfig, head_dim_v: int) -> tuple[FwdConfig, int | None]:
    """Deduplicate exact split counts that share main and combine codegen."""
    main = replace(config, num_splits=2 if config.num_splits > 1 else 1)
    if config.num_splits == 1:
        return main, None
    tile_m, _ = fwd_combine_tile(head_dim_v)
    return main, combine_log_max_splits(config.num_splits, tile_m)


def num_splits_heuristic(
    total_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """Resolve automatic SplitKV to a concrete positive split count."""
    if num_n_blocks <= 4 or total_mblocks == 0:
        return 1
    return max(1, min(num_sms // total_mblocks, max_splits, num_n_blocks))


def fixed_seqlen_k_num_splits(inputs: FwdHeuristicInputs) -> int:
    """Return the minimum split count required by a fixed KV extent."""
    seqlen_k_per_split = inputs.seqlen_k_per_split
    if seqlen_k_per_split is None:
        return 1
    if seqlen_k_per_split <= 0:
        raise ValueError("seqlen_k_per_split must be positive")
    return max(
        1,
        (inputs.max_seqlen_k + seqlen_k_per_split - 1) // seqlen_k_per_split,
    )


def select_sm90_tile(
    head_dim: int,
    head_dim_v: int,
    causal: bool,
    local: bool,
    sparse_q_block_size: int | None,
) -> tuple[int, int, bool, bool]:
    """Return the existing SM90 tile, RS, and overlap heuristic."""
    match head_dim:
        case value if value <= 64:
            # C++: 192×192 non-causal, 192×128 causal/local.
            # Python: 192×128 RS+OL is consistently best across seqlens.
            if sparse_q_block_size is not None and sparse_q_block_size % 192 != 0:
                return 128, 128, True, True
            return 192, 128, True, True
        case value if value <= 96:
            # C++: 192×144 noRS+OL for all cases.
            # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
            # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
            if sparse_q_block_size is not None and sparse_q_block_size % 192 != 0:
                return 128, 128, False, True
            return (192, 128, False, True) if causal or local else (192, 144, False, True)
        case value if value <= 128:
            return 128, 128, True, True
        case value if value <= 192:
            match local, head_dim_v <= 128:
                case True, _:
                    tile_n = 96
                case False, True:
                    tile_n = 128
                case False, False:
                    tile_n = 112
            return 128, tile_n, True, True
        case _:
            return 128, 64 if local else 80, True, True


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
        case 12 if inputs.head_dim > 64:
            # SM120 has 99 KB SMEM: 128×128 is preferred for D<=64, and 128×64
            # avoids the occupancy loss from a 96 KB tile for larger head dimensions.
            tile_n = 64

    if inputs.requested_tile_m is not None:
        tile_m = inputs.requested_tile_m
    if inputs.requested_tile_n is not None:
        tile_n = inputs.requested_tile_n
    if inputs.device_capacity == 9 and (
        inputs.requested_tile_m is not None or inputs.requested_tile_n is not None
    ):
        mma_pv_is_rs = True
        intra_wg_overlap = True
    if inputs.requested_mma_pv_is_rs is not None:
        mma_pv_is_rs = inputs.requested_mma_pv_is_rs
    if inputs.requested_intra_wg_overlap is not None:
        intra_wg_overlap = inputs.requested_intra_wg_overlap
    return tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap


def sm90_num_threads(tile_m: int) -> int:
    """Return the thread count implied by an SM90 tile."""
    if tile_m not in (64, 128, 192):
        raise ValueError(f"SM90 tile_m must be 64, 128, or 192, got {tile_m}")
    return 128 * (tile_m // 64 + 1)


def select_sm90_register_allocation(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    tile_n: int,
) -> FwdSm90RegisterAllocation:
    """Resolve the existing SM90 warp-group register allocation."""
    num_mma_warp_groups = tile_m // 64
    use_tma_q = not (inputs.pack_gqa and tile_m % inputs.qhead_per_kvhead != 0)
    use_tma_kv = inputs.page_size in (None, tile_n)
    if num_mma_warp_groups == 2 and (not use_tma_q or not use_tma_kv):
        return FwdSm90RegisterAllocation(mma=224, producer=40)
    return {
        1: FwdSm90RegisterAllocation(mma=256, producer=56),
        2: FwdSm90RegisterAllocation(mma=240, producer=24),
        3: FwdSm90RegisterAllocation(mma=160, producer=32),
    }[num_mma_warp_groups]


def has_s_o_s_q_overlap(inputs: FwdHeuristicInputs, num_splits: int) -> bool:
    """Return whether O and Q share storage for this SM100 problem."""
    return (inputs.head_dim_padded == 192 and inputs.head_dim_v_padded >= 64) or (
        inputs.head_dim_v_padded >= 128 and num_splits > 1
    )


def can_use_tma_o(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    num_splits: int,
) -> bool:
    """Return whether the generic SM100 output layout supports TMA."""
    return (
        not (inputs.pack_gqa and tile_m % inputs.qhead_per_kvhead != 0)
        and not (inputs.pack_gqa and num_splits > 1)
        and not inputs.is_varlen_q
    )


def uses_dedicated_hd256_kernel(inputs: FwdHeuristicInputs) -> bool:
    """Return whether this problem uses the fixed SM100 head-dim-256 kernel."""
    return (
        inputs.device_capacity in (10, 11) and inputs.head_dim == 256 and inputs.head_dim_v == 256
    )


# The 12-warp dedicated kernel does not use the generic 512-thread budget.
_HD256_REGISTER_ALLOCATION = FwdSm100RegisterAllocation(256, 160, 32)
# The generic kernel redistributes a 4 × 128-register CTA budget.
_DEFAULT_REGISTER_ALLOCATION = FwdSm100RegisterAllocation(192, 80, 48)
_PAGED_REGISTER_ALLOCATION = FwdSm100RegisterAllocation(184, 64, 80)
# Key: (use_2cta_instrs, causal, head_dim_padded, is_sm103_family).
_SM100_REGISTER_OVERRIDES = {
    (True, False, 128, False): FwdSm100RegisterAllocation(176, 88, 72),
    (False, True, 128, False): FwdSm100RegisterAllocation(192, 72, 56),
    (True, False, 192, False): FwdSm100RegisterAllocation(184, 80, 64),
    (False, True, 192, False): FwdSm100RegisterAllocation(192, 72, 56),
    (True, False, 128, True): FwdSm100RegisterAllocation(176, 80, 80),
    (False, True, 128, True): FwdSm100RegisterAllocation(176, 64, 96),
    (True, False, 192, True): FwdSm100RegisterAllocation(176, 64, 96),
    (False, True, 192, True): FwdSm100RegisterAllocation(176, 72, 88),
}
_FP8_SM100_REGISTER_OVERRIDES = {
    (True, False, 128, False): FwdSm100RegisterAllocation(160, 72, 120),
}


def select_sm100_register_allocation(
    inputs: FwdHeuristicInputs,
    *,
    tile_n: int,
    use_2cta_instrs: bool,
) -> FwdSm100RegisterAllocation:
    """Resolve the generic SM100 warp-group register allocation."""
    head_dim_padded = inputs.head_dim_padded
    paged_kv_non_tma = inputs.page_size not in (None, tile_n)
    is_fp8 = inputs.dtype in ("torch.float8_e4m3fn", "torch.float8_e5m2")
    if head_dim_padded < 96:
        if is_fp8:
            return (
                FwdSm100RegisterAllocation(152, 96, 112)
                if paged_kv_non_tma
                else FwdSm100RegisterAllocation(168, 96, 80)
            )
        return (
            _PAGED_REGISTER_ALLOCATION
            if paged_kv_non_tma
            else FwdSm100RegisterAllocation(200, 64, 48)
        )
    if paged_kv_non_tma:
        return _PAGED_REGISTER_ALLOCATION

    is_sm103_family = inputs.device_capacity == 10 and inputs.device_arch >= 103
    key = (use_2cta_instrs, inputs.causal, head_dim_padded, is_sm103_family)
    registers = _SM100_REGISTER_OVERRIDES.get(key, _DEFAULT_REGISTER_ALLOCATION)
    return _FP8_SM100_REGISTER_OVERRIDES.get(key, registers) if is_fp8 else registers


def fixed_mla_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Represent the dedicated MLA implementation with one non-tunable config."""
    return FwdConfig(
        device_capacity=inputs.device_capacity,
        tile_m=64,
        tile_n=128,
        num_stages=0,
        num_threads=(512 if inputs.has_gather_kv or inputs.page_size not in (None, 128) else 384),
        mma_pv_is_rs=False,
        intra_wg_overlap=False,
        q_stage=1,
        use_clc_scheduler=True,
        is_static_persistent=False,
        use_tma_o=True,
        num_splits=1,
        use_2cta_instrs=True,
        registers=None,
    )


@lru_cache(maxsize=1024)
def select_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Select one fully resolved config from host-visible metadata."""
    device_capacity = inputs.device_capacity
    if device_capacity == 12 and inputs.requested_num_splits != 1:
        raise AssertionError("SM120 forward only supports num_splits=1")
    if inputs.has_qv:
        if inputs.requested_num_splits not in (None, 1):
            raise ValueError("The dedicated MLA kernel does not support SplitKV")
        config = fixed_mla_fwd_config(inputs)
        validate_fwd_config(config, inputs)
        return config
    if uses_dedicated_hd256_kernel(inputs):
        if inputs.requested_num_splits not in (None, 1):
            raise ValueError("The SM100 head-dim-256 kernel does not support SplitKV")
        config = fixed_hd256_fwd_config(inputs)
        requested_overrides = (
            (inputs.requested_tile_m, config.tile_m),
            (inputs.requested_tile_n, config.tile_n),
            (inputs.requested_mma_pv_is_rs, config.mma_pv_is_rs),
            (inputs.requested_intra_wg_overlap, config.intra_wg_overlap),
        )
        if any(requested not in (None, resolved) for requested, resolved in requested_overrides):
            raise ValueError("The SM100 head-dim-256 kernel has one fixed configuration")
        validate_fwd_config(config, inputs)
        return config

    tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap = select_initial_fwd_tile(inputs)
    is_sm100_family = device_capacity in (10, 11)
    q_stage = 2 if is_sm100_family and inputs.max_seqlen_q * inputs.qhead_per_kvhead > tile_m else 1
    seqlen_k_loaded = inputs.max_seqlen_k
    if inputs.local:
        window_left = inputs.window_size_left or inputs.max_seqlen_k
        window_right = inputs.window_size_right or inputs.max_seqlen_k
        seqlen_k_loaded = max(
            0,
            min(inputs.max_seqlen_k, window_left + window_right + 1 + tile_m),
        )

    effective_tile_m = q_stage * tile_m
    packed_seqlen_q = inputs.max_seqlen_q * inputs.qhead_per_kvhead
    num_m_blocks = (packed_seqlen_q + effective_tile_m - 1) // effective_tile_m
    total_mblocks = inputs.batch_size * inputs.num_heads_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    fixed_num_splits = fixed_seqlen_k_num_splits(inputs)
    if inputs.requested_num_splits is not None:
        num_splits = max(1, inputs.requested_num_splits)
    elif inputs.seqlen_k_per_split is not None:
        num_splits = fixed_num_splits
    else:
        num_splits = num_splits_heuristic(total_mblocks, inputs.num_sms, num_n_blocks, 128)

    # SplitKV's float32 partial O doubles its shared-memory footprint for diff-head.
    if is_sm100_family and inputs.head_dim != inputs.head_dim_v and num_splits > 1:
        match inputs.requested_num_splits:
            case None if (num_n_blocks >= 64 or fixed_num_splits > 1) and inputs.head_dim_v != 512:
                tile_n = 64
                if inputs.seqlen_k_per_split is None:
                    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
                    num_splits = num_splits_heuristic(
                        total_mblocks, inputs.num_sms, num_n_blocks, 128
                    )
            case None:
                num_splits = 1
            case _ if inputs.head_dim_v != 512:
                tile_n = 64

    is_split_kv = num_splits > 1
    use_2cta_instrs = not inputs.disable_2cta and is_2cta_eligible(
        inputs, tile_m=tile_m, num_splits=num_splits
    )

    overlap_s_o_s_q = is_sm100_family and has_s_o_s_q_overlap(inputs, num_splits)
    # CLC regresses varlen MHA and dense noncausal: the former increases K/V
    # traffic under imbalance, while the latter mostly pays work-stealing overhead.
    is_varlen = (
        inputs.is_varlen_q
        or inputs.has_cu_seqlens_q
        or inputs.has_cu_seqlens_k
        or inputs.has_seqused
    )
    is_varlen_mha = is_varlen and inputs.qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not inputs.causal and not inputs.local
    use_clc_scheduler = (
        is_sm100_family
        and inputs.requested_use_clc_scheduler
        and not is_varlen_mha
        and not is_dense_noncausal
        and inputs.page_size in (None, tile_n)
        and not overlap_s_o_s_q
    )
    is_static_persistent = (
        is_sm100_family
        and not is_split_kv
        and not overlap_s_o_s_q
        and not use_clc_scheduler
        and (
            (not inputs.causal and not inputs.local and not inputs.is_varlen_q)
            or packed_seqlen_q <= effective_tile_m
        )
    )
    use_tma_o = is_sm100_family and can_use_tma_o(inputs, tile_m=tile_m, num_splits=num_splits)

    match device_capacity:
        case 9:
            num_stages = 2
            num_threads = sm90_num_threads(tile_m)
            registers = select_sm90_register_allocation(
                inputs,
                tile_m=tile_m,
                tile_n=tile_n,
            )
        case 8 | 12:
            num_stages = 1
            num_threads = 128
            registers = None
        case 10 | 11:
            num_stages = 0
            num_threads = 512
            registers = select_sm100_register_allocation(
                inputs,
                tile_n=tile_n,
                use_2cta_instrs=use_2cta_instrs,
            )
        case _:
            num_stages = 0
            num_threads = 512
            registers = None
    config = FwdConfig(
        device_capacity=device_capacity,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=num_stages,
        num_threads=num_threads,
        mma_pv_is_rs=mma_pv_is_rs,
        intra_wg_overlap=intra_wg_overlap,
        q_stage=q_stage,
        use_clc_scheduler=use_clc_scheduler,
        is_static_persistent=is_static_persistent,
        use_tma_o=use_tma_o,
        num_splits=num_splits,
        use_2cta_instrs=use_2cta_instrs,
        registers=registers,
    )
    validate_fwd_config(config, inputs)
    return config


@lru_cache(maxsize=1024)
def validate_fwd_config(config: FwdConfig, inputs: FwdHeuristicInputs) -> None:
    """Reject known unsupported or silently normalized config combinations."""
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
    if config.num_threads <= 0 or config.num_threads % 32 != 0:
        raise ValueError(f"num_threads must be a positive multiple of 32, got {config.num_threads}")
    if config.num_stages < 0:
        raise ValueError(f"num_stages must be nonnegative, got {config.num_stages}")
    if config.num_splits < 1 or config.num_splits > 256:
        raise ValueError(f"num_splits must be in [1, 256], got {config.num_splits}")
    fixed_num_splits = fixed_seqlen_k_num_splits(inputs)
    if config.num_splits < fixed_num_splits:
        raise ValueError(
            f"seqlen_k_per_split={inputs.seqlen_k_per_split} requires "
            f"num_splits >= {fixed_num_splits}, got {config.num_splits}"
        )
    if inputs.seqlen_k_per_split is not None and inputs.seqlen_k_per_split % config.tile_n != 0:
        raise ValueError(f"seqlen_k_per_split must be divisible by tile_n={config.tile_n}")

    is_sm90 = inputs.device_capacity == 9
    is_sm100_family = inputs.device_capacity in (10, 11)
    if is_sm90:
        if not isinstance(config.registers, FwdSm90RegisterAllocation):
            raise TypeError("SM90 forward requires an SM90 register allocation")
        registers = config.registers
        if any(value < 24 or value > 256 or value % 8 != 0 for value in registers):
            raise ValueError(
                f"SM90 register allocations must be multiples of 8 in [24, 256], got {registers}"
            )
        if registers.mma < 128:
            raise ValueError("SM90 MMA register allocation must be at least 128 for setmaxnreg.inc")
        if registers.producer > 128:
            raise ValueError(
                "SM90 producer register allocation must be at most 128 for setmaxnreg.dec"
            )
        num_mma_warp_groups = config.tile_m // 64
        if num_mma_warp_groups * registers.mma + registers.producer > 512:
            raise ValueError(
                f"SM90 register allocation exceeds the 512-register budget: {registers}"
            )
    elif not is_sm100_family and config.registers is not None:
        raise ValueError(f"SM{inputs.device_capacity} does not expose register allocation")

    if not is_sm100_family:
        match inputs.device_capacity:
            case 8 if inputs.page_size is not None:
                raise ValueError("SM80 forward does not support paged KV")
            case 12 if inputs.page_size is not None or inputs.use_block_sparsity:
                raise ValueError("SM120 forward does not support paged KV or block sparsity")
        if config.num_stages == 0:
            raise ValueError(f"SM{inputs.device_capacity} requires at least one pipeline stage")
        if is_sm90:
            expected_threads = sm90_num_threads(config.tile_m)
            if config.num_threads != expected_threads:
                raise ValueError(
                    f"SM90 tile_m={config.tile_m} requires num_threads={expected_threads}"
                )
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
        ):
            raise ValueError(f"SM{inputs.device_capacity} does not support SM100 scheduler options")
        return

    is_dedicated_hd256 = uses_dedicated_hd256_kernel(inputs)
    if is_dedicated_hd256:
        if config != fixed_hd256_fwd_config(inputs):
            raise ValueError("The SM100 head-dim-256 kernel has one fixed configuration")
        return
    if inputs.has_qv:
        if config != fixed_mla_fwd_config(inputs):
            raise ValueError("The dedicated MLA kernel has one fixed configuration")
        return
    if not isinstance(config.registers, FwdSm100RegisterAllocation):
        raise TypeError("Generic SM100 forward requires an SM100 register allocation")
    registers = config.registers
    if any(value < 24 or value > 256 or value % 8 != 0 for value in registers):
        raise ValueError(
            f"SM100 register allocations must be multiples of 8 in [24, 256], got {registers}"
        )
    if registers.softmax < 128:
        raise ValueError(
            "SM100 softmax register allocation must be at least 128 for setmaxnreg.inc"
        )
    if registers.correction > 128 or registers.other > 128:
        raise ValueError(
            "SM100 correction and other register allocations must be at most 128 for setmaxnreg.dec"
        )
    if 2 * registers.softmax + registers.correction + registers.other > 512:
        raise ValueError(f"SM100 register allocation exceeds the 512-register budget: {registers}")
    if config.num_stages != 0:
        raise ValueError("SM100 forward derives internal pipeline stages and requires num_stages=0")
    if config.num_threads != 512:
        raise ValueError("SM100 forward uses a fixed 512-thread CTA")
    if config.mma_pv_is_rs or config.intra_wg_overlap:
        raise ValueError("Generic SM100 forward does not expose SM90 MMA flags")
    if config.q_stage not in (1, 2):
        raise ValueError(f"SM100 q_stage must be 1 or 2, got {config.q_stage}")
    if config.num_splits > 1:
        if config.use_2cta_instrs:
            raise ValueError("SplitKV and 2CTA cannot be enabled together")
        if inputs.head_dim_v_padded >= 192:
            raise ValueError("SplitKV does not support padded value head dimensions >= 192")
        if inputs.head_dim != inputs.head_dim_v and config.tile_n != 64:
            raise ValueError("Diff-head SplitKV requires tile_n=64")

    overlap_s_o_s_q = has_s_o_s_q_overlap(inputs, config.num_splits)
    paged_kv_non_tma = inputs.page_size not in (None, config.tile_n)
    if paged_kv_non_tma and (inputs.head_dim % 16 != 0 or inputs.head_dim_v % 16 != 0):
        raise ValueError("Non-TMA paged KV requires head dimensions divisible by 16")
    if config.use_clc_scheduler and (paged_kv_non_tma or overlap_s_o_s_q):
        raise ValueError("CLC requires TMA KV and a non-overlapping O/Q shared-memory layout")
    persistent_shape = (not inputs.causal and not inputs.local and not inputs.is_varlen_q) or (
        inputs.max_seqlen_q * inputs.qhead_per_kvhead <= config.q_stage * config.tile_m
    )
    if config.is_static_persistent and (
        not persistent_shape or config.num_splits > 1 or overlap_s_o_s_q or config.use_clc_scheduler
    ):
        raise ValueError("Static persistent scheduling is not effective for this forward problem")

    if config.use_tma_o and not can_use_tma_o(
        inputs, tile_m=config.tile_m, num_splits=config.num_splits
    ):
        raise ValueError("TMA O is not supported for this GQA, SplitKV, or varlen layout")

    if config.use_2cta_instrs and not is_2cta_eligible(
        inputs, tile_m=config.tile_m, num_splits=config.num_splits
    ):
        raise ValueError("2CTA instructions are not supported for this forward problem")


def fixed_hd256_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Return the canonical fixed head-dim-256 config."""
    if inputs.head_dim == 256 and inputs.head_dim_v == 256:
        return FwdConfig(
            device_capacity=inputs.device_capacity,
            tile_m=128,
            tile_n=128,
            num_stages=0,
            num_threads=384,
            mma_pv_is_rs=False,
            intra_wg_overlap=False,
            q_stage=2,
            use_clc_scheduler=False,
            is_static_persistent=False,
            use_tma_o=False,
            num_splits=1,
            use_2cta_instrs=True,
            registers=_HD256_REGISTER_ALLOCATION,
        )
    raise ValueError("No fixed config is defined for this problem")


def is_2cta_eligible(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    num_splits: int,
) -> bool:
    """Return whether a generic SM100 config can use 2CTA instructions."""
    return (
        inputs.device_capacity in (10, 11)
        and not inputs.causal
        and not inputs.local
        and num_splits == 1
        and not inputs.is_varlen_q
        and not inputs.use_block_sparsity
        and inputs.page_size in (None, 128)
        and inputs.head_dim_padded in (128, 192)
        and inputs.head_dim_v_padded == 128
        and inputs.max_seqlen_q * inputs.qhead_per_kvhead > 2 * tile_m
        and (tile_m % inputs.qhead_per_kvhead == 0 or not inputs.pack_gqa)
    )
