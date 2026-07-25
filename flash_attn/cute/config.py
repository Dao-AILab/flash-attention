# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Typed boundaries for forward configuration and compilation caches.

`FwdHeuristicInputs` contains immutable, host-visible problem metadata and
directly keys the bounded default selector. Offline profiling wraps that same
default in a named `FwdConfigBucket` with valid alternatives. `FwdConfig` is
fully resolved: one split means the nonsplit path and larger values are exact
SplitKV counts. Explicit configs are validated without ranking or normalization.

`FwdMainKernelSpec` and `FwdCombineKernelSpec` are independent compile-cache
projections containing only values that affect their generated kernels. The
main projection distinguishes nonsplit from SplitKV; the combine projection
independently buckets concrete split counts. Selection must not read tensor
contents, synchronize CUDA, or allocate outputs.
"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class FwdConfig:
    """Fully resolved forward launch and algorithm configuration."""

    device_capacity: int
    tile_m: int
    tile_n: int
    num_stages: int
    num_threads: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    q_stage: int
    use_clc_scheduler: bool
    is_persistent: bool
    use_tma_o: bool
    num_splits: int
    use_2cta_instrs: bool


@dataclass(frozen=True)
class FwdHeuristicInputs:
    """Host-visible metadata used to select and validate a forward config."""

    device_capacity: int
    device_arch: int
    num_sms: int
    dtype: str
    head_dim: int
    head_dim_v: int
    num_heads: int
    num_heads_kv: int
    batch_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    causal: bool
    local: bool
    window_size_left: int | None
    window_size_right: int | None
    is_varlen: bool
    is_varlen_q: bool
    pack_gqa: bool
    page_size: int | None
    use_block_sparsity: bool
    sparse_q_block_size: int | None
    has_qv: bool
    has_gather_kv: bool
    requested_tile_m: int | None
    requested_tile_n: int | None
    requested_num_threads: int
    requested_mma_pv_is_rs: bool | None
    requested_intra_wg_overlap: bool | None
    requested_num_splits: int | None
    requested_use_clc_scheduler: bool
    disable_2cta: bool

    @property
    def qhead_per_kvhead(self) -> int:
        return self.num_heads // self.num_heads_kv


@dataclass(frozen=True)
class FwdConfigBucket:
    """Named production default and its curated tuning neighborhood."""

    name: str
    default: FwdConfig
    candidates: tuple[FwdConfig, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Forward config bucket name must not be empty")
        if self.default not in self.candidates:
            raise ValueError(f"Bucket {self.name!r} does not contain its default config")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError(f"Bucket {self.name!r} contains duplicate configs")


@dataclass(frozen=True)
class FwdMainKernelSpec:
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
    has_learnable_sink: bool
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
    is_persistent: bool | None
    use_tma_o: bool | None
    has_q: bool
    has_qv: bool
    has_p: bool
    has_row_max: bool
    gather_kv_length: int | None
    sparse_kv: bool | None
    disable_sparse_kv_bitmask: bool | None
    log_level: int


@dataclass(frozen=True)
class FwdCombineKernelSpec:
    """Typed cache key for one generated SplitKV combine kernel."""

    dtype: object
    dtype_partial: object
    head_dim: int
    tile_m: int
    k_block_size: int
    log_max_splits: int
    has_cu_seqlens: bool
    has_seqused: bool
    has_lse: bool
    has_varlen_batch_idx: bool


def fwd_combine_tile(head_dim: int) -> tuple[int, int]:
    """Return the combine kernel's tile and padded head-dimension block."""
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    tile_m = 8 if k_block_size % 128 == 0 else 16
    return tile_m, k_block_size


def combine_log_max_splits(num_splits: int, tile_m: int) -> int:
    """Project an exact split count into the combine kernel's codegen bucket."""
    if num_splits < 1:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal with this by using 128 threads instead.
        log_max_splits = max(log_max_splits, 5)
    return log_max_splits


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
        if inputs.device_capacity == 9:
            mma_pv_is_rs = True
            intra_wg_overlap = True
    if inputs.requested_tile_n is not None:
        tile_n = inputs.requested_tile_n
        if inputs.device_capacity == 9:
            mma_pv_is_rs = True
            intra_wg_overlap = True
    if inputs.requested_mma_pv_is_rs is not None:
        mma_pv_is_rs = inputs.requested_mma_pv_is_rs
    if inputs.requested_intra_wg_overlap is not None:
        intra_wg_overlap = inputs.requested_intra_wg_overlap
    return tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap


def has_s_o_s_q_overlap(inputs: FwdHeuristicInputs, num_splits: int) -> bool:
    """Return whether O and Q share storage for this SM100 problem."""
    return (
        math.ceil(inputs.head_dim / 16) * 16 == 192 and math.ceil(inputs.head_dim_v / 16) * 16 >= 64
    ) or (math.ceil(inputs.head_dim_v / 16) * 16 >= 128 and num_splits > 1)


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
        is_persistent=False,
        use_tma_o=True,
        num_splits=1,
        use_2cta_instrs=True,
    )


def default_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Reproduce the current analytical forward choices as one resolved config."""
    device_capacity = inputs.device_capacity
    if device_capacity not in (8, 9, 10, 11, 12):
        raise ValueError(f"Unsupported forward architecture family SM{device_capacity}")
    if device_capacity == 12 and inputs.requested_num_splits != 1:
        raise AssertionError("SM120 forward only supports num_splits=1")
    if inputs.has_qv:
        if inputs.requested_num_splits not in (None, 1):
            raise ValueError("The dedicated MLA kernel does not support SplitKV")
        config = fixed_mla_fwd_config(inputs)
        validate_fwd_config(config, inputs)
        return config

    tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap = select_initial_fwd_tile(inputs)
    is_sm100_family = device_capacity in (10, 11)
    is_dedicated_hd256 = uses_dedicated_hd256_kernel(inputs)
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
    match inputs.requested_num_splits:
        case None:
            num_splits = num_splits_heuristic(total_mblocks, inputs.num_sms, num_n_blocks, 128)
        case requested_num_splits:
            num_splits = max(1, requested_num_splits)

    # SplitKV's float32 partial O doubles its shared-memory footprint for diff-head.
    if is_sm100_family and inputs.head_dim != inputs.head_dim_v and num_splits > 1:
        match inputs.requested_num_splits:
            case None if num_n_blocks >= 64 and inputs.head_dim_v != 512:
                tile_n = 64
                num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
                num_splits = num_splits_heuristic(total_mblocks, inputs.num_sms, num_n_blocks, 128)
            case None:
                num_splits = 1
            case _ if inputs.head_dim_v != 512:
                tile_n = 64

    is_split_kv = num_splits > 1
    use_2cta_instrs = (
        not inputs.disable_2cta and is_2cta_eligible(inputs, tile_m=tile_m, num_splits=num_splits)
    ) or is_dedicated_hd256

    overlap_s_o_s_q = is_sm100_family and has_s_o_s_q_overlap(inputs, num_splits)
    # CLC regresses varlen MHA and dense noncausal: the former increases K/V
    # traffic under imbalance, while the latter mostly pays work-stealing overhead.
    is_varlen_mha = inputs.is_varlen and inputs.qhead_per_kvhead == 1
    is_dense_noncausal = not inputs.is_varlen and not inputs.causal and not inputs.local
    use_clc_scheduler = (
        is_sm100_family
        and inputs.requested_use_clc_scheduler
        and not is_varlen_mha
        and not is_dense_noncausal
        and inputs.page_size in (None, tile_n)
        and not overlap_s_o_s_q
        and not is_dedicated_hd256
    )
    is_persistent = (
        is_sm100_family
        and not inputs.causal
        and not inputs.local
        and not inputs.is_varlen_q
        and not is_split_kv
        and not overlap_s_o_s_q
        and not use_clc_scheduler
        and not is_dedicated_hd256
    )
    use_tma_o = (
        is_sm100_family
        and not is_dedicated_hd256
        and can_use_tma_o(inputs, tile_m=tile_m, num_splits=num_splits)
    )

    match device_capacity:
        case 9:
            num_stages = 2
            num_threads = inputs.requested_num_threads
        case 8 | 12:
            num_stages = 1
            num_threads = 128
        case _ if is_dedicated_hd256:
            num_stages = 0
            num_threads = 384
        case _:
            num_stages = 0
            num_threads = 512

    config = FwdConfig(
        device_capacity=device_capacity,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=num_stages,
        num_threads=num_threads,
        mma_pv_is_rs=mma_pv_is_rs,
        intra_wg_overlap=intra_wg_overlap,
        q_stage=2 if is_dedicated_hd256 else q_stage,
        use_clc_scheduler=use_clc_scheduler,
        is_persistent=is_persistent,
        use_tma_o=use_tma_o,
        num_splits=num_splits,
        use_2cta_instrs=use_2cta_instrs,
    )
    validate_fwd_config(config, inputs)
    return config


def validate_fwd_config(config: FwdConfig, inputs: FwdHeuristicInputs) -> None:
    """Reject configs that would be unsupported or silently normalized by a kernel."""
    if inputs.device_capacity not in (8, 9, 10, 11, 12):
        raise ValueError(f"Unsupported forward architecture family SM{inputs.device_capacity}")
    if inputs.device_arch // 10 != inputs.device_capacity:
        raise ValueError(
            f"Architecture SM{inputs.device_arch} does not match capacity {inputs.device_capacity}"
        )
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

    is_sm100_family = inputs.device_capacity in (10, 11)
    if not is_sm100_family:
        match inputs.device_capacity:
            case 8 if inputs.page_size is not None:
                raise ValueError("SM80 forward does not support paged KV")
            case 12 if inputs.page_size is not None or inputs.use_block_sparsity:
                raise ValueError("SM120 forward does not support paged KV or block sparsity")
        if config.num_stages == 0:
            raise ValueError(f"SM{inputs.device_capacity} requires at least one pipeline stage")
        if inputs.device_capacity in (8, 12):
            smem_usage = (
                config.tile_m * inputs.head_dim * 2
                + config.tile_n * (inputs.head_dim + inputs.head_dim_v) * config.num_stages * 2
            )
            smem_capacity = 166912 if inputs.device_capacity == 8 else 101376
            if (
                inputs.dtype not in ("torch.float16", "torch.bfloat16")
                or inputs.head_dim % 8 != 0
                or inputs.head_dim_v % 8 != 0
                or (config.tile_m * 2) % config.num_threads != 0
                or smem_usage > smem_capacity
            ):
                raise ValueError(
                    f"SM{inputs.device_capacity} cannot implement this tile/thread/stage configuration"
                )
        if inputs.device_capacity != 9 and (config.mma_pv_is_rs or config.intra_wg_overlap):
            raise ValueError(f"SM{inputs.device_capacity} does not expose SM90 MMA flags")
        if config.q_stage != 1:
            raise ValueError(f"SM{inputs.device_capacity} requires q_stage=1")
        if config.num_splits != 1:
            raise ValueError(f"SM{inputs.device_capacity} does not support SplitKV")
        if (
            config.use_2cta_instrs
            or config.use_clc_scheduler
            or config.is_persistent
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
        if math.ceil(inputs.head_dim_v / 16) * 16 >= 192:
            raise ValueError("SplitKV does not support padded value head dimensions >= 192")
        if inputs.head_dim != inputs.head_dim_v and config.tile_n != 64:
            raise ValueError("Diff-head SplitKV requires tile_n=64")

    overlap_s_o_s_q = has_s_o_s_q_overlap(inputs, config.num_splits)
    paged_kv_non_tma = inputs.page_size not in (None, config.tile_n)
    if paged_kv_non_tma and (inputs.head_dim % 16 != 0 or inputs.head_dim_v % 16 != 0):
        raise ValueError("Non-TMA paged KV requires head dimensions divisible by 16")
    if config.use_clc_scheduler and (paged_kv_non_tma or overlap_s_o_s_q):
        raise ValueError("CLC requires TMA KV and a non-overlapping O/Q shared-memory layout")
    if config.is_persistent and (
        inputs.causal
        or inputs.local
        or inputs.is_varlen_q
        or config.num_splits > 1
        or overlap_s_o_s_q
        or config.use_clc_scheduler
    ):
        raise ValueError("Persistent scheduling is not effective for this forward problem")

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
            is_persistent=False,
            use_tma_o=False,
            num_splits=1,
            use_2cta_instrs=True,
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
        and math.ceil(inputs.head_dim / 16) * 16 in (128, 192)
        and math.ceil(inputs.head_dim_v / 16) * 16 == 128
        and inputs.max_seqlen_q * inputs.qhead_per_kvhead > 2 * tile_m
        and (tile_m % inputs.qhead_per_kvhead == 0 or not inputs.pack_gqa)
    )


def candidate_fwd_configs(
    inputs: FwdHeuristicInputs,
    default: FwdConfig,
) -> tuple[FwdConfig, ...]:
    """Build a curated local neighborhood around the production default."""
    if (
        inputs.device_capacity not in (10, 11)
        or inputs.has_qv
        or uses_dedicated_hd256_kernel(inputs)
    ):
        return (default,)

    candidates = [default]
    alternatives = (
        {"q_stage": 1 if default.q_stage == 2 else 2},
        {
            "use_clc_scheduler": not default.use_clc_scheduler,
            "is_persistent": False,
        },
        {"use_clc_scheduler": False, "is_persistent": True},
        {"is_persistent": not default.is_persistent},
        {"use_tma_o": not default.use_tma_o},
        {"use_2cta_instrs": not default.use_2cta_instrs},
    )
    for changes in alternatives:
        candidate = replace(default, **changes)
        try:
            validate_fwd_config(candidate, inputs)
        except ValueError:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def fwd_bucket_name(inputs: FwdHeuristicInputs, default: FwdConfig) -> str:
    """Name a coarse production region without embedding exact sequence lengths."""
    mode = "varlen" if inputs.is_varlen else "dense"
    match inputs:
        case FwdHeuristicInputs(causal=True):
            mask = "causal"
        case FwdHeuristicInputs(local=True):
            mask = "local"
        case _:
            mask = "noncausal"
    heads = "mha" if inputs.qhead_per_kvhead == 1 else f"gqa{inputs.qhead_per_kvhead}"
    dims = (
        f"d{inputs.head_dim}"
        if inputs.head_dim == inputs.head_dim_v
        else f"d{inputs.head_dim}v{inputs.head_dim_v}"
    )
    split = f"split{default.num_splits}" if default.num_splits > 1 else "nosplit"
    cta = "2cta" if default.use_2cta_instrs else "1cta"
    match default:
        case FwdConfig(use_clc_scheduler=True):
            scheduler = "clc"
        case FwdConfig(is_persistent=True):
            scheduler = "persistent"
        case _:
            scheduler = "single"
    epilogue = "tma-o" if default.use_tma_o else "direct-o"
    match inputs:
        case FwdHeuristicInputs(has_qv=True, has_gather_kv=True):
            feature = "mla-topk"
        case FwdHeuristicInputs(has_qv=True, page_size=page_size) if page_size is not None:
            feature = f"mla-paged{page_size}"
        case FwdHeuristicInputs(has_qv=True):
            feature = "mla"
        case FwdHeuristicInputs(use_block_sparsity=True):
            feature = "sparse"
        case FwdHeuristicInputs(page_size=page_size) if page_size is not None:
            feature = f"paged{page_size}"
        case _:
            feature = "standard"
    return ".".join(
        (
            f"sm{inputs.device_arch}",
            inputs.dtype.removeprefix("torch."),
            mode,
            mask,
            heads,
            dims,
            feature,
            f"tile{default.tile_m}x{default.tile_n}",
            f"q{default.q_stage}",
            split,
            cta,
            scheduler,
            epilogue,
        )
    )


@lru_cache(maxsize=1024)
def get_fwd_config_bucket(inputs: FwdHeuristicInputs) -> FwdConfigBucket:
    """Return the cached named default and candidates for one immutable input."""
    default = default_fwd_config(inputs)
    return FwdConfigBucket(
        name=fwd_bucket_name(inputs, default),
        default=default,
        candidates=candidate_fwd_configs(inputs, default),
    )


@lru_cache(maxsize=1024)
def select_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
    """Return the cached baked-in default without constructing tuning candidates."""
    return default_fwd_config(inputs)
