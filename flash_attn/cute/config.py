# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Typed forward configuration, selection, and validation.

`FwdHeuristicInputs` contains immutable host-visible metadata. `FwdConfig` is
fully resolved: one split means the nonsplit path and larger values are exact
SplitKV counts. Explicit configs are validated without ranking or normalization.
Selection must not read tensor contents, synchronize CUDA, or allocate.

MLA (``qv``) forward is outside this contract: it plans its own kernel, tile,
and split count from the routing heuristic in ``interface.py``.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple


@dataclass(frozen=True)
class FwdConfig:
    """Fully resolved forward launch and algorithm configuration.

    ``is_static_persistent`` controls static persistence; dynamic persistence is
    derived from scheduler metadata.
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


class FwdHeuristicInputs(NamedTuple):
    """Host-visible metadata used to select and validate a forward config.

    ``max_seqlen_q`` is a host bound; ``has_max_seqlen_q_hint`` records whether
    the caller provided it for varlen Q instead of the total-token fallback.
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
        return math.ceil(self.head_dim / 16) * 16

    @property
    def head_dim_v_padded(self) -> int:
        return math.ceil(self.head_dim_v / 16) * 16

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
    return max(
        1,
        (inputs.max_seqlen_k + seqlen_k_per_split - 1) // seqlen_k_per_split,
    )


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
    sparse_block_size_q: int | None = None,
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
    # hd256 prefers 2CTA over cp.async-Q PackGQA when 2CTA can engage.
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
    """Return whether the SM100 output layout supports TMA."""
    return (
        # hd256 SplitKV uses a chunked half-width epilogue without TMA.
        not (num_splits > 1 and inputs.head_dim_v == 256)
        and not (inputs.pack_gqa and tile_m % inputs.qhead_per_kvhead != 0)
        and not (inputs.pack_gqa and num_splits > 1)
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
    """Return whether the SM100 kernel supports S/P ping-pong for this problem."""
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
    """Return whether an SM100 config can use 2CTA instructions."""
    # Single-M-block hd256 2CTA halves each CTA's K/V loads; worth it while the
    # doubled grid fits in one wave.
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
    """Return whether static persistence maps correctly onto this problem."""
    if num_splits > 1 or use_clc_scheduler:
        return False
    dense_noncausal = not inputs.causal and not inputs.local and not inputs.is_varlen_q
    single_m_block = (
        inputs.packed_max_seqlen_q <= q_stage * tile_m
        and (not inputs.has_cu_seqlens_q or inputs.has_max_seqlen_q_hint)
        # Dense causal/local uses the LPT scheduler, which maps 2CTA clusters only
        # when not persistent.
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
    """Return whether the SM100 kernel honors CLC scheduling for this problem."""
    return (
        inputs.page_size in (None, tile_n)
        # CLC does not map hd256 2CTA tiles correctly.
        and not (inputs.is_hdim256 and use_2cta_instrs)
    )


class TunedSm100Overrides(NamedTuple):
    """Measured scheduling wins applied on top of the default policy."""

    clc: bool = False
    nonpersistent: bool = False


def select_tuned_sm100_overrides(
    inputs: FwdHeuristicInputs,
    *,
    tile_m: int,
    tile_n: int,
    num_n_blocks: int,
    is_split_kv: bool,
) -> TunedSm100Overrides:
    """Return measured BF16 output-only scheduling wins for one problem.

    Each rule was promoted from explicit-config A/B campaigns on the named GPU and
    covers only plain output-only BF16 attention on 128x128 nonsplit tiles.
    """
    plain = (
        inputs.dtype == "torch.bfloat16"
        and inputs.page_size is None
        and not inputs.use_block_sparsity
        and not inputs.has_score_mod
        and not inputs.has_mask_mod
        and not inputs.has_learnable_sink
        and not inputs.has_lse
        and tile_m == 128
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
                dense_noncausal
                and head_dim == 64
                and packs_all_q_heads
                and num_n_blocks >= 32
            )
            # CLC pays off for balanced medium-head MHA and broadly for packed high-head
            # varlen once Q has enough work, and for dense causal short-K batches.
            packed_varlen = (
                inputs.has_cu_seqlens_q
                and inputs.has_cu_seqlens_k
                and not inputs.has_seqused_q
                and not inputs.has_seqused_k
                # Caller-built metadata with a tile semaphore selects dynamic scheduling.
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
    device_capacity = inputs.device_capacity
    if device_capacity == 12 and inputs.requested_num_splits != 1:
        raise AssertionError("SM120 forward only supports num_splits=1")

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

    effective_tile_m = q_stage * tile_m
    num_m_blocks = (packed_seqlen_q + effective_tile_m - 1) // effective_tile_m
    # Without PackGQA every Q head gets its own M blocks.
    heads_per_m_block = 1 if inputs.pack_gqa else inputs.qhead_per_kvhead
    total_mblocks = (
        inputs.batch_size * inputs.num_heads_kv * heads_per_m_block * num_m_blocks
    )
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    if inputs.requested_num_splits is not None:
        num_splits = max(1, inputs.requested_num_splits)
    else:
        num_splits = num_splits_heuristic(
            total_mblocks, inputs.num_sms, num_n_blocks, 128
        )

    # SplitKV's float32 partial O doubles its shared-memory footprint for diff-head:
    # long KV re-splits with tile_n=64, and short KV or DV512 falls back to no split.
    # Like the legacy num_splits argument, this also re-resolves a requested count;
    # an explicit FwdConfig is the way to force an exact diff-head split.
    if is_sm100_family and inputs.head_dim != inputs.head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and inputs.head_dim_v != 512:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(
                total_mblocks, inputs.num_sms, num_n_blocks, 128
            )
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    use_2cta_instrs = (
        is_sm100_family
        and not inputs.disable_2cta
        and is_2cta_eligible(
            inputs, tile_m=tile_m, q_stage=q_stage, num_splits=num_splits
        )
    )

    # CLC regresses varlen MHA and dense noncausal: the former increases K/V
    # traffic under imbalance, while the latter mostly pays work-stealing overhead.
    is_varlen_mha = inputs.is_varlen and inputs.qhead_per_kvhead == 1
    is_dense_noncausal = not inputs.is_varlen and not inputs.causal and not inputs.local
    tuned = select_tuned_sm100_overrides(
        inputs,
        tile_m=tile_m,
        tile_n=tile_n,
        num_n_blocks=num_n_blocks,
        is_split_kv=is_split_kv,
    )
    use_clc_scheduler = (
        is_sm100_family
        and can_use_clc(inputs, tile_n=tile_n, use_2cta_instrs=use_2cta_instrs)
        and (
            tuned.clc
            or (
                inputs.requested_use_clc_scheduler
                and not is_varlen_mha
                and not is_dense_noncausal
            )
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
    use_tma_o = is_sm100_family and can_use_tma_o(
        inputs, tile_m=tile_m, num_splits=num_splits
    )
    n_blocks_per_split = (num_n_blocks + num_splits - 1) // num_splits
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
    )
    validate_fwd_config(config, inputs)
    return config


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
    # A fixed KV extent applies only under SplitKV, where each split covers one extent.
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
        return

    if config.mma_pv_is_rs or config.intra_wg_overlap:
        raise ValueError("SM100 forward does not expose SM90 MMA flags")
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
        inputs, tile_n=config.tile_n, use_2cta_instrs=config.use_2cta_instrs
    ):
        raise ValueError(
            "CLC requires TMA KV and is not supported for head-dim-256 2CTA tiles"
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
