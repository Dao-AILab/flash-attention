"""CPU tests of the SwapAB decode branch of the forward config selector (config.py) and of
the launch grid an admitted config gets."""

import itertools
from dataclasses import replace

import cutlass.cute as cute
import pytest

from flash_attn.cute.config import (
    SWAPAB,
    FwdHeuristicInputs,
    derive_fwd_config,
    select_fwd_config,
    needs_split_metadata,
    validate_fwd_config,
)
from flash_attn.cute.tile_scheduler import SingleTileScheduler, TileSchedulerArguments

# Dense bf16 decode: 4 sequences of one query token, 8 query heads on 2 KV heads, 4096-token KV.
_DECODE = FwdHeuristicInputs(
    device_arch=100, num_sms=148, dtype="torch.bfloat16", head_dim=128, head_dim_v=128,
    num_heads=8, num_heads_kv=2, batch_size=4, total_q=4, total_k=4 * 4096,
    max_seqlen_q=1, max_seqlen_k=4096,
)
# Varlen Q decode (cu_seqlens_q, host max_seqlen_q=1) over paged KV with used lengths.
_VARLEN_PAGED = dict(has_cu_seqlens_q=True, total_q=3, has_seqused_k=True, page_size=128)


def decode(**changes) -> FwdHeuristicInputs:
    return _DECODE._replace(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        _VARLEN_PAGED,
        # Pages hold whole 128-token subtiles; a 256-token KV block may span two pages.
        *({"has_seqused_k": True, "page_size": page} for page in (128, 256, 384, 512)),
        {"head_dim": 64, "head_dim_v": 128},
        {"head_dim": 96, "head_dim_v": 64, "dtype": "torch.float16"},
        {"head_dim": 256, "head_dim_v": 256},
        {"num_heads": 64, "num_heads_kv": 2},
        {"num_heads": 2, "num_heads_kv": 2},
        {"device_arch": 103},
        {"causal": True, "local": True, "window_size_left": 31, "window_size_right": 0,
         "has_score_mod": True, "has_mask_mod": True, "pack_gqa": True, "disable_2cta": True},
    ],
)
def test_admitted_decode_selects_swapab(changes):
    inputs = decode(**changes)

    config = select_fwd_config(inputs)

    assert config.use_swapab
    # A power-of-two query-head tile covering the GQA group, and the 256-token KV tile.
    group = inputs.num_heads // inputs.num_heads_kv
    assert config.tile_m >= group and config.tile_m & (config.tile_m - 1) == 0
    assert config.tile_m < 2 * group
    assert (config.tile_n, config.q_stage, config.pack_gqa) == (256, 1, True)
    assert select_fwd_config(inputs._replace(requested_use_swapab=True)) == config


# Problems the SwapAB kernel cannot run, with the rejection reason.
_UNSUPPORTED = [
    ({"has_cu_seqlens_k": True}, "cu_seqlens_k"),
    ({"has_seqused_q": True}, "seqused_q"),
    ({"has_learnable_sink": True}, "learnable_sink"),
    ({"use_block_sparsity": True}, "block_sparse_tensors"),
    ({"has_aux_scalars": True}, "aux_scalars"),
    ({"has_caller_scheduler_metadata": True}, "scheduler_metadata"),
    ({"disable_scheduler_metadata": True}, "disable_scheduler_metadata"),
    ({"seqlen_k_per_split": 1024}, "seqlen_k_per_split"),
    ({"device_arch": 90}, "SM100 or SM103"),
    ({"device_arch": 110}, "SM100 or SM103"),
    ({"requires_grad": True}, "training"),
    ({"total_q": 8, "max_seqlen_q": 2}, "single query token"),
    ({**_VARLEN_PAGED, "max_seqlen_q": 2}, "max_seqlen_q == 1"),
    # A tensor (device) max_seqlen_q is no host bound.
    ({**_VARLEN_PAGED, "has_max_seqlen_q_hint": False}, "max_seqlen_q == 1"),
    ({"dtype": "torch.float8_e4m3fn"}, "dtype"),
    ({"head_dim": 192, "head_dim_v": 128}, "head dimensions"),
    ({"num_heads": 66, "num_heads_kv": 2}, "GQA ratio"),
    ({"page_size": 64}, "page size must be a multiple of 128"),
]
# Selection requests that keep the selector off SwapAB.
_VETOES = [
    {"requested_tile_m": 128, "requested_tile_n": 128},
    {"requested_mma_pv_is_rs": False},
    {"requested_intra_wg_overlap": False},
    {"pack_gqa": False},
    {"requested_use_clc_scheduler": True},
    {"disable_s_ping_pong": True},
]


@pytest.mark.parametrize("changes", [changes for changes, _ in _UNSUPPORTED] + _VETOES)
def test_swapab_request_falls_back_to_the_standard_kernel(changes):
    """Automatic selection and a SwapAB request both take the standard kernel's config for a
    problem SwapAB cannot run or a request vetoes, like an FA_CLC request CLC cannot honor."""
    inputs = decode(**changes)
    standard = select_fwd_config(inputs._replace(requested_use_swapab=False))

    assert not standard.use_swapab
    assert select_fwd_config(inputs) == standard
    assert select_fwd_config(inputs._replace(requested_use_swapab=True)) == standard


@pytest.mark.parametrize(("changes", "match"), _UNSUPPORTED)
def test_explicit_swapab_config_for_unsupported_problem_raises(changes, match):
    """An explicit SwapAB config is forced exactly, so it raises where a request falls back."""
    inputs = decode(**changes)
    swapab = replace(select_fwd_config(decode()), device_capacity=inputs.device_capacity)

    with pytest.raises(ValueError, match=f"SwapAB unsupported: .*{match}"):
        validate_fwd_config(swapab, inputs)


@pytest.mark.parametrize(
    ("changes", "use_swapab"),
    [
        # Many short sequences at head dims <= 128: SwapAB measured 1.12-1.33x slower.
        ({"batch_size": 256, "total_q": 256, "max_seqlen_k": 512}, False),
        ({"batch_size": 256, "total_q": 256, "max_seqlen_k": 512, "has_seqused_k": True,
          "page_size": 256}, False),
        ({"batch_size": 256, "total_q": 256, "max_seqlen_k": 512,
          "head_dim": 64, "head_dim_v": 64}, False),
        # Fewer CTAs, longer KV, or hd256 keep SwapAB.
        ({"batch_size": 255, "total_q": 255, "max_seqlen_k": 512}, True),
        ({"batch_size": 256, "total_q": 256, "max_seqlen_k": 513}, True),
        ({"batch_size": 256, "total_q": 256, "max_seqlen_k": 512,
          "head_dim": 256, "head_dim_v": 256}, True),
        # A large page table hides the real lengths: its capacity keeps SwapAB.
        ({"batch_size": 256, "total_q": 256, "max_seqlen_k": 512, "seqlen_k_capacity": 262144,
          "has_seqused_k": True, "page_size": 128}, True),
    ],
)
def test_automatic_selection_skips_measured_swapab_losses(changes, use_swapab):
    inputs = decode(**changes)

    assert select_fwd_config(inputs).use_swapab is use_swapab
    # A SwapAB request overrides the measured-loss rule.
    assert select_fwd_config(inputs._replace(requested_use_swapab=True)).use_swapab


def test_standard_request_keeps_the_standard_kernel():
    config = select_fwd_config(decode(requested_use_swapab=False))

    assert not config.use_swapab and config.tile_m == 128


@pytest.mark.parametrize(
    ("changes", "num_splits"),
    [
        # Static policy: the largest power of two up to about 64 CTAs (128 for hd64), then at
        # most 32 splits and at least 512 tokens per split.
        ({"batch_size": 1, "total_q": 1, "num_heads_kv": 1, "max_seqlen_k": 10**6}, 32),
        ({"batch_size": 1, "total_q": 1, "num_heads_kv": 1, "head_dim": 64, "head_dim_v": 64,
          "max_seqlen_k": 10**6}, 32),
        ({"batch_size": 16, "total_q": 16, "head_dim": 64, "head_dim_v": 64}, 4),
        ({"batch_size": 16, "total_q": 16}, 2),
        ({"batch_size": 64, "total_q": 64}, 1),
        # The token floor can leave a count that is no power of two.
        ({"max_seqlen_k": 1536}, 3),
        ({"max_seqlen_k": 700}, 1),
        ({"max_seqlen_k": 0}, 1),
        # An empty batch has nothing to split (direct selector callers; the interface returns
        # before selection).
        ({"batch_size": 0, "total_q": 0, "total_k": 0}, 1),
        ({"batch_size": 0, "total_q": 0, "total_k": 0, "has_seqused_k": True,
          "max_seqlen_k": 32768}, 1),
        # The paged capacity, not a caller's max_seqlen_k hint, bounds the split count.
        ({"max_seqlen_k": 600, "seqlen_k_capacity": 4096, "page_size": 256}, 8),
        # An explicit count is kept.
        ({"requested_num_splits": 3}, 3),
    ],
)
def test_split_count(changes, num_splits):
    config = select_fwd_config(decode(**changes))

    assert (config.use_swapab, config.num_splits) == (True, num_splits)


def test_split_count_bounds():
    """Automatic counts stay within the policy's caps over a grid of decode problems."""
    for batch, kv_heads, capacity, head_dim, seqused in itertools.product(
        (1, 3, 16, 256), (1, 2, 8), (0, 255, 512, 4096, 32768, 10**6), (64, 128, 256),
        (False, True),
    ):
        inputs = decode(
            batch_size=batch, total_q=batch, num_heads=4 * kv_heads, num_heads_kv=kv_heads,
            head_dim=head_dim, head_dim_v=head_dim, max_seqlen_k=capacity,
            has_seqused_k=seqused, requested_use_swapab=True,
        )
        config = select_fwd_config(inputs)
        cap = -(-capacity // 256) if needs_split_metadata(config, inputs) else capacity // 512
        assert 1 <= config.num_splits <= min(32, max(1, cap)), inputs


@pytest.mark.parametrize(
    ("changes", "per_sequence", "num_splits"),
    [
        # Below 32K tokens of capacity the static count is kept, even with used lengths.
        ({"has_seqused_k": True, "page_size": 256, "max_seqlen_k": 32767}, False, 8),
        # From 32K, used lengths or varlen Q take per-sequence counts; num_splits caps them at
        # about 8 CTAs per SM and at most 32.
        ({"has_seqused_k": True, "page_size": 256, "max_seqlen_k": 32768}, True, 32),
        ({"has_cu_seqlens_q": True, "total_q": 3, "max_seqlen_k": 32768}, True, 32),
        ({"has_seqused_k": True, "batch_size": 109, "total_q": 109, "max_seqlen_k": 262144},
         True, 6),
        ({"has_seqused_k": True, "batch_size": 296, "total_q": 296, "max_seqlen_k": 262144},
         True, 2),
        # Without used lengths every sequence has the full length: static splits.
        ({"max_seqlen_k": 262144}, False, 8),
        # One split needs no counts.
        ({"has_seqused_k": True, "max_seqlen_k": 262144, "requested_num_splits": 1}, False, 1),
    ],
)
def test_per_sequence_splits(changes, per_sequence, num_splits):
    inputs = decode(**changes)

    config = select_fwd_config(inputs)

    assert config.use_swapab
    assert (needs_split_metadata(config, inputs), config.num_splits) == (
        per_sequence, num_splits
    )


def swapab_grid_shape(num_heads_kv: int, batch: int, num_splits: int, split_major: bool):
    """The SwapAB kernel's SingleTileScheduler launch grid, traced on the host (no launch)."""
    grid = []

    @cute.jit
    def trace():
        args = TileSchedulerArguments(
            1, num_heads_kv, batch, num_splits, 0, 0, 0, 0, tile_shape_mn=(1, 1),
            is_split_kv=True, split_major=split_major,
        )
        params = SingleTileScheduler.to_underlying_arguments(args)
        grid.extend(SingleTileScheduler.get_grid_shape(params))

    cute.compile(trace)
    return tuple(grid)


@pytest.mark.parametrize(
    ("batch", "num_heads_kv", "num_splits", "seqused_k", "use_swapab"),
    [
        (257, 1, 255, True, True),
        (256, 1, 256, True, True),
        (4096, 2, 256, True, True),
        # The batch takes grid Z (or Y when split-major).
        (65535, 1, 1, True, True),
        (65536, 1, 1, True, False),
        # Without per-sequence splits the (KV head, split) pairs take grid Y.
        (1, 2047, 32, False, True),
        (1, 2048, 32, False, False),
    ],
    ids=["work-65535", "work-65536", "work-2M", "batch-65535", "batch-65536", "y-65504",
         "y-65536"],
)
def test_swapab_grid_fits_launch_limits(batch, num_heads_kv, num_splits, seqused_k, use_swapab):
    """Admitted SwapAB configs launch a legal grid that covers every (split, KV head, sequence)
    tile once, also with batch * num_splits CTAs per KV head past 65535 (the grid Y/Z limit).
    Grids past the limit take the standard kernel, and an explicit SwapAB config raises."""
    inputs = decode(
        batch_size=batch, total_q=batch, num_heads=8 * num_heads_kv, num_heads_kv=num_heads_kv,
        head_dim=64, head_dim_v=64, max_seqlen_k=32768, has_seqused_k=seqused_k,
        requested_num_splits=num_splits,
    )
    config = select_fwd_config(inputs)
    if not use_swapab:
        assert not config.use_swapab
        assert not select_fwd_config(inputs._replace(requested_use_swapab=True)).use_swapab
        with pytest.raises(ValueError, match="SwapAB unsupported: .*at most 65535"):
            validate_fwd_config(SWAPAB.config(inputs, num_splits), inputs)
        return
    assert config.use_swapab and config.num_splits == num_splits

    grid = swapab_grid_shape(
        num_heads_kv, batch, num_splits, split_major=needs_split_metadata(config, inputs)
    )

    assert grid[0] <= 2**31 - 1 and max(grid[1:]) <= 65535, grid
    assert grid[0] * grid[1] * grid[2] == batch * num_heads_kv * num_splits


@pytest.mark.parametrize(
    ("changes", "per_sequence"),
    [
        ({"has_cu_seqlens_q": True}, True),
        ({"has_seqused_q": True}, True),
        ({"has_cu_seqlens_q": True, "has_caller_scheduler_metadata": True}, False),
        ({"has_cu_seqlens_q": True, "disable_scheduler_metadata": True}, False),
        ({"has_seqused_k": True}, False),
    ],
)
def test_standard_kernel_per_sequence_splits(changes, per_sequence):
    """The standard kernel takes per-sequence counts with varlen Q unless the caller owns
    (or disables) the scheduler metadata."""
    inputs = decode(**changes, requested_use_swapab=False, max_seqlen_k=16384, total_q=4)

    config = select_fwd_config(inputs)

    assert config.num_splits > 1
    assert needs_split_metadata(config, inputs) is per_sequence


def test_explicit_configs_force_either_kernel():
    inputs = decode()
    swapab = select_fwd_config(inputs)
    standard = select_fwd_config(inputs._replace(requested_use_swapab=False))

    # Both validate for the same problem, whatever the backend request.
    for request in (None, True, False):
        validate_fwd_config(swapab, inputs._replace(requested_use_swapab=request))
        validate_fwd_config(standard, inputs._replace(requested_use_swapab=request))
    # Any split count; derive_fwd_config keeps the SwapAB geometry.
    validate_fwd_config(replace(swapab, num_splits=7), inputs)
    assert derive_fwd_config(swapab, inputs, num_splits=4) == replace(swapab, num_splits=4)


@pytest.mark.parametrize(
    "requests",
    [
        {"requested_tile_m": 128, "requested_tile_n": 128},
        {"requested_mma_pv_is_rs": True},
        {"requested_use_clc_scheduler": True},
        {"disable_s_ping_pong": True},
        {"requested_use_swapab": False},
    ],
)
def test_selection_requests_do_not_veto_an_explicit_config(requests):
    """Requests steer only the selector; an explicit SwapAB config runs as given."""
    inputs = decode(**requests)
    swapab = select_fwd_config(decode())

    assert not select_fwd_config(inputs).use_swapab
    validate_fwd_config(swapab, inputs)


@pytest.mark.parametrize(
    ("changes", "config_changes", "match"),
    [
        ({}, {"tile_m": 16}, "SwapAB tunes only num_splits"),
        ({}, {"tile_n": 128}, "SwapAB tunes only num_splits"),
        ({}, {"use_s_ping_pong": True}, "SwapAB tunes only num_splits"),
        ({}, {"num_splits": 300}, r"num_splits must be in \[1, 256\]"),
        ({"pack_gqa": True}, {"pack_gqa": False}, "disagrees"),
    ],
)
def test_explicit_swapab_config_validation(changes, config_changes, match):
    swapab = replace(select_fwd_config(decode()), **config_changes)

    with pytest.raises(ValueError, match=match):
        validate_fwd_config(swapab, decode(**changes))
