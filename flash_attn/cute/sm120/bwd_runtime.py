# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host side of the SM12x backward: tile policy, scratch buffers and kernel launches."""

import math
from functools import lru_cache
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

from flash_attn.cute.bwd_accum_tile import AccumTileOrder
from flash_attn.cute.block_sparsity import (
    normalize_block_sparse_config_bwd,
    to_cute_block_sparse_tensors,
)
from flash_attn.cute.config import (
    _SM90_ONLY_NEUTRAL_FIELDS,
    _SM100_ONLY_NEUTRAL_FIELDS,
    BwdConfig,
    BwdHeuristicInputs,
)
from flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from flash_attn.cute.utils import AuxData
from flash_attn.cute.sm120.flash_bwd import (
    FlashAttentionBackwardDKDVSm120,
    FlashAttentionBackwardDQSm120,
)

# Head-dim multiple of the padded tiles, including the row stride of the fp32 accumulate tiles
# (a 96-wide tile is not padded to 128; the pre / postprocess copies tile any such stride).
HDIM_MULTIPLE_OF = 32


def _wide_head_dim(head_dim: int, head_dim_v: int) -> bool:
    """Head dims whose dK / dV accumulators do not fit one register set per KV row."""
    return max(head_dim, head_dim_v) > 192 or head_dim_v > 128


def _fused_dq(head_dim: int, head_dim_v: int, deterministic: bool) -> bool:
    """Whether the dK / dV kernel also reduces dQ (else the split pass: dK / dV, then a dQ
    kernel). At hd 256 the 32-wide KV tile makes the fused dQ reduction (one fp32 tile per Q tile
    per KV tile) cost more than the dQ kernel's second pass over K / V, so the split pass is the
    default there too."""
    return not deterministic and (
        not _wide_head_dim(head_dim, head_dim_v) or (head_dim, head_dim_v) == (192, 192)
    )


def bwd_config(inputs: BwdHeuristicInputs) -> BwdConfig:
    """The dK / dV kernel's tiles as a BwdConfig, for the FA4 interface's shared shape checks
    before it hands the call to flash_attn_bwd, which plans the launch itself."""
    head_dim, head_dim_v = inputs.head_dim, inputs.head_dim_v
    tile_m, tile_n, stages_q, stages_do, warps, _, _ = dkdv_config(
        head_dim,
        head_dim_v,
        not _fused_dq(head_dim, head_dim_v, inputs.deterministic),
        inputs.num_heads // inputs.num_heads_kv,
    )
    return BwdConfig(
        device_capacity=inputs.device_capacity,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages_Q=stages_q,
        num_stages_dO=stages_do,
        num_threads=warps * 32,
        spt=False,
        registers=None,
        **_SM90_ONLY_NEUTRAL_FIELDS,
        **_SM100_ONLY_NEUTRAL_FIELDS,
    )


def dkdv_config(head_dim: int, head_dim_v: int, split_dq: bool, qhead_per_kvhead: int = 1):
    """(tile_m, tile_n, stages_Q, stages_dO, warps, warps_on_kv_SdP, warps_on_kv_dKV).

    Every KV tile re-reads the visible Q / dO tiles and, fused, reduces a dQ piece into L2,
    so L2 traffic per FLOP falls with tile_n; the 99 KB shared-memory budget then sets
    tile_m. With all warps on the KV axis P^T and dS^T are register operands. Wide head dims
    split dK / dV over the head dim instead (2 x 2 warps) and stage P^T / dS^T in shared
    memory. The split pass (deterministic, and the hd-256 default) shares the padded per-tile
    scratch with the dQ kernel, so it uses the dQ kernel's tile_m.
    """
    hd_max = max(head_dim, head_dim_v)
    if hd_max <= 64:
        return 64, 128, 2, 2, 8, 8, 8
    if (head_dim, head_dim_v) == (192, 192) and not split_dq:
        # Fused hd 192: 8 warps (S^T on 4 KV x 2 Q, dK / dV on 4 KV x 2 head-dim warps, P / dS
        # in smem) hold both 192-wide fp32 accumulators without spills, and avoid the split
        # pass, whose 2-warp dQ kernel cannot keep the tensor pipe busy.
        return 32, 64, 2, 1, 8, 4, 4
    if _wide_head_dim(head_dim, head_dim_v):
        # hd 256: a 32-row K + V tile is 32 KB; dK + dV split over two head-dim halves.
        return 32, 32, 2, 1, 4, 2, 2
    if head_dim > 128:
        # hd 192 (-> 128): K / V alone take 40 KB of a 64-row KV tile. With GQA / MQA, 8 warps
        # (S^T on 4 KV x 2 Q, dK / dV on 4 KV x 2 head-dim warps, P / dS in smem) avoid the
        # 4-warp register-operand layout's spills; for MHA the P / dS smem round trip costs
        # more than that saves.
        if split_dq:
            # The split pass (deterministic) keeps 64-row tiles. At (192, 128) its 4-warp
            # layout holds 224 fp32 accumulators per thread (dK, dV, S, dP) and spills heavily,
            # so it takes 8 warps for every group size.
            eight_warps = (head_dim, head_dim_v) == (192, 128)
            return 64, 64, 1, 1, 8 if eight_warps else 4, 4, 4
        eight_warps = qhead_per_kvhead > 1 and (head_dim, head_dim_v) == (192, 128)
        return (32, 64, 2, 1, 8, 4, 4) if eight_warps else (32, 64, 2, 1, 4, 4, 4)
    # The fused dQ GEMM runs in swizzle-atom-wide head-dim pieces: 64 columns for padded head
    # dims that are multiples of 64, else 32 (each of the 4 head-dim warps takes 8 columns).
    if split_dq:
        return 64, 64, 2, 1, 4, 4, 4
    return 32, 128, 2, 1, 8, 8, 8


def dq_config(head_dim: int, head_dim_v: int):
    """(tile_m, tile_n, stages_KV, warps) of the split (deterministic) dQ kernel."""
    if _wide_head_dim(head_dim, head_dim_v):
        if (head_dim, head_dim_v) in ((256, 256), (256, 128)):
            # A 16-row single-stage K / V tile halves shared memory to 50 KB, so two CTAs
            # (4 warps) share an SM instead of one at 2 warps.
            return 32, 16, 1, 2
        return 32, 32, 2, 2
    return (64, 64, 1, 4) if head_dim > 128 else (64, 64, 2, 4)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _num_sms(device_index: int) -> int:
    """SMs of the current context, as the forward's selection counts them."""
    # Lazy: the interface imports this module (and tests patch the cute_dsl_utils function).
    from flash_attn.cute import cute_dsl_utils
    from flash_attn.cute.interface import _get_device_arch

    return cute_dsl_utils.get_num_sms_for_selection(device_index, _get_device_arch())


# Longest dK/dV CTA, in Q-tile iterations, past which a further head split pays off: the last
# partial wave (and, causal, the CTAs LPT cannot pack) then costs more than the fp32 dK / dV
# accumulation of the split. Uniform (non-causal) CTAs stop splitting at _HEAD_SPLIT_MAX_WAVES,
# where the last wave is a small fraction.
_HEAD_SPLIT_MAX_CTA_ITERS = 320
_HEAD_SPLIT_MAX_WAVES = 8


def _head_split(
    q,
    k,
    cu_seqlens_k,
    tile_m,
    tile_n,
    qhead_per_kvhead,
    num_head_kv,
    max_seqlen_q,
    causal,
    row_bytes,
) -> int:
    """Split of each KV head's Q-head group over CTAs: the smallest that gives the dK/dV grid
    two waves and keeps the longest CTA within _HEAD_SPLIT_MAX_CTA_ITERS Q-tile iterations
    (or, non-causal, reaches _HEAD_SPLIT_MAX_WAVES); dense causal calls then split further
    until the (batch, split) units in flight fit in L2 (row_bytes: one Q head's Q, dO and
    fused dQaccum row).

    Without a split one CTA serves a KV tile for every Q head of its group, so MQA and
    wide GQA can leave most SMs idle; split groups reduce their dK / dV through fp32 adds.
    """
    if qhead_per_kvhead == 1:
        return 1
    kv_tiles = (k.shape[1] + tile_n - 1) // tile_n
    if cu_seqlens_k is None:
        n_tiles = k.shape[0] * kv_tiles
        max_seqlen_q = q.shape[1]
    else:
        n_tiles = (k.shape[0] + tile_n - 1) // tile_n + cu_seqlens_k.shape[0] - 1
    ctas = n_tiles * num_head_kv
    num_sms = _num_sms(q.device.index)
    # Without max_seqlen_q (varlen) only the wave target applies.
    m_tiles = 0 if max_seqlen_q is None else (max_seqlen_q + tile_m - 1) // tile_m
    splits = [s for s in range(1, qhead_per_kvhead + 1) if qhead_per_kvhead % s == 0]
    split = next(
        (
            s
            for s in splits
            if ctas * s >= 2 * num_sms
            and (
                qhead_per_kvhead // s * m_tiles <= _HEAD_SPLIT_MAX_CTA_ITERS
                or (not causal and ctas * s >= _HEAD_SPLIT_MAX_WAVES * num_sms)
            )
        ),
        qhead_per_kvhead,
    )
    if not causal or cu_seqlens_k is not None:
        return split
    # Causal KV tiles of one (batch, split) start at different Q tiles and run different
    # lengths, so they drift apart and reuse the unit's rows only while every unit in flight (a
    # wave over its KV tiles) still fits in L2. Non-causal tiles walk the Q tiles in lockstep.
    units = max(num_sms // kv_tiles, 1)
    l2_bytes = _l2_bytes(q.device.index)
    return next(
        (
            s
            for s in splits
            if s >= split and units * (qhead_per_kvhead // s) * q.shape[1] * row_bytes <= l2_bytes
        ),
        qhead_per_kvhead,
    )


# Deterministic head splits keep one fp32 dK + dV slot per split, summed after the dK/dV
# kernel. Peak slot scratch, the slots plus their sum, stays within this many bytes.
_SLOT_SCRATCH_BUDGET = 256 << 20
# dK/dV MMA rate of one SM, FLOP / us; only weighs compute against the slot traffic.
_DKDV_FLOP_PER_SM_US = 2.0e6
_SLOT_MIN_GAIN = 0.05


@lru_cache(maxsize=None)
def _l2_bytes(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).L2_cache_size


@lru_cache(maxsize=None)
def _is_integrated(device_index: int) -> bool:
    return bool(getattr(torch.cuda.get_device_properties(device_index), "is_integrated", False))


@lru_cache(maxsize=None)
def _dram_bytes_per_us(device_index: int) -> float:
    # Cached: the property query costs more than a short backward call.
    props = torch.cuda.get_device_properties(device_index)
    # CUDA reports the effective data rate on integrated (LPDDR) devices.
    rate = 1 if getattr(props, "is_integrated", False) else 2
    return props.memory_clock_rate * 1e3 * props.memory_bus_width / 8 * rate / 1e6


def _dkdv_grid(q, k, tile_n, heads) -> tuple[int, int]:
    """(dK/dV CTAs of a dense call, SMs of the current context)."""
    num_sms = _num_sms(q.device.index)
    return k.shape[0] * ((k.shape[1] + tile_n - 1) // tile_n) * heads, num_sms


def _l2_group_bytes(device_index: int) -> int:
    """L2 bytes of Q / dO / dQaccum per head group of the dense dK/dV launch order: half the L2,
    leaving room for the K / V tiles and dQ reductions that pass through it too."""
    return _l2_bytes(device_index) // 2


# Per-part cost of splitting a KV tile's Q range, relative to the unsplit CTA (prologue, the
# dK / dV fixup and the fp32 dQ accumulator's reduction contention), fitted to short-K grids.
_Q_SPLIT_PART_COST_DQ_ACCUM = 0.05
# Cost of merging a split KV tile's parts, in Q rows of the dK / dV main loop: every part stores
# its fp32 dK / dV slot and arrives (fixed), and the last to arrive loads one slot per part.
# Both move tile_n x head-dim fp32 data, as one Q row of the main loop does tile_n x head-dim
# MMA work, so the cost in rows holds across head dims.
_MERGE_ROWS_FIXED = 45
_MERGE_ROWS_PER_PART = 22


def _q_range_parts(total: int, num_sms: int, m_tiles: int, part_cost: float) -> int:
    """Parts per KV tile for a grid of `total` equal CTAs: the split minimizing
    ceil(total * p / SMs) / p + part_cost * (p - 1) (whole waves of 1 / p CTAs plus a per-part
    cost), at least 2 Q tiles per part, taken if it beats no split by 15 %."""
    full = -(-total // num_sms)
    best, best_est = 1, float(full)
    for parts in range(2, min(16, m_tiles // 2) + 1):
        est = -(-total * parts // num_sms) / parts + part_cost * (parts - 1)
        if est < best_est:
            best, best_est = parts, est
    return best if best_est <= 0.85 * full else 1


def _direct_dq_split(q, k, tile_m, tile_n, heads) -> tuple[int, int]:
    """(tail_tiles, parts) of a single-KV / KV-sweep grid (one CTA per (batch, KV head slot),
    over every KV tile): every tile's Q range is split into the parts minimizing
    ceil(total * p / SMs) * (ceil(m_tiles / p) + merge(p)), whole waves of CTAs that each run
    their Q tiles and then the merge (in Q tiles), at least 2 Q tiles per part, taken if it
    beats no split by 15 %."""
    _, num_sms = _dkdv_grid(q, k, tile_n, heads)
    total = k.shape[0] * heads
    m_tiles = (q.shape[1] + tile_m - 1) // tile_m
    full = -(-total // num_sms) * m_tiles
    best, best_est = 1, float(full)
    for parts in range(2, min(16, m_tiles // 2) + 1):
        merge = (_MERGE_ROWS_FIXED + _MERGE_ROWS_PER_PART * parts) / tile_m
        est = -(-total * parts // num_sms) * (-(-m_tiles // parts) + merge)
        if est < best_est:
            best, best_est = parts, est
    return (total, best) if best_est <= 0.85 * full else (0, 1)


def _tail_split(q, k, tile_m, tile_n, heads, causal, local, seq_tensors):
    """(tail_tiles, parts): the dK/dV grid's partial last wave, r of its uniform CTAs, runs as
    r * parts CTAs of 1 / parts of the work, parts = SMs // r (at most 4). A grid under one
    wave splits every tile (_q_range_parts).

    Uniform (dense, non-causal, non-local) grids only: there the last r tiles of the launch
    order are the last wave.
    """
    if causal or local or any(t is not None for t in seq_tensors):
        return 0, 1
    total, num_sms = _dkdv_grid(q, k, tile_n, heads)
    if total < num_sms:
        # Under one wave: split every tile's Q range.
        m_tiles = (q.shape[1] + tile_m - 1) // tile_m
        parts = _q_range_parts(total, num_sms, m_tiles, _Q_SPLIT_PART_COST_DQ_ACCUM)
        return (total, parts) if parts >= 2 else (0, 1)
    tail = total % num_sms
    if tail == 0:
        return 0, 1
    parts = min(4, num_sms // tail)
    return (tail, parts) if parts >= 2 else (0, 1)


# Names of the dK/dV compile-cache key fields, in order.
DKDV_KEY_FIELDS = (
    "kernel",
    "dtype",
    "head_dim",
    "head_dim_v",
    "qhead_per_kvhead",
    "causal",
    "local",
    "window_left",
    "window_right",
    "fused",
    "tile_m",
    "tile_n",
    "stages_q",
    "stages_do",
    "warps",
    "kv_warps_sdp",
    "kv_warps_dkv",
    "head_split",
    "split_slots",
    "tail_split",
    "l2_group_bytes",
    "l2_group_ctas",
    "tma_ws",
    "direct_dq",
    "kv_sweep",
    "sparse_key",
    "seq_key",
    "mod_key",
)


def _tma_producer_eligible(fused: bool, head_dim: int, head_dim_v: int) -> bool:
    """Whether the fused dK / dV kernel takes the TMA producer path: every fused d64 / d96 /
    d128 call."""
    return fused and (head_dim, head_dim_v) in ((64, 64), (96, 96), (128, 128))


def _dkv_accum_shapes(k, cu_seqlens_k, tile_n, heads, hdims):
    """Shapes of the row-major fp32 dK / dV accumulators (padded like the Q scratch, per
    tile_n) for `heads` accumulator heads; hdims are the rounded head dims."""
    if cu_seqlens_k is None:
        lead, rows = (k.shape[0], heads), _round_up(k.shape[1], tile_n)
    else:
        lead, rows = (heads,), _round_up(k.shape[0] + cu_seqlens_k.shape[0] * tile_n - 1, tile_n)
    return [(*lead, rows * hd) for hd in hdims]


def _slot_split_choice(
    ctas,
    group,
    m_tiles,
    tile_m,
    tile_n,
    head_dim,
    head_dim_v,
    slot_bytes,
    causal,
    num_sms,
    dram_bytes_per_us,
) -> int:
    """Deterministic head split for `ctas` unsplit dK/dV CTAs (one per SM at a time).

    est(s) = max(waves * CTA time, longest CTA time) / s + slot traffic, waves = ceil(ctas s /
    SMs) (causal: ctas s / SMs, as LPT fills the waves with unequal CTAs). A CTA's
    time is its S^T, dP^T, dV, dK FLOPs at _DKDV_FLOP_PER_SM_US (half on average for causal,
    whose first KV tile still sees every Q tile). Slot traffic is the logical bytes of zeroing
    s slots, adding into them (read + write) and summing them into one (read s, write 1), at
    DRAM bandwidth: (4 s + 1) * slot_bytes. The best split within _SLOT_SCRATCH_BUDGET
    ((s + 1) * slot_bytes) is taken if est is at least _SLOT_MIN_GAIN below no split.
    """
    if group == 1:
        return 1
    longest_us = group * m_tiles * 4 * tile_m * tile_n * (head_dim + head_dim_v)
    longest_us /= _DKDV_FLOP_PER_SM_US
    cta_us = longest_us * (0.5 if causal else 1.0)

    def est_us(split):
        # Equal non-causal CTAs run in whole waves; LPT packs unequal causal ones.
        waves = ctas * split / num_sms if causal else -(-ctas * split // num_sms)
        traffic = 0 if split == 1 else (4 * split + 1) * slot_bytes / dram_bytes_per_us
        return max(waves * cta_us, longest_us) / split + traffic

    best = 1
    for split in range(2, group + 1):
        if group % split or (split + 1) * slot_bytes > _SLOT_SCRATCH_BUDGET:
            continue
        if est_us(split) < est_us(best):
            best = split
    return best if est_us(best) <= (1 - _SLOT_MIN_GAIN) * est_us(1) else 1


def _deterministic_head_split(
    q,
    k,
    v,
    seq_tensors,
    local,
    mods,
    tile_m,
    tile_n,
    qhead_per_kvhead,
    num_head_kv,
    causal,
    hdims,
) -> int:
    """The shapes the split model covers: dense full-length calls at d = dv = 64 / 128 without
    mods (causal: square, the model's triangle). Other calls keep the unsplit deterministic
    pass."""
    if (
        (q.shape[-1], v.shape[-1]) not in ((64, 64), (128, 128))
        or any(t is not None for t in seq_tensors)
        or local
        or mods
        or (causal and q.shape[1] != k.shape[1])
    ):
        return 1
    return _slot_head_split(q, k, v, tile_m, tile_n, qhead_per_kvhead, num_head_kv, causal, hdims)


def _slot_head_split(q, k, v, tile_m, tile_n, qhead_per_kvhead, num_head_kv, causal, hdims) -> int:
    """_slot_split_choice for a dense call with the device's SM count and DRAM bandwidth."""
    shapes = _dkv_accum_shapes(k, None, tile_n, num_head_kv, hdims)
    return _slot_split_choice(
        ctas=k.shape[0] * ((k.shape[1] + tile_n - 1) // tile_n) * num_head_kv,
        group=qhead_per_kvhead,
        m_tiles=(q.shape[1] + tile_m - 1) // tile_m,
        tile_m=tile_m,
        tile_n=tile_n,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        slot_bytes=sum(math.prod(shape) for shape in shapes) * 4,
        causal=causal,
        num_sms=_num_sms(q.device.index),
        dram_bytes_per_us=_dram_bytes_per_us(q.device.index),
    )


def flash_attn_bwd(
    q,
    k,
    v,
    out,
    dout,
    lse,
    *,
    softmax_scale: float,
    causal: bool,
    local: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    max_seqlen_q: Optional[int],
    deterministic: bool,
    dq,
    dk,
    dv,
    dlse,
    learnable_sink,
    block_sparse_tensors=None,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    fake_mode: bool,
    compile_cache: dict,
):
    # Lazy: interface imports this module.
    from flash_attn.cute import utils
    from flash_attn.cute.flash_bwd_postprocess import LearnableSinkBwdTensors
    from flash_attn.cute.interface import (
        _bwd_postprocess_convert,
        _bwd_preprocess,
        torch2cute_dtype_map,
    )

    num_head, head_dim = q.shape[-2:]
    num_head_kv, head_dim_v = v.shape[-2], v.shape[-1]
    qhead_per_kvhead = num_head // num_head_kv
    dtype = torch2cute_dtype_map[q.dtype]
    device = q.device
    fused = _fused_dq(head_dim, head_dim_v, deterministic)
    tile_m, tile_n, stages_q, stages_do, warps, kv_warps_sdp, kv_warps_dkv = dkdv_config(
        head_dim, head_dim_v, not fused, qhead_per_kvhead
    )
    accum_multiple = HDIM_MULTIPLE_OF
    accum_hdims = (_round_up(head_dim, accum_multiple), _round_up(head_dim_v, accum_multiple))
    if not deterministic:
        head_split = _head_split(
            q,
            k,
            cu_seqlens_k,
            tile_m,
            tile_n,
            qhead_per_kvhead,
            num_head_kv,
            max_seqlen_q,
            causal and not local,
            # Q, dO and (fused) the fp32 dQaccum row of one Q head.
            q.element_size() * (head_dim + head_dim_v) + (4 * accum_hdims[0] if fused else 0),
        )
    else:
        head_split = _deterministic_head_split(
            q,
            k,
            v,
            (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k),
            local,
            score_mod is not None or mask_mod is not None,
            tile_m,
            tile_n,
            qhead_per_kvhead,
            num_head_kv,
            causal,
            accum_hdims,
        )
    # Deterministic splits add into one fp32 slot each, summed after the dK/dV kernel.
    split_slots = deterministic and head_split > 1
    seq_tensors = (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    # Fused d64 / d96 / d128: a TMA producer warpgroup feeds two out-of-phase consumer groups.
    tma_ws = _tma_producer_eligible(fused, head_dim, head_dim_v)
    # Dense calls whose keys fit one KV tile: the kernel stores complete dQ tiles, so there is
    # no fp32 dQ accumulator (zero fill, reductions, postprocess). Non-causal, non-local calls
    # over two KV tiles take the same route with one CTA sweeping both for its Q range
    # (kv_sweep). Its fp32 partials move 2 (N - 1) dQ-sized passes against the accumulator's
    # N + 2 (zero fill, N reductions, postprocess read); past two KV tiles that shrinking saving
    # does not pay for the longer serial sweep of each CTA.
    n_kv = (k.shape[1] + tile_n - 1) // tile_n
    sparse, sparse_key = None, None
    if block_sparse_tensors is not None:
        sparse, sparse_key = _block_sparse_bwd(
            block_sparse_tensors, q, k, fused, deterministic, tile_m, tile_n
        )
    direct_dq = (
        tma_ws
        and sparse is None
        and all(t is None for t in seq_tensors)
        and (n_kv == 1 or (n_kv == 2 and not causal and not local))
        and learnable_sink is None
    )
    kv_sweep = direct_dq and n_kv > 1
    tail_tiles, tail_split = 0, 1
    if tma_ws:
        # The producer refills dO as soon as both groups' dV is done: one stage.
        stages_do = 1
        heads = num_head_kv * head_split
        if direct_dq:
            tail_tiles, tail_split = _direct_dq_split(q, k, tile_m, tile_n, heads)
        elif sparse is None:
            tail_tiles, tail_split = _tail_split(
                q, k, tile_m, tile_n, heads, causal, local, seq_tensors
            )
    if not fused:
        dq_tile_m, dq_tile_n, dq_stages, dq_warps = dq_config(head_dim, head_dim_v)
        # The dPsum / LSE scratch is padded per tile_m and shared by both kernels.
        assert dq_tile_m == tile_m
    scratch_tile_m = tile_m

    # Per-head scratch, padded like SeqlenInfoQK's padded offsets: one tile of slack per
    # sequence for varlen Q.
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        scratch_shape = (batch_size, num_head, _round_up(seqlen_q, scratch_tile_m))
    else:
        total_q = q.shape[0]
        scratch_shape = (
            num_head,
            _round_up(total_q + cu_seqlens_q.shape[0] * scratch_tile_m - 1, scratch_tile_m),
        )
    head_dim_rounded = _round_up(head_dim, accum_multiple)
    dpsum = torch.empty(*scratch_shape, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(*scratch_shape, dtype=torch.float32, device=device)
    if direct_dq:
        # The kernel stores complete dQ tiles; Q tiles that see no key (causal with seqlen_q >
        # seqlen_k, local) are not visited, so their dQ is zeroed here.
        dq_accum = dq
        if local or (causal and q.shape[1] > k.shape[1]):
            dq.zero_()
        # fp32 dQ partials between KV sweeps (no zero fill: the first sweep stores).
        dq_part = (
            torch.empty(
                *scratch_shape[:-1],
                scratch_shape[-1] * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            if kv_sweep
            else None
        )
    else:
        dq_part = None
        dq_accum = (
            torch.empty(
                *scratch_shape[:-1],
                scratch_shape[-1] * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            if fused
            else None
        )
    _bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        None if direct_dq else dq_accum,
        cu_seqlens_q,
        seqused_q,
        dlse,
        dtype,
        head_dim,
        head_dim_v,
        scratch_tile_m,
        hdim_multiple_of=accum_multiple,
        fake_mode=fake_mode,
    )

    mod_key = (
        utils.hash_callable(score_mod) if score_mod is not None else None,
        utils.hash_callable(score_mod_bwd) if score_mod_bwd is not None else None,
        utils.hash_callable(mask_mod) if mask_mod is not None else None,
        get_aux_tensor_metadata(aux_tensors) if aux_tensors is not None else None,
        tuple(type(x) for x in aux_scalars) if aux_scalars is not None else None,
    )
    # Compile-time aux tensors carry the static layout facts that key the cache; launches
    # pass the torch tensors.
    aux_compile = AuxData(
        [to_cute_aux_tensor(t) for t in aux_tensors] if aux_tensors is not None else None,
        aux_scalars,
    )
    aux_call = AuxData(aux_tensors, aux_scalars)
    tail_args = (None,) * 4
    # A KV-sweep CTA meets the other parts of its Q range once per KV tile.
    fixups = tail_tiles * (n_kv if kv_sweep else 1)
    if tail_split > 1:
        # Without a head split each tail part stores to its own fp32 scratch slot, summed in
        # part order by the last to arrive (else the parts add into dk_out / dv_out).
        tail_scratch = (
            [
                torch.empty(fixups, tail_split * tile_n * hd, dtype=torch.float32, device=device)
                for hd in accum_hdims
            ]
            if head_split == 1
            else [None, None]
        )
        tail_args = (
            tail_tiles,
            *tail_scratch,
            torch.zeros(fixups, dtype=torch.int32, device=device) if head_split == 1 else None,
        )
    if head_split == 1:
        dk_out, dv_out = dk, dv
    else:
        heads = num_head_kv * (head_split if split_slots else 1)
        dk_out, dv_out = (
            torch.zeros(*shape, dtype=torch.float32, device=device)
            for shape in _dkv_accum_shapes(k, cu_seqlens_k, tile_n, heads, accum_hdims)
        )
    seq_key = tuple(t is None for t in seq_tensors)
    l2_group_bytes = _l2_group_bytes(device.index)
    # With uniform CTAs a head group spans at most one wave of dK / dV CTAs, so its KV tiles
    # share Q / dO and dQaccum in L2 (see SingleTileLPTBwdScheduler). Causal, local and
    # block-sparse CTAs keep the wider group, whose longest-first order packs the last wave.
    uniform = not causal and not local and sparse_key is None
    l2_group_ctas = _num_sms(device.index) if uniform else 0
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def cute_seq(t):
        return to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None

    key = (
        "sm120_dkdv",
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        local,
        window_size_left is not None,
        window_size_right is not None,
        fused,
        tile_m,
        tile_n,
        stages_q,
        stages_do,
        warps,
        kv_warps_sdp,
        kv_warps_dkv,
        head_split,
        split_slots,
        tail_split,
        l2_group_bytes,
        l2_group_ctas,
        tma_ws,
        direct_dq,
        kv_sweep,
        sparse_key,
        seq_key,
        mod_key,
    )
    assert len(key) == len(DKDV_KEY_FIELDS)
    if key not in compile_cache:
        kernel = FlashAttentionBackwardDKDVSm120(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            tile_m,
            tile_n,
            stages_q,
            stages_do,
            warps * 32,
            False,
            causal,
            local,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=warps // kv_warps_sdp,
            AtomLayoutNdKV=kv_warps_dkv,
            AtomLayoutMdQ=min(warps, tile_m // 16),
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            mask_mod=mask_mod,
            fuse_dq=fused,
            head_split=head_split,
            split_slots=split_slots,
            tail_split=tail_split,
            l2_group_bytes=l2_group_bytes,
            l2_group_ctas=l2_group_ctas,
            tma_ws=tma_ws,
            direct_dq=direct_dq,
            kv_sweep=kv_sweep,
            q_subtile_factor=sparse_key[0] if sparse_key is not None else 1,
            kv_subtile_factor=sparse_key[1] if sparse_key is not None else 1,
            accum_hdim_multiple=accum_multiple,
        )
        compile_cache[key] = cute.compile(
            kernel,
            *[
                to_cute_tensor(t) if t is not None else None
                for t in (q, k, v, dout, lse_log2, dpsum, dq_accum, dk_out, dv_out)
            ],
            softmax_scale,
            *[cute_seq(t) for t in seq_tensors],
            window_size_left,
            window_size_right,
            # Runtime value: shapes sharing the key differ in their tail.
            cutlass.Int32(tail_args[0]) if tail_args[0] is not None else None,
            *[
                to_cute_tensor(t, assumed_align=4, leading_dim=t.ndim - 1)
                if t is not None
                else None
                for t in tail_args[1:]
            ],
            to_cute_tensor(dq_part) if dq_part is not None else None,
            to_cute_block_sparse_tensors(sparse) if sparse is not None else None,
            aux_compile,
            stream,
            options="--enable-tvm-ffi",
        )
    if not fake_mode:
        compile_cache[key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk_out,
            dv_out,
            softmax_scale,
            *seq_tensors,
            window_size_left,
            window_size_right,
            *tail_args,
            dq_part,
            _block_sparse_args(sparse),
            aux_call,
        )
    if split_slots:
        dk_out, dv_out = (_sum_slots(t, head_split) for t in (dk_out, dv_out))
    if head_split > 1:
        for accum, grad, hdim, scale in (
            (dk_out, dk, head_dim, softmax_scale),
            (dv_out, dv, head_dim_v, 1.0),
        ):
            _bwd_postprocess_convert(
                accum,
                grad,
                scale,
                cu_seqlens_k,
                seqused_k,
                120,
                dtype,
                hdim,
                tile_n,
                warps * 32,
                min(warps, tile_n // 16),
                False,
                accum_order=AccumTileOrder.ROW_MAJOR,
                hdim_multiple_of=accum_multiple,
                fake_mode=fake_mode,
            )

    if direct_dq:
        return dq, dk, dv
    if fused:
        dsink = torch.empty_like(learnable_sink) if learnable_sink is not None else None
        _bwd_postprocess_convert(
            dq_accum,
            dq,
            softmax_scale,
            cu_seqlens_q,
            seqused_q,
            120,
            dtype,
            head_dim,
            tile_m,
            warps * 32,
            min(warps, tile_m // 16),
            False,
            accum_order=AccumTileOrder.ROW_MAJOR,
            sink_tensors=(
                LearnableSinkBwdTensors(dpsum, lse, learnable_sink, dsink)
                if learnable_sink is not None
                else None
            ),
            hdim_multiple_of=accum_multiple,
            fake_mode=fake_mode,
        )
        return (dq, dk, dv) if learnable_sink is None else (dq, dk, dv, dsink)

    dq_head_group = _dq_head_group(
        q, k, causal, local, seq_tensors, head_dim, head_dim_v, qhead_per_kvhead
    )
    key = (
        "sm120_dq",
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        local,
        window_size_left is not None,
        window_size_right is not None,
        dq_tile_m,
        dq_tile_n,
        dq_stages,
        dq_warps,
        seq_key,
        mod_key,
    )
    if key not in compile_cache:
        kernel = FlashAttentionBackwardDQSm120(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            dq_tile_m,
            dq_tile_n,
            dq_stages,
            1,
            dq_warps * 32,
            False,
            causal,
            local,
            SdP_swapAB=False,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=dq_warps,
            AtomLayoutNdKV=dq_warps,
            AtomLayoutMdQ=dq_warps,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            mask_mod=mask_mod,
        )
        compile_cache[key] = cute.compile(
            kernel,
            *[to_cute_tensor(t) for t in (q, k, v, dout, lse_log2, dpsum, dq)],
            softmax_scale,
            *[cute_seq(t) for t in seq_tensors],
            window_size_left,
            window_size_right,
            dq_head_group,
            aux_compile,
            stream,
            options="--enable-tvm-ffi",
        )
    if not fake_mode:
        compile_cache[key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq,
            softmax_scale,
            *seq_tensors,
            window_size_left,
            window_size_right,
            dq_head_group,
            aux_call,
        )
    if learnable_sink is None:
        return dq, dk, dv
    return dq, dk, dv, _dsink(learnable_sink, lse, dpsum, cu_seqlens_q, seqused_q, tile_m)


def _block_sparse_bwd(tensors, q, k, fused, deterministic, tile_m, tile_n):
    """Normalized Q-direction block-sparse tensors and their compile-key part
    (q_subtile_factor, kv_subtile_factor, broadcast pattern) for the fused dK/dV kernel.

    The block size defaults to (128, 128), as on SM90; it must be a multiple of the kernel
    tile. Deterministic calls (the split dQ kernel would need the forward-direction lists)
    and the wide head dims (which take that kernel by default) are not supported.
    """
    if deterministic:
        raise NotImplementedError(
            "deterministic block-sparse backward is supported on SM100 / SM110 only"
        )
    if not fused:
        raise NotImplementedError(
            "block-sparse backward on SM 12.x needs head_dim <= 192 and head_dim_v <= 128"
        )
    block_q, block_kv = tensors.block_size if tensors.block_size is not None else (128, 128)
    if block_q % tile_m or block_kv % tile_n:
        raise ValueError(
            f"Block sparsity on SM 12.x needs block_size {(block_q, block_kv)} to be a "
            f"multiple of the backward tile ({tile_m}, {tile_n})"
        )
    normalized, pattern = normalize_block_sparse_config_bwd(
        tensors._replace(block_size=(block_q, block_kv)),
        batch_size=q.shape[0],
        num_head=q.shape[2],
        seqlen_q=q.shape[1],
        seqlen_k=k.shape[1],
        block_size=(tile_m, tile_n),
        q_subtile_factor=block_q // tile_m,
        kv_subtile_factor=block_kv // tile_n,
    )
    return normalized, (block_q // tile_m, block_kv // tile_n, pattern)


def _block_sparse_args(tensors):
    """Launch-time block-sparse argument: the tensors in BlockSparseTensors field order."""
    if tensors is None:
        return None
    return (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
        tensors.cu_total_m_blocks,
        tensors.cu_block_idx_offsets,
        tensors.dq_write_order,
        tensors.dq_write_order_full,
    )


def _dq_head_group(q, k, causal, local, seq_tensors, head_dim, head_dim_v, qhead_per_kvhead):
    """(head, batch) pairs per longest-first group of the dQ kernel's causal tiles; 0 keeps the
    original tile order.

    Groups are whole KV-head groups whose K / V fit half the L2. Integrated devices keep the
    original order.
    """
    if (
        not causal
        or local
        or any(t is not None for t in seq_tensors)
        or _is_integrated(q.device.index)
    ):
        return 0
    kv_bytes = k.shape[1] * (head_dim + head_dim_v) * k.element_size()
    return min(
        q.shape[2] * q.shape[0],
        max(1, _l2_bytes(q.device.index) // 2 // kv_bytes) * qhead_per_kvhead,
    )


def _sum_slots(accum, slots):
    """(.., h_kv * slots, n) fp32 slot accumulators -> (.., h_kv, n) in one reduction over the
    slot dimension: no atomics, so a fixed configuration gives the same bits every call."""
    return accum.view(*accum.shape[:-2], -1, slots, accum.shape[-1]).sum(dim=-2)


def _dsink(learnable_sink, lse, dpsum, cu_seqlens_q, seqused_q, tile_m):
    """dsink[h] = -sum over valid rows of exp(sink[h] - lse) * dPsum, without host syncs."""
    if cu_seqlens_q is None:
        seqlen = lse.shape[-1]
        dpsum_rows = dpsum[..., :seqlen]
        weight = torch.exp(learnable_sink.float()[None, :, None] - lse) * dpsum_rows
        if seqused_q is not None:
            valid = torch.arange(seqlen, device=lse.device)[None, :] < seqused_q[:, None]
            weight = weight * valid[:, None, :]
        return (-weight.sum(dim=(0, 2))).to(learnable_sink.dtype)
    # Varlen: token t of sequence b sits at row padded_offset(b) + t - cu[b] of the scratch.
    total = lse.shape[-1]
    tokens = torch.arange(total, device=lse.device)
    batch = torch.searchsorted(cu_seqlens_q[1:], tokens, right=True)
    start = cu_seqlens_q[batch]
    padded = (start + batch * tile_m) // tile_m * tile_m
    weight = torch.exp(learnable_sink.float()[:, None] - lse) * dpsum[:, padded + tokens - start]
    if seqused_q is not None:
        weight = weight * (tokens - start < seqused_q[batch])[None, :]
    return (-weight.sum(dim=1)).to(learnable_sink.dtype)
