# Copyright (c) 2025, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

from enum import IntEnum, auto
from typing import Optional, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import cutlass
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisorV2
from cutlass.utils import ClcDynamicPersistentTileScheduler, ClcDynamicPersistentTileSchedulerParams
from cutlass.cute.typing import Boolean
from cutlass.utils.hardware_info import HardwareInfo

from quack.cute_dsl_utils import ParamsBase

import flash_attn.cute.utils as utils
from flash_attn.cute.fast_math import clz


class SchedulingMode(IntEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    CLC = auto()


@dataclass
class SchedulerState(ParamsBase):
    """Runtime state shared by CLC and dynamic persistent tile schedulers:
    the async pipeline and its producer/consumer states.

    Main kernels construct this via `create_clc` / `create_dynamic_persistent`,
    which return the appropriate concrete state (`ClcSchedulerState` or
    `DynamicPersistentSchedulerState`). Schedulers consume it through the
    `ctx: SchedulerState | None` parameter on their `__init__(...)`.
    """

    _pipeline: cutlass.pipeline.PipelineAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create_clc(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcSchedulerState":
        return ClcSchedulerState(pipeline, consumer_state, producer_state, hw_scheduler)

    @staticmethod
    def create_dynamic_persistent(
        *,
        work_info: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "DynamicPersistentSchedulerState":
        return DynamicPersistentSchedulerState(pipeline, consumer_state, producer_state, work_info)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def advance_consumer_state(self, *, loc=None, ip=None):
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)


@dataclass
class ClcSchedulerState(SchedulerState):
    """Owns the runtime state shared by CLC-capable tile schedulers.

    `FlashAttentionForwardSm100` constructs this state because it owns the CLC
    response buffer, mbarrier storage, and launch geometry needed to initialize
    the hardware scheduler and async pipeline. Individual tile schedulers then
    consume this state and map the returned hardware work tiles into their own
    logical `WorkTileInfo` coordinates.

    To add CLC support to a scheduler:
    - implement `clc_problem_shape(params)` so the kernel can create the hardware scheduler
    - map `ctx.initial_work_tile_info()` and `ctx.get_current_work()` into scheduler coordinates
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(self._producer_state, loc=loc, ip=ip)
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)


@dataclass
class DynamicPersistentSchedulerState(SchedulerState):
    """Semaphore-backed: the scheduler class drives atomicAdd + warp-prefix-sum
    and writes the resolved work tile via `write_work_info`."""

    _work_info: cute.Tensor

    def producer_acquire(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)

    def producer_commit(self, *, loc=None, ip=None):
        self._pipeline.producer_commit(self._producer_state, loc=loc, ip=ip)

    def advance_producer_state(self, *, loc=None, ip=None):
        self._producer_state.advance(loc=loc, ip=ip)

    def write_work_info(self, block: Int32, head: Int32, batch: Int32, split: Int32):
        self._work_info[0] = block
        self._work_info[1] = head
        self._work_info[2] = batch
        self._work_info[3] = split


# NOTE [SM100 forward work coordinates]
# With C = cta_group_size (2 for 2CTA, else 1), every CTA in a cluster receives the same
# logical (work_m, head, batch, split) and validity; the CTA rank is not part of work_m.
# Grids count physical CTAs: physical_x = C * cluster_slot + cta_rank. Persistence can run
# many logical tiles on one physical cluster, so a work tile id is not a cluster id.
#
# The SM100 forward passes logical M work tiles in num_block with use_cluster_idx = (C > 1).
# SingleTileLPTScheduler's CLC grid has grid_x = num_block * C; clc_work_to_coords divides
# claimed_x by C and optionally reverses M for LPT, while unsplit 1CTA grids use the
# L2-swizzled static mapping. StaticPersistentTileScheduler takes num_block as a cluster
# count, caps the grid at the SM count rounded down to whole clusters, and strides over work
# by the launched cluster count. Varlen metadata units and the MLA contracts are separate;
# the backward's is NOTE [SM100 backward work coordinates] below. Row mapping: NOTE [SM100
# forward coordinate spaces] in flash_fwd_sm100.py.


# NOTE [SM100 backward work coordinates]
# The SM100 backward tiles KV rows. With C = cta_group_size (2 for 2CTA, else 1), every CTA
# of a cluster receives the same logical (work_n, head, batch) and validity: work_n is the
# C * tile_n-row KV work tile, the CTA rank is not part of it. The kernel applies the rank
# once, work_kv_block(work_n, cta_rank) = work_n * C + cta_rank (NOTE [SM100 backward
# coordinate spaces] in flash_bwd_sm100.py).
#
# The backward passes work tiles in num_block (ceil_div(seqlen_k, C * tile_n)) with
# use_cluster_idx = (C > 1). SingleTileScheduler then reads cluster_idx(); SingleTileLPTBwd-
# Scheduler and SingleTileVarlenScheduler decode blockIdx.x // C and skip their CTA-rank fold.
# Grids still count physical CTAs (physical_x = C * cluster_slot + cta_rank). Without
# use_cluster_idx (sm90 backward, cluster 1) the same schedulers hand out CTA tiles as before.
# The head is a scheduler head (NOTE [SM100 backward coordinate spaces]): num_head counts
# query heads, packed groups, or KV heads under the head loop; qhead_fold tells the
# deterministic schedulers how many query heads' Q / dO / dQaccum one tile touches.


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split).

    For the SM100 forward these are logical work coordinates, shared by every CTA of a
    cluster; see NOTE [SM100 forward work coordinates].
    """

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@runtime_checkable
class TileSchedulerProtocol(Protocol):
    """Protocol defining the interface all tile schedulers must implement.

    Schedulers are responsible for:
    1. Coordinate mapping: linear tile index -> (m_block, head, batch, split)
    2. Work distribution: how to get the next tile (static grid-stride vs dynamic)

    Grids are in physical CTAs, work tiles in logical coordinates; for the SM100 forward
    see NOTE [SM100 forward work coordinates].
    """

    def initial_work_tile_info(self) -> WorkTileInfo:
        """Get the initial work tile for this CTA."""
        ...

    def advance_to_next_work(self, *, loc=None, ip=None):
        """Consumer-side advance: move to next tile and return it.

        For static schedulers: grid-stride increment + get_current_work.
        For dynamic schedulers: consumer wait + get_current_work + consumer release + state advance.
        """
        ...

    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        """Producer-side prefetch of next work tile (no-op for static schedulers).

        For dynamic schedulers: producer acquire (+ issue CLC query) + producer state advance.
        Only called by the single producer warp (CLC: the scheduler warp, or the load warp for
        late claims; dynamic: the first load warp).
        """
        ...

    def producer_tail(self, *, loc=None, ip=None) -> None:
        """Producer-side cleanup after the last tile.

        No-op for static schedulers. For dynamic schedulers: pipeline producer_tail.
        DynamicPersistentVarlenScheduler also reports this CTA finished; the last CTA zeroes
        the semaphore.
        """
        ...


@dataclass
class TileSchedulerArguments(ParamsBase):
    # num_block is in logical work tiles with use_cluster_idx, else in CTA tiles; see
    # NOTE [SM100 forward work coordinates].
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False
    use_cluster_idx: cutlass.Constexpr[bool] = False
    # SingleTileLPTScheduler: one L2 section over the whole problem; see NOTE [global LPT].
    lpt_global: cutlass.Constexpr[bool] = False
    num_splits_dynamic_ptr: Optional[cute.Tensor] = None
    num_m_blocks_ptr: Optional[cute.Tensor] = None
    virtual_batch_idx_ptr: Optional[cute.Tensor] = None
    num_nheads_in_l2_ptr: Optional[cute.Tensor] = None
    cu_total_m_blocks_ptr: Optional[cute.Tensor] = None
    cu_total_splits_m_blocks_ptr: Optional[cute.Tensor] = None
    blocks_to_batch_idx_ptr: Optional[cute.Tensor] = None
    tile_count_semaphore: Optional[cute.Pointer] = None
    persistent_cta_multiplier: cutlass.Constexpr[int] = 1
    # SingleTileScheduler: launch splits as the slowest grid mode, so with per-batch split counts
    # every batch's first splits launch before the excess splits, which exit at once. Cluster
    # shape (1, 1) only.
    split_major: cutlass.Constexpr[bool] = False
    # Backward GQA: each tile processes qhead_fold query heads in total (its packed heads
    # and, under the head loop, every sub-group of the KV head). Deterministic schedulers size
    # their L2 sections using the combined Q, dO, and dQaccum memory footprint of all
    # qhead_fold query heads; the backward passes seqlen_k as unpacked Q rows for that.
    qhead_fold: cutlass.Constexpr[int] = 1


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisorV2
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        use_cluster_idx: cutlass.Constexpr[bool] = False
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None
        num_head_divmod: Optional[FastDivmodDivisorV2] = None  # split-major grid

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisorV2(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
                args.use_cluster_idx,
                args.num_splits_dynamic_ptr,
                FastDivmodDivisorV2(args.num_head) if args.split_major else None,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        assert not args.split_major or args.cluster_shape_mn == (1, 1), (
            "split_major folds the heads into grid X, which clusters span"
        )
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "SingleTileScheduler":
        if const_expr(cute.size(params.cluster_shape_mn) == 1 or not params.use_cluster_idx):
            blk_coord = cute.arch.block_idx()
        else:
            blk_coord = cute.arch.cluster_idx()
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        if const_expr(params.use_cluster_idx):
            # Grid must have num_block * cluster_m physical blocks so that there are num_block clusters
            grid_x = params.num_block * params.cluster_shape_mn[0]
        else:
            grid_x = cute.round_up(params.num_block, params.cluster_shape_mn[0])
        if const_expr(params.num_head_divmod is not None):
            # Split-major: heads vary fastest, then batches, then splits. Only the splits sit in
            # grid Z, since batch * num_splits can pass the 65535 grid Y/Z limit (num_batch
            # alone shares that limit with the default layout).
            return (grid_x * params.num_head, params.num_batch, params.num_splits)
        return (
            grid_x,
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        is_valid = self._is_first_block
        if const_expr(self.params.num_head_divmod is not None):
            # Grid (block * head, batch, split); see get_grid_shape.
            split_idx = batch_idx
            batch_idx = head_idx
            block_idx, head_idx = divmod(block_idx, self.params.num_head_divmod)
        elif const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        # Pack dynamic per-batch num_splits into high 16 bits of split_idx
        if const_expr(self.params.is_split_kv and self.params.num_splits_dynamic_ptr is not None):
            if is_valid:
                num_splits = Int32(self.params.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            is_valid,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)
        # Note: _is_first_block is a Python-only attribute omitted from MLIR values,
        # so it must be restored explicitly after reconstruction.
        scheduler._is_first_block = self._is_first_block
        return scheduler


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_cluster_divmod: FastDivmodDivisorV2
        num_head_divmod: FastDivmodDivisorV2
        total_blocks_cluster: Int32
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            # use_cluster_idx: num_block already counts clusters (NOTE [SM100 forward work coordinates]).
            num_block_cluster = (
                args.num_block
                if const_expr(args.use_cluster_idx)
                else cute.ceil_div(args.num_block, cute.size(args.cluster_shape_mn))
            )
            total_blocks_cluster = num_block_cluster * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmodDivisorV2(num_block_cluster),
                FastDivmodDivisorV2(args.num_head),
                total_blocks_cluster,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"StaticPersistentTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "StaticPersistentTileScheduler":
        if const_expr(cute.size(params.cluster_shape_m) == 1):
            tile_idx = cute.arch.block_idx()[0]
        else:
            tile_idx = cute.arch.cluster_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        max_ctas = (sm_count // params.cluster_shape_m) * params.cluster_shape_m
        grid_x = cutlass.min(max_ctas, params.total_blocks_cluster * params.cluster_shape_m)
        return (grid_x, Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = divmod(self._tile_idx, self.params.num_block_cluster_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self._tile_idx < self.params.total_blocks_cluster
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.cluster_shape_m == 1):
            self._tile_idx += cute.arch.grid_dim()[0]
        else:
            self._tile_idx += cute.arch.cluster_dim()[0]
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_splits: Int32
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisorV2
        l2_minor_divmod: FastDivmodDivisorV2
        l2_major_divmod: FastDivmodDivisorV2
        l2_minor_residual_divmod: FastDivmodDivisorV2
        num_hb_quotient: Int32
        num_splits_divmod: FastDivmodDivisorV2
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
        lpt: cutlass.Constexpr[bool] = True
        use_cluster_idx: cutlass.Constexpr[bool] = True
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileLPTScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            # int64: this product overflows int32 once seqlen_k * (headdim +
            # headdim_v) * element_size > 2**31 (seqlen_k > ~4M for hdim-128 bf16).
            size_one_kv_head = (
                cutlass.Int64(args.seqlen_k) * (args.headdim + args.headdim_v) * args.element_size
            )
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # Seems faster if swizzle is a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = (
                1 if size_l2 < size_one_head else (1 << log2_floor(Int32(size_l2 // size_one_head)))
            )
            if const_expr(args.lpt_global):
                swizzle = Int32(args.num_head * args.num_batch)
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block=args.num_block,
                num_head=args.num_head,
                num_batch=args.num_batch,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisorV2(args.num_head),
                l2_minor_divmod=FastDivmodDivisorV2(swizzle),
                l2_major_divmod=FastDivmodDivisorV2(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmodDivisorV2(max(num_hb_remainder, 1)),
                num_hb_quotient=Int32(num_hb_quotient),
                num_splits=args.num_splits,
                num_splits_divmod=FastDivmodDivisorV2(args.num_splits),
                is_split_kv=args.is_split_kv,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
                lpt=args.lpt,
                use_cluster_idx=args.use_cluster_idx,
                num_splits_dynamic_ptr=args.num_splits_dynamic_ptr,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        ctx: SchedulerState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._ctx = ctx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileLPTScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    def _clc_grid_shape(params: Params):
        num_batch_splits = (
            params.num_batch * params.num_splits
            if const_expr(params.is_split_kv)
            else params.num_batch
        )
        if const_expr(params.use_cluster_idx):
            # Grid must have num_block * cluster_m physical blocks so that there are num_block clusters
            grid_x = params.num_block * params.cluster_shape_m
        else:
            grid_x = cute.round_up(params.num_block, params.cluster_shape_m)
        return (
            grid_x,
            params.num_head,
            num_batch_splits,
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileLPTScheduler._clc_grid_shape(params),
            cluster_shape_mnk=(params.cluster_shape_m, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "SingleTileLPTScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler(
                params, cute.arch.block_idx()[0], Int32(0), ctx, loc=loc, ip=ip
            )
        if const_expr(params.cluster_shape_m > 1 and params.use_cluster_idx):
            # Both CTAs share one m block.
            tile_idx, split_idx, _ = cute.arch.cluster_idx()
        else:
            tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return SingleTileLPTScheduler._clc_grid_shape(params)
        grid_x = params.total_blocks
        if const_expr(params.cluster_shape_m > 1 and params.use_cluster_idx):
            # Convert clusters to CTAs.
            grid_x = grid_x * params.cluster_shape_m
        return (grid_x, params.num_splits, Int32(1))

    @cute.jit
    def clc_work_to_coords(self, work) -> WorkTileInfo:
        """Convert CLC response (block, head, batch_split) to WorkTileInfo.

        Unsplit 1CTA grids reuse the static L2-swizzled LPT mapping.
        Clustered grids use one m block per cluster; num_block counts clusters. Both CTAs
        of a cluster decode the same logical tile; see NOTE [SM100 forward work coordinates].
        """
        if const_expr(self.params.cluster_shape_m == 1 and not self.params.is_split_kv):
            tile_idx = work.tile_idx[0] + self.params.num_block * (
                work.tile_idx[1] + self.params.num_head * work.tile_idx[2]
            )
            return WorkTileInfo(self._static_coords(tile_idx).tile_idx, work.is_valid_tile)
        assert self.params.cluster_shape_m == 1 or self.params.use_cluster_idx, (
            "CLC with clusters requires use_cluster_idx"
        )
        block_idx = work.tile_idx[0]
        if const_expr(self.params.cluster_shape_m > 1):
            block_idx = block_idx // self.params.cluster_shape_m
        if const_expr(self.params.lpt):
            # Longest-processing-time-first: reverse block order
            block_idx = self.params.num_block - 1 - block_idx
        split_idx = Int32(0)
        if const_expr(self.params.is_split_kv):
            batch_idx, split_idx = divmod(work.tile_idx[2], self.params.num_splits_divmod)
        else:
            batch_idx = work.tile_idx[2]
        # Pack dynamic per-batch num_splits into high 16 bits of split_idx
        if const_expr(self.params.is_split_kv and self.params.num_splits_dynamic_ptr is not None):
            if work.is_valid_tile:
                num_splits = Int32(self.params.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (Int32(block_idx), Int32(work.tile_idx[1]), Int32(batch_idx), Int32(split_idx)),
            work.is_valid_tile,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._ctx.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        return self._static_coords(self._tile_idx)

    @cute.jit
    def _static_coords(self, tile_idx: Int32) -> WorkTileInfo:
        # L2-swizzled coordinate mapping (static grid, or unsplit 1CTA CLC)
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(tile_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        # Longest-processing-time-first
        if const_expr(params.lpt):
            block = params.num_block - 1 - block
        is_valid = tile_idx < params.total_blocks
        split_idx = self._split_idx
        # Pack dynamic per-batch num_splits into high 16 bits of split_idx
        if const_expr(params.is_split_kv and params.num_splits_dynamic_ptr is not None):
            if is_valid:
                num_splits = Int32(params.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(split_idx)), is_valid
        )

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._ctx.initial_work_tile_info()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self._ctx.consumer_release(loc=loc, ip=ip)
            return work
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)


class SingleTileLPTBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisorV2
        l2_minor_divmod: FastDivmodDivisorV2
        l2_major_divmod: FastDivmodDivisorV2
        l2_minor_residual_divmod: FastDivmodDivisorV2
        num_hb_quotient: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        spt: cutlass.Constexpr[bool] = True
        # num_block counts work tiles (clusters); see NOTE [SM100 backward work coordinates]
        use_cluster_idx: cutlass.Constexpr[bool] = False

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTBwdScheduler.Params":
            size_l2 = 50 * 1024 * 1024
            # int64: these products overflow int32 at large seqlen_k (> ~4M for
            # hdim-128 bf16; the dqaccum *4 term wraps even sooner).
            size_one_qdo_head = (
                cutlass.Int64(args.seqlen_k) * (args.headdim + args.headdim_v) * args.element_size
            )
            size_one_dqaccum_head = cutlass.Int64(args.seqlen_k) * (args.headdim) * 4
            # size_one_dqaccum_head = 0
            # seqlen_k here is the unpacked Q rows: every query head of the fold adds a full
            # Q / dO / dQaccum footprint
            size_one_head = (size_one_qdo_head + size_one_dqaccum_head) * args.qhead_fold
            log2_floor = lambda n: 31 - clz(n)
            swizzle = (
                1 if size_l2 < size_one_head else (1 << log2_floor(Int32(size_l2 // size_one_head)))
            )
            # swizzle = 8
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            if const_expr(args.use_cluster_idx):
                num_block = args.num_block  # already in work tiles (clusters)
            else:
                num_block = cute.ceil_div(args.num_block, args.cluster_shape_mn[0])
            return SingleTileLPTBwdScheduler.Params(
                total_blocks=(num_block * args.cluster_shape_mn[0])
                * args.num_head
                * args.num_batch,
                num_block=num_block,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisorV2(args.num_head),
                l2_minor_divmod=FastDivmodDivisorV2(swizzle),
                l2_major_divmod=FastDivmodDivisorV2(swizzle * num_block),
                l2_minor_residual_divmod=FastDivmodDivisorV2(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                cluster_shape_mn=args.cluster_shape_mn,
                spt=args.lpt,
                use_cluster_idx=args.use_cluster_idx,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileLPTBwdScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_idx = self._tile_idx // self.params.cluster_shape_mn[0]
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(cluster_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        if cutlass.const_expr(params.spt):
            block = params.num_block - 1 - block
        if cutlass.const_expr(params.cluster_shape_mn[0] > 1 and not params.use_cluster_idx):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block = block * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


@dataclass
class VarlenDecoder(ParamsBase):
    """Per-batch m-block lookup + warp-prefix-sum search-and-decode of the
    varlen work tile. Composed into both `SingleTileVarlenScheduler.Params`
    and `DynamicPersistentVarlenScheduler.Params`.

    `fold_splits_into_scan` controls whether the prefix-sum scan folds per-batch
    `num_splits` into the per-batch tile count (DynamicPersistent) or always
    counts only m_blocks (SingleTileVarlen, where splits are dispatched at the
    grid level and resolved post-scan).
    """

    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    max_kvblock_in_l2: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    is_split_kv: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False
    cluster_shape_m: cutlass.Constexpr[int] = 1
    use_cluster_idx: cutlass.Constexpr[bool] = False
    fold_splits_into_scan: cutlass.Constexpr[bool] = False
    scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    num_m_blocks_ptr: Optional[cute.Tensor] = None
    num_splits_dynamic_ptr: Optional[cute.Tensor] = None
    virtual_batch_idx_ptr: Optional[cute.Tensor] = None
    num_nheads_in_l2_ptr: Optional[cute.Tensor] = None
    cu_total_m_blocks_ptr: Optional[cute.Tensor] = None
    cu_total_splits_m_blocks_ptr: Optional[cute.Tensor] = None
    blocks_to_batch_idx_ptr: Optional[cute.Tensor] = None

    @staticmethod
    @cute.jit
    def create(
        args: TileSchedulerArguments,
        *,
        fold_splits_into_scan: bool,
        head_swizzle: bool = False,
        cluster_shape_m: int = 1,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> "VarlenDecoder":
        size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
        # if backward, this is qdo block size
        kv_block_size = (args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1]
        # if backward, add dqaccum block size to calculate swizzle
        if head_swizzle:
            kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
            # num_n_blocks below counts unpacked query rows (from the K tiles): the whole fold
            kv_block_size *= args.qhead_fold
        max_kvblock_in_l2 = size_l2 // kv_block_size
        return VarlenDecoder(
            num_head=args.num_head,
            num_batch=args.num_batch,
            num_splits=args.num_splits,
            max_kvblock_in_l2=max_kvblock_in_l2,
            tile_shape_mn=args.tile_shape_mn,
            qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
            is_split_kv=args.is_split_kv,
            lpt=args.lpt,
            head_swizzle=head_swizzle,
            cluster_shape_m=cluster_shape_m,
            use_cluster_idx=args.use_cluster_idx,
            fold_splits_into_scan=fold_splits_into_scan,
            scheduling_mode=scheduling_mode,
            mCuSeqlensQ=args.mCuSeqlensQ,
            mSeqUsedQ=args.mSeqUsedQ,
            num_m_blocks_ptr=args.num_m_blocks_ptr,
            num_splits_dynamic_ptr=args.num_splits_dynamic_ptr,
            virtual_batch_idx_ptr=args.virtual_batch_idx_ptr,
            num_nheads_in_l2_ptr=args.num_nheads_in_l2_ptr,
            cu_total_m_blocks_ptr=args.cu_total_m_blocks_ptr,
            cu_total_splits_m_blocks_ptr=args.cu_total_splits_m_blocks_ptr,
            blocks_to_batch_idx_ptr=args.blocks_to_batch_idx_ptr,
        )

    @cute.jit
    def _num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        """Per-batch m-block count"""
        batch_idx = lane + bidb_start
        is_valid = batch_idx < self.num_batch and lane < cute.arch.WARP_SIZE - 1
        if cutlass.const_expr(self.num_m_blocks_ptr is not None):
            num_m_blocks_raw = Int32(0)
            if is_valid:
                if cutlass.const_expr(self.virtual_batch_idx_ptr is not None):
                    real_batch_idx = self.virtual_batch_idx_ptr[batch_idx]
                else:
                    real_batch_idx = batch_idx
                num_m_blocks_raw = Int32(self.num_m_blocks_ptr[real_batch_idx])
            return cute.ceil_div(num_m_blocks_raw, self.cluster_shape_m) if is_valid else Int32(0)
        if cutlass.const_expr(self.virtual_batch_idx_ptr is not None):
            seqlen = Int32(0)
            if is_valid:
                real_batch_idx = self.virtual_batch_idx_ptr[batch_idx]
                if cutlass.const_expr(self.mSeqUsedQ is not None):
                    seqlen = self.mSeqUsedQ[real_batch_idx]
                else:
                    seqlen = self.mCuSeqlensQ[real_batch_idx + 1] - self.mCuSeqlensQ[real_batch_idx]
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                seqlen *= self.qhead_per_kvhead_packgqa
            return (
                cute.ceil_div(cute.ceil_div(seqlen, self.tile_shape_mn[0]), self.cluster_shape_m)
                if is_valid
                else Int32(0)
            )
        if cutlass.const_expr(self.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < self.num_batch:
                seqlen = self.mSeqUsedQ[batch_idx]
        else:
            assert self.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= self.num_batch:
                cur_cu_seqlen = self.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
            seqlen *= self.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(cute.ceil_div(seqlen, self.tile_shape_mn[0]), self.cluster_shape_m)
            if is_valid
            else Int32(0)
        )

    @cute.jit
    def _num_splits(self, lane: Int32, bidb_start: Int32) -> Int32:
        if cutlass.const_expr(not self.fold_splits_into_scan):
            return Int32(1)
        batch_idx = lane + bidb_start
        is_valid = batch_idx < self.num_batch and lane < cute.arch.WARP_SIZE - 1
        if cutlass.const_expr(not self.is_split_kv):
            return Int32(1)
        elif cutlass.const_expr(self.num_splits_dynamic_ptr is not None):
            num_splits = Int32(0)
            if is_valid:
                if cutlass.const_expr(self.virtual_batch_idx_ptr is not None):
                    batch_idx = self.virtual_batch_idx_ptr[batch_idx]
                num_splits = self.num_splits_dynamic_ptr[batch_idx]
            return num_splits
        else:
            return Int32(0) if not is_valid else self.num_splits

    @cute.jit
    def decode(
        self,
        next_tile_idx: Int32,
        bidb_start: Int32,
        group_start_tile: Int32,
    ) -> Tuple[Int32, Int32, Int32, Int32, Int32, Int32, Boolean]:
        """Search varlen batches via warp-level prefix sums and decode the work tile.

        Returns
            - block
            - head_idx
            - batch_idx
            - split_idx
            - num_splits
            - group_start_tile
            - is_valid
        """
        # The scan counts m_blocks unless splits are folded into it, so the cumsum used as a
        # hint must match: cu_total_splits_m_blocks only applies to the folded (persistent)
        # layout, where SingleTileVarlen keeps splits in a separate grid dim.
        if const_expr(self.fold_splits_into_scan):
            cu_hint_ptr = self.cu_total_splits_m_blocks_ptr
        else:
            cu_hint_ptr = self.cu_total_m_blocks_ptr
        # Both SingleTileVarlen STATIC and CLC; not DynamicPersistent (where
        # warp-scan's _bidb_start resumption already amortizes per-call cost).
        hint_mode_ok = const_expr(
            cu_hint_ptr is not None
            and (
                self.scheduling_mode == SchedulingMode.STATIC
                or self.scheduling_mode == SchedulingMode.CLC
            )
        )
        # O(1) inverted index (flat block -> batch) replaces the warp scan outright. Needs the
        # unfolded layout, where a batch owns a contiguous tile range given by the cumsum.
        use_blocks_to_batch = const_expr(
            hint_mode_ok
            and self.blocks_to_batch_idx_ptr is not None
            and not self.fold_splits_into_scan
        )
        use_cumsum_hint = const_expr(hint_mode_ok and not use_blocks_to_batch)

        lane_idx = cute.arch.lane_idx()
        if const_expr(use_blocks_to_batch):
            batch_idx = Int32(self.blocks_to_batch_idx_ptr[next_tile_idx // self.num_head])
            num_m_blocks, num_splits = Int32(0), Int32(1)
            is_valid = batch_idx < self.num_batch
            if is_valid:
                cu_lo = cu_hint_ptr[batch_idx]
                num_m_blocks = cu_hint_ptr[batch_idx + 1] - cu_lo
                group_start_tile = cu_lo * self.num_head
            else:
                batch_idx = Int32(self.num_batch)
        else:
            if const_expr(use_cumsum_hint):
                target = next_tile_idx // self.num_head
                lo = utils.get_batch_from_cu_tensor(target, cu_hint_ptr)
                group_size = Int32(cute.arch.WARP_SIZE - 1)
                bidb_start = (lo // group_size) * group_size
                group_start_tile = cu_hint_ptr[bidb_start] * self.num_head

            num_m_blocks = self._num_m_blocks(lane_idx, bidb_start=bidb_start)
            num_splits = self._num_splits(lane_idx, bidb_start=bidb_start)
            per_batch = num_m_blocks * num_splits if const_expr(self.is_split_kv) else num_m_blocks
            cumulative = utils.warp_prefix_sum(per_batch, lane_idx)
            m_blocks_in_group = cute.arch.shuffle_sync(cumulative, cute.arch.WARP_SIZE - 1)
            group_end_tile = m_blocks_in_group * self.num_head + group_start_tile

            # Padded cu_seqlens repeat the last offset; once a group starts there, every
            # remaining batch is empty and the tile is invalid without scanning the tail.
            stop_at_padding = const_expr(
                self.mCuSeqlensQ is not None
                and self.mSeqUsedQ is None
                and self.virtual_batch_idx_ptr is None
            )
            batch_idx = bidb_start
            while group_end_tile <= next_tile_idx:
                batch_idx += cute.arch.WARP_SIZE - 1
                at_padding = Boolean(False)
                if const_expr(stop_at_padding):
                    if batch_idx < self.num_batch:
                        at_padding = self.mCuSeqlensQ[batch_idx] == self.mCuSeqlensQ[self.num_batch]
                if batch_idx >= self.num_batch or at_padding:
                    batch_idx = Int32(self.num_batch)
                    group_end_tile = next_tile_idx + 1
                else:
                    num_m_blocks = self._num_m_blocks(lane_idx, bidb_start=batch_idx)
                    num_splits = self._num_splits(lane_idx, bidb_start=batch_idx)
                    per_batch = (
                        num_m_blocks * num_splits if const_expr(self.is_split_kv) else num_m_blocks
                    )
                    cumulative = utils.warp_prefix_sum(per_batch, lane_idx)
                    m_blocks_in_group = cute.arch.shuffle_sync(cumulative, cute.arch.WARP_SIZE - 1)
                    group_end_tile += m_blocks_in_group * self.num_head

            is_valid = batch_idx < self.num_batch
            if is_valid:
                group_start_tile = group_end_tile - m_blocks_in_group * self.num_head
                batch_idx_in_group = cute.arch.popc(
                    cute.arch.vote_ballot_sync(
                        group_start_tile + cumulative * self.num_head <= next_tile_idx
                    )
                )
                batch_idx += batch_idx_in_group
                num_m_blocks_prev_lane = (
                    Int32(0)
                    if batch_idx_in_group == 0
                    else cute.arch.shuffle_sync(cumulative, batch_idx_in_group - 1)
                )
                group_start_tile += num_m_blocks_prev_lane * self.num_head
                num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
                if const_expr(self.is_split_kv):
                    num_splits = cute.arch.shuffle_sync(num_splits, batch_idx_in_group)

        block, head_idx, split_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            mh_block = next_tile_idx - group_start_tile

            if const_expr(self.lpt or self.head_swizzle):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # TODO: is there any case where num_m_blocks is 0?
                if const_expr(not self.is_split_kv) or num_splits == 1:
                    if const_expr(self.num_nheads_in_l2_ptr is not None):
                        if const_expr(self.virtual_batch_idx_ptr is not None):
                            nheads_in_l2 = Int32(
                                self.num_nheads_in_l2_ptr[self.virtual_batch_idx_ptr[batch_idx]]
                            )
                        else:
                            nheads_in_l2 = Int32(self.num_nheads_in_l2_ptr[batch_idx])
                    else:
                        # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                        num_n_blocks = (
                            num_m_blocks
                            * self.tile_shape_mn[0]
                            * self.cluster_shape_m
                            // self.qhead_per_kvhead_packgqa
                            // self.tile_shape_mn[1]
                        )
                        # Seems faster to have nheads_in_l2 be a power of 2
                        nheads_in_l2 = (
                            16
                            if num_n_blocks * 16 <= self.max_kvblock_in_l2
                            else (
                                8
                                if num_n_blocks * 8 <= self.max_kvblock_in_l2
                                else (
                                    4
                                    if num_n_blocks * 4 <= self.max_kvblock_in_l2
                                    else (2 if num_n_blocks * 2 <= self.max_kvblock_in_l2 else 1)
                                )
                            )
                        )
                        nheads_in_l2 = min(nheads_in_l2, self.num_head)
                    mh_in_l2 = nheads_in_l2 * num_m_blocks
                    section_idx = mh_block // mh_in_l2
                    l2_mod = mh_block - section_idx * mh_in_l2
                    nheads_in_this_section = (
                        nheads_in_l2
                        if nheads_in_l2 * (section_idx + 1) <= self.num_head
                        else self.num_head - section_idx * nheads_in_l2
                    )
                    block = l2_mod // nheads_in_this_section
                    head_idx_residual = l2_mod - block * nheads_in_this_section
                    head_idx = section_idx * nheads_in_l2 + head_idx_residual
                else:
                    head_split_idx = mh_block // num_m_blocks
                    block = mh_block - head_split_idx * num_m_blocks
                    head_idx = head_split_idx // num_splits
                    split_idx = head_split_idx - head_idx * num_splits
                if const_expr(self.lpt):
                    block = num_m_blocks - 1 - block
            else:
                head_split_idx = mh_block // num_m_blocks
                block = mh_block - head_split_idx * num_m_blocks
                if const_expr(self.is_split_kv):
                    head_idx = head_split_idx // num_splits
                    split_idx = head_split_idx - head_idx * num_splits
                else:
                    head_idx = head_split_idx

            if const_expr(self.cluster_shape_m > 1 and not self.use_cluster_idx):
                bidx_in_cluster = cute.arch.block_in_cluster_idx()
                block = block * self.cluster_shape_m + bidx_in_cluster[0]

        return block, head_idx, batch_idx, split_idx, num_splits, group_start_tile, is_valid


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        total_q: Int32
        scheduling_mode: cutlass.Constexpr[SchedulingMode]
        decoder: VarlenDecoder

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileVarlenScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            assert args.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
            decoder = VarlenDecoder.create(
                args,
                fold_splits_into_scan=False,
                head_swizzle=args.head_swizzle,
                cluster_shape_m=args.cluster_shape_mn[0],
                scheduling_mode=scheduling_mode,
                loc=loc,
                ip=ip,
            )
            return SingleTileVarlenScheduler.Params(
                total_q=args.total_q,
                scheduling_mode=scheduling_mode,
                decoder=decoder,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        ctx: SchedulerState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._is_first_block = True
        self._ctx = ctx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        return SingleTileVarlenScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileVarlenScheduler.get_grid_shape(params),
            cluster_shape_mnk=(1, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, ctx: SchedulerState | None = None, *, loc=None, ip=None
    ) -> "SingleTileVarlenScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            block_idx = cute.arch.block_idx()
            split_idx = Int32(0)
            if const_expr(params.decoder.is_split_kv):
                split_idx = block_idx[1]
            return SingleTileVarlenScheduler(
                params,
                block_idx[0],
                split_idx,
                ctx,
                loc=loc,
                ip=ip,
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileVarlenScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        d = params.decoder
        total_blocks_max = (
            params.total_q + d.num_batch * (d.cluster_shape_m * d.tile_shape_mn[0] - 1)
        ) // d.tile_shape_mn[0]
        # Round down to nearest multiple of cluster since odd excess is always padding.
        total_blocks_max = total_blocks_max // d.cluster_shape_m * d.cluster_shape_m
        return (total_blocks_max * d.num_head, d.num_splits, Int32(1))

    @cute.jit
    def _decode_work_tile(self) -> WorkTileInfo:
        """Map self._tile_idx to (block, head, batch, split) via warp-level prefix sums."""
        d = self.params.decoder
        next_tile_idx = self._tile_idx // d.cluster_shape_m
        block, head_idx, batch_idx, _, _, _, is_valid = d.decode(next_tile_idx, Int32(0), Int32(0))
        is_valid = is_valid and self._is_first_block
        split_idx = self._split_idx if const_expr(d.is_split_kv) else Int32(0)
        if const_expr(d.virtual_batch_idx_ptr is not None):
            if is_valid:
                batch_idx = d.virtual_batch_idx_ptr[batch_idx]
        # Pack dynamic per-batch num_splits into high 16 bits of split_idx
        if const_expr(d.is_split_kv and d.num_splits_dynamic_ptr is not None):
            if is_valid:
                num_splits = Int32(d.num_splits_dynamic_ptr[batch_idx])
                split_idx = split_idx | (num_splits << 16)
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(split_idx)),
            is_valid,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self._ctx.get_current_work()
            # Default to grid_dim (one past last valid flat index) so _decode_work_tile
            # returns is_valid=False when CLC is exhausted. CLC tile_idx is garbage when
            # invalid, so we can't trust it. Local-then-assign avoids CuTe DSL structural
            # mismatch on self inside the runtime if.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.decoder.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._decode_work_tile()

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self._ctx.initial_work_tile_info()
            # See get_current_work for why grid_dim and local-then-assign.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.decoder.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._decode_work_tile()

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self._ctx.consumer_release(loc=loc, ip=ip)
            return work
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._ctx.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._ctx]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        scheduler = self.__class__(*obj_list, loc=self._loc)
        # See the note on Python-only attributes in SingleTileScheduler.
        scheduler._is_first_block = self._is_first_block
        return scheduler


# DynamicPersistentVarlenScheduler semaphore: the low bits count claimed tiles (all tiles handed out
# in one launch), the high bits count finished CTAs (at most the grid, about the SM count).
SEMAPHORE_CLAIM_BITS = 24
SEMAPHORE_FINISHED_BITS = 32 - SEMAPHORE_CLAIM_BITS
SEMAPHORE_CLAIM_MASK = (1 << SEMAPHORE_CLAIM_BITS) - 1
SEMAPHORE_MAX_CTAS = (1 << SEMAPHORE_FINISHED_BITS) - 1


class DynamicPersistentVarlenScheduler:
    """Persistent varlen scheduler: a grid of at most SEMAPHORE_MAX_CTAS CTAs claims tiles from
    ``tile_count_semaphore`` with an atomic add, after each CTA's first tile (its block index).

    The semaphore must be zero at launch and belong to one in-flight launch at a time; the last
    CTA to finish resets it (see ``producer_tail``), so launches ordered on one stream can reuse it
    without a host-side reset.
    """

    # Claimed tiles per launch must stay below this (the semaphore's claim field).
    MAX_TILES = 1 << SEMAPHORE_CLAIM_BITS

    @staticmethod
    def max_claimable_tiles(
        total_q: int,
        batch_size: int,
        m_tile: int,
        num_head: int,
        num_head_kv: int,
        pack_gqa: bool,
        num_splits: int,
    ) -> int:
        """Host-side upper bound on the tiles one launch can claim, in the grid's geometry: M
        tiles of ``m_tile`` (= q_stage * tile_m) rows over packed rows (PackGQA) per scheduled head.
        Each nonempty request adds at most one partial tile and needs at least one of the total_q
        rows, so padded request slots beyond total_q add none."""
        rows_per_token = num_head // num_head_kv if pack_gqa else 1
        heads = num_head_kv if pack_gqa else num_head
        requests = min(batch_size, total_q)
        return (total_q * rows_per_token // m_tile + requests + 1) * heads * max(num_splits, 1)

    @dataclass
    class Params(ParamsBase):
        total_q: Int32
        decoder: VarlenDecoder
        tile_count_semaphore: Optional[cute.Pointer] = None
        persistent_cta_multiplier: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "DynamicPersistentVarlenScheduler.Params":
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            # TODO: support non-trivial cluster shapes in a follow-on PR
            assert args.cluster_shape_mn[0] == 1 and args.cluster_shape_mn[1] == 1, (
                "DynamicPersistentVarlenScheduler currently requires cluster_shape_mn == (1, 1)"
            )
            decoder = VarlenDecoder.create(
                args,
                fold_splits_into_scan=True,
                scheduling_mode=SchedulingMode.DYNAMIC,
                loc=loc,
                ip=ip,
            )
            return DynamicPersistentVarlenScheduler.Params(
                total_q=args.total_q,
                decoder=decoder,
                tile_count_semaphore=args.tile_count_semaphore,
                persistent_cta_multiplier=args.persistent_cta_multiplier,
            )

    def __init__(
        self,
        params: Params,
        ctx: SchedulerState,
        bidb_start: Int32,
        group_start_tile: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._ctx = ctx
        self._bidb_start = bidb_start
        self._group_start_tile = group_start_tile
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.DYNAMIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert scheduling_mode == SchedulingMode.DYNAMIC, (
            f"DynamicPersistentVarlenScheduler only supports DYNAMIC, got {scheduling_mode!r}"
        )
        return DynamicPersistentVarlenScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        ctx: SchedulerState,
        *,
        loc=None,
        ip=None,
    ) -> "DynamicPersistentVarlenScheduler":
        return DynamicPersistentVarlenScheduler(params, ctx, Int32(0), Int32(0), loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        d = params.decoder
        total_blocks_max = (
            params.total_q + d.num_batch * (d.tile_shape_mn[0] - 1)
        ) // d.tile_shape_mn[0]
        total_blocks = total_blocks_max * d.num_head * d.num_splits
        hardware_info = HardwareInfo()
        sm_count = min(
            hardware_info.get_device_multiprocessor_count() * params.persistent_cta_multiplier,
            SEMAPHORE_MAX_CTAS,
        )
        return (cutlass.min(sm_count, total_blocks), Int32(1), Int32(1))

    @cute.jit
    def get_current_work(
        self,
        next_tile_idx: Int32,
        bidb_start: Int32,
        group_start_tile: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[WorkTileInfo, Int32]:
        d = self.params.decoder
        block, head_idx, batch_idx, split_idx, num_splits, group_start_tile, is_valid = d.decode(
            next_tile_idx, bidb_start, group_start_tile
        )
        if const_expr(d.is_split_kv and d.num_splits_dynamic_ptr is not None):
            if is_valid:
                split_idx = split_idx | (num_splits << 16)
        if const_expr(d.virtual_batch_idx_ptr is not None):
            if is_valid:
                batch_idx = d.virtual_batch_idx_ptr[batch_idx]
        return (
            WorkTileInfo(
                (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(split_idx)),
                is_valid,
            ),
            group_start_tile,
        )

    @cute.jit
    def prefetch_next_work(self, *, loc=None, ip=None):
        """Claim the next tile and publish it to the CTA's warps.

        The first load warp calls this after issuing the current tile's loads. Claiming only once
        the one-slot pipeline is free keeps a CTA at most one tile ahead.
        """
        ctx = self._ctx
        ctx.producer_acquire()
        next_tile_idx = Int32(0)
        if cute.arch.lane_idx() == 0:
            claimed = utils.atomic_add_i32(1, self.params.tile_count_semaphore)
            next_tile_idx = cute.arch.grid_dim()[0] + (claimed & SEMAPHORE_CLAIM_MASK)
        next_tile_idx = cute.arch.shuffle_sync(next_tile_idx, 0)
        work_info, new_group_start_tile = self.get_current_work(
            next_tile_idx, self._bidb_start, self._group_start_tile
        )
        # Advance scan state so the next prefetch resumes from this tile's batch
        # group instead of restarting at batch 0.
        self._bidb_start = Int32(work_info.tile_idx[2])
        self._group_start_tile = new_group_start_tile
        with cute.arch.elect_one():
            block, head_idx, batch_idx, split_idx = work_info.tile_idx
            ctx.write_work_info(block, head_idx, batch_idx, split_idx)
            ctx.producer_commit()
        ctx.advance_producer_state()

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        ctx = self._ctx
        ctx.consumer_wait()
        block = ctx._work_info[0]
        head_idx = ctx._work_info[1]
        batch_idx = ctx._work_info[2]
        split_idx = ctx._work_info[3]
        is_valid = batch_idx < self.params.decoder.num_batch
        work_info = WorkTileInfo((block, head_idx, batch_idx, split_idx), is_valid)
        ctx.consumer_release()
        return work_info

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        cta_tile_idx, _, _ = cute.arch.block_idx()
        work_info, new_group_start_tile = self.get_current_work(cta_tile_idx, Int32(0), Int32(0))
        self._bidb_start = Int32(work_info.tile_idx[2])
        self._group_start_tile = new_group_start_tile
        return work_info

    @cute.jit
    def producer_tail(self, *, loc=None, ip=None):
        """Report this CTA finished; the last CTA resets the semaphore for the next launch.

        Each CTA calls this once, after its last claim (including CTAs whose first tile is
        invalid). Claims and the report come from the same thread to the same address, so the CTA
        that sees ``grid - 1`` others finished observes every claim and can zero the semaphore. A
        finished count of at least ``grid`` means the semaphore was not zero at launch or is shared
        with a concurrent launch, so tiles may have been skipped: trap instead.
        """
        self._ctx.producer_tail(loc=loc, ip=ip)
        if cute.arch.lane_idx() == 0:
            finished = Int32(1 << SEMAPHORE_CLAIM_BITS)
            old = utils.atomic_add_i32(finished, self.params.tile_count_semaphore)
            num_finished = (old >> SEMAPHORE_CLAIM_BITS) & SEMAPHORE_MAX_CTAS
            if num_finished == cute.arch.grid_dim()[0] - 1:
                utils.atomic_add_i32(-(old + finished), self.params.tile_count_semaphore)
            elif num_finished >= cute.arch.grid_dim()[0]:
                cute.arch.inline_ptx("trap;", read_only_args=[])

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._ctx, self._bidb_start, self._group_start_tile]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._ctx, self._bidb_start, self._group_start_tile],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)
