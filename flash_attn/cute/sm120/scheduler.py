# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tile schedulers of the SM12x forward."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    WorkTileInfo,
)


class Sm120UniformBatchScheduler:
    """Map uniform varlen batches without the generic prefix-sum walk.

    SM120 paged decode uses one equally sized query segment per request. Its
    compile-time dispatch proves that invariant before selecting this scheduler,
    so each CTA can recover ``(block, head, batch)`` arithmetically.
    """

    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        num_splits: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        is_split_kv: cutlass.Constexpr[bool] = False
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None
        split_slot_batch_ptr: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "Sm120UniformBatchScheduler.Params":
            assert args.cluster_shape_mn == (
                1,
                1,
            ), "SM120 uniform-batch scheduling requires a 1x1 cluster"
            return Sm120UniformBatchScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                num_splits=args.num_splits,
                tile_shape_mn=args.tile_shape_mn,
                is_split_kv=args.is_split_kv,
                num_splits_dynamic_ptr=args.num_splits_dynamic_ptr,
                split_slot_batch_ptr=args.split_slot_batch_ptr,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
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
            f"SM120 uniform-batch scheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return Sm120UniformBatchScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, clc=None, *, loc=None, ip=None) -> "Sm120UniformBatchScheduler":
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return Sm120UniformBatchScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        if cutlass.const_expr(params.split_slot_batch_ptr is not None):
            # Compact split layout: one CTA per (slot, head), the slots the prepare kernel laid
            # out for the batches' own split counts plus a bounded tail.
            return (cute.size(params.split_slot_batch_ptr) * params.num_head, Int32(1), Int32(1))
        rows_per_batch = params.total_q // params.num_batch
        num_m_blocks = cute.ceil_div(rows_per_batch, params.tile_shape_mn[0])
        return (
            num_m_blocks * params.num_head * params.num_batch,
            params.num_splits,
            Int32(1),
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        params = self.params
        rows_per_batch = params.total_q // params.num_batch
        num_m_blocks = cute.ceil_div(rows_per_batch, params.tile_shape_mn[0])
        if cutlass.const_expr(params.split_slot_batch_ptr is not None):
            slot = self._tile_idx // params.num_head
            head_idx = self._tile_idx - slot * params.num_head
            entry = Int32(params.split_slot_batch_ptr[slot])
            # Slots past the last batch (-1) carry no work: split 0 of 0 splits.
            batch_idx, local, num_splits = Int32(0), Int32(0), Int32(0)
            if entry >= 0:
                batch_idx = entry >> 16
                local = entry & 0xFFFF
                num_splits = Int32(params.num_splits_dynamic_ptr[batch_idx])
            split_idx = local // num_m_blocks
            block = local - split_idx * num_m_blocks
            return WorkTileInfo(
                (Int32(block), Int32(head_idx), Int32(batch_idx), split_idx | (num_splits << 16)),
                self._is_first_block and entry >= 0,
            )
        mh_blocks_per_batch = num_m_blocks * params.num_head
        batch_idx = self._tile_idx // mh_blocks_per_batch
        mh_block = self._tile_idx - batch_idx * mh_blocks_per_batch
        block = mh_block // params.num_head
        head_idx = mh_block - block * params.num_head
        split_idx = self._split_idx if cutlass.const_expr(params.is_split_kv) else Int32(0)
        if cutlass.const_expr(params.is_split_kv and params.num_splits_dynamic_ptr is not None):
            # Per-batch split count in the top 16 bits, as SingleTileVarlenScheduler packs it.
            split_idx = split_idx | (Int32(params.num_splits_dynamic_ptr[batch_idx]) << 16)
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), split_idx),
            self._is_first_block,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        self._is_first_block = False
        return self.get_current_work(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in (self.params, self._tile_idx, self._split_idx):
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        objects = []
        for obj, n_items in zip((self.params, self._tile_idx, self._split_idx), self._values_pos):
            objects.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return Sm120UniformBatchScheduler(*objects, loc=self._loc)
