# Copyright (c) 2025, Tri Dao.

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32


class TileSchedulerParams:
    def __init__(
        self,
        # block_size: cutlass.Constexpr[int],
        num_blocks: Int32,
        num_head: Int32,
        num_batch: Int32,
        is_persistent: cutlass.Constexpr[bool] = False,
        # qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1,  # Only pass in if using packed GQA
        *,
        loc=None,
        ip=None,
    ):
        # self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_head = num_head
        self.num_batch = num_batch
        self.is_persistent = is_persistent
        # self.qhead_per_kvhead_packgqa = qhead_per_kvhead_packgqa
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.num_blocks, self.num_head, self.num_batch]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.num_blocks, self.num_head, self.num_batch], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return TileSchedulerParams(
            # self.block_size, *(tuple(obj_list)), self.qhead_per_kvhead_packgqa, loc=self._loc
            *(tuple(obj_list)),
            self.is_persistent,
            loc=self._loc,
        )


class SingleTileScheduler:
    def __init__(self, blk_coord: cute.Coord, *, loc=None, ip=None):
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def create(params: TileSchedulerParams, *, loc=None, ip=None) -> "SingleTileScheduler":
        blk_coord = cute.arch.block_idx()
        return SingleTileScheduler(blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: TileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return params.num_blocks, params.num_head, params.num_batch

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        return cutlass.utils.WorkTileInfo(self._blk_coord, self._is_first_block)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    def __init__(
        self,
        num_blocks: Int32,
        num_head: Int32,
        total_blocks: Int32,
        tile_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.num_blocks = num_blocks
        self.num_head = num_head
        self.total_blocks = total_blocks
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def create(params: TileSchedulerParams, *, loc=None, ip=None) -> "SingleTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        total_blocks = params.num_blocks * params.num_head * params.num_batch
        return StaticPersistentTileScheduler(
            params.num_blocks, params.num_head, total_blocks, tile_idx, loc=loc, ip=ip
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: TileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        total_blocks = params.num_blocks * params.num_head * params.num_batch
        return (cutlass.min(sm_count, total_blocks), Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        hn_idx = self._tile_idx // self.num_blocks
        block_idx = self._tile_idx - hn_idx * self.num_blocks
        batch_idx = hn_idx // self.num_head
        head_idx = hn_idx - batch_idx * self.num_head
        is_valid = self._tile_idx < self.total_blocks
        return cutlass.utils.WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._tile_idx += cute.arch.grid_dim()[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.num_blocks, self.num_head, self.total_blocks, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.num_blocks, self.num_head, self.total_blocks, self._tile_idx], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)
