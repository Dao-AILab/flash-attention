# Copyright (c) 2025, Tri Dao.

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from flash_attn.cute.fast_math import FastDivmod


class TileSchedulerParams:
    def __init__(
        self,
        # block_size: cutlass.Constexpr[int],
        num_block: Int32,
        num_head: Int32,
        num_batch: Int32,
        num_block_divmod: FastDivmod,
        num_head_divmod: FastDivmod,
        is_persistent: cutlass.Constexpr[bool] = False,
        # qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1,  # Only pass in if using packed GPA
        *,
        loc=None,
        ip=None,
    ):
        # self.block_size = block_size
        self.num_block = num_block
        self.num_head = num_head
        self.num_batch = num_batch
        self.num_block_divmod = num_block_divmod
        self.num_head_divmod = num_head_divmod
        self.is_persistent = is_persistent
        # self.qhead_per_kvhead_packgqa = qhead_per_kvhead_packgqa
        self._loc = loc

    @staticmethod
    def create(
        num_block: Int32,
        num_head: Int32,
        num_batch: Int32,
        is_persistent: cutlass.Constexpr[bool] = False,
        *,
        loc=None,
        ip=None,
    ) -> "TileSchedulerParams":
        num_block_divmod = FastDivmod.create(num_block, loc=loc, ip=ip)
        num_head_divmod = FastDivmod.create(num_head, loc=loc, ip=ip)
        return TileSchedulerParams(
            num_block,
            num_head,
            num_batch,
            num_block_divmod,
            num_head_divmod,
            is_persistent,
            loc=loc,
            ip=ip,
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.num_block,
            self.num_head,
            self.num_batch,
            self.num_block_divmod,
            self.num_head_divmod,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.num_block,
                self.num_head,
                self.num_batch,
                self.num_block_divmod,
                self.num_head_divmod,
            ],
            self._values_pos,
        ):
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
        return params.num_block, params.num_head, params.num_batch

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
        num_block_divmod: FastDivmod,
        num_head_divmod: FastDivmod,
        total_blocks: Int32,
        tile_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.num_block_divmod = num_block_divmod
        self.num_head_divmod = num_head_divmod
        self.total_blocks = total_blocks
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def create(params: TileSchedulerParams, *, loc=None, ip=None) -> "SingleTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        total_blocks = params.num_block * params.num_head * params.num_batch
        return StaticPersistentTileScheduler(
            params.num_block_divmod, params.num_head_divmod, total_blocks, tile_idx, loc=loc, ip=ip
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
        total_blocks = params.num_block * params.num_head * params.num_batch
        return (cutlass.min(sm_count, total_blocks), Int32(1), Int32(1))

    # @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        hn_idx, block_idx = self.num_block_divmod.divmod(self._tile_idx)
        batch_idx, head_idx = self.num_head_divmod.divmod(hn_idx)
        is_valid = self._tile_idx < self.total_blocks
        # if cute.arch.thread_idx()[0] == 0:
        #     cute.printf("TileScheduler: tile_idx=%d, hn_idx=%d, block_idx=%d, batch_idx=%d, head_idx=%d, is_valid=%d", self._tile_idx, hn_idx, block_idx, batch_idx, head_idx, is_valid)
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
        for obj in [self.num_block_divmod, self.num_head_divmod, self.total_blocks, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.num_block_divmod, self.num_head_divmod, self.total_blocks, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)
