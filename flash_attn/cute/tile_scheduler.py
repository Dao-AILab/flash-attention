# Copyright (c) 2025, Tri Dao.

from typing import Optional, Tuple
from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from flash_attn.cute.fast_math import FastDivmod, clz


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(args.num_block, args.num_head, args.num_batch)

    def __init__(self, blk_coord: cute.Coord, *, loc=None, ip=None):
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        blk_coord = cute.arch.block_idx()
        return SingleTileScheduler(blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
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
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        total_blocks: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            total_blocks = args.num_block * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmod.create(args.num_block), FastDivmod.create(args.num_head), total_blocks
            )

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
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "StaticPersistentTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return StaticPersistentTileScheduler(
            params.num_block_divmod,
            params.num_head_divmod,
            params.total_blocks,
            tile_idx,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (cutlass.min(sm_count, params.total_blocks), Int32(1), Int32(1))

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


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        l2_minor_divmod: FastDivmod
        l2_major_divmod: FastDivmod
        l2_minor_residual_divmod: FastDivmod
        num_hb_quotient: Int32

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTScheduler.Params":
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            log2_floor = lambda n: 31 - clz(n)
            # swizzle is how many heads can fit in L2
            # Seems faster if swizzle if a power of 2
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block_divmod=FastDivmod.create(args.num_block),
                num_head_divmod=FastDivmod.create(args.num_head),
                l2_minor_divmod=FastDivmod.create(swizzle),
                l2_major_divmod=FastDivmod.create(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmod.create(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
            )

    def __init__(
        self,
        total_blocks: Int32,
        num_block_divmod: FastDivmod,
        num_head_divmod: FastDivmod,
        l2_minor_divmod: FastDivmod,
        l2_major_divmod: FastDivmod,
        l2_minor_residual_divmod: FastDivmod,
        num_hb_quotient: Int32,
        tile_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.total_blocks = total_blocks
        self.num_block_divmod = num_block_divmod
        self.num_head_divmod = num_head_divmod
        self.l2_minor_divmod = l2_minor_divmod
        self.l2_major_divmod = l2_major_divmod
        self.l2_minor_residual_divmod = l2_minor_residual_divmod
        self.num_hb_quotient = num_hb_quotient
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileLPTScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTScheduler(
            params.total_blocks,
            params.num_block_divmod,
            params.num_head_divmod,
            params.l2_minor_divmod,
            params.l2_major_divmod,
            params.l2_minor_residual_divmod,
            params.num_hb_quotient,
            tile_idx,
            loc=loc,
            ip=ip,
        )

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
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = self.l2_major_divmod.divmod(self._tile_idx)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < self.num_hb_quotient:
            block, bidhb_residual = self.l2_minor_divmod.divmod(l2_mod)
        else:
            block, bidhb_residual = self.l2_minor_residual_divmod.divmod(l2_mod)
        bidhb_actual = bidhb * self.l2_minor_divmod.divisor + bidhb_residual
        batch_idx, head_idx = self.num_head_divmod.divmod(bidhb_actual)
        # Longest-processing-time-first
        block = self.num_block_divmod.divisor - 1 - block
        is_valid = self._tile_idx < self.total_blocks
        return cutlass.utils.WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.total_blocks

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.total_blocks,
            self.num_block_divmod,
            self.num_head_divmod,
            self.l2_minor_divmod,
            self.l2_major_divmod,
            self.l2_minor_residual_divmod,
            self.num_hb_quotient,
            self._tile_idx,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.total_blocks,
                self.num_block_divmod,
                self.num_head_divmod,
                self.l2_minor_divmod,
                self.l2_major_divmod,
                self.l2_minor_residual_divmod,
                self.num_hb_quotient,
                self._tile_idx,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileLPTScheduler(*(tuple(obj_list)), loc=self._loc)
