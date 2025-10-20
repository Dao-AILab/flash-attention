import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.arch import ProxyKind, SharedSpace
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from flash_attn.cute import hopper_helpers as sm90_utils
from flash_attn.cute import utils
from flash_attn.cute import copy_utils
from flash_attn.cute.hopper_helpers import gemm_zero_init, gemm_w_idx
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, ParamsBase
from flash_attn.cute.named_barrier import NamedBarrierFwd, NamedBarrierBwd


def mma_partition_fragment_AB(
    thr_mma: cute.core.ThrMma, sA: Optional[cute.Tensor], sB: Optional[cute.Tensor], swap_AB: bool
):
    if const_expr(not swap_AB):
        return (
            thr_mma.make_fragment_A(thr_mma.partition_A(sA)) if sA is not None else None,
            thr_mma.make_fragment_B(thr_mma.partition_B(sB)) if sB is not None else None,
        )
    else:
        return (
            thr_mma.make_fragment_B(thr_mma.partition_B(sA)) if sA is not None else None,
            thr_mma.make_fragment_A(thr_mma.partition_A(sB)) if sB is not None else None,
        )


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
    ):
        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = False
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage]
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_mma_warp_groups = (self.num_threads // 128) - 1
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_mma_warp_groups
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        # These are tuned for speed
        # Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share
        # them and then shuffle to get the value whenever we need? This can reduce register
        # pressure when SdP_swapAB, where each thread needs to keep statistics for (kBlockM / 4)
        # rows. If !SdP_swapAB, each thread only needs to keep statistics for 2 rows.
        # TODO: impl these for hdim 64
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        Q_stage,
        num_threads,
        V_in_regs=False,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        else:
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sdO_layout, self.sPdS_layout = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), self.Q_stage),
                ((self.tile_n, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdimv), None),
                ((self.tile_m, self.tile_hdimv), self.dO_stage),
                ((self.tile_m, self.tile_n), self.PdS_stage),
            ]
        ]
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_mma_warp_groups, self.num_mma_warp_groups)
        )
        # dQaccum R->S
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            # thr_layout
            cute.make_layout((self.num_threads_per_warp_group, self.num_mma_warp_groups)),
            cute.make_layout(128 // Float32.width),  # val_layout
        )

    def _get_tiled_mma(self):
        # S = Q @ K.T, dP = dO @ V.T
        atom_layout_SdP = (self.AtomLayoutMSdP, self.num_mma_warp_groups // self.AtomLayoutMSdP)
        tiler_mn_SdP = (self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1])
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(atom_layout_SdP if not self.SdP_swapAB else atom_layout_SdP[::-1])
            + (1,),
            tiler_mn=tiler_mn_SdP if not self.SdP_swapAB else tiler_mn_SdP[::-1],
        )
        # dV = P.T @ dO, dK = dS.T @ Q
        atom_layout_dKV = (self.AtomLayoutNdKV, self.num_mma_warp_groups // self.AtomLayoutNdKV)
        tiler_mn_dK = (self.tile_n // atom_layout_dKV[0], self.tile_hdim // atom_layout_dKV[1])
        tiler_mn_dV = (self.tile_n // atom_layout_dKV[0], self.tile_hdimv // atom_layout_dKV[1])
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=(atom_layout_dKV if not self.dKV_swapAB else atom_layout_dKV[::-1])
                + (1,),
                tiler_mn=tiler_mn_d if not self.dKV_swapAB else tiler_mn_d[::-1],
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        # dQ = dS @ K
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_mma_warp_groups // self.AtomLayoutMdQ)
        tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K if not self.dQ_swapAB else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN if not self.dQ_swapAB else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(atom_layout_dQ if not self.dQ_swapAB else atom_layout_dQ[::-1]) + (1,),
            tiler_mn=tiler_mn_dQ if not self.dQ_swapAB else tiler_mn_dQ[::-1],
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_alignment = sK_alignment = sV_alighment = sdQaccum_alignment = sdO_alignment = 1024

        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[type, cute.cosize(layout)], alignment]
            for (layout, type, alignment) in [
                (self.sQ_layout, self.dtype, sQ_alignment),
                (self.sK_layout, self.dtype, sK_alignment),
                (self.sV_layout, self.dtype, sV_alighment),
                (self.sdO_layout, self.dtype, sdO_alignment),
                (self.sdQaccum_layout, Float32, sdQaccum_alignment),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.Q_stage], 128
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
    ):
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        layout_transpose = [1, 3, 2, 0]  # (b, s, n, h) --> (s, h, n, b)
        mQ, mK, mV, mdK, mdV, mdO = [
            utils.select(t, layout_transpose) for t in (mQ, mK, mV, mdK, mdV, mdO)
        ]
        LSE_dPsum_dQaccum_transpose = [2, 1, 0]  # (b, n, s) -> (s, n, b)
        mLSE, mdPsum, mdQaccum = [
            utils.select(t, LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()

        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32

        self.num_mma_regs = 240
        self.num_producer_regs = 24
        # self.num_mma_regs = 232
        # self.num_producer_regs = 40

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_mma_warp_groups
        )

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )

        TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mK.shape[2]),
            cute.size(mK.shape[3]),
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=False,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK,
            tma_tensor_dV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // self.num_threads_per_warp_group
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            init_wait=False,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            init_wait=True,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=None,
            window_size_right=None,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
            if warp_idx == 1:
                for warp_group_idx in cutlass.range(self.num_mma_warp_groups):
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                    )
                self.dQaccum_store(mdQaccum, sdQaccum, block_info, TileSchedulerCls, SeqlenInfoCls)
        else:
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mK_cur = mK[None, None, head_idx, batch_idx]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                mV_cur = mV[None, None, head_idx, batch_idx]
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))

                mQ_cur = mQ[None, None, head_idx, batch_idx]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                mdO_cur = mdO[None, None, head_idx, batch_idx]
                gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                mdPsum_cur = mdPsum[None, head_idx, batch_idx]
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
                # First iteration: load K together w Q & LSE, then V together w dO & dPsum
                m_block = m_block_min
                pipeline_Q.producer_acquire(
                    producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"]
                )
                load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                load_Q(m_block, producer_state=producer_state_Q)
                # cp.async.bulk is using ptx, so we need to elect one thread to do it
                with cute.arch.elect_one():
                    load_LSE(m_block, producer_state=producer_state_Q)
                producer_state_dO_cur = (
                    producer_state_dO
                    if const_expr(self.Q_stage != self.dO_stage)
                    else producer_state_Q
                )
                pipeline_dO.producer_acquire(
                    producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"]
                )
                load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                load_dO(m_block, producer_state=producer_state_dO_cur)
                with cute.arch.elect_one():
                    load_dPsum(m_block, producer_state=producer_state_dO_cur)
                producer_state_Q.advance()
                producer_state_dO.advance()
                # Subsequent iterations: load Q & LSE, then dO & dPsum
                for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                    pipeline_Q.producer_acquire(producer_state_Q)
                    load_Q(m_block, producer_state=producer_state_Q)
                    # cp.async.bulk is using ptx, so we need to elect one thread to do it
                    with cute.arch.elect_one():
                        load_LSE(m_block, producer_state=producer_state_Q)
                    producer_state_dO_cur = (
                        producer_state_dO
                        if const_expr(self.Q_stage != self.dO_stage)
                        else producer_state_Q
                    )
                    pipeline_dO.producer_acquire(producer_state_dO_cur)
                    load_dO(m_block, producer_state=producer_state_dO_cur)
                    with cute.arch.elect_one():
                        load_dPsum(m_block, producer_state=producer_state_dO_cur)
                    producer_state_Q.advance()
                    producer_state_dO.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx))
        # S = Q @ K.T
        tSrQ, tSrK = mma_partition_fragment_AB(wg_mma_SdP, sQ, sK, self.SdP_swapAB)
        # dP = dO @ V.T
        tdPrdO, tdPrV = mma_partition_fragment_AB(wg_mma_SdP, sdO, sV, self.SdP_swapAB)
        # dV += P.T @ dO
        sPt = utils.transpose_view(sP) if sP is not None else None
        sdOt = utils.transpose_view(sdO)
        tdVrPt, tdVrdOt = mma_partition_fragment_AB(wg_mma_dV, sPt, sdOt, self.dKV_swapAB)
        # dK += dS.T @ Q
        sdSt = utils.transpose_view(sdS)
        sQt = utils.transpose_view(sQ)
        tdKrdSt, tdKrQt = mma_partition_fragment_AB(wg_mma_dK, sdSt, sQt, self.dKV_swapAB)
        # dQ = dS @ K
        sKt = utils.transpose_view(sK)
        tdQrdS, tdQrKt = mma_partition_fragment_AB(wg_mma_dQ, sdS, sKt, self.dQ_swapAB)

        # Smem copy atom tiling
        smem_copy_atom_PdS = utils.get_smem_store_atom(
            self.arch, self.dtype, transpose=self.SdP_swapAB
        )
        smem_thr_copy_PdS = cute.make_tiled_copy_C(smem_copy_atom_PdS, tiled_mma_SdP).get_slice(
            tidx
        )
        tPsP = None
        if const_expr(sP is not None):
            tPsP = smem_thr_copy_PdS.partition_D(sP if const_expr(not self.SdP_swapAB) else sPt)
        tdSsdS = smem_thr_copy_PdS.partition_D(sdS if const_expr(not self.SdP_swapAB) else sdSt)

        sLSE_mma = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_mma = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        if const_expr(self.SdP_swapAB):
            sLSE_mma = utils.transpose_view(sLSE_mma)
            sdPsum_mma = utils.transpose_view(sdPsum_mma)
        LSEslice = (None, 0, None) if const_expr(not self.SdP_swapAB) else (0, None, None)
        tLSEsLSE = utils.make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sLSE_mma))[LSEslice]
        tLSEsdPsum = utils.make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sdPsum_mma))[LSEslice]

        smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        dV_shape = (self.tile_n, self.tile_hdimv)
        acc_dV = cute.make_fragment(
            tiled_mma_dV.partition_shape_C(dV_shape if not self.dKV_swapAB else dV_shape[::-1]),
            Float32,
        )
        dK_shape = (self.tile_n, self.tile_hdim)
        acc_dK = cute.make_fragment(
            tiled_mma_dK.partition_shape_C(dK_shape if not self.dKV_swapAB else dK_shape[::-1]),
            Float32,
        )

        mma_qk_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
            swap_AB=self.SdP_swapAB,
        )
        mma_dov_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            (self.tile_m, self.tile_n),
            tdPrdO,
            tdPrV,
            swap_AB=self.SdP_swapAB,
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(
                gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=self.dKV_swapAB
            )
            mma_dsq_fn = partial(
                gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=self.dKV_swapAB
            )
        else:
            assert not self.dKV_swapAB
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        mma_dsk_fn = partial(
            gemm_zero_init,
            tiled_mma_dQ,
            (self.tile_m, self.tile_hdim),
            tdQrdS,
            tdQrKt,
            swap_AB=self.dQ_swapAB,
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tPsP=tPsP,
            tdSsdS=tdSsdS,
            tdQsdQaccum=tdQsdQaccum,
            smem_thr_copy_PdS=smem_thr_copy_PdS,
            smem_thr_copy_dQaccum=smem_thr_copy_dQaccum,
            softmax_scale_log2=softmax_scale_log2,
            # acc_dV=acc_dV,
            # acc_dK=acc_dK,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask,
                n_block=n_block,
                thr_mma=thr_mma_SdP,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("tidx = {}, m_block_min = {}, m_block_max = {}", cute.arch.thread_idx()[0], m_block_min, m_block_max)
            dKV_accumulate = False
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                    m_block,
                    consumer_state_Q,
                    consumer_state_dO,
                    mask_fn=mask_fn,
                    dKV_accumulate=dKV_accumulate,
                )
                dKV_accumulate = True

            # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dV)
            # scale dK
            acc_dK.store(acc_dK.load() * softmax_scale)
            self.epilogue_dKV(
                acc_dV,
                mdV,
                sV,
                acc_dK,
                mdK,
                sK,
                seqlen,
                tma_atom_dK,
                tma_atom_dV,
                tiled_mma_dK,
                tiled_mma_dV,
                tidx,
                n_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tPsP: Optional[cute.Tensor],
        tdSsdS: Optional[cute.Tensor],
        tdQsdQaccum: cute.Tensor,
        smem_thr_copy_PdS: cute.TiledCopy,
        smem_thr_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        mask_fn: Optional[Callable] = None,
        # acc_dV,
        # acc_dK,
        dKV_accumulate: Boolean = True,
    ):
        consumer_state_dO_cur = (
            consumer_state_dO if const_expr(self.Q_stage == self.dO_stage) else consumer_state_Q
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        # (1) [GEMM 1] S = Q @ K^T
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        # (2) [GEMM 2] dP = dO @ V.T
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)
        # (3) [Pointwise 1] P = exp(S - LSE)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S, transpose=self.SdP_swapAB)
        # if cute.arch.thread_idx()[0] == 256: cute.print_tensor(acc_S_mn)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - tLSErLSE[r], fastmath=True
                )
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_S_mn)
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])

        # Convert P from f32 -> f16
        tdVrP = utils.cvt_f16(utils.make_acc_tensor_frgA_view(acc_S), self.dtype)
        # R2S for P
        if const_expr(not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads
                )
            tPrP = smem_thr_copy_PdS.retile(tdVrP)
            cute.copy(smem_thr_copy_PdS, tPrP, tPsP[None, None, None, smem_idx_PdS])

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        warpgroup.wait_group(0)
        acc_dP_mn = utils.make_acc_tensor_mn_view(acc_dP, transpose=self.SdP_swapAB)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dP_mn)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - tLSErdPsum[r])
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dP_mn)
        # Convert dS from f32 -> f16
        tdKrdS = utils.cvt_f16(utils.make_acc_tensor_frgA_view(acc_dP), self.dtype)

        # If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure (1) P is written in case of !mma_dkv_is_rs and
        # (2) dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)):
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads
            )

        # R2S for dS
        tdSrdS = smem_thr_copy_PdS.retile(tdKrdS)
        cute.copy(smem_thr_copy_PdS, tdSrdS, tdSsdS[None, None, None, smem_idx_PdS])

        # (5) [GEMM 3] dV += P.T @ dO
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )
        else:
            mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads
        )
        # (6) [GEMM 4] dQ = dS @ K
        acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dV)
        pipeline_dO.consumer_release(consumer_state_dO_cur)  # release dO as dV mma is done

        # (7) [GEMM 5] dK += dS.T @ Q
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
            )
        else:
            mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dQ)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        tdQrdQaccum_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
        cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )

        warpgroup.wait_group(0)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dK)
        pipeline_Q.consumer_release(consumer_state_Q)
        # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("tidx = {}, m_block = {}, after pipeline_Q consumer release", cute.arch.thread_idx()[0], m_block)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        rdV = cute.make_fragment_like(acc_dV, self.dtype)
        rdV.store(acc_dV.load().to(self.dtype))
        rdK = utils.cvt_f16(acc_dK, self.dtype)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_mma_threads
        )

        smem_copy_atom_dKV = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=self.dKV_swapAB, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_dK = cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma_dK).get_slice(tidx)
        smem_thr_copy_dV = cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma_dV).get_slice(tidx)
        mdV_cur = mdV[None, None, head_idx, batch_idx]
        mdK_cur = mdK[None, None, head_idx, batch_idx]
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
        store_dK, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
        )
        store_dV, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
        )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # rmem -> smem
        taccdVrdV = smem_thr_copy_dV.retile(rdV)
        sdV = sV if const_expr(not self.dKV_swapAB) else utils.transpose_view(sV)  # reuse sV SMEM
        taccdVsdV = smem_thr_copy_dV.partition_D(sdV)
        cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)
        # ensure smem writes are visible to TMA
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_mma_threads
        )
        if warp_idx == 4:
            store_dV()
        taccdKrdK = smem_thr_copy_dK.retile(rdK)
        sdK = sK if const_expr(not self.dKV_swapAB) else utils.transpose_view(sK)  # reuse sK SMEM
        taccdKsdK = smem_thr_copy_dK.partition_D(sdK)  # reuse sK SMEM
        cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)
        # ensure smem writes are visible to TMA
        cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_mma_threads
        )
        # smem -> gmem
        if warp_idx == 4:
            store_dK()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
    ):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / WG, WG, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.num_mma_warp_groups,)
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                for warp_group_idx in cutlass.range_constexpr(self.num_mma_warp_groups):
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                    )
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdQaccum[None, warp_group_idx].iterator,
                            gdQaccum[None, warp_group_idx, m_block].iterator,
                            self.tma_copy_bytes["dQ"],
                        )
                    cute.arch.cp_async_bulk_commit_group()
                for warp_group_idx in cutlass.range_constexpr(self.num_mma_warp_groups):
                    cute.arch.cp_async_bulk_wait_group(
                        self.num_mma_warp_groups - 1 - warp_group_idx, read=True
                    )
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                    )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
