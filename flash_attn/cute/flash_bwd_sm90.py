import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from flash_attn.cute import hopper_helpers as sm90_utils
from flash_attn.cute import utils
from flash_attn.cute import copy_utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, ParamsBase
from flash_attn.cute.named_barrier import NamedBarrierFwd, NamedBarrierBwd


def mma_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
) -> cute.Tensor:
    acc = cute.make_fragment(tiled_mma.partition_shape_C(shape), Float32)
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sm90_utils.gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
    return acc


def mma_sm90(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sm90_utils.gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        tile_m: int = 64,
        tile_n: int = 128,
        num_stages: int = 2,
        num_threads: int = 384,
        Q_in_regs: bool = False,
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
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.dS_stage = 2
        self.Q_in_regs = Q_in_regs

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        Q_in_regs=False,
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
                ((self.tile_m, self.tile_hdim), self.num_stages),
                ((self.tile_n, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdimv), None),
                ((self.tile_m, self.tile_hdimv), self.num_stages),
                ((self.tile_m, self.tile_n), self.dS_stage),
            ]
        ]

        self.sdQaccum_layout = cute.make_layout(self.tile_m * self.tile_hdim)
        # dQaccum R->S
        self.r2s_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            Float32, self.num_mma_threads, num_copy_elems=128 // Float32.width
        )

    def _get_tiled_mma(self):
        # S = Q @ K.T, dP = dO @ V.T
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 2, 1),
            tiler_mn=(64, self.tile_n // 2),
        )
        # dV = P.T @ dO, dK = dS.T @ Q
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=(self.tile_n // 64, 1, 1),
                tiler_mn=(64, tile_hdim),
            )
            for tile_hdim in (self.tile_hdim, self.tile_hdimv)
        ]
        # dQ = dS @ K
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 2, 1),
            tiler_mn=(64, self.tile_hdim // 2),
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
        cosize_sP = cute.cosize(self.sPdS_layout)  # Could be zero
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.num_stages], 128
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.num_stages], 128
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_KV: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
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

        self.num_threads_per_warp_group = 128
        self.num_mma_warp_groups = self.num_mma_threads // self.num_threads_per_warp_group
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

        mbar_ptr_KV = storage.mbar_ptr_KV.data_ptr()

        # mbarrier init
        if warp_idx == 1:
            cute.arch.mbarrier_init(mbar_ptr_KV, 1)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // self.num_threads_per_warp_group
        )
        pipeline_q = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            init_wait=False,
        )
        pipeline_do = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            init_wait=True,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.num_stages),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.num_stages),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            False,
            False,
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
                    pipeline_q,
                    pipeline_do,
                    mbar_ptr_KV,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
            if warp_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.dQEmpty),
                    number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
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
                pipeline_q,
                pipeline_do,
                mbar_ptr_KV,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
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
        pipeline_q: cutlass.pipeline.PipelineAsync,
        pipeline_do: cutlass.pipeline.PipelineAsync,
        mbar_ptr_KV: cutlass.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
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
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_do)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_do)

                # TODO: need to wait if we do persistent kernel
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_ptr_KV, self.tma_copy_bytes["K"] + self.tma_copy_bytes["V"]
                    )
                load_K(tma_bar_ptr=mbar_ptr_KV)
                load_V(tma_bar_ptr=mbar_ptr_KV)

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
                for i in cutlass.range(m_block_max - m_block_min, unroll=2):
                    m_block = m_block_max - i - 1
                    pipeline_q.producer_acquire(producer_state)
                    load_Q(m_block, producer_state=producer_state)
                    # cp.async.bulk is using ptx, so we need to elect one thread to do it
                    with cute.arch.elect_one():
                        load_LSE(m_block, producer_state=producer_state)
                    pipeline_do.producer_acquire(producer_state)
                    load_dO(m_block, producer_state=producer_state)
                    with cute.arch.elect_one():
                        load_dPsum(m_block, producer_state=producer_state)
                    producer_state.advance()

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
        pipeline_q: cutlass.pipeline.PipelineAsync,
        pipeline_do: cutlass.pipeline.PipelineAsync,
        mbar_ptr_KV: cutlass.Pointer,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
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
        tSrQ = tiled_mma_SdP.make_fragment_A(wg_mma_SdP.partition_A(sQ))
        tSrK = tiled_mma_SdP.make_fragment_B(wg_mma_SdP.partition_B(sK))
        # dP = dO @ V.T
        tdPrdO = tiled_mma_SdP.make_fragment_A(wg_mma_SdP.partition_A(sdO))
        tdPrV = tiled_mma_SdP.make_fragment_B(wg_mma_SdP.partition_B(sV))
        # dV += P.T @ dO
        sPt = utils.transpose_view(sP)
        sdOt = utils.transpose_view(sdO)
        tdVrPt = tiled_mma_dV.make_fragment_A(wg_mma_dV.partition_A(sPt))
        tdVrdOt = tiled_mma_dV.make_fragment_B(wg_mma_dV.partition_B(sdOt))
        # dK += dS.T @ Q
        sdSt = utils.transpose_view(sdS)
        sQt = utils.transpose_view(sQ)
        tdKrdSt = tiled_mma_dK.make_fragment_A(wg_mma_dK.partition_A(sdSt))
        tdKrQt = tiled_mma_dK.make_fragment_B(wg_mma_dK.partition_B(sQt))
        # dQ = dS @ K
        sKt = utils.transpose_view(sK)
        tdQrdS = tiled_mma_dQ.make_fragment_A(wg_mma_dQ.partition_A(sdS))
        tdQrKt = tiled_mma_dQ.make_fragment_B(wg_mma_dQ.partition_B(sKt))

        # Smem copy atom tiling
        smem_copy_atom_PdS = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_PdS = cute.make_tiled_copy_C(smem_copy_atom_PdS, tiled_mma_SdP).get_slice(
            tidx
        )
        tPsP = smem_thr_copy_PdS.partition_D(sP)
        tdSsdS = smem_thr_copy_PdS.partition_D(sdS)

        sLSE_mma = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.num_stages),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_mma = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.num_stages),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        LSEslice = (None, 0, None)
        tLSEsLSE = utils.make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sLSE_mma))[LSEslice]
        tLSEsdPsum = utils.make_acc_tensor_mn_view(thr_mma_SdP.partition_C(sdPsum_mma))[LSEslice]

        smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        acc_dV = cute.make_fragment(
            tiled_mma_dV.partition_shape_C((self.tile_n, self.tile_hdimv)),
            Float32,
        )
        acc_dK = cute.make_fragment(
            tiled_mma_dK.partition_shape_C((self.tile_n, self.tile_hdim)),
            Float32,
        )

        mma_qk_fn = partial(mma_zero_init, tiled_mma_SdP, (self.tile_m, self.tile_n), tSrQ, tSrK)
        mma_dov_fn = partial(
            mma_zero_init, tiled_mma_SdP, (self.tile_m, self.tile_n), tdPrdO, tdPrV
        )
        mma_pdo_fn = partial(mma_sm90, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt)
        mma_dsq_fn = partial(mma_sm90, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt)
        mma_dsk_fn = partial(
            mma_zero_init, tiled_mma_dQ, (self.tile_m, self.tile_hdim), tdQrdS, tdQrKt
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            pipeline_q=pipeline_q,
            pipeline_do=pipeline_do,
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

        kv_consumer_phase = Int32(0)
        consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("tidx = {}, m_block_min = {}, m_block_max = {}", cute.arch.thread_idx()[0], m_block_min, m_block_max)

            cute.arch.mbarrier_wait(mbar_ptr_KV, phase=kv_consumer_phase)
            kv_consumer_phase ^= 1

            dKV_should_accumulate = False
            for m_tile in cutlass.range(m_block_max - m_block_min, unroll=1):
                m_block = m_block_max - 1 - m_tile
                consumer_state = mma_one_m_block_all(
                    m_block, consumer_state, dKV_should_accumulate=dKV_should_accumulate
                )
                dKV_should_accumulate = True

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
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        pipeline_q: cutlass.pipeline.PipelineAsync,
        pipeline_do: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tPsP: Optional[cute.Tensor],
        tdSsdS: Optional[cute.Tensor],
        tdQsdQaccum: cute.Tensor,
        smem_thr_copy_PdS: cute.TiledCopy,
        smem_thr_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        # acc_dV,
        # acc_dK,
        dKV_should_accumulate: Boolean = True,
    ):
        smem_idx = smem_pipe_read.index
        # (1) [GEMM 1] S = Q @ K^T
        pipeline_q.consumer_wait(smem_pipe_read, pipeline_q.consumer_try_wait(smem_pipe_read))
        acc_S = mma_qk_fn(A_idx=smem_idx, wg_wait=-1)
        # S2R for LSE
        tLSErLSE = cute.make_fragment_like(tLSEsLSE[None, 0])
        cute.autovec_copy(tLSEsLSE[None, smem_idx], tLSErLSE)
        # (2) [GEMM 2] dP = dO @ V.T
        pipeline_do.consumer_wait(smem_pipe_read, pipeline_do.consumer_try_wait(smem_pipe_read))
        acc_dP = mma_dov_fn(A_idx=smem_idx, wg_wait=1)
        # (3) [Pointwise 1] P = exp(S - LSE)
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_S_mn)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            acc_S_mn[r, None].store(
                cute.math.exp2(
                    acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r], fastmath=True
                )
            )
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_S_mn)
        # Convert P from f32 -> f16
        tdVrP_acc = cute.make_tensor(acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout))
        tdVrP = cute.make_fragment_like(tdVrP_acc, self.dtype)
        utils.cvt_f16(tdVrP_acc, tdVrP)
        # S2R for dPsum
        tLSErdPsum = cute.make_fragment_like(tLSEsdPsum[None, 0])
        cute.autovec_copy(tLSEsdPsum[None, smem_idx], tLSErdPsum)

        PdS_smem_idx = smem_idx if const_expr(self.dS_stage > 1) else 0
        # R2S for P
        tPrP = smem_thr_copy_PdS.retile(tdVrP)
        # sync to make sure P has already been used in the previous iteration before writing new vals
        if const_expr(self.dS_stage == 1):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads
            )
        cute.copy(smem_thr_copy_PdS, tPrP, tPsP[None, None, None, PdS_smem_idx])

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        warpgroup.wait_group(0)
        acc_dP_mn = utils.make_acc_tensor_mn_view(acc_dP)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dP_mn)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            acc_dP_mn[r, None].store(
                acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r])
            )
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dP_mn)
        # Convert dS from f32 -> f16
        tdKrdS_acc = cute.make_tensor(acc_dP.iterator, utils.convert_layout_acc_frgA(acc_dP.layout))
        tdKrdS = cute.make_fragment_like(tdKrdS_acc, self.dtype)
        utils.cvt_f16(tdKrdS_acc, tdKrdS)

        # If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
        # (2) dS is already read by the Mma in the previous iteration in case of Mma_dKV_is_RS.
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads
        )

        # R2S for dS
        tdSrdS = smem_thr_copy_PdS.retile(tdKrdS)
        cute.copy(smem_thr_copy_PdS, tdSrdS, tdSsdS[None, None, None, PdS_smem_idx])

        # (4) [GEMM 3] dV += P.T @ dO
        mma_pdo_fn(A_idx=PdS_smem_idx, B_idx=smem_idx, zero_init=not dKV_should_accumulate, wg_wait=-1)

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads
        )
        # (6) [GEMM 4] dQ = dS @ K
        acc_dQ = mma_dsk_fn(A_idx=PdS_smem_idx, wg_wait=1)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dV)
        pipeline_do.consumer_release(smem_pipe_read)  # release dO as dV mma is done

        # (7) [GEMM 5] dK += dS.T @ Q
        mma_dsq_fn(A_idx=PdS_smem_idx, B_idx=smem_idx, zero_init=not dKV_should_accumulate, wg_wait=1)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dQ)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQEmpty),
            number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
        )
        tdQrdQaccum_tmp = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
        cute.copy(smem_thr_copy_dQaccum, tdQrdQaccum_tmp, tdQsdQaccum)
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQFull),
            number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
        )

        warpgroup.wait_group(0)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(acc_dK)
        pipeline_q.consumer_release(smem_pipe_read)
        # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("tidx = {}, m_block = {}, after pipeline_q consumer release", cute.arch.thread_idx()[0], m_block)

        smem_pipe_read.advance()
        return smem_pipe_read

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
        rdK = cute.make_fragment_like(acc_dK, self.dtype)
        rdK.store(acc_dK.load().to(self.dtype))

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_mma_threads
        )

        smem_copy_atom_dKV = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_dK = cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma_dK).get_slice(tidx)
        smem_thr_copy_dV = cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma_dV).get_slice(tidx)

        # rmem -> smem
        taccdVrdV = smem_thr_copy_dV.retile(rdV)
        taccdVsdV = smem_thr_copy_dV.partition_D(sV)  # reuse sV SMEM
        cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)

        taccdKrdK = smem_thr_copy_dK.retile(rdK)
        taccdKsdK = smem_thr_copy_dK.partition_D(sK)  # reuse sK SMEM
        cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)

        # smem -> gmem
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
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            store_dV()
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
        tile_elems = cute.cosize(sdQaccum.layout)
        tile_bytes = Int32(tile_elems * 4)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            base_flat = cute.domain_offset((seqlen.offset_q * self.tile_hdim,), mdQaccum_cur)

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            for it_m in cutlass.range(m_block_max - m_block_min, unroll=1):
                m_block = m_block_max - 1 - it_m
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.dQFull),
                    number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
                )
                gdQaccum_block = cute.local_tile(base_flat, (tile_elems,), (m_block,))
                with cute.arch.elect_one():
                    sm90_utils.tma_reduce_add_bulk_f32(
                        sdQaccum.iterator, gdQaccum_block.iterator, tile_bytes
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.dQEmpty),
                    number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
