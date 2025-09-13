import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warpgroup
#import cutlass.pipeline
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import const_expr

from flash_attn.cute import hopper_helpers as sm90_utils
from flash_attn.cute import utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, ParamsBase
from flash_attn.cute.named_barrier import NamedBarrierFwd, NamedBarrierBwd

class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_stages: int = 2,
        num_threads: int = 384,
        Q_in_regs: bool = False,
    ):

        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.Q_in_regs = Q_in_regs

    @staticmethod
    def can_implement(
        dtype, head_dim, head_dim_v, m_block_size, n_block_size, num_stages, num_threads,
        Q_in_regs=False
    ) -> bool:

        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if n_block_size % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False

        if (m_block_size * 2) % num_threads != 0:
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
        if const_expr(mLSE_type not in [cutlass.Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [cutlass.Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [cutlass.Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        else:
            if const_expr(not (mdK_type == mdV_type == cutlass.Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        assert mQ_type == self.dtype

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.head_dim_padded
            ),
            self.dtype
        )
        sK_layout_atom = sQ_layout_atom

        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.head_dim_v_padded
            ),
            self.dtype
        )
        sPdS_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.n_block_size
            ),
            self.dtype
        )
        sdO_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.dtype,
                self.head_dim_padded
            ),
            self.dtype
        )

        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sPdS_layout_atom, sdO_layout_atom


    def _setup_attributes(self):
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sPdS_layout_atom, sdO_layout_atom  = self._get_smem_layout_atom()

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width

        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        self.sQ_layout =   cute.tile_to_shape(sQ_layout_atom, (self.m_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2),)
        self.sK_layout =   cute.tile_to_shape(sK_layout_atom, (self.n_block_size, self.head_dim_padded),     (0, 1),)
        self.sV_layout =   cute.tile_to_shape(sV_layout_atom, (self.n_block_size, self.head_dim_v_padded),   (0, 1),)
        self.sdO_layout =  cute.tile_to_shape(sdO_layout_atom, (self.m_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2),)

        self.sPdS_layout = cute.tile_to_shape(sPdS_layout_atom, (self.m_block_size, self.n_block_size), (0, 1),)
        self.sdQaccum_layout = cute.make_layout(shape=(self.m_block_size * self.head_dim_padded, ),)


        # dQaccum R->S
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
                    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32,  num_bits_per_copy=universal_copy_bits),
                    cute.make_layout(self.num_mma_threads),
                    cute.make_layout(universal_copy_bits // cutlass.Float32.width)
        )

        # dV: S->G
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tdV_layout = cute.make_ordered_layout(
            (self.num_mma_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        self.gmem_tiled_copy_dV = cute.make_tiled_copy_tv(
                                    atom_universal_copy,
                                    tdV_layout,
                                    cute.make_layout((1, async_copy_elems))
        )

        # dK: S->G
        tK_shape_dim_1 = sK_layout_atom.outer.shape[1] // async_copy_elems
        tdK_layout = cute.make_ordered_layout(
            (self.num_mma_threads // tK_shape_dim_1, tK_shape_dim_1),
            order=(1, 0),
        )
        self.gmem_tiled_copy_dK = cute.make_tiled_copy_tv(
                                    atom_universal_copy,
                                    tdK_layout,
                                    cute.make_layout((1, async_copy_elems))
        )

    def _get_tiled_mma(self):

        # C = A @ B.T
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            cutlass.Float32,
            atom_layout_mnk=(self.m_block_size // 64, 1, 1),
            tiler_mn=(64, self.n_block_size),
        )
        # C = A.T @ B
        tiled_mma_dKV = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN,
            cutlass.Float32,
            atom_layout_mnk=(self.n_block_size // 64 , 1, 1),
            tiler_mn=(64, self.head_dim_padded),
        )
        # C = A @ B
        tiled_mma_dQaccum = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            cutlass.Float32,
            atom_layout_mnk=(self.m_block_size // 64, 1, 1),
            tiler_mn=(64, self.head_dim_padded),
        )

        return tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQaccum


    def _get_shared_storage_cls(self):
        sQ_alignment = sK_alignment = sV_alighment = sdQaccum_alignment = sdO_alignment = 128

        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[type, cute.cosize(layout)], alignment]
            for (layout, type, alignment) in [
                (self.sQ_layout,       self.dtype,      sQ_alignment),
                (self.sK_layout,       self.dtype,      sK_alignment),
                (self.sV_layout,       self.dtype,      sV_alighment),
                (self.sdO_layout,      self.dtype,      sdO_alignment),
                (self.sdQaccum_layout, cutlass.Float32, sdQaccum_alignment)
            ]
        ]

        cosize_sPdS   = cute.cosize(self.sPdS_layout)
        sPdS_struct   = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sPdS], 1024]
        sLSE_struct   = cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.m_block_size * self.num_stages], 128]
        sdPsum_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.m_block_size * self.num_stages], 128]

        mbar_ptr_Q_struct     = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_LSE_struct   = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_dPsum_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_dO_struct    = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        mbar_ptr_K_struct   = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_V_struct   = cute.struct.MemRange[cutlass.Int64, 2]


        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q:     mbar_ptr_Q_struct
            mbar_ptr_K:     mbar_ptr_K_struct
            mbar_ptr_V:     mbar_ptr_V_struct
            mbar_ptr_lse:   mbar_ptr_LSE_struct
            mbar_ptr_dpsum: mbar_ptr_dPsum_struct
            mbar_ptr_dO:    mbar_ptr_dO_struct

            sQ:       sQ_struct
            sV:       sV_struct
            sK:       sK_struct
            sPdS:     sPdS_struct
            sLSE:     sLSE_struct
            sdPsum:   sdPsum_struct
            sdO:      sdO_struct
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,

        mdO:  cute.Tensor,
        mLSE: cute.Tensor,

        mdPsum:   cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK:      cute.Tensor,
        mdV:      cute.Tensor,

        softmax_scale: cutlass.Float32,
        stream:        cuda.CUstream,

        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ:   Optional[cute.Tensor] = None,
        mSeqUsedK:   Optional[cute.Tensor] = None,

        softcap:           cutlass.Float32 | float | None = None,
        window_size_left:  cutlass.Int32 | int | None = None,
        window_size_right: cutlass.Int32 | int | None = None,
    ):

        self._check_type(
            *(t.element_type if t is not None else None
              for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV))
        )

        layout_transpose = [1, 3, 2, 0] # (b, s, n, h) --> (s, h, n, b)
        mQ, mK, mV, mdK, mdV, mdO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=layout_transpose))
            for t in (mQ, mK, mV, mdK, mdV, mdO)
        ]

        LSE_dPsum_dQaccum_transpose = [2, 1, 0] # (b, n, s) -> (s, n, b)
        mLSE, mdPsum, mdQaccum = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=LSE_dPsum_dQaccum_transpose))
            for t in (mLSE, mdPsum, mdQaccum)
        ]


        tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dQaccum = self._get_tiled_mma()

        self.tiled_mma_SdP      = tiled_mma_SdP
        self.tiled_mma_dKV      = tiled_mma_dKV
        self.tiled_mma_sdQaccum = tiled_mma_dQaccum

        self.num_mma_threads = tiled_mma_SdP.size

        self.num_threads_per_warp_group = 128
        self.num_mma_warp_groups = self.num_mma_threads // self.num_threads_per_warp_group
        self.num_producer_threads = 32

        self.num_mma_regs = 240
        self.num_producer_regs = 24

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()


        self.tma_copy_q_bytes = cute.size_in_bytes(mQ.element_type, cute.select(self.sQ_layout, mode=[0, 1]))
        self.tma_copy_k_bytes = cute.size_in_bytes(mK.element_type, cute.select(self.sK_layout, mode=[0, 1]))
        self.tma_copy_v_bytes = cute.size_in_bytes(mV.element_type, cute.select(self.sK_layout, mode=[0, 1]))

        self.tma_copy_do_bytes    =  cute.size_in_bytes(mdO.element_type, cute.select(self.sdO_layout, mode=[0,1]))
        self.tma_copy_lse_bytes   =  self.m_block_size * 4
        self.tma_copy_dPsum_bytes =  self.m_block_size * 4


        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.m_block_size, self.head_dim_padded),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
            1
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0,1]),
            (self.n_block_size, self.head_dim_v_padded),
            1
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0,1]),
            (self.m_block_size, self.head_dim_padded)
        )
        tma_atom_LSE, tma_tensor_LSE = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mLSE,
            cute.make_layout(self.m_block_size), (self.m_block_size,),
        )
        tma_atom_dPsum, tma_tensor_dPsum = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdPsum,
            cute.make_layout(self.m_block_size), (self.m_block_size, ),
        )
        TileScheduler = SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.n_block_size),
            cute.size(mK.shape[2]),
            cute.size(mK.shape[3]),
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.m_block_size, self.n_block_size),
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa= 1,
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
            tma_tensor_LSE,
            tma_tensor_dPsum,
            tma_tensor_dO,

            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_LSE,
            tma_atom_dPsum,
            tma_atom_dO,

            mdK,
            mdV,
            mdQaccum,

            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,

            self.gmem_tiled_copy_dV,
            self.gmem_tiled_copy_dK,
            self.r2s_tiled_copy_dQaccum,

            tiled_mma_SdP,
            tiled_mma_dKV,
            tiled_mma_dQaccum,

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
        mQ:     cute.Tensor,
        mK:     cute.Tensor,
        mV:     cute.Tensor,
        mLSE:   cute.Tensor,
        mdPsum: cute.Tensor,
        mdO:    cute.Tensor,

        tma_atom_Q:     Optional[cute.CopyAtom],
        tma_atom_K:     Optional[cute.CopyAtom],
        tma_atom_V:     Optional[cute.CopyAtom],
        tma_atom_LSE:   Optional[cute.CopyAtom],
        tma_atom_dPsum: Optional[cute.CopyAtom],
        tma_atom_dO:    Optional[cute.CopyAtom],

        mdK:      cute.Tensor,
        mdV:      cute.Tensor,
        mdQaccum: cute.Tensor,

        sQ_layout:       cute.ComposedLayout,
        sK_layout:       cute.ComposedLayout,
        sV_layout:       cute.ComposedLayout,
        sPdS_layout:     cute.ComposedLayout,
        sdO_layout:      cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,

        gmem_tiled_copy_dV:      cute.TiledCopy,
        gmem_tiled_copy_dK:      cute.TiledCopy,
        r2s_tiled_copy_dQaccum:  cute.TiledCopy,

        tiled_mma_SdP:     cute.TiledMma,
        tiled_mma_dKV:     cute.TiledMma,
        tiled_mma_dQaccum: cute.TiledMma,

        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler:     cutlass.Constexpr[Callable],
        SharedStorage:     cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx     = cute.arch.thread_idx()[0]

        # prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_LSE)
            cpasync.prefetch_descriptor(tma_atom_dPsum)
            cpasync.prefetch_descriptor(tma_atom_dO)


        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_ptr_K = storage.mbar_ptr_K.data_ptr()
        mbar_ptr_V = storage.mbar_ptr_V.data_ptr()

        # mbarrier init
        if warp_idx == 1:
            cute.arch.mbarrier_init(mbar_ptr_K, 1)
            cute.arch.mbarrier_init(mbar_ptr_V, 1)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, self.num_mma_threads // self.num_threads_per_warp_group)

        pipeline_q = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_q_bytes,
            init_wait=False,
        )
        pipeline_lse = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_lse.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_lse_bytes,
            init_wait=False,
        )
        pipeline_dpsum = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_dpsum.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_dPsum_bytes,
            init_wait=False,
        )
        pipeline_do = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_do_bytes,
            init_wait=False,
        )
        sQ  = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQt = utils.transpose_view(sQ)

        sK  = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV  = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)

        sLSE_load = storage.sLSE.get_tensor(cute.make_layout(
                                    (self.m_block_size, self.num_stages),
                                    stride=(1, cute.round_up(self.m_block_size, 64))
        ))
        sLSE_mma = storage.sLSE.get_tensor(cute.make_layout(
                                    (self.m_block_size, self.n_block_size, self.num_stages),
                                    stride=(1, 0, cute.round_up(self.m_block_size, 64))
        ))
        sdPsum_load = storage.sdPsum.get_tensor(cute.make_layout(
                                    (self.m_block_size, self.num_stages),
                                    stride=(1, cute.round_up(self.m_block_size, 64))
        ))
        sdPsum_mma = storage.sdPsum.get_tensor(cute.make_layout(
                                    (self.m_block_size, self.n_block_size, self.num_stages),
                                    stride=(1, 0, cute.round_up(self.m_block_size, 64))
        ))

        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)



        sP = storage.sPdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sPt = utils.transpose_view(sP)

        sdS = storage.sPdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdSt = utils.transpose_view(sdS)

        sdO = storage.sdO.get_tensor(sdO_layout.outer,  swizzle=sdO_layout.inner)
        sdOt = utils.transpose_view(sdO)


        block_info = BlockInfo(self.m_block_size, self.n_block_size, False, False,None, None, qhead_per_kvhead_packgqa=1,)
        SeqlenInfoCls = partial(
            SeqlenInfoQK, seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None, mCuSeqlensK=None,
            mSeqUsedQ=None, mSeqUsedK=None
        )

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if  warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            if warp_idx  == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mLSE,
                    mdPsum,
                    mdO,

                    sQ,
                    sK,
                    sV,
                    sLSE_load,
                    sdPsum_load,
                    sdO,

                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_LSE,
                    tma_atom_dPsum,
                    tma_atom_dO,

                    pipeline_q,
                    pipeline_lse,
                    pipeline_dpsum,
                    pipeline_do,

                    mbar_ptr_K,
                    mbar_ptr_V,

                    SeqlenInfoCls,
                    TileSchedulerCls,
                )
            if warp_idx == 1:
                cute.arch.barrier_arrive(barrier_id=int(NamedBarrierBwd.dQEmpty), number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE)
                self.dQaccum_writer(
                    mdQaccum,
                    sdQaccum,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                )
        else:
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx  - 128

            self.mma(
                tiled_mma_SdP,
                tiled_mma_dKV,
                tiled_mma_dQaccum,

                mdK,
                mdV,
                mdQaccum,

                sQ,
                sQt,
                sK,
                sV,

                sP,
                sPt,

                sdS,
                sdSt,

                sdO,
                sdOt,

                sLSE_mma,
                sdPsum_mma,

                sdQaccum,

                pipeline_q,
                pipeline_lse,
                pipeline_dpsum,
                pipeline_do,

                mbar_ptr_K,
                mbar_ptr_V,
                tidx,
                gmem_tiled_copy_dV,
                gmem_tiled_copy_dK,
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
        mQ:     cute.Tensor,
        mK:     cute.Tensor,
        mV:     cute.Tensor,
        mLSE:   cute.Tensor,
        mdPsum: cute.Tensor,
        mdO:    cute.Tensor,

        sQ:     cute.Tensor,
        sK:     cute.Tensor,
        sV:     cute.Tensor,
        sLSE:   cute.Tensor,
        sdPsum: cute.Tensor,
        sdO:    cute.Tensor,

        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,

        tma_atom_LSE:   cute.CopyAtom,
        tma_atom_dPsum: cute.CopyAtom,
        tma_atom_dO:    cute.CopyAtom,

        pipeline_q:     cutlass.pipeline.PipelineAsync,
        pipeline_lse:   cutlass.pipeline.PipelineAsync,
        pipeline_dpsum: cutlass.pipeline.PipelineAsync,
        pipeline_dO:    cutlass.pipeline.PipelineAsync,

        mbar_ptr_K: cutlass.Pointer,
        mbar_ptr_V: cutlass.Pointer,

        SeqlenInfoCls:    Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.num_stages)


            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()


            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)

                mK_cur = mK[None, None, head_idx, batch_idx]
                gK =     cute.local_tile(mK_cur, (self.n_block_size, self.head_dim_padded), (n_block, 0))

                mV_cur = mV[None, None, head_idx, batch_idx]
                gV =     cute.local_tile(mV_cur, (self.n_block_size, self.head_dim_padded), (n_block, 0))

                mQ_cur = mQ[None, None, head_idx, batch_idx]
                gQ =     cute.local_tile(mQ_cur, (self.m_block_size, self.head_dim_padded), (None, 0))

                mLSE_cur = mLSE[None, head_idx, batch_idx]
                gLSE =     cute.local_tile(mLSE_cur, (self.m_block_size,), (None,))

                mdPsum_cur = mdPsum[None, head_idx, batch_idx]
                gdPsum =     cute.local_tile(mdPsum_cur, (self.m_block_size,), (None,))

                mdO_cur = mdO[None, None, head_idx, batch_idx]
                gdO =     cute.local_tile(mdO_cur, (self.m_block_size, self.head_dim_padded), (None, 0))

                tQsQ, tQgQ = cpasync.tma_partition(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 2),
                    cute.group_modes(gQ, 0, 2),
                )
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 2),
                    cute.group_modes(gK, 0, 2),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 2),
                    cute.group_modes(gV, 0, 2),
                )
                tLSEsLSE, tLSEgLSE = cpasync.tma_partition(
                    tma_atom_LSE,
                    0,
                    cute.make_layout(1),
                    sLSE,
                    gLSE,
                )
                tdPsumsdPsum, tdPsumgdPsum = cpasync.tma_partition(
                    tma_atom_dPsum,
                    0,
                    cute.make_layout(1),
                    sdPsum,
                    gdPsum,
                )
                tdOsdO, tdOgdO = cpasync.tma_partition(
                    tma_atom_dO,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sdO, 0, 2),
                    cute.group_modes(gdO, 0, 2),
                )

                load_Q     =  partial(self.load_m_tile,     tma_atom_Q,     tQgQ, tQsQ, pipeline_q)
                load_LSE   =  partial(self.load_m_tile,     tma_atom_LSE,   tLSEgLSE, tLSEsLSE, pipeline_lse)
                load_dPsum =  partial(self.load_m_tile,     tma_atom_dPsum, tdPsumgdPsum, tdPsumsdPsum, pipeline_dpsum)
                load_dO    =  partial(self.load_m_tile,     tma_atom_dO,    tdOgdO, tdOsdO, pipeline_dO)

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_K, self.tma_copy_k_bytes)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_V, self.tma_copy_v_bytes)

                cute.copy(tma_atom_K, tKgK, tKsK, tma_bar_ptr=mbar_ptr_K)
                cute.copy(tma_atom_V, tVgV, tVsV, tma_bar_ptr=mbar_ptr_V)

                m_block_min, m_block_max = 0, cute.ceil_div(seqlen.seqlen_q, self.m_block_size)

                for i in cutlass.range(m_block_max - m_block_min, unroll=2):
                    m_block = m_block_max - i - 1

                    load_Q(m_block,     producer_state=producer_state)
                    load_LSE(m_block,   producer_state=producer_state)
                    load_dPsum(m_block, producer_state=producer_state)
                    load_dO(m_block,    producer_state=producer_state)

                    producer_state.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()


    @cute.jit
    def mma(
        self,
        tiled_mma_SdP:      cute.TiledMma,
        tiled_mma_dKV:      cute.TiledMma,
        tiled_mma_dQaccum:  cute.TiledMma,

        mdK:      cute.Tensor,
        mdV:      cute.Tensor,
        mdQaccum: cute.Tensor,

        sQ:   cute.Tensor,
        sQt:  cute.Tensor,
        sK:   cute.Tensor,
        sV:   cute.Tensor,

        sP:   cute.Tensor,
        sPt:  cute.Tensor,

        sdS:  cute.Tensor,
        sdSt: cute.Tensor,

        sdO:  cute.Tensor,
        sdOt: cute.Tensor,

        sLSE_mma:   cute.Tensor,
        sdPsum_mma: cute.Tensor,

        sdQaccum:   cute.Tensor,

        pipeline_q:     cutlass.pipeline.PipelineAsync,
        pipeline_lse:   cutlass.pipeline.PipelineAsync,
        pipeline_dPsum: cutlass.pipeline.PipelineAsync,
        pipeline_dO:    cutlass.pipeline.PipelineAsync,

        mbar_ptr_K: cutlass.Pointer,
        mbar_ptr_V: cutlass.Pointer,

        tidx: cutlass.Int32,
        gmem_tiled_copy_dV:      cute.TiledCopy,
        gmem_tiled_copy_dK:      cute.TiledCopy,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,

        softmax_scale_log2: cutlass.Float32,
        softmax_scale:      cutlass.Float32,

        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(self.num_mma_warp_groups, stride=self.num_threads_per_warp_group)

        wg_mma_SdP =     tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dKV =     tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQaccum = tiled_mma_dQaccum.get_slice(warp_group_thread_layout(warp_group_idx))

        smem_copy_atom_PdS = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_PdS  = cute.make_tiled_copy_C(smem_copy_atom_PdS, tiled_mma_SdP).get_slice(tidx)

        # S = Q @ K.T
        tSrQ  =  tiled_mma_SdP.make_fragment_A(wg_mma_SdP.partition_A(sQ))
        tSrK  =  tiled_mma_SdP.make_fragment_B(wg_mma_SdP.partition_B(sK))

        # dP = dO @ V.T
        tdPrdO = tiled_mma_SdP.make_fragment_A(wg_mma_SdP.partition_A(sdO))
        tdPrV  = tiled_mma_SdP.make_fragment_B(wg_mma_SdP.partition_B(sV))

        # P = exp(S-LSE)
        tPsP = smem_thr_copy_PdS.partition_D(sP)

        LSEslice = (None, 0, None)
        tLSEsLSE_2D = utils.make_acc_tensor_mn_view(tiled_mma_SdP.get_slice(tidx).partition_C(sLSE_mma))[LSEslice]

        # dS = P*(dP-dPsum)
        tdSsdS = smem_thr_copy_PdS.partition_D(sdS)

        dPsumslice = (None, 0, None)
        tdPsumsdPsum_2D = utils.make_acc_tensor_mn_view(tiled_mma_SdP.get_slice(tidx).partition_C(sdPsum_mma))[dPsumslice]

        # dV += P.T @ dO
        tdVrPt  = tiled_mma_dKV.make_fragment_A(wg_mma_dKV.partition_A(sPt))
        tdVrdOt = tiled_mma_dKV.make_fragment_B(wg_mma_dKV.partition_B(sdOt))

        # dK += dS.T @ Q
        tdKrdSt  = tiled_mma_dKV.make_fragment_A(wg_mma_dKV.partition_A(sdSt))
        tdKrQt   = tiled_mma_dKV.make_fragment_B(wg_mma_dKV.partition_B(sQt))

        # dQ  = dS @ K
        sKt = utils.transpose_view(sK)
        tdQaccumrdS = tiled_mma_dQaccum.make_fragment_A(wg_mma_dQaccum.partition_A(sdS))
        tdQaccumrK  = tiled_mma_dQaccum.make_fragment_B(wg_mma_dQaccum.partition_B(sKt))


        smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQaccumsdQaccum =      smem_thr_copy_dQaccum.partition_D(sdQaccum)

        acc_dV = cute.make_fragment(
            tiled_mma_dKV.partition_shape_C((self.n_block_size, self.head_dim_padded)),
            cutlass.Float32
        )
        acc_dK = cute.make_fragment(
            tiled_mma_dKV.partition_shape_C((self.n_block_size, self.head_dim_padded)),
            cutlass.Float32
        )

        acc_dV.fill(0.0)
        acc_dK.fill(0.0)

        mma_one_m_block_all = partial(self.mma_one_m_block,
                                      tiled_mma_SdP=tiled_mma_SdP, tiled_mma_dKV=tiled_mma_dKV, tiled_mma_dQaccum=tiled_mma_dQaccum,
                                      pipeline_q=pipeline_q, pipeline_lse=pipeline_lse,
                                      pipeline_dPsum=pipeline_dPsum, pipeline_dO=pipeline_dO,
                                      tLSEsLSE_2D=tLSEsLSE_2D, tdPsumsdPsum_2D=tdPsumsdPsum_2D, sP=sP, sdS=sdS, sdQaccum=sdQaccum, acc_dV=acc_dV, acc_dK=acc_dK,
                                      tSrQ=tSrQ, tSrK=tSrK,
                                      tPsP=tPsP, tdSsdS=tdSsdS,
                                      tdVrPt=tdVrPt, tdVrdOt=tdVrdOt,
                                      tdKrdSt=tdKrdSt, tdKrQt=tdKrQt,
                                      tdPrdO=tdPrdO, tdPrV=tdPrV,
                                      tdQaccumrdS=tdQaccumrdS, tdQaccumrK=tdQaccumrK, tdQaccumsdQaccum=tdQaccumsdQaccum,
                                      smem_thr_copy_PdS=smem_thr_copy_PdS,
                                      smem_thr_copy_dQaccum=smem_thr_copy_dQaccum,
                            )

        KV_consumer_phase = cutlass.Int32(0)
        consumer_state    = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.num_stages)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx

            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            cute.arch.mbarrier_wait(mbar_ptr_K, phase=KV_consumer_phase)
            cute.arch.mbarrier_wait(mbar_ptr_V, phase=KV_consumer_phase)

            KV_consumer_phase ^= 1

            for m_block in cutlass.range(m_block_max - m_block_min, unroll=1):
                m_block_idx = m_block_max - 1 - m_block

                consumer_state = mma_one_m_block_all(
                    warp_group_idx,
                    n_block,
                    m_block_idx,
                    head_idx,
                    batch_idx,
                    consumer_state,
                    softmax_scale_log2=softmax_scale_log2,
                )

            #scale dK
            acc_dK.store(acc_dK.load() * softmax_scale)

            self.epilogue_dKV(
                acc_dV, mdV, sV,
                acc_dK, mdK, sK,
                seqlen,
                gmem_tiled_copy_dV, gmem_tiled_copy_dK,
                tiled_mma_dKV,
                tidx, n_block, head_idx, batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


    @cute.jit
    def mma_one_m_block(
        self,
        warp_group_idx,
        n_block: cutlass.Int32,
        m_block: cutlass.Int32,
        head_idx: cutlass.Int32,
        batch_idx: cutlass.Int32,

        smem_pipe_read:    cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,

        tiled_mma_SdP:     cute.TiledMma,
        tiled_mma_dKV:     cute.TiledMma,
        tiled_mma_dQaccum: cute.TiledMma,

        pipeline_q:     cutlass.pipeline.PipelineAsync,
        pipeline_lse:   cutlass.pipeline.PipelineAsync,
        pipeline_dPsum: cutlass.pipeline.PipelineAsync,
        pipeline_dO:    cutlass.pipeline.PipelineAsync,

        tLSEsLSE_2D:     cute.Tensor,
        tdPsumsdPsum_2D: cute.Tensor,
        sP:          Optional[cute.Tensor],
        sdS:         Optional[cute.Tensor],
        sdQaccum:    cute.Tensor,

        acc_dV:      cute.Tensor,
        acc_dK:      cute.Tensor,


        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,

        tPsP:   Optional[cute.Tensor],
        tdSsdS: Optional[cute.Tensor],

        tdVrPt:  cute.Tensor,
        tdVrdOt: cute.Tensor,

        tdKrdSt: cute.Tensor,
        tdKrQt:  cute.Tensor,

        tdPrdO:  cute.Tensor,
        tdPrV:   cute.Tensor,
        tdQaccumrdS: cute.Tensor,
        tdQaccumrK:  cute.Tensor,
        tdQaccumsdQaccum: cute.Tensor,

        smem_thr_copy_PdS:  cute.TiledCopy,
        smem_thr_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: cutlass.Float32 = 1.0,
    ):


        # (1) [GEMM 1] S = Q @ K^T
        pipeline_q.consumer_wait(smem_pipe_read, pipeline_q.consumer_try_wait(smem_pipe_read))
        acc_S = cute.make_fragment(
            tiled_mma_SdP.partition_shape_C((self.m_block_size, self.n_block_size)),
            cutlass.Float32
        )

        sm90_utils.gemm(
            tiled_mma_SdP, acc_S,
            tSrQ[None, None, None, smem_pipe_read.index],
            tSrK,
            zero_init=True,
            wg_wait=0
        )

        # (2) [Pointwise 1] P = exp(S - LSE)
        pipeline_lse.consumer_wait(smem_pipe_read, pipeline_lse.consumer_try_wait(smem_pipe_read))

        tLSErLSE = cute.make_fragment_like(tLSEsLSE_2D[None, 0])
        cute.autovec_copy(tLSEsLSE_2D[None, smem_pipe_read.index], tLSErLSE)

        acc_P_mn = utils.make_acc_tensor_mn_view(acc_S)
        for r in cutlass.range_constexpr(cute.size(acc_P_mn, mode=[0])):
            acc_P_mn[r, None].store(cute.exp2(acc_P_mn[r, None].load() * softmax_scale_log2  - tLSErLSE[r]))

        # fp32->bf16
        tdVrP_acc = cute.make_tensor(acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout))
        tdVrP = cute.make_fragment_like(tdVrP_acc, self.dtype)
        utils.cvt_f16(tdVrP_acc, tdVrP)

        # cp: rmem->smem
        tPrP = smem_thr_copy_PdS.retile(tdVrP)

        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.Epilogue), number_of_threads=self.num_mma_threads)
        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads)
        cute.copy(smem_thr_copy_PdS, tPrP, tPsP)


        '''
        if warp_group_idx == 0 and cute.arch.thread_idx()[0] == 128 and m_block == 0 and n_block == 0 and head_idx == 0 and batch_idx == 0:
            for j in cutlass.range_constexpr(16):
                cute.printf("%.15f", tPrP[j].to(cutlass.Float32))
        '''

        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads)

        pipeline_lse.consumer_release(smem_pipe_read)


        # (3) [GEMM 2] dP = dO @ V.T
        pipeline_dO.consumer_wait(smem_pipe_read, pipeline_dO.consumer_try_wait(smem_pipe_read))
        acc_dP = cute.make_fragment(
            tiled_mma_SdP.partition_shape_C((self.m_block_size, self.n_block_size)),
            cutlass.Float32
        )

        sm90_utils.gemm(
            tiled_mma_SdP, acc_dP,
            tdPrdO[None, None, None, smem_pipe_read.index],
            tdPrV,
            zero_init=True,
            wg_wait=-0
        )

        # (4) [GEMM 3] dV += P.T @ dO
        sm90_utils.gemm(
            tiled_mma_dKV, acc_dV,
            tdVrPt,
            tdVrdOt[None, None, None, smem_pipe_read.index],
            zero_init=False,
            wg_wait=0
        )

        pipeline_dO.consumer_release(smem_pipe_read)

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        pipeline_dPsum.consumer_wait(smem_pipe_read, pipeline_dPsum.consumer_try_wait(smem_pipe_read))

        # dPsum
        tdPsumrdPsum = cute.make_fragment_like(tdPsumsdPsum_2D[None, 0])
        cute.autovec_copy(tdPsumsdPsum_2D[None, smem_pipe_read.index], tdPsumrdPsum)

        acc_dP_mn = utils.make_acc_tensor_mn_view(acc_dP)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            acc_dP_mn[r, None].store(
                        acc_P_mn[r, None].load() * (acc_dP_mn[r, None].load() - tdPsumrdPsum[r])
                        )

        # fp32->bf16
        tdKrdS_acc = cute.make_tensor(acc_dP.iterator, utils.convert_layout_acc_frgA(acc_dP.layout))
        tdKrdS = cute.make_fragment_like(tdKrdS_acc, self.dtype)
        utils.cvt_f16(tdKrdS_acc, tdKrdS)

        tdSrdS = smem_thr_copy_PdS.retile(tdKrdS)

        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.Epilogue), number_of_threads=self.num_mma_threads)
        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads)

        cute.copy(smem_thr_copy_PdS, tdSrdS, tdSsdS)

        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.PdS), number_of_threads=self.num_mma_threads)

        pipeline_dPsum.consumer_release(smem_pipe_read)



        # (6) [GEMM 4] dQ = dS @ K
        acc_dQ = cute.make_fragment(
            tiled_mma_dQaccum.partition_shape_C((self.m_block_size, self.head_dim_padded)),
            cutlass.Float32
        )
        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.Epilogue), number_of_threads=self.num_mma_threads)
        sm90_utils.gemm(
            tiled_mma_dQaccum, acc_dQ,
            tdQaccumrdS,
            tdQaccumrK,
            zero_init=True,
            wg_wait=0
        )

        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.Epilogue), number_of_threads=self.num_mma_threads)
        cute.arch.barrier(barrier_id=int(NamedBarrierBwd.dQEmpty), number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE)

        tdQaccumrdQaccum_tmp = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQaccumsdQaccum.shape))
        cute.copy(smem_thr_copy_dQaccum, tdQaccumrdQaccum_tmp, tdQaccumsdQaccum)

        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier_arrive(barrier_id=int(NamedBarrierBwd.dQFull), number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE)

        # (7) [GEMM 5] dK += dS.T @ Q
        sm90_utils.gemm(
            tiled_mma_dKV, acc_dK,
            tdKrdSt,
            tdKrQt[None, None, None, smem_pipe_read.index],
            zero_init=False,
            wg_wait=0
        )
        pipeline_q.consumer_release(smem_pipe_read)

        smem_pipe_read.advance()
        return smem_pipe_read


    @cute.jit
    def epilogue_dKV(
                self,
                acc_dV: cute.Tensor,
                mdV:    cute.Tensor,
                sV:     cute.Tensor,

                acc_dK: cute.Tensor,
                mdK:    cute.Tensor,
                sK:     cute.Tensor,


                seqlen: SeqlenInfoQK,

                gmem_tiled_copy_dV: cute.TiledCopy,
                gmem_tiled_copy_dK: cute.TiledCopy,

                tiled_mma_dKV: cute.TiledMma,

                tidx:      cutlass.Int32,
                n_block:   cutlass.Int32,
                head_idx:  cutlass.Int32,
                batch_idx: cutlass.Int32
            ):

            ### RMEM --> SMEM
            rdV = cute.make_fragment_like(acc_dV, self.dtype)
            rdV.store(acc_dV.load().to(self.dtype))

            rdK = cute.make_fragment_like(acc_dK, self.dtype)
            rdK.store(acc_dK.load().to(self.dtype))

            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_mma_threads)


            smem_copy_atom_dKV = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,)
            smem_thr_copy_dKV =  cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma_dKV).get_slice(tidx)


            taccdVrdV = smem_thr_copy_dKV.retile(rdV)
            taccdVsdV = smem_thr_copy_dKV.partition_D(sV)  # reuse sV SMEM
            cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)

            taccdKrdK = smem_thr_copy_dKV.retile(rdK)
            taccdKsdK = smem_thr_copy_dKV.partition_D(sK)  # reuse sK SMEM
            cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)


            # SMEM -> GMEM
            cdV = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
            mdV_cur = mdV[None, None, head_idx, batch_idx]

            cdK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
            mdK_cur = mdK[None, None, head_idx, batch_idx]

            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_mma_threads)
            gmem_thr_copy_dV = gmem_tiled_copy_dV.get_slice(tidx)
            gmem_thr_copy_dK = gmem_tiled_copy_dK.get_slice(tidx)

            tdVsdV = gmem_thr_copy_dV.partition_S(sV)
            tdVrdV = cute.make_fragment_like(tdVsdV, self.dtype)
            cute.autovec_copy(tdVsdV, tdVrdV)

            tdKsdK = gmem_thr_copy_dK.partition_S(sK)
            tdKrdK = cute.make_fragment_like(tdKsdK, self.dtype)
            cute.autovec_copy(tdKsdK, tdKrdK)

            gdV = cute.local_tile(mdV_cur, (self.n_block_size, self.head_dim_padded), (n_block, 0))
            tdVgdV = gmem_thr_copy_dV.partition_D(gdV)

            gdK = cute.local_tile(mdK_cur, (self.n_block_size, self.head_dim_padded), (n_block, 0))
            tdKgdK = gmem_thr_copy_dK.partition_D(gdK)

            tdVcdV = gmem_thr_copy_dV.partition_S(cdV)
            t0dVcdV = gmem_tiled_copy_dV.get_slice(0).partition_S(cdV)
            tdVpdV = utils.predicate_k(tdVcdV, limit=mdV.shape[1])

            tdKcdK = gmem_thr_copy_dK.partition_S(cdK)
            tdKpdK = utils.predicate_k(tdKcdK, limit=mdK.shape[1])

            for rest_m in cutlass.range_constexpr(cute.size(tdVrdV.shape[1])):
                row_idx = n_block * self.n_block_size + t0dVcdV[0, rest_m, 0][0]
                if row_idx < seqlen.seqlen_k:
                    cute.copy(
                        gmem_tiled_copy_dV,
                        tdVrdV[None, rest_m, None],
                        tdVgdV[None, rest_m, None],
                        pred=tdVpdV[None, rest_m, None] if cutlass.const_expr(self.check_hdim_v_oob) else None,
                    )
                    cute.copy(
                        gmem_tiled_copy_dK,
                        tdKrdK[None, rest_m, None],
                        tdKgdK[None, rest_m, None],
                        pred=tdKpdK[None, rest_m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )


    @cute.jit
    def dQaccum_writer(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls:    cutlass.Constexpr[Callable],
    ):

        tile_elems = cute.cosize(sdQaccum.layout)
        tile_bytes = cutlass.Int32(tile_elems * 4)

        tile_scheduler = TileSchedulerCls()
        work_tile      = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            # GMEM
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]

            base_flat = cute.domain_offset(
                            (seqlen.offset_q * self.head_dim_padded, ),
                            mdQaccum_cur
                        )

            m_block_min = cutlass.Int32(0)
            m_block_max = cute.ceil_div(seqlen.seqlen_q, self.m_block_size)

            for it_m in cutlass.range(m_block_max - m_block_min, unroll=1):
                m_block = m_block_max -1 - it_m

                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.dQFull),
                    number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE
                )

                gdQaccum_block = cute.local_tile(
                    base_flat,
                    (tile_elems, ),
                    (m_block, )
                )

                with cute.arch.elect_one():
                    sm90_utils.tma_reduce_add_bulk_f32(
                            sdQaccum.iterator,
                            gdQaccum_block.iterator,
                            tile_bytes,
                            )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

                cute.arch.barrier_arrive(
                            barrier_id=int(NamedBarrierBwd.dQEmpty),
                            number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


    @cute.jit
    def load_m_tile(
            self,
            tma_atom: cute.CopyAtom,
            tXgX: cute.Tensor,
            tXsX: cute.Tensor,
            pipeline: cutlass.pipeline.PipelineAsync,
            block: cutlass.Int32,
            producer_state: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
    ):
        pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tXgX[None, block],
            tXsX[None, producer_state.index],
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state)
        )
