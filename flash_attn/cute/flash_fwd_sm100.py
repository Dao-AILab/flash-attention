# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - varlen
# - sliding window
# Unsupported features that will be added later:
# - split-kv (optimizing for inference)
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import enum
import math
from typing import Type, Tuple, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

import flash_attn.cute.utils as utils
# import flash_attn.cute.pipeline as pipeline
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.seqlen_info import SeqlenInfo
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute.fast_math import FastDivmod
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, StaticPersistentTileScheduler, SingleTileLPTScheduler, SingleTileVarlenScheduler, ParamsBase


# class NamedBarrierFwd(enum.IntEnum):
#     Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


class FlashAttentionForwardSm100:

    arch = 100

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
    ):
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        assert head_dim == head_dim_v, "head_dim and head_dim_v must be the same for now"
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        # 2 Q tile per CTA
        self.cta_tiler = (2 * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.pv_mma_tiler = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        self.s0_s1_barrier = False

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epilogue_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        self.tmem_alloc_sync_bar_id = 1

        self.tmem_s0_offset = 0
        self.tmem_s1_offset = self.tmem_s0_offset + self.n_block_size
        self.tmem_o0_offset = self.tmem_s1_offset + self.n_block_size
        self.tmem_o1_offset = self.tmem_o0_offset + self.head_dim_v_padded
        self.tmem_total = self.tmem_o1_offset + self.head_dim_v_padded
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_p_offset = 0
        self.tmem_p0_offset = self.tmem_s0_offset + self.tmem_p_offset
        self.tmem_p1_offset = self.tmem_s1_offset + self.tmem_p_offset

        # vec buffer for row_max & row_sum
        self.tmem_vec0_offset = 0
        self.tmem_vec1_offset = self.tmem_vec0_offset + self.n_block_size

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64
            self.num_regs_other = 48
        else:
            self.num_regs_softmax = 192 if self.is_causal or self.is_local else 184
            # self.num_regs_softmax = 176
            # self.num_regs_correction = 96
            # self.num_regs_correction = 80
            # self.num_regs_correction = 64 if self.is_causal or self.is_local else 80
            self.num_regs_correction = 64
            # self.num_regs_other = 32
            # self.num_regs_other = 64
            # self.num_regs_other = 80
            # self.num_regs_other = 48
            # self.num_regs_other = 96 if self.is_causal or self.is_local else 80
            self.num_regs_other = 64 if self.is_causal or self.is_local else 80
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = 2
        self.kv_stage = 4 if self.q_dtype.width == 8 else 3
        self.acc_stage = 1
        self.epi_stage = 2

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
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
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=QO_layout_transpose))
            for t in (mQ, mO)
        ]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)) if const_expr(mLSE is not None) else None
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of mV is not supported")

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None and mSeqUsedQ is None and not self.pack_gqa

        cta_group = tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.pv_mma_tiler[:2]

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.pv_mma_tiler, self.q_dtype, self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.pv_mma_tiler, self.v_dtype, self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage,
        )

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.pv_mma_tiler,
            tiled_mma_pv,
            self.cluster_layout_vmnk.shape,
        )

        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)

        # print(sO_layout.outer)
        if const_expr(not self.use_tma_O):
            self.epilogue_warp_ids = (14, 15)
            self.empty_warp_ids = ()
        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.o_dtype, num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1), order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2]))

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = SingleTileScheduler if const_expr(not self.is_persistent) else StaticPersistentTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]) if const_expr(mCuSeqlensQ is not None) else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            block_size=self.cta_tiler[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_O_rescaled_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_O_rescaled_offset + 2
        self.mbar_O_full_offset = self.mbar_S_full_offset + 2
        self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + 2
        self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + 2
        self.mbar_corr_epi_full_offset = self.mbar_softmax_corr_empty_offset + self.epi_stage
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.epi_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + 2
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        self.mbar_total = self.mbar_P_full_2_offset + 2

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            sScale: cute.struct.MemRange[Float32, 2 * self.m_block_size * (1 if const_expr(mLSE is None) else 2)]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(sO_layout)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        # Right after this, we multiply by log2(e) before applying exp2.
        # To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        # (assigning it to softcap_val) and pre-multiply softcap_val * log2(e)
        # (assigning it to softmax_scale_log2).
        LOG2_E = math.log2(math.e)
        if const_expr(softcap is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = None
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = Float32(softmax_scale / softcap)
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)
        # Launch the kernel synchronously
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softcap_val,
            window_size_left,
            window_size_right,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softcap_val: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            if const_expr(not self.pack_gqa):
                cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(self.use_tma_O):
                cpasync.prefetch_descriptor(tma_atom_O)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_full_offset + i, len([self.load_warp_id]))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id]))
        if warp_idx == 2:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_softmax_corr_empty_offset + i, cute.arch.WARP_SIZE * 4)
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_softmax_corr_full_offset + i, cute.arch.WARP_SIZE * 4)
        if warp_idx == 3:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE)
        if warp_idx == 4:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_corr_epi_full_offset + i, cute.arch.WARP_SIZE * len(self.correction_warp_ids))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_corr_epi_empty_offset + i, cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        if warp_idx == 5:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_O_rescaled_offset + i, cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids) + len(self.correction_warp_ids)))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id]))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id]))
        if warp_idx == 6:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_2_offset + i, cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        if warp_idx == 7:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                        *self.correction_warp_ids,
                    )
                ),
            )
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # sQ_pi = storage.sQ.get_tensor(sQ_layout)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # sK_pi = storage.sK.get_tensor(sK_layout)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        sScale = storage.sScale.get_tensor(cute.make_layout(256))

        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM

        qk_acc_shape = thr_mma_qk.partition_shape_C((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        # TODO: this is a fake tensor, need to retrieve tmem_ptr
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem,
                                 assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = thr_mma_pv.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)

        tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout)

        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem_o0_offset, tOtO.layout)
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem_o1_offset, tOtO.layout)

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrP0 = cute.make_tensor(
            tOrP.iterator
            + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p0_offset,
            tOrP.layout,
        )
        tOrP1 = cute.make_tensor(
            tOrP.iterator
            + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p1_offset,
            tOrP.layout,
        )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0], self.cta_tiler[1], self.is_causal, self.is_local,
            window_size_left, window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfo, seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ, mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask, self.m_block_size, self.n_block_size,
            window_size_left=window_size_left, window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_kv,
                mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
        # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
                cute.arch.sync_warp()

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                sQ_layout.inner,
                sK_layout.inner,
                sV_layout.inner,
                tStS0,
                tStS1,
                tOtO0,
                tOtO1,
                tOrP0,
                tOrP1,
                pipeline_kv,
                mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.epilogue_s2g(mO, sO, gmem_tiled_copy_O, tma_atom_O, mbar_ptr, SeqlenInfoCls, TileSchedulerCls)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(
                    stage=stage,
                    tStSi=cute.make_tensor(tStS.iterator + (self.tmem_s0_offset if stage == 0 else self.tmem_s1_offset), tStS.layout))
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout)
                    softmax_loop(stage=0, tStSi=tStSi)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout)
                    softmax_loop(stage=1, tStSi=tStSi)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO0,
                tOtO1,
                sScale,
                mO,
                mLSE,
                sO,
                tma_atom_O,
                mbar_ptr,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):

        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.kv_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mQ_cur = mQ[None, None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mQ_cur = cute.domain_offset((offset, 0), mQ[None, None, head_idx])
            head_idx_kv = head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
            else:
                mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))
            tSgQ = thr_mma_qk.partition_A(gQ)
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
            tSgK = thr_mma_qk.partition_B(gK)
            gV = cute.local_tile(mV_cur, cute.select(self.pv_mma_tiler, mode=[1, 2]), (0, None))
            tOgV = thr_mma_pv.partition_B(gV)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ, 0, 3),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            def load_Q(stage: int):
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_empty_offset + stage, q_producer_phase)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr + self.mbar_load_q_full_offset + stage, self.tma_copy_q_bytes)
                cute.copy(
                    tma_atom_Q,
                    tQgQ[None, 2 * m_block + stage],
                    tQsQ[None, stage],
                    tma_bar_ptr=mbar_ptr + self.mbar_load_q_full_offset + stage,
                )

            load_K = partial(self.load_K, tma_atom_K, tKgK, tKsK, pipeline_kv)
            load_V = partial(self.load_K, tma_atom_V, tVgV, tVsV, pipeline_kv)

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            load_Q(0)  # Q0
            load_K(n_block_max - 1, kv_producer_state)  # K0
            kv_producer_state.advance()
            load_Q(1)  # Q1
            q_producer_phase ^= 1
            load_V(n_block_max - 1, kv_producer_state)  # V0
            kv_producer_state.advance()
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                n_block = n_block_max - 2 - i
                load_K(n_block, kv_producer_state)  # Ki
                kv_producer_state.advance()
                load_V(n_block, kv_producer_state)  # Vi
                kv_producer_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ_swizzle: cute.Swizzle,
        sK_swizzle: cute.Swizzle,
        sV_swizzle: cute.Swizzle,
        tStS0: cute.Tensor,
        tStS1: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tOrP0: cute.Tensor,
        tOrP1: cute.Tensor,
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        thr_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        thr_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM
        tSrQ = thr_mma_qk.make_fragment_A(sQ)
        tSrK = thr_mma_qk.make_fragment_B(sK)
        tOrV = thr_mma_pv.make_fragment_B(sV)
        tStSs = (tStS0, tStS1)
        tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        tOrPs = (tOrP0, tOrP1)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_partial,
                qk_mma_op, self.tmem_s0_offset if const_expr(stage == 0) else self.tmem_s1_offset, tSrQs[stage],
                sA=sQ[None, None, None, stage],
                sA_swizzle=sQ_swizzle, sB_swizzle=sK_swizzle, zero_init=True
            )
            for stage in range(2)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op, self.tmem_o0_offset if const_expr(stage == 0) else self.tmem_o1_offset, tOrPs[stage],
                sA=None, sA_swizzle=None, sB_swizzle=sV_swizzle
            )
            for stage in range(2)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            for stage in cutlass.range_constexpr(2):
                # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                # 1. wait for Q0 / Q1
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase)
                # 2. wait for K0
                if const_expr(stage == 0):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                # We don't need to acquire empty S0 / S1.
                # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                # are empty. For subsequent iterations, the wait happened at the end
                # of the while loop.
                # 3. gemm
                # sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                gemm_Si[stage](tCrB=tSrKi, sB=sK[None, None, None, mma_kv_consumer_state.index])
                # 4. release S0 / S1
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
            mma_q_consumer_phase ^= 1
            # 5. release K0
            pipeline_kv.consumer_release(mma_kv_consumer_state)
            mma_kv_consumer_state.advance()
            # End of GEMM (Q1 * K0 -> S1)
            # Note: Q0 & Q1 are still needed in the seqlen_kv loop
            # so we need to release them after the seqlen_kv loop

            # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
            O_should_accumulate = False
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                mma_kv_release_state = mma_kv_consumer_state.clone()
                Vi_index = mma_kv_consumer_state.index
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(2):
                    # 2. acquire corrected O0/O1_partial and P0 / P1
                    # For the first iteration in this work tile, waiting for O0/O1_partial
                    # means that the correction warps has finished reading tO during
                    # the last iteration of the previous work tile has finished.
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase)
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate, mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage, mbar_phase= P_full_O_rescaled_phase)
                    # 4. release accumulated O0_partial / O1_partial
                    # Don't need to signal O_full to the correction warps anymore since the
                    # correction warps wait for the softmax warps anyway. By the time the softmax
                    # warps finished, S_i for the next iteration must have been done, so O_i-1
                    # must have been done as well.
                    # with cute.arch.elect_one():
                    #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # 5. release V(i-1)
                    if const_expr(stage == 1):
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        mma_kv_release_state.advance()
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    # GEMM_QK0i (Q0 * Ki -> S0)
                    # 1. wait for Ki
                    if const_expr(stage == 0):
                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index = mma_kv_consumer_state.index
                    # 2. gemm
                    # Don't need to wait for the softmax warp to have finished reading the previous
                    # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                    # has been read and Pi has been written.
                    # sm100_utils.gemm(tiled_mma_qk, tStS0, tSrQs[0], tSrK[None, None, None, Ki_index], zero_init=True)
                    gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK[None, None, None, Ki_index])
                    # 3. release S0
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                    # End of GEMM_QK0i (Q0 * Ki -> S0)
                # 4. release Ki
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                P_full_O_rescaled_phase ^= 1
                O_should_accumulate = True
            # End of seqlen_kv loop

            # release Q0 & Q1
            with cute.arch.elect_one():
                tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + 0)
                tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + 1)

            # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
            # 1. wait for V0
            pipeline_kv.consumer_wait(mma_kv_consumer_state)
            Vi_index = mma_kv_consumer_state.index
            tOrVi = tOrV[None, None, None, Vi_index]
            for stage in cutlass.range_constexpr(2):
                # 2. acquire corrected Oi_partial and Pi
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, P_full_O_rescaled_phase)
                # 3. gemm
                # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate, mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage, mbar_phase=P_full_O_rescaled_phase)
                # 4. release accumulated O0_partial
                # We do need O_full here since for the last tile, by the time the softmax warp
                # has signaled to the correction warp, the softmax warp has just finished compute
                # the row sum of the current tile. It does not guarantee that the 1st tile
                # of the next work tile has been computed yet.
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                # End of GEMM_PV00 (P0 * V0 -> O0_partial)
            P_full_O_rescaled_phase ^= 1
            # 5. release Vi_end
            pipeline_kv.consumer_release(mma_kv_consumer_state)
            mma_kv_consumer_state.advance()
            # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int,
        # stage: Int32,
        softmax_scale_log2: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStSi: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids)
            )
        )

        cS_base = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        tScS = thr_mma_qk.partition_C(cS_base)

        tStS_scale_layout = cute.composition(tStSi.layout, cute.make_layout((self.m_block_size, 1)))
        tStScale = cute.make_tensor(tStSi.iterator, tStS_scale_layout)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((self.m_block_size, 1)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32,
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(tidx)

        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tSrScale_r2t_shape = thr_tmem_store_scale.partition_S(tScS_vec).shape
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP)
        thr_tmem_store = tiled_tmem_store.get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        mma_si_consumer_phase = Int32(0)
        si_corr_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask_sm100, m_block=m_block * 2 + stage, thr_mma=thr_mma_qk, thr_tmem_load=thr_tmem_load, mask_causal=self.is_causal, mask_local=self.is_local
            )
            softmax = SoftmaxSm100(softmax_scale_log2, rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0)
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                thr_mma_qk=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
            )

            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase)
            si_corr_producer_phase ^= 1

            # 1 masking iter
            mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block_max - 1, is_first=True, mask_fn=partial(mask_fn, mask_seqlen=True))
            n_block_max -= 1
            # Next couple of iterations with causal masking
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block, mask_fn=partial(mask_fn, mask_seqlen=False))
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
            # The remaining iterations have no masking
            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                n_block = n_block_max - n_tile - 1
                mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block)
            # Separate iterations with local masking on the left
            if const_expr(self.is_local and block_info.window_size_left is not None):
                n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, si_corr_producer_phase, s0_s1_sequence_phase, n_block, mask_fn=partial(mask_fn, mask_seqlen=False))
                    # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

            # tSrScale_r2t = cute.make_fragment(tSrScale_r2t_shape, Float32)
            # tSrScale_r2t[0] = softmax.row_sum[0]
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
            if const_expr(mLSE is not None):
                sScale[tidx + stage * self.m_block_size + self.m_block_size * 2] = softmax.row_max[0]
            # if tidx == 0:
            #     cute.printf("softmax row sum stage %d: %f, row_max = %f\n", stage, softmax.row_sum[0], softmax.row_max[0])
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)
            # if tidx == 0: cute.printf("softmax row sum stage %d: %f\n", stage, softmax.row_sum[0])

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + utils.log2f(softmax.row_sum[0])) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def softmax_step(
        self,
        # stage: Int32,
        mma_si_consumer_phase: Int32,
        si_corr_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        thr_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1])))
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((self.m_block_size, 1)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tScP_layout = cute.composition(tScS.layout, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tScP = cute.make_tensor(tScS.iterator, tScP_layout)
        tScS_t2r_shape = thr_tmem_load.partition_D(tScS).shape

        # Wait for Si
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(tScS_t2r_shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        # tSrScale_r2t = cute.make_fragment(thr_tmem_store_scale.partition_S(tScS_vec).shape, Float32)
        # tSrScale_r2t[0] = acc_scale
        # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
        # cute.arch.fence_view_async_tmem_store()
        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # if thread_idx == 0: cute.printf("softmax acc_scale stage %d: %f, row_max = %f\n", stage, acc_scale, row_max)
        # Notify correction wg that row_max is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_full_offset + stage)

        # if thread_idx == 0 and stage == 0: cute.print_tensor(tSrS_t2r)
        # print(tSrS_t2r)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_fragment(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout,
        )
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        softmax.apply_exp2_convert(tSrS_t2r, tSrP_r2t, e2e=mask_fn is None, e2e_freq=16 if self.head_dim_padded <= 64 else 16)
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)
        # print(tSrP_r2t_f32, tStP_r2t)
        # cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2]) // 4 * 3):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2]) // 4 * 3, cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage, si_corr_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.arch.exp2(acc_scale_)
        return mma_si_consumer_phase ^ 1, si_corr_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        mbar_ptr: cute.Pointer,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1])))
        tStS_scale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScale_0 = cute.make_tensor(tStS.iterator + self.tmem_vec0_offset, tStS_scale_layout)
        tStScale_1 = cute.make_tensor(tStS.iterator + self.tmem_vec1_offset, tStS_scale_layout)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((self.m_block_size, 1)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype,
        )
        tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScale_0)
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(tidx)

        tStScale_0_t2r = thr_tmem_load_vec.partition_S(tStScale_0)
        tStScale_1_t2r = thr_tmem_load_vec.partition_S(tStScale_1)
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScS_vec).shape

        tOtOs = [tOtO0, tOtO1]
        tStScales_t2r = [tStScale_0_t2r, tStScale_1_t2r]

        # First iter: no correction is required
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 0)
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + 1)

        softmax_corr_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            # Ignore first signal from softmax as no correction is required
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_full_offset + 0, softmax_corr_consumer_phase)
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 0)
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_full_offset + 1, softmax_corr_consumer_phase)
            softmax_corr_consumer_phase ^= 1

            tSrScale_t2r = cute.make_fragment(tSrScale_t2r_shape, Float32)
            for i in cutlass.range(n_block_max - n_block_min - 1, unroll=1):
                for stage in cutlass.range_constexpr(2):
                    # wait for S0 / S1
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_full_offset + stage, softmax_corr_consumer_phase)
                    # cute.copy(tiled_tmem_load_vec, tStScale_1_t2r, tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[stage]
                    scale = sScale[tidx + stage * self.m_block_size]
                    should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                    # should_rescale = True
                    # if tidx == 0: cute.printf("Correction scale i = %d, for stage %d: %f, should_rescale = %d\n", i, stage, scale, should_rescale)
                    # Don't need O_full anymore, since by the time softmax has signaled the correction
                    # warps, S_i must have been done, so O_i-1 must have been done as well.
                    # cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                    if should_rescale:
                        self.correction_rescale(thr_mma_pv, tOtOs[stage], tidx, scale)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + (1 - stage))
                softmax_corr_consumer_phase ^= 1
                # o_corr_consumer_phase ^= 1
            # End of seqlen_corr_loop_steps

            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + 1)

            stats = [None, None]
            for stage in cutlass.range_constexpr(2):
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_softmax_corr_full_offset + stage, softmax_corr_consumer_phase)
                # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                # cute.arch.fence_view_async_tmem_load()
                # scale = tSrScale_t2r[0]
                row_sum = sScale[tidx + stage * self.m_block_size]
                if const_expr(mLSE is not None):
                    row_max = sScale[tidx + stage * self.m_block_size + self.m_block_size * 2]
                else:
                    row_max = None
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_softmax_corr_empty_offset + stage)
                acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
                cute.arch.mbarrier_wait(mbar_ptr + self.mbar_corr_epi_empty_offset + stage, corr_epi_producer_phase)
                self.correction_epilogue(
                    thr_mma_pv, tOtOs[stage], tidx, scale, sO[None, None, stage],
                )
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_full_offset + stage)
                # Signal for the next work tile that O buffers in tmem are already read, so
                # mma warp can write to them
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
                # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)
            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2,))
                for stage in cutlass.range_constexpr(2):
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    # if tidx == 0 and stage <= 1:
                    #     cute.printf("row_sum = {}, row_max = {}, acc_O_mn_row_is_zero_or_nan = {}\n", row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + utils.log2f(row_sum)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
                    )
                    if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
                        gLSE[tidx + stage * self.m_block_size] = lse

            o_corr_consumer_phase ^= 1
            softmax_corr_consumer_phase ^= 1
            corr_epi_producer_phase ^= 1

            # gO_qdhb = cute.local_tile(mO, cute.select(self.pv_mma_tiler, mode=[0, 1]), (None, 0, None, None))
            # gO = gO_qdhb[None, None, None, head_idx, batch_idx]
            # tOsO, tOgO = cpasync.tma_partition(
            #     tma_atom_O,
            #     0,
            #     cute.make_layout(1),
            #     cute.group_modes(sO, 0, 2),
            #     cute.group_modes(gO, 0, 2),
            # )
            # warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
            # stage = warp_idx_in_wg
            # if stage < 2:
            #     # wait from corr, issue tma store on smem
            #     # 1. wait for O0 / O1 final
            #     cute.arch.mbarrier_wait(mbar_ptr + self.mbar_corr_epi_full_offset + stage, corr_epi_producer_phase)
            #     # 2. copy O0 / O1 to gmem
            #     cute.copy(tma_atom_O, tOsO[None, stage], tOgO[None, 2 * m_block + stage])
            #     cute.arch.cp_async_bulk_commit_group()
            #     # Ensure O0 / O1 buffer is ready to be released
            #     cute.arch.cp_async_bulk_wait_group(0, read=True)
            #     cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        thread_idx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOcO = thr_mma.partition_C(cO)

        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )

        tOtO_i_layout = cute.composition(tOtO.layout, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i_layout = cute.composition(tOcO.layout, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
        tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg_i = tOrO_frg[None, i]
            tTMrO_i_layout = cute.composition(tOrO_frg_i.layout, cute.make_layout(tOrO_frg.shape[0]))
            tTMrO_i = cute.make_tensor(tOrO_frg_i.iterator, tTMrO_i_layout)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tTMrO_i)
            for j in cutlass.range_constexpr(0, cute.size(tTMrO_i), 2):
                tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO_i[j], tTMrO_i[j + 1]), (scale, scale),
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(tiled_tmem_store, tTMrO_i, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        thread_idx: Int32,
        scale: Float32,
        sO: cute.Tensor,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        cO = cute.make_identity_tensor((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cO)

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy(
            smem_copy_atom,
            layout_tv=tiled_tmem_load.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_load.tiler_mn,
        )

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale),
                )
            tSMrO = cute.make_fragment(tOrO_frg.shape, self.o_dtype)
            o_vec = tOrO_frg.load()
            tSMrO.store(o_vec.to(self.o_dtype))
            cute.copy(tiled_smem_store, tSMrO, tOsO_r2s_i)

        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mO_cur = mO[None, None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mO_cur = cute.domain_offset((offset, 0), mO[None, None, head_idx])
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
            if const_expr(self.use_tma_O):
                tOsO, tOgO = cpasync.tma_partition(
                    tma_atom_O,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sO, 0, 2),
                    cute.group_modes(gO, 0, 2),
                )
                for stage in cutlass.range_constexpr(2):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 / O1 final
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase)
                    # 2. copy O0 / O1 to gmem
                    cute.copy(tma_atom_O, tOsO[None, stage], tOgO[None, 2 * m_block + stage])
                    cute.arch.cp_async_bulk_commit_group()
                for stage in cutlass.range_constexpr(2):
                    # Ensure O0 / O1 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
            else:
                tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOsO = gmem_thr_copy_O.partition_S(sO)
                cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                for stage in cutlass.range_constexpr(2):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 / O1 final
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase)
                    # 2. copy O0 / O1 to gmem
                    # load acc O from smem to rmem for wider vectorization
                    tOrO = cute.make_fragment_like(tOsO[None, None, None, 0], self.o_dtype)
                    cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                    # copy acc O from rmem to gmem
                    for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                        if t0OcO[0, rest_m, 0][0] < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size - tOcO[0][0]:
                            cute.copy(
                                gmem_tiled_copy_O,
                                tOrO[None, rest_m, None],
                                tOgO[None, rest_m, None, 2 * m_block + stage],
                                pred=tOpO[None, rest_m, None] if self.check_hdim_v_oob else None,
                            )
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)

            # Advance to next tile
            epi_consumer_phase ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # @cute.jit
    def load_K(
        self,
        tma_atom: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineAsync,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
    ):
        pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tKgK[None, block],
            tKsK[None, producer_state.index],
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state)
        )

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))
        return cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=load_kv_producer_group,
            consumer_group=load_kv_consumer_group,
            tx_count=self.tma_copy_kv_bytes,
        )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )
