# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 256
# - varlen
# Unsupported features that will be added later:
# - sliding window
# - score_mod / mask_mod
# - paged KV
# - split-kv

from functools import partial
from typing import Callable, Tuple, Optional, Literal

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import flash_attn.cute.pipeline as pipeline_custom
from cutlass import const_expr, Boolean
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.cute.typing import Int32, Int64, Float32

from quack import copy_utils
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    SM100_TMEM_CAPACITY_COLUMNS,
)
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors, _TUNING_CONFIG


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        is_persistent: bool = True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
    ):
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 dedicated kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert score_mod is None, "SM100 forward with head_dim=256 does not support score_mod"
        assert mask_mod is None, "SM100 forward with head_dim=256 does not support mask_mod"
        assert not has_aux_tensors, "SM100 forward with head_dim=256 does not support aux tensors"
        assert not paged_kv_non_tma, "SM100 forward with head_dim=256 does not support paged KV"
        assert not pack_gqa, "SM100 forward with head_dim=256 does not support pack_gqa"
        assert not is_split_kv, "SM100 forward with head_dim=256 does not support SplitKV"
        assert q_subtile_factor is None, (
            "SM100 forward with head_dim=256 does not support q_subtile_factor"
        )
        assert m_block_size == 128 and n_block_size == 128, (
            "SM100 dedicated kernel only supports tile_m=128 and tile_n=128"
        )
        assert use_2cta_instrs, (
            "SM100 forward with head_dim=256 requires use_2cta_instrs=True"
        )
        self.head_dim_padded = head_dim
        self.head_dim_v_padded = head_dim_v
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        self.q_subtile_factor = q_subtile_factor
        self.use_2cta_instrs = use_2cta_instrs
        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        qk_acc_dtype = cutlass.Float32
        pv_acc_dtype = cutlass.Float32
        mma_tiler = (self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.mma_tiler = mma_tiler
        assert mma_tiler[0] == 128 and mma_tiler[1] == 128, "Only 128x128 tile impl is supported"
        assert mma_tiler[2] == 256, "Only 256 is supported for 128x128 tile impl"
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.mma_tiler_qk = (
            self.cta_group_size * mma_tiler[0],
            mma_tiler[1],
            min(self.cta_tiler[2], 128),
        )
        self.mma_tiler_pv = self.mma_tiler_qk
        self.block_tiler_pv = (
            self.mma_tiler_pv[0] // 2,
            self.mma_tiler_pv[1],
            self.mma_tiler_pv[2],
        )
        self.q_load_stage = self.cta_tiler[2] // self.mma_tiler_qk[2]
        self.qk_hdim_stage = self.q_load_stage
        self.pv_hdim_stage = self.cta_tiler[2] // self.mma_tiler_pv[1]
        self.cluster_shape_mn = (self.cta_group_size, 1)
        self.tmem_warp_shape_mn = (4, 1)
        # Dedicated hd256 kernel uses fixed scheduling policy.
        self.is_persistent = False
        self.is_varlen_q = is_varlen_q
        self.is_causal = is_causal
        self.is_local = is_local
        self.use_semantic_trip_range = is_causal or is_local
        self.scheduling_mode = SchedulingMode.STATIC
        self.scheduler_m_block_is_logical = not is_varlen_q
        self.TileScheduler = (
            SingleTileVarlenScheduler
            if is_varlen_q
            else SingleTileScheduler
        )

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_ids = (10, 11)
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_ids,
            )
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=self.threads_per_warp
            * (len([self.mma_warp_id]) + len(self.softmax_warp_ids) + len(self.correction_warp_ids)),
        )

        self.tmem_s_offset = 0
        self.tmem_o_offset = 256
        self.tmem_p_offset = self.tmem_s_offset

        _tune_key = (True, is_causal, 256, False)  # hd256: always 2cta, no sm103 variant
        _tune = _TUNING_CONFIG.get(_tune_key, {})
        self.num_regs_softmax = _tune.get("num_regs_softmax", 256)
        self.num_regs_correction = _tune.get("num_regs_correction", 160)
        self.num_regs_other = 32  # fixed for hd256; not derived from 512 budget like other kernels
        self.ex2_emu_freq = _tune.get("ex2_emu_freq", 4)
        self.ex2_emu_res = _tune.get("ex2_emu_res", 3)
        self.ex2_emu_start_frg = _tune.get("ex2_emu_start_frg", 0)

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.kv_stage = 4
        self.s_stage = 2
        self.mma_corr_stage = 1

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        stream: cuda.CUstream = None,
    ):
        # Keep parity with FlashAttentionForwardSm100.__call__ while this
        # dedicated hdim=256 path only supports the feature subset above.
        assert mSeqUsedQ is None and mSeqUsedK is None, (
            "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
        )
        assert mPageTable is None, "SM100 forward with head_dim=256 does not support paged KV"
        assert learnable_sink is None, (
            "SM100 forward with head_dim=256 does not support learnable_sink"
        )
        assert blocksparse_tensors is None, (
            "SM100 forward with head_dim=256 does not support block sparsity"
        )
        assert aux_tensors is None, "SM100 forward with head_dim=256 does not support aux_tensors"
        assert not self.is_local, (
            "SM100 forward with head_dim=256 does not support local attention yet"
        )
        assert window_size_left is None and window_size_right is None, (
            "SM100 forward with head_dim=256 does not support runtime window_size overrides"
        )
        assert descale_tensors is None, (
            "SM100 forward with head_dim=256 does not support descale_tensors"
        )

        softmax_scale_log2, _ = utils.compute_softmax_scale_log2(softmax_scale, None)
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]

        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]

        O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & k-major
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
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.block_tiler_pv[:2]

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.q_load_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        tP_layout = cute.select(
            sm100_utils_basic.make_smem_layout_a(
                tiled_mma_pv,
                self.mma_tiler_pv,
                self.q_dtype,
                self.s_stage,
            ),
            mode=[0, 1, 2],
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv,
            self.mma_tiler_pv,
            self.v_dtype,
            self.kv_stage,
        )
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cta_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cta_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            self.cta_layout_vmnk.shape,
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(dtype, cute.select(layout, mode=[0, 1, 2]))
            for name, dtype, layout in [
                ("Q", self.q_dtype, sQ_layout),
                ("K", self.k_dtype, sK_layout),
                ("V", self.v_dtype, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size
        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0] * (
            self.cta_group_size
            if not self.is_persistent and self.cta_group_size > 1
            else 1
        )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            Int32(1),
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[0],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=not self.is_persistent and self.cta_group_size > 1,
        )
        self.tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(self.tile_sched_params)

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[
                Int64, self.q_load_stage * 2
            ]
            mbar_load_KV: cute.struct.MemRange[
                Int64, self.kv_stage * 2
            ]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[
                Int64, self.s_stage * 2
            ]
            mbar_O_full: cute.struct.MemRange[
                Int64, self.mma_corr_stage * 2
            ]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32

        @cute.struct
        class TensorStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)], 128
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)], 128
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)], 128
            ]
            sScale: cute.struct.Align[
                cute.struct.MemRange[
                    self.qk_acc_dtype,
                    len(self.softmax_warp_ids) * self.threads_per_warp * 3,
                ],
                128,
            ]

        self.shared_storage = SharedStorage
        self.tensor_storage = TensorStorage

        tma_atom_O = None
        sO_layout = None
        gmem_tiled_copy_Q = None
        gmem_tiled_copy_O = None
        num_splits = Int32(1)
        fastdiv_mods = (None, None)
        head_divmod = None

        grid_dim = cute.round_up(grid_dim, self.cluster_shape_mnk)
        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            descale_tensors,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            self.tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
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
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: Optional[cute.ComposedLayout],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Shared storage
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tensor_storage = smem.allocate(self.tensor_storage)

        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            num_stages=self.q_load_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_bytes["Q"],
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_bytes["K"],
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            num_stages=self.s_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            num_stages=self.s_stage,
            producer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            num_stages=self.s_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            defer_sync=True,
        )
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mbar_O_full.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # Generate smem tensor Q/K/V.
        sQ = tensor_storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = tensor_storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = tensor_storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sScale = tensor_storage.sScale.get_tensor(
            cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp * 3)
        )

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.pv_hdim_stage,
                stride=self.mma_tiler_pv[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)
        tP_width_ratio = Float32.width // self.v_dtype.width
        tP_stage_stride = self.mma_tiler_qk[1] * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for _i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[_i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), (
            f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"
        )
        block_info = BlockInfo(
            self.mma_tiler_qk[0],
            self.mma_tiler_qk[1],
            self.is_causal,
            self.is_local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        if const_expr(self.pack_gqa):
            seqlen_q_static = mQ.shape[0][1]
        else:
            seqlen_q_static = mQ.shape[0]
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=seqlen_q_static,
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.mma_tiler_qk[0],
            tile_n=self.mma_tiler_qk[1],
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        scale_output = Float32(1.0)

        # Cluster wait before tensor memory use
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
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
                pipeline_q,
                pipeline_kv,
                block_info,
                SeqlenInfoCls,
                cta_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                thr_mma_qk,
                thr_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                SeqlenInfoCls,
                mma_tile_coord_v,
                tile_scheduler=tile_scheduler,
            )
            tmem.relinquish_alloc_permit()
            self.tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                tile_scheduler=tile_scheduler,
            )
            softmax_loop(tStS=tStS, mma_tile_coord_v=mma_tile_coord_v)
            self.tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.wait_for_alloc()
            tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_pv,
                tOtO,
                sScale,
                mO,
                mLSE,
                scale_output,
                softmax_scale,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_acc,
                block_info,
                SeqlenInfoCls,
                mma_tile_coord_v,
                tile_scheduler=tile_scheduler,
            )
            self.tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty warps reg dealloc
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx > self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        return

    @cute.jit
    def normalize_work_tile(self, work_tile, mma_tile_coord_v: Int32):
        scheduler_m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        m_block = scheduler_m_block
        if const_expr(not self.scheduler_m_block_is_logical):
            m_block = scheduler_m_block // self.cta_group_size
        m_tile_idx = self.get_m_tile_idx(m_block, mma_tile_coord_v)
        head_idx_kv = self._kv_head_idx(head_idx)
        return m_block, m_tile_idx, head_idx, head_idx_kv, batch_idx, split_idx

    @cute.jit
    def get_m_tile_idx(self, m_block: Int32, mma_tile_coord_v: Int32) -> Int32:
        return m_block * self.cta_group_size + mma_tile_coord_v

    @cute.jit
    def get_mask_m_block(self, m_block: Int32) -> Int32:
        return m_block * self.cta_group_size

    @cute.jit
    def get_sparse_m_block(self, m_block: Int32) -> Int32:
        # Sparse metadata is indexed by the logical 256-row Q tile in this hd256 2CTA kernel.
        return m_block

    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        if const_expr(self.pack_gqa):
            return head_idx
        return head_idx // self.qhead_per_kvhead

    def load_Q(
        self,
        load_Q_fn,
        stage: Int32,
        pipeline_q,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(
            src_idx=stage,
            dst_idx=stage,
            tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
        )

    def load_KV(
        self,
        tma_atom,
        tXgX,
        tXsX,
        block: Int32,
        hdim_stage: Int32,
        producer_state,
        K_or_V: Literal["K", "V"],
        pipeline_kv,
    ):
        assert K_or_V in ("K", "V")
        pipeline_kv.producer_acquire(producer_state)
        stage = producer_state.index
        tXgX_cur = (
            tXgX[None, block, hdim_stage]
            if const_expr(K_or_V == "K")
            else tXgX[None, hdim_stage, block]
        )
        tXsX_cur = tXsX[None, stage]
        cute.copy(
            tma_atom,
            tXgX_cur,
            tXsX_cur,
            tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
        )

    @cute.jit
    def load(
        self,
        thr_mma_qk,
        thr_mma_pv,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q,
        pipeline_kv,
        block_info: BlockInfo,
        SeqlenInfoCls,
        cta_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk,
        mma_tile_coord_v: Int32,
        tile_scheduler=None,
    ):
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, _, head_idx, head_idx_kv, batch_idx, _ = self.normalize_work_tile(
                work_tile, mma_tile_coord_v
            )
            seqlen = SeqlenInfoCls(batch_idx)
            process_tile = m_block * self.mma_tiler_qk[0] < seqlen.seqlen_q
            if process_tile:
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur = mK[None, None, head_idx_kv, batch_idx]
                    mV_cur = mV[None, None, head_idx_kv, batch_idx]
                else:
                    mV_cur = cute.domain_offset(
                        (Int32(0), seqlen.offset_k), mV[None, None, head_idx_kv]
                    )
                q_cta_layout = cute.make_layout(
                    cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape
                )
                # (bM, bK, loopM, loopK, loopL)
                gQ = cute.flat_divide(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]))
                gQ = gQ[None, None, m_block, None]
                tSgQ = thr_mma_qk.partition_A(gQ)
                load_Q_fn = copy_utils.tma_get_copy_fn(
                    tma_atom_Q,
                    block_in_cluster_coord_vmnk[2],
                    q_cta_layout,
                    tSgQ,
                    sQ,
                )[0]
                kv_cta_layout = cute.make_layout(
                    cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
                )
                gK = cute.flat_divide(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]))
                tSgK = thr_mma_qk.partition_B(gK)
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    block_in_cluster_coord_vmnk[1],
                    kv_cta_layout,
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )

                gV = cute.flat_divide(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]))
                tSgV = thr_mma_pv.partition_B(gV)
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    block_in_cluster_coord_vmnk[1],
                    kv_cta_layout,
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tSgV, 0, 3),
                )
                # ((atom_v, rest_v), RestN, RestK)
                tKgK = tKgK[None, None, None]
                # ((atom_v, rest_v), RestN, RestK)
                tVgV = tVgV[None, None, None]
                load_Q = partial(
                    self.load_Q,
                    load_Q_fn,
                    pipeline_q=pipeline_q,
                    phase=q_producer_phase,
                )
                load_K = partial(
                    self.load_KV,
                    tma_atom_K,
                    tKgK,
                    tKsK,
                    pipeline_kv=pipeline_kv,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_atom_V,
                    tVgV,
                    tVsV,
                    pipeline_kv=pipeline_kv,
                    K_or_V="V",
                )

                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, Int32(0), Int32(1)
                )
                tile_block_count = n_block_max - n_block_min
                # Q tile
                for iter in cutlass.range(self.qk_hdim_stage, unroll=1):
                    load_Q(stage=iter)
                q_producer_phase ^= 1

                # First logical KV block: n_block_max - 1.
                kv_coord = n_block_max - 1
                for iter in cutlass.range(self.qk_hdim_stage, unroll=1):
                    load_K(block=kv_coord, hdim_stage=iter, producer_state=kv_producer_state)
                    kv_producer_state.advance()
                kv_coord -= 1

                for i in cutlass.range(1, tile_block_count, 1, unroll=1):
                    # QK-ahead issue order: load next K before the previous V.
                    for iter in cutlass.range(self.qk_hdim_stage, unroll=1):
                        load_K(block=kv_coord, hdim_stage=iter, producer_state=kv_producer_state)
                        kv_producer_state.advance()
                    # V for the previously produced score tile.
                    for iter in cutlass.range(self.pv_hdim_stage, unroll=1):
                        load_V(
                            block=kv_coord + 1,
                            hdim_stage=iter,
                            producer_state=kv_producer_state,
                        )
                        kv_producer_state.advance()
                    kv_coord -= 1
                # Final V tile for n_block_min.
                for iter in cutlass.range(self.pv_hdim_stage, unroll=1):
                    load_V(
                        block=n_block_min,
                        hdim_stage=iter,
                        producer_state=kv_producer_state,
                    )
                    kv_producer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()
            # End of persistent scheduler loop
        pipeline_kv.producer_tail(kv_producer_state)
        pipeline_q.producer_acquire_w_index_phase(self.q_load_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        thr_mma_qk,
        thr_mma_pv,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q,
        pipeline_kv,
        pipeline_s_p_o,
        pipeline_p_lastsplit,
        pipeline_o_acc,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        SeqlenInfoCls,
        mma_tile_coord_v: Int32,
        tile_scheduler=None,
    ):
        tSrQ = thr_mma_qk.make_fragment_A(sQ)
        tSrK = thr_mma_qk.make_fragment_B(sK)
        tOrV = thr_mma_pv.make_fragment_B(sV)
        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        q_smem_start = [
            sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
            for stage in range(self.qk_hdim_stage)
        ]
        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.qk_hdim_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="fa_fwd_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")
        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.qk_hdim_stage == 1):
            sQ_stage_stride = 0
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fa_fwd_q_smem_desc",
                idesc_var_name="fa_fwd_qk_mma_idesc",
                kind=qk_mma_kind,
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.qk_hdim_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                tOtO[None, None, None, stage].iterator.toint(),
                sA=None,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.pv_hdim_stage)
        ]
        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        s_p_o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.s_stage
        )
        p_lastsplit_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.s_stage
        )
        o_acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_corr_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, _, _, _, batch_idx, _ = self.normalize_work_tile(
                work_tile, mma_tile_coord_v
            )
            seqlen = SeqlenInfoCls(batch_idx)
            process_tile = m_block * self.mma_tiler_qk[0] < seqlen.seqlen_q

            if process_tile:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, Int32(0), Int32(1)
                )
                tile_block_count = n_block_max - n_block_min

                O_should_accumulate = False
                if tile_block_count > 1:
                    # First QK for logical n_block_max - 1.
                    if is_leader_cta:
                        pipeline_s_p_o.producer_acquire(s_p_o_producer_state)
                        s_stage = s_p_o_producer_state.index
                        tSAcc = tStS[None, None, None, s_stage]
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                        for iter in cutlass.range_constexpr(self.qk_hdim_stage):
                            pipeline_q.consumer_wait_w_index_phase(iter, mma_q_consumer_phase)
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                            Ki_index = mma_kv_consumer_state.index
                            sK_cur = sK[None, None, None, Ki_index]
                            gemm_Si[iter](
                                acc_tmem_addr=tSAcc.iterator.toint(),
                                smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                    sK_cur.iterator
                                ),
                                zero_init=iter == 0,
                            )
                            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                            pipeline_kv.consumer_release(mma_kv_consumer_state)
                            mma_kv_consumer_state.advance()
                        pipeline_s_p_o.producer_commit(s_p_o_producer_state)
                        s_p_o_producer_state.advance()
                    for i in cutlass.range(1, tile_block_count - 1, 1, unroll=1):
                        # Next QK in reverse logical n_block order.
                        if is_leader_cta:
                            pipeline_s_p_o.producer_acquire(s_p_o_producer_state)
                            s_stage = s_p_o_producer_state.index
                            tSAcc = tStS[None, None, None, s_stage]
                            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range_constexpr(self.qk_hdim_stage):
                                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                                Ki_index = mma_kv_consumer_state.index
                                sK_cur = sK[None, None, None, Ki_index]
                                gemm_Si[iter](
                                    acc_tmem_addr=tSAcc.iterator.toint(),
                                    smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                        sK_cur.iterator
                                    ),
                                    zero_init=iter == 0,
                                )
                                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                                pipeline_kv.consumer_release(mma_kv_consumer_state)
                                mma_kv_consumer_state.advance()
                            pipeline_s_p_o.producer_commit(s_p_o_producer_state)
                            s_p_o_producer_state.advance()

                            # PV for the previous softmax tile.
                            pipeline_p_lastsplit.consumer_wait(p_lastsplit_consumer_state)
                            p_stage = p_lastsplit_consumer_state.index
                            pipeline_o_acc.producer_acquire(o_acc_producer_state)
                            for iter in cutlass.range_constexpr(self.pv_hdim_stage):
                                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                                Vi_index = mma_kv_consumer_state.index
                                sV_cur = sV[None, None, None, Vi_index]
                                gemm_Pi[iter](
                                    tOrP[None, None, None, p_stage],
                                    tOrV[None, None, None, Vi_index],
                                    sB=sV_cur,
                                    zero_init=not O_should_accumulate,
                                )
                                pipeline_kv.consumer_release(mma_kv_consumer_state)
                                mma_kv_consumer_state.advance()
                            O_should_accumulate = True
                            pipeline_o_acc.producer_commit(o_acc_producer_state)
                            o_acc_producer_state.advance()
                            pipeline_p_lastsplit.consumer_release(p_lastsplit_consumer_state)
                            p_lastsplit_consumer_state.advance()
                    if is_leader_cta:
                        # Last QK in this work tile.
                        pipeline_s_p_o.producer_acquire(s_p_o_producer_state)
                        s_stage = s_p_o_producer_state.index
                        tSAcc = tStS[None, None, None, s_stage]
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                        for iter in cutlass.range_constexpr(self.qk_hdim_stage):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                            Ki_index = mma_kv_consumer_state.index
                            sK_cur = sK[None, None, None, Ki_index]
                            gemm_Si[iter](
                                acc_tmem_addr=tSAcc.iterator.toint(),
                                smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                    sK_cur.iterator
                                ),
                                zero_init=iter == 0,
                            )
                            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                            pipeline_kv.consumer_release(mma_kv_consumer_state)
                            mma_kv_consumer_state.advance()
                            pipeline_q.consumer_release_w_index(iter)
                        mma_q_consumer_phase ^= 1
                        pipeline_s_p_o.producer_commit(s_p_o_producer_state)
                        s_p_o_producer_state.advance()

                        # PV for the penultimate produced P tile.
                        pipeline_p_lastsplit.consumer_wait(p_lastsplit_consumer_state)
                        p_stage = p_lastsplit_consumer_state.index
                        pipeline_o_acc.producer_acquire(o_acc_producer_state)
                        for iter in cutlass.range_constexpr(self.pv_hdim_stage):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                            Vi_index = mma_kv_consumer_state.index
                            sV_cur = sV[None, None, None, Vi_index]
                            gemm_Pi[iter](
                                tOrP[None, None, None, p_stage],
                                tOrV[None, None, None, Vi_index],
                                sB=sV_cur,
                                zero_init=not O_should_accumulate,
                            )
                            pipeline_kv.consumer_release(mma_kv_consumer_state)
                            mma_kv_consumer_state.advance()
                        O_should_accumulate = True
                        pipeline_o_acc.producer_commit(o_acc_producer_state)
                        o_acc_producer_state.advance()
                        pipeline_p_lastsplit.consumer_release(p_lastsplit_consumer_state)
                        p_lastsplit_consumer_state.advance()
                else:
                    if is_leader_cta:
                        # Only QK for this work tile.
                        pipeline_s_p_o.producer_acquire(s_p_o_producer_state)
                        s_stage = s_p_o_producer_state.index
                        tSAcc = tStS[None, None, None, s_stage]
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                        for iter in cutlass.range_constexpr(self.qk_hdim_stage):
                            pipeline_q.consumer_wait_w_index_phase(iter, mma_q_consumer_phase)
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                            Ki_index = mma_kv_consumer_state.index
                            sK_cur = sK[None, None, None, Ki_index]
                            gemm_Si[iter](
                                acc_tmem_addr=tSAcc.iterator.toint(),
                                smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                    sK_cur.iterator
                                ),
                                zero_init=iter == 0,
                            )
                            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                            pipeline_kv.consumer_release(mma_kv_consumer_state)
                            mma_kv_consumer_state.advance()
                            pipeline_q.consumer_release_w_index(iter)
                        mma_q_consumer_phase ^= 1
                        pipeline_s_p_o.producer_commit(s_p_o_producer_state)
                        s_p_o_producer_state.advance()

                if is_leader_cta:
                    # Final PV for the last produced P tile.
                    pipeline_p_lastsplit.consumer_wait(p_lastsplit_consumer_state)
                    p_stage = p_lastsplit_consumer_state.index
                    pipeline_o_acc.producer_acquire(o_acc_producer_state)
                    for iter in cutlass.range_constexpr(self.pv_hdim_stage):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Vi_index = mma_kv_consumer_state.index
                        sV_cur = sV[None, None, None, Vi_index]
                        gemm_Pi[iter](
                            tOrP[None, None, None, p_stage],
                            tOrV[None, None, None, Vi_index],
                            sB=sV_cur,
                            zero_init=not O_should_accumulate,
                        )
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                    O_should_accumulate = True
                    pipeline_o_acc.producer_commit(o_acc_producer_state)
                    o_acc_producer_state.advance()
                    pipeline_p_lastsplit.consumer_release(p_lastsplit_consumer_state)
                    p_lastsplit_consumer_state.advance()
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

    @cute.jit
    def softmax_loop(
        self,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk: cute.core.ThrMma,
        tStS: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o,
        pipeline_p_lastsplit,
        pipeline_sm_stats,
        sm_stats_barrier,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls,
        AttentionMaskCls,
        mma_tile_coord_v: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.softmax_warp_ids)
        )
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        sm_stats_producer_phase = Int32(1)
        sm_stats_stage = Int32(0)
        softmax_stage = Int32(0)
        mma_si_consumer_phase = Int32(0)
        p_lastsplit_producer_phase = Int32(1)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, m_tile_idx, head_idx, _, batch_idx, _ = self.normalize_work_tile(
                work_tile, mma_tile_coord_v
            )
            seqlen = SeqlenInfoCls(batch_idx)
            seqlen_q = seqlen.seqlen_q
            process_tile = m_block * self.mma_tiler_qk[0] < seqlen_q
            if process_tile:
                mask_m_block = self.get_mask_m_block(m_block)
                mask = AttentionMaskCls(seqlen)
                shared_mask_kwargs = dict(
                    m_block=mask_m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    aux_tensors=aux_tensors,
                )
                mask_fn = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )

                softmax = SoftmaxSm100.create(softmax_scale_log2)
                softmax.reset()
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, Int32(0), Int32(1)
                )
                tile_block_count = n_block_max - n_block_min
                has_work = tile_block_count > Int32(0)
                softmax_step = partial(
                    self.softmax_step,
                    softmax=softmax,
                    thr_mma_qk=thr_mma_qk,
                    pipeline_s_p_o=pipeline_s_p_o,
                    pipeline_p_lastsplit=pipeline_p_lastsplit,
                    pipeline_sm_stats=pipeline_sm_stats,
                    sm_stats_barrier=sm_stats_barrier,
                    sm_stats_stage=sm_stats_stage,
                    tStS=tStS,
                    tScS=tScS,
                    sScale=sScale,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=mask_m_block,
                    seqlen=seqlen,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                )

                if has_work:
                    pipeline_sm_stats.producer_acquire_w_index_phase(
                        sm_stats_stage, sm_stats_producer_phase
                    )
                    sm_stats_producer_phase ^= 1
                    (
                        softmax_stage,
                        mma_si_consumer_phase,
                        p_lastsplit_producer_phase,
                        sm_stats_producer_phase,
                    ) = softmax_step(
                        softmax_stage,
                        mma_si_consumer_phase,
                        p_lastsplit_producer_phase,
                        sm_stats_producer_phase,
                        n_block_max - 1,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                        is_first=True,
                    )
                    n_block_max -= 1
                    if const_expr(self.use_semantic_trip_range):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                softmax_stage,
                                mma_si_consumer_phase,
                                p_lastsplit_producer_phase,
                                sm_stats_producer_phase,
                            ) = softmax_step(
                                softmax_stage,
                                mma_si_consumer_phase,
                                p_lastsplit_producer_phase,
                                sm_stats_producer_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        n_block = n_block_max - 1 - n_tile
                        (
                            softmax_stage,
                            mma_si_consumer_phase,
                            p_lastsplit_producer_phase,
                            sm_stats_producer_phase,
                        ) = softmax_step(
                            softmax_stage,
                            mma_si_consumer_phase,
                            p_lastsplit_producer_phase,
                            sm_stats_producer_phase,
                            n_block,
                        )
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            (
                                softmax_stage,
                                mma_si_consumer_phase,
                                p_lastsplit_producer_phase,
                                sm_stats_producer_phase,
                            ) = softmax_step(
                                softmax_stage,
                                mma_si_consumer_phase,
                                p_lastsplit_producer_phase,
                                sm_stats_producer_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                sScale[tidx + self.m_block_size] = softmax.row_sum[0]
                if const_expr(mLSE is not None):
                    sScale[tidx + self.m_block_size * 2] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=sm_stats_stage * 4 + warp_idx)
            work_tile = tile_scheduler.advance_to_next_work()
        pipeline_sm_stats.producer_acquire_w_index_phase(sm_stats_stage, sm_stats_producer_phase)

    @cute.jit
    def correction_loop(
        self,
        thr_mma_pv: cute.core.ThrMma,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        scale_output: Float32,
        softmax_scale: Float32,
        pipeline_sm_stats,
        sm_stats_barrier,
        pipeline_o_acc,
        block_info: BlockInfo,
        SeqlenInfoCls,
        mma_tile_coord_v: Int32,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        sm_stats_stage = Int32(0)
        o_acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_corr_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, m_tile_idx, head_idx, _, batch_idx, _ = self.normalize_work_tile(
                work_tile, mma_tile_coord_v
            )
            seqlen = SeqlenInfoCls(batch_idx)
            seqlen_q = seqlen.seqlen_q
            process_tile = m_block * self.mma_tiler_qk[0] < seqlen_q

            if process_tile:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[
                    None, None, head_idx
                ]

                # (bM, bN, loopM, loopN, loopL)
                gO = cute.local_tile(
                    mO_cur,
                    cute.select(self.block_tiler_pv, mode=[0, 1]),
                    (m_tile_idx, None),
                )
                tOcO = cute.local_tile(
                    cute.make_identity_tensor(mO_cur.shape),
                    cute.select(self.block_tiler_pv, mode=[0, 1]),
                    (m_tile_idx, None),
                )
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, Int32(0), Int32(1)
                )
                total_block_count = n_block_max - n_block_min

                # The first accumulated O tile has no previous scale correction.
                sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_stage * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(sm_stats_stage)
                for step in cutlass.range(1, total_block_count, 1, unroll=1):
                    # Rescale O(i-1) before accumulating O(i).
                    sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_stage * 4 + warp_idx)
                    scale = sScale[tidx]
                    pipeline_o_acc.consumer_wait(o_acc_consumer_state)
                    for stage in cutlass.range_constexpr(self.pv_hdim_stage):
                        self.correction_rescale(
                            thr_mma_pv,
                            tOtO[None, None, None, stage],
                            tidx,
                            scale,
                        )
                    pipeline_o_acc.consumer_release(o_acc_consumer_state)
                    o_acc_consumer_state.advance()
                    pipeline_sm_stats.consumer_release_w_index(sm_stats_stage)
                # Normalize and store the final accumulated O tile.
                sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_stage * 4 + warp_idx)
                row_sum = sScale[tidx + self.m_block_size]
                row_max = sScale[tidx + self.m_block_size * 2]
                pipeline_sm_stats.consumer_release_w_index(sm_stats_stage)
                acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                scale = scale_output / (row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                pipeline_o_acc.consumer_wait(o_acc_consumer_state)
                self.correction_epilogue(
                    seqlen_q,
                    scale,
                    gO,
                    tOcO,
                    tOtO,
                    self.epi_tile,
                )
                pipeline_o_acc.consumer_release(o_acc_consumer_state)
                o_acc_consumer_state.advance()
                if const_expr(mLSE is not None):
                    q_idx = m_tile_idx * self.cta_tiler[0] + tidx
                    lse = (
                        softmax_scale * row_max + cute.math.log(row_sum, fastmath=True)
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if cute.elem_less(q_idx, seqlen.seqlen_q):
                        global_q_idx = (
                            q_idx + seqlen.offset_q
                            if const_expr(seqlen.has_cu_seqlens_q)
                            else q_idx
                        )
                        if const_expr(not seqlen.has_cu_seqlens_q):
                            mLSE[global_q_idx, head_idx, batch_idx] = lse
                        else:
                            mLSE[global_q_idx, head_idx] = lse
            work_tile = tile_scheduler.advance_to_next_work()
        # TMEM free is owned by the MMA warp after softmax/correction arrive.

    @cute.jit
    def softmax_step(
        self,
        stage: Int32,
        mma_si_consumer_phase: Int32,
        p_lastsplit_producer_phase: Int32,
        sm_stats_producer_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        pipeline_s_p_o,
        pipeline_p_lastsplit,
        pipeline_sm_stats,
        sm_stats_barrier,
        sm_stats_stage: Int32,
        tStS: cute.Tensor,
        tScS: cute.Tensor,
        sScale: cute.Tensor,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: cutlass.Constexpr[bool] = False,
    ) -> Tuple[Int32, Int32, Int32, Int32]:
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.softmax_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSAcc = tStS[(None, None), 0, 0, stage]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(
            tidx
        )
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        cute.arch.fence_view_async_tmem_load()
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block, thr_tmem_load=thr_tmem_load)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            sScale[tidx] = acc_scale
        sm_stats_barrier.arrive_w_index(index=sm_stats_stage * 4 + warp_idx)

        tSrP_r2t = cute.make_rmem_tensor(tSrS_t2r.shape, self.q_dtype)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_res=self.ex2_emu_res,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )

        pipeline_p_lastsplit.producer_acquire_w_index_phase(
            stage, p_lastsplit_producer_phase
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tSAcc.shape[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((tSAcc.shape[0], tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator, tStP_layout)
        tScP_layout = cute.composition(
            tScS.layout, cute.make_layout((tScS.shape[0], tilePlikeFP32))
        )
        tScP = cute.make_tensor(tScS.iterator, tScP_layout)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(
            tidx
        )
        tStP_r2t = thr_tmem_store.partition_D(tStP)
        tScP_r2t = thr_tmem_store.partition_S(tScP)
        tSrP_r2t_f32 = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t.iterator, dtype=self.qk_acc_dtype),
            tScP_r2t.shape,
        )
        cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t)
        cute.arch.fence_view_async_tmem_store()

        pipeline_p_lastsplit.producer_commit_w_index(stage)
        pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(
            sm_stats_stage, sm_stats_producer_phase
        )
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        stage ^= 1
        phase_advance = Int32(1) - stage
        return (
            stage,
            mma_si_consumer_phase ^ phase_advance,
            p_lastsplit_producer_phase ^ phase_advance,
            sm_stats_producer_phase ^ 1,
        )

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.mma_tiler_pv[1] // corr_tile_size
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        seqlen_q: Int32,
        scale: Float32,
        gO: cute.Tensor,
        tOcO: cute.Tensor,
        tOtO: cute.Tensor,
        epi_tile: cute.Tile,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        for stage in cutlass.range(self.pv_hdim_stage):
            gO_stage = gO[None, None, stage]
            tOcO_stage = tOcO[None, None, stage]
            tOtO_stage = tOtO[(None, None), 0, 0, stage]
            tOtO_i = cute.zipped_divide(tOtO_stage, epi_tile)
            tOcO_i = cute.zipped_divide(tOcO_stage, epi_tile)
            tOgO_i = cute.zipped_divide(gO_stage, epi_tile)
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i)
            thr_tmem_load = tiled_tmem_load.get_slice(tidx)
            tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
            tOgO_r2g = thr_tmem_load.partition_D(tOgO_i)
            tOcO_t2r = thr_tmem_load.partition_D(tOcO_i)
            for i in cutlass.range(cute.size(tOtO_t2r, mode=[1]), unroll_full=True):
                tOtO_t2r_i = tOtO_t2r[None, i, 0]
                tOgO_r2g_i = tOgO_r2g[None, i, 0]
                tOcO_t2r_i = tOcO_t2r[None, i, 0]
                tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
                for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                    tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                        (tOrO_frg[j], tOrO_frg[j + 1]),
                        (scale, scale),
                    )
                tOrO_cvt = cute.make_rmem_tensor(tOrO_frg.shape, self.o_dtype)
                o_vec = tOrO_frg.load()
                tOrO_cvt.store(o_vec.to(self.o_dtype))
                if cute.elem_less(tOcO_t2r_i[0][0], seqlen_q):
                    cute.autovec_copy(tOrO_cvt, tOgO_r2g_i)
