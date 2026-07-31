# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 256
# - varlen
# - sliding window
# - learnable sink
# - softcap / custom score_mod
# - mask_mod
# - block sparsity
# - TMA paged KV
# Unsupported features that will be added later:
# - non-TMA paged KV
# - split-kv

import math
from functools import partial
from typing import Callable, Tuple, Optional, Literal

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import flash_attn.cute.pipeline as pipeline_custom
from cutlass import const_expr
from cutlass.cute import FastDivmodDivisor
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
)
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100_qk_ahead,
    softmax_block_sparse_sm100,
)
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors, _TUNING_CONFIG
from flash_attn.cute.utils import AuxData


@cute.jit
def _producer_tail_with_cutlass_state(pipeline_obj, state):
    tail_state = cutlass.pipeline.PipelineState(
        state.stages,
        Int32(0),
        state.index,
        state.phase,
    )
    pipeline_obj.producer_tail(tail_state)


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
    ):
        self.use_tma_KV = not paged_kv_non_tma
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 dedicated kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert head_dim % hdim_multiple_of == 0 and head_dim_v % hdim_multiple_of == 0
        assert not paged_kv_non_tma, (
            "SM100 forward with head_dim=256 does not support non-TMA paged KV"
        )
        assert not pack_gqa, "SM100 forward with head_dim=256 does not support pack_gqa"
        assert not is_split_kv, "SM100 forward with head_dim=256 does not support SplitKV"
        assert not use_clc_scheduler, (
            "SM100 forward with head_dim=256 does not support CLC scheduler"
        )
        assert kv_subtile_factor == 1, (
            "SM100 forward with head_dim=256 does not support kv_subtile_factor"
        )
        assert m_block_size == 128 and n_block_size == 128, (
            "SM100 dedicated kernel only supports tile_m=128 and tile_n=128"
        )
        assert use_2cta_instrs, "SM100 forward with head_dim=256 requires use_2cta_instrs=True"
        self.head_dim_padded = head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = head_dim_v
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        self.use_2cta_instrs = use_2cta_instrs
        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        self.cta_tiler = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (
            self.cta_group_size * m_block_size,
            n_block_size,
            min(self.cta_tiler[2], 128),
        )
        self.mma_tiler_pv = self.mma_tiler_qk
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
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
        self.is_persistent = False
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.use_correction_warps_for_epi = is_varlen_q
        self.pack_gqa = pack_gqa
        self.q_subtile_factor = q_subtile_factor
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.score_vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        self.mask_vec_size: cutlass.Constexpr = getattr(mask_mod, "__vec_size__", 1)
        self.use_clc_scheduler = False
        self.scheduling_mode = SchedulingMode.STATIC
        self.use_tma_Q = True

        if is_varlen_q:
            self.TileScheduler = SingleTileVarlenScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_ids = (9,)
        self.epilogue_warp_ids = (10,)
        self.empty_warp_ids = (11,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_warp = cute.arch.WARP_SIZE
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )
        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids

        self.tmem_s_offset = 0
        self.tmem_o_offset = 256
        self.tmem_total = self.tmem_o_offset + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_p_offset = self.tmem_s_offset

        # Look up tuning config for register counts and ex2_emu params
        self.is_sm103 = False
        _tune_key = (self.use_2cta_instrs, self.is_causal, self.head_dim_padded, self.is_sm103)
        self._tune = _TUNING_CONFIG.get(_tune_key, {})
        self.num_regs_softmax = self._tune.get("num_regs_softmax", 256)
        self.num_regs_correction = self._tune.get("num_regs_correction", 160)
        self.num_regs_other = 32
        self.ex2_emu_freq = self._tune.get("ex2_emu_freq", 4)
        self.ex2_emu_res = self._tune.get("ex2_emu_res", 3)
        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 0)

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.kv_stage = 4
        self.s_stage = 2
        self.o_stage = 1

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
        aux_data: AuxData = AuxData(),
        stream: cuda.CUstream = None,
    ):
        assert descale_tensors is None, (
            "SM100 forward with head_dim=256 does not support descale_tensors"
        )

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

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
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.o_stage,
        )
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        # TMA load for Q
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        self.use_tma_O = mCuSeqlensQ is None and mSeqUsedQ is None
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                self.epi_tile,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0] * (
            self.cta_group_size if not self.is_persistent and self.cta_group_size > 1 else 1
        )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            Int32(1),
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
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
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = 0
        sQ_size = cutlass.max(
            cute.cosize(sQ_layout),
            cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
        )

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_load_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.o_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.o_stage * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store correction scale, row sum and row max
            sScale: cute.struct.MemRange[
                self.qk_acc_dtype,
                len(self.softmax_warp_ids) * self.threads_per_warp * 3,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        gmem_tiled_copy_Q = None
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable
        )
        head_divmod = None
        if cutlass.const_expr(self.pack_gqa):
            head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")
        if cutlass.const_expr(self.use_block_sparsity and self.is_varlen_q):
            assert const_expr(blocksparse_tensors.cu_total_m_blocks is not None), (
                "blocksparse_tensors.cu_total_m_blocks must be provided for varlen blocksparsity"
            )

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
            tile_sched_params,
            num_splits,
            aux_data,
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
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k
        mO: cute.Tensor,  # (s_q, dv, h, b) or (total_q, dv, h) if there is cu_seqlens_q
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
        softmax_scale: Float32 | None,
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
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases
        of the hd256 2CTA FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Rescale, normalize, and store O

        The kernel uses TMA for Q/K/V loads, warp specialization for different
        computation phases, and the hd256-specific split-hdim TMEM layout.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.softmax_warp_ids, *self.correction_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax_warp_ids))
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )

        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_load_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem, they signal as "consumer". The mma warp then
        # waits for that signal to do the P @ V gemm. The hd256 path keeps correction on the
        # separate O pipeline, so correction threads are not participants here.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=softmax_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.o_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.o_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # Generate smem tensor Q/K/V/O.
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype),
            sO_layout.outer,
        )
        sScale = storage.sScale.get_tensor(
            cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp * 3)
        )

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
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

        block_info = BlockInfo(
            # This is the logical 2CTA Q tile height.
            self.cta_tiler[0] * self.cta_group_size,
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), (
            f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                mPageTable,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
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
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    mma_tile_coord_v,
                    tile_scheduler=tile_scheduler,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
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
                pipeline_s0_s1_sequence=None,
                learnable_sink=learnable_sink,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )
            stage = Int32(0)
            softmax_loop(stage=stage, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                gmem_tiled_copy_O,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                learnable_sink,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        if cutlass.const_expr(self.pack_gqa):
            return head_idx
        return head_idx // self.qhead_per_kvhead

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(
            src_idx=block,
            dst_idx=stage,
            tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
        )

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        block: Int32,
        K_or_V: Literal["K", "V"],
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        hdim_stage: Optional[Int32] = None,
        hdim_stage_count: cutlass.Constexpr[int] = 1,
        page_idx: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        if const_expr(hdim_stage is None):
            for iter in cutlass.range_constexpr(hdim_stage_count):
                stage = producer_state.index
                pipeline_kv.producer_acquire(producer_state)
                if const_expr(page_idx is None):
                    tXgX_cur = (
                        tXgX[None, block, iter]
                        if const_expr(K_or_V == "K")
                        else tXgX[None, iter, block]
                    )
                else:
                    # TMA paged-KV requires page_size == tile_n. The logical
                    # n-block therefore always selects tile 0 within page_idx.
                    tXgX_cur = (
                        tXgX[None, 0, iter, page_idx]
                        if const_expr(K_or_V == "K")
                        else tXgX[None, iter, 0, page_idx]
                    )
                tXsX_cur = tXsX[None, stage]
                cute.copy(
                    tma_atom,
                    tXgX_cur,
                    tXsX_cur,
                    tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
                )
                if const_expr(iter < hdim_stage_count - 1):
                    producer_state.advance()
        else:
            stage = producer_state.index
            pipeline_kv.producer_acquire(producer_state)
            if const_expr(page_idx is None):
                tXgX_cur = (
                    tXgX[None, block, hdim_stage]
                    if const_expr(K_or_V == "K")
                    else tXgX[None, hdim_stage, block]
                )
            else:
                tXgX_cur = (
                    tXgX[None, 0, hdim_stage, page_idx]
                    if const_expr(K_or_V == "K")
                    else tXgX[None, hdim_stage, 0, page_idx]
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
        mPageTable: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q,
        pipeline_kv,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler: TileSchedulerProtocol,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        issue_kv_for_this_warp = (
            const_expr(not self.use_tma_KV or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )
        issue_q_for_this_warp = (
            const_expr(not self.use_tma_Q or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (thr_mma_qk.thr_id.shape,)
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(thr_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(thr_mma_qk.thr_id.shape)

        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx_kv = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            if const_expr(mPageTable is not None):
                # Preserve the physical-page mode for page_idx-based TMA.
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
            elif const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
            else:
                mK_cur = cute.domain_offset(
                    (seqlen.offset_k, Int32(0)), mK[None, None, head_idx_kv]
                )
                mV_cur = cute.domain_offset(
                    (Int32(0), seqlen.offset_k), mV[None, None, head_idx_kv]
                )
            if const_expr(mPageTable is None):
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (None, None)
                )
            else:
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, None, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (None, None, None)
                )
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)

            q_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape)
            # (bM, bK, loopM, loopK, loopL)
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (m_block, None)
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                block_in_cluster_coord_vmnk[2],
                q_cta_layout,
                tSgQ,
                sQ,
            )
            kv_cta_layout = cute.make_layout(cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape)
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                block_in_cluster_coord_vmnk[1],
                kv_cta_layout,
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                block_in_cluster_coord_vmnk[1],
                kv_cta_layout,
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )
            # Paged TMA carries an additional physical-page mode.
            if const_expr(mPageTable is None):
                tKgK = tKgK[None, None, None]
                tVgV = tVgV[None, None, None]
            else:
                tKgK = tKgK[None, None, None, None]
                tVgV = tVgV[None, None, None, None]
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
                K_or_V="K",
                pipeline_kv=pipeline_kv,
                hdim_stage_count=self.qk_hdim_stage,
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                K_or_V="V",
                pipeline_kv=pipeline_kv,
                hdim_stage_count=self.pv_hdim_stage,
            )

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    # Q tile, split across the head-dim stages.
                    if issue_q_for_this_warp:
                        for iter in cutlass.range(self.qk_hdim_stage, unroll=1):
                            load_Q(block=iter, stage=iter)
                    q_producer_phase ^= 1

                    # First logical KV block: n_block_max - 1.
                    kv_coord = n_block_max - 1
                    page_idx = (
                        mPageTable[batch_idx, kv_coord]
                        if const_expr(mPageTable is not None)
                        else None
                    )
                    v_page_idx = page_idx
                    if issue_kv_for_this_warp:
                        load_K(
                            block=kv_coord,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )
                        kv_producer_state.advance()

                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None)
                            else None
                        )
                        # QK-ahead issue order: load next K before the previous V.
                        if issue_kv_for_this_warp:
                            load_K(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                            # V for the previously produced score tile.
                            load_V(
                                block=n_block + 1,
                                producer_state=kv_producer_state,
                                page_idx=v_page_idx,
                            )
                            kv_producer_state.advance()
                        v_page_idx = page_idx
                    # Final V tile for n_block_min.
                    if issue_kv_for_this_warp:
                        load_V(
                            block=n_block_min,
                            producer_state=kv_producer_state,
                            page_idx=v_page_idx,
                        )
                        kv_producer_state.advance()
            else:
                if issue_kv_for_this_warp or issue_q_for_this_warp:
                    kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100_qk_ahead(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        m_block,
                        seqlen,
                        kv_producer_state,
                        load_Q,
                        load_K,
                        load_V,
                        pipeline_kv,
                        self.q_load_stage,
                        q_producer_phase,
                        self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                        self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    )

            work_tile = tile_scheduler.advance_to_next_work()
            # End of persistent scheduler loop
        if issue_kv_for_this_warp:
            _producer_tail_with_cutlass_state(pipeline_kv, kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail for the TMA-Q producer warp.
        if issue_q_for_this_warp:
            pipeline_q.producer_acquire_w_index_phase(self.q_load_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk,
        tiled_mma_pv,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler=None,
    ):
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_load_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_idesc, pv_mma_idesc = (
            sm100_desc.mma_op_to_idesc(qk_mma_op),
            sm100_desc.mma_op_to_idesc(pv_mma_op),
        )
        qk_mma_kind = sm100_utils._tcgen05_mma_kind(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        v_smem_base = sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)
        q_smem_start = [
            sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
            for stage in range(self.q_load_stage)
        ]
        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.q_load_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="fa_fwd_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")
        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_load_stage == 1):
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
            for stage in range(self.q_load_stage)
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
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    split_idx,
                    num_splits,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile:
                O_should_accumulate = False
                if block_iter_count > 1:
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
                    for i in cutlass.range(1, block_iter_count - 1, 1, unroll=1):
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
    def softmax_step_block_sparse(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        softmax_step_state: Int32,
        n_block: Int32,
        softmax_step: Callable,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        p_lastsplit_producer_phase = softmax_step_state % Int32(2)
        stage = softmax_step_state // Int32(2)
        (
            mma_si_consumer_phase,
            sm_stats_producer_phase,
            p_lastsplit_producer_phase,
        ) = softmax_step(
            mma_si_consumer_phase,
            sm_stats_producer_phase,
            p_lastsplit_producer_phase,
            n_block,
            mask_fn=mask_fn,
            is_first=is_first,
            stage=stage,
        )
        stage ^= 1
        return (
            mma_si_consumer_phase,
            sm_stats_producer_phase,
            p_lastsplit_producer_phase + stage * Int32(2),
        )

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.softmax_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        aux_tensors = aux_data.tensors

        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        sm_stats_stage = Int32(0)
        p_lastsplit_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=m_block * self.cta_group_size,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                vec_size=self.mask_vec_size,
            )

            # Recompute fastdiv_mods if necessary
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                # Full blocks don't need mask_mod.
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            max_offset = 8 if cutlass.const_expr(self.q_dtype.width == 8) else 0
            if const_expr(self.score_mod is None):
                softmax_scale_log2_eff = softmax_scale_log2
                softmax_scale_eff = None
            else:
                softmax_scale_log2_eff = softmax_scale_log2
                softmax_scale_eff = softmax_scale
            rescale_threshold = 0.0
            softmax = SoftmaxSm100.create(
                softmax_scale_log2_eff,
                rescale_threshold=rescale_threshold,
                softmax_scale=softmax_scale_eff,
                max_offset=max_offset,
            )
            softmax.reset()
            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    split_idx,
                    num_splits,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen,
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)
            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                sm_stats_stage=sm_stats_stage,
                tStS=tStS,
                tScS=tScS,
                sScale=sScale,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block * self.cta_group_size,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
            )

            if const_expr(self.use_block_sparsity) or has_work:
                pipeline_sm_stats.producer_acquire_w_index_phase(
                    sm_stats_stage, sm_stats_producer_phase
                )
                sm_stats_producer_phase ^= 1

            if const_expr(self.use_block_sparsity):
                if const_expr(aux_tensors is not None):
                    m_tile_end = (m_block + 1) * self.cta_group_size * self.m_block_size
                    check_m_boundary = m_tile_end > seqlen.seqlen_q
                else:
                    check_m_boundary = False
                softmax_step_state = p_lastsplit_producer_phase + stage * Int32(2)
                (
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    softmax_step_state,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    split_idx,
                    num_splits,
                    partial(self.softmax_step_block_sparse, softmax_step=softmax_step),
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    softmax_step_state,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    1,
                    sm_stats_stage,
                    check_m_boundary,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                p_lastsplit_producer_phase = softmax_step_state % Int32(2)
                stage = softmax_step_state // Int32(2)
                if not empty_tile:
                    sScale[tidx + self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[tidx + self.m_block_size * 2] = softmax.row_max[0]
                    sm_stats_barrier.arrive_w_index(index=sm_stats_stage * 4 + warp_idx)

            if has_work and const_expr(not self.use_block_sparsity):
                mma_si_consumer_phase, sm_stats_producer_phase, p_lastsplit_producer_phase = (
                    softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        p_lastsplit_producer_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                        stage=stage,
                    )
                )
                stage ^= 1
                n_block_max -= 1
                # Next couple of iterations with causal masking
                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        n_block = n_block_max - 1 - n_tile
                        (
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                            n_block,
                            mask_fn=partial(mask_fn, mask_seqlen=False),
                            stage=stage,
                        )
                        stage ^= 1
                    n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                # The remaining iterations have no masking (but may still need mask_mod)
                n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                    n_block = n_block_max - n_tile - 1
                    if const_expr(self.mask_mod is not None):
                        (
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                            n_block,
                            mask_fn=partial(mask_fn, mask_seqlen=False),
                            stage=stage,
                        )
                    else:
                        (
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                            n_block,
                            stage=stage,
                        )
                    stage ^= 1
                # Separate iterations with local masking on the left
                if const_expr(self.is_local and block_info.window_size_left is not None):
                    n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                    for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                        n_block = n_block_max - 1 - n_tile
                        (
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            p_lastsplit_producer_phase,
                            n_block,
                            mask_fn=partial(mask_fn, mask_seqlen=False),
                            stage=stage,
                        )
                        # Now that we no longer already have the 1st iteration, need mask_seqlen=True here
                        stage ^= 1
                # Dense path always writes scale / signals
                sScale[tidx + self.m_block_size] = softmax.row_sum[0]
                if const_expr(mLSE is not None or learnable_sink is not None):
                    sScale[tidx + self.m_block_size * 2] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=sm_stats_stage * 4 + warp_idx)
            work_tile = tile_scheduler.advance_to_next_work()
        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(sm_stats_stage, sm_stats_producer_phase)

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk,
        thr_mma_pv,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats,
        sm_stats_barrier,
        pipeline_o_epi: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        mma_tile_coord_v = thr_mma_qk.thr_idx

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.o_stage
        )
        o_epi_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            m_tile_idx = m_block * self.cta_group_size + mma_tile_coord_v
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = cute.local_tile(
                mO_cur,
                self.epi_tile,
                (m_tile_idx, 0),
            )
            stats = (
                Float32(0.0),
                -Float32.inf
                if const_expr(mLSE is not None or learnable_sink is not None)
                else None,
                True,
            )
            softmax_scale_log2_eff = softmax_scale_log2
            max_offset = (
                Float32(8.0) if cutlass.const_expr(self.q_dtype.width == 8) else Float32(0.0)
            )
            max_offset_scale = (
                Float32(256.0) if cutlass.const_expr(self.q_dtype.width == 8) else Float32(1.0)
            )

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    split_idx,
                    num_splits,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:
                # The first accumulated O tile has no previous scale correction.
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                sm_stats_consumer_phase ^= 1
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    # Rescale O(i-1) before accumulating O(i).
                    sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                    scale = sScale[tidx]
                    pipeline_o_acc.consumer_wait(o_corr_consumer_state)
                    for stage in cutlass.range_constexpr(self.pv_hdim_stage):
                        self.correction_rescale(
                            thr_mma_pv,
                            tOtO[None, None, None, stage],
                            tidx,
                            scale,
                        )
                    pipeline_o_acc.consumer_release(o_corr_consumer_state)
                    o_corr_consumer_state.advance()
                    pipeline_sm_stats.consumer_release_w_index(0)
                    sm_stats_consumer_phase ^= 1
                # Normalize and store the final accumulated O tile.
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                row_sum = sScale[tidx + self.m_block_size]
                row_max = (
                    sScale[tidx + self.m_block_size * 2]
                    if const_expr(mLSE is not None or learnable_sink is not None)
                    else None
                )
                pipeline_sm_stats.consumer_release_w_index(0)
                if const_expr(learnable_sink is not None):
                    LOG2_E = math.log2(math.e)
                    sink_val = Float32(learnable_sink[head_idx])
                    if row_max == -Float32.inf:
                        row_max = sink_val * (LOG2_E / softmax_scale_log2_eff)
                        row_sum = max_offset_scale
                    else:
                        row_sum += cute.math.exp2(
                            sink_val * LOG2_E - row_max * softmax_scale_log2_eff + max_offset,
                            fastmath=True,
                        )
                acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                stats = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                scale = (
                    cute.arch.rcp_approx(row_sum)
                    if not acc_O_mn_row_is_zero_or_nan
                    else Float32(0.0)
                )
                pipeline_o_acc.consumer_wait(o_corr_consumer_state)
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_acquire(o_epi_producer_state)
                self.correction_epilogue(
                    thr_mma_pv,
                    tOtO,
                    tidx,
                    m_tile_idx,
                    seqlen.seqlen_q,
                    scale,
                    sO[None, None, 0],
                    mO_cur,
                    gO,
                    gmem_tiled_copy_O,
                )
                pipeline_o_acc.consumer_release(o_corr_consumer_state)
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_commit(o_epi_producer_state)
                    o_epi_producer_state.advance()
                o_corr_consumer_state.advance()
                sm_stats_consumer_phase ^= 1
            elif const_expr(self.use_block_sparsity):
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                sm_stats_consumer_phase ^= 1
                row_sum = Float32(1.0)
                row_max = (
                    -Float32.inf
                    if const_expr(mLSE is not None or learnable_sink is not None)
                    else None
                )
                if const_expr(learnable_sink is not None):
                    LOG2_E = math.log2(math.e)
                    sink_val = Float32(learnable_sink[head_idx])
                    if row_max == -Float32.inf:
                        row_max = sink_val * (LOG2_E / softmax_scale_log2_eff)
                        row_sum = max_offset_scale
                    else:
                        row_sum += cute.math.exp2(
                            sink_val * LOG2_E - row_max * softmax_scale_log2_eff + max_offset,
                            fastmath=True,
                        )
                acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                stats = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_acquire(o_epi_producer_state)
                self.correction_epilogue(
                    thr_mma_pv,
                    tOtO,
                    tidx,
                    m_tile_idx,
                    seqlen.seqlen_q,
                    Float32(0.0),
                    sO[None, None, 0],
                    mO_cur,
                    gO,
                    gmem_tiled_copy_O,
                )
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_commit(o_epi_producer_state)
                    o_epi_producer_state.advance()

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
                row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats
                LN2 = math.log(2.0)
                lse = (
                    (
                        row_max * softmax_scale_log2_eff
                        + (cute.math.log2(row_sum, fastmath=True) - max_offset)
                    )
                    * LN2
                    if not acc_O_mn_row_is_zero_or_nan
                    else -Float32.inf
                )
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                if tidx < seqlen.seqlen_q - m_tile_idx * self.m_block_size:
                    gLSE[tidx] = lse
            work_tile = tile_scheduler.advance_to_next_work()
        # TMEM free is owned by the MMA warp after softmax/correction arrive.
        if const_expr(not self.use_correction_warps_for_epi):
            _producer_tail_with_cutlass_state(pipeline_o_epi, o_epi_producer_state)

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        mma_tile_coord_v: Int32 = 0,
        tile_scheduler=None,
    ):
        o_epi_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.o_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_tile_idx = m_block * self.cta_group_size + mma_tile_coord_v
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]

            gO = cute.local_tile(
                mO_cur,
                self.epi_tile,
                (m_tile_idx, 0),
            )

            pipeline_o_epi.consumer_wait(o_epi_consumer_state)
            if const_expr(self.use_tma_O):
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O,
                    0,
                    cute.make_layout(1),
                    sO[None, None, 0],
                    gO,
                    single_stage=True,
                )
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                tidx = cute.arch.thread_idx()[0] % (
                    cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                )
                self._store_O_to_gmem(
                    sO[None, None, 0],
                    gO,
                    mO_cur,
                    gmem_tiled_copy_O,
                    tidx,
                    seqlen.seqlen_q,
                    m_tile_idx,
                )
            pipeline_o_epi.consumer_release(o_epi_consumer_state)
            o_epi_consumer_state.advance()

            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        p_lastsplit_producer_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        sm_stats_stage: Int32,
        tStS: cute.Tensor,
        tScS: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
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
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.softmax_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSAcc = tStS[(None, None), 0, 0, stage]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_data,
                fastdiv_mods,
                head_divmod,
            )
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

        pipeline_p_lastsplit.producer_acquire_w_index_phase(stage, p_lastsplit_producer_phase)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
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
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
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
        pipeline_sm_stats.producer_acquire_w_index_phase(sm_stats_stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        phase_advance = stage
        return (
            mma_si_consumer_phase ^ phase_advance,
            sm_stats_producer_phase ^ 1,
            p_lastsplit_producer_phase ^ phase_advance,
        )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            assert head_divmod is not None
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical, head_offset = divmod(q_physical, head_divmod)
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_data.tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1,
        )

    @cute.jit
    def correction_rescale(
        self,
        thr_mma,
        tOtO: cute.Tensor,
        tidx: Int32,
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
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
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
        tOrO_frg = cute.make_rmem_tensor((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma,
        tOtO: cute.Tensor,
        tidx: Int32,
        m_tile_idx: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and stage the attention output in shared memory."""
        corr_tile_size = 8 * 32 // self.o_dtype.width
        epi_subtile = (self.m_block_size, corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )

        for hdim_stage in cutlass.range_constexpr(self.pv_hdim_stage):
            tOtO_stage = tOtO[(None, None), 0, 0, hdim_stage]
            sO_stage = cute.local_tile(
                sO,
                cute.select(self.block_tiler_pv, mode=[0, 1]),
                (0, hdim_stage),
            )

            # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized.
            tOsO = thr_mma.get_slice(0).partition_C(sO_stage)
            tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

            tOtO_i = cute.logical_divide(
                tOtO_stage, cute.make_layout((self.m_block_size, corr_tile_size))
            )
            tOcO_i = cute.logical_divide(
                tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
            )
            tOsO_i = cute.logical_divide(
                tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
            )

            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
            thr_tmem_load = tiled_tmem_load.get_slice(tidx)
            smem_copy_atom = sm100_utils_basic.get_smem_store_op(
                self.o_layout,
                self.o_dtype,
                self.pv_acc_dtype,
                tiled_tmem_load,
            )
            tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

            tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
            tOsO_s2r = copy_utils.partition_D_position_independent(
                thr_tmem_load, tOsO_i[(None, None), None]
            )
            tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
            for i in cutlass.range(self.mma_tiler_pv[1] // corr_tile_size, unroll_full=True):
                tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
                tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
                tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
                for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                    tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                        (tOrO_frg[j], tOrO_frg[j + 1]),
                        (scale, scale),
                    )
                copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )
            self._store_O_to_gmem(
                sO,
                gO,
                mO_cur,
                gmem_tiled_copy_O,
                tidx,
                seqlen_q,
                m_tile_idx,
            )

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: cute.Tensor,
        mO_cur: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        m_tile_idx: Int32,
    ):
        """Copy O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=mO_cur.shape[1])

        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                )
