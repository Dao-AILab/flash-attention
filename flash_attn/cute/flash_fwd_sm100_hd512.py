# Copyright (c) 2026, Colfax International.

import math
from functools import partial
from typing import Callable, Optional


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Uint32, Boolean, const_expr
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05

from quack import copy_utils

from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.block_info import BlockInfo
import flash_attn.cute.blackwell_helpers as fa_sm100_utils
from flash_attn.cute.tile_scheduler import (
    TileSchedulerProtocol,
)
from flash_attn.cute.utils import get_batch_from_cu_tensor


from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100


class FusedD512Forward(FlashAttentionMLAForwardSm100):
    """Reuse the large-Dv pipeline for ordinary Q512/K512/V512 attention.

    Each CTA owns 64 query rows. QK and online softmax are shared by both
    resident O256 accumulators. Qv storage/scoring are disabled; V has two
    stages whose storage is reused by the two final output stores.
    """

    def __init__(self, causal=False, ratio=1, kvheads=1):
        super().__init__(
            is_causal=causal,
            hdim=512,
            hdimv=512,
            is_topk_gather=False,
            pack_gqa=False,
            qhead_per_kvhead=ratio,
            nheads_kv=kvheads,
            use_clc_scheduler=False,
            has_qk=True,
        )
        self.num_stages_V = 2

    def _get_shared_storage_cls(self):
        self.buffer_align_bytes = 1024

        def smem_struct_align(dtype, staged_layout, disabled=False):
            if disabled:
                return cute.struct.MemRange[dtype, 0]
            return cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(staged_layout)],
                self.buffer_align_bytes,
            ]

        def mbar_struct(num_stages):
            return cute.struct.MemRange[Int64, 2 * num_stages]

        (sQ_struct, sK_struct, sQv_struct, sV_struct, sP_struct) = (
            smem_struct_align(dtype, layout, disabled)
            for dtype, layout, disabled in [
                (self.dtype_Q, self.sQ_layout_staged, not self.has_qk),
                (self.dtype_K, self.sK_layout_staged, not self.has_qk),
                (self.dtype_Qv, self.sQv_layout_staged, True),
                (self.dtype_V, self.sV_layout_staged, False),
                (self.dtype_P, self.sP_layout_staged, False),
            ]
        )
        sStats_struct = cute.struct.MemRange[Float32, cute.cosize(self.sStats_layout)]
        sScale_struct = cute.struct.MemRange[Float32, cute.cosize(self.sScale_layout)]
        sBitmask_struct = cute.struct.MemRange[Uint32, cute.cosize(self.sBitmask_layout)]

        (
            mbar_ptr_Q_struct,
            mbar_ptr_K_struct,
            mbar_ptr_Qv_struct,
            mbar_ptr_V_struct,
            mbar_ptr_S_struct,
            mbar_ptr_P_struct,
            mbar_ptr_O0_struct,
            mbar_ptr_O1_struct,
            mbar_sm_stats_struct,
            mbar_bitmask_struct,
        ) = (
            mbar_struct(n)
            for n in [
                self.num_stages_Q,
                self.num_stages_K,
                self.num_stages_Qv,
                self.num_stages_V,
                self.num_stages_S,
                self.num_stages_P,
                self.num_stages_Oi,
                self.num_stages_Oi,
                self.num_stages_sm_stats,
                self.num_stages_bitmask,
            ]
        )
        tmem_dealloc_mbar_struct = Int64
        tmem_holding_buf_struct = Int32

        self.sched_stages = 1
        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        @cute.struct
        class SharedStorage:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_Qv: mbar_ptr_Qv_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_S: mbar_ptr_S_struct
            mbar_ptr_P: mbar_ptr_P_struct
            mbar_ptr_O0: mbar_ptr_O0_struct
            mbar_ptr_O1: mbar_ptr_O1_struct
            mbar_ptr_K_cpasync: mbar_ptr_K_struct
            mbar_ptr_V_cpasync: mbar_ptr_V_struct
            mbar_ptr_sm_stats: mbar_sm_stats_struct
            mbar_ptr_bitmask: mbar_bitmask_struct
            tmem_dealloc_mbar: tmem_dealloc_mbar_struct
            tmem_holding_buf: tmem_holding_buf_struct
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            sO_empty_mbar_ptr: cutlass.Int64

            sRowMax: sStats_struct
            sRowSum: sStats_struct
            sScale: sScale_struct
            sBitmask: sBitmask_struct
            sQv: sQv_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            sP: sP_struct

        # print("smem bytes = ", SharedStorage.size_in_bytes())

        return SharedStorage

    @cute.jit
    def load(
        self,
        mQ: Optional[cute.Tensor],
        mK: Optional[cute.Tensor],
        mQv: cute.Tensor,
        mV: cute.Tensor,
        mVt: cute.Tensor,
        sQ: Optional[cute.Tensor],
        sK: Optional[cute.Tensor],
        sQv: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_Qv: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_Vt: cute.CopyAtom,
        pipeline_Q: Optional[pipeline.PipelineAsync],
        pipeline_K: Optional[pipeline.PipelineAsync],
        pipeline_Qv: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        thr_mma_QK: cute.ThrMma,
        thr_mma_QvV: cute.ThrMma,
        thr_mma_PVt: cute.ThrMma,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mPageTable: Optional[cute.Tensor] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        # ==== Load warp ====
        # Load stationary Q and moving K/V tiles with TMA.
        # Produces Q, K and the two V256 slices.
        # consumes: -

        # ==== Make pipeline states ====
        Producer = pipeline.PipelineUserType.Producer
        if const_expr(self.has_qk):
            producer_state_Q = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Q)
        if const_expr(self.use_tma_KV):
            if const_expr(self.has_qk):
                producer_state_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
            producer_state_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        if const_expr(self.use_tma_O):
            producer_phase_O = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            # ==== Partition GMEM tensors ====
            # (seqlen_q, hdim or hdimv//2)
            if const_expr(self.has_qk):
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                # (mma_tile_m, hdim or hdimv//2)
                gQ = cute.local_tile(
                    mQ_cur,
                    (self.mma_tiler_QK[0], self.mma_tiler_QK[2]),
                    (cluster_m_block, 0),
                )
                tSgQ = thr_mma_QK.partition_A(gQ)
                tQsQ, tQgQ = cpasync.tma_partition(
                    atom=tma_atom_Q,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sQ, 0, 3),
                    gmem_tensor=cute.group_modes(tSgQ, 0, 3),
                )
            if const_expr(self.use_tma_KV):
                if const_expr(mPageTable is None):
                    mPageTable_cur = None
                    # Non-paged: select batch, tile over seqlen_k
                    if const_expr(self.has_qk):
                        # (seqlen_k, hdim)
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        # (tile_n, hdim, num_n_blocks)
                        gK = cute.local_tile(
                            mK_cur,
                            (self.mma_tiler_QK[1], self.mma_tiler_QK[2]),
                            (None, 0),
                        )
                    # (seqlen_k, hdimv)
                    mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                    # (hdimv, seqlen_k)
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        mVt_cur = mVt[None, None, head_idx_kv, batch_idx]
                    else:
                        mVt_cur = cute.domain_offset(
                            (0, seqlen.offset_k), mVt[None, None, head_idx_kv]
                        )
                    # (tile_n, hdimv//4, num_n_blocks, num_d_blocks=4)
                    gV = cute.local_tile(
                        mV_cur,
                        (self.mma_tiler_QvV[1], self.mma_tiler_QvV[2]),
                        (None, None),
                    )
                    # (tile_n, hdimv//4, num_d_blocks=4, num_n_blocks)
                    gV = cute.make_tensor(gV.iterator, cute.select(gV.layout, mode=[0, 1, 3, 2]))
                    # (hdimv//4, tile_n, num_d_blocks=4, num_n_blocks)
                    gVt = cute.local_tile(
                        mVt_cur,
                        (self.mma_tiler_PVt[1], self.mma_tiler_PVt[2]),
                        (None, None),
                    )
                else:
                    mPageTable_cur = mPageTable[batch_idx, None]
                    # Paged KV: keep pages dim, index by page_idx at load time
                    # TMA path assumes page_size == tile_n
                    if const_expr(self.has_qk):
                        # (page_size, hdim, num_pages)
                        mK_cur = mK[None, None, head_idx_kv, None]
                        # (tile_n, hdim, num_pages)
                        gK = cute.local_tile(
                            mK_cur,
                            (self.mma_tiler_QK[1], self.mma_tiler_QK[2]),
                            (0, 0, None),
                        )
                    # (page_size, hdimv, num_pages)
                    mV_cur = mV[None, None, head_idx_kv, None]
                    # (hdimv, page_size, num_pages)
                    mVt_cur = mVt[None, None, head_idx_kv, None]
                    # (tile_n, hdimv//4, num_d_blocks=4, num_pages)
                    gV = cute.local_tile(
                        mV_cur,
                        (self.mma_tiler_QvV[1], self.mma_tiler_QvV[2]),
                        (0, None, None),
                    )
                    # (hdimv//4, tile_n, num_d_blocks=4, num_pages)
                    gVt = cute.local_tile(
                        mVt_cur,
                        (self.mma_tiler_PVt[1], self.mma_tiler_PVt[2]),
                        (None, 0, None),
                    )

                if const_expr(self.has_qk):
                    tSgK = thr_mma_QK.partition_B(gK)
                    tKsK, tKgK = cpasync.tma_partition(
                        atom=tma_atom_K,
                        cta_coord=0,
                        cta_layout=cute.make_layout(1),
                        smem_tensor=cute.group_modes(sK, 0, 3),
                        gmem_tensor=cute.group_modes(tSgK, 0, 3),
                    )

                tSgV = thr_mma_QvV.partition_B(gV)
                tOgVt = thr_mma_PVt.partition_B(gVt)
                tVsV, tVgV = cpasync.tma_partition(
                    atom=tma_atom_V,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sV, 0, 3),
                    gmem_tensor=cute.group_modes(tSgV, 0, 3),
                )
                tVtsVt, tVtgVt = cpasync.tma_partition(
                    atom=tma_atom_Vt,
                    cta_coord=0,
                    cta_layout=cute.make_layout(1),
                    smem_tensor=cute.group_modes(sVt, 0, 3),
                    gmem_tensor=cute.group_modes(tOgVt, 0, 3),
                )

            if const_expr(self.has_qk):
                load_Q = partial(self.load_inner, tma_atom_Q, tQgQ, tQsQ, pipeline_Q)

            if const_expr(self.use_tma_KV):
                if const_expr(self.has_qk):
                    load_K = partial(self.load_inner, tma_atom_K, tKgK, tKsK, pipeline_K)
                load_Vt = partial(self.load_inner, tma_atom_Vt, tVtgVt, tVtsVt, pipeline_V)

            # ==== Load stationary operands ====

            # Copy Q from global to shared memory.
            if const_expr(self.has_qk):
                producer_state_Q = load_Q(producer_state_Q)
            if const_expr(self.use_tma_KV):
                # ==== Prologue ====
                n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                block = self._get_block_idx(n_block_first, mPageTable_cur)
                # copy K gmem -> smem
                if const_expr(self.has_qk):
                    producer_state_K = load_K(producer_state_K, block=block)
                # copy Vi gmem -> smem
                if const_expr(self.use_tma_O and self.overlap_sO_sV):
                    cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                    producer_phase_O ^= 1

                # ==== Main loop ====
                for n_block_group in cutlass.range(num_n_block_groups - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        n_block = n_block_max - 1 - n_block_group * self.num_stages_S - stage
                        block_next = self._get_block_idx(n_block - 1, mPageTable_cur)
                        block = self._get_block_idx(n_block, mPageTable_cur)
                        if const_expr(self.has_qk):
                            # copy K gmem -> smem
                            producer_state_K = load_K(producer_state_K, block=block_next)
                        # copy Vi gmem -> smem
                        # copy Vti gmem -> smem
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_V = load_Vt(producer_state_V, block=block, split=split)

                # ==== Epilogue ====
                num_final_n_blocks = self.num_stages_S if even_n_blocks else self.num_stages_S - 1
                for stage in cutlass.range(num_final_n_blocks, unroll_full=True):
                    n_block = num_final_n_blocks - 1 - stage
                    block = self._get_block_idx(n_block, mPageTable_cur)
                    if n_block > 0:
                        block_next = self._get_block_idx(n_block - 1, mPageTable_cur)
                        if const_expr(self.has_qk):
                            # copy K gmem -> smem
                            producer_state_K = load_K(producer_state_K, block=block_next)
                        # copy Vi gmem -> smem
                    # copy Vti gmem -> smem
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_V = load_Vt(producer_state_V, block=block, split=split)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(self.has_qk):
            pipeline_Q.producer_tail(producer_state_Q)
        if const_expr(self.use_tma_KV):
            if const_expr(self.has_qk):
                pipeline_K.producer_tail(producer_state_K)
            pipeline_V.producer_tail(producer_state_V)

    @cute.jit
    def mma(
        self,
        sQ: Optional[cute.Tensor],
        sK: Optional[cute.Tensor],
        sQv: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sP: cute.Tensor,
        tStS: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_QvV: cute.TiledMma,
        tiled_mma_PVt: cute.TiledMma,
        pipeline_Q: Optional[pipeline.PipelineAsync],
        pipeline_K: Optional[pipeline.PipelineAsync],
        pipeline_Qv: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        is_leader_cta: Boolean,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
    ):
        # ==== mma warp ====
        # Compute Q @ K^T once and both P @ V256 products.
        # Produces: S, O
        # Consumes Q, K, V and P.

        pipelines_O = [pipeline_O0, pipeline_O1]
        tOtOs = [tOtO0, tOtO1]

        use_ptx_gemm_QK = True
        use_ptx_gemm_PVt = True

        # Operands for S = Q @ K^T
        if const_expr(self.has_qk):
            tSrQ = tiled_mma_QK.make_fragment_A(sQ)
            tSrK = tiled_mma_QK.make_fragment_B(sK)

        # Operands for Oi = P @ Vi
        tOrP = tiled_mma_PVt.make_fragment_A(sP)
        tOrVt = tiled_mma_PVt.make_fragment_B(sVt)

        # GEMM functions
        if const_expr(self.has_qk):
            if const_expr(use_ptx_gemm_QK):
                gemm_QK = [
                    partial(
                        fa_sm100_utils.gemm_ptx_partial,
                        tiled_mma_QK.op,
                        self.tmem_offset_S[stage],
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )
                    for stage in range(self.num_stages_S)
                ]
            else:
                gemm_QK = [
                    partial(
                        fa_sm100_utils.gemm,
                        tiled_mma_QK,
                        tStS[None, None, None, stage],
                        zero_init=True,
                    )
                    for stage in range(self.num_stages_S)
                ]
        if const_expr(use_ptx_gemm_PVt):
            gemm_PVt = [
                partial(
                    fa_sm100_utils.gemm_ptx_partial,
                    tiled_mma_PVt.op,
                    self.tmem_offsets_O[split],
                    cta_group=self.cta_group_size,
                )
                for split in range(self.num_hdimv_splits)
            ]
        else:
            gemm_PVt = [
                partial(
                    fa_sm100_utils.gemm,
                    tiled_mma_PVt,
                    tOtOs[split],
                )
                for split in range(self.num_hdimv_splits)
            ]

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        if const_expr(self.has_qk):
            consumer_state_Q = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Q)
            consumer_state_K = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_K)
        consumer_state_V = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        producer_state_S = pipeline.make_pipeline_state(Producer, stages=self.num_stages_S)
        consumer_state_P = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_P)
        producer_state_O0 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)
        producer_state_O1 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)

        mma_fn = self.mma_inner
        if const_expr(self.has_qk):
            mma_QK = partial(
                mma_fn, gemm_QK, pipeline_K, tSrQ, sQ, tSrK, sK, use_ptx=use_ptx_gemm_QK
            )
        mma_PVt = partial(
            mma_fn, gemm_PVt, pipeline_V, tOrP, sP, tOrVt, sVt, use_ptx=use_ptx_gemm_PVt
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        O_should_accumulate = False
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                # n_block_max = self.topk_length // self.tile_n
                n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min
            even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
            num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

            if is_leader_cta:
                if const_expr(self.has_qk):
                    pipeline_Q.consumer_wait(consumer_state_Q)

                producer_states_O = [producer_state_O0, producer_state_O1]

                # ==== Prologue ====
                pipeline_S.producer_acquire(producer_state_S)
                if const_expr(self.has_qk):
                    # S = Q @ K^T
                    consumer_state_K = mma_QK(consumer_state_K, acc_stage=0)
                pipeline_S.producer_commit(producer_state_S)
                producer_state_S.advance()

                # ==== Mainloop ====
                for _ in cutlass.range(num_n_block_groups - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        next_stage = const_expr((stage + 1) % self.num_stages_S)
                        pipeline_S.producer_acquire(producer_state_S)
                        if const_expr(self.has_qk):
                            # S = Q @ K^T
                            consumer_state_K = mma_QK(consumer_state_K, acc_stage=next_stage)
                        pipeline_S.producer_commit(producer_state_S)
                        producer_state_S.advance()
                        # Oi += P @ Vi
                        pipeline_P.consumer_wait(consumer_state_P)
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_Oi = producer_states_O[split]
                            pipelines_O[split].producer_acquire(producer_state_Oi)
                            consumer_state_V = mma_PVt(
                                consumer_state_V,
                                acc_stage=split,
                                a_stage=consumer_state_P.index,
                                zero_init=not O_should_accumulate,
                            )
                            pipelines_O[split].producer_commit(producer_state_Oi)
                            producer_state_Oi.advance()
                            producer_states_O[split] = producer_state_Oi
                        pipeline_P.consumer_release(consumer_state_P)
                        consumer_state_P.advance()
                        O_should_accumulate = True

                # ==== Epilogue ====
                num_final_n_blocks = self.num_stages_S if even_n_blocks else self.num_stages_S - 1
                for stage in cutlass.range_constexpr(self.num_stages_S):
                    n_block = num_final_n_blocks - 1 - stage
                    if const_expr(stage == 0):
                        if n_block > 0:
                            pipeline_S.producer_acquire(producer_state_S)
                            if const_expr(self.has_qk):
                                # S = Q @ K^T
                                consumer_state_K = mma_QK(consumer_state_K, acc_stage=stage + 1)
                            pipeline_S.producer_commit(producer_state_S)
                            producer_state_S.advance()
                    if n_block >= 0:
                        # Oi += P @ Vi
                        pipeline_P.consumer_wait(consumer_state_P)
                        for split in cutlass.range_constexpr(self.num_hdimv_splits):
                            producer_state_Oi = producer_states_O[split]
                            pipelines_O[split].producer_acquire(producer_state_Oi)
                            consumer_state_V = mma_PVt(
                                consumer_state_V,
                                acc_stage=split,
                                a_stage=consumer_state_P.index,
                                zero_init=not O_should_accumulate,
                            )
                            pipelines_O[split].producer_commit(producer_state_Oi)
                            producer_state_Oi.advance()
                            producer_states_O[split] = producer_state_Oi
                        pipeline_P.consumer_release(consumer_state_P)
                        consumer_state_P.advance()
                    O_should_accumulate = True

                producer_state_O0, producer_state_O1 = producer_states_O

                if const_expr(self.has_qk):
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                # if we overlap sOi with sQvi for tma store, need to acquire signal
                if const_expr(self.use_tma_O and not self.overlap_sO_sV):
                    pipeline_O0.producer_tail(producer_state_O0.clone())
                    pipeline_O1.producer_tail(producer_state_O1.clone())

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            O_should_accumulate = False

        pipeline_S.producer_tail(producer_state_S)
        pipeline_O0.producer_tail(producer_state_O0)
        pipeline_O1.producer_tail(producer_state_O1)

    @cute.jit
    def correction_loop(
        self,
        softmax_scale_log2: Float32,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_O: Optional[cute.CopyAtom],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sO: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
        tiled_copy_O_r2g: cute.TiledCopy,
        topk_length_dynamic: Optional[Int32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        learnable_sink: Optional[cute.Tensor] = None,
    ):
        assert learnable_sink is None, "D512 does not support learnable sinks"
        ### ==== correction/epilogue warpgroup ====
        # Correction: copy scale smem -> rmem, copy O tmem -> rmem, rescale O, store O rmem -> tmem
        # Epilogue:   copy O tmem -> rmem, do final scaling of O, store O rmem -> gmem,
        #             optionally store LSE
        # Produces: -
        # Consumes: O, softmax stats

        tidx = cute.arch.thread_idx()[0] % self.num_epilogue_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_epilogue_threads // 32
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        leader_warp = warp_idx == 0

        tOtO0 = tOtO0[(None, None), 0, 0]  # (64, (128, 2))
        tOtO1 = tOtO1[(None, None), 0, 0]  # (64, (128, 2))
        tOtOs = [tOtO0, tOtO1]

        # tuneable parameter
        corr_tile_size = math.gcd(32, self.tmem_cols_Oi)

        tmem_load_atom_O = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.dtype_acc,
        )
        tmem_store_atom_O = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.dtype_acc,
        )
        thr_tmem_load_O = tcgen05.make_tmem_copy(tmem_load_atom_O, tOtO0).get_slice(tidx)
        thr_tmem_store_O = tcgen05.make_tmem_copy(tmem_store_atom_O, tOtO0).get_slice(tidx)

        # ((32,1),1,4)
        tOtOs_t2r = [
            thr_tmem_load_O.partition_S(tOtOs[split]) for split in range(self.num_hdimv_splits)
        ]
        tOtOs_r2t = [
            thr_tmem_store_O.partition_D(tOtOs[split]) for split in range(self.num_hdimv_splits)
        ]

        cOi = cute.make_identity_tensor((self.cta_tile_m, self.hdimv // self.num_hdimv_splits))
        thr_tiled_copy_O_r2g = tiled_copy_O_r2g.get_slice(tidx)
        tOicOi = thr_tiled_copy_O_r2g.partition_S(cOi)

        tOicOi_t2r = thr_tmem_load_O.partition_D(tOicOi[(None, None), 0, 0])

        pipelines_O = [pipeline_O0, pipeline_O1]

        Consumer = pipeline.PipelineUserType.Consumer
        consumer_state_O0 = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Oi)
        consumer_state_O1 = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Oi)
        consumer_state_sm_stats = pipeline.make_pipeline_state(
            Consumer, stages=self.num_stages_sm_stats
        )

        do_correction_rescale = partial(
            self.correction_rescale,
            thr_tmem_load_O,
            thr_tmem_store_O,
            tOicOi_t2r,
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            cta_m_block = cluster_m_block * self.cta_group_size + cta_rank_in_cluster

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                )
            num_n_blocks = n_block_max - n_block_min

            consumer_states_O = [consumer_state_O0, consumer_state_O1]

            # acquire first signal and release immediately
            pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
            pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
            consumer_state_sm_stats.advance()

            for _ in cutlass.range(num_n_blocks - 1, unroll=1):
                pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
                scale = sScale[tidx % self.cta_tile_m, consumer_state_sm_stats.index]
                should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
                consumer_state_sm_stats.advance()

                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_state_Oi = consumer_states_O[split]
                    pipelines_O[split].consumer_wait(consumer_state_Oi)
                    if should_rescale:
                        do_correction_rescale(
                            tOtOs_t2r[split],
                            tOtOs_r2t[split],
                            scale,
                        )
                    pipelines_O[split].consumer_release(consumer_state_Oi)
                    consumer_state_Oi.advance()
                    consumer_states_O[split] = consumer_state_Oi

            # (seqlen_q, hdimv)
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=self.ragged_tma_O)[
                None, None, head_idx
            ]
            # (cta_tile_m, hdimv//2, 2)
            gO = cute.local_tile(
                mO_cur,
                (self.cta_tile_m, self.hdimv // self.num_hdimv_splits),
                (cta_m_block, None),
            )
            tOgO = thr_tiled_copy_O_r2g.partition_D(gO)
            # ((32, 1), 1, 4)
            tOrOs_t2r = [
                cute.make_rmem_tensor(tOicOi_t2r.shape, self.dtype_acc)
                for split in range(self.num_hdimv_splits)
            ]
            tOrOs_r2g_f32 = [
                thr_tiled_copy_O_r2g.retile(tOrOs_t2r[split])
                for split in range(self.num_hdimv_splits)
            ]
            tOrOs_r2g = [
                cute.make_rmem_tensor_like(tOrOs_r2g_f32[split], self.dtype_O)
                for split in range(self.num_hdimv_splits)
            ]
            if const_expr(self.use_tma_O):
                tOsO = thr_tiled_copy_O_r2g.partition_D(sO)
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O,
                    0,
                    cute.make_layout(1),
                    sO,
                    gO,
                )

            self.sm_stats_barrier_full.arrive_and_wait()

            row_sum0 = sRowSum[tidx % self.cta_tile_m, 0]
            row_sum1 = sRowSum[tidx % self.cta_tile_m, 1]
            row_sum = row_sum0 + row_sum1
            acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)

            row_max = 0.0
            if const_expr(mLSE is not None):
                if tidx < self.cta_tile_m:
                    row_max = sRowMax[tidx, 0]

            self.sm_stats_barrier_empty.arrive()

            seqlen_q = (
                seqlen.seqlen_q
                if const_expr(not self.pack_gqa)
                else seqlen.seqlen_q * self.qhead_per_kvhead
            )

            # compute and store lse to gmem
            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    lse_offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    mLSE_cur = cute.domain_offset((lse_offset,), mLSE[None, head_idx])
                gLSE = cute.local_tile(mLSE_cur, (self.cta_tile_m,), (cta_m_block,))
                if tidx < self.cta_tile_m:
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + cute.math.log2(row_sum, fastmath=True))
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if tidx < seqlen_q - cta_m_block * self.cta_tile_m:
                        gLSE[tidx] = lse

            row_idx = cta_m_block * self.cta_tile_m + tOicOi[0][0]

            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                consumer_state_Oi = consumer_states_O[split]
                pipelines_O[split].consumer_wait(consumer_state_Oi)
                # copy Oi tmem -> rmem
                cute.copy(
                    thr_tmem_load_O,
                    tOtOs_t2r[split],
                    tOrOs_t2r[split],
                )

                # scale and downcast Oi
                tOrOs_r2g[split].store((tOrOs_r2g_f32[split].load() * scale).to(self.dtype_O))

                if const_expr(not self.use_tma_O):
                    # copy Oi rmem -> gmem
                    if row_idx < seqlen_q:
                        cute.copy(
                            thr_tiled_copy_O_r2g,
                            tOrOs_r2g[split],
                            tOgO[None, None, None, split],
                        )
                else:
                    # copy Oi rmem -> smem
                    # V has two stages: reuse slots 0/1 after PV completes.
                    sO_idx = split
                    cute.copy(
                        thr_tiled_copy_O_r2g,
                        tOrOs_r2g[split],
                        tOsO[None, None, None, sO_idx],
                    )
                    cute.arch.fence_view_async_shared()
                    self.epi_barrier.arrive_and_wait()
                    # tma store Oi smem -> gmem
                    if leader_warp:
                        # An empty CTA must still participate in the pipeline,
                        # but must not issue an out-of-range TMA output store.
                        if cta_m_block * self.cta_tile_m < seqlen_q:
                            store_O(src_idx=sO_idx, dst_idx=split)
                            cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(1 - split, read=True)
                        if const_expr(split == 1 and self.overlap_sO_sV):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(sO_empty_mbar_ptr)

            consumer_state_O0, consumer_state_O1 = consumer_states_O

            cute.arch.fence_view_async_tmem_load()
            pipeline_O0.consumer_release(consumer_state_O0)
            pipeline_O1.consumer_release(consumer_state_O1)
            consumer_state_O0.advance()
            consumer_state_O1.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()


def forward_sm100_d512(q, k, v, out, lse, scale, causal, *, arch: int):
    """Dense FP16/BF16 attention with one QK/softmax and two resident O slices.

    Q is also passed as the inherited host's Qv shape descriptor. No Qv data
    is loaded and no Qv-V score term is computed by this specialization.
    LSE uses a view of the public B,H,S buffer; no transpose copy is needed.
    """
    from cutlass.cute.runtime import from_dlpack

    lse_view = lse.transpose(1, 2) if lse is not None else None
    tensors = (q, q, k, v, out, lse_view)
    signature = tuple(
        (tuple(t.shape), tuple(t.stride()), t.dtype) if t is not None else None for t in tensors
    )
    key = (arch, signature, causal, q.device.index)
    if key not in _forward_cache:
        args = [
            from_dlpack(t.detach(), assumed_align=4 if i == 5 else 16) if t is not None else None
            for i, t in enumerate(tensors)
        ]
        _forward_cache[key] = cute.compile(
            FusedD512Forward(causal, q.shape[2] // k.shape[2], k.shape[2]),
            *args,
            Float32(0),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    # Exported TVM FFI functions do not retain Python's default arguments.
    _forward_cache[key](
        *tensors,
        scale,
        None,  # mP
        None,  # mRowMax
        None,  # mCuSeqlensQ
        None,  # mCuSeqlensK
        None,  # mSeqUsedQ
        None,  # mSeqUsedK
        None,  # mIndexTopk
        None,  # mPageTable
        None,  # window_size_left
        None,  # window_size_right
        None,  # learnable_sink
    )
    return out, lse


_forward_cache = get_jit_cache("fwd_sm100_d512_fused")
