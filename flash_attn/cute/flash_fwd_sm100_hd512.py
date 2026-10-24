# Copyright (c) 2026, Colfax International.

import math
import torch
from functools import partial
from typing import Callable, Optional


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Uint32, Boolean, const_expr
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute import FastDivmodDivisorV2

from quack import copy_utils

from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import (
    dynamic_shape_signature,
    is_compact_layout,
    to_compact_dynamic_tensor,
)
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors
import flash_attn.cute.blackwell_helpers as fa_sm100_utils
from flash_attn.cute.tile_scheduler import (
    TileSchedulerProtocol,
)
from flash_attn.cute.utils import get_batch_from_cu_tensor


from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100


class D512BlockInfo(BlockInfo):
    @cute.jit
    def get_n_block_min_max(self, seqlen_info, m_block, split_idx=0, num_splits=1):
        n_min, n_max = BlockInfo.get_n_block_min_max(
            self, seqlen_info, m_block, split_idx, num_splits
        )
        # Every pipeline role must execute the same single, fully masked
        # iteration for an empty range. Load a valid physical tile, never a
        # negative or beyond-sequence tile selected by a signed local bound.
        if n_max <= n_min:
            if const_expr(self.is_split_kv):
                n_min, n_max = Int32(-1), Int32(0)
            else:
                n_min, n_max = Int32(0), Int32(1)
        return n_min, n_max


class FusedD512Forward(FlashAttentionMLAForwardSm100):
    """Reuse the large-Dv pipeline for ordinary Q512/K512/V512 attention.

    Each CTA owns 64 query rows. QK and online softmax are shared by both
    resident O256 accumulators. Qv storage/scoring are disabled; V has two
    stages whose storage is reused by the two final output stores.
    """

    def __init__(
        self, causal=False, ratio=1, kvheads=1, *, has_cu_seqlens_q=False,
        has_seqused_q=False, window_size_left=None, window_size_right=None,
        softcap=None, page_size=None, is_fp8=False, num_splits=1, pack_gqa=False,
        has_cu_seqlens_k=False, has_seqused_k=False,
    ):
        super().__init__(
            is_causal=causal,
            hdim=512,
            hdimv=512,
            is_topk_gather=False,
            pack_gqa=pack_gqa,
            qhead_per_kvhead=ratio,
            nheads_kv=kvheads,
            use_clc_scheduler=False,
            has_qk=True,
            has_cu_seqlens_q=has_cu_seqlens_q,
            has_seqused_q=has_seqused_q,
            use_cpasync_load_KV=page_size is not None and page_size != 128,
            num_splits=num_splits,
        )
        # Two FP32/BF16 output slices reuse V storage. FP8 halves each V
        # stage's bytes, so four stages preserve enough epilogue storage and
        # allow paired K loads without altering the 16-bit default pipeline.
        self.num_stages_V = 4 if is_fp8 else 2
        if is_fp8:
            self.num_stages_K = 2
        self.num_stages_sO = self.num_hdimv_splits
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.is_local = window_size_left is not None or window_size_right is not None
        general_blocks = (
            self.is_local or self.is_split_kv or self.is_varlen_q
            or has_cu_seqlens_k or has_seqused_k or page_size is not None
        )
        self.block_info_type = D512BlockInfo if general_blocks else BlockInfo
        self.softcap = softcap
        self.supports_fp8 = is_fp8
        self.is_paged = page_size is not None
        # Ordinary packed varlen has no ragged TMA output descriptor. A full
        # tail-tile store would overwrite the following sequence's output.
        # Reuse the inherited predicated register-to-global epilogue instead.
        self.use_tma_O = not self.is_varlen_q and not self.is_split_kv

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
    def softcap_and_mask(self, scores, n_block, rBitmask, softmax_scale, mask_fn):
        # Keep the inherited softmax/LSE scale contract by storing the capped
        # score in its original (unscaled) units. Apply masks after tanh so
        # that a masked -inf cannot turn into a finite saturated score.
        if const_expr(self.softcap is not None):
            inv_scale = cute.arch.rcp_approx(softmax_scale) if softmax_scale != 0.0 else 0.0
            capped = self.softcap * cute.math.tanh(
                scores.load() * (softmax_scale / self.softcap), fastmath=True
            )
            scores.store(capped * inv_scale)
        if const_expr(mask_fn is not None):
            mask_fn(scores, n_block=n_block, rBitmask=rBitmask)
        if const_expr(self.is_split_kv):
            if n_block < 0:
                scores.fill(-Float32.inf)

    @cute.jit
    def softmax_step(
        self, softmax, sRowMax, sScale, sBitmask, tStS_t2r_staged,
        tSrS_t2r, sP_smem_view, tmem_load_thr, smem_store_thr,
        pipeline_S, pipeline_P, pipeline_sm_stats, pipeline_bitmask,
        tidx, warp_idx, consumer_state_S, producer_state_P,
        producer_state_sm_stats, consumer_state_bitmask, stage: cutlass.Constexpr,
        n_block, mask_fn=None, is_first=False, store_P=None, gRowMax=None,
        packed_row0=0,
    ):
        if const_expr(self.softcap is not None or self.is_split_kv):
            mask_fn = partial(
                self.softcap_and_mask, softmax_scale=softmax.softmax_scale,
                mask_fn=mask_fn,
            )
        return FlashAttentionMLAForwardSm100.softmax_step(
            self, softmax, sRowMax, sScale, sBitmask, tStS_t2r_staged,
            tSrS_t2r, sP_smem_view, tmem_load_thr, smem_store_thr,
            pipeline_S, pipeline_P, pipeline_sm_stats, pipeline_bitmask,
            tidx, warp_idx, consumer_state_S, producer_state_P,
            producer_state_sm_stats, consumer_state_bitmask, stage, n_block,
            mask_fn=mask_fn, is_first=is_first, store_P=store_P,
            gRowMax=gRowMax, packed_row0=packed_row0,
        )

    @cute.jit
    def _get_block_idx(self, n_block, mPageTable_cur):
        # Empty SplitKV ranges execute one masked logical block -1 while
        # all memory traffic targets physical block zero.
        if const_expr(self.is_split_kv):
            n_block = max(n_block, 0)
        if const_expr(mPageTable_cur is not None):
            if const_expr(cute.size(mPageTable_cur.shape) == 0):
                return Int32(0)
            return mPageTable_cur[n_block]
        return n_block

    @cute.jit
    def relay(
        self, pipeline_K, pipeline_V, pipeline_K_cpasync, pipeline_V_cpasync,
        sO_empty_mbar_ptr, topk_length_dynamic, block_info, SeqlenInfoCls,
        tile_scheduler, mCuSeqlensQ=None,
    ):
        """Paged loaders produce K and only the two transposed V slices."""
        Producer, Consumer = pipeline.PipelineUserType.Producer, pipeline.PipelineUserType.Consumer
        producer_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
        consumer_K = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_K)
        producer_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        consumer_V = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, _, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            seqlen = SeqlenInfoCls(batch_idx)
            n_min, n_max = block_info.get_n_block_min_max(seqlen, cluster_m_block, split_idx)
            for _ in cutlass.range(n_max - n_min, unroll=1):
                consumer_K, producer_K = self.relay_inner(
                    pipeline_K_cpasync, pipeline_K, consumer_K, producer_K
                )
                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_V, producer_V = self.relay_inner(
                        pipeline_V_cpasync, pipeline_V, consumer_V, producer_V
                    )
            work_tile = tile_scheduler.advance_to_next_work()
        pipeline_K.producer_tail(producer_K)
        pipeline_V.producer_tail(producer_V)

    @cute.jit
    def load_cpasync(
        self, mIndexTopk, mK, mV, mVt, sK, sV, sVt, sBitmask,
        pipeline_K, pipeline_V, pipeline_K_cpasync, pipeline_V_cpasync,
        pipeline_bitmask, sO_empty_mbar_ptr, topk_length_dynamic,
        block_info, SeqlenInfoCls, tile_scheduler, mPageTable=None,
        mCuSeqlensQ=None,
    ):
        """Direct paged K/V loads; no contiguous materialization or Qv reads."""
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        tidx = cute.arch.thread_idx()[0] % self.num_cpasync_load_threads
        Producer = pipeline.PipelineUserType.Producer
        producer_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
        producer_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        if const_expr(self.use_tma_O):
            producer_phase_O = Int32(1)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            cluster_m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            seqlen = SeqlenInfoCls(batch_idx)
            n_min, n_max = block_info.get_n_block_min_max(seqlen, cluster_m_block, split_idx)
            head_kv = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
            page_divmod = FastDivmodDivisorV2(cute.size(mV.shape[0]))
            split_per_cta = self.hdimv // self.num_hdimv_splits // self.cta_group_size
            mVt_cta = cute.tiled_divide(mVt, (split_per_cta,))
            mVt_cta = cute.logical_divide(mVt_cta, (1, self.cta_group_size, 1, 1, 1))
            mVt_cta = mVt_cta[
                (0, None), (cta_rank, None), (0, None), (0, None), (0, None)
            ]
            mVt_cta = cute.group_modes(mVt_cta, 0, 2)
            manager_K = PagedKVManager.create(
                mPageTable, mK, mK, page_divmod, batch_idx, head_kv, tidx,
                seqlen.seqlen_k, 0, self.tile_n, self.hdim, self.hdim,
                self.num_cpasync_load_threads, mK.element_type, arch=100,
            )
            manager_V = PagedKVManager.create(
                mPageTable, mV, mVt_cta, page_divmod, batch_idx, head_kv, tidx,
                seqlen.seqlen_k, 0, self.tile_n,
                self.hdimv // self.num_hdimv_splits, split_per_cta,
                self.num_cpasync_load_threads, mV.element_type, arch=100,
            )
            if const_expr(self.use_tma_O and self.overlap_sO_sV):
                cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                producer_phase_O ^= 1
            for step in cutlass.range(n_max - n_min, unroll=1):
                n_block = max(0, n_max - 1 - step)
                manager_K.load_page_table(n_block)
                producer_K = self.cpasync_paged_load_KV(
                    manager_K, pipeline_K, pipeline_K_cpasync, sK,
                    False, "K", cta_rank, n_block, producer_K,
                )
                manager_V.load_page_table(n_block)
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    producer_V = self.cpasync_paged_load_KV(
                        manager_V, pipeline_V, pipeline_V_cpasync, sVt,
                        True, "V", cta_rank, n_block, producer_V,
                        d_offset=split * split_per_cta,
                    )
            work_tile = tile_scheduler.advance_to_next_work()

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
            cluster_m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
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
                    split_idx,
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
                    n_block = n_block_min + num_final_n_blocks - 1 - stage
                    block = self._get_block_idx(n_block, mPageTable_cur)
                    if n_block > n_block_min:
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
            cluster_m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
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
                    split_idx,
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
        mOlo: Optional[cute.Tensor] = None,
        descale_tensors=None,
    ):
        assert learnable_sink is None, "D512 does not support learnable sinks"
        assert mOlo is None, "D512 does not produce an output rounding residual"
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
            cluster_m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            if const_expr(self.use_packed_varlen_sched):
                batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            cta_m_block = cluster_m_block * self.cta_group_size + cta_rank_in_cluster

            kv_head_idx = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
            qk_descale, v_descale = self._load_effective_descales(
                descale_tensors, batch_idx, kv_head_idx
            )
            softmax_scale_log2_eff = softmax_scale_log2 * qk_descale

            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(self.is_topk_gather):
                n_block_min = 0
                n_block_max = self.topk_length // self.tile_n
                # n_block_max = topk_length_dynamic // self.tile_n
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    cluster_m_block,
                    split_idx,
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
            mO_split = mO
            if const_expr(self.is_split_kv):
                mO_split = mO[(None,) * (cute.rank(mO) - 1) + (split_idx,)]
            mO_cur = seqlen.offset_batch_Q(mO_split, batch_idx, dim=3, ragged=self.ragged_tma_O)[
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
            scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0) * v_descale

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
                mLSE_split = mLSE
                if const_expr(self.is_split_kv):
                    mLSE_split = mLSE[(None,) * (cute.rank(mLSE) - 1) + (split_idx,)]
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE_split[None, head_idx, batch_idx]
                else:
                    lse_offset = (
                        seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    )
                    mLSE_cur = cute.domain_offset((lse_offset,), mLSE_split[None, head_idx])
                gLSE = cute.local_tile(mLSE_cur, (self.cta_tile_m,), (cta_m_block,))
                if tidx < self.cta_tile_m:
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2_eff + cute.math.log2(row_sum, fastmath=True)
                         - (8.0 if const_expr(self.dtype_P.width == 8) else 0.0))
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if tidx < seqlen_q - cta_m_block * self.cta_tile_m:
                        gLSE[tidx] = lse

            row_idx = cta_m_block * self.cta_tile_m + tOicOi[0][0]

            if const_expr(self.supports_fp8 and self.use_tma_O):
                # One BF16 output stage overlaps two FP8 V stages. Both
                # final PV slices must finish before the first shared store.
                pipeline_O1.consumer_wait(consumer_states_O[1])

            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                consumer_state_Oi = consumer_states_O[split]
                pipelines_O[split].consumer_wait(consumer_state_Oi)
                # copy Oi tmem -> rmem
                cute.copy(
                    thr_tmem_load_O,
                    tOtOs_t2r[split],
                    tOrOs_t2r[split],
                )
                if const_expr(self.is_split_kv or self.is_paged):
                    if acc_O_mn_row_is_zero_or_nan:
                        tOrOs_t2r[split].fill(0.0)

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


def select_sm100_d512_config(
    *, num_splits, pack_gqa, batch_size, total_q, num_head, num_head_kv,
    seqlen_q_hint, seqlen_k_hint, window_size_left, window_size_right, num_sms=None,
    is_fp8=False, is_bf16=False, is_causal=False, is_plain_dense=False,
):
    """Host-only plan using physical 128-row / two-CTA work and native SplitKV policy.

    Lengths are static storage bounds or explicit host hints, never GPU scalar
    reads. Packed batches also use their known total token count to bound the
    sum of per-sequence partial tiles.
    """
    ratio = num_head // num_head_kv
    can_pack = ratio > 1 and ratio <= 128 and 128 % ratio == 0
    q_bound = total_q if seqlen_q_hint is None else min(seqlen_q_hint, total_q)
    # Matched short-Q probes retain the unpacked single-split plan for small
    # FP8 MQA groups where grouping regresses. Long prefill is also unchanged;
    # explicit False wins, and missing packed hints use the conservative total.
    implicit_pack = (
        0 < q_bound <= 4 and ratio >= 8
        and not (is_fp8 and num_head_kv == 1 and ratio < 64 and num_splits == 1)
    )
    # Matched BF16 causal decode controls favor independent heads for these
    # two short-Q GQA groups at KV8192. This is empirical bucket tuning;
    # unmeasured KV lengths, explicit packing, and automatic SplitKV keep
    # their existing behavior.
    if (
        is_bf16 and is_causal and is_plain_dense
        and num_splits == 1 and batch_size == 1 and num_head_kv == 2
        and seqlen_k_hint == 8192
        and (q_bound, ratio) in ((2, 32), (4, 16))
        and window_size_left is None and window_size_right is None
    ):
        implicit_pack = False
    packed = can_pack and (implicit_pack if pack_gqa is None else bool(pack_gqa))
    if num_splits > 0:
        return num_splits, packed
    from flash_attn.cute.interface import num_splits_heuristic
    if num_sms is None:
        raise ValueError("Automatic D512 SplitKV planning requires the target SM count")
    row_pack = ratio if packed else 1
    ceil_div = lambda value, divisor: (value + divisor - 1) // divisor
    partial_tiles = 0 if row_pack == 128 else max(0, batch_size - 1)
    q_clusters = min(
        batch_size * ceil_div(q_bound * row_pack, 128),
        ceil_div(total_q * row_pack, 128) + partial_tiles,
    )
    work_heads = num_head_kv if packed else num_head
    physical_ctas = 2 * max(1, q_clusters * work_heads)
    kv_bound = max(0, seqlen_k_hint)
    n_blocks = ceil_div(kv_bound, 128)
    union_width = None
    if window_size_left is not None:
        union_width = (
            window_size_left + q_bound if window_size_right is None else
            window_size_left + window_size_right + 1 + min(q_bound, ceil_div(128, row_pack))
        )
    elif window_size_right is not None:
        union_width = kv_bound + window_size_right
    if union_width is not None:
        union_width = max(0, union_width)
        # A local interval can straddle both end tiles at an arbitrary offset.
        local_blocks = ceil_div(union_width, 128) + (1 if union_width else 0)
        n_blocks = min(n_blocks, local_blocks)
    # MQA follows the native MLA four-tile minimum to bound FP32 partials.
    # Multi-KV-head GQA keeps the native occupancy policy: the measured
    # four-tile cap regresses that route, so do not extrapolate the MQA rule.
    max_splits = min(128, max(1, n_blocks // 4)) if num_head_kv == 1 else 128
    splits = num_splits_heuristic(physical_ctas, num_sms, n_blocks, max_splits)
    return max(1, splits), packed


def forward_sm100_d512(
    q, k, v, out, lse, scale, causal, *, arch: int,
    cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
    window_size_left=None, window_size_right=None, softcap=None,
    page_table=None, q_descale=None, k_descale=None, v_descale=None,
    num_splits=1, pack_gqa=False,
):
    """FP16/BF16/FP8 attention with one QK/softmax and two resident O slices.

    Q is also passed as the inherited host's Qv shape descriptor. No Qv data
    is loaded and no Qv-V score term is computed by this specialization.
    LSE uses a view of the public B,H,S / H,total buffer; no copy is needed.
    Q and KV can be packed independently, with independent used lengths.
    """
    is_split_kv = num_splits > 1
    kernel_out, kernel_lse = out, lse
    if is_split_kv:
        kernel_out = torch.empty((num_splits, *out.shape), dtype=torch.float32, device=out.device)
        lse_shape = (
            (q.shape[0], q.shape[-2], q.shape[1]) if cu_seqlens_q is None
            else (q.shape[-2], q.shape[0])
        )
        kernel_lse = torch.empty((num_splits, *lse_shape), dtype=torch.float32, device=out.device)
    lse_view = kernel_lse.transpose(-1, -2) if kernel_lse is not None else None
    tensors = (q, q, k, v, kernel_out, lse_view)
    metadata = (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table)
    descales = (q_descale, k_descale, v_descale)
    descale_modes = tuple(
        (0,) if t is not None and is_compact_layout(t) else () for t in descales
    )
    descale_tensors = DescaleTensors(*descales) if any(t is not None for t in descales) else None
    metadata_modes = tuple(
        (0,) if t is not None and is_compact_layout(t) else () for t in metadata
    )
    # Batch and sequence extents change from call to call, so mark them dynamic and keep them out
    # of the compile key; only the head shape the tiling depends on has to be baked in.
    q_modes = (0, 1) if cu_seqlens_q is None else (0,)
    k_modes = (0, 1) if cu_seqlens_k is None and page_table is None else (0,)
    out_modes = tuple(mode + 1 for mode in q_modes) if is_split_kv else q_modes
    tensor_modes = (q_modes, q_modes, k_modes, k_modes, out_modes, out_modes)
    tensor_modes = tuple(
        modes if t is not None and is_compact_layout(t) else ()
        for t, modes in zip(tensors, tensor_modes)
    )
    signature = tuple(
        dynamic_shape_signature(t, modes) for t, modes in zip(tensors, tensor_modes)
    ) + tuple(dynamic_shape_signature(t, modes) for t, modes in zip(metadata, metadata_modes)) + tuple(
        dynamic_shape_signature(t, modes) for t, modes in zip(descales, descale_modes)
    )
    key = (arch, signature, causal, window_size_left, window_size_right, softcap,
           num_splits, pack_gqa, q.device.index)
    if key not in _forward_cache:
        args = [
            to_compact_dynamic_tensor(t.detach(), 4 if i == 5 else 16, modes)
            if t is not None
            else None
            for i, (t, modes) in enumerate(zip(tensors, tensor_modes))
        ]
        metadata_args = [
            to_compact_dynamic_tensor(t.detach(), 4, modes) if t is not None else None
            for t, modes in zip(metadata, metadata_modes)
        ]
        descale_args = DescaleTensors(*(
            to_compact_dynamic_tensor(t.detach(), 4, modes) if t is not None else None
            for t, modes in zip(descales, descale_modes)
        )) if descale_tensors is not None else None
        _forward_cache[key] = cute.compile(
            FusedD512Forward(
                causal, q.shape[-2] // k.shape[-2], k.shape[-2],
                has_cu_seqlens_q=cu_seqlens_q is not None,
                has_seqused_q=seqused_q is not None,
                has_cu_seqlens_k=cu_seqlens_k is not None,
                has_seqused_k=seqused_k is not None,
                window_size_left=window_size_left, window_size_right=window_size_right,
                softcap=softcap,
                page_size=k.shape[1] if page_table is not None else None,
                is_fp8=q.element_size() == 1,
                num_splits=num_splits, pack_gqa=pack_gqa,
            ),
            *args,
            Float32(0),
            None,  # mP
            None,  # mRowMax
            *metadata_args[:4],
            None,  # mIndexTopk
            metadata_args[4],
            descale_args,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    # Exported TVM FFI functions do not retain Python's default arguments.
    _forward_cache[key](
        *tensors,
        scale,
        None,  # mP
        None,  # mRowMax
        *metadata[:4],
        None,  # mIndexTopk
        metadata[4],
        descale_tensors,
        None,  # window_size_left
        None,  # window_size_right
        None,  # learnable_sink
        None,  # mOlo
    )
    if is_split_kv:
        from flash_attn.cute.interface import _flash_attn_fwd_combine
        _flash_attn_fwd_combine(
            kernel_out, kernel_lse.transpose(-1, -2), out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q, seqused_q, _arch=arch,
        )
    return out, lse


_forward_cache = get_jit_cache("fwd_sm100_d512_fused")
