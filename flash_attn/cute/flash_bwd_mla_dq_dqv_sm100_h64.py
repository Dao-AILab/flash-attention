# Copyright (c) 2026, Colfax International.

"""
CuTe DSL implementation of the dQ+dQv gemm of the sparse-MLA (DSA) backward for exactly
64 Q heads per KV head (see AI/SPARSE_MLA_64H.md, section C6-dq).

    dQ  = dS @ K_rope   (per token: [64 heads, top_k] x [top_k gathered rows, hdim])
    dQv = dS @ V        (per token: [64 heads, top_k] x [top_k gathered rows, hdim_v])

where the K/V rows are gathered according to the top-k index tensor mIdxTopK.

Inputs:
    - dS:      [batch, seqlen_q, 64, top_k] or [total_q, 64, top_k]
    - K:       [batch, seqlen_k, hdim]      or [total_k, hdim]      (optional)
    - V:       [batch, seqlen_k, hdim_v]    or [total_k, hdim_v]
    - IdxTopK: [batch, seqlen_q, top_k]     or [total_q, top_k]

Outputs:
    - dQ:  [batch, seqlen_q, 64, hdim]   or [total_q, 64, hdim]
    - dQv: [batch, seqlen_q, 64, hdim_v] or [total_q, 64, hdim_v]

One token per CTA, no cluster: the 64 heads are the M=64 of ``tcgen05.mma.cta_group::1``
instructions (dQv as two N=256 tiles accumulated in the two TMEM lane halves, dQ as one N=64
tile), and each top-k key's whole latent + rope row is gathered exactly once per CTA by the
64-key whole-row producer ``CpasyncGatherKVManagerH64`` (shared with the 64-head forward):
16-B ``cp.async.cg`` copies from 128 threads, the indices of block n+2 loaded while block n
is in flight, completion signalled with ``cp.async.mbarrier.arrive.noinc``. The same gathered
stage is the B operand of both GEMMs: K-major for the gather, re-viewed MN-major (dims
contiguous per key row, same bytes and swizzle) for the MMAs. dS is TMA-loaded per 64-key
block. The epilogue drains the accumulators through dedicated smem tiles and TMA stores, so
the next token's gather never waits for it.
"""

from functools import partial
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisorV2
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from flash_attn.cute.pack_gqa import sparse_mla_qhead_tile
from flash_attn.cute.topk_gather_kv import CpasyncGatherKVManagerH64
from flash_attn.cute.utils import get_batch_from_cu_tensor


class dQdQvGemmKernelH64:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        nheads: int,
        head_dim_k: Optional[int],
        head_dim_v: int,
        top_k: int,
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.nheads = nheads
        # 64-row tile == the head count: no padded rows (pack_gqa.sparse_mla_qhead_tile).
        self.tile_m = sparse_mla_qhead_tile(nheads, min_tile=64)
        assert self.tile_m == 64 and nheads == 64, (
            f"dQdQvGemmKernelH64 serves exactly 64 Q heads, got {nheads}"
        )
        self.head_dim_k = head_dim_k or 0  # when head_dim_k not provided, dQ is not computed
        self.head_dim_v = head_dim_v
        self.top_k = top_k
        # keys per gathered stage / MMA k-tile (whole rows of 64 keys, E1 producer rule)
        self.tile_k = 64
        assert self.top_k % (2 * self.tile_k) == 0, "top_k must be a multiple of 128"
        assert self.head_dim_v % 256 == 0 and self.head_dim_v // 256 == 2, (
            "dQv is accumulated as two N=256 tiles (hdim_v == 512)"
        )
        assert self.head_dim_k in (0, 64), "dQ tile is one N=64 MMA"

        self.cluster_shape_mn = (1, 1)
        self.hdimv_ntile = self.head_dim_v // 2
        self.num_hdimv_ntiles = 2
        self.mma_tiler_dQ = (self.tile_m, self.head_dim_k, self.tile_k)
        self.mma_tiler_dQv = (self.tile_m, self.hdimv_ntile, self.tile_k)
        self.num_mainloop_iters = self.top_k // self.tile_k
        self.arch = "sm_100"

        self.cta_group = tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.threads_per_warp = cute.arch.WARP_SIZE

        # ---- Set specialized warp ids ----
        self.epilogue_warp_ids = (0, 1, 2, 3)
        self.kv_load_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_ids,
                *self.kv_load_warp_ids,
            )
        )
        self.num_kv_load_threads = 32 * len(self.kv_load_warp_ids)
        # ---- Set barrier id for cta sync, epilogue sync and tmem ptr sync ----
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=self.threads_per_warp * len(self.epilogue_warp_ids),
        )

        self.is_persistent = False

        # ---- pipeline stages ----
        # 2 whole-row KV stages (72 KiB each): a third does not fit beside the dS ring under
        # the 232,448-B cap (3 x 73,728 + 2 x 8,192 = 237,568).
        self.num_stages_dS = 4
        self.num_stages_KV = 2
        self.num_stages_acc = 1
        self.num_stages_epi = 2  # dQv epilogue smem tiles (dQ has one)
        self.num_stages_clc = 1

        # ---- register allocation (honoured with min_blocks_per_mp=1 at launch) ----
        self.num_regs_KV = 224
        self.num_regs_epi = 128
        self.num_regs_other = 112

    @cute.jit
    def __call__(
        self,
        mdS: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        mdQ: cute.Tensor,
        mdQv: cute.Tensor,
        mIdxTopK: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.compute_dQ = const_expr(mK is not None)

        # ---- dtype info ----
        self.ds_dtype: Type[cutlass.Numeric] = mdS.element_type
        self.kv_dtype: Type[cutlass.Numeric] = mV.element_type
        self.dq_dtype: Type[cutlass.Numeric] = mdQv.element_type
        if const_expr(self.compute_dQ):
            assert self.kv_dtype == mK.element_type
            assert self.dq_dtype == mdQ.element_type

        varlen_q = const_expr(mCuSeqlensQ is not None)
        varlen_k = const_expr(mCuSeqlensK is not None)

        # ------------------------------------------------------------------ #
        # Reshape GMEM layouts for static strides                            #
        # ------------------------------------------------------------------ #
        seqlen_q = Int32(0) if const_expr(varlen_q) else mdS.shape[1]
        seqlen_q_divmod = FastDivmodDivisorV2(seqlen_q)
        seqlen_k = Int32(0) if const_expr(varlen_k) else mV.shape[1]

        # ---- group batch and seqlen modes in nonvarlen case ----
        def group_batch_seqlen(t: cute.Tensor, varlen: bool) -> cute.Tensor:
            if const_expr(not varlen):
                t = cute.make_tensor(
                    t.iterator,
                    cute.make_layout(
                        (t.shape[1], t.shape[0], *t.shape[2:]),
                        stride=(t.stride[1], t.stride[0], *t.stride[2:]),
                    ),
                )
                t = cute.group_modes(t, 0, 2)
            return t

        mdS = group_batch_seqlen(mdS, varlen_q)
        mdQv = group_batch_seqlen(mdQv, varlen_q)
        mV = group_batch_seqlen(mV, varlen_k)
        mIdxTopK = group_batch_seqlen(mIdxTopK, varlen_q)
        if const_expr(self.compute_dQ):
            mdQ = group_batch_seqlen(mdQ, varlen_q)
            mK = group_batch_seqlen(mK, varlen_k)

        # ---- transpose and make static modes static ----
        def static_reshape(t: cute.Tensor, *static_shapes) -> cute.Tensor:
            static_modes = range(1, len(t.shape))
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (*static_shapes, t.shape[0]),
                    stride=(*(t.stride[i] for i in static_modes), t.stride[0]),
                ),
            )

        # (tokens, head_dim) with a static head dim: the whole-row gather addresses rows.
        def rows_static_hdim(t: cute.Tensor, head_dim: int) -> cute.Tensor:
            return cute.make_tensor(
                t.iterator,
                cute.make_layout((t.shape[0], head_dim), stride=(t.stride[0], t.stride[1])),
            )

        mdS = static_reshape(mdS, self.nheads, self.top_k)
        mdQv = static_reshape(mdQv, self.nheads, self.head_dim_v)
        mV = rows_static_hdim(mV, self.head_dim_v)
        mIdxTopK = static_reshape(mIdxTopK, self.top_k)
        if const_expr(self.compute_dQ):
            mdQ = static_reshape(mdQ, self.nheads, self.head_dim_k)
            mK = rows_static_hdim(mK, self.head_dim_k)

        # ---- layout info ----
        self.ds_major_mode = utils.LayoutEnum.from_tensor(mdS).mma_major_mode()
        assert self.ds_major_mode == tcgen05.OperandMajorMode.K
        # gathered rows are dims-contiguous: MN-major B of both GEMMs
        self.kv_major_mode = tcgen05.OperandMajorMode.MN
        self.dq_layout = utils.LayoutEnum.from_tensor(mdQv)
        if const_expr(self.compute_dQ):
            assert self.dq_layout == utils.LayoutEnum.from_tensor(mdQ)

        # ------------------------------------------------------------------ #
        # Setup attributes that depend on kernel inputs                      #
        # ------------------------------------------------------------------ #
        tiled_mma_v = utils.sm100.make_trivial_tiled_mma(
            self.ds_dtype,
            self.ds_dtype,
            self.ds_major_mode,
            self.kv_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dQv[:2],
        )
        tiled_mma_k = None
        if const_expr(self.compute_dQ):
            tiled_mma_k = utils.sm100.make_trivial_tiled_mma(
                self.ds_dtype,
                self.ds_dtype,
                self.ds_major_mode,
                self.kv_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_dQ[:2],
            )
        # K-major (keys x dims) views of the same stage buffers for the gather: the tiled
        # MMA is a dummy (M=64, N=64 keys, K=dims, K-major B) used only for the layout.
        tiled_mma_gather_v = utils.sm100.make_trivial_tiled_mma(
            self.kv_dtype,
            self.kv_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            (self.tile_m, self.tile_k),
        )

        self.cta_tile_shape_dQv = (self.tile_m, self.head_dim_v, self.tile_k)

        # ---- Compute cluster layout (1,1): one CTA per token ----
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_v.thr_id.shape,),
        )

        # ---- Compute epi tiles for dQ/dQv (per 256-dim N tile) ----
        self.epi_tile_dQv = utils.sm100.compute_epilogue_tile_shape(
            self.mma_tiler_dQv,
            False,  # use_2cta_instrs
            self.dq_layout,
            self.dq_dtype,
        )
        self.epi_tile_dQ = None
        if const_expr(self.compute_dQ):
            self.epi_tile_dQ = utils.sm100.compute_epilogue_tile_shape(
                self.mma_tiler_dQ,
                False,
                self.dq_layout,
                self.dq_dtype,
            )

        # ---- Device-specific attributes ----
        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        self.num_tmem_alloc_cols = 512

        # ------------------------------------------------------------------ #
        # Make SMEM layouts                                                  #
        # ------------------------------------------------------------------ #
        sdS_layout = utils.sm100.make_smem_layout_a(
            tiled_mma_v,
            self.mma_tiler_dQv,
            self.ds_dtype,
            self.num_stages_dS,
        )
        # gather view: 64 keys x hdim_v, K-major SW128 (what CpasyncGatherKVManagerH64 fills)
        sV_layout = utils.sm100.make_smem_layout_b(
            tiled_mma_gather_v,
            (self.tile_m, self.tile_k, self.head_dim_v),
            self.kv_dtype,
            self.num_stages_KV,
        )
        # MMA view: one 256-dim N tile of the stage, MN-major (dims contiguous per key row).
        # Same bytes and swizzle as the gather view: N tile 1 starts 256 x 64 elements in.
        sVt_layout = utils.sm100.make_smem_layout_b(
            tiled_mma_v,
            self.mma_tiler_dQv,
            self.kv_dtype,
            self.num_stages_KV,
        )
        sV_stage_elems = cute.cosize(cute.select(sV_layout, mode=[0, 1, 2]))
        self.sVt_ntile_offset = self.hdimv_ntile * self.tile_k
        assert cute.cosize(cute.select(sVt_layout, mode=[0, 1, 2])) == self.sVt_ntile_offset
        assert self.num_hdimv_ntiles * self.sVt_ntile_offset == sV_stage_elems
        assert sV_layout.inner == sVt_layout.inner, "gather / MMA views must share the swizzle"
        # the N-tile view's stages must step by the whole latent stage, not by one N tile
        sVt_layout_outer = cute.append(
            cute.select(sVt_layout.outer, mode=[0, 1, 2]),
            cute.make_layout(self.num_stages_KV, stride=sV_stage_elems),
        )
        sdQv_layout = utils.sm100.make_smem_layout_epi(
            self.dq_dtype,
            self.dq_layout,
            self.epi_tile_dQv,
            self.num_stages_epi,
        )
        sK_layout, sKt_layout, sdQ_layout = None, None, None
        if const_expr(self.compute_dQ):
            sK_layout = utils.sm100.make_smem_layout_b(
                tiled_mma_gather_v,
                (self.tile_m, self.tile_k, self.head_dim_k),
                self.kv_dtype,
                self.num_stages_KV,
            )
            sKt_layout = utils.sm100.make_smem_layout_b(
                tiled_mma_k,
                self.mma_tiler_dQ,
                self.kv_dtype,
                self.num_stages_KV,
            )
            assert cute.cosize(sK_layout) == cute.cosize(sKt_layout)
            assert cute.cosize(cute.select(sK_layout, mode=[0, 1, 2])) == cute.cosize(
                cute.select(sKt_layout, mode=[0, 1, 2])
            )
            assert sK_layout.inner == sKt_layout.inner
            sdQ_layout = utils.sm100.make_smem_layout_epi(
                self.dq_dtype,
                self.dq_layout,
                self.epi_tile_dQ,
                1,
            )

        # ------------------------------------------------------------------ #
        # Set up TMA load/stores                                             #
        # ------------------------------------------------------------------ #
        # ---- Setup TMA load for dS (no multicast: one CTA per token) ----
        dS_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma_v.thr_id)
        dS_smem_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        tma_atom_dS, tma_tensor_dS = cute.nvgpu.make_tiled_tma_atom_A(
            dS_op,
            mdS,
            dS_smem_layout,
            self.mma_tiler_dQv,
            tiled_mma_v,
            self.cluster_layout_vmnk.shape,
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.ds_dtype, dS_smem_layout)

        # ---- Setup TMA store for dQ and dQV ----
        dQv_epi_smem_layout = cute.select(sdQv_layout, mode=[0, 1])
        tma_atom_dQv, tma_tensor_dQv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mdQv, dQv_epi_smem_layout, self.epi_tile_dQv
        )
        tma_atom_dQ, tma_tensor_dQ = None, None
        if const_expr(self.compute_dQ):
            dQ_epi_smem_layout = cute.select(sdQ_layout, mode=[0, 1])
            tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mdQ, dQ_epi_smem_layout, self.epi_tile_dQ
            )

        # ------------------------------------------------------------------ #
        # Set up shared storage for SMEM                                     #
        # ------------------------------------------------------------------ #

        self.buffer_align_bytes = 1024

        sdS_size = cute.cosize(sdS_layout)
        sK_size = cute.cosize(sK_layout) if const_expr(self.compute_dQ) else 0
        sV_size = cute.cosize(sV_layout)
        sdQ_size = cute.cosize(sdQ_layout) if const_expr(self.compute_dQ) else 0
        sdQv_size = cute.cosize(sdQv_layout)

        @cute.struct
        class SharedStorage:
            mbar_ptr_dS: cute.struct.MemRange[cutlass.Int64, self.num_stages_dS * 2]
            mbar_ptr_KV: cute.struct.MemRange[cutlass.Int64, self.num_stages_KV * 2]
            mbar_ptr_dQ_dQv: cute.struct.MemRange[cutlass.Int64, self.num_stages_acc * 2]
            # Tmem holding buffer
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # Clc pointers
            clc_ptr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.num_stages_clc * 2], 16
            ]
            clc_response_ptr: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            # Smem tensors
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, sdS_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, sK_size],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.kv_dtype, sV_size],
                self.buffer_align_bytes,
            ]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[self.dq_dtype, sdQ_size],
                self.buffer_align_bytes,
            ]
            sdQv: cute.struct.Align[
                cute.struct.MemRange[self.dq_dtype, sdQv_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        assert self.shared_storage.size_in_bytes() <= self.smem_capacity, (
            f"smem {self.shared_storage.size_in_bytes()} > {self.smem_capacity}"
        )

        # ---- Compute grid size: (1, 1, tokens) ----
        self.tile_sched_params, grid = self._compute_grid(
            mdQv,
            self.cta_tile_shape_dQv,
            self.cluster_shape_mn,
        )
        self.num_clc_response_bytes = 16
        # permute grid to conform to grid_dim_z <= 65536 constraint;
        # this is undone in the kernel
        grid = (grid[2], grid[1], grid[0])

        # ---- Launch the kernel synchronously ----
        self.kernel(
            tiled_mma_k,
            tiled_mma_v,
            tma_atom_dS,
            tma_tensor_dS,
            mK,
            mV,
            tma_atom_dQ,
            tma_tensor_dQ,
            tma_atom_dQv,
            tma_tensor_dQv,
            mIdxTopK,
            mCuSeqlensQ,
            mCuSeqlensK,
            seqlen_q_divmod,
            self.cluster_layout_vmnk,
            sdS_layout,
            sK_layout,
            sKt_layout,
            sV_layout,
            sVt_layout,
            sVt_layout_outer,
            sdQ_layout,
            sdQv_layout,
            self.epi_tile_dQ,
            self.epi_tile_dQv,
            self.tile_sched_params,
            seqlen_k,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma_k: Optional[cute.TiledMma],
        tiled_mma_v: cute.TiledMma,
        tma_atom_dS: cute.CopyAtom,
        mdS: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        tma_atom_dQ: Optional[cute.CopyAtom],
        mdQ: Optional[cute.Tensor],
        tma_atom_dQv: cute.CopyAtom,
        mdQv: cute.Tensor,
        mIdxTopK: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        seqlen_q_divmod: FastDivmodDivisorV2,
        cluster_layout_vmnk: cute.Layout,
        sdS_layout: cute.ComposedLayout,
        sK_layout: Optional[cute.ComposedLayout],
        sKt_layout: Optional[cute.ComposedLayout],
        sV_layout: cute.ComposedLayout,
        sVt_layout: cute.ComposedLayout,
        sVt_layout_outer: cute.Layout,
        sdQ_layout: Optional[cute.ComposedLayout],
        sdQv_layout: cute.ComposedLayout,
        epi_tile_dQ: Optional[cute.Tile],
        epi_tile_dQv: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        seqlen_k_static: Int32,
    ):
        """
        GPU device kernel performing the persistent per-token gather GEMMs.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # ------------------------------------------------------------------ #
        # Prefetch TMA descriptors                                           #
        # ------------------------------------------------------------------ #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_dS)
            cpasync.prefetch_descriptor(tma_atom_dQv)
            if const_expr(self.compute_dQ):
                cpasync.prefetch_descriptor(tma_atom_dQ)

        tidx, _, _ = cute.arch.thread_idx()

        # ------------------------------------------------------------------ #
        # Shared storage allocation                                          #
        # ------------------------------------------------------------------ #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # ------------------------------------------------------------------ #
        # Initialize pipelines                                               #
        # ------------------------------------------------------------------ #
        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        clc_producer_group = ThreadCooperativeGroup(1)
        num_clc_consumer_threads = 32 * (
            1  # sched warp
            + (1 + len(self.epilogue_warp_ids) + len(self.kv_load_warp_ids) + 1)
            # tma + epi + kv_load + mma
        )
        clc_consumer_group = ThreadCooperativeGroup(num_clc_consumer_threads)
        mma_warp = ThreadCooperativeGroup(1)
        tma_warp = ThreadCooperativeGroup(1)
        epilogue_warps = ThreadCooperativeGroup(len(self.epilogue_warp_ids))
        gather_threads = ThreadCooperativeGroup(self.num_kv_load_threads)

        pipeline_dS = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_ptr_dS.data_ptr(),
            num_stages=self.num_stages_dS,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # the 128 gather threads arrive on the full barrier with cp.async.mbarrier.arrive.noinc
        pipeline_KV = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_ptr_KV.data_ptr(),
            num_stages=self.num_stages_KV,
            producer_group=gather_threads,
            consumer_group=mma_warp,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_dQ_dQv = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_ptr_dQ_dQv.data_ptr(),
            num_stages=self.num_stages_acc,
            producer_group=mma_warp,
            consumer_group=epilogue_warps,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_clc = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_ptr.data_ptr(),
            num_stages=self.num_stages_clc,
            producer_group=clc_producer_group,
            consumer_group=clc_consumer_group,
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # ------------------------------------------------------------------ #
        # TMEM Allocation                                                    #
        # ------------------------------------------------------------------ #
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_ids)),
        )
        # ---- Tensor memory dealloc barrier init ----
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_ids[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # ---- Cluster arrive after barrier init ----
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # ---- Initial clc response pointer ----
        clc_response_ptr = storage.clc_response_ptr.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages_clc
        )

        # ------------------------------------------------------------------ #
        # SMEM tensors                                                       #
        # ------------------------------------------------------------------ #
        # (MMA, MMA_M, MMA_K, STAGE)
        sdS = storage.sdS.get_tensor(sdS_layout.outer, swizzle=sdS_layout.inner)
        # gather views (MMA, MMA_N=keys, MMA_K=dims, STAGE), K-major
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        # MMA views of the two 256-dim N tiles: same bytes, MN-major, stages stepped by the
        # whole latent stage
        sVt0 = cute.make_tensor(
            cute.recast_ptr(sV.iterator, sVt_layout.inner, self.kv_dtype), sVt_layout_outer
        )
        sVt1 = cute.make_tensor(
            cute.recast_ptr(sV.iterator + self.sVt_ntile_offset, sVt_layout.inner, self.kv_dtype),
            sVt_layout_outer,
        )
        sK, sKt, sdQ = None, None, None
        if const_expr(self.compute_dQ):
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sKt = cute.make_tensor(
                cute.recast_ptr(sK.iterator, sKt_layout.inner, self.kv_dtype), sKt_layout.outer
            )
            sdQ = storage.sdQ.get_tensor(sdQ_layout.outer, swizzle=sdQ_layout.inner)
        sdQv = storage.sdQv.get_tensor(sdQv_layout.outer, swizzle=sdQv_layout.inner)

        # ------------------------------------------------------------------ #
        # Global tile partitioning                                           #
        # ------------------------------------------------------------------ #
        # (bM, bK, RestM, RestK, RestL)
        gdS = cute.local_tile(
            mdS, cute.slice_(self.mma_tiler_dQv, (None, 0, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gdQ = None
        if const_expr(self.compute_dQ):
            gdQ = cute.local_tile(
                mdQ, cute.slice_(self.mma_tiler_dQ, (None, None, 0)), (None, None, None)
            )
        gdQv = cute.local_tile(
            mdQv, cute.slice_(self.mma_tiler_dQv, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gdS, mode=[3])

        # ------------------------------------------------------------------ #
        # TiledMMA partitioning                                              #
        # ------------------------------------------------------------------ #
        thr_mma_v = tiled_mma_v.get_slice(0)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tdQvgdS = thr_mma_v.partition_A(gdS)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tdQvgdQv = thr_mma_v.partition_C(gdQv)
        tdQgdQ = None
        if const_expr(self.compute_dQ):
            thr_mma_k = tiled_mma_k.get_slice(0)
            tdQgdQ = thr_mma_k.partition_C(gdQ)

        # ------------------------------------------------------------------ #
        # TMA partition for dS                                               #
        # ------------------------------------------------------------------ #
        dS_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tdSsdS, tdSgdS = cpasync.tma_partition(
            tma_atom_dS,
            0,
            dS_cta_layout,
            cute.group_modes(sdS, 0, 3),
            cute.group_modes(tdQvgdS, 0, 3),
        )

        # ------------------------------------------------------------------ #
        # MMA fragments                                                      #
        # ------------------------------------------------------------------ #
        # (MMA, MMA_M, MMA_K, STAGE)
        tdQvrdS = tiled_mma_v.make_fragment_A(sdS)
        # (MMA, MMA_N, MMA_K, STAGE) per 256-dim N tile
        tdQvrV0 = tiled_mma_v.make_fragment_B(sVt0)
        tdQvrV1 = tiled_mma_v.make_fragment_B(sVt1)
        # (MMA, MMA_M, MMA_N)
        acc_v_shape = tiled_mma_v.partition_shape_C(self.mma_tiler_dQv[:2])
        # (MMA, MMA_M, MMA_N, STAGE): one N tile; tile 1 sits in the other TMEM lane half
        tdQvtAcc_fake = tiled_mma_v.make_fragment_C(cute.append(acc_v_shape, self.num_stages_acc))
        tdQrK, tdQtAcc_fake = None, None
        if const_expr(self.compute_dQ):
            # (MMA, MMA_N, MMA_K, STAGE)
            tdQrK = tiled_mma_k.make_fragment_B(sKt)
            # (MMA, MMA_M, MMA_N)
            acc_k_shape = tiled_mma_k.partition_shape_C(self.mma_tiler_dQ[:2])
            # (MMA, MMA_M, MMA_N, STAGE)
            tdQtAcc_fake = tiled_mma_k.make_fragment_C(
                cute.append(acc_k_shape, self.num_stages_acc)
            )
        # TMEM column plan: dQ [0, 64) then dQv N tile 0 / 1 (lane halves) [64, 320)
        tmem_cols_dQ = (
            tcgen05.find_tmem_tensor_col_offset(tdQtAcc_fake) if const_expr(self.compute_dQ) else 0
        )
        tmem_cols_dQv = tcgen05.find_tmem_tensor_col_offset(tdQvtAcc_fake)
        assert tmem_cols_dQ + tmem_cols_dQv <= self.num_tmem_alloc_cols
        # the second 256-dim N tile: same columns, lanes 16-31 of each 32-lane subpartition
        # (the M=64 accumulator occupies lanes 0-15; TMEM address lane field = bits [31:16])
        tmem_lane_half_offset = 16 << 16

        # ------------------------------------------------------------------ #
        # Cluster wait before tensor memory alloc                            #
        # ------------------------------------------------------------------ #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ------------------------------------------------------------------ #
        # Tile Scheduler                                                     #
        # ------------------------------------------------------------------ #
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        # ------------------------------------------------------------------ #
        # TMA load warp: dS, one 64 x 64 tile per k-tile                      #
        # ------------------------------------------------------------------ #
        if warp_idx == self.tma_warp_id:
            producer_state_dS = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_dS
            )

            while work_tile.is_valid_tile:
                # ---- Get tile coord from tile scheduler ----
                token, _, _ = work_tile.tile_idx

                # ((atom_v, rest_v), RestK)
                tdSgdS_slice = tdSgdS[(None, 0, None, token)]

                # ---- mainloop ----
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    pipeline_dS.producer_acquire(producer_state_dS)
                    index_dS = producer_state_dS.index

                    # ---- TMA load dS ----
                    cute.copy(
                        tma_atom_dS,
                        tdSgdS_slice[(None, k_tile)],
                        tdSsdS[(None, index_dS)],
                        tma_bar_ptr=pipeline_dS.producer_get_barrier(producer_state_dS),
                    )

                    producer_state_dS.advance()

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            # ---- Wait dS buffer empty ----
            pipeline_dS.producer_tail(producer_state_dS)

        # ------------------------------------------------------------------ #
        # Clc Scheduler warp                                                 #
        # ------------------------------------------------------------------ #
        if warp_idx == self.sched_warp_id:
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_stages_clc
            )

            while work_tile.is_valid_tile:
                pipeline_clc.producer_acquire(clc_producer_state)
                mbar_addr = pipeline_clc.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbar_addr)
                clc_producer_state.advance()

                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            pipeline_clc.producer_tail(clc_producer_state)

        # ------------------------------------------------------------------ #
        # CpAsync KV gather warps: whole rows, 64 keys per stage             #
        # ------------------------------------------------------------------ #
        if warp_idx >= self.kv_load_warp_ids[0] and warp_idx <= self.kv_load_warp_ids[-1]:
            find_batch = partial(
                self.find_batch_from_q, seqlen_q_divmod=seqlen_q_divmod, mCuSeqlensQ=mCuSeqlensQ
            )

            producer_state_KV = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, stages=self.num_stages_KV
            )

            kv_tidx = tidx % self.num_kv_load_threads
            kv_warp_idx = warp_idx % len(self.kv_load_warp_ids)

            while work_tile.is_valid_tile:
                # ---- Get tile coord from tile scheduler ----
                token, _, _ = work_tile.tile_idx

                batch_idx = find_batch(token)
                k_batch_offset = (
                    mCuSeqlensK[batch_idx] if const_expr(mCuSeqlensK is not None) else Int32(0)
                )
                seqlen_k = (
                    mCuSeqlensK[batch_idx + 1] - k_batch_offset
                    if const_expr(mCuSeqlensK is not None)
                    else seqlen_k_static
                )
                # (seqlen_k, head_dim) of this batch; rows addressed by the top-k index
                mK_cur = None
                if const_expr(mCuSeqlensK is not None):
                    if const_expr(self.compute_dQ):
                        mK_cur = cute.domain_offset((k_batch_offset, 0), mK)
                    mV_cur = cute.domain_offset((k_batch_offset, 0), mV)
                else:
                    if const_expr(self.compute_dQ):
                        mK_cur = cute.domain_offset(((0, batch_idx), 0), mK)
                    mV_cur = cute.domain_offset(((0, batch_idx), 0), mV)
                mIdxTopK_cur = mIdxTopK[None, token]

                # no causal limit here (out-of-limit slots carry dS = 0); rows with index -1 or
                # >= seqlen_k are zero-filled by the predicated cp.async
                gather = CpasyncGatherKVManagerH64.create(
                    mIdxTopK_cur,
                    kv_tidx,
                    kv_warp_idx,
                    seqlen_k,
                    self.tile_k,
                    self.head_dim_k if const_expr(self.compute_dQ) else 64,
                    self.head_dim_v,
                    self.num_kv_load_threads,
                    mV.element_type,
                    None,
                    False,
                    None,
                    None,
                )

                # ---- K/V gather mainloop: indices two blocks ahead in two register sets ----
                gather.load_index_topk(Int32(0), 0)
                gather.load_index_topk(Int32(1), 1)
                for it in cutlass.range(k_tile_cnt // 2, unroll=1):
                    for buf in cutlass.range_constexpr(2):
                        stage = producer_state_KV.index
                        pipeline_KV.producer_acquire(producer_state_KV)
                        if const_expr(self.compute_dQ):
                            gather.load_X(mK_cur, sK[None, None, None, stage], "K", buf)
                        gather.load_X(mV_cur, sV[None, None, None, stage], "V", buf)
                        cute.arch.cp_async_commit_group()
                        # fires once every copy this thread issued so far has landed
                        pipeline_KV.sync_object_full.arrive_cp_async_mbarrier(stage)
                        producer_state_KV.advance()
                        n_prefetch = 2 * it + buf + 2
                        n_prefetch = (
                            n_prefetch if n_prefetch < k_tile_cnt else Int32(k_tile_cnt - 1)
                        )
                        gather.load_index_topk(n_prefetch, buf)

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            pipeline_KV.producer_tail(producer_state_KV)

        # ------------------------------------------------------------------ #
        # MMA warp                                                           #
        # ------------------------------------------------------------------ #
        if warp_idx == self.mma_warp_id:
            # --- Retrieve TMEM ptr and make accumulator tensors
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tdQtAcc_base = None
            if const_expr(self.compute_dQ):
                tdQtAcc_base = cute.make_tensor(tmem_ptr, tdQtAcc_fake.layout)
            tdQvtAcc0_base = cute.make_tensor(tmem_ptr + tmem_cols_dQ, tdQvtAcc_fake.layout)
            tdQvtAcc1_base = cute.make_tensor(
                tmem_ptr + (tmem_cols_dQ + tmem_lane_half_offset), tdQvtAcc_fake.layout
            )

            consumer_state_dS = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages=self.num_stages_dS
            )
            consumer_state_KV = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, stages=self.num_stages_KV
            )
            producer_state_dQ_dQv = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages_acc
            )

            while work_tile.is_valid_tile:
                # ---- Set tensor memory buffer for current tile ----
                # (MMA, MMA_M, MMA_N)
                acc_idx = producer_state_dQ_dQv.index
                tdQvtAcc0 = tdQvtAcc0_base[(None, None, None, acc_idx)]
                tdQvtAcc1 = tdQvtAcc1_base[(None, None, None, acc_idx)]
                if const_expr(self.compute_dQ):
                    tdQtAcc = tdQtAcc_base[(None, None, None, acc_idx)]

                # ---- Wait for accumulator buffer empty ----
                pipeline_dQ_dQv.producer_acquire(producer_state_dQ_dQv)

                # ---- Reset the ACCUMULATE field for each tile ----
                tiled_mma_v.set(tcgen05.Field.ACCUMULATE, False)
                if const_expr(self.compute_dQ):
                    tiled_mma_k.set(tcgen05.Field.ACCUMULATE, False)

                # ---- Mma mainloop ----
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    pipeline_KV.consumer_wait(consumer_state_KV)
                    dS_stage = consumer_state_dS.index
                    KV_stage = consumer_state_KV.index

                    num_kblocks = cute.size(tdQvrdS, mode=[2])
                    for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                        # dQv[:, 0:256] += dS @ V[:, 0:256]; dQv[:, 256:512] += dS @ V[:, 256:512]
                        cute.gemm(
                            tiled_mma_v,
                            tdQvtAcc0,
                            tdQvrdS[(None, None, kblk_idx, dS_stage)],
                            tdQvrV0[(None, None, kblk_idx, KV_stage)],
                            tdQvtAcc0,
                        )
                        cute.gemm(
                            tiled_mma_v,
                            tdQvtAcc1,
                            tdQvrdS[(None, None, kblk_idx, dS_stage)],
                            tdQvrV1[(None, None, kblk_idx, KV_stage)],
                            tdQvtAcc1,
                        )
                        # Enable accumulate on both dQv tiles after the first kblock
                        tiled_mma_v.set(tcgen05.Field.ACCUMULATE, True)

                        if const_expr(self.compute_dQ):
                            # dQ += dS @ K
                            cute.gemm(
                                tiled_mma_k,
                                tdQtAcc,
                                tdQvrdS[(None, None, kblk_idx, dS_stage)],
                                tdQrK[(None, None, kblk_idx, KV_stage)],
                                tdQtAcc,
                            )
                            # Enable accumulate on tdQtAcc after first kblock
                            tiled_mma_k.set(tcgen05.Field.ACCUMULATE, True)

                    pipeline_dS.consumer_release(consumer_state_dS)
                    pipeline_KV.consumer_release(consumer_state_KV)
                    consumer_state_dS.advance()
                    consumer_state_KV.advance()

                pipeline_dQ_dQv.producer_commit(producer_state_dQ_dQv)
                producer_state_dQ_dQv.advance()

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            pipeline_dQ_dQv.producer_tail(producer_state_dQ_dQv)

        # ------------------------------------------------------------------ #
        # Epilogue warps                                                     #
        # ------------------------------------------------------------------ #
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            # ---- Alloc tensor memory buffer ----
            tmem.allocate(self.num_tmem_alloc_cols)

            # ---- Retrieving tensor memory ptr and make accumulator tensor ----
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tdQtAcc_base = None
            if const_expr(self.compute_dQ):
                tdQtAcc_base = cute.make_tensor(tmem_ptr, tdQtAcc_fake.layout)
            tdQvtAcc0_base = cute.make_tensor(tmem_ptr + tmem_cols_dQ, tdQvtAcc_fake.layout)
            tdQvtAcc1_base = cute.make_tensor(
                tmem_ptr + (tmem_cols_dQ + tmem_lane_half_offset), tdQvtAcc_fake.layout
            )

            epi_idx = tidx
            # ---- TMEM -> RMEM -> SMEM -> GMEM copies + partitions ----
            tiled_copy_dQv_t2r, tTR_dQvtAcc0_base, tTR_dQvrAcc = (
                self.epilogue_tmem_copy_and_partition(
                    epi_idx,
                    tdQvtAcc0_base,
                    tdQvgdQv,
                    epi_tile_dQv,
                    self.mma_tiler_dQv,
                    self.dq_layout,
                    self.dq_dtype,
                )
            )
            _, tTR_dQvtAcc1_base, _ = self.epilogue_tmem_copy_and_partition(
                epi_idx,
                tdQvtAcc1_base,
                tdQvgdQv,
                epi_tile_dQv,
                self.mma_tiler_dQv,
                self.dq_layout,
                self.dq_dtype,
            )
            tTR_rdQv = cute.make_rmem_tensor(tTR_dQvrAcc.shape, self.dq_dtype)
            (
                tiled_copy_dQv_r2s,
                tRS_rdQv,
                tRS_sdQv,
            ) = self.epilogue_smem_copy_and_partition(
                self.dq_layout,
                self.dq_dtype,
                tiled_copy_dQv_t2r,
                tTR_rdQv,
                epi_idx,
                sdQv,
            )
            bSG_sdQv, bSG_gdQv_partitioned = self.epilogue_gmem_copy_and_partition(
                tma_atom_dQv,
                tdQvgdQv,
                epi_tile_dQv,
                sdQv,
            )
            if const_expr(self.compute_dQ):
                (tiled_copy_dQ_t2r, tTR_dQtAcc_base, tTR_dQrAcc) = (
                    self.epilogue_tmem_copy_and_partition(
                        epi_idx,
                        tdQtAcc_base,
                        tdQgdQ,
                        epi_tile_dQ,
                        self.mma_tiler_dQ,
                        self.dq_layout,
                        self.dq_dtype,
                    )
                )

                tTR_rdQ = cute.make_rmem_tensor(tTR_dQrAcc.shape, self.dq_dtype)
                (
                    tiled_copy_dQ_r2s,
                    tRS_rdQ,
                    tRS_sdQ,
                ) = self.epilogue_smem_copy_and_partition(
                    self.dq_layout,
                    self.dq_dtype,
                    tiled_copy_dQ_t2r,
                    tTR_rdQ,
                    epi_idx,
                    sdQ,
                )
                bSG_sdQ, bSG_gdQ_partitioned = self.epilogue_gmem_copy_and_partition(
                    tma_atom_dQ,
                    tdQgdQ,
                    epi_tile_dQ,
                    sdQ,
                )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages_acc
            )

            epi_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                cute.arch.WARP_SIZE,
            )
            pipeline_epi = pipeline.PipelineTmaStore.create(
                num_stages=self.num_stages_epi,
                producer_group=epi_producer_group,
            )

            # ---- Persistent tile scheduling loop for epilogue ----
            while work_tile.is_valid_tile:
                # ---- Get current work tile ----
                token, _, _ = work_tile.tile_idx

                acc_idx = acc_consumer_state.index
                # dQv: (T2R, T2R_M, T2R_N, EPI_M*EPI_N) per N tile; gmem tiles per (N tile, token)
                tTR_dQvtAccs = []
                bSG_gdQvs = []
                for n_tile in cutlass.range_constexpr(self.num_hdimv_ntiles):
                    tTR_base = tTR_dQvtAcc0_base if n_tile == 0 else tTR_dQvtAcc1_base
                    t = tTR_base[(None, None, None, None, None, acc_idx)]
                    tTR_dQvtAccs.append(cute.group_modes(t, 3, cute.rank(t)))
                    g = bSG_gdQv_partitioned[(None, None, None, 0, n_tile, token)]
                    bSG_gdQvs.append(cute.group_modes(g, 1, cute.rank(g)))
                subtile_cnt_v = cute.size(tTR_dQvtAccs[0].shape, mode=[3])
                epi_subtile_counter = 0

                if const_expr(self.compute_dQ):
                    bSG_gdQ = bSG_gdQ_partitioned[(None, None, None, 0, 0, token)]
                    tTR_dQtAcc = tTR_dQtAcc_base[(None, None, None, None, None, acc_idx)]
                    tTR_dQtAcc = cute.group_modes(tTR_dQtAcc, 3, cute.rank(tTR_dQtAcc))
                    bSG_gdQ = cute.group_modes(bSG_gdQ, 1, cute.rank(bSG_gdQ))
                    subtile_cnt_k = cute.size(tTR_dQtAcc.shape, mode=[3])

                pipeline_dQ_dQv.consumer_wait(acc_consumer_state)

                # ---- dQ: its own smem tile (one subtile of 64 x 64) ----
                if const_expr(self.compute_dQ):
                    for subtile_idx in cutlass.range(subtile_cnt_k, unroll_full=True):
                        tTR_dQtAcc_mn = tTR_dQtAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_dQ_t2r, tTR_dQtAcc_mn, tTR_dQrAcc)
                        cute.arch.fence_view_async_tmem_load()
                        tRS_rdQ.store(tiled_copy_dQ_r2s.retile(tTR_dQrAcc).load().to(self.dq_dtype))
                        if warp_idx == self.epilogue_warp_ids[0]:
                            # the dQ tile of the previous token must have left smem
                            pipeline_epi.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(tiled_copy_dQ_r2s, tRS_rdQ, tRS_sdQ[(None, None, None, 0)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilogue_warp_ids[0]:
                            cute.copy(tma_atom_dQ, bSG_sdQ[(None, 0)], bSG_gdQ[(None, subtile_idx)])
                            pipeline_epi.producer_commit()

                # ---- dQv: two N tiles x subtiles through the 2-stage smem ring ----
                for n_tile in cutlass.range_constexpr(self.num_hdimv_ntiles):
                    for subtile_idx in cutlass.range(subtile_cnt_v, unroll_full=True):
                        tTR_dQvtAcc_mn = tTR_dQvtAccs[n_tile][(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_dQv_t2r, tTR_dQvtAcc_mn, tTR_dQvrAcc)
                        cute.arch.fence_view_async_tmem_load()
                        # convert to output dtype
                        tRS_rdQv.store(
                            tiled_copy_dQv_r2s.retile(tTR_dQvrAcc).load().to(self.dq_dtype)
                        )
                        epi_buffer = epi_subtile_counter % self.num_stages_epi
                        if warp_idx == self.epilogue_warp_ids[0]:
                            # the TMA store that read this buffer two subtiles ago is done
                            pipeline_epi.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_dQv_r2s, tRS_rdQv, tRS_sdQv[(None, None, None, epi_buffer)]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilogue_warp_ids[0]:
                            cute.copy(
                                tma_atom_dQv,
                                bSG_sdQv[(None, epi_buffer)],
                                bSG_gdQvs[n_tile][(None, subtile_idx)],
                            )
                            pipeline_epi.producer_commit()
                        epi_subtile_counter += 1

                with cute.arch.elect_one():
                    pipeline_dQ_dQv.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # ---- Advance to next tile ----
                pipeline_clc.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                pipeline_clc.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            # ---- Dealloc the tensor memory buffer ----
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            pipeline_epi.producer_tail()

    def epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        mma_tiler_mnk,
        c_layout,
        c_dtype,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = utils.sm100.get_tmem_load_op(
            mma_tiler_mnk,
            c_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs=False,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)

        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        rAcc_shape = tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape
        tTR_rAcc = cute.make_rmem_tensor(rAcc_shape, self.acc_dtype)

        return (tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

    def epilogue_smem_copy_and_partition(
        self,
        c_layout,
        c_dtype: Type[cutlass.Numeric],
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = utils.sm100.get_smem_store_op(
            c_layout, c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilogue_gmem_copy_and_partition(
        self,
        tma_atom,
        gC,
        epi_tile,
        sC,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(gC[((None, None), 0, 0, None, None, None)], epi_tile)

        sC_for_tma_partition = cute.group_modes(sC, 0, 2)

        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)

        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )

        return bSG_sC, bSG_gC

    @cute.jit
    def find_batch_from_q(
        self,
        token: Int32,
        seqlen_q_divmod: FastDivmodDivisorV2,
        mCuSeqlensQ: Optional[cute.Tensor],
    ) -> Int32:
        """Find batch index from q token (binary search for varlen, divmod otherwise)"""
        if const_expr(mCuSeqlensQ is not None):
            return get_batch_from_cu_tensor(token, mCuSeqlensQ)
        else:
            batch, _ = divmod(token, seqlen_q_divmod)
            return batch

    @staticmethod
    def _compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn):
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        c_logical_shape = gc[(0, (None, None, None))].shape

        num_ctas_mnl = (
            cute.size(c_logical_shape[0]),
            cute.size(c_logical_shape[1]),
            cute.size(c_logical_shape[2]),
        )

        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl, (*cluster_shape_mn, 1)
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        return tile_sched_params, grid
