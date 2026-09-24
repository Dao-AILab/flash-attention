# Copyright (c) 2026, Colfax International.
"""Native 64-head sparse-MLA (top-k gather) forward for Blackwell, one token per CTA.

The 2-CTA sparse-MLA forward (flash_fwd_mla_sm100.py) pads 64 Q heads to its 128-row tile split
across a CTA pair and gathers every top-k latent row twice. This kernel puts one token's 64 heads
on M = 64 of ``tcgen05.mma.ws`` UMMAs and gathers each top-k key once (its 512-dim latent row and
64-dim rope row; 64 keys per stage, three latent stages, one rope tile, each stage landing in
parts with their own barriers so S streams behind the fill). Q lives in TMEM: the gather stages
the token's Q tile through the KV ring once per tile and the MMA warp copies it to TMEM with
``tcgen05.cp`` (blackwell_helpers.utccp_128x256b_ptx); S = Q K^T is then the ``.ws`` TS "dual
GEMM" (M 64 x N 128: the two lane halves of the accumulator hold the two dim halves' partial sums,
summed by the softmax warps; blackwell_helpers.gemm_ws_ts_ptx_partial), which frees the 72 KiB of
Q shared memory for the third latent stage. O += P V re-views the latent tile MN-major (SS). The
MMA warp issues S(n) before P(n-1) V(n-1), so the stage refill, the softmax step and the GEMMs
overlap. It serves the forwards that store no P / row_max (recompute-P training and inference).
Design, measurements and the test contract: AI/SPARSE_MLA_64H.md.
"""

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Uint32, Boolean, const_expr
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import ClcDynamicPersistentTileScheduler

from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
import flash_attn.cute.blackwell_helpers as fa_sm100_utils
from flash_attn.cute.softmax import SoftmaxSm100, apply_learnable_sink, load_learnable_sink
from flash_attn.cute.tile_scheduler import (
    SchedulerState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)
from flash_attn.cute.fa_logging import fa_log, fa_printf
from flash_attn.cute.utils import smid, get_batch_from_cu_tensor
from flash_attn.cute.topk_gather_kv import CpasyncGatherKVManagerH64
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100_MLA2CTA
from cutlass._mlir.dialects import llvm


@cute.jit
def pair_barrier_sync(barrier_id: Int32, num_threads: cutlass.Constexpr[int]) -> None:
    """`bar.sync id, n` with a run-time (warp-uniform) barrier id: the two softmax warps that hold
    the two lane halves of the same rows synchronize pairwise instead of all four warps."""
    llvm.inline_asm(
        None,
        [Int32(barrier_id).ir_value()],
        f"bar.sync $0, {num_threads};",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionMLAForwardSm100H64:
    def __init__(
        self,
        is_causal: bool = False,
        use_cpasync_load_KV: bool = True,
        topk_length: int = 2048,
        is_topk_gather: bool = True,
        pack_gqa: bool = True,
        qhead_per_kvhead: int = 64,
        nheads_kv: int = 1,
        hdim: int = 64,
        hdimv: int = 512,
        has_seqused_q: bool = False,
        has_cu_seqlens_q: bool = False,
        disable_bitmask: bool = False,
        use_clc_scheduler: bool = True,
        has_qk: bool = True,
        rescale_threshold: float = 8.0,
        debug_clock_addr: Optional[int] = None,
        debug_epilogue: Optional[str] = None,
    ):
        # Same constructor as FlashAttentionMLAForwardSm100 so the interface can pick either class.
        # debug_clock_addr (probes only, never set by the interface): device address of an
        # int64 (num_n_blocks + 8, 20) buffer that receives %clock64 stamps of CTA 0's first tiles.
        # debug_epilogue (probes only): "skip_olo_store" / "skip_all_stores" keep the epilogue
        # arithmetic but drop the corresponding global stores (wrong output; timing A/B).
        self.debug_clock_addr = debug_clock_addr
        assert debug_epilogue in (None, "skip_olo_store", "skip_all_stores")
        self.debug_epilogue = debug_epilogue
        assert is_topk_gather and use_cpasync_load_KV and pack_gqa, (
            "H64 forward: sparse-MLA top-k gather with cp.async only"
        )
        assert qhead_per_kvhead == 64, "H64 forward: exactly 64 Q heads per KV head (no padding)"
        assert hdim == 64 and hdimv == 512, "H64 forward: 64 rope + 512 latent dims"
        self.is_causal = is_causal
        self.is_local = False
        # Lazy online-softmax rescaling threshold (0.0 = exact running max); see the 2-CTA kernel.
        self.rescale_threshold = float(rescale_threshold)
        self.pack_gqa = True
        self.qhead_per_kvhead = qhead_per_kvhead
        self.qhead_per_kvhead_valid = qhead_per_kvhead
        self.pad_qheads = False
        self.nheads_kv = nheads_kv
        # O is stored thread-wise (16-B stores, like the o_lo residual): no sV alias for a TMA
        # staging tile and no sO_empty hand-off with the gather warps.
        self.use_tma_O = False
        self.topk_length = topk_length
        self.is_topk_gather = True
        self.disable_bitmask = disable_bitmask
        self.has_qk = has_qk

        # ==== tile scheduler ====
        self.is_persistent = False
        self.use_clc_scheduler = use_clc_scheduler
        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )
        self.is_varlen_q = has_seqused_q or has_cu_seqlens_q
        # one token per tile: the packed (head, token) block index is the token index
        self.use_packed_varlen_sched = has_cu_seqlens_q
        self.use_varlen_scheduler = self.is_varlen_q and not self.use_packed_varlen_sched
        if const_expr(self.use_varlen_scheduler):
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        else:
            self.TileScheduler = SingleTileScheduler
        fa_log(
            1,
            f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}",
        )

        # ==== thread info ====
        # warps 0-3 softmax, 4-7 correction/epilogue, 8 idle (Q is staged by the gather warps, no
        # TMA), 9 MMA + TMEM alloc, 10 CLC, 11 idle, 12-15 cp.async gather
        self.num_softmax_threads = 128
        self.num_epilogue_threads = 128
        self.num_load_threads = 32
        self.num_mma_threads = 32
        self.num_empty_threads = 64
        self.num_cpasync_load_threads = 128
        self.num_threads = (
            self.num_softmax_threads
            + self.num_epilogue_threads
            + self.num_load_threads
            + self.num_mma_threads
            + self.num_empty_threads
            + self.num_cpasync_load_threads
        )
        self.num_warps = self.num_threads // 32
        assert self.num_warps == 16
        self.softmax_warp_indices = (0, 1, 2, 3)
        # pairwise softmax barriers (warps 0 + 2, warps 1 + 3): ids after the shared MLA enum (1-6)
        self.pair_barrier_id0 = max(int(b) for b in NamedBarrierFwdSm100_MLA2CTA) + 1
        self.epilogue_warp_indices = (4, 5, 6, 7)
        self.load_warp_id = 8
        self.mma_warp_id = 9
        self.clc_scheduler_warp_id = 10
        self.idle_warp_id = 11
        self.empty_warp_ids = tuple(
            w
            for w, active in [
                (self.load_warp_id, True),
                (self.idle_warp_id, True),
                (self.clc_scheduler_warp_id, not self.use_clc_scheduler),
            ]
            if active
        )
        self.cpasync_load_warp_indices = (12, 13, 14, 15)

        # ==== register usage (honoured only with min_blocks_per_mp=1 at launch) ====
        self.num_regs_load = 112
        self.num_regs_mma = 112
        self.num_regs_softmax = 192
        self.num_regs_epilogue = 128
        self.num_regs_cpasync = 80
        self.num_regs_other = 48
        self.num_regs_per_thread = 128
        self.num_regs_total = 512
        assert (
            self.num_regs_mma
            + self.num_regs_softmax
            + self.num_regs_epilogue
            + self.num_regs_cpasync
            <= self.num_regs_total
        )

        # ==== 1-CTA info ====
        self.use_2cta_instrs = False
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cta_group_size = 1
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (1, 1, 1)

        # ==== problem shape info ====
        self.hdim = hdim
        self.hdimv = hdimv
        self.cta_tile_m = 64
        self.cluster_tile_m = self.cta_tile_m
        self.tile_n = 64
        # two softmax / correction threads per row (lane t and t + 64 hold row t % 64)
        self.threads_per_row = self.num_softmax_threads // self.cta_tile_m
        assert self.threads_per_row == 2
        # even block count: the gather keeps two index register sets (indices two blocks ahead)
        assert self.topk_length % (2 * self.tile_n) == 0 and self.topk_length >= 2 * self.tile_n
        self.num_n_blocks = self.topk_length // self.tile_n
        # O epilogue splits == P.V N-tiles of 256 dims (not the gather geometry)
        self.num_hdimv_splits = 2
        self.hdimv_split = self.hdimv // self.num_hdimv_splits
        self.epi_tile = (self.cta_tile_m, self.hdimv_split)
        self.tile_P = (self.cta_tile_m, self.tile_n)

        # ==== MMA info ====
        # S = Q K^T is the .ws TS "dual GEMM": M = 64 heads, N = 128 = 2 x 64 keys (the two N halves
        # of the instruction see the two dim halves of every 128-dim chunk of the K tile and of Q,
        # so lane half h of the accumulator holds the partial sum over dim half h), K = 16 per
        # instruction: 16 k-blocks over the latent (K = 256 per lane half) + 2 over the rope (32).
        self.n_dual = 2 * self.tile_n
        self.hdimv_dual = self.hdimv // 2
        self.hdim_dual = self.hdim // 2
        self.mma_tiler_stage = (self.cta_tile_m, self.tile_n, self.hdimv)  # physical latent stage
        self.mma_tiler_rope = (self.cta_tile_m, self.tile_n, self.hdim_dual)  # rope tile, 2 halves
        self.mma_tiler_Sd = (
            self.cta_tile_m,
            self.n_dual,
            self.hdimv_dual,
        )  # (128, 256) latent view
        self.mma_tiler_Srd = (self.cta_tile_m, self.n_dual, self.hdim_dual)  # (128, 32) rope view
        self.mma_tiler_PV = (self.cta_tile_m, self.hdimv_split, self.tile_n)
        self.major_mode_K = cute.nvgpu.OperandMajorMode.K
        self.major_mode_V = cute.nvgpu.OperandMajorMode.K
        self.major_mode_Vt = cute.nvgpu.OperandMajorMode.MN
        self.major_mode_P = cute.nvgpu.OperandMajorMode.K
        self.num_k_Sd = self.hdimv_dual // 16  # 16 k-blocks of the latent dual GEMM
        self.num_k_Srd = self.hdim_dual // 16  # 2 k-blocks of the rope dual GEMM

        # ==== pipeline info ====
        # three latent stages (72 KiB of Q shared memory freed by Q in TMEM) and one rope tile: the
        # rope rows of a block are written after the rope GEMM of the previous block released the
        # tile, so they are issued last and the rope GEMM is the last instruction of S(n)
        self.num_stages_KV = 3
        self.num_stages_K = 1
        # A latent stage lands in num_kv_parts parts (whole 128-B column blocks of the latent row),
        # each with its own cp.async mbarrier, so the S GEMM streams behind the fill. With rope
        # every latent part has its own barrier and the pipeline's full barrier covers the rope
        # rows; without rope the last part is the full barrier.
        self.num_kv_parts = 4
        num_col_blocks_V = self.hdimv * 2 // 128  # 128-B column blocks of a bf16 latent row
        assert num_col_blocks_V % self.num_kv_parts == 0
        self.col_blocks_per_part = num_col_blocks_V // self.num_kv_parts
        assert self.num_k_Sd % self.num_kv_parts == 0
        self.k_per_part = self.num_k_Sd // self.num_kv_parts
        self.num_latent_part_mbars = self.num_kv_parts if self.has_qk else self.num_kv_parts - 1
        # one S stage (64 TMEM columns): O 256 + Q 144 + S 64 = 464 of 512, a second stage does
        # not fit; the MMA warp issues S(n) before PV(n-1) so the softmax of block n-1 overlaps
        # PV(n-1) and S(n) needs only the t2r of S(n-1) to have happened
        self.num_stages_S = 1
        self.num_stages_P = 1
        self.num_stages_Oi = 1
        self.num_stages_sm_stats = 2
        self.num_stages_bitmask = 2

        # ==== dtype info ====
        self.dtype_acc = Float32
        # epilogue O / o_lo stores: 256-bit `st.global.v8.b32` (sm_100), one request per 32-B sector
        self.o_store_bits = 256

        # ==== TMEM info ("2x2" .ws accumulator layout: 64 x N fp32 = 128 lanes x N/2 columns) ====
        # every region 32-column aligned (tcgen05.ld 32x32b.x32 sub-tiles of the softmax)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_cols_Oi = self.hdimv_split // 2
        self.tmem_cols_S = self.n_dual // 2
        self.tmem_cols_Qv = self.hdimv // 4  # 64 x 512 bf16 = 128 lanes x 128 columns of pairs
        self.tmem_cols_Qr = self.hdim // 4
        self.tmem_offset_O0 = 0
        self.tmem_offset_O1 = self.tmem_offset_O0 + self.tmem_cols_Oi
        self.tmem_offsets_O = [self.tmem_offset_O0, self.tmem_offset_O1]
        self.tmem_offset_S = self.tmem_offset_O1 + self.tmem_cols_Oi
        self.tmem_offset_Qv = self.tmem_offset_S + self.tmem_cols_S
        self.tmem_offset_Qr = self.tmem_offset_Qv + self.tmem_cols_Qv
        self.total_tmem = self.tmem_offset_Qr + (self.tmem_cols_Qr if self.has_qk else 0)
        assert self.total_tmem <= self.tmem_alloc_cols
        for off in (
            self.tmem_offset_O1,
            self.tmem_offset_S,
            self.tmem_offset_Qv,
            self.tmem_offset_Qr,
        ):
            assert off % 32 == 0

    @cute.jit
    def is_valid_qhead_row(self, packed_row) -> Boolean:
        """No padded heads in this kernel; kept for symmetry with the 2-CTA kernel's guards."""
        return Boolean(True)

    @staticmethod
    def tmem_acc_layout(n: int) -> cute.Layout:
        """TMEM layout of a 64 x n fp32 `.ws` M=64 accumulator: row r in lane r (columns 0 ..
        n/2) and lane r + 64 (columns n/2 .. n); identical to the per-CTA layout of the 2-CTA
        kernel's M=128 fragments, so its softmax / correction partitions apply unchanged."""
        return cute.make_layout(((64, (n // 2, 2)), 1, 1), stride=((1 << 16, (1, 64 << 16)), 0, 0))

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

        (sK_struct, sV_struct, sP_struct) = (
            smem_struct_align(dtype, layout, disabled)
            for dtype, layout, disabled in [
                (self.dtype_K, self.sK_layout_staged, not self.has_qk),
                (self.dtype_V, self.sV_layout_staged, False),
                (self.dtype_P, self.sP_layout_staged, False),
            ]
        )
        sX_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.cosize(self.sX_layout)], self.buffer_align_bytes
        ]
        sStats_struct = cute.struct.MemRange[Float32, cute.cosize(self.sStats_layout)]
        sRowMax_struct = cute.struct.MemRange[Float32, cute.cosize(self.sRowMax_layout)]
        sScale_struct = cute.struct.MemRange[Float32, cute.cosize(self.sScale_layout)]
        sBitmask_struct = cute.struct.MemRange[Uint32, cute.cosize(self.sBitmask_layout)]

        (
            mbar_ptr_KV_struct,
            mbar_ptr_K_struct,
            mbar_ptr_S_struct,
            mbar_ptr_P_struct,
            mbar_ptr_O0_struct,
            mbar_ptr_O1_struct,
            mbar_sm_stats_struct,
            mbar_bitmask_struct,
        ) = (
            mbar_struct(n)
            for n in [
                self.num_stages_KV,
                self.num_stages_K if self.has_qk else 0,
                self.num_stages_S,
                self.num_stages_P,
                self.num_stages_Oi,
                self.num_stages_Oi,
                self.num_stages_sm_stats,
                self.num_stages_bitmask,
            ]
        )
        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0
        num_part_mbars = self.num_latent_part_mbars * self.num_stages_KV

        @cute.struct
        class SharedStorage:
            mbar_ptr_KV: mbar_ptr_KV_struct
            mbar_ptr_KV_part: cute.struct.MemRange[Int64, num_part_mbars]
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_S: mbar_ptr_S_struct
            mbar_ptr_P: mbar_ptr_P_struct
            mbar_ptr_O0: mbar_ptr_O0_struct
            mbar_ptr_O1: mbar_ptr_O1_struct
            mbar_ptr_sm_stats: mbar_sm_stats_struct
            mbar_ptr_bitmask: mbar_bitmask_struct
            mbar_ptr_utccp: Int64
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            # the scheduler reads the CLC response with one 16-B load
            clc_response: cute.struct.Align[cute.struct.MemRange[Int32, clc_response_size], 16]

            sRowMax: sRowMax_struct
            sRowSum: sStats_struct
            sScale: sScale_struct
            sBitmask: sBitmask_struct
            sV: sV_struct
            sK: sK_struct
            sP: sP_struct
            sX: sX_struct

        return SharedStorage

    # fmt: off
    @cute.jit
    def __call__(
        self,
        mQ: Optional[cute.Tensor],    # (b, s_q, h, d)     or (total_q, h, d)    if there is cu_seqlens_q
        mQv: cute.Tensor,             # (b, s_q, h, dv)    or (total_q, h, dv)
        mK: Optional[cute.Tensor],    # (b, s_k, h_k, d)   or (total_k, h_k, d)
        mV: cute.Tensor,              # (b, s_k, h_k, dv)  or (total_k, h_k, dv)
        mO: cute.Tensor,              # (b, s_q, h, dv)    or (total_q, h, dv)
        mLSE: Optional[cute.Tensor],  # (b, s_q, h)        or (total_q, h)
        softmax_scale: Float32,
        mP: Optional[cute.Tensor] = None,           # must be None (P is laid out per 128-key tile)
        mRowMax: Optional[cute.Tensor] = None,      # must be None
        mCuSeqlensQ: Optional[cute.Tensor] = None,  # (b + 1)
        mCuSeqlensK: Optional[cute.Tensor] = None,  # (b + 1)
        mSeqUsedQ: Optional[cute.Tensor] = None,    # (b)
        mSeqUsedK: Optional[cute.Tensor] = None,    # (b)
        mIndexTopk: Optional[cute.Tensor] = None,   # (b, s_q, topk)  or (total_q, topk)
        mPageTable: Optional[cute.Tensor] = None,   # must be None
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        mOlo: Optional[cute.Tensor] = None,          # same shape/dtype as mO: bf16 rounding residual of O
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # fmt: on
        assert mP is None and mRowMax is None, "H64 forward serves p is None / row_max is None only"
        assert mPageTable is None, "H64 forward: top-k gather only"
        assert mIndexTopk is not None
        self.store_O_residual = mOlo is not None
        mDebugClock = None
        if const_expr(self.debug_clock_addr is not None):
            mDebugClock = cute.make_tensor(
                cute.make_ptr(Int64, self.debug_clock_addr, cute.AddressSpace.gmem, assumed_align=8),
                cute.make_layout((self.num_n_blocks + 8, 20), stride=(20, 1)),
            )

        if const_expr(self.has_qk):
            assert mQ is not None and mK is not None, "has_qk requires mQ and mK"
        else:
            assert mQ is None and mK is None, "not has_qk disallows mQ and mK"

        # ==== dtype info ====
        self.dtype_Q = mQ.element_type if self.has_qk else cutlass.BFloat16
        self.dtype_K = mK.element_type if self.has_qk else cutlass.BFloat16
        self.dtype_Qv = mQv.element_type
        self.dtype_V = mV.element_type
        self.dtype_P = mV.element_type
        self.dtype_O = mO.element_type
        assert self.dtype_Qv == self.dtype_V, "Q and the latent share the stage layout"
        if const_expr(self.has_qk):
            assert self.dtype_Q == self.dtype_K, "Q_rope and K_rope share the rope tile layout"
        if const_expr(self.store_O_residual):
            assert mOlo.element_type == self.dtype_O, "O residual must have O's dtype"

        # ==== Prepare Tensors ====
        new_stride = lambda mX, bits=128: (
            *(cute.assume(s, divby=bits // mX.element_type.width) for s in mX.stride[:-1]),
            mX.stride[-1],
        )
        mQ, mQv, mK, mV = [
            cute.make_tensor(mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX)))
            if mX is not None
            else None
            for mX in (mQ, mQv, mK, mV)
        ]
        # O and o_lo: strides divisible by the epilogue store width (rows are 1024 B, torch allocations)
        mO, mOlo = [
            cute.make_tensor(
                mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX, self.o_store_bits))
            )
            if mX is not None
            else None
            for mX in (mO, mOlo)
        ]
        # O and o_lo are torch allocations (>= 256-B aligned) with 1024-B rows: the 256-bit stores of
        # the epilogue need a 32-B alignment promise on the pointer (the interface asserts 16 B)
        if const_expr(self.o_store_bits > 128):
            mO, mOlo = [
                cute.make_tensor(
                    cute.make_ptr(
                        mX.element_type,
                        mX.iterator.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=self.o_store_bits // 8,
                    ),
                    mX.layout,
                )
                if mX is not None
                else None
                for mX in (mO, mOlo)
            ]
        # (b, s, h, d) -> (s, d, h, b) or (total, h, d) -> (total, d, h)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ, mQv, mO, mOlo = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            if mX is not None
            else None
            for mX in (mQ, mQv, mO, mOlo)
        ]
        mK, mV = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=KV_layout_transpose))
            if mX is not None
            else None
            for mX in (mK, mV)
        ]
        # (b, s_q, topk) -> (topk, s_q, b) or (total_q, topk) -> (topk, total_q)
        topk_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mIndexTopk = cute.make_tensor(
            mIndexTopk.iterator, cute.select(mIndexTopk.layout, mode=topk_layout_transpose)
        )
        # (b, s_q, h) -> (s_q, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )
        # pack the 64 heads into the token mode: ((64, seqlen), d, 1, b)
        mQ, mQv, mO, mOlo = [
            pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
            if mX is not None
            else None
            for mX in (mQ, mQv, mO, mOlo)
        ]
        if const_expr(mLSE is not None):
            mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)

        # ==== Prepare MMAs (layout / descriptor providers of the .ws PTX helpers) ====
        K, MN = cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN
        make_mma = sm100_utils.make_trivial_tiled_mma
        # physical stage / rope tile layouts (what the gather writes): an M=64 N=64 K-major op
        tiled_mma_64 = make_mma(self.dtype_V, self.dtype_V, K, K, self.dtype_acc, self.cta_group, (self.cta_tile_m, self.tile_n))
        # the dual GEMM (idesc M=64 N=128) and the (128 rows, K/2) re-views of the same bytes
        tiled_mma_Sd = make_mma(self.dtype_V, self.dtype_V, K, K, self.dtype_acc, self.cta_group, (self.cta_tile_m, self.n_dual))
        tiled_mma_PV = make_mma(self.dtype_P, self.dtype_V, K, MN, self.dtype_acc, self.cta_group, (self.cta_tile_m, self.hdimv_split))

        # ==== Prepare SMEM layouts ====
        # fmt: off
        _smem_layout_specs = [
            # latent stages: 64 keys x 512 dims, K-major SW128, 8 column tiles of 64 x 64; 3 stages
            ("sV_layout",  sm100_utils.make_smem_layout_b, tiled_mma_64, self.mma_tiler_stage, self.dtype_V, self.num_stages_KV),
            # the same bytes as the B operand of the dual GEMM: (128 rows, 256) SW128, rows 64-127 = the
            # second column tile of each pair = the upper dim half of the same keys; 3 stages
            ("sVd_layout", sm100_utils.make_smem_layout_b, tiled_mma_Sd, self.mma_tiler_Sd, self.dtype_V, self.num_stages_KV),
            # MN-major re-view of one 256-dim N-tile of the latent stage for P.V (tile 1 at +16384 elements)
            ("sVt_layout", sm100_utils.make_smem_layout_b, tiled_mma_PV, self.mma_tiler_PV, self.dtype_V, self.num_stages_KV),
            ("sP_layout",  sm100_utils.make_smem_layout_a, tiled_mma_PV, self.mma_tiler_PV, self.dtype_P, self.num_stages_P),
        ]
        if const_expr(self.has_qk):
            _smem_layout_specs += [
                # rope tile: 64 keys x 64 dims as two K-major SW64 column tiles of 64 x 32 (the "2 stages" are
                # the two dim halves at 2048 elements), one tile
                ("sK_layout",  sm100_utils.make_smem_layout_b, tiled_mma_64, self.mma_tiler_rope, self.dtype_K, 2),
                # its dual view: (128 rows, 32) SW64, rows 64-127 = dims 32-63
                ("sKd_layout", sm100_utils.make_smem_layout_b, tiled_mma_Sd, self.mma_tiler_Srd, self.dtype_K, self.num_stages_K),
            ]
        for attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages in _smem_layout_specs:
            ab_kwarg = "a_dtype" if make_fn is sm100_utils.make_smem_layout_a else "b_dtype"
            staged = make_fn(tiled_mma=tiled_mma, mma_tiler_mnk=mma_tiler, num_stages=num_stages, **{ab_kwarg: dtype})
            setattr(self, f"{attr}_staged", staged)
            setattr(self, attr, cute.select(staged, mode=[0, 1, 2]))
        # fmt: on
        stage_elems_V = cute.cosize(self.sV_layout)
        assert stage_elems_V == self.tile_n * self.hdimv
        assert cute.cosize(self.sVd_layout) == stage_elems_V, "dual view must cover exactly one stage"
        assert cute.cosize(self.sVd_layout_staged) == cute.cosize(self.sV_layout_staged)
        assert cute.size(self.sVd_layout_staged.outer, mode=[2]) == self.num_k_Sd
        if const_expr(self.has_qk):
            assert cute.cosize(self.sK_layout_staged) == self.tile_n * self.hdim
            assert cute.cosize(self.sKd_layout_staged) == self.tile_n * self.hdim
            assert cute.size(self.sKd_layout_staged.outer, mode=[2]) == self.num_k_Srd
        else:
            self.sK_layout_staged = self.sKd_layout_staged = None
        # The MN-major view of one 256-dim N-tile spans half a latent stage (16384 elements), so
        # the layout builder stacks its stages at that stride; the view must step by the full
        # latent stage (32768 elements) instead. Re-stage the per-stage view with sV's stride.
        self.sVt_ntile_offset = self.hdimv_split * self.tile_n  # elements: 256 dims x 64 keys
        assert cute.cosize(self.sVt_layout) == self.sVt_ntile_offset
        assert self.num_hdimv_splits * self.sVt_ntile_offset == stage_elems_V
        self.sVt_layout_outer = cute.append(
            cute.select(self.sVt_layout_staged.outer, mode=[0, 1, 2]),
            cute.make_layout(self.num_stages_KV, stride=stage_elems_V),
        )

        self.sStats_layout = cute.make_layout((self.cta_tile_m, self.threads_per_row))
        # row-max exchange double-buffered by block parity: one named barrier per block
        self.sRowMax_layout = cute.make_layout((self.cta_tile_m, 2 * self.threads_per_row))
        self.sScale_layout = cute.make_layout((self.cta_tile_m, self.num_stages_sm_stats))
        self.sBitmask_layout = cute.make_layout((self.tile_n // 32, self.num_stages_bitmask))
        # lane-half exchange of the dual GEMM: 32 fp32 per softmax thread, thread-contiguous per
        # value (a warp's 32 scalar stores / loads of one value hit 32 consecutive words)
        self.sX_layout = cute.make_layout(
            (self.num_softmax_threads, self.tile_n // 2), stride=(1, self.num_softmax_threads)
        )

        # ==== O rmem -> gmem copy: thread t stores row t % 64, dims 128 * (t // 64) .. +128 of a split ====
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype_O, num_bits_per_copy=self.o_store_bits
        )
        thread_layout_O_r2g = cute.make_layout((self.cta_tile_m, self.threads_per_row), stride=(1, self.cta_tile_m))
        value_layout_O_r2g = cute.make_layout((1, self.hdimv_split // self.threads_per_row))
        tiled_copy_O_r2g = cute.make_tiled_copy_tv(
            atom=atom_universal_copy, thr_layout=thread_layout_O_r2g, val_layout=value_layout_O_r2g
        )

        # ==== Allocate shared memory ====
        SharedStorage = self._get_shared_storage_cls()
        fa_log(1, f"H64 fwd smem = {SharedStorage.size_in_bytes()} B, TMEM = {self.total_tmem} cols")

        # ==== Tile scheduler ====
        TileScheduler = self.TileScheduler
        batch_size_for_sched = (
            cute.size(mQv.shape[3]) if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1) if self.use_varlen_scheduler
            else 1
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQv.shape[0]), self.cluster_tile_m),
            num_head=cute.size(mQv.shape[2]),
            num_batch=batch_size_for_sched,
            num_splits=1,
            seqlen_k=cute.size(mV.shape[0]),
            headdim=self.hdim,
            headdim_v=self.hdimv,
            total_q=cute.size(mQv.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQv.shape[0]) * cute.size(mQv.shape[3]),
            tile_shape_mn=(self.cta_tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
            element_size=self.dtype_V.width // 8,
            is_persistent=self.is_persistent,
            lpt=False,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=True,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        fa_printf(1, "grid = {}", grid_dim)

        # ==== Named barriers (ids shared with the 2-CTA kernel's enum; per-kernel scope) ====
        self.cpasync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Cpasync),
            num_threads=self.num_cpasync_load_threads,
        )
        self.softmax_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Softmax),
            num_threads=self.num_softmax_threads,
        )
        self.epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Epilogue),
            num_threads=self.num_epilogue_threads,
        )
        self.sm_stats_barrier_full = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsFull),
            num_threads=self.num_softmax_threads + self.num_epilogue_threads,
        )
        self.sm_stats_barrier_empty = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsEmpty),
            num_threads=self.num_softmax_threads + self.num_epilogue_threads,
        )

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        # ==== Launch kernel ====
        self.kernel(
            mQ,
            mQv,
            mK,
            mV,
            mO,
            mOlo,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mIndexTopk,
            learnable_sink,
            mDebugClock,
            tiled_copy_O_r2g,
            self.sV_layout_staged,
            self.sVd_layout_staged,
            self.sK_layout_staged,
            self.sKd_layout_staged,
            self.sVt_layout_staged,
            self.sVt_layout_outer,
            self.sP_layout_staged,
            self.sX_layout,
            self.sStats_layout,
            self.sRowMax_layout,
            self.sScale_layout,
            self.sBitmask_layout,
            tiled_mma_Sd,
            tiled_mma_PV,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=(self.num_threads, 1, 1),
            cluster=None,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: Optional[cute.Tensor],
        mQv: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        mO: cute.Tensor,
        mOlo: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mIndexTopk: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        mDebugClock: Optional[cute.Tensor],
        tiled_copy_O_r2g: cute.TiledCopy,
        sV_layout_staged: cute.ComposedLayout,
        sVd_layout_staged: cute.ComposedLayout,
        sK_layout_staged: Optional[cute.ComposedLayout],
        sKd_layout_staged: Optional[cute.ComposedLayout],
        sVt_layout_staged: cute.ComposedLayout,
        sVt_layout_outer: cute.Layout,
        sP_layout_staged: cute.ComposedLayout,
        sX_layout: cute.Layout,
        sStats_layout: cute.Layout,
        sRowMax_layout: cute.Layout,
        sScale_layout: cute.Layout,
        sBitmask_layout: cute.Layout,
        tiled_mma_Sd: cute.TiledMma,
        tiled_mma_PV: cute.TiledMma,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        tile_sched_params: ParamsBase,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_Sd.thr_id.shape,)
        )

        # ==== Allocate SMEM ====
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ==== TMEM stuff ====
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.TmemPtr),
            num_threads=self.num_mma_threads + self.num_softmax_threads + self.num_epilogue_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # ==== Construct pipelines ====
        mma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads)
        epi_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilogue_threads)
        gather_threads = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_cpasync_load_threads
        )

        AsyncUmma = pipeline.PipelineAsyncUmma
        UmmaAsync = pipeline.PipelineUmmaAsync
        Async = pipeline.PipelineAsync

        def make_pipeline(cls, mbar_ptr, num_stages, producer, consumer, tx_count=None):
            return cls.create(
                barrier_storage=mbar_ptr.data_ptr(),
                num_stages=num_stages,
                producer_group=producer,
                consumer_group=consumer,
                defer_sync=True,
                **({"cta_layout_vmnk": cta_layout_vmnk} if cls is not Async else {}),
                **({"tx_count": tx_count} if tx_count is not None else {}),
            )

        # fmt: off
        # the 128 gather threads arrive on the full barriers with cp.async.mbarrier.arrive.noinc; the
        # MMA warp releases with tcgen05.commit (after PV(n), or after the UTCCP of the Q stage)
        pipeline_KV       = make_pipeline(AsyncUmma, storage.mbar_ptr_KV,       self.num_stages_KV,       gather_threads, mma_warp)
        # the rope tile: producer_acquire = "the rope GEMM (or the rope UTCCP) of the previous block
        # released it"; its full side is unused (the stage's full barrier covers the rope rows)
        pipeline_K = None
        if const_expr(self.has_qk):
            pipeline_K    = make_pipeline(AsyncUmma, storage.mbar_ptr_K,        self.num_stages_K,        gather_threads, mma_warp)
        pipeline_S        = make_pipeline(UmmaAsync, storage.mbar_ptr_S,        self.num_stages_S,        mma_warp,       sm_threads)
        pipeline_P        = make_pipeline(AsyncUmma, storage.mbar_ptr_P,        self.num_stages_P,        sm_threads,     mma_warp)
        pipeline_O0       = make_pipeline(UmmaAsync, storage.mbar_ptr_O0,       self.num_stages_Oi,       mma_warp,       epi_threads)
        pipeline_O1       = make_pipeline(UmmaAsync, storage.mbar_ptr_O1,       self.num_stages_Oi,       mma_warp,       epi_threads)
        pipeline_sm_stats = make_pipeline(Async,     storage.mbar_ptr_sm_stats, self.num_stages_sm_stats, sm_threads,     epi_threads)
        pipeline_bitmask  = (
            make_pipeline(Async, storage.mbar_ptr_bitmask, self.num_stages_bitmask, gather_threads, sm_threads)
            if const_expr(not self.disable_bitmask) else None
        )
        # fmt: on
        # part barriers of the KV ring (raw mbarriers: 128 cp.async arrivals per landed part) and the
        # MMA warp's private UTCCP-done barrier (1 tcgen05.commit arrival per tile)
        mbar_KV_part = storage.mbar_ptr_KV_part.data_ptr()
        mbar_utccp = storage.mbar_ptr_utccp.ptr
        if warp_idx == 0:
            if cute.arch.lane_idx() == 0:
                for i in range(self.num_latent_part_mbars * self.num_stages_KV):
                    cute.arch.mbarrier_init(mbar_KV_part + i, self.num_cpasync_load_threads)
                cute.arch.mbarrier_init(mbar_utccp, 1)

        pipeline.pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # ==== Get SMEM tensors ====
        sV = storage.sV.get_tensor(sV_layout_staged.outer, swizzle=sV_layout_staged.inner)
        sK = None
        sKd = None
        if const_expr(self.has_qk):
            sK = storage.sK.get_tensor(sK_layout_staged.outer, swizzle=sK_layout_staged.inner)
            sKd = cute.make_tensor(
                cute.recast_ptr(sK.iterator, sKd_layout_staged.inner, self.dtype_K),
                sKd_layout_staged.outer,
            )
        # the dual-GEMM view of the latent stages (same bytes, same swizzle, same stage stride)
        sVd = cute.make_tensor(
            cute.recast_ptr(sV.iterator, sVd_layout_staged.inner, self.dtype_V),
            sVd_layout_staged.outer,
        )
        # the single P buffer as a per-stage (3-mode) tensor straight from storage: slicing the staged
        # tensor with a static stage index loses the 1024-B alignment the 16-B P stores need
        sP_stage_layout = cute.select(sP_layout_staged, mode=[0, 1, 2])
        sP = storage.sP.get_tensor(sP_stage_layout.outer, swizzle=sP_stage_layout.inner)
        # the two 256-dim N-tiles of the latent stage viewed MN-major (same bytes, same swizzle;
        # sVt_layout_outer steps stages by the full latent stage)
        sVt0 = cute.make_tensor(
            cute.recast_ptr(sV.iterator, sVt_layout_staged.inner, self.dtype_V),
            sVt_layout_outer,
        )
        sVt1 = cute.make_tensor(
            cute.recast_ptr(sV.iterator + self.sVt_ntile_offset, sVt_layout_staged.inner, self.dtype_V),
            sVt_layout_outer,
        )
        sX = storage.sX.get_tensor(sX_layout)
        sRowMax = storage.sRowMax.get_tensor(sRowMax_layout)
        sRowSum = storage.sRowSum.get_tensor(sStats_layout)
        sScale = storage.sScale.get_tensor(sScale_layout)
        sBitmask = storage.sBitmask.get_tensor(sBitmask_layout)

        # ==== TMEM accumulator layouts (2x2 .ws layout, see tmem_acc_layout) ====
        # S (64 x 128 fp32 dual GEMM): TMEM column c (< 64) of lane r holds the lower-dim-half
        # partial sum of key c for row r, of lane 64 + r the upper-half partial. Viewed as two 64 x 64
        # accumulators (keys 0-31 at the S offset, keys 32-63 at + 32 columns) so the 2-CTA kernel's
        # per-thread 32-column partition applies: thread t reads its lane half's partial of 32 keys.
        tStS_layout = self.tmem_acc_layout(self.tile_n)
        tOtO_layout = self.tmem_acc_layout(self.hdimv_split)

        block_info = BlockInfo(
            self.cta_tile_m,
            self.tile_n,
            is_causal=self.is_causal,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQv.shape[0][1],
            seqlen_k_static=mV.shape[0],
            tile_m=self.cta_tile_m,
            tile_n=self.tile_n,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_clc_consumer_warps = self.num_threads // cute.arch.WARP_SIZE
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cute.arch.WARP_SIZE * num_clc_consumer_warps
            )
            sched_ctx = SchedulerState.create_clc(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, ctx=sched_ctx)
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), (
            f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"
        )

        pipeline.pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                if const_expr(self.num_regs_other < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.clc_scheduler_warp(tile_scheduler)
        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                if const_expr(self.num_regs_other < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.empty_warp(tile_scheduler)

        if warp_idx in self.cpasync_load_warp_indices:
            if const_expr(self.num_regs_cpasync < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_cpasync)
            self.load_cpasync(
                mIndexTopk,
                mQ,
                mQv,
                mK,
                mV,
                sK,
                sV,
                sBitmask,
                pipeline_KV,
                pipeline_K,
                mbar_KV_part,
                pipeline_bitmask,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                mCuSeqlensQ=mCuSeqlensQ,
                mDebugClock=mDebugClock,
            )

        if warp_idx == self.mma_warp_id:
            if const_expr(self.num_regs_mma < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # ==== Allocate TMEM ====
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            self.mma(
                sKd,
                sVd,
                sVt0,
                sVt1,
                sP,
                tmem_ptr,
                tiled_mma_Sd,
                tiled_mma_PV,
                pipeline_KV,
                pipeline_K,
                mbar_KV_part,
                mbar_utccp,
                pipeline_S,
                pipeline_P,
                pipeline_O0,
                pipeline_O1,
                tile_scheduler=tile_scheduler,
                mCuSeqlensQ=mCuSeqlensQ,
                mDebugClock=mDebugClock,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx in self.softmax_warp_indices:
            if const_expr(self.num_regs_softmax > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            # keys 0-31 and keys 32-63 of the dual-GEMM accumulator (this lane half's partials)
            tStS0 = cute.make_tensor(tmem_ptr + self.tmem_offset_S, tStS_layout)
            tStS1 = cute.make_tensor(tmem_ptr + self.tmem_offset_S + self.tile_n // 2, tStS_layout)
            self.softmax_loop(
                softmax_scale,
                softmax_scale_log2,
                sRowMax,
                sRowSum,
                sScale,
                sBitmask,
                sX,
                sP,
                tStS0,
                tStS1,
                pipeline_S,
                pipeline_P,
                pipeline_sm_stats,
                pipeline_bitmask,
                tile_scheduler=tile_scheduler,
                mCuSeqlensQ=mCuSeqlensQ,
                learnable_sink=learnable_sink,
                mDebugClock=mDebugClock,
            )
            tmem_alloc_barrier.arrive()

        if warp_idx in self.epilogue_warp_indices:
            if const_expr(self.num_regs_epilogue < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            elif const_expr(self.num_regs_epilogue > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            tOtO0 = cute.make_tensor(tmem_ptr + self.tmem_offset_O0, tOtO_layout)
            tOtO1 = cute.make_tensor(tmem_ptr + self.tmem_offset_O1, tOtO_layout)
            self.correction_loop(
                softmax_scale_log2,
                mO,
                mLSE,
                sRowMax,
                sRowSum,
                sScale,
                tOtO0,
                tOtO1,
                pipeline_O0,
                pipeline_O1,
                pipeline_sm_stats,
                tiled_copy_O_r2g,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                mCuSeqlensQ=mCuSeqlensQ,
                learnable_sink=learnable_sink,
                mOlo=mOlo,
                mDebugClock=mDebugClock,
            )
            tmem_alloc_barrier.arrive()

    @cute.jit
    def clc_scheduler_warp(self, tile_scheduler: TileSchedulerProtocol):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            if cute.arch.thread_idx()[0] == self.clc_scheduler_warp_id * cute.arch.WARP_SIZE:
                fa_printf(
                    3,
                    "[CLC] query sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(self, tile_scheduler: TileSchedulerProtocol):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def tile_coords(self, work_tile, mCuSeqlensQ: Optional[cute.Tensor]):
        """(token block, KV head, batch) of a work tile; with cu_seqlens the block index is the
        global token index (the top-k table is indexed by it) and the batch is looked up."""
        cluster_m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        if const_expr(self.use_packed_varlen_sched):
            batch_idx = get_batch_from_cu_tensor(cluster_m_block, mCuSeqlensQ)
        return cluster_m_block, head_idx, batch_idx

    # ------------------------------------------------------------------------------------------
    # gather warps (12-15): per tile the token's Q tile as a pseudo-block of the KV ring (its 64
    # head rows in the latent-stage and rope-tile layouts, copied to TMEM by the MMA warp), then one
    # 64-key stage of latent + rope rows per block, indices two blocks ahead
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def load_cpasync(
        self,
        mIndexTopk: cute.Tensor,
        mQ: Optional[cute.Tensor],
        mQv: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        sK: Optional[cute.Tensor],
        sV: cute.Tensor,
        sBitmask: cute.Tensor,
        pipeline_KV: pipeline.PipelineAsyncUmma,
        pipeline_K: Optional[pipeline.PipelineAsyncUmma],
        mbar_KV_part: cute.Pointer,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mDebugClock: Optional[cute.Tensor] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % self.num_cpasync_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_cpasync_load_threads // 32
        )
        Producer = pipeline.PipelineUserType.Producer
        producer_state_KV = pipeline.make_pipeline_state(Producer, stages=self.num_stages_KV)
        producer_state_K = None
        if const_expr(self.has_qk):
            producer_state_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
        producer_state_bitmask = pipeline.make_pipeline_state(
            Producer, stages=self.num_stages_bitmask
        )

        tile_count = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            dbg_tile = Boolean(cute.arch.block_idx()[0] == 0 and tile_count < 8 and tidx == 0)
            tile_row = self.num_n_blocks + tile_count
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 13] = cute.arch.clock64()
            m_idx, head_idx, batch_idx = self.tile_coords(work_tile, mCuSeqlensQ)
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mIndexTopk_cur = mIndexTopk[None, m_idx, batch_idx]
            else:
                mIndexTopk_cur = mIndexTopk[None, m_idx]
            m_local_idx = m_idx - seqlen.offset_q if const_expr(self.use_packed_varlen_sched) else m_idx
            if const_expr(self.is_causal):
                seqlen_k_limit = m_local_idx + 1 + seqlen.seqlen_k - seqlen.seqlen_q
            else:
                seqlen_k_limit = seqlen.seqlen_k
            gather = CpasyncGatherKVManagerH64.create(
                mIndexTopk_cur,
                tidx,
                warp_idx,
                seqlen_k_limit,
                self.tile_n,
                self.hdim,
                self.hdimv,
                self.num_cpasync_load_threads,
                mV.element_type,
                self.cpasync_barrier,
                self.disable_bitmask,
                sBitmask,
                pipeline_bitmask,
            )
            # (seqlen_k, hdim) and (seqlen_k, hdimv) of this batch / KV head
            mK_cur = None
            if const_expr(self.has_qk):
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx]
            # the token's Q tile: (64 heads, dims), rows in the packed (head, token) order
            gQ = None
            if const_expr(self.has_qk):
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                gQ = cute.local_tile(mQ_cur, (self.cta_tile_m, self.hdim), (m_local_idx, 0))
            mQv_cur = seqlen.offset_batch_Q(mQv, batch_idx, dim=3)[None, None, head_idx]
            gQv = cute.local_tile(mQv_cur, (self.cta_tile_m, self.hdimv), (m_local_idx, 0))

            dbg = Boolean(cute.arch.block_idx()[0] == 0 and tile_count == 0 and tidx == 0)
            # Q pseudo-block first: the MMA warp copies it to TMEM before the first S GEMM
            producer_state_KV, producer_state_K = self.gather_q(
                gather, pipeline_KV, pipeline_K, mbar_KV_part, sK, sV, gQ, gQv,
                producer_state_KV, producer_state_K,
            )
            gather_block = partial(
                self.gather_block, gather, pipeline_KV, pipeline_K, mbar_KV_part, sK, sV, mK_cur, mV_cur,
                mDebugClock,
            )
            # blocks in decreasing order, two index register sets, indices two blocks ahead
            n_block = Int32(self.num_n_blocks - 1)
            gather.load_index_topk(n_block, 0)
            gather.load_index_topk(n_block - 1, 1)
            for it in cutlass.range(self.num_n_blocks // 2, unroll=1):
                for buf in cutlass.range_constexpr(2):
                    producer_state_KV, producer_state_K, producer_state_bitmask = gather_block(
                        producer_state_KV, producer_state_K, producer_state_bitmask, buf, dbg, 2 * it + buf
                    )
                    n_prefetch = n_block - 2 - buf
                    n_prefetch = n_prefetch if n_prefetch >= 0 else Int32(0)
                    gather.load_index_topk(n_prefetch, buf)
                n_block -= 2
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 15] = cute.arch.clock64()

            work_tile = tile_scheduler.advance_to_next_work()
            tile_count += 1

        pipeline_KV.producer_tail(producer_state_KV)
        if const_expr(not self.disable_bitmask):
            pipeline_bitmask.producer_tail(producer_state_bitmask)

    @cute.jit
    def gather_q(
        self,
        gather: CpasyncGatherKVManagerH64,
        pipeline_KV: pipeline.PipelineAsyncUmma,
        pipeline_K: Optional[pipeline.PipelineAsyncUmma],
        mbar_KV_part: cute.Pointer,
        sK: Optional[cute.Tensor],
        sV: cute.Tensor,
        gQ: Optional[cute.Tensor],
        gQv: cute.Tensor,
        producer_state_KV: pipeline.PipelineState,
        producer_state_K: Optional[pipeline.PipelineState],
    ):
        """The token's Q tile as a pseudo-block of the ring: Qv rows into a latent stage, Q_rope rows
        into the rope tile, with the same copies, part barriers and full-barrier arrival as a KV
        block (so every barrier phase advances once per ring slot). No bitmask."""
        stage = producer_state_KV.index
        pipeline_KV.producer_acquire(producer_state_KV)
        for p in cutlass.range_constexpr(self.num_kv_parts):
            gather.load_X(
                gQv,
                sV[None, None, None, stage],
                "V",
                0,
                col_blocks=(p * self.col_blocks_per_part, (p + 1) * self.col_blocks_per_part),
                identity_rows=True,
            )
            if const_expr(p < self.num_latent_part_mbars):
                cute.arch.cp_async_mbarrier_arrive_noinc(
                    mbar_KV_part + (p * self.num_stages_KV + stage)
                )
        if const_expr(self.has_qk):
            pipeline_K.producer_acquire(producer_state_K)
            gather.load_X(gQ, sK, "K", 0, identity_rows=True)
            producer_state_K.advance()
        cute.arch.cp_async_commit_group()
        pipeline_KV.sync_object_full.arrive_cp_async_mbarrier(stage)
        producer_state_KV.advance()
        return producer_state_KV, producer_state_K

    @cute.jit
    def gather_block(
        self,
        gather: CpasyncGatherKVManagerH64,
        pipeline_KV: pipeline.PipelineAsyncUmma,
        pipeline_K: Optional[pipeline.PipelineAsyncUmma],
        mbar_KV_part: cute.Pointer,
        sK: Optional[cute.Tensor],
        sV: cute.Tensor,
        mK_cur: Optional[cute.Tensor],
        mV_cur: cute.Tensor,
        mDebugClock: Optional[cute.Tensor],
        producer_state_KV: pipeline.PipelineState,
        producer_state_K: Optional[pipeline.PipelineState],
        producer_state_bitmask: pipeline.PipelineState,
        buf: cutlass.Constexpr[int],
        dbg: Boolean,
        dbg_row: Int32,
    ):
        stage = producer_state_KV.index
        pipeline_KV.producer_acquire(producer_state_KV)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 14] = cute.arch.clock64()
        # The latent lands in parts (column blocks [0, cb), [cb, 2cb), ...): a part's
        # cp.async.mbarrier.arrive.noinc fires once every copy this thread issued so far has
        # landed, so the MMA warp starts S on part 0 while the rest streams in. The rope rows go
        # last: the single rope tile is free only once the rope GEMM of the previous block (the
        # last instruction of its S) has completed.
        for p in cutlass.range_constexpr(self.num_kv_parts):
            gather.load_X(
                mV_cur,
                sV[None, None, None, stage],
                "V",
                buf,
                col_blocks=(p * self.col_blocks_per_part, (p + 1) * self.col_blocks_per_part),
            )
            if const_expr(p < self.num_latent_part_mbars):
                cute.arch.cp_async_mbarrier_arrive_noinc(
                    mbar_KV_part + (p * self.num_stages_KV + stage)
                )
        if const_expr(self.has_qk):
            pipeline_K.producer_acquire(producer_state_K)
            gather.load_X(mK_cur, sK, "K", buf)
            producer_state_K.advance()
        cute.arch.cp_async_commit_group()
        pipeline_KV.sync_object_full.arrive_cp_async_mbarrier(stage)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 15] = cute.arch.clock64()
        producer_state_KV.advance()
        if const_expr(not self.disable_bitmask):
            producer_state_bitmask = gather.compute_bitmask(producer_state_bitmask, buf)
        return producer_state_KV, producer_state_K, producer_state_bitmask

    # ------------------------------------------------------------------------------------------
    # MMA warp (9): per tile Q -> TMEM (tcgen05.cp from the Q pseudo-block), then
    #   S(0); for n = 1 .. N-1: S(n), O += P(n-1) V(n-1); O += P(N-1) V(N-1).
    #   S(n) waits only for the softmax's t2r of S(n-1) (one S stage), so the softmax step of
    #   block n-1 runs under S(n) and PV(n-1); the refill of a stage (released after its PV) has
    #   two block periods to land with three stages.
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def mma(
        self,
        sKd: Optional[cute.Tensor],
        sVd: cute.Tensor,
        sVt0: cute.Tensor,
        sVt1: cute.Tensor,
        sP: cute.Tensor,
        tmem_ptr: cute.Pointer,
        tiled_mma_Sd: cute.TiledMma,
        tiled_mma_PV: cute.TiledMma,
        pipeline_KV: pipeline.PipelineAsync,
        pipeline_K: Optional[pipeline.PipelineAsync],
        mbar_KV_part: cute.Pointer,
        mbar_utccp: cute.Pointer,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mDebugClock: Optional[cute.Tensor] = None,
    ):
        tmem_base = tmem_ptr.toint()
        # operand descriptors (k-block offsets) of the .ws PTX helpers
        tSrVd = tiled_mma_Sd.make_fragment_B(sVd)
        tSrKd = None
        if const_expr(self.has_qk):
            tSrKd = tiled_mma_Sd.make_fragment_B(sKd)
        tOrP = tiled_mma_PV.make_fragment_A(sP)
        sP_ = sP
        tOrVt0 = tiled_mma_PV.make_fragment_B(sVt0)
        tOrVt1 = tiled_mma_PV.make_fragment_B(sVt1)

        gemm_ts = fa_sm100_utils.gemm_ws_ts_ptx_partial
        gemm_ws = fa_sm100_utils.gemm_ws_ptx_partial
        utccp = fa_sm100_utils.utccp_128x256b_ptx
        acc_S = Int32(tmem_base + self.tmem_offset_S)
        a_Qv = Int32(tmem_base + self.tmem_offset_Qv)
        a_Qr = Int32(tmem_base + self.tmem_offset_Qr)
        gemm_Sd = partial(gemm_ts, tiled_mma_Sd.op, acc_S, a_Qv)  # zero_init / k_range per landed part
        gemm_Srd = None
        if const_expr(self.has_qk):
            gemm_Srd = partial(gemm_ts, tiled_mma_Sd.op, acc_S, a_Qr, zero_init=False)
        gemm_PV0 = partial(gemm_ws, tiled_mma_PV.op, Int32(tmem_base + self.tmem_offset_O0))
        gemm_PV1 = partial(gemm_ws, tiled_mma_PV.op, Int32(tmem_base + self.tmem_offset_O1))

        mma_S = partial(
            self.mma_S, gemm_Sd, gemm_Srd, tSrVd, sVd, tSrKd, sKd, pipeline_KV, pipeline_K,
            mbar_KV_part, pipeline_S, mDebugClock,
        )
        mma_PV = partial(
            self.mma_PV, gemm_PV0, gemm_PV1, tOrP, sP_, tOrVt0, sVt0, tOrVt1, sVt1,
            pipeline_KV, pipeline_P, pipeline_O0, pipeline_O1, mDebugClock,
        )

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        # two consumer states of the KV ring: the S GEMMs run one block ahead of the PV release
        kv_state_S = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_KV)
        kv_state_PV = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_KV)
        k_state = None
        if const_expr(self.has_qk):
            k_state = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_K)
        producer_state_S = pipeline.make_pipeline_state(Producer, stages=self.num_stages_S)
        consumer_state_P = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_P)
        producer_state_O0 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)
        producer_state_O1 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)
        utccp_phase = Int32(0)

        lane_idx = cute.arch.lane_idx()
        tile_count = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # %clock64 stamps of CTA 0: per block for the first tile (rows 0..n_blocks-1), per tile
            # for the first 8 tiles (rows n_blocks + t); mDebugClock is None in production
            dbg = Boolean(cute.arch.block_idx()[0] == 0 and tile_count == 0 and lane_idx == 0)
            dbg_tile = Boolean(cute.arch.block_idx()[0] == 0 and tile_count < 8 and lane_idx == 0)
            tile_row = self.num_n_blocks + tile_count
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 1] = cute.arch.clock64()

            # ---- Q -> TMEM from the pseudo-block: 16 UTCCP for Qv, 2 for Q_rope ----
            stage_q = kv_state_S.index
            pipeline_KV.consumer_wait(kv_state_S)
            fa_sm100_utils.tcgen05_fence_after_thread_sync()
            utccp(a_Qv, tSrVd[None, None, None, stage_q], sVd[None, None, None, stage_q])
            if const_expr(self.has_qk):
                utccp(a_Qr, tSrKd[None, None, None, 0], sKd[None, None, None, 0])
                pipeline_K.consumer_release(k_state)  # rope tile free once the copies complete
                k_state.advance()
            pipeline_KV.consumer_release(kv_state_S)  # the stage too (tcgen05.commit tracks the cp)
            kv_state_S.advance()
            kv_state_PV.advance()
            with cute.arch.elect_one():
                tcgen05.commit(mbar_utccp)
            cute.arch.mbarrier_wait(mbar_utccp, phase=utccp_phase)
            utccp_phase ^= 1
            fa_sm100_utils.tcgen05_fence_after_thread_sync()
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 0] = cute.arch.clock64()

            # ---- blocks (counted in issue order: 0 = the last key block) ----
            kv_state_S, producer_state_S, k_state = mma_S(kv_state_S, producer_state_S, k_state, dbg, Int32(0))
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 2] = cute.arch.clock64()
            kv_state_S, producer_state_S, k_state = mma_S(kv_state_S, producer_state_S, k_state, dbg, Int32(1))
            kv_state_PV, consumer_state_P, producer_state_O0, producer_state_O1 = mma_PV(
                kv_state_PV, consumer_state_P, producer_state_O0, producer_state_O1, True, dbg, Int32(0)
            )
            blk = Int32(2)
            for _ in cutlass.range(self.num_n_blocks - 2, unroll=1):
                kv_state_S, producer_state_S, k_state = mma_S(kv_state_S, producer_state_S, k_state, dbg, blk)
                kv_state_PV, consumer_state_P, producer_state_O0, producer_state_O1 = mma_PV(
                    kv_state_PV, consumer_state_P, producer_state_O0, producer_state_O1, False, dbg, blk - 1
                )
                blk += 1
            kv_state_PV, consumer_state_P, producer_state_O0, producer_state_O1 = mma_PV(
                kv_state_PV, consumer_state_P, producer_state_O0, producer_state_O1, False, dbg, blk - 1
            )
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 3] = cute.arch.clock64()

            work_tile = tile_scheduler.advance_to_next_work()
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 4] = cute.arch.clock64()
            tile_count += 1

        pipeline_S.producer_tail(producer_state_S)
        pipeline_O0.producer_tail(producer_state_O0)
        pipeline_O1.producer_tail(producer_state_O1)

    @cute.jit
    def mma_S(
        self,
        gemm_Sd,
        gemm_Srd,
        tSrVd: cute.Tensor,
        sVd: cute.Tensor,
        tSrKd: Optional[cute.Tensor],
        sKd: Optional[cute.Tensor],
        pipeline_KV: pipeline.PipelineAsync,
        pipeline_K: Optional[pipeline.PipelineAsync],
        mbar_KV_part: cute.Pointer,
        pipeline_S: pipeline.PipelineAsync,
        mDebugClock: Optional[cute.Tensor],
        kv_state: pipeline.PipelineState,
        s_state: pipeline.PipelineState,
        k_state: Optional[pipeline.PipelineState],
        dbg: Boolean,
        dbg_row: Int32,
    ):
        """S = Qv V^T (16 TS k-steps of .ws M=64 N=128 over the (128, 256) latent view, issued part
        by part as the stage lands) + Q_rope K_rope^T (2 k-steps over the (128, 32) rope view, last:
        the rope rows land last) into the single S TMEM stage; then the rope tile is released."""
        pipeline_S.producer_acquire(s_state)
        stage = kv_state.index
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 6] = cute.arch.clock64()
        for p in cutlass.range_constexpr(self.num_kv_parts):
            if const_expr(p < self.num_latent_part_mbars):
                cute.arch.mbarrier_wait(
                    mbar_KV_part + (p * self.num_stages_KV + stage), phase=kv_state.phase
                )
            else:
                pipeline_KV.consumer_wait(kv_state)
            gemm_Sd(
                tCrB=tSrVd[None, None, None, stage],
                sB=sVd[None, None, None, stage],
                zero_init=p == 0,
                k_range=(p * self.k_per_part, (p + 1) * self.k_per_part),
            )
        if const_expr(self.has_qk):
            pipeline_KV.consumer_wait(kv_state)  # the rope rows (the whole stage) have landed
            gemm_Srd(tCrB=tSrKd[None, None, None, 0], sB=sKd[None, None, None, 0])
            pipeline_K.consumer_release(k_state)  # the next block's rope rows may be written
            k_state.advance()
        pipeline_S.producer_commit(s_state)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 3] = cute.arch.clock64()
        s_state.advance()
        kv_state.advance()
        return kv_state, s_state, k_state

    @cute.jit
    def mma_PV(
        self,
        gemm_PV0,
        gemm_PV1,
        tOrP: cute.Tensor,
        sP_: cute.Tensor,
        tOrVt0: cute.Tensor,
        sVt0: cute.Tensor,
        tOrVt1: cute.Tensor,
        sVt1: cute.Tensor,
        pipeline_KV: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        mDebugClock: Optional[cute.Tensor],
        kv_state: pipeline.PipelineState,
        p_state: pipeline.PipelineState,
        o0_state: pipeline.PipelineState,
        o1_state: pipeline.PipelineState,
        zero_init: cutlass.Constexpr[bool],
        dbg: Boolean,
        dbg_row: Int32,
    ):
        """O_t += P V_t for the two 256-dim N-tiles (4 k-steps each of .ws M=64 N=256 SS, B = the
        latent stage re-viewed MN-major), then release the KV stage and the P buffer."""
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 0] = cute.arch.clock64()
        pipeline_P.consumer_wait(p_state)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 1] = cute.arch.clock64()
        stage = kv_state.index
        pipeline_O0.producer_acquire(o0_state)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 16] = cute.arch.clock64()
        gemm_PV0(
            tCrA=tOrP,
            tCrB=tOrVt0[None, None, None, stage],
            sA=sP_,
            sB=sVt0[None, None, None, stage],
            zero_init=zero_init,
        )
        pipeline_O0.producer_commit(o0_state)
        o0_state.advance()
        pipeline_O1.producer_acquire(o1_state)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 17] = cute.arch.clock64()
        gemm_PV1(
            tCrA=tOrP,
            tCrB=tOrVt1[None, None, None, stage],
            sA=sP_,
            sB=sVt1[None, None, None, stage],
            zero_init=zero_init,
        )
        pipeline_O1.producer_commit(o1_state)
        o1_state.advance()
        # both GEMMs of this block (S earlier, PV now) are done with the stage once this commit fires
        pipeline_KV.consumer_release(kv_state)
        kv_state.advance()
        pipeline_P.consumer_release(p_state)
        p_state.advance()
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 2] = cute.arch.clock64()
        return kv_state, p_state, o0_state, o1_state

    # ------------------------------------------------------------------------------------------
    # softmax warps (0-3): thread t and t + 64 own row t % 64, 32 keys each, after summing the
    # two lane halves of the dual GEMM through the exchange buffer
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def softmax_loop(
        self,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sBitmask: cute.Tensor,
        sX: cute.Tensor,
        sP: cute.Tensor,
        tStS0: cute.Tensor,
        tStS1: cute.Tensor,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        learnable_sink: Optional[cute.Tensor] = None,
        mDebugClock: Optional[cute.Tensor] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % self.num_softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_softmax_threads // 32
        )

        # (64, (32, 2)) fp32: keys 0-31 / keys 32-63 partials of this thread's lane half
        tSAcc0 = tStS0[(None, None), 0, 0]
        tSAcc1 = tStS1[(None, None), 0, 0]

        # S tmem -> rmem copy objects: 32 consecutive columns per thread (lane t = row t % 64)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.dtype_acc
        )
        tmem_load_tiled = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc0)
        tmem_load_thr = tmem_load_tiled.get_slice(tidx)
        tStS0_t2r = tmem_load_thr.partition_S(tSAcc0)
        tStS1_t2r = tmem_load_thr.partition_S(tSAcc1)
        cS = cute.make_identity_tensor(tSAcc0.shape)
        tScS_t2r = tmem_load_thr.partition_D(cS)
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.dtype_acc)  # the summed 32 keys
        tSrS_A = cute.make_rmem_tensor(tScS_t2r.shape, self.dtype_acc)  # keys 0-31 partial
        tSrS_B = cute.make_rmem_tensor(tScS_t2r.shape, self.dtype_acc)  # keys 32-63 partial

        # P rmem -> smem copy objects (K-major SW128 64 x 64 P tile)
        universal_copy_bits = 128
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype_P, num_bits_per_copy=universal_copy_bits
        )
        smem_store_tiled = cute.make_tiled_copy_D(smem_store_atom, tmem_load_tiled)
        smem_store_thr = smem_store_tiled.get_slice(tidx)
        sP_mn_layout = cute.make_ordered_layout(self.tile_P, order=(0, 1))
        sP_smem_view = smem_store_thr.partition_D(cute.composition(sP, sP_mn_layout))

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        consumer_state_S = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_S)
        producer_state_P = pipeline.make_pipeline_state(Producer, stages=self.num_stages_P)
        producer_state_sm_stats = pipeline.make_pipeline_state(
            Producer, stages=self.num_stages_sm_stats
        )
        consumer_state_bitmask = None
        if const_expr(not self.disable_bitmask):
            consumer_state_bitmask = pipeline.make_pipeline_state(
                Consumer, stages=self.num_stages_bitmask
            )

        tile_count = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            dbg = Boolean(
                cute.arch.block_idx()[0] == 0 and tile_count == 0 and tidx == 0
            )
            dbg_tile = Boolean(cute.arch.block_idx()[0] == 0 and tile_count < 8 and tidx == 0)
            tile_row = self.num_n_blocks + tile_count
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 5] = cute.arch.clock64()
            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=self.rescale_threshold if const_expr(self.dtype_Qv.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step_fn = partial(
                self.softmax_step,
                softmax,
                sRowMax,
                sScale,
                sBitmask,
                sX,
                tStS0_t2r,
                tStS1_t2r,
                tSrS_t2r,
                tSrS_A,
                tSrS_B,
                sP_smem_view,
                tmem_load_thr,
                smem_store_thr,
                pipeline_S,
                pipeline_P,
                pipeline_sm_stats,
                pipeline_bitmask,
                tidx,
                warp_idx,
                mDebugClock,
            )

            (
                consumer_state_S,
                producer_state_P,
                producer_state_sm_stats,
                consumer_state_bitmask,
            ) = softmax_step_fn(
                consumer_state_S,
                producer_state_P,
                producer_state_sm_stats,
                consumer_state_bitmask,
                0,
                True,
                dbg,
                Int32(0),
            )
            (
                consumer_state_S,
                producer_state_P,
                producer_state_sm_stats,
                consumer_state_bitmask,
            ) = softmax_step_fn(
                consumer_state_S,
                producer_state_P,
                producer_state_sm_stats,
                consumer_state_bitmask,
                1,
                False,
                dbg,
                Int32(1),
            )
            blk = Int32(2)
            for _ in cutlass.range(self.num_n_blocks // 2 - 1, unroll=1):
                # the row-max exchange slots alternate with the block parity (compile-time here)
                for parity in cutlass.range_constexpr(2):
                    (
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                        consumer_state_bitmask,
                    ) = softmax_step_fn(
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                        consumer_state_bitmask,
                        parity,
                        False,
                        dbg,
                        blk + parity,
                    )
                blk += 2

            # write row max and sum to smem (the row max is also the LSE input)
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 6] = cute.arch.clock64()
            sRowSum[tidx % self.cta_tile_m, warp_idx // self.threads_per_row] = softmax.row_sum[0]
            if tidx < self.cta_tile_m:
                sRowMax[tidx, 0] = softmax.row_max[0]
            self.sm_stats_barrier_full.arrive()

            work_tile = tile_scheduler.advance_to_next_work()
            tile_count += 1
            self.sm_stats_barrier_empty.arrive_and_wait()
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 7] = cute.arch.clock64()

        pipeline_P.producer_tail(producer_state_P)
        pipeline_sm_stats.producer_tail(producer_state_sm_stats)

    @cute.jit
    def softmax_step(
        self,
        softmax: SoftmaxSm100,
        sRowMax: cute.Tensor,
        sScale: cute.Tensor,
        sBitmask: cute.Tensor,
        sX: cute.Tensor,
        tStS0_t2r: cute.Tensor,
        tStS1_t2r: cute.Tensor,
        tSrS_t2r: cute.Tensor,
        tSrS_A: cute.Tensor,
        tSrS_B: cute.Tensor,
        sP_smem_view: cute.Tensor,
        tmem_load_thr: cute.CopyAtom,
        smem_store_thr: cute.CopyAtom,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        tidx: Int32,
        warp_idx: Int32,
        mDebugClock: Optional[cute.Tensor],
        consumer_state_S: pipeline.PipelineState,
        producer_state_P: pipeline.PipelineState,
        producer_state_sm_stats: pipeline.PipelineState,
        consumer_state_bitmask: Optional[pipeline.PipelineState],
        parity: cutlass.Constexpr[int],
        is_first: cutlass.Constexpr[bool],
        dbg: Boolean,
        dbg_row: Int32,
    ):
        row = tidx % self.cta_tile_m
        half = warp_idx // self.threads_per_row  # 0: keys 0-31, 1: keys 32-63 of the block (warp-uniform)
        tSrP = cute.make_rmem_tensor(tSrS_t2r.shape, self.dtype_P)
        rP_smem_view = smem_store_thr.retile(tSrP)

        # Everything that does not depend on S is waited for / read before the S wait: the bitmask
        # word (produced with the gather) and the stats stage (released by the correction two
        # blocks ago) come for free here instead of on the P hand-off path.
        if const_expr(not self.disable_bitmask):
            pipeline_bitmask.consumer_wait(consumer_state_bitmask)
            bitmask = sBitmask[half, consumer_state_bitmask.index]
        pipeline_sm_stats.producer_acquire(producer_state_sm_stats)

        pipeline_S.consumer_wait(consumer_state_S)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 4] = cute.arch.clock64()
        # this lane half's partial sums: keys 0-31 (tStS0 -> tSrS_A) and keys 32-63 (tStS1 ->
        # tSrS_B); threads of half 0 keep keys 0-31 and hand keys 32-63 to their partner (thread
        # ^ 64), half 1 the reverse. Both loads are unconditional (register tensors written inside
        # a dynamic branch miscompile); the ownership is a per-element select.
        cute.copy(tmem_load_thr, tStS0_t2r, tSrS_A)
        cute.copy(tmem_load_thr, tStS1_t2r, tSrS_B)
        cute.arch.fence_view_async_tmem_load()
        pipeline_S.consumer_release(consumer_state_S)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 7] = cute.arch.clock64()

        # lane-half sum through the exchange buffer (slot = the partner's thread index); a thread's
        # write of block n+1 cannot overtake its partner's read of block n: the read precedes the
        # partner's arrival at block n's row-max barrier, which this thread passes first
        partner = tidx ^ self.cta_tile_m
        keep_lo = half == 0
        # warps w and w ^ 2 hold the two lane halves of the same rows: they synchronize pairwise
        # (64 threads, barrier id by warp parity) for both exchanges of the step
        pair_bar = Int32(self.pair_barrier_id0 + warp_idx % 2)
        for j in cutlass.range_constexpr(cute.size(tSrS_A)):
            sX[partner, j] = tSrS_B[j] if keep_lo else tSrS_A[j]
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 18] = cute.arch.clock64()
        pair_barrier_sync(pair_bar, 2 * cute.arch.WARP_SIZE)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 19] = cute.arch.clock64()
        for j in cutlass.range_constexpr(cute.size(tSrS_A)):
            mine = tSrS_A[j] if keep_lo else tSrS_B[j]
            tSrS_t2r[j] = mine + sX[tidx, j]
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 8] = cute.arch.clock64()

        if const_expr(not self.disable_bitmask):
            # key j of this thread's 32 is valid iff bit j; -1 sentinels / keys past the causal
            # limit were zero-filled by the gather and are masked to -inf here. The word is
            # uniform over the 64 threads of a key half, so the all-valid case (every block
            # of a long-context token) skips the 32 selects with a uniform branch.
            if bitmask != Uint32(0xFFFFFFFF):
                for j in cutlass.range_constexpr(cute.size(tSrS_t2r)):
                    keep = Boolean((bitmask >> j) & 1)
                    tSrS_t2r[j] = tSrS_t2r[j] if keep else -Float32.inf

        # threadwise row max, then the 2-thread exchange through smem; the exchange slots
        # alternate with the block parity, so the barrier below is the only one per block besides
        # the lane-half exchange (a thread cannot overwrite a slot its partner still reads: the
        # partner's read of block n precedes its arrival at block n+1's barriers, which this
        # thread passes before writing block n+2)
        row_max = softmax.compute_row_max_local(tSrS_t2r.load(), is_first)
        xcol = self.threads_per_row * parity
        sRowMax[row, xcol + half] = row_max
        pair_barrier_sync(pair_bar, 2 * cute.arch.WARP_SIZE)
        if const_expr(not self.disable_bitmask):
            pipeline_bitmask.consumer_release(consumer_state_bitmask)
        row_max0 = sRowMax[row, xcol]
        row_max1 = sRowMax[row, xcol + 1]
        row_max = max(row_max0, row_max1)
        row_max, acc_scale = softmax.update_row_max_from_local(row_max, is_first)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 9] = cute.arch.clock64()

        # note: acc_scales agree for the two threads of a row (stage acquired above)
        if warp_idx < self.threads_per_row:
            sScale[row, producer_state_sm_stats.index] = acc_scale
        pipeline_sm_stats.producer_commit(producer_state_sm_stats)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 10] = cute.arch.clock64()

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        softmax.apply_exp2_convert(tSrS_t2r, tSrP)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 11] = cute.arch.clock64()
        # the single P buffer is released by PV(n-1), issued after S(n): long done by now
        pipeline_P.producer_acquire(producer_state_P)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 12] = cute.arch.clock64()
        cute.copy(smem_store_thr, rP_smem_view, sP_smem_view)
        cute.arch.fence_view_async_shared()
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 13] = cute.arch.clock64()
        pipeline_P.producer_commit(producer_state_P)
        if const_expr(mDebugClock is not None):
            if dbg:
                mDebugClock[dbg_row, 5] = cute.arch.clock64()

        consumer_state_S.advance()
        producer_state_P.advance()
        producer_state_sm_stats.advance()
        if const_expr(not self.disable_bitmask):
            consumer_state_bitmask.advance()

        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)

        return consumer_state_S, producer_state_P, producer_state_sm_stats, consumer_state_bitmask

    # ------------------------------------------------------------------------------------------
    # correction / epilogue warps (4-7)
    # ------------------------------------------------------------------------------------------
    @cute.jit
    def correction_loop(
        self,
        softmax_scale_log2: Float32,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        tiled_copy_O_r2g: cute.TiledCopy,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        learnable_sink: Optional[cute.Tensor] = None,
        mOlo: Optional[cute.Tensor] = None,
        mDebugClock: Optional[cute.Tensor] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % self.num_epilogue_threads

        tOtO0 = tOtO0[(None, None), 0, 0]  # (64, (128, 2))
        tOtO1 = tOtO1[(None, None), 0, 0]
        tOtOs = [tOtO0, tOtO1]

        corr_tile_size = math.gcd(32, self.tmem_cols_Oi)
        tmem_load_atom_O = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.dtype_acc
        )
        tmem_store_atom_O = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.dtype_acc
        )
        thr_tmem_load_O = tcgen05.make_tmem_copy(tmem_load_atom_O, tOtO0).get_slice(tidx)
        thr_tmem_store_O = tcgen05.make_tmem_copy(tmem_store_atom_O, tOtO0).get_slice(tidx)
        tOtOs_t2r = [thr_tmem_load_O.partition_S(tOtOs[split]) for split in range(self.num_hdimv_splits)]
        tOtOs_r2t = [thr_tmem_store_O.partition_D(tOtOs[split]) for split in range(self.num_hdimv_splits)]

        cOi = cute.make_identity_tensor((self.cta_tile_m, self.hdimv_split))
        thr_tiled_copy_O_r2g = tiled_copy_O_r2g.get_slice(tidx)
        tOicOi = thr_tiled_copy_O_r2g.partition_S(cOi)
        tOicOi_t2r = thr_tmem_load_O.partition_D(tOicOi[(None, None), 0, 0])

        pipelines_O = [pipeline_O0, pipeline_O1]
        Consumer = pipeline.PipelineUserType.Consumer
        consumer_state_O0 = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Oi)
        consumer_state_O1 = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Oi)
        consumer_state_sm_stats = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_sm_stats)

        do_correction_rescale = partial(
            self.correction_rescale, thr_tmem_load_O, thr_tmem_store_O, tOicOi_t2r
        )

        tile_count = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            dbg_tile = Boolean(cute.arch.block_idx()[0] == 0 and tile_count < 8 and tidx == 0)
            tile_row = self.num_n_blocks + tile_count
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 8] = cute.arch.clock64()
            cluster_m_block, head_idx, batch_idx = self.tile_coords(work_tile, mCuSeqlensQ)
            if const_expr(self.use_packed_varlen_sched):
                cluster_m_block -= mCuSeqlensQ[batch_idx]
            cta_m_block = cluster_m_block
            seqlen = SeqlenInfoCls(batch_idx)
            consumer_states_O = [consumer_state_O0, consumer_state_O1]

            # first block: acc_scale is 0 by construction, nothing to rescale
            pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
            pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
            consumer_state_sm_stats.advance()
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 9] = cute.arch.clock64()

            for _ in cutlass.range(self.num_n_blocks - 1, unroll=1):
                pipeline_sm_stats.consumer_wait(consumer_state_sm_stats)
                scale = sScale[tidx % self.cta_tile_m, consumer_state_sm_stats.index]
                should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                pipeline_sm_stats.consumer_release(consumer_state_sm_stats)
                consumer_state_sm_stats.advance()
                for split in cutlass.range_constexpr(self.num_hdimv_splits):
                    consumer_state_Oi = consumer_states_O[split]
                    pipelines_O[split].consumer_wait(consumer_state_Oi)
                    if should_rescale:
                        do_correction_rescale(tOtOs_t2r[split], tOtOs_r2t[split], scale)
                    pipelines_O[split].consumer_release(consumer_state_Oi)
                    consumer_state_Oi.advance()
                    consumer_states_O[split] = consumer_state_Oi

            # (seqlen_q, hdimv) -> (cta_tile_m, hdimv//2, 2)
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=False)[None, None, head_idx]
            gO = cute.local_tile(mO_cur, (self.cta_tile_m, self.hdimv_split), (cta_m_block, None))
            tOgO = thr_tiled_copy_O_r2g.partition_D(gO)
            tOgOlo = None
            if const_expr(mOlo is not None):
                mOlo_cur = seqlen.offset_batch_Q(mOlo, batch_idx, dim=3, ragged=False)[None, None, head_idx]
                gOlo = cute.local_tile(mOlo_cur, (self.cta_tile_m, self.hdimv_split), (cta_m_block, None))
                tOgOlo = thr_tiled_copy_O_r2g.partition_D(gOlo)

            self.sm_stats_barrier_full.arrive_and_wait()
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 10] = cute.arch.clock64()
            row_sum0 = sRowSum[tidx % self.cta_tile_m, 0]
            row_sum1 = sRowSum[tidx % self.cta_tile_m, 1]
            row_sum = row_sum0 + row_sum1
            row_max = sRowMax[tidx % self.cta_tile_m, 0]
            if const_expr(learnable_sink is not None):
                sink_val = load_learnable_sink(
                    learnable_sink,
                    head_idx,
                    cta_m_block * self.cta_tile_m + tidx % self.cta_tile_m,
                    self.qhead_per_kvhead,
                    self.pack_gqa,
                    None,
                )
                row_max, row_sum = apply_learnable_sink(row_max, row_sum, sink_val, softmax_scale_log2)
            acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
            self.sm_stats_barrier_empty.arrive()

            seqlen_q = seqlen.seqlen_q * self.qhead_per_kvhead  # packed (head, token) rows

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset(((0, seqlen.offset_q),), mLSE[None, head_idx])
                gLSE = cute.local_tile(mLSE_cur, (self.cta_tile_m,), (cta_m_block,))
                if tidx < self.cta_tile_m:
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + cute.math.log2(row_sum, fastmath=True)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if tidx < seqlen_q - cta_m_block * self.cta_tile_m:
                        gLSE[tidx] = lse

            row_idx = cta_m_block * self.cta_tile_m + tOicOi[0][0]
            store_row = row_idx < seqlen_q

            # O (and the o_lo residual) are streamed 32 TMEM columns at a time: one t2r, the
            # fp32 scale, the bf16 output and its rounding residual, 4 x 16-B stores each. Holding
            # the whole 64 x 256 fp32 split in registers (128 per thread at the 128-register cap)
            # plus its bf16 copy and the per-atom gmem addresses spilled ~500 registers and made
            # the epilogue ~50k cycles, stalling the next tile's first P.V on the O release.
            tOrO_chunk_f32 = cute.make_rmem_tensor_like(tOicOi_t2r[None, None, 0], self.dtype_acc)
            tOrO_chunk = cute.make_rmem_tensor_like(tOrO_chunk_f32, self.dtype_O)
            elems_per_store = self.o_store_bits // self.dtype_O.width
            n_atoms = cute.size(tOrO_chunk) // elems_per_store
            tOrO_chunk_v = cute.make_tensor(
                tOrO_chunk.iterator, cute.make_layout((elems_per_store, n_atoms))
            )
            if const_expr(mOlo is not None):
                tOrOres_lo = cute.make_rmem_tensor_like(tOrO_chunk_f32, self.dtype_O)
                tOrOres_lo_v = cute.make_tensor(
                    tOrOres_lo.iterator, cute.make_layout((elems_per_store, n_atoms))
                )
            for split in cutlass.range_constexpr(self.num_hdimv_splits):
                consumer_state_Oi = consumer_states_O[split]
                pipelines_O[split].consumer_wait(consumer_state_Oi)
                tOgO_cur = tOgO[None, None, None, split]
                if const_expr(mOlo is not None):
                    tOgOlo_cur = tOgOlo[None, None, None, split]
                for i in cutlass.range_constexpr(cute.size(tOtOs_t2r[split], mode=[2])):
                    cute.copy(thr_tmem_load_O, tOtOs_t2r[split][None, None, i], tOrO_chunk_f32)
                    o_f32 = tOrO_chunk_f32.load() * scale
                    o_bf16 = o_f32.to(self.dtype_O)
                    tOrO_chunk.store(o_bf16)
                    if const_expr(mOlo is not None):
                        tOrOres_lo.store((o_f32 - o_bf16.to(self.dtype_acc)).to(self.dtype_O))
                    if store_row:
                        for j in cutlass.range_constexpr(n_atoms):
                            if const_expr(self.debug_epilogue != "skip_all_stores"):
                                cute.copy(
                                    thr_tiled_copy_O_r2g,
                                    tOrO_chunk_v[None, j],
                                    tOgO_cur[(None, i * n_atoms + j), 0, 0],
                                )
                            if const_expr(mOlo is not None and self.debug_epilogue is None):
                                cute.copy(
                                    thr_tiled_copy_O_r2g,
                                    tOrOres_lo_v[None, j],
                                    tOgOlo_cur[(None, i * n_atoms + j), 0, 0],
                                )
                # release this split's TMEM as soon as it is drained: the next tile's first P.V
                # acquires O0 first, so it overlaps this tile's second split
                cute.arch.fence_view_async_tmem_load()
                pipelines_O[split].consumer_release(consumer_state_Oi)
                consumer_state_Oi.advance()
                consumer_states_O[split] = consumer_state_Oi

            consumer_state_O0, consumer_state_O1 = consumer_states_O
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 11] = cute.arch.clock64()

            work_tile = tile_scheduler.advance_to_next_work()
            if const_expr(mDebugClock is not None):
                if dbg_tile:
                    mDebugClock[tile_row, 12] = cute.arch.clock64()
            tile_count += 1

    @cute.jit
    def correction_rescale(
        self,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tOcO_t2r: cute.Tensor,
        tOtO_t2r: cute.Tensor,
        tOtO_r2t: cute.Tensor,
        scale: Float32,
    ):
        tOrO_t2r_frg = cute.make_rmem_tensor_like(tOcO_t2r[None, None, 0], self.dtype_acc)
        for i in cutlass.range_constexpr(cute.size(tOtO_t2r, mode=[2])):
            tOtO_t2r_cur = tOtO_t2r[None, None, i]
            tOtO_r2t_cur = tOtO_r2t[None, None, i]
            cute.copy(thr_tmem_load, tOtO_t2r_cur, tOrO_t2r_frg)
            for j in cutlass.range(0, cute.size(tOrO_t2r_frg), 2, unroll_full=True):
                tOrO_t2r_frg[j], tOrO_t2r_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_t2r_frg[j], tOrO_t2r_frg[j + 1]), (scale, scale)
                )
            cute.copy(thr_tmem_store, tOrO_t2r_frg, tOtO_r2t_cur)
        cute.arch.fence_view_async_tmem_store()
