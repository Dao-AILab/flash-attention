"""Optimized SM120 forward kernel definition.

This file contains a full SM120 kernel class implementation (derived from
FlashAttentionForwardSm80) with fixed behavior:
- Non-causal/non-local: persistent scheduling with single-tile scheduler.
- Causal/local: non-persistent scheduling with LPT ordering.
"""

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional, List
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Constexpr, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.arch import ProxyKind, SharedSpace
import cutlass.utils as utils_basic
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.pipeline import Agent, CooperativeGroup

from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import utils
from flash_attn.cute import copy_utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax, apply_score_mod_inner
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    produce_block_sparse_loads,
    consume_block_sparse_loads,
)
from flash_attn.cute import pipeline
from flash_attn.cute.pack_gqa import PackGQA, make_packgqa_tiled_tma_atom
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80


class FlashAttentionForwardSm120TMAOptimized(FlashAttentionForwardSm80):

    arch = 120

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = False,
        tile_m: int = None,
        tile_n: int = None,
        num_stages: int = None,
        num_threads: int = 160,
        use_tma_Q: bool = True,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        direct_q_gmem_regs_decode: bool = False,
    ):
        # SM120 shared memory budget - max per block is ~101KB based on empirical testing
        # (99KB documented limit may be conservative). We account for:
        # - Pipeline barrier arrays (num_stages * 2 * 8 bytes = ~32 bytes)
        # - Alignment padding (4 buffers * 1024 bytes = 4KB)
        self.SM120_SMEM_BUDGET_BYTES = 101 * 1024
        # Store basic config
        self.dtype = dtype
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == self.head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        # SM120 uses warp-level MMA (no warpgroup ops), and we keep all MMA operands in
        # registers for both QK and PV paths by default.
        self.Q_in_regs = True
        self.K_in_regs = True
        # Decode-only optimization: in causal q1 mode, load Q directly from GMEM to
        # registers and reserve shared memory primarily for staged KV tiles.
        self.use_direct_q_gmem_regs_decode = direct_q_gmem_regs_decode
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.alignment_width = 1024  # bytes
        self.buffer_align_bytes = cutlass.const_expr(self.alignment_width)
        self.use_tma_Q = use_tma_Q
        self.use_tma_O = True
        self.unroll_consumer = 0
        self.unroll_producer_load = 0
        self.unroll_producer_sync = 0

        # KV TMA always uses shared-KV SMEM on SM120.
        self.share_kv_smem = True

        # Pad head dimensions to multiples of 16
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.tile_hdimv = int(math.ceil(self.head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = self.head_dim_v != self.tile_hdimv

        # MMA instruction shape
        self.mma_inst_mnk = (16, 8, 16)
        
        bytes_per_elem = 2  # fp16/bf16
        # Single-tile SM120 kernel: Q staging is dead by epilogue time, so O can reuse
        # the same SMEM buffer even when the final O write uses TMA.
        self.alias_sO_with_sQ = True
        # Decode direct-Q mode keeps only a KV shared buffer; otherwise we keep
        # both KV and aliased Q/O shared buffers.
        num_aligned_buffers = 1 if self.use_direct_q_gmem_regs_decode else 2
        alignment_overhead = num_aligned_buffers * self.alignment_width + 64

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_stages = num_stages

        if use_tma_Q:
            assert self.tile_m % self.qhead_per_kvhead == 0, "tile_m must be a multiple of qhead_per_kvhead when using TMA Q"

        # Optimized SM120 launch topology is fixed to 1 DMA warp + 4 MMA warps.
        self.num_producer_warps = 1
        self.num_mma_warps = 4
        expected_num_threads = (self.num_producer_warps + self.num_mma_warps) * 32
        if num_threads is None:
            num_threads = expected_num_threads
        if num_threads != expected_num_threads:
            raise ValueError(
                "SM120 config invalid: num_threads must equal "
                f"(num_dma_warps + num_mma_warps) * 32 = {expected_num_threads}, "
                f"got {num_threads}."
            )

        self.k_num_stages = self.num_stages
        self.v_num_stages = self.num_stages

        # Match SM100-style shared K/V stage-offset scheduling when stage counts align.
        self.share_kv_tma_stage_offset = self.num_stages > 1
        self.shared_kv_mbar = self.share_kv_tma_stage_offset
        self.use_tma_Q_mainloop = self.use_tma_Q and (not self.use_direct_q_gmem_regs_decode)
        # In shared-KV stage-offset mode, piggyback the first Q TMA transaction on K's mbarrier.
        self.unify_qk_tma_barrier = self.use_tma_Q_mainloop and self.shared_kv_mbar

        # SM100-style mbarrier layout (no TMEM fields on SM120).
        self.mbar_load_q_full_offset = 0
        q_mbar_slots = 0 if self.unify_qk_tma_barrier else (2 if self.use_tma_Q_mainloop else 0)
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + (1 if q_mbar_slots else 0)
        self.mbar_load_k_full_offset = self.mbar_load_q_full_offset + q_mbar_slots
        self.mbar_load_k_empty_offset = self.mbar_load_k_full_offset + self.k_num_stages
        if self.shared_kv_mbar:
            self.mbar_load_v_full_offset = self.mbar_load_k_full_offset
            self.mbar_load_v_empty_offset = self.mbar_load_k_empty_offset
            self.mbar_total = self.mbar_load_k_empty_offset + self.k_num_stages
        else:
            self.mbar_load_v_full_offset = self.mbar_load_k_empty_offset + self.k_num_stages
            self.mbar_load_v_empty_offset = self.mbar_load_v_full_offset + self.v_num_stages
            self.mbar_total = self.mbar_load_v_empty_offset + self.v_num_stages
        
        # Calculate and verify shared memory usage for shared-KV path.
        kv_elems = max(self.tile_hdim, self.tile_hdimv)
        kv_smem = self.tile_n * kv_elems * self.num_stages * bytes_per_elem
        if self.use_direct_q_gmem_regs_decode:
            # Q staging is removed. O is stored by reusing KV shared storage.
            qo_smem = 0
            kv_smem = max(kv_smem, self.tile_m * self.tile_hdimv * bytes_per_elem)
        else:
            q_elems = self.tile_hdim
            o_elems = 0 if self.alias_sO_with_sQ else self.tile_hdimv
            qo_smem = self.tile_m * (q_elems + o_elems) * bytes_per_elem
        barrier_smem = self.mbar_total * 8  # Int64 barrier slots
        total_smem = kv_smem + qo_smem + alignment_overhead + barrier_smem
        
        if total_smem > self.SM120_SMEM_BUDGET_BYTES:
            raise ValueError(
                f"SM120 kernel configuration exceeds shared memory budget: "
                f"tile_m={self.tile_m}, tile_n={self.tile_n}, hdim={self.tile_hdim}, "
                f"stages={self.num_stages} requires ~{total_smem // 1024}KB, limit is ~101KB"
            )
        
        mma_m = self.mma_inst_mnk[0]
        mma_m_factor = self.num_mma_warps
        if self.tile_m % (mma_m_factor * mma_m) != 0:
            raise ValueError(
                "SM120 config invalid: tile_m must be divisible by "
                "(mma_m_factor * mma_m). "
                f"tile_m={self.tile_m}, mma_m_factor={mma_m_factor}, "
                f"mma_m={mma_m}."
            )

        self.num_threads = expected_num_threads
        # Cluster shape (single CTA for now, no multicast)
        self.cluster_shape = (1, 1, 1)

        # Optimized defaults baked into the kernel behavior:
        # - non-causal/non-local: persistent + single scheduler
        # - causal/local: non-persistent + LPT scheduler
        self.use_persistent_schedule = (not self.is_causal) and (not self.is_local)
        self.persistent_noncausal_scheduler = (
            "single" if self.use_persistent_schedule else "lpt"
        )
    
    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=True,
        use_tma_Q: bool = False,
        pack_gqa: bool = False,
        decode_q1_direct_q_g2r: bool = False,
    ) -> bool:
        """Check whether the optimized SM120 forward kernel is implementable."""
        del is_causal, Q_in_regs, pack_gqa
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim is None or head_dim_v is None:
            return False
        if head_dim % 8 != 0 or head_dim_v % 8 != 0:
            return False
        if tile_m is None or tile_n is None or num_stages is None:
            return False
        if tile_m <= 0 or tile_n <= 0 or num_stages <= 0:
            return False
        if tile_n % 16 != 0:
            return False
        # Optimized SM120 launch topology is fixed to 1 DMA warp + 4 MMA warps.
        expected_num_threads = (1 + 4) * 32
        if num_threads != expected_num_threads:
            return False

        # MMA shape requires tile_m to be divisible by (num_mma_warps * mma_m) = 64.
        if tile_m % (4 * 16) != 0:
            return False

        tile_hdim = int(math.ceil(head_dim / 16) * 16)
        tile_hdimv = int(math.ceil(head_dim_v / 16) * 16)
        bytes_per_elem = 2  # fp16/bf16
        # shared_kv_smem=True and alias_sO_with_sQ=True in optimized SM120.
        kv_smem = tile_n * max(tile_hdim, tile_hdimv) * num_stages * bytes_per_elem
        o_smem = tile_m * tile_hdimv * bytes_per_elem
        if decode_q1_direct_q_g2r:
            # Decode q1 direct-Q path aliases O with KV shared storage and keeps only
            # one aligned shared buffer.
            qo_smem = 0
            kv_smem = max(kv_smem, o_smem)
            alignment_overhead = 1 * 1024 + 64
            use_tma_Q_effective = False
        else:
            qo_smem = tile_m * tile_hdim * bytes_per_elem
            alignment_overhead = 2 * 1024 + 64
            use_tma_Q_effective = use_tma_Q

        k_num_stages = num_stages
        v_num_stages = num_stages
        shared_kv_mbar = (num_stages > 1) and (k_num_stages == v_num_stages)
        unify_qk_tma_barrier = use_tma_Q_effective and shared_kv_mbar
        q_mbar_slots = 0 if unify_qk_tma_barrier else (2 if use_tma_Q_effective else 0)
        mbar_load_k_full_offset = q_mbar_slots
        mbar_load_k_empty_offset = mbar_load_k_full_offset + k_num_stages
        if shared_kv_mbar:
            mbar_total = mbar_load_k_empty_offset + k_num_stages
        else:
            mbar_load_v_full_offset = mbar_load_k_empty_offset + k_num_stages
            mbar_load_v_empty_offset = mbar_load_v_full_offset + v_num_stages
            mbar_total = mbar_load_v_empty_offset + v_num_stages
        barrier_smem = mbar_total * 8

        total_smem = kv_smem + qo_smem + alignment_overhead + barrier_smem
        return total_smem <= 101 * 1024

    def _setup_attributes(self, use_decode_q1_direct_q_g2r: bool = False):
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        # Epilogue only involves MMA warps (not DMA warp).
        self.num_epilogue_threads = self.num_mma_warps * 32
        self._setup_layouts()
        if use_decode_q1_direct_q_g2r:
            SharedStorage = self._get_shared_storage_cls_decode_q1()
        else:
            SharedStorage = self._get_shared_storage_cls()
        return tiled_mma_qk, tiled_mma_pv, SharedStorage

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
    ):
        FlashAttentionForwardSm80.load_Q(
            self,
            gmem_thr_copy,
            gQ,
            sQ,
            block,
            seqlen,
            headdim,
        )

    @cute.jit
    def load_K(
        self,
        tma_atom: cute.CopyAtom,
        tKgK_tma: cute.Tensor,
        tKsK_tma: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        smem_pipe_phase: Int32,
        tma_bar_ptr: cutlass.Pointer,
    ):
        self.load_tma_partitioned(
            tma_atom,
            tKgK_tma,
            tKsK_tma,
            src_idx=block,
            dst_idx=smem_pipe_write,
            dst_phase=smem_pipe_phase,
            tma_bar_ptr=tma_bar_ptr,
        )

    @cute.jit
    def load_V(
        self,
        tma_atom: cute.CopyAtom,
        tVgV_tma: cute.Tensor,
        tVsV_tma: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        smem_pipe_phase: Int32,
        tma_bar_ptr: cutlass.Pointer,
    ):
        self.load_tma_partitioned(
            tma_atom,
            tVgV_tma,
            tVsV_tma,
            src_idx=block,
            dst_idx=smem_pipe_write,
            dst_phase=smem_pipe_phase,
            tma_bar_ptr=tma_bar_ptr,
        )

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric] | None,
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None = None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None = None,
        mSeqUsedQ_type: Type[cutlass.Numeric] | None = None,
        mSeqUsedK_type: Type[cutlass.Numeric] | None = None,
        mPageTable_type: Type[cutlass.Numeric] | None = None,
    ):
        if const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Tensors must be either FP16 or BF16")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("mCuSeqlensQ tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("mCuSeqlensK tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("mSeqUsedQ tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("mSeqUsedK tensor must be Int32")
        if const_expr(mPageTable_type not in [None, Int32]):
            raise TypeError("mPageTable tensor must be Int32")
        assert mQ_type == self.dtype

    def _setup_layouts(self):
        def _make_smem_layout(
            dtype: Type[cutlass.Numeric],
            layout: LayoutEnum,
            shape: tuple[int, int],
            stage: Optional[int] = None,
            return_atom: bool = False,
        ):
            major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
            smem_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(layout, dtype, major_mode_size),
                dtype,
            )
            order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
            tiled_layout = cute.tile_to_shape(
                smem_layout_atom,
                cute.append(shape, stage) if const_expr(stage is not None) else shape,
                order=order if const_expr(stage is not None) else order[:2],
            )
            return (tiled_layout, smem_layout_atom) if return_atom else tiled_layout

        if self.use_tma_Q:
            self.sQ_layout = _make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None
            )
        else:
            self.sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
            self.sQ_layout = cute.tile_to_shape(
                self.sQ_layout_atom,
                (self.tile_m, self.tile_hdim),
                (0, 1),
            )
        self.sK_layout = _make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), self.num_stages
        )
        self.sV_layout = _make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), self.num_stages
        )
        self.sO_layout, self.sO_layout_atom = _make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdimv), None, return_atom=True
        )

    def _get_tiled_mma(self):
        # Reuse the SM80 tiled-MMA construction with SM120's MMA warp count
        # (exclude the producer warp from MMA partitioning).
        orig_num_threads = self.num_threads
        self.num_threads = self.num_mma_warps * 32
        try:
            return FlashAttentionForwardSm80._get_tiled_mma(self)
        finally:
            self.num_threads = orig_num_threads
    
    def _get_q_gmem_tiled_copy(self):
        """Create cp.async tiled copy for Q using all threads (SM80-style)."""
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQK_shape_dim_1 = self.sQ_layout_atom.outer.shape[1] // async_copy_elems
        num_q_threads = self.num_mma_warps * 32
        assert num_q_threads % tQK_shape_dim_1 == 0
        tQ_layout = cute.make_ordered_layout(
            (num_q_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0)
        )
        assert self.tile_m % tQ_layout.shape[0] == 0
        vQ_layout = cute.make_layout((1, async_copy_elems))
        return cute.make_tiled_copy_tv(atom_async_copy, tQ_layout, vQ_layout)

    def _get_o_gmem_tiled_copy(self):
        """Create cp.async tiled copy for O using thread dedicated to MMA warps."""
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tV_shape_dim_1 = self.sO_layout_atom.outer.shape[1] // async_copy_elems
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        vO_layout = cute.make_layout((1, async_copy_elems))
        return cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

    def _get_shared_storage_cls(self):
        # SM120 forward keeps a single shared-KV path and aliases O with Q.
        assert self.share_kv_smem
        assert self.alias_sO_with_sQ

        align_bytes = cutlass.const_expr(self.alignment_width)
        sQ_cosize = max(cute.cosize(self.sQ_layout), cute.cosize(self.sO_layout))
        sK_cosize = cute.cosize(self.sK_layout)
        sV_cosize = cute.cosize(self.sV_layout)
        sKV_cosize = max(sK_cosize, sV_cosize)
        sK_shared_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, sKV_cosize], align_bytes
        ]

        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, sQ_cosize], align_bytes
            ]
            sK: sK_shared_struct

        return SharedStorage

    def _get_shared_storage_cls_decode_q1(self):
        # Decode q1 direct-Q path keeps only KV shared storage. O reuses this
        # storage during epilogue after mainloop consumption is complete.
        assert self.share_kv_smem
        align_bytes = cutlass.const_expr(self.alignment_width)
        sK_cosize = cute.cosize(self.sK_layout)
        sV_cosize = cute.cosize(self.sV_layout)
        sO_cosize = cute.cosize(self.sO_layout)
        sKV_cosize = max(sK_cosize, sV_cosize, sO_cosize)
        sK_shared_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, sKV_cosize], align_bytes
        ]

        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            sK: sK_shared_struct

        return SharedStorage

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
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        stream: cuda.CUstream = None,
    ):
        if const_expr(blocksparse_tensors is not None):
            raise NotImplementedError("SM120 TMA kernel does not support block sparsity.")
        if const_expr(mPageTable is not None and (mCuSeqlensK is not None or mSeqUsedK is not None)):
            raise NotImplementedError(
                "SM120 optimized paged KV currently requires batched (non-varlen) K/V."
            )
        self._check_type(
            *(
                t.element_type
                if t is not None
                else None
                for t in (
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
                )
            )
        )
        is_split_kv = cute.rank(mO) == cute.rank(mQ) + 1
        num_splits = Int32(mO.shape[0]) if const_expr(is_split_kv) else Int32(1)
        if const_expr(is_split_kv and mLSE is None):
            raise ValueError("SplitKV mode requires mLSE partial tensor.")
        varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        use_decode_q1_direct_q_g2r = self.use_direct_q_gmem_regs_decode
        if const_expr(use_decode_q1_direct_q_g2r and varlen_q):
            raise NotImplementedError(
                "direct_q_gmem_regs_decode supports only dense inputs."
            )
        tiled_mma_qk, tiled_mma_pv, SharedStorage = self._setup_attributes(
            use_decode_q1_direct_q_g2r=use_decode_q1_direct_q_g2r
        )

        # Align strides
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            for t in (mQ, mK, mV, mO)
        ]
        page_blocks_per_entry = Int32(1)
        if const_expr(mPageTable is not None):
            page_blocks_per_entry = cute.ceil_div(cute.size(mK.shape[1]), self.tile_n)
            # Normalize paged-KV storage so source block indices are tile_n-granular.
            shape_K_blocked = (
                mK.shape[0] * page_blocks_per_entry,
                self.tile_n,
                mK.shape[2],
                mK.shape[3],
            )
            stride_K_blocked = (
                self.tile_n * mK.stride[1],
                mK.stride[1],
                mK.stride[2],
                mK.stride[3],
            )
            shape_V_blocked = (
                mV.shape[0] * page_blocks_per_entry,
                self.tile_n,
                mV.shape[2],
                mV.shape[3],
            )
            stride_V_blocked = (
                self.tile_n * mV.stride[1],
                mV.stride[1],
                mV.stride[2],
                mV.stride[3],
            )
            mK = cute.make_tensor(
                mK.iterator, cute.make_layout(shape_K_blocked, stride=stride_K_blocked)
            )
            mV = cute.make_tensor(
                mV.iterator, cute.make_layout(shape_V_blocked, stride=stride_V_blocked)
            )
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        if const_expr(not is_split_kv):
            O_layout_transpose = Q_layout_transpose
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        else:
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        mQ_tma = mQ
        if const_expr(mLSE is not None):
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (self.qhead_per_kvhead, mQ.shape[0]),
                mQ.shape[1],
                mK.shape[2],
                *mQ.shape[3:],
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mO.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed)
                )

        # TMA atoms for Q/K/V
        self.tma_copy_bytes = {}
        gmem_tiled_copy_Q, tma_atom_Q, tma_tensor_Q = None, None, None
        if const_expr(self.use_tma_Q and not use_decode_q1_direct_q_g2r):
            tma_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
            self.tma_copy_bytes["Q"] = cute.size_in_bytes(
                self.dtype, cute.select(self.sQ_layout, mode=[0, 1])
            )
            if const_expr(self.pack_gqa):
                tma_atom_Q, tma_tensor_Q = make_packgqa_tiled_tma_atom(
                    tma_copy_Q,
                    mQ_tma,
                    self.sQ_layout,
                    (self.tile_m, self.tile_hdim),
                    qhead_per_kvhead=self.qhead_per_kvhead,
                    head_idx=2,
                )
            else:
                tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
                    tma_copy_Q, mQ, self.sQ_layout, (self.tile_m, self.tile_hdim)
                )
        elif const_expr(not self.use_tma_Q):
            gmem_tiled_copy_Q = self._get_q_gmem_tiled_copy()
        tma_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            tma_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        gmem_tiled_copy_O = self._get_o_gmem_tiled_copy()
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(
            self.use_tma_O
            and not self.pack_gqa
            and not varlen_q
            and not use_decode_q1_direct_q_g2r
        ):
            tma_copy_O = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
                tma_copy_O,
                mO,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),
            )

        is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        use_persistent_schedule = self.use_persistent_schedule and const_expr(not is_varlen_q)
        if const_expr(use_persistent_schedule):
            if const_expr(self.is_causal or self.is_local):
                raise NotImplementedError(
                    "Dedicated SM120 persistent kernel currently supports only non-causal, non-local attention."
                )
            if const_expr(self.persistent_noncausal_scheduler == "lpt"):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = SingleTileScheduler
        elif const_expr(is_varlen_q):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileLPTScheduler if const_expr(self.is_causal or self.is_local) else SingleTileScheduler
            )
        num_m_blocks = cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m)
        num_heads = cute.size(mQ.shape[2])
        num_batches = (
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1)
        )
        tile_sched_args = TileSchedulerArguments(
            num_m_blocks,
            num_heads,
            num_batches,
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else cute.size(mK.shape[0])
            * cute.size(mPageTable.shape[1])
            * page_blocks_per_entry,
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            cluster_shape_mn=(self.cluster_shape[0], self.cluster_shape[1]),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            is_persistent=use_persistent_schedule,
            element_size=self.dtype.width // 8,
            is_split_kv=is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = Float32(softmax_scale * LOG2_E)
            softmax_scale = None
        else:
            softmax_scale_log2 = Float32(LOG2_E)
            softmax_scale = Float32(softmax_scale)

        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )

        self.kernel(
            gmem_tiled_copy_Q,
            tma_atom_Q,
            tma_tensor_Q if const_expr(self.use_tma_Q and not use_decode_q1_direct_q_g2r) else mQ,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            gmem_tiled_copy_O,
            tma_atom_O,
            tma_tensor_O if const_expr(tma_atom_O is not None) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            self.qhead_per_kvhead,
            num_splits,
            softmax_scale_log2, softmax_scale,
            window_size_left, window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            tiled_mma_qk, tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            use_decode_q1_direct_q_g2r,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            cluster=list(self.cluster_shape),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        tma_atom_Q: Optional[cute.CopyAtom],
        mQ: cute.Tensor,
        tma_atom_K: Optional[cute.CopyAtom],
        mK: cute.Tensor,
        tma_atom_V: Optional[cute.CopyAtom],
        mV: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        qhead_per_kvhead: cutlass.Int32,
        num_splits: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        use_decode_q1_direct_q_g2r: cutlass.Constexpr[bool],
        aux_tensors=None,
        fastdiv_mods=None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        dma_local_tidx = Int32(0)
        
        # Prefetch TMA descriptors
        if warp_idx == 0:
            if const_expr(self.use_tma_Q and not use_decode_q1_direct_q_g2r):
                cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)
        
        # Block info / seqlen
        is_split_kv = const_expr(cute.rank(mO) in (4, 5))
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

        # Allocate SMEM
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_ptr = storage.mbar_ptr.data_ptr()

        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sK.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        # Q uses SM90-style layout for TMA or SM80-style layout for cp.async fallback.
        # In decode q1 direct-Q mode, we alias Q and O onto KV shared storage.
        if const_expr(use_decode_q1_direct_q_g2r):
            sQ = storage.sK.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        elif const_expr(self.use_tma_Q):
            sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        else:
            sQ = storage.sQ.get_tensor(sQ_layout)
        if const_expr(self.alias_sO_with_sQ):
            if const_expr(use_decode_q1_direct_q_g2r):
                sO = storage.sK.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
            else:
                sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
        else:
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        # V for P*V gemm: transposed view for B operand dimension matching
        # (tile_n, tile_hdimv, stages) -> (tile_hdimv, tile_n, stages)
        sVt = layout_utils.transpose_view(sV)
        sQ_stage0 = sQ
        sO_stage0 = sO

        # SM100-style barrier layout packed in one shared mbarrier array.
        mainloop_pipeline_array_ptr_K = mbar_ptr + self.mbar_load_k_full_offset
        mainloop_pipeline_array_ptr_V = (
            mainloop_pipeline_array_ptr_K
            if const_expr(self.shared_kv_mbar)
            else mbar_ptr + self.mbar_load_v_full_offset
        )
        kv_producer_group = CooperativeGroup(Agent.Thread)
        kv_consumer_group = CooperativeGroup(Agent.Thread, self.num_mma_warps)
        sK_single = cute.slice_(sK_layout, (None, None, 0))
        sV_single = cute.slice_(sV_layout, (None, None, 0))
        tma_k_bytes = cute.size_in_bytes(self.dtype, sK_single)
        tma_v_bytes = cute.size_in_bytes(self.dtype, sV_single)
        if cute.size(self.cluster_shape) > 1:
            cute.arch.cluster_arrive_relaxed()

        cutlass.pipeline.sync(barrier_id=1)

        if cute.size(self.cluster_shape) > 1:
            cute.arch.cluster_wait()

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen_k_static = (
                mK.shape[0]
                if const_expr(mPageTable is None)
                else mK.shape[0]
                * mPageTable.shape[1]
                * (
                    mK.shape[3]
                    // (mPageTable.shape[0] * mPageTable.shape[1])
                )
            )
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                seqlen_q_static=mQ.shape[0]
                if const_expr(not self.pack_gqa)
                else mQ.shape[0][1],
                seqlen_k_static=seqlen_k_static,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                tile_m=self.tile_m,
                tile_n=self.tile_n,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                m_block,
                split_idx=split_idx,
                num_splits=num_splits,
            )
            n_block = n_block_max - 1

            kv_head_idx = head_idx if const_expr(self.pack_gqa) else head_idx // qhead_per_kvhead

            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            if const_expr(mPageTable is None):
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, kv_head_idx]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, kv_head_idx]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
            else:
                mK_cur = mK[None, None, kv_head_idx, None]
                mV_cur = mV[None, None, kv_head_idx, None]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None))
    
            # SM100-style TMA partitioning for K/V.
            tKsK_tma, tKgK_tma = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 2),
                cute.group_modes(gK, 0, 2),
            )
            tVsV_tma, tVgV_tma = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gV, 0, 2),
            )
            # Reinitialize per tile to avoid phase carryover across scheduler iterations.
            pipeline_K = pipeline.PipelineTmaAsync.create(
                num_stages=self.k_num_stages,
                producer_group=kv_producer_group,
                consumer_group=kv_consumer_group,
                tx_count=tma_k_bytes,
                barrier_storage=mainloop_pipeline_array_ptr_K,
                defer_sync=True,
            )
            pipeline_V = pipeline.PipelineTmaAsync.create(
                num_stages=self.v_num_stages,
                producer_group=kv_producer_group,
                consumer_group=kv_consumer_group,
                tx_count=tma_v_bytes,
                barrier_storage=mainloop_pipeline_array_ptr_V,
                defer_sync=False,
            )
            q_pipeline = None
            if const_expr(self.use_tma_Q and not self.unify_qk_tma_barrier and not use_decode_q1_direct_q_g2r):
                q_pipeline = pipeline.PipelineTmaAsync.create(
                    num_stages=1,
                    producer_group=CooperativeGroup(Agent.Thread),
                    consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_warps),
                    tx_count=self.tma_copy_bytes["Q"],
                    barrier_storage=mbar_ptr + self.mbar_load_q_full_offset,
                )
    
            if const_expr(self.use_tma_Q and not use_decode_q1_direct_q_g2r):
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ_stage0, single_stage=True
                )
    
            # Load Q (cp.async) BEFORE the main if/elif split
            # This ensures Q is loaded before MMA warps need it (when using SMEM staging)
            if const_expr(not self.use_tma_Q and not use_decode_q1_direct_q_g2r):
                if (
                    warp_idx >= self.num_producer_warps
                    and warp_idx < self.num_producer_warps + self.num_mma_warps
                ):
                    mma_tidx = tidx - self.num_producer_warps * 32
                    if const_expr(self.pack_gqa):
                        pack_gqa = PackGQA(
                            self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                        )
                        pack_gqa.load_Q(
                            mQ_cur, sQ_stage0, gmem_tiled_copy_Q, mma_tidx, m_block, seqlen.seqlen_q
                        )
                    else:
                        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(mma_tidx)
                        self.load_Q(
                            gmem_thr_copy_Q,
                            gQ,
                            sQ_stage0,
                            m_block,
                            seqlen=seqlen.seqlen_q,
                            headdim=mQ.shape[1],
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)

            # All threads sync after Q is loaded (cp.async only).
            if const_expr(not self.use_tma_Q and not use_decode_q1_direct_q_g2r):
                cute.arch.barrier()

            # MMA warps
            if (
                warp_idx >= self.num_producer_warps
                and warp_idx < self.num_producer_warps + self.num_mma_warps
                and n_block_max > 0
            ):
                mma_tidx = tidx - self.num_producer_warps * 32
                thr_mma_qk = tiled_mma_qk.get_slice(mma_tidx)
                thr_mma_pv = tiled_mma_pv.get_slice(mma_tidx)
    
                tSsQ = thr_mma_qk.partition_A(sQ_stage0)
                tSsK = thr_mma_qk.partition_B(sK)
                tOrVt = thr_mma_pv.partition_B(sVt)

                tSrQ = thr_mma_qk.make_fragment_A(tSsQ)
                tSrK = thr_mma_qk.make_fragment_B(tSsK[None, None, None, 0])
                tOrV = thr_mma_pv.make_fragment_B(tOrVt[None, None, None, 0])
    
                smem_copy_atom_Q = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    self.dtype,
                )
                smem_copy_atom_K = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    self.dtype,
                )
                smem_copy_atom_V = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                    self.dtype,
                )

                smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma_qk)
                smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma_qk)
                smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv)
                thr_copy_Q = smem_tiled_copy_Q.get_slice(mma_tidx)
                thr_copy_K = smem_tiled_copy_K.get_slice(mma_tidx)
                thr_copy_V = smem_tiled_copy_V.get_slice(mma_tidx)
                tSsQ_copy = thr_copy_Q.partition_S(sQ_stage0)
                tSrQ_copy = thr_copy_Q.retile(tSrQ)
                tSsK_copy = thr_copy_K.partition_S(sK)
                tSrK_copy = thr_copy_K.retile(tSrK)
                tOsVt_copy = thr_copy_V.partition_S(sVt)
    
                acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
                acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
                acc_O.fill(0.0)
                acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
    
                softmax = Softmax.create(
                    softmax_scale_log2,
                    num_rows=(acc_O.shape[0][0] * acc_O.shape[1]),
                    softmax_scale=softmax_scale if const_expr(self.score_mod is not None) else None,
                )
                softmax.reset()
                mask = AttentionMask(
                    self.tile_m,
                    self.tile_n,
                    seqlen,
                    window_size_left,
                    window_size_right,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                )

                if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                    # SM100-style interleaving: K and V consume consecutive states from one stream.
                    consumer_state_kv = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.k_num_stages
                    )
                else:
                    consumer_state_k = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.k_num_stages
                    )
                    consumer_state_v = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.v_num_stages
                    )

                num_k_blocks = cute.size(tSrQ, mode=[2])
                if const_expr(use_decode_q1_direct_q_g2r and self.Q_in_regs):
                    copy_atom_q_g2r = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.dtype,
                        num_bits_per_copy=32,
                    )
                    gmem_tiled_copy_q_g2r = cute.make_tiled_copy_A(copy_atom_q_g2r, tiled_mma_qk)
                    gmem_thr_copy_q_g2r = gmem_tiled_copy_q_g2r.get_slice(mma_tidx)
                    tQgQ_copy = gmem_thr_copy_q_g2r.partition_S(gQ)
                    tQrQ_copy = gmem_thr_copy_q_g2r.retile(tSrQ)
                    cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
                    tQcQ = gmem_thr_copy_q_g2r.partition_S(cQ)
                    t0QcQ = gmem_tiled_copy_q_g2r.get_slice(0).partition_S(cQ)
                    tQpQ = utils.predicate_k(tQcQ, limit=mQ.shape[1])
                    q_rows_total = seqlen.seqlen_q
                    if const_expr(self.pack_gqa):
                        q_rows_total *= self.qhead_per_kvhead
                    for m in cutlass.range_constexpr(cute.size(tQrQ_copy.shape[1])):
                        if t0QcQ[0, m, 0][0] < q_rows_total - m_block * self.tile_m - tQcQ[0][0]:
                            cute.copy(
                                gmem_tiled_copy_q_g2r,
                                tQgQ_copy[None, m, None],
                                tQrQ_copy[None, m, None],
                                pred=tQpQ[None, m, None] if const_expr(self.check_hdim_oob) else None,
                            )

                if const_expr(self.use_tma_Q and not use_decode_q1_direct_q_g2r):
                    if const_expr(not self.unify_qk_tma_barrier):
                        q_consumer_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Consumer, 1
                        )
                        q_pipeline.consumer_wait(
                            q_consumer_state,
                            q_pipeline.consumer_try_wait(q_consumer_state),
                        )
                        if const_expr(self.Q_in_regs):
                            for k in cutlass.range_constexpr(num_k_blocks):
                                cute.copy(
                                    smem_tiled_copy_Q,
                                    tSsQ_copy[None, None, k],
                                    tSrQ_copy[None, None, k],
                                )
                        q_pipeline.consumer_release(q_consumer_state)
                elif const_expr(not use_decode_q1_direct_q_g2r):
                    if const_expr(self.Q_in_regs):
                        for k in cutlass.range_constexpr(num_k_blocks):
                            cute.copy(smem_tiled_copy_Q, tSsQ_copy[None, None, k], tSrQ_copy[None, None, k])
    
                if n_block_max > 0:
                    # First n-block
                    current_n_block = n_block_max - 1
    
                    if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                        pipeline_K.consumer_wait(
                            consumer_state_kv,
                            pipeline_K.consumer_try_wait(consumer_state_kv),
                        )
                        stage_k = consumer_state_kv.index
                        stage_k_phase = consumer_state_kv.phase
                    else:
                        pipeline_K.consumer_wait(
                            consumer_state_k,
                            pipeline_K.consumer_try_wait(consumer_state_k),
                        )
                        stage_k = consumer_state_k.index
                        stage_k_phase = consumer_state_k.phase

                    if const_expr(
                        self.use_tma_Q
                        and self.unify_qk_tma_barrier
                        and self.Q_in_regs
                        and not use_decode_q1_direct_q_g2r
                    ):
                        # Q and first-stage K share the same TMA barrier. Once K is available,
                        # Q is guaranteed available as well.
                        for k in cutlass.range_constexpr(num_k_blocks):
                            cute.copy(
                                smem_tiled_copy_Q,
                                tSsQ_copy[None, None, k],
                                tSrQ_copy[None, None, k],
                            )
    
                    acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
                    acc_S.fill(0.0)
                    tSsK_stage = tSsK_copy[None, None, None, stage_k]

                    # Pre-copy K to registers if enabled (allows earlier SMEM release)
                    if const_expr(self.K_in_regs):
                        for k in cutlass.range_constexpr(num_k_blocks):
                            cute.copy(
                                smem_tiled_copy_K,
                                tSsK_stage[None, None, k],
                                tSrK_copy[None, None, k],
                            )

                    sm80_utils.gemm(
                        tiled_mma_qk,
                        acc_S,
                        tSrQ,
                        tSrK,
                        tSsQ_copy,
                        tSsK_stage,
                        thr_copy_Q,
                        thr_copy_K,
                        A_in_regs=cutlass.const_expr(self.Q_in_regs),
                        B_in_regs=cutlass.const_expr(self.K_in_regs),
                    )
                    if const_expr(not (self.share_kv_tma_stage_offset and self.num_stages > 1)):
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierFwd.PFull),
                            number_of_threads=self.num_threads,
                        )

                    # Release K immediately after QK GEMM so producer can refill while we run
                    # masking + softmax before the V wait point.
                    if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                        pipeline_K.consumer_release(consumer_state_kv)
                        consumer_state_kv.advance()
                        v_wait_token = pipeline_V.consumer_try_wait(consumer_state_kv)
                    else:
                        pipeline_K.consumer_release(consumer_state_k)
                        consumer_state_k.advance()
                        v_wait_token = pipeline_V.consumer_try_wait(consumer_state_v)

                    mask.apply_mask(
                        acc_S,
                        batch_idx,
                        head_idx,
                        m_block,
                        current_n_block,
                        thr_mma_qk,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                    )

                    # Rescale acc_O before P*V gemm (no-op for first block).
                    row_scale = softmax.online_softmax(
                        acc_S,
                        is_first=cutlass.const_expr(True),
                    )
                    softmax.rescale_O(acc_O, row_scale)

                    # Wait for V stage after softmax work.
                    if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                        pipeline_V.consumer_wait(
                            consumer_state_kv,
                            v_wait_token,
                        )
                        stage_v = consumer_state_kv.index
                        stage_v_phase = consumer_state_kv.phase
                    else:
                        pipeline_V.consumer_wait(
                            consumer_state_v,
                            v_wait_token,
                        )
                        stage_v = consumer_state_v.index
                        stage_v_phase = consumer_state_v.phase

                    tOsVt_stage = tOsVt_copy[None, None, None, stage_v]

                    # Convert acc_S to A fragment layout for PV gemm (back-to-back gemm pattern)
                    rP = cute.make_fragment_like(acc_S, self.dtype)
                    rP.store(acc_S.load().to(self.dtype))

                    tOrP = cute.make_tensor(
                        rP.iterator,
                        layout_utils.convert_layout_acc_frgA(rP.layout),
                    )
                    sm80_utils.gemm_rs(
                        tiled_mma_pv,
                        acc_O,
                        tOrP,
                        tOrV,
                        tOsVt_stage,
                        thr_copy_V,
                    )

                    if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                        pipeline_V.consumer_release(consumer_state_kv)
                        consumer_state_kv.advance()
                    else:
                        pipeline_V.consumer_release(consumer_state_v)
                        consumer_state_v.advance()
                    if const_expr(not (self.share_kv_tma_stage_offset and self.num_stages > 1)):
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierFwd.PEmpty),
                            number_of_threads=self.num_threads,
                        )
    
                    # Remaining n-blocks
                    for n_tile in cutlass.range(n_block_max - 1, unroll=self.unroll_consumer):
                        current_n_block = n_block_max - 2 - n_tile
    
                        if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                            pipeline_K.consumer_wait(
                                consumer_state_kv,
                                pipeline_K.consumer_try_wait(consumer_state_kv),
                            )
                            stage_k = consumer_state_kv.index
                            stage_k_phase = consumer_state_kv.phase
                        else:
                            pipeline_K.consumer_wait(
                                consumer_state_k,
                                pipeline_K.consumer_try_wait(consumer_state_k),
                            )
                            stage_k = consumer_state_k.index
                            stage_k_phase = consumer_state_k.phase
    
                        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
                        acc_S.fill(0.0)
                        tSsK_stage = tSsK_copy[None, None, None, stage_k]

                        # Pre-copy K to registers if enabled (allows earlier SMEM release)
                        if const_expr(self.K_in_regs):
                            for k in cutlass.range_constexpr(num_k_blocks):
                                cute.copy(
                                    smem_tiled_copy_K,
                                    tSsK_stage[None, None, k],
                                    tSrK_copy[None, None, k],
                                )

                        sm80_utils.gemm(
                            tiled_mma_qk,
                            acc_S,
                            tSrQ,
                            tSrK,
                            tSsQ_copy,
                            tSsK_stage,
                            thr_copy_Q,
                            thr_copy_K,
                            A_in_regs=cutlass.const_expr(self.Q_in_regs),
                            B_in_regs=cutlass.const_expr(self.K_in_regs),
                        )
                        if const_expr(not (self.share_kv_tma_stage_offset and self.num_stages > 1)):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierFwd.PFull),
                                number_of_threads=self.num_threads,
                            )

                        # Release K immediately after QK GEMM so producer can refill while
                        # mask/softmax runs.
                        if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                            pipeline_K.consumer_release(consumer_state_kv)
                            consumer_state_kv.advance()
                            v_wait_token = pipeline_V.consumer_try_wait(consumer_state_kv)
                        else:
                            pipeline_K.consumer_release(consumer_state_k)
                            consumer_state_k.advance()
                            v_wait_token = pipeline_V.consumer_try_wait(consumer_state_v)

                        if const_expr(self.is_causal):
                            mask.apply_mask(
                                acc_S,
                                batch_idx,
                                head_idx,
                                m_block,
                                current_n_block,
                                thr_mma_qk,
                                mask_seqlen=False,
                                mask_causal=self.is_causal,
                                mask_local=self.is_local,
                            )

                        # Rescale acc_O before adding new P*V contribution.
                        row_scale = softmax.online_softmax(
                            acc_S,
                            is_first=cutlass.const_expr(False),
                        )
                        softmax.rescale_O(acc_O, row_scale)

                        # Wait for V stage after softmax work.
                        if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                            pipeline_V.consumer_wait(
                                consumer_state_kv,
                                v_wait_token,
                            )
                            stage_v = consumer_state_kv.index
                            stage_v_phase = consumer_state_kv.phase
                        else:
                            pipeline_V.consumer_wait(
                                consumer_state_v,
                                v_wait_token,
                            )
                            stage_v = consumer_state_v.index
                            stage_v_phase = consumer_state_v.phase

                        tOsVt_stage = tOsVt_copy[None, None, None, stage_v]

                        # Convert acc_S to A fragment layout for PV gemm (back-to-back gemm pattern)
                        rP = cute.make_fragment_like(acc_S, self.dtype)
                        rP.store(acc_S.load().to(self.dtype))

                        tOrP = cute.make_tensor(
                            rP.iterator,
                            layout_utils.convert_layout_acc_frgA(rP.layout),
                        )
                        sm80_utils.gemm_rs(
                            tiled_mma_pv,
                            acc_O,
                            tOrP,
                            tOrV,
                            tOsVt_stage,
                            thr_copy_V,
                        )

                        if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                            pipeline_V.consumer_release(consumer_state_kv)
                            consumer_state_kv.advance()
                        else:
                            pipeline_V.consumer_release(consumer_state_v)
                            consumer_state_v.advance()
                        if const_expr(not (self.share_kv_tma_stage_offset and self.num_stages > 1)):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierFwd.PEmpty),
                                number_of_threads=self.num_threads,
                            )
    
                row_scale = softmax.finalize()
                softmax.rescale_O(acc_O, row_scale)
                lse = softmax.row_sum
                self.epilogue(
                    acc_O,
                    lse,
                    mO,
                    mLSE,
                    sO_stage0,
                    seqlen,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    tiled_mma_pv,
                    mma_tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                    split_idx,
                )
    
            # Producer warps - load K/V tiles via TMA
            elif warp_idx < self.num_producer_warps and n_block_max > 0:
                producer_warp_idx = warp_idx
                page_blocks_per_entry = (
                    Int32(1)
                    if const_expr(mPageTable is None)
                    else mK.shape[3] // (mPageTable.shape[0] * mPageTable.shape[1])
                )
                if const_expr(self.share_kv_tma_stage_offset and self.num_stages > 1):
                    # SM100-style K/V interleaving: both streams advance one shared state.
                    producer_state_kv = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.k_num_stages
                    )
                else:
                    producer_state_k = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.k_num_stages
                    )
                    producer_state_v = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.v_num_stages
                    )
                if producer_warp_idx == 0:
                    if const_expr(
                        self.use_tma_Q and not self.unify_qk_tma_barrier and not use_decode_q1_direct_q_g2r
                    ):
                        q_producer_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Producer, 1
                        )
                        q_pipeline.producer_acquire(q_producer_state)
                        load_Q(tma_bar_ptr=q_pipeline.producer_get_barrier(q_producer_state))
                        q_pipeline.producer_commit(q_producer_state)
                        q_producer_state.advance()
                    if const_expr(
                        self.shared_kv_mbar and self.unify_qk_tma_barrier and not use_decode_q1_direct_q_g2r
                    ):
                        # First stage: load Q and K together, using K pipeline's mbarrier.
                        first_n_block = n_block_max - 1
                        first_kv_block = (
                            mPageTable[batch_idx, first_n_block // page_blocks_per_entry]
                            * page_blocks_per_entry
                            + (first_n_block % page_blocks_per_entry)
                            if const_expr(mPageTable is not None)
                            else first_n_block
                        )
                        pipeline_K.producer_acquire(
                            producer_state_kv,
                            extra_tx_count=self.tma_copy_bytes["Q"],
                        )
                        load_Q(tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_kv))
                        self.load_K(
                            tma_atom_K,
                            tKgK_tma,
                            tKsK_tma,
                            block=first_kv_block,
                            smem_pipe_write=producer_state_kv.index,
                            smem_pipe_phase=producer_state_kv.phase,
                            tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_kv),
                        )
                        pipeline_K.producer_commit(producer_state_kv)
                        producer_state_kv.advance()
                        pipeline_V.producer_acquire(producer_state_kv)
                        self.load_V(
                            tma_atom_V,
                            tVgV_tma,
                            tVsV_tma,
                            block=first_kv_block,
                            smem_pipe_write=producer_state_kv.index,
                            smem_pipe_phase=producer_state_kv.phase,
                            tma_bar_ptr=pipeline_V.producer_get_barrier(producer_state_kv),
                        )
                        pipeline_V.producer_commit(producer_state_kv)
                        producer_state_kv.advance()
                        for n_tile in cutlass.range(n_block_max - 1, unroll=self.unroll_producer_load):
                            n_block_to_load = n_block_max - 2 - n_tile
                            kv_block_to_load = (
                                mPageTable[batch_idx, n_block_to_load // page_blocks_per_entry]
                                * page_blocks_per_entry
                                + (n_block_to_load % page_blocks_per_entry)
                                if const_expr(mPageTable is not None)
                                else n_block_to_load
                            )
                            pipeline_K.producer_acquire(producer_state_kv)
                            self.load_K(
                                tma_atom_K,
                                tKgK_tma,
                                tKsK_tma,
                                block=kv_block_to_load,
                                smem_pipe_write=producer_state_kv.index,
                                smem_pipe_phase=producer_state_kv.phase,
                                tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_kv),
                            )
                            pipeline_K.producer_commit(producer_state_kv)
                            producer_state_kv.advance()
                            pipeline_V.producer_acquire(producer_state_kv)
                            self.load_V(
                                tma_atom_V,
                                tVgV_tma,
                                tVsV_tma,
                                block=kv_block_to_load,
                                smem_pipe_write=producer_state_kv.index,
                                smem_pipe_phase=producer_state_kv.phase,
                                tma_bar_ptr=pipeline_V.producer_get_barrier(producer_state_kv),
                            )
                            pipeline_V.producer_commit(producer_state_kv)
                            producer_state_kv.advance()
                    else:
                        for n_tile in cutlass.range(n_block_max, unroll=self.unroll_producer_load):
                            n_block_to_load = n_block_max - 1 - n_tile
                            kv_block_to_load = (
                                mPageTable[batch_idx, n_block_to_load // page_blocks_per_entry]
                                * page_blocks_per_entry
                                + (n_block_to_load % page_blocks_per_entry)
                                if const_expr(mPageTable is not None)
                                else n_block_to_load
                            )
                            if const_expr(self.shared_kv_mbar):
                                pipeline_K.producer_acquire(producer_state_kv)
                                self.load_K(
                                    tma_atom_K,
                                    tKgK_tma,
                                    tKsK_tma,
                                    block=kv_block_to_load,
                                    smem_pipe_write=producer_state_kv.index,
                                    smem_pipe_phase=producer_state_kv.phase,
                                    tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_kv),
                                )
                                pipeline_K.producer_commit(producer_state_kv)
                                producer_state_kv.advance()
                                pipeline_V.producer_acquire(producer_state_kv)
                                self.load_V(
                                    tma_atom_V,
                                    tVgV_tma,
                                    tVsV_tma,
                                    block=kv_block_to_load,
                                    smem_pipe_write=producer_state_kv.index,
                                    smem_pipe_phase=producer_state_kv.phase,
                                    tma_bar_ptr=pipeline_V.producer_get_barrier(producer_state_kv),
                                )
                                pipeline_V.producer_commit(producer_state_kv)
                                producer_state_kv.advance()
                            else:
                                pipeline_K.producer_acquire(producer_state_k)
                                self.load_K(
                                    tma_atom_K,
                                    tKgK_tma,
                                    tKsK_tma,
                                    block=kv_block_to_load,
                                    smem_pipe_write=producer_state_k.index,
                                    smem_pipe_phase=producer_state_k.phase,
                                    tma_bar_ptr=pipeline_K.producer_get_barrier(producer_state_k),
                                )
                                pipeline_K.producer_commit(producer_state_k)
                                producer_state_k.advance()
                                if const_expr(
                                    not (self.share_kv_tma_stage_offset and self.num_stages > 1)
                                ):
                                    cute.arch.barrier(
                                        barrier_id=int(NamedBarrierFwd.PFull),
                                        number_of_threads=self.num_threads,
                                    )
                                pipeline_V.producer_acquire(producer_state_v)
                                self.load_V(
                                    tma_atom_V,
                                    tVgV_tma,
                                    tVsV_tma,
                                    block=kv_block_to_load,
                                    smem_pipe_write=producer_state_v.index,
                                    smem_pipe_phase=producer_state_v.phase,
                                    tma_bar_ptr=pipeline_V.producer_get_barrier(producer_state_v),
                                )
                                pipeline_V.producer_commit(producer_state_v)
                                producer_state_v.advance()
                                if const_expr(
                                    not (self.share_kv_tma_stage_offset and self.num_stages > 1)
                                ):
                                    cute.arch.barrier(
                                        barrier_id=int(NamedBarrierFwd.PEmpty),
                                        number_of_threads=self.num_threads,
                                    )
                    if const_expr(self.shared_kv_mbar):
                        # Avoid relying on PipelineTmaAsync.producer_tail(loc=..., ip=...)
                        # because local pipeline overrides may not accept loc/ip in producer_acquire.
                        for _ in cutlass.range(self.k_num_stages - 1, unroll_full=True):
                            producer_state_kv.advance()
                        pipeline_K.producer_acquire(producer_state_kv)
                    else:
                        for _ in cutlass.range(self.k_num_stages - 1, unroll_full=True):
                            producer_state_k.advance()
                        pipeline_K.producer_acquire(producer_state_k)
                        for _ in cutlass.range(self.v_num_stages - 1, unroll_full=True):
                            producer_state_v.advance()
                        pipeline_V.producer_acquire(producer_state_v)
                else:
                    # Keep extra producer warps synchronized with the producer hand-off.
                    if const_expr(
                        not (self.share_kv_tma_stage_offset and self.num_stages > 1)
                    ):
                        for _ in cutlass.range(n_block_max, unroll=self.unroll_producer_sync):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierFwd.PFull),
                                number_of_threads=self.num_threads,
                            )
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierFwd.PEmpty),
                                number_of_threads=self.num_threads,
                            )
            cute.arch.barrier()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        return

    @cute.jit
    def load_tma_partitioned(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        src_idx: Int32,
        dst_idx: Int32,
        dst_phase: Int32,
        tma_bar_ptr: cutlass.Pointer,
    ):
        tXsX_cur = tXsX[None, dst_idx]
        cute.copy(
            tma_atom,
            tXgX[None, src_idx],
            tXsX_cur,
            tma_bar_ptr=tma_bar_ptr,
        )

    @cute.jit
    def store_lse(
        self,
        lse: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        seqlen: SeqlenInfoQK,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        if const_expr(mLSE is None):
            return
        is_split_kv = const_expr(
            (seqlen.has_cu_seqlens_q and cute.rank(mLSE) == 3)
            or (not seqlen.has_cu_seqlens_q and cute.rank(mLSE) == 4)
        )
        if const_expr(not seqlen.has_cu_seqlens_q):
            mLSE_cur = (
                mLSE[None, head_idx, batch_idx, split_idx]
                if const_expr(is_split_kv)
                else mLSE[None, head_idx, batch_idx]
            )
        else:
            offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
            mLSE_base = (
                mLSE[None, head_idx, split_idx]
                if const_expr(is_split_kv)
                else mLSE[None, head_idx]
            )
            mLSE_cur = cute.domain_offset((offset,), mLSE_base)
        if const_expr(not self.pack_gqa):
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
            gLSE_expanded_layout = cute.append(
                gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
            )
            gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
            thr_mma = tiled_mma.get_slice(tidx)
            taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
            assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
            cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
            taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
            t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
            if taccOcO[0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                    if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]:
                        taccOgLSE[m, 0] = lse[m]
        else:
            pack_gqa = PackGQA(
                self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
            )
            pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

    @cute.jit
    def store_O_from_smem(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )
        is_split_kv = const_expr(
            (seqlen.has_cu_seqlens_q and cute.rank(mO) == 4)
            or (not seqlen.has_cu_seqlens_q and cute.rank(mO) == 5)
        )
        if const_expr(not seqlen.has_cu_seqlens_q):
            mO_cur = (
                mO[None, None, head_idx, batch_idx, split_idx]
                if const_expr(is_split_kv)
                else mO[None, None, head_idx, batch_idx]
            )
        else:
            offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
            mO_base = (
                mO[None, None, head_idx, split_idx]
                if const_expr(is_split_kv)
                else mO[None, None, head_idx]
            )
            mO_cur = cute.domain_offset((offset, 0), mO_base)
        if const_expr(tma_atom_O is not None and not self.pack_gqa and not self.check_hdim_v_oob):
            cute.arch.fence_proxy(ProxyKind.async_shared, space=SharedSpace.shared_cta)
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == (self.num_producer_warps + self.num_mma_warps - 1):
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOrO = cute.make_fragment_like(tOsO, self.dtype)
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if t0OcO[0, rest_m, 0][0] < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]:
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                        )
            else:
                pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q)

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads
        )
        smem_copy_atom_O = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(
                transpose=False,  # ROW_MAJOR
                num_matrices=4,
            ),
            self.dtype,
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
        self.store_lse(
            lse,
            mLSE,
            seqlen,
            tiled_mma,
            tidx,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
        )
        self.store_O_from_smem(
            mO,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            tma_atom_O,
            tiled_mma,
            tidx,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
        )
