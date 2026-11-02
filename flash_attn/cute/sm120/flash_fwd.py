# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.

import dataclasses
import heapq
import math
import operator
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineState,
    PipelineTmaAsync,
    pipeline_init_arrive,
    pipeline_init_wait,
)
from quack import copy_utils, layout_utils

from flash_attn.cute.sm120.paged_kv import Sm120PagedKVManager
from flash_attn.cute.sm120.policy import (
    LOW_HD_DECODE_SHAPES,
    LOW_HD_DECODE_TILE_N,
    low_hd_paged_decode_tile_m,
    visible_decode_seqlen_k,
)
from flash_attn.cute.sm120.scheduler import (
    Sm120UniformBatchScheduler,
)
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparse_utils import (
    get_curr_blocksparse_tensors,
    sparse_tensor_m_block,
)
from flash_attn.cute.block_sparsity import (
    BlockSparseTensors,
)
from flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned,
)
from flash_attn.cute.flash_fwd import (
    FlashAttentionForwardBase,
)
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.pack_gqa import (
    PackGQA,
    pack_gqa_layout,
)
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import (
    Softmax,
    apply_score_mod_inner,
)
from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)
from flash_attn.cute.utils import AuxData


class FlashAttentionForwardSm120(FlashAttentionForwardBase):
    """SM120 warp-MMA forward kernel with TMA fused into the QK warps."""

    # Paged K / V loaded by TMA, one page per KV tile (the transposed decode kernel sets it);
    # otherwise the DMA warp gathers the pages.
    paged_tma = False

    def __init__(
        self,
        *args,
        direct_uniform_batch: bool = False,
        paged_kv: bool = False,
        split_qk_n: bool = False,
        l2_group_bytes: int = 50 * 1024 * 1024,
        split_kv_blocks_per_cta: int = 0,
        is_split_kv: bool = False,
        kv_subtile_factor: int = 1,
        lpt_order: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_split_kv = is_split_kv
        self.kv_subtile_factor = kv_subtile_factor
        self.lpt_order = lpt_order
        self.direct_uniform_batch = direct_uniform_batch
        self.paged_kv = paged_kv
        self.split_qk_n = split_qk_n
        self.l2_group_bytes = l2_group_bytes
        self.split_kv_blocks_per_cta = split_kv_blocks_per_cta

    # SplitKV writes float32 partial outputs that FlashAttentionForwardCombine merges.

    def _check_type(
        self,
        mQ_type: type,
        mK_type: type,
        mV_type: type,
        mO_type: type,
        mLSE_type: type | None,
        mCuSeqlensQ_type: type | None,
        mCuSeqlensK_type: type | None,
        mSeqUsedQ_type: type | None,
        mSeqUsedK_type: type | None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(self.is_split_kv):
            # SplitKV writes float32 partial outputs; Q/K/V still fp16/bf16.
            if const_expr(not (mQ_type == mK_type == mV_type)):
                raise TypeError("Q/K/V must have the same data type")
            if const_expr(mO_type != Float32):
                raise TypeError("SplitKV partial output (mO) must be Float32")
        elif const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")
        assert mQ_type == self.dtype

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
        split_idx: Int32 = 0,
        single_split_out: Optional[tuple] = None,
    ):
        if const_expr(self.is_split_kv):
            split_args = (seqlen, tiled_mma, tidx, m_block, head_idx, batch_idx)
            if const_expr(single_split_out is not None):
                mO_final, mLSE_final, num_splits = single_split_out
                if num_splits == 1:
                    self.epilogue_split(acc_O, lse, mO_final, mLSE_final, *split_args, Int32(0))
                else:
                    self.epilogue_split(acc_O, lse, mO, mLSE, *split_args, split_idx)
            else:
                self.epilogue_split(acc_O, lse, mO, mLSE, *split_args, split_idx)
        else:
            super().epilogue(
                acc_O,
                lse,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

    @cute.jit
    def epilogue_split(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,  # (s_q, dv, h, b, num_splits) or (total_q, dv, h, num_splits)
        mLSE: Optional[cute.Tensor],  # (s_q, h, b, num_splits) or (total_q, h, num_splits)
        seqlen: SeqlenInfoQK,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)

        if const_expr(not self.pack_gqa):
            # Write LSE: only the thread that owns column 0 of each row writes.
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx, split_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                if taccOcO[0][1] == 0:
                    for r in cutlass.range(cute.size(taccOcO, mode=[0]), unroll_full=True):
                        row = taccOcO[r, 0][0]
                        if m_block * self.tile_m + row < seqlen.seqlen_q:
                            gLSE[row] = lse[r]

            # Write O partials (float32) directly from the accumulator.
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            taccOgO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gO))
            for r in cutlass.range(cute.size(acc_O_mn, mode=[0]), unroll_full=True):
                if m_block * self.tile_m + taccOcO[r, 0][0] < seqlen.seqlen_q:
                    for c in cutlass.range(cute.size(acc_O_mn, mode=[1]), unroll_full=True):
                        if const_expr(not self.check_hdim_v_oob) or taccOcO[r, c][1] < mO.shape[1]:
                            taccOgO[r, c] = mO.element_type(acc_O_mn[r, c])
        else:
            # pack_gqa: mO/mLSE arrive pre-packed by pack_gqa_layout — mode 0 is the
            # hierarchical (qhead_per_kvhead, seqlen_q) row and the head mode indexes
            # KV heads. An integer row coordinate decomposes colexicographically as
            # (row % qhead, row // qhead), which is exactly the kernel's packed row
            # (q-head fastest), so we index mode 0 with the tile row directly.
            row_limit = seqlen.seqlen_q * self.qhead_per_kvhead
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx, split_idx]
                if taccOcO[0][1] == 0:
                    for r in cutlass.range(cute.size(taccOcO, mode=[0]), unroll_full=True):
                        row = m_block * self.tile_m + taccOcO[r, 0][0]
                        if row < row_limit:
                            mLSE_cur[row] = lse[r]
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            for r in cutlass.range(cute.size(acc_O_mn, mode=[0]), unroll_full=True):
                row = m_block * self.tile_m + taccOcO[r, 0][0]
                if row < row_limit:
                    for c in cutlass.range(cute.size(acc_O_mn, mode=[1]), unroll_full=True):
                        if const_expr(not self.check_hdim_v_oob) or taccOcO[r, c][1] < mO.shape[1]:
                            mO_cur[row, taccOcO[r, c][1]] = mO.element_type(acc_O_mn[r, c])

    @staticmethod
    def _has_split_counts(params) -> bool:
        """Whether work tiles carry a per-batch split count (in the top 16 bits)."""
        # SingleTileVarlenScheduler keeps its problem shape in a VarlenDecoder.
        decoder = getattr(params, "decoder", None)
        shape = params if decoder is None else decoder
        return (
            getattr(shape, "is_split_kv", False)
            and getattr(shape, "num_splits_dynamic_ptr", None) is not None
        )

    @staticmethod
    def _work_split(split_idx, params):
        """(split, split count) of a work tile. A per-batch count rides in the top 16 bits."""
        decoder = getattr(params, "decoder", None)
        shape = params if decoder is None else decoder
        num_splits = shape.num_splits
        if const_expr(FlashAttentionForwardSm120._has_split_counts(params)):
            num_splits = split_idx >> 16
            split_idx = split_idx & 0xFFFF
        return split_idx, num_splits

    @staticmethod
    @cute.jit
    def _absolute_n_block_min_max(
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
    ):
        """KV block range of an M block before any SplitKV partitioning."""
        return dataclasses.replace(block_info, is_split_kv=False).get_n_block_min_max(
            seqlen, m_block
        )

    @cute.jit
    def _get_n_block_min_max(
        self,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32,
        num_splits: Int32,
    ):
        """Partition an M block's KV range over num_splits uniform ceil-div chunks.

        Ragged decode (split_kv_blocks_per_cta > 0) balances requests by deriving the
        split count from each request's own range, without changing the workspace bound.
        """
        n_block_min, n_block_max = self._absolute_n_block_min_max(block_info, seqlen, m_block)
        if const_expr(block_info.is_split_kv):
            if const_expr(self.split_kv_blocks_per_cta > 0):
                num_splits = cute.ceil_div(
                    cutlass.max(n_block_max - n_block_min, 1),
                    self.split_kv_blocks_per_cta,
                )
            num_n_blocks_per_split = (
                Int32(0)
                if n_block_max <= n_block_min
                else (n_block_max - n_block_min + num_splits - 1) // num_splits
            )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max

    num_dma_threads = 32
    # Relative CTA critical-path costs keyed by the complete kernel config:
    # (fixed work, work per N block, extra local-mask work). Sequence length,
    # window, packed heads, and SM count remain analytic inputs to the generic
    # LPT model below.
    _lpt_cost_by_config = {
        (256, 256, 32, 64): (213, 95, 95),
        (256, 256, 48, 64): (221, 99, 74),
        (256, 256, 64, 64): (212, 101, 67),
    }
    _lpt_tie_margin = 1
    _qualified_wave_tile_shapes = frozenset(((32, 32), (64, 64), (96, 96), (128, 128), (192, 128)))

    @staticmethod
    def _estimate_lpt_makespan(
        tile_m: int,
        tile_n: int,
        cost: tuple[int, int, int],
        *,
        seqlen_q: int,
        seqlen_k: int,
        num_sms: int,
        num_head_kv: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        window_size_left: int | None,
        window_size_right: int | None,
    ) -> int | None:
        """Estimate the one- or two-wave LPT critical path in relative units."""
        if (
            seqlen_q <= 0
            or seqlen_k <= 0
            or num_sms <= 0
            or num_head_kv <= 0
            or qhead_per_kvhead <= 0
        ):
            return None

        packed_q = seqlen_q * qhead_per_kvhead
        num_m_blocks = (packed_q + tile_m - 1) // tile_m
        num_ctas = num_m_blocks * num_head_kv
        # The exact boundary expression below is intentionally limited to two
        # physical waves. Returning None also keeps this estimator O(1).
        if num_ctas > 2 * num_sms:
            return None

        fixed_cost, n_block_cost, local_mask_cost = cost
        fixed_cost += local_mask_cost if is_local else 0
        num_k_blocks = (seqlen_k + tile_n - 1) // tile_n
        seqlen_delta = seqlen_k - seqlen_q

        def cta_cost(launch_idx: int) -> int:
            # SingleTileVarlenScheduler repeats each reversed (LPT) M block
            # across the packed KV heads before advancing to the next block.
            m_block = num_m_blocks - 1 - launch_idx // num_head_kv
            m_idx_min = m_block * tile_m // qhead_per_kvhead
            m_idx_max = ((m_block + 1) * tile_m + qhead_per_kvhead - 1) // qhead_per_kvhead

            if is_causal or (is_local and window_size_right is not None):
                n_idx_right = m_idx_max + seqlen_delta
                if not is_causal:
                    n_idx_right += window_size_right
                n_block_max = min(
                    num_k_blocks,
                    max(0, (n_idx_right + tile_n - 1) // tile_n),
                )
            else:
                n_block_max = num_k_blocks

            n_block_min = 0
            if is_local and window_size_left is not None:
                n_idx_left = m_idx_min + seqlen_delta - window_size_left
                n_block_min = max(n_idx_left // tile_n, 0)
            num_n_blocks = max(n_block_max - n_block_min, 0)
            return fixed_cost + n_block_cost * num_n_blocks

        heaviest_cta = cta_cost(0)
        if num_ctas <= num_sms:
            return heaviest_cta
        # With at most two waves, the first tail CTA is paired with the lightest
        # CTA in the first hardware wave. This captures the discrete tail that
        # an average-work occupancy model misses.
        wave_boundary = cta_cost(num_sms - 1) + cta_cost(num_sms)
        return max(heaviest_cta, wave_boundary)

    @staticmethod
    def _fits_lpt_equivalent_wave(
        tile_n: int,
        resident_ctas_per_sm: int,
        *,
        seqlen_q: int,
        seqlen_k: int,
        num_sms: int,
        num_head_kv: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        window_size_left: int | None,
        window_size_right: int | None,
    ) -> bool:
        """Return whether structural CTA work fits one LPT-equivalent wave."""
        if resident_ctas_per_sm <= 0:
            return False
        packed_q = seqlen_q * qhead_per_kvhead
        num_ctas = ((packed_q + 63) // 64) * num_head_kv
        structural_cost = (0, 1, 0)
        workload = FlashAttentionForwardSm120._estimate_lpt_makespan(
            64,
            tile_n,
            structural_cost,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_sms=num_sms * resident_ctas_per_sm,
            num_head_kv=num_head_kv,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        heaviest_cta = FlashAttentionForwardSm120._estimate_lpt_makespan(
            64,
            tile_n,
            structural_cost,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_sms=max(num_ctas, 1),
            num_head_kv=num_head_kv,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        return workload is not None and heaviest_cta is not None and workload <= heaviest_cta

    @staticmethod
    def _select_qualified_tile_n(
        head_dim: int,
        head_dim_v: int,
        *,
        seqlen_q: int,
        seqlen_k: int,
        num_sms: int,
        num_head_kv: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        window_size_left: int | None,
        window_size_right: int | None,
    ) -> int | None:
        """Tile N from the LPT wave fit and CTA residency."""
        shape = (head_dim, head_dim_v)
        if shape not in FlashAttentionForwardSm120._qualified_wave_tile_shapes:
            return None
        if (
            seqlen_q <= 0
            or seqlen_k <= 0
            or num_sms <= 0
            or num_head_kv <= 0
            or qhead_per_kvhead <= 0
        ):
            return None

        has_steady_state_k_loop = seqlen_k >= 1024
        # Small SM arrays reach several waves at shorter K, where the narrower N64 loop
        # balances better.
        has_compact_sm_array = num_sms <= 64

        def fits_lpt_wave(tile_n: int, resident_ctas_per_sm: int) -> bool:
            return FlashAttentionForwardSm120._fits_lpt_equivalent_wave(
                tile_n,
                resident_ctas_per_sm,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                num_sms=num_sms,
                num_head_kv=num_head_kv,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_local=is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )

        if is_local:
            if shape == (32, 32):
                # N128 retains two resident CTAs/SM.
                return 128 if fits_lpt_wave(128, 2) else 64
            if shape == (64, 64):
                return 128 if fits_lpt_wave(128, 1) else 64
            return 64

        if shape == (32, 32):
            if has_compact_sm_array and seqlen_k >= 1536:
                return 64
            # N128 keeps two CTAs per SM; N256's single CTA per SM loses once the grid nears a
            # full wave.
            return 128
        if shape == (64, 64):
            if not has_steady_state_k_loop:
                return 128
            return 192 if fits_lpt_wave(192, 1) else 64
        if shape == (96, 96):
            if has_compact_sm_array and has_steady_state_k_loop:
                return 64
            # Larger SM arrays: N128, fewer K-loop iterations per CTA.
            return 128
        if shape == (128, 128):
            # N64 runs the pipelined two-CTA path (see _qk_pipelined); N128 needs a DMA warp and
            # 80 KB of SMEM, one CTA per SM, and loses the second CTA's overlap.
            return 64
        # hd 192: N64, the configuration without spills.
        return 64

    @staticmethod
    def _smem_usage_in_bytes(
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
    ) -> int:
        """Return SMEM usage after padding head dimensions to kernel alignment."""
        head_dim = math.ceil(head_dim / 16) * 16
        head_dim_v = math.ceil(head_dim_v / 16) * 16
        element_size = 2
        # The O epilogue tile reuses the Q buffer.
        smem_usage_Q = tile_m * max(head_dim, head_dim_v) * element_size
        smem_usage_K = tile_n * head_dim * num_stages * element_size
        smem_usage_V = tile_n * head_dim_v * num_stages * element_size
        smem_usage = smem_usage_Q + smem_usage_V + smem_usage_K
        if (head_dim, head_dim_v, tile_m, tile_n) in (
            (256, 256, 16, 64),
            (256, 256, 16, 80),
            (256, 256, 32, 64),
            (256, 256, 48, 64),
            (256, 256, 64, 48),
            (256, 256, 64, 64),
        ):
            # Q is copied to QK registers before the mainloop, so its allocation
            # can hold both P stages and later the O epilogue tile.  Four FP32
            # rows hold the per-stage rescale values and final scale/LSE.
            smem_usage += 4 * tile_m * 4
        # Q/K/V/P/final full-empty mbarriers and conservative field alignment.
        return smem_usage + 2048

    # Schedule model of a dense grid per kernel configuration (tile_hdim, tile_m, CTAs per SM):
    # SM cycles of one 64-column K / V block on one CTA slot, and a tile's prologue / epilogue
    # in blocks. Fitted per configuration on MQA (K / V stay in L2: no K / V DRAM traffic) at
    # base clock. Cycles are SM-local, so they carry across SM12x boards; a configuration not
    # listed scales its head dim's first entry at the same per-SM rate.
    _schedule = {
        (64, 64, 2): (2501, 1.5),
        (64, 128, 2): (4586, 1.5),
        (96, 64, 2): (3474, 1.5),
        (96, 128, 2): (6601, 1.25),
        (128, 64, 2): (4238, 2.0),
        (128, 128, 1): (4099, 2.75),
        (192, 128, 1): (6253, 1.75),
        (256, 64, 1): (5767, 1.5),
    }
    # Speedup of a CTA left alone on its SM: a two-CTA tail of at most half a wave costs
    # 1 / speedup of a round where one CTA per SM pays a full one.
    _m128_lone_speedup = 1.7
    # 128-row tiles halve the K/V L2 traffic, which buys clock under the power cap (a lower
    # bound on that gain).
    _m128_min_clock_gain = 1.02
    # Predicted gains of the 128-row tile below this fraction are within the schedule model's
    # error and count as ties, which keep 64 x 64.
    _m128_tie_margin = 0.01
    # Streaming DRAM bandwidth (elementwise read + write) against transfer rate x bus width; it
    # varies by board, and a middle value keeps the model's errors on either side small.
    _dram_efficiency = 0.87

    @staticmethod
    def _schedule_params(head_dim: int, tile_m: int, ctas_per_sm: int) -> tuple[float, float]:
        """(cycles per K / V block per CTA slot, tile overhead in blocks) of a configuration."""
        C = FlashAttentionForwardSm120
        tile_hdim = next(d for d in (64, 96, 128, 192, 256) if d >= head_dim)
        params = C._schedule.get((tile_hdim, tile_m, ctas_per_sm))
        if params is not None:
            return params
        (_, ref_m, ref_ctas), (cycles, overhead) = next(
            item for item in C._schedule.items() if item[0][0] == tile_hdim
        )
        return cycles * tile_m / ref_m * ctas_per_sm / ref_ctas, overhead

    @staticmethod
    def _tile_blocks(
        seqlen_q: int,
        seqlen_k: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        window: tuple[int | None, int | None] | None,
        tile_m: int,
    ) -> list[int]:
        """K / V blocks of each M block of a head. window is the local (left, right) window."""
        rows = seqlen_q * qhead_per_kvhead
        n_blocks = (seqlen_k + 63) // 64
        blocks = []
        for m in range((rows + tile_m - 1) // tile_m):
            lo, hi = 0, n_blocks
            if is_causal or window is not None:
                q_first = m * tile_m // qhead_per_kvhead + seqlen_k - seqlen_q
                q_last = -(-(m + 1) * tile_m // qhead_per_kvhead) + seqlen_k - seqlen_q
                right = 0 if is_causal else window[1]
                if right is not None:
                    hi = min(n_blocks, max(0, -(-(q_last + right) // 64)))
                if window is not None and window[0] is not None:
                    lo = max(0, (q_first - window[0]) // 64)
            blocks.append(max(hi - lo, 0))
        return blocks

    @staticmethod
    def _makespan(
        num_sms: int,
        num_heads: int,
        head_bytes: int,
        l2_group_bytes: int,
        costs: list[float],
        ctas_per_sm: int,
        lpt: bool,
    ) -> float:
        """Finish time of a dense grid of tiles costing costs (per M block of a head),
        dispatched as the hardware does: list scheduling in grid order (each head's M blocks
        lightest first), or with lpt heaviest first within the L2 head groups of
        SingleTileLPTScheduler."""
        C = FlashAttentionForwardSm120
        if lpt:
            group = C._lpt_head_group(head_bytes, l2_group_bytes)
            order = []
            for g in range(0, num_heads, group):
                for cost in reversed(costs):
                    order.extend([cost] * min(group, num_heads - g))
        else:
            order = costs * num_heads
        slots = [(0.0, k, sm) for k in range(ctas_per_sm) for sm in range(num_sms)]
        if len(order) > 64 * len(slots):
            # Past 64 rounds the tail is at most a round in 64: compare the work per slot.
            return sum(order) / len(slots)
        for cost in order:
            t, k, sm = heapq.heappop(slots)
            heapq.heappush(slots, (t + cost, k, sm))
        if ctas_per_sm == 1:
            return max(t for t, _, _ in slots)
        ends = [[] for _ in range(num_sms)]
        for t, _, sm in slots:
            ends[sm].append(t)
        return max(min(e) + (max(e) - min(e)) / C._m128_lone_speedup for e in ends)

    @staticmethod
    def _lpt_head_group(head_bytes: int, l2_group_bytes: int) -> int:
        """Heads per L2 group of SingleTileLPTScheduler: a power of two."""
        if head_bytes > l2_group_bytes:
            return 1
        return 1 << (l2_group_bytes // head_bytes).bit_length() - 1

    @staticmethod
    def _order_time(
        num_sms: int,
        head_dim: int,
        num_heads: int,
        heads_per_kv: int,
        seqlen_q: int,
        seqlen_k: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        window: tuple[int | None, int | None] | None,
        l2_group_bytes: int,
        tile_m: int,
        ctas_per_sm: int,
        rates: tuple[float, float] | None,
        lpt: bool,
        clock_gain: float = 1.0,
    ) -> float:
        """Modelled time (in blocks when rates is None) of a dense causal / local grid in grid
        or LPT order. num_heads counts scheduled heads, heads_per_kv of them reading one K / V
        head; rates is (SM clock in Hz, DRAM bytes / s), and clock_gain speeds up the compute
        only.

        Both orders read the same compulsory DRAM bytes but not at the same time.
        Grid order runs each head's M blocks lightest first, so K / V streams in about as fast
        as the grid computes it. LPT starts each L2 head group with its heaviest tiles, and a
        causal heaviest tile reads all of its head's K / V; the kernel loads K / V only a block
        ahead, so every wave of P = num_sms * ctas_per_sm new heads needs its K / V, plus one
        Q tile per slot, within one heaviest-tile time, else the wave stalls for the rest.
        The stalls of successive waves add."""
        C = FlashAttentionForwardSm120
        cycles, overhead = C._schedule_params(head_dim, tile_m, ctas_per_sm)
        blocks = C._tile_blocks(seqlen_q, seqlen_k, qhead_per_kvhead, is_causal, window, tile_m)
        head_bytes = seqlen_k * 2 * head_dim * 2
        block_time = (1.0 if rates is None else cycles / rates[0]) / clock_gain
        costs = [(b + overhead) * block_time for b in blocks]
        time = C._makespan(num_sms, num_heads, head_bytes, l2_group_bytes, costs, ctas_per_sm, lpt)
        if rates is None:
            return time
        bandwidth = rates[1]
        q_bytes = tile_m * head_dim * 2
        if lpt and is_causal:
            slots = num_sms * ctas_per_sm
            heaviest = max(costs)

            def stall(heads: int) -> float:
                kv = -(-heads // heads_per_kv) * head_bytes
                return max(0.0, (kv + slots * q_bytes) / bandwidth - heaviest)

            def group_stall(heads: int) -> float:
                return heads // slots * stall(slots) + (
                    stall(heads % slots) if heads % slots else 0.0
                )

            group = C._lpt_head_group(head_bytes, l2_group_bytes)
            time += num_heads // group * group_stall(group)
            if num_heads % group:
                time += group_stall(num_heads % group)
        total = num_heads * len(blocks) * 2 * q_bytes + -(-num_heads // heads_per_kv) * head_bytes
        return max(time, total / bandwidth)

    @staticmethod
    def _m128_no_slower(
        num_sms: int,
        head_dim: int,
        num_heads: int,
        heads_per_kv: int,
        seqlen_q: int,
        seqlen_k: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        window: tuple[int | None, int | None] | None,
        l2_group_bytes: int,
        rates: tuple[float, float] | None,
    ) -> bool:
        """Whether the 128-row tile finishes before 64 x 64, each in its better launch order
        (see use_lpt_order). window is the local window, None for causal and full attention."""
        C = FlashAttentionForwardSm120

        def best(tile_m: int, ctas_per_sm: int, clock_gain: float) -> float:
            def time(lpt: bool) -> float:
                return C._order_time(
                    num_sms,
                    head_dim,
                    num_heads,
                    heads_per_kv,
                    seqlen_q,
                    seqlen_k,
                    qhead_per_kvhead,
                    is_causal,
                    window,
                    l2_group_bytes,
                    tile_m,
                    ctas_per_sm,
                    rates,
                    lpt,
                    clock_gain,
                )

            return min(time(False), time(True)) if is_causal or window is not None else time(False)

        # Ties (both at the DRAM bound, where the clock gain buys nothing, or within the model's
        # error) keep 64 x 64.
        return best(128, 1 if head_dim == 128 else 2, C._m128_min_clock_gain) < best(64, 2, 1.0) * (
            1 - C._m128_tie_margin
        )

    @staticmethod
    @lru_cache(maxsize=4096)
    def use_lpt_order(
        num_sms: int,
        head_dim: int,
        tile_m: int,
        ctas_per_sm: int,
        num_heads: int,
        heads_per_kv: int,
        seqlen_q: int,
        seqlen_k: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        window_size_left: int | None,
        window_size_right: int | None,
        l2_group_bytes: int,
        rates: tuple[float, float] | None = None,
    ) -> bool:
        """Whether a dense causal / local grid launches heaviest M blocks first: whichever order
        _order_time finishes first."""
        C = FlashAttentionForwardSm120
        window = None if is_causal else (window_size_left, window_size_right)
        args = (
            num_sms,
            head_dim,
            num_heads,
            heads_per_kv,
            seqlen_q,
            seqlen_k,
            qhead_per_kvhead,
            is_causal,
            window,
            l2_group_bytes,
            tile_m,
            ctas_per_sm,
            rates,
        )
        return C._order_time(*args, True) < C._order_time(*args, False)

    @staticmethod
    @lru_cache(maxsize=4096)
    def get_fwd_tile_size(
        head_dim: int,
        head_dim_v: int,
        total_q_rows: int | None = None,
        num_sms: int | None = None,
        num_batch: int | None = None,
        seqlen_q: int | None = None,
        seqlen_k: int | None = None,
        num_head_kv: int | None = None,
        qhead_per_kvhead: int | None = None,
        is_causal: bool = False,
        is_local: bool = False,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        pack_gqa: bool = False,
        paged_kv: bool = False,
        l2_group_bytes: int = 50 * 1024 * 1024,
        schedule_rates: tuple[float, float] | None = None,
    ) -> tuple[int, int]:
        """Select an SM120 tile that fits the architecture's 99 KB SMEM. schedule_rates is
        the device's (SM clock in Hz, DRAM bytes / s) for the schedule model."""
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        shape = (head_dim, head_dim_v)
        qualified_shape = shape in FlashAttentionForwardSm120._qualified_wave_tile_shapes
        has_compact_q_groups = pack_gqa or (paged_kv and qhead_per_kvhead == 1)
        is_short_compact_q = (
            has_compact_q_groups
            and seqlen_q is not None
            and qhead_per_kvhead is not None
            and seqlen_q * qhead_per_kvhead <= 16
        )
        visible_seqlen_k = (
            None
            if seqlen_k is None
            else visible_decode_seqlen_k(
                seqlen_k,
                is_local=is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )
        )
        low_hd_m16_total_mblocks = None
        if (
            num_batch is not None
            and num_head_kv is not None
            and seqlen_q is not None
            and qhead_per_kvhead is not None
        ):
            packed_q_rows = seqlen_q * qhead_per_kvhead
            low_hd_m16_total_mblocks = num_batch * num_head_kv * ((packed_q_rows + 16 - 1) // 16)
        low_hd_tile_m = low_hd_paged_decode_tile_m(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            paged_kv=paged_kv,
            seqlen_q=seqlen_q,
            visible_seqlen_k=visible_seqlen_k,
            qhead_per_kvhead=qhead_per_kvhead,
            num_sms=num_sms,
            total_mblocks=low_hd_m16_total_mblocks,
        )
        if low_hd_tile_m is not None:
            fallback_candidates = ((low_hd_tile_m, LOW_HD_DECODE_TILE_N, 1),)
        elif (head_dim, head_dim_v) == (256, 256):
            if is_short_compact_q:
                fallback_candidates = (
                    (16, 64, 1),
                    (16, 80, 1),
                    (32, 64, 1),
                    (64, 64, 1),
                )
            elif (
                total_q_rows is not None
                and num_sms is not None
                and total_q_rows > 76 * num_sms
                and total_q_rows <= 92 * num_sms
            ):
                # Multi-batch / unpacked grids: no per-sequence LPT costs on the host, so a
                # fixed preference order.
                fallback_candidates = (
                    (48, 64, 1),
                    (64, 64, 1),
                    (64, 48, 1),
                    (32, 64, 1),
                )
            else:
                fallback_candidates = (
                    (64, 64, 1),
                    (64, 48, 1),
                    (48, 64, 1),
                    (32, 64, 1),
                )
        elif qualified_shape:
            # M64 is the spill-free fallback for these shapes. hd 32 without a local mask
            # takes N128, which keeps two CTAs per SM; local masks keep the more parallel N64.
            safe_tile_n = 128 if shape == (32, 32) and not is_local else 64
            fallback_candidates = ((64, safe_tile_n, 1),) + (
                ((64, 64, 1),) if safe_tile_n != 64 else ()
            )
        else:
            fallback_candidates = (((128, 128, 1),) if max(head_dim, head_dim_v) <= 64 else ()) + (
                (128, 64, 1),
                (64, 64, 1),
            )

        can_select_qualified_tile = (
            qualified_shape
            and num_batch == 1
            and pack_gqa
            and total_q_rows is not None
            and num_sms is not None
            and seqlen_q is not None
            and seqlen_k is not None
            and num_head_kv is not None
            and qhead_per_kvhead is not None
            and total_q_rows == seqlen_q * num_head_kv * qhead_per_kvhead
            and (is_causal or is_local)
            and not is_short_compact_q
        )
        candidates = fallback_candidates
        if can_select_qualified_tile:
            preferred_tile_n = FlashAttentionForwardSm120._select_qualified_tile_n(
                head_dim,
                head_dim_v,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                num_sms=num_sms,
                num_head_kv=num_head_kv,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_local=is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )
            if preferred_tile_n is not None:
                preferred = (64, preferred_tile_n, 1)
                candidates = (preferred,) + tuple(
                    candidate for candidate in fallback_candidates if candidate != preferred
                )

        # hd 256: pick among the modelled tiles (_lpt_cost_by_config) by the LPT makespan.
        model_candidates = tuple(
            (config[2], config[3], 1)
            for config in FlashAttentionForwardSm120._lpt_cost_by_config
            if config[:2] == shape
            and FlashAttentionForwardSm120._smem_usage_in_bytes(*config, 1) <= smem_capacity
        )
        can_model_lpt = (
            len(model_candidates) >= 2
            and num_batch == 1
            and pack_gqa
            and total_q_rows is not None
            and num_sms is not None
            and seqlen_q is not None
            and seqlen_k is not None
            and num_head_kv is not None
            and qhead_per_kvhead is not None
            and total_q_rows == seqlen_q * num_head_kv * qhead_per_kvhead
            and (is_causal or is_local)
            and not is_short_compact_q
        )
        if can_model_lpt:
            scores = {
                candidate: FlashAttentionForwardSm120._estimate_lpt_makespan(
                    candidate[0],
                    candidate[1],
                    FlashAttentionForwardSm120._lpt_cost_by_config[
                        (head_dim, head_dim_v, candidate[0], candidate[1])
                    ],
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    num_sms=num_sms,
                    num_head_kv=num_head_kv,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=is_causal,
                    is_local=is_local,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                )
                for candidate in model_candidates
            }
            valid_scores = {
                candidate: score for candidate, score in scores.items() if score is not None
            }
            if valid_scores:
                exact_best = min(valid_scores, key=valid_scores.get)
                packed_q = seqlen_q * qhead_per_kvhead

                def num_ctas(candidate: tuple[int, int, int]) -> int:
                    return ((packed_q + candidate[0] - 1) // candidate[0]) * num_head_kv

                max_model_ctas = max(num_ctas(candidate) for candidate in model_candidates)
                if num_ctas(exact_best) == max_model_ctas:
                    # Keep a genuinely faster high-parallelism tile. Once that
                    # tile loses outright, a near tie favors fewer CTAs.
                    preferred = exact_best
                else:
                    cutoff = valid_scores[exact_best] + FlashAttentionForwardSm120._lpt_tie_margin
                    near = tuple(
                        candidate for candidate, score in valid_scores.items() if score <= cutoff
                    )
                    preferred = min(
                        near,
                        key=lambda candidate: (
                            num_ctas(candidate),
                            -(candidate[0] * candidate[1]),
                        ),
                    )
                candidates = (preferred,) + tuple(
                    candidate for candidate in fallback_candidates if candidate != preferred
                )
        compact_mha_local_candidate = None
        if (
            shape == (256, 256)
            and num_sms is not None
            and num_sms <= 64
            and is_local
            and not paged_kv
            and num_batch == 1
            and pack_gqa
            and qhead_per_kvhead == 1
            and seqlen_q is not None
            and seqlen_q >= 2048
            and window_size_left is not None
            and window_size_left > 0
            and window_size_right == 0
        ):
            # On the compact SM array, long-query MHA with a narrow left
            # window benefits from smaller M tiles: they reduce masked local
            # work while the long Q dimension amortizes the extra CTAs.  Use
            # Q/window ratios so the crossover scales with both dimensions.
            if window_size_left <= 128 and seqlen_q >= 80 * window_size_left:
                compact_mha_local_candidate = (32, 64, 1)
            elif seqlen_q >= 40 * window_size_left:
                compact_mha_local_candidate = (48, 64, 1)
        if compact_mha_local_candidate is not None:
            candidates = (compact_mha_local_candidate,) + tuple(
                candidate for candidate in candidates if candidate != compact_mha_local_candidate
            )
        if (
            shape == (256, 256)
            and num_sms is not None
            and num_sms <= 64
            and is_local
            and seqlen_k is not None
            and seqlen_k >= 512
            and qhead_per_kvhead is not None
            and qhead_per_kvhead > 1
        ):
            # Small SM arrays: N48 cuts the masked KV work of packed-GQA local attention; MHA
            # and larger arrays keep the LPT model's choice.
            preferred = (64, 48, 1)
            candidates = (preferred,) + tuple(
                candidate for candidate in candidates if candidate != preferred
            )
        # The schedule model takes a dense grid. A varlen grid (fewer Q rows than batch x max
        # length) is modelled as the same rows in the fewest max-length sequences: the least
        # parallel grid those rows can form, the one least favourable to the 128-row tile.
        model_batch = None
        if (
            total_q_rows is not None
            and num_batch is not None
            and seqlen_q
            and num_head_kv is not None
            and qhead_per_kvhead is not None
        ):
            model_batch = min(
                num_batch, max(1, total_q_rows // (seqlen_q * num_head_kv * qhead_per_kvhead))
            )
        if (
            shape in ((64, 64), (96, 96), (128, 128))
            and candidates[0] == (64, 64, 1)
            and not paged_kv
            and num_sms is not None
            and model_batch is not None
            and seqlen_k is not None
            and FlashAttentionForwardSm120._m128_no_slower(
                num_sms,
                head_dim,
                model_batch * (num_head_kv if pack_gqa else num_head_kv * qhead_per_kvhead),
                1 if pack_gqa else qhead_per_kvhead,
                seqlen_q,
                seqlen_k,
                qhead_per_kvhead if pack_gqa else 1,
                is_causal,
                (window_size_left, window_size_right) if is_local else None,
                l2_group_bytes,
                schedule_rates,
            )
        ):
            candidates = ((128, 64, 1),) + candidates
        for tile_m, tile_n, num_stages in candidates:
            if (
                FlashAttentionForwardSm120._smem_usage_in_bytes(
                    head_dim,
                    head_dim_v,
                    tile_m,
                    tile_n,
                    num_stages,
                )
                <= smem_capacity
            ):
                return tile_m, tile_n
        raise ValueError(
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) exceeds "
            f"SM120 shared-memory capacity ({smem_capacity} bytes)"
        )

    @staticmethod
    def get_fwd_num_stages(head_dim: int, head_dim_v: int, tile_m: int, tile_n: int) -> int:
        """Return the public pipeline specialization depth."""
        return 1

    @staticmethod
    def get_fwd_num_threads(
        head_dim: int,
        head_dim_v: int,
        tile_m: int,
        tile_n: int,
        paged_kv: bool = False,
    ) -> int:
        """Return the number of warp-MMA consumer threads for an SM120 tile."""
        if (
            paged_kv
            and (head_dim, head_dim_v) in LOW_HD_DECODE_SHAPES
            and tile_m in (16, 32)
            and tile_n == LOW_HD_DECODE_TILE_N
        ):
            return tile_m * 2
        if paged_kv and head_dim == 256 and head_dim_v == 256:
            if (tile_m, tile_n) == (16, 64):
                # Decode is SMEM-limited to one CTA/SM. Four consumer warps
                # expose QK/PV parallelism without reducing CTA residency.
                return 128
            # The contiguous HD256 path assigns distinct QK and PV warp sets.
            # Paged KV instead reserves one DMA warp for gather and has each
            # consumer warp own the same 16 rows through both MMA phases.
            return tile_m * 2
        config = (head_dim, head_dim_v, tile_m, tile_n)
        if config in ((256, 256, 16, 64), (256, 256, 16, 80)):
            return 64
        if config == (256, 256, 32, 64):
            return 128
        if config == (256, 256, 48, 64):
            return 192
        if config in ((256, 256, 64, 48), (256, 256, 64, 64)):
            return 256
        if config == (256, 256, 96, 48):
            return 192
        if config in ((128, 128, 128, 64), (192, 192, 128, 64)) and not paged_kv:
            # 16 rows per warp, as for 64 x 64: same registers, one CTA per SM.
            return 256
        return 128

    def _uses_split_pv_warps(self) -> bool:
        """Whether dedicated QK/PV warp sets exchange P through SMEM."""
        config = (
            self.tile_hdim,
            self.tile_hdimv,
            self.tile_m,
            self.tile_n,
            self.num_threads,
        )
        if self.paged_kv:
            return config == (256, 256, 16, 64, 128)
        return config in (
            (256, 256, 16, 64, 64),
            (256, 256, 16, 80, 64),
            (256, 256, 32, 64, 128),
            (256, 256, 48, 64, 192),
            (256, 256, 64, 48, 256),
            (256, 256, 64, 64, 256),
        )

    def _q_in_regs_pipeline(self) -> bool:
        """Whether Q remains resident while its SMEM allocation holds P."""
        if self.paged_kv:
            return False
        config = (
            self.tile_hdim,
            self.tile_hdimv,
            self.tile_m,
            self.tile_n,
            self.num_threads,
        )
        return config in (
            (256, 256, 16, 64, 64),
            (256, 256, 16, 80, 64),
            (256, 256, 32, 64, 128),
            (256, 256, 48, 64, 192),
            (256, 256, 64, 48, 256),
            (256, 256, 64, 64, 256),
        )

    def _num_k_stages(self) -> int:
        return self.num_stages

    def _num_v_stages(self) -> int:
        return 1 if self._uses_split_pv_warps() else self.num_stages

    def _num_p_stages(self) -> int:
        return 2

    def _num_softmax_stat_rows(self) -> int:
        if self._uses_n_distributed_qk():
            num_qk_warps = self.num_qk_threads // cute.arch.WARP_SIZE
            return 3 * num_qk_warps + 4
        return 4

    def _uses_n_distributed_qk(self) -> bool:
        return self.split_qk_n

    def _qk_pipelined(self) -> bool:
        """hd 128 64 x 64 (4 warps) and 128 x 64 (8 warps) tiles issue QK(j + 1) before the
        online softmax and PV of block j (two S accumulators) and fold the DMA warp (see
        _fold_dma): the register file's four partitions take a CTA's warps in turn, so a fifth
        warp would put three warps of two CTAs on one partition and leave one CTA per SM. The
        128 x 64 tile shares each K / V block across twice the rows."""
        return (
            (self.tile_hdim, self.tile_hdimv, self.tile_m, self.tile_n, self.num_threads)
            in ((128, 128, 64, 64, 128), (128, 128, 128, 64, 256))
            and not self._uses_split_pv_warps()
            and not self._uses_n_distributed_qk()
            and not self.paged_kv
        )

    def _fold_dma(self) -> bool:
        """No DMA warp: warp 0 issues each next K / V load right after the consumers release
        the stage (the K load of block j - 1 after QK(j), the V load after PV(j)), so a CTA is
        just its consumer warps."""
        return self._qk_pipelined() or self._fold_generic()

    def _fold_generic(self) -> bool:
        """fold_dma on the unpipelined loop: hd 64 / 96 128 x 64 tiles (4 warps of 32 rows,
        two CTAs per SM at the full register budget) and hd 192 128 x 64 tiles (8 warps of 16
        rows). A DMA warp would put three warps on one register-file partition, where the O
        accumulator spills; a second S accumulator for QK(j + 1) would spill too."""
        return (
            (self.tile_hdim, self.tile_hdimv, self.tile_m, self.tile_n, self.num_threads)
            in ((64, 64, 128, 64, 128), (96, 96, 128, 64, 128), (192, 192, 128, 64, 256))
            and not self._uses_split_pv_warps()
            and not self._uses_n_distributed_qk()
            and not self.paged_kv
        )

    def _num_dma_threads(self) -> int:
        if self._fold_dma():
            return 0
        return self.num_dma_threads if self.paged_kv or not self._uses_split_pv_warps() else 0

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
        paged_kv=False,
    ) -> bool:
        """Check the constraints of the dedicated SM120 TMA kernel."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0 or head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_stages != FlashAttentionForwardSm120.get_fwd_num_stages(
            head_dim, head_dim_v, tile_m, tile_n
        ):
            return False
        if num_threads != FlashAttentionForwardSm120.get_fwd_num_threads(
            head_dim, head_dim_v, tile_m, tile_n, paged_kv=paged_kv
        ):
            return False
        smem_usage = FlashAttentionForwardSm120._smem_usage_in_bytes(
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            num_stages,
        )
        if smem_usage > utils_basic.get_smem_capacity_in_bytes("sm_120"):
            return False
        if paged_kv and (head_dim, head_dim_v, tile_m, tile_n, num_threads) == (
            256,
            256,
            16,
            64,
            128,
        ):
            return True
        if (head_dim, head_dim_v, tile_m, tile_n, num_threads) in (
            (256, 256, 16, 64, 64),
            (256, 256, 16, 80, 64),
            (256, 256, 32, 64, 128),
            (256, 256, 48, 64, 192),
            (256, 256, 64, 48, 256),
            (256, 256, 64, 64, 256),
        ):
            return True
        return (tile_m * 2) % num_threads == 0

    def _get_smem_layout_atom(self):
        sQ_layout_atom = self._make_smem_layout_atom(self.dtype, self.tile_hdim, is_k_major=True)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = self._make_smem_layout_atom(self.dtype, self.tile_hdimv, is_k_major=True)
        sO_layout_atom = sV_layout_atom
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, None

    def _setup_attributes(self):
        super()._setup_attributes()
        if const_expr(self._uses_split_pv_warps()):
            sK_layout_atom = self._make_smem_layout_atom(
                self.dtype, self.tile_hdim, is_k_major=True
            )
            self.sK_layout = cute.tile_to_shape(
                sK_layout_atom,
                (self.tile_n, self.tile_hdim, self._num_k_stages()),
                (0, 1, 2),
            )
        sV_layout_atom = self._make_smem_layout_atom(self.dtype, self.tile_hdimv, is_k_major=False)
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.tile_hdimv, self.tile_n, self._num_v_stages()),
            (1, 0, 2),
        )
        self.sP_layout = None
        if const_expr(self._uses_split_pv_warps()):
            sP_layout_atom = self._make_smem_layout_atom(self.dtype, self.tile_n, is_k_major=True)
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom,
                (self.tile_m, self.tile_n, self._num_p_stages()),
                (0, 1, 2),
            )

    @staticmethod
    def _make_smem_layout_atom(
        dtype: type[cutlass.Numeric],
        major_dim: int,
        *,
        is_k_major: bool,
    ) -> cute.ComposedLayout:
        """Build a TMA-compatible SMEM layout for SM120 warp MMA."""
        major_mode_bits = const_expr(major_dim * dtype.width)
        if const_expr(major_mode_bits % 1024 == 0):
            contiguous_bits, swizzle_bits = 1024, 3
        elif const_expr(major_mode_bits % 512 == 0):
            contiguous_bits, swizzle_bits = 512, 2
        elif const_expr(major_mode_bits % 256 == 0):
            contiguous_bits, swizzle_bits = 256, 1
        else:
            contiguous_bits, swizzle_bits = 128, 0
        contiguous_elems = const_expr(contiguous_bits // dtype.width)
        layout = (
            cute.make_layout(
                (8, contiguous_elems),
                stride=(contiguous_elems, 1),
            )
            if const_expr(is_k_major)
            else cute.make_layout(
                (contiguous_elems, 8),
                stride=(1, contiguous_elems),
            )
        )
        return cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 4, 3),
            0,
            layout,
        )

    def _get_tiled_mma(self):
        split_pv_warps = self._uses_split_pv_warps()
        num_qk_warps = (
            self.num_threads // cute.arch.WARP_SIZE
            if split_pv_warps and self._uses_n_distributed_qk()
            else (self.tile_m // 16 if split_pv_warps else self.num_threads // cute.arch.WARP_SIZE)
        )
        num_pv_warps_m = self.tile_m // 16 if split_pv_warps else self.num_threads // 32
        num_pv_warps_n = (
            self.num_threads // cute.arch.WARP_SIZE if split_pv_warps and self.paged_kv else 1
        )
        tiled_mma_qk = (
            cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                (1, num_qk_warps, 1),
                permutation_mnk=(self.tile_m, self.tile_n, 16),
            )
            if self.split_qk_n
            else cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                (num_qk_warps, 1, 1),
                permutation_mnk=(num_qk_warps * 16, 16, 16),
            )
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (num_pv_warps_m, num_pv_warps_n, 1),
            permutation_mnk=(
                num_pv_warps_m * 16,
                num_pv_warps_n * 16,
                16,
            ),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        # The O epilogue tile reuses the Q buffer, which is wider when head_dim_v > head_dim.
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cosize], 1024]
            for cosize in (
                max(cute.cosize(self.sQ_layout), cute.cosize(self.sO_layout)),
                cute.cosize(self.sK_layout),
                cute.cosize(self.sV_layout),
            )
        ]
        mbar_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_K_struct = cute.struct.MemRange[cutlass.Int64, self._num_k_stages() * 2]
        mbar_V_struct = cute.struct.MemRange[cutlass.Int64, self._num_v_stages() * 2]
        mbar_P_struct = cute.struct.MemRange[
            cutlass.Int64,
            2 * self._num_p_stages() if self._uses_split_pv_warps() else 0,
        ]
        mbar_final_struct = cute.struct.MemRange[
            cutlass.Int64, 2 if self._uses_split_pv_warps() else 0
        ]
        num_stats = (
            self._num_softmax_stat_rows() * self.tile_m if self._uses_split_pv_warps() else 0
        )
        softmax_stats_struct = cute.struct.MemRange[Float32, num_stats]
        num_p_elements = (
            0
            if self._q_in_regs_pipeline() or not self._uses_split_pv_warps()
            else cute.cosize(self.sP_layout)
        )
        sP_struct = cute.struct.Align[
            cute.struct.MemRange[
                self.dtype,
                num_p_elements,
            ],
            1024,
        ]

        @cute.struct
        class SharedStorage:
            mbar_Q: mbar_Q_struct
            mbar_K: mbar_K_struct
            mbar_V: mbar_V_struct
            mbar_P: mbar_P_struct
            mbar_final: mbar_final_struct
            softmax_stats: softmax_stats_struct
            sP: sP_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

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
        aux_data: AuxData = AuxData(),
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mBlocksToBatch: Optional[cute.Tensor] = None,
        mNumSplitsDynamic: Optional[cute.Tensor] = None,
        mOFinal: Optional[cute.Tensor] = None,  # mO / mLSE shapes with one split
        mLSEFinal: Optional[cute.Tensor] = None,
        mSplitSlotBatch: Optional[cute.Tensor] = None,  # compact split layout
        stream: cuda.CUstream = None,
    ):
        if const_expr(blocksparse_tensors is not None):
            assert not self._uses_n_distributed_qk(), (
                "SM120 block sparsity does not run the N-distributed paged decode"
            )
        assert mPageTable is None or self.paged_kv, (
            "SM120 paged KV requires the dedicated DMA-warp specialization"
        )
        self._check_type(
            *(
                t.element_type if t is not None else None
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
                )
            )
        )
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_qk_threads = tiled_mma_qk.size
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = (
            self.num_qk_threads if self._uses_split_pv_warps() else self.num_threads
        )
        self.num_epilogue_threads = (
            self.num_mma_threads if self._uses_split_pv_warps() else self.num_threads
        )
        self.use_tma_O = False
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(mPageTable is None):
            V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
            mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))
        if const_expr(self.is_split_kv):
            num_splits = mO.shape[0]
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
        else:
            num_splits = Int32(1)
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mO, mOFinal = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=O_layout_transpose))
            if t is not None
            else None
            for t in (mO, mOFinal)
        ]
        mLSE, mLSEFinal = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=LSE_layout_transpose))
            if t is not None
            else None
            for t in (mLSE, mLSEFinal)
        ]
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO, mOFinal = [
                pack_gqa_layout(t, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                if t is not None
                else None
                for t in (mO, mOFinal)
            ]
            mLSE, mLSEFinal = [
                pack_gqa_layout(t, self.qhead_per_kvhead, nheads_kv, head_idx=1)
                if t is not None
                else None
                for t in (mLSE, mLSEFinal)
            ]

        if const_expr(mPageTable is None or self.paged_tma):
            tma_copy_op = cpasync.CopyBulkTensorTileG2SOp()
            mV_tma = (
                mV
                if const_expr(mPageTable is None)
                else cute.make_tensor(
                    mV.iterator,
                    cute.select(mV.layout, mode=[1, 0, 2, 3]),
                )
            )
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                tma_copy_op,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                tma_copy_op,
                mV_tma,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_hdimv, self.tile_n),
                1,
            )
            self.tma_copy_bytes_K = cute.size_in_bytes(
                mK.element_type, cute.select(self.sK_layout, mode=[0, 1])
            )
            self.tma_copy_bytes_V = cute.size_in_bytes(
                mV_tma.element_type,
                cute.select(self.sV_layout, mode=[0, 1]),
            )
        else:
            tma_atom_K = None
            tma_atom_V = None
            tma_tensor_K = mK
            tma_tensor_V = mV
        is_varlen = const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None)
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[3])
        )
        # Dense causal / local grids the host picks (use_lpt_order) dispatch heaviest M blocks
        # first within L2-sized head groups; in grid order each head's heaviest blocks land last.
        use_lpt = const_expr(
            self.lpt_order
            and not is_varlen
            and (self.is_causal or self.is_local)
            and not self.paged_kv
            and not self.is_split_kv
        )
        TileScheduler = (
            Sm120UniformBatchScheduler
            if is_varlen and self.direct_uniform_batch
            else SingleTileVarlenScheduler
            if is_varlen
            else SingleTileLPTScheduler
            if use_lpt
            else SingleTileScheduler
        )
        assert mSplitSlotBatch is None or TileScheduler is Sm120UniformBatchScheduler, (
            "the compact split layout needs the uniform-batch scheduler"
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            num_splits=num_splits,
            seqlen_k=cute.size(mK.shape[0]) if const_expr(use_lpt) else 0,
            headdim=mQ.shape[1],
            headdim_v=mO.shape[1],
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=(self.tile_m, self.tile_n),
            lpt=(self.is_causal or self.is_local) and not self.direct_uniform_batch,
            qhead_per_kvhead_packgqa=(self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            blocks_to_batch_idx_ptr=mBlocksToBatch,
            num_splits_dynamic_ptr=mNumSplitsDynamic,
            split_slot_batch_ptr=mSplitSlotBatch,
            element_size=self.dtype.width // 8,
            l2_group_bytes=self.l2_group_bytes,
            is_persistent=False,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args,
            scheduling_mode=SchedulingMode.STATIC,
        )
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ,
            mK,
            self.qhead_per_kvhead,
            self.pack_gqa,
            aux_data.tensors,
            mPageTable,
        )

        self.kernel(
            mQ,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_K,
            tma_atom_V,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
            mOFinal,
            mLSEFinal,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads + self._num_dma_threads(), 1, 1],
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
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mOFinal: Optional[cute.Tensor] = None,
        mLSEFinal: Optional[cute.Tensor] = None,
    ):
        # Splits past their batch's own count (per-batch split counts) have no work: they
        # leave before any setup, so a grid sized for the longest request costs little.
        if const_expr(self._has_split_counts(tile_sched_params)):
            split_idx = TileScheduler.create(tile_sched_params).initial_work_tile_info().tile_idx[3]
            split_idx, num_splits = self._work_split(split_idx, tile_sched_params)
            if split_idx < num_splits:
                self._kernel_main(
                    mQ=mQ,
                    mK=mK,
                    mV=mV,
                    mO=mO,
                    mLSE=mLSE,
                    mCuSeqlensQ=mCuSeqlensQ,
                    mCuSeqlensK=mCuSeqlensK,
                    mSeqUsedQ=mSeqUsedQ,
                    mSeqUsedK=mSeqUsedK,
                    mPageTable=mPageTable,
                    tma_atom_K=tma_atom_K,
                    tma_atom_V=tma_atom_V,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_scale=softmax_scale,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    learnable_sink=learnable_sink,
                    sQ_layout=sQ_layout,
                    sK_layout=sK_layout,
                    sV_layout=sV_layout,
                    sO_layout=sO_layout,
                    sP_layout=sP_layout,
                    gmem_tiled_copy_Q=gmem_tiled_copy_Q,
                    gmem_tiled_copy_O=gmem_tiled_copy_O,
                    tiled_mma_qk=tiled_mma_qk,
                    tiled_mma_pv=tiled_mma_pv,
                    SharedStorage=SharedStorage,
                    tile_sched_params=tile_sched_params,
                    TileScheduler=TileScheduler,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    blocksparse_tensors=blocksparse_tensors,
                    mOFinal=mOFinal,
                    mLSEFinal=mLSEFinal,
                )
        else:
            self._kernel_main(
                mQ=mQ,
                mK=mK,
                mV=mV,
                mO=mO,
                mLSE=mLSE,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                mPageTable=mPageTable,
                tma_atom_K=tma_atom_K,
                tma_atom_V=tma_atom_V,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                learnable_sink=learnable_sink,
                sQ_layout=sQ_layout,
                sK_layout=sK_layout,
                sV_layout=sV_layout,
                sO_layout=sO_layout,
                sP_layout=sP_layout,
                gmem_tiled_copy_Q=gmem_tiled_copy_Q,
                gmem_tiled_copy_O=gmem_tiled_copy_O,
                tiled_mma_qk=tiled_mma_qk,
                tiled_mma_pv=tiled_mma_pv,
                SharedStorage=SharedStorage,
                tile_sched_params=tile_sched_params,
                TileScheduler=TileScheduler,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                blocksparse_tensors=blocksparse_tensors,
                mOFinal=mOFinal,
                mLSEFinal=mLSEFinal,
            )

    @cute.jit
    def _kernel_main(
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
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mOFinal: Optional[cute.Tensor] = None,
        mLSEFinal: Optional[cute.Tensor] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Per-batch split counts: a batch with one split writes O and LSE directly.
        single_split_out = None if const_expr(mOFinal is None) else (mOFinal, mLSEFinal)

        if const_expr(mPageTable is None or self.paged_tma):
            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        sP = None
        sRowScale = None
        sFinalScale = None
        sLSE = None
        if const_expr(sP_layout is not None):
            if const_expr(self._q_in_regs_pipeline()):
                sP = storage.sQ.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
            else:
                sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
            sSoftmaxStats = storage.softmax_stats.get_tensor(
                cute.make_layout(
                    (self._num_softmax_stat_rows(), self.tile_m),
                    stride=(self.tile_m, 1),
                )
            )
            sRowScale = sSoftmaxStats
            if const_expr(self._uses_n_distributed_qk()):
                num_qk_warps = self.num_qk_threads // cute.arch.WARP_SIZE
                sFinalScale = sSoftmaxStats[2 * num_qk_warps + 2, None]
                sLSE = sSoftmaxStats[3 * num_qk_warps + 3, None]
            else:
                sFinalScale = sSoftmaxStats[2, None]
                sLSE = sSoftmaxStats[3, None]

        tma_group = CooperativeGroup(Agent.Thread)
        qk_group = CooperativeGroup(Agent.Thread, self.num_qk_threads // cute.arch.WARP_SIZE)
        pv_group = CooperativeGroup(Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(mPageTable is None or self.paged_tma):
            pipeline_k = PipelineTmaAsync.create(
                num_stages=self._num_k_stages(),
                producer_group=tma_group,
                consumer_group=qk_group,
                tx_count=self.tma_copy_bytes_K,
                barrier_storage=storage.mbar_K.data_ptr(),
                defer_sync=True,
            )
            pipeline_v = PipelineTmaAsync.create(
                num_stages=self._num_v_stages(),
                producer_group=tma_group,
                consumer_group=pv_group,
                tx_count=self.tma_copy_bytes_V,
                barrier_storage=storage.mbar_V.data_ptr(),
                defer_sync=True,
            )
        else:
            dma_group = CooperativeGroup(Agent.Thread, self.num_dma_threads)
            k_consumer_group = CooperativeGroup(
                Agent.Thread,
                (self.num_qk_threads if self._uses_split_pv_warps() else self.num_threads),
            )
            v_consumer_group = CooperativeGroup(
                Agent.Thread,
                (self.num_mma_threads if self._uses_split_pv_warps() else self.num_threads),
            )
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                num_stages=self._num_k_stages(),
                producer_group=dma_group,
                consumer_group=k_consumer_group,
                barrier_storage=storage.mbar_K.data_ptr(),
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                num_stages=self._num_v_stages(),
                producer_group=dma_group,
                consumer_group=v_consumer_group,
                barrier_storage=storage.mbar_V.data_ptr(),
                defer_sync=True,
            )
        pipeline_p = None
        pipeline_final = None
        if const_expr(sP_layout is not None):
            pipeline_p = PipelineAsync.create(
                num_stages=self._num_p_stages(),
                producer_group=CooperativeGroup(Agent.Thread, self.num_qk_threads),
                consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_threads),
                barrier_storage=storage.mbar_P.data_ptr(),
                defer_sync=True,
                name="sm120_p",
            )
            pipeline_final = PipelineAsync.create(
                num_stages=1,
                producer_group=CooperativeGroup(Agent.Thread, self.num_qk_threads),
                consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_threads),
                barrier_storage=storage.mbar_final.data_ptr(),
                defer_sync=True,
                name="sm120_final",
            )
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        if work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            split_idx, num_splits = self._work_split(split_idx, tile_scheduler.params)
            seqlen = SeqlenInfoQK.create(
                batch_idx=batch_idx,
                seqlen_q_static=(mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]),
                seqlen_k_static=(
                    mK.shape[0]
                    if const_expr(mPageTable is None)
                    else mK.shape[0] * mPageTable.shape[1]
                ),
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                **self._seqlen_sparse_kwargs(blocksparse_tensors),
                tile_m=self.tile_m,
                tile_n=self.tile_n,
            )
            run_mainloop = True
            # Block-sparse splits run their (possibly empty) share of the lists in the mainloop.
            if const_expr(self.is_split_kv and blocksparse_tensors is None):
                block_info = BlockInfo(
                    self.tile_m,
                    self.tile_n,
                    self.is_causal,
                    self.is_local,
                    self.is_split_kv,
                    window_size_left,
                    window_size_right,
                    qhead_per_kvhead_packgqa=(
                        self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
                    ),
                )
                n_block_min, n_block_max = self._get_n_block_min_max(
                    block_info,
                    seqlen,
                    m_block,
                    split_idx,
                    num_splits,
                )
                is_empty_split = n_block_min >= n_block_max
                if is_empty_split:
                    # Splits past a batch's own count are never read by the combine.
                    if split_idx < num_splits:
                        empty_args = (
                            learnable_sink,
                            seqlen,
                            tiled_mma_pv,
                            tidx,
                            warp_idx,
                            m_block,
                            head_idx,
                            batch_idx,
                        )
                        if const_expr(single_split_out is not None):
                            if num_splits == 1:
                                self._epilogue_empty_split_by_role(
                                    *single_split_out, *empty_args, Int32(0)
                                )
                            else:
                                self._epilogue_empty_split_by_role(mO, mLSE, *empty_args, split_idx)
                        else:
                            self._epilogue_empty_split_by_role(mO, mLSE, *empty_args, split_idx)
                run_mainloop = not is_empty_split

            if run_mainloop:
                pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
                pipeline_init_wait(cluster_shape_mn=(1, 1))

                if const_expr(mPageTable is not None and self._uses_split_pv_warps()):
                    if warp_idx == 0:
                        self.load_paged_persistent(
                            mK,
                            mV,
                            mPageTable,
                            tma_atom_K,
                            tma_atom_V,
                            sK,
                            sV,
                            pipeline_k,
                            pipeline_v,
                            tile_scheduler,
                            mQ,
                            mCuSeqlensQ,
                            mSeqUsedQ,
                            mSeqUsedK,
                            window_size_left,
                            window_size_right,
                            tidx,
                            blocksparse_tensors=blocksparse_tensors,
                        )
                    elif const_expr(self._uses_n_distributed_qk()):
                        if warp_idx <= self.num_mma_threads // cute.arch.WARP_SIZE:
                            self.mma_persistent(
                                mQ,
                                mK,
                                mO,
                                mLSE,
                                sQ,
                                sK,
                                sV,
                                sO,
                                sP,
                                sRowScale,
                                sLSE,
                                learnable_sink,
                                pipeline_k,
                                pipeline_v,
                                gmem_tiled_copy_Q,
                                gmem_tiled_copy_O,
                                tiled_mma_qk,
                                tiled_mma_pv,
                                tidx - self.num_dma_threads,
                                softmax_scale_log2,
                                softmax_scale,
                                tile_scheduler,
                                mCuSeqlensQ,
                                mCuSeqlensK,
                                mSeqUsedQ,
                                mSeqUsedK,
                                mPageTable,
                                window_size_left,
                                window_size_right,
                                True,
                                aux_data,
                                fastdiv_mods,
                                single_split_out=single_split_out,
                            )
                    elif warp_idx == 1:
                        self.mma_persistent(
                            mQ,
                            mK,
                            mO,
                            mLSE,
                            sQ,
                            sK,
                            sV,
                            sO,
                            sP,
                            sRowScale,
                            sLSE,
                            learnable_sink,
                            pipeline_k,
                            pipeline_v,
                            gmem_tiled_copy_Q,
                            gmem_tiled_copy_O,
                            tiled_mma_qk,
                            tiled_mma_pv,
                            tidx - self.num_dma_threads,
                            softmax_scale_log2,
                            softmax_scale,
                            tile_scheduler,
                            mCuSeqlensQ,
                            mCuSeqlensK,
                            mSeqUsedQ,
                            mSeqUsedK,
                            mPageTable,
                            window_size_left,
                            window_size_right,
                            True,
                            aux_data,
                            fastdiv_mods,
                            single_split_out=single_split_out,
                        )
                    elif warp_idx <= self.num_mma_threads // cute.arch.WARP_SIZE:
                        self.mma_persistent(
                            mQ,
                            mK,
                            mO,
                            mLSE,
                            sQ,
                            sK,
                            sV,
                            sO,
                            sP,
                            sRowScale,
                            sLSE,
                            learnable_sink,
                            pipeline_k,
                            pipeline_v,
                            gmem_tiled_copy_Q,
                            gmem_tiled_copy_O,
                            tiled_mma_qk,
                            tiled_mma_pv,
                            tidx - self.num_dma_threads,
                            softmax_scale_log2,
                            softmax_scale,
                            tile_scheduler,
                            mCuSeqlensQ,
                            mCuSeqlensK,
                            mSeqUsedQ,
                            mSeqUsedK,
                            mPageTable,
                            window_size_left,
                            window_size_right,
                            False,
                            aux_data,
                            fastdiv_mods,
                            single_split_out=single_split_out,
                        )
                elif const_expr(self._uses_split_pv_warps()):
                    if warp_idx < self.num_qk_threads // cute.arch.WARP_SIZE:
                        self.mma_qk_pipeline_persistent(
                            mQ,
                            mK,
                            mV,
                            sQ,
                            sK,
                            sV,
                            tma_atom_K,
                            tma_atom_V,
                            sP,
                            sRowScale,
                            sFinalScale,
                            sLSE,
                            learnable_sink,
                            pipeline_k,
                            pipeline_v,
                            pipeline_p,
                            pipeline_final,
                            gmem_tiled_copy_Q,
                            tiled_mma_qk,
                            tidx,
                            softmax_scale_log2,
                            softmax_scale,
                            tile_scheduler,
                            mCuSeqlensQ,
                            mCuSeqlensK,
                            mSeqUsedQ,
                            mSeqUsedK,
                            window_size_left,
                            window_size_right,
                            aux_data,
                            fastdiv_mods,
                            blocksparse_tensors=blocksparse_tensors,
                        )
                    else:
                        self.mma_pv_pipeline_persistent(
                            mQ,
                            mK,
                            mO,
                            mLSE,
                            sV,
                            sO,
                            sP,
                            sRowScale,
                            sFinalScale,
                            sLSE,
                            pipeline_v,
                            pipeline_p,
                            pipeline_final,
                            gmem_tiled_copy_O,
                            tiled_mma_pv,
                            tidx - self.num_qk_threads,
                            tile_scheduler,
                            mCuSeqlensQ,
                            mCuSeqlensK,
                            mSeqUsedQ,
                            mSeqUsedK,
                            window_size_left,
                            window_size_right,
                            blocksparse_tensors=blocksparse_tensors,
                            single_split_out=single_split_out,
                        )
                elif const_expr(self._fold_dma()):
                    self.mma_persistent(
                        mQ,
                        mK,
                        mO,
                        mLSE,
                        sQ,
                        sK,
                        sV,
                        sO,
                        sP,
                        sRowScale,
                        sLSE,
                        learnable_sink,
                        pipeline_k,
                        pipeline_v,
                        gmem_tiled_copy_Q,
                        gmem_tiled_copy_O,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        tidx,
                        softmax_scale_log2,
                        softmax_scale,
                        tile_scheduler,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mSeqUsedQ,
                        mSeqUsedK,
                        mPageTable,
                        window_size_left,
                        window_size_right,
                        True,
                        aux_data,
                        fastdiv_mods,
                        fold_kv=(mK, mV, tma_atom_K, tma_atom_V),
                        blocksparse_tensors=blocksparse_tensors,
                        single_split_out=single_split_out,
                    )
                elif warp_idx == 0:
                    if const_expr(mPageTable is None):
                        self.load_tma_persistent(
                            mK,
                            mV,
                            sK,
                            sV,
                            tma_atom_K,
                            tma_atom_V,
                            pipeline_k,
                            pipeline_v,
                            tile_scheduler,
                            mQ,
                            mCuSeqlensQ,
                            mCuSeqlensK,
                            mSeqUsedQ,
                            mSeqUsedK,
                            window_size_left,
                            window_size_right,
                            blocksparse_tensors=blocksparse_tensors,
                        )
                    else:
                        self.load_paged_persistent(
                            mK,
                            mV,
                            mPageTable,
                            tma_atom_K,
                            tma_atom_V,
                            sK,
                            sV,
                            pipeline_k,
                            pipeline_v,
                            tile_scheduler,
                            mQ,
                            mCuSeqlensQ,
                            mSeqUsedQ,
                            mSeqUsedK,
                            window_size_left,
                            window_size_right,
                            tidx,
                            blocksparse_tensors=blocksparse_tensors,
                        )
                elif warp_idx <= self.num_threads // cute.arch.WARP_SIZE:
                    self.mma_persistent(
                        mQ,
                        mK,
                        mO,
                        mLSE,
                        sQ,
                        sK,
                        sV,
                        sO,
                        sP,
                        sRowScale,
                        sLSE,
                        learnable_sink,
                        pipeline_k,
                        pipeline_v,
                        gmem_tiled_copy_Q,
                        gmem_tiled_copy_O,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        tidx - self.num_dma_threads,
                        softmax_scale_log2,
                        softmax_scale,
                        tile_scheduler,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mSeqUsedQ,
                        mSeqUsedK,
                        mPageTable,
                        window_size_left,
                        window_size_right,
                        True,
                        aux_data,
                        fastdiv_mods,
                        blocksparse_tensors=blocksparse_tensors,
                        single_split_out=single_split_out,
                    )

    @cute.jit
    def _epilogue_empty_split_by_role(
        self,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        seqlen: SeqlenInfoQK,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        warp_idx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        """epilogue_empty_split from the warps that hold the O accumulator."""
        if const_expr(self._uses_split_pv_warps()):
            if warp_idx >= self.num_qk_threads // cute.arch.WARP_SIZE:
                self.epilogue_empty_split(
                    mO,
                    mLSE,
                    learnable_sink,
                    seqlen,
                    tiled_mma_pv,
                    tidx - self.num_qk_threads,
                    m_block,
                    head_idx,
                    batch_idx,
                    split_idx,
                )
        elif const_expr(self._fold_dma()):
            self.epilogue_empty_split(
                mO,
                mLSE,
                learnable_sink,
                seqlen,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )
        else:
            if 0 < warp_idx <= self.num_threads // cute.arch.WARP_SIZE:
                self.epilogue_empty_split(
                    mO,
                    mLSE,
                    learnable_sink,
                    seqlen,
                    tiled_mma_pv,
                    tidx - self.num_dma_threads,
                    m_block,
                    head_idx,
                    batch_idx,
                    split_idx,
                )

    @cute.jit
    def epilogue_empty_split(
        self,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        seqlen: SeqlenInfoQK,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        """Write the reduction identity for a split with no visible K/V tile."""
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        tOcO_mn = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
        qhead_pack = self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
        row_limit = seqlen.seqlen_q * qhead_pack
        row_offset = (
            seqlen.offset_q * qhead_pack if const_expr(seqlen.has_cu_seqlens_q) else Int32(0)
        )

        if const_expr(mLSE is not None):
            if tOcO_mn[0][1] == 0:
                for r in cutlass.range(cute.size(tOcO_mn, mode=[0]), unroll_full=True):
                    row = m_block * self.tile_m + tOcO_mn[r, 0][0]
                    if row < row_limit:
                        lse = -Float32.inf
                        if const_expr(learnable_sink is not None):
                            if split_idx == 0:
                                q_head_idx = (
                                    row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                                    if const_expr(self.pack_gqa)
                                    else head_idx
                                )
                                lse = Float32(learnable_sink[q_head_idx])
                        if const_expr(seqlen.has_cu_seqlens_q):
                            mLSE[row_offset + row, head_idx, split_idx] = lse
                        else:
                            mLSE[row, head_idx, batch_idx, split_idx] = lse

        for r in cutlass.range(cute.size(tOcO_mn, mode=[0]), unroll_full=True):
            row = m_block * self.tile_m + tOcO_mn[r, 0][0]
            if row < row_limit:
                for c in cutlass.range(cute.size(tOcO_mn, mode=[1]), unroll_full=True):
                    col = tOcO_mn[r, c][1]
                    if const_expr(not self.check_hdim_v_oob) or col < mO.shape[1]:
                        if const_expr(seqlen.has_cu_seqlens_q):
                            mO[row_offset + row, col, head_idx, split_idx] = mO.element_type(0.0)
                        else:
                            mO[row, col, head_idx, batch_idx, split_idx] = mO.element_type(0.0)

    @cute.jit
    def load_tma_persistent(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mQ: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        producer_state_k = PipelineState(self._num_k_stages(), Int32(0), Int32(0), Int32(1))
        producer_state_v = PipelineState(self._num_v_stages(), Int32(0), Int32(0), Int32(1))
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        split_idx, num_splits = self._work_split(split_idx, tile_scheduler.params)
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]),
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            **self._seqlen_sparse_kwargs(blocksparse_tensors),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = self._get_n_block_min_max(
            block_info,
            seqlen,
            m_block,
            split_idx,
            num_splits,
        )
        head_idx_kv = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        producer_state_k, producer_state_v = self.load_tma(
            mK,
            mV,
            sK,
            sV,
            tma_atom_K,
            tma_atom_V,
            pipeline_k,
            pipeline_v,
            producer_state_k,
            producer_state_v,
            seqlen,
            n_block_min,
            n_block_max,
            head_idx_kv,
            batch_idx,
            sparse=self._sparse_kv_list(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                m_block,
                seqlen,
                split_idx,
                num_splits,
            ),
        )

        pipeline_k.producer_tail(producer_state_k)
        pipeline_v.producer_tail(producer_state_v)

    @cute.jit
    def load_paged_persistent(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPageTable: cute.Tensor,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        sK: cute.Tensor,
        sV: cute.Tensor,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mQ: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tidx: Int32,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        producer_state_k = PipelineState(self._num_k_stages(), Int32(0), Int32(0), Int32(1))
        producer_state_v = PipelineState(self._num_v_stages(), Int32(0), Int32(0), Int32(1))
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        split_idx, num_splits = self._work_split(split_idx, tile_scheduler.params)
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]),
            seqlen_k_static=mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=None,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            **self._seqlen_sparse_kwargs(blocksparse_tensors),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = self._get_n_block_min_max(
            block_info,
            seqlen,
            m_block,
            split_idx,
            num_splits,
        )
        head_idx_kv = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        sparse = self._sparse_kv_list(
            blocksparse_tensors,
            batch_idx,
            head_idx,
            m_block,
            seqlen,
            split_idx,
            num_splits,
        )
        if const_expr(self.paged_tma):
            mK_cur = mK[None, None, head_idx_kv, None]
            mV_cur = mV[None, None, head_idx_kv, None]
            gK = cute.local_tile(
                mK_cur,
                (self.tile_n, self.tile_hdim),
                (0, 0, None),
            )
            gV = cute.local_tile(
                mV_cur,
                (self.tile_hdimv, self.tile_n),
                (0, 0, None),
            )
            copy_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                0,
                cute.make_layout(1),
                gK,
                sK,
            )
            copy_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                gV,
                sV,
            )
            num_n_blocks = self._kv_iters(sparse, n_block_min, n_block_max)
            for n_tile in cutlass.range(num_n_blocks, unroll=1):
                n_block = self._kv_block_of(sparse, n_tile, n_block_min, n_block_max)
                page_idx = mPageTable[batch_idx, n_block]
                pipeline_k.producer_acquire(producer_state_k)
                copy_K(
                    src_idx=page_idx,
                    dst_idx=producer_state_k.index,
                    tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
                )
                pipeline_k.producer_commit(producer_state_k)
                producer_state_k.advance()

                pipeline_v.producer_acquire(producer_state_v)
                copy_V(
                    src_idx=page_idx,
                    dst_idx=producer_state_v.index,
                    tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state_v),
                )
                pipeline_v.producer_commit(producer_state_v)
                producer_state_v.advance()

            pipeline_k.producer_tail(producer_state_k)
            pipeline_v.producer_tail(producer_state_v)
            return

        paged_kv_manager = Sm120PagedKVManager.create(
            mPageTable,
            mK,
            mV,
            FastDivmodDivisor(mK.shape[0]),
            batch_idx,
            head_idx_kv,
            tidx,
            seqlen.seqlen_k,
            0,
            self.tile_n,
            self.tile_hdim,
            self.tile_hdimv,
            self.num_dma_threads,
            mK.element_type,
        )
        num_n_blocks = self._kv_iters(sparse, n_block_min, n_block_max)
        for n_tile in cutlass.range(num_n_blocks, unroll=1):
            n_block = self._kv_block_of(sparse, n_tile, n_block_min, n_block_max)
            paged_kv_manager.load_page_table(n_block)
            pipeline_k.producer_acquire(producer_state_k)
            paged_kv_manager.load_KV(
                n_block,
                sK[None, None, producer_state_k.index],
                "K",
            )
            cute.arch.cp_async_commit_group()
            pipeline_k.producer_commit(producer_state_k)
            producer_state_k.advance()

            pipeline_v.producer_acquire(producer_state_v)
            sV_stage = layout_utils.transpose_view(sV[None, None, producer_state_v.index])
            paged_kv_manager.load_KV(n_block, sV_stage, "V")
            cute.arch.cp_async_commit_group()
            pipeline_v.producer_commit(producer_state_v)
            producer_state_v.advance()

        pipeline_k.producer_tail(producer_state_k)
        pipeline_v.producer_tail(producer_state_v)

    @cute.jit
    def mma_qk_pipeline_persistent(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        split_idx, num_splits = self._work_split(split_idx, tile_scheduler.params)
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]),
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            **self._seqlen_sparse_kwargs(blocksparse_tensors),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = self._get_n_block_min_max(
            block_info,
            seqlen,
            m_block,
            split_idx,
            num_splits,
        )
        sparse = self._sparse_kv_list(
            blocksparse_tensors,
            batch_idx,
            head_idx,
            m_block,
            seqlen,
            split_idx,
            num_splits,
        )
        head_idx_kv = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        self.mma_qk_pipeline(
            mQ,
            mK,
            mV,
            sQ,
            sK,
            sV,
            tma_atom_K,
            tma_atom_V,
            sP,
            sRowScale,
            sFinalScale,
            sLSE,
            learnable_sink,
            pipeline_k,
            pipeline_v,
            pipeline_p,
            pipeline_final,
            gmem_tiled_copy_Q,
            tiled_mma_qk,
            tidx,
            softmax_scale_log2,
            softmax_scale,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            head_idx,
            head_idx_kv,
            batch_idx,
            split_idx,
            aux_data,
            fastdiv_mods,
            sparse=sparse,
        )

    @cute.jit
    def mma_pv_pipeline_persistent(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        single_split_out: Optional[tuple] = None,
    ):
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        split_idx, num_splits = self._work_split(split_idx, tile_scheduler.params)
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]),
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            **self._seqlen_sparse_kwargs(blocksparse_tensors),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = self._get_n_block_min_max(
            block_info,
            seqlen,
            m_block,
            split_idx,
            num_splits,
        )
        sparse = self._sparse_kv_list(
            blocksparse_tensors,
            batch_idx,
            head_idx,
            m_block,
            seqlen,
            split_idx,
            num_splits,
        )
        self.mma_pv_pipeline(
            mO,
            mLSE,
            sV,
            sO,
            sP,
            sRowScale,
            sFinalScale,
            sLSE,
            pipeline_v,
            pipeline_p,
            pipeline_final,
            gmem_tiled_copy_O,
            tiled_mma_pv,
            tidx,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
            sparse=sparse,
            single_split_out=(
                None if const_expr(single_split_out is None) else (*single_split_out, num_splits)
            ),
        )

    @cute.jit
    def _run_sparse_schedule(self, compute_one_n_block, role_state, mask_fn, sparse):
        """_run_n_block_schedule over the block-sparse iterations (the QK and PV warp sets run
        the same count; only the QK side masks)."""
        masks = None if const_expr(mask_fn is None) else self._sparse_masks(mask_fn)
        if sparse[5] > 0:
            role_state = self._sparse_schedule_step(
                compute_one_n_block, role_state, masks, sparse, 0, True
            )
            for it in cutlass.range(1, sparse[5], unroll=1):
                role_state = self._sparse_schedule_step(
                    compute_one_n_block, role_state, masks, sparse, it, False
                )
        return role_state

    @cute.jit
    def _sparse_schedule_step(
        self, compute_one_n_block, role_state, masks, sparse, it, is_first: cutlass.Constexpr[bool]
    ):
        n_block, is_full, list_head = self._sparse_kv_block(sparse, it)
        mask_fn = None
        if const_expr(masks is not None):
            mask_fn = partial(masks, is_full=is_full, list_head=list_head)
        return compute_one_n_block(n_block, role_state, mask_fn=mask_fn, is_first_n_block=is_first)

    def _local_edge_masks(self, mask_fn, block_info, seqlen, m_block, n_block_min):
        """(diagonal, left) masks of a local tile without a mask_mod: a block that only the
        window's right edge can cut takes the right edge alone (one R2P bitmask, as causal),
        and the blocks past the left edge take the left edge alone, instead of both edges on
        every block. None when not local or with a mask_mod."""
        if mask_fn is None or not self.is_local or self.mask_mod is not None:
            return None, None
        n_left = block_info.get_n_block_min_before_local_mask(seqlen, m_block, n_block_min)
        diag = partial(
            self._local_diag_mask,
            right=partial(mask_fn, mask_seqlen=True, local_sides="right"),
            both=partial(mask_fn, mask_seqlen=True),
            n_left=n_left,
        )
        return diag, partial(mask_fn, mask_seqlen=False, local_sides="left")

    @cute.jit
    def _local_diag_mask(self, acc_S, n_block, right, both, n_left):
        if n_block >= n_left:
            right(acc_S, n_block=n_block)
        else:
            both(acc_S, n_block=n_block)

    def _band_mask_fn(self, mask_fn: Optional[Callable]) -> Optional[Callable]:
        """Mask of a KV block that the causal / local band leaves fully visible: only a mask_mod
        still applies. Running the band mask there masks nothing but costs its per-element
        compares on every block. The hd-128 QK-pipelined loops keep the band mask: they run at
        the 255-register cap, and the separate masked and unmasked block bodies spill there."""
        if mask_fn is None or self.mask_mod is None:
            return None
        return partial(
            mask_fn, mask_mod=self.mask_mod, mask_seqlen=False, mask_causal=False, mask_local=False
        )

    @cute.jit
    def _run_n_block_schedule(
        self,
        compute_one_n_block: Callable,
        role_state,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        mask_fn: Optional[Callable],
        sparse: Optional[tuple] = None,
    ):
        if const_expr(sparse is not None):
            return self._run_sparse_schedule(compute_one_n_block, role_state, mask_fn, sparse)
        if const_expr(mask_fn is not None):
            mask_fn_seqlen = partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True)
            mask_fn_no_seqlen = partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False)
        else:
            mask_fn_seqlen = None
            mask_fn_no_seqlen = None
        local_diag, local_left = self._local_edge_masks(
            mask_fn, block_info, seqlen, m_block, n_block_min
        )
        if const_expr(local_diag is not None):
            mask_fn_seqlen, mask_fn_no_seqlen = local_diag, local_left

        n_block = cutlass.max(n_block_max - 1, 0)
        role_state = compute_one_n_block(
            n_block,
            role_state,
            mask_fn=mask_fn_seqlen,
            is_first_n_block=True,
        )
        n_block_upper = n_block
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_local_mask, unroll=1):
                n_block = n_block_max - 2 - n_tile
                role_state = compute_one_n_block(
                    n_block,
                    role_state,
                    mask_fn=mask_fn_seqlen,
                )
            n_block_upper = cutlass.min(n_block_upper, n_block_min_causal_local_mask)
        n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
            seqlen, m_block, n_block_min
        )
        mask_fn_band = self._band_mask_fn(mask_fn)
        for n_tile in cutlass.range(n_block_upper - n_block_min_before_local_mask, unroll=1):
            role_state = compute_one_n_block(
                n_block_upper - n_tile - 1,
                role_state,
                mask_fn=mask_fn_band,
            )
        if const_expr(self.is_local and block_info.window_size_left is not None):
            n_block_upper = cutlass.min(n_block_upper, n_block_min_before_local_mask)
            for n_tile in cutlass.range(n_block_upper - n_block_min, unroll=1):
                role_state = compute_one_n_block(
                    n_block_upper - n_tile - 1,
                    role_state,
                    mask_fn=mask_fn_no_seqlen,
                )
        return role_state

    @cute.jit
    def mma_qk_pipeline(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        head_idx: Int32,
        head_idx_kv: Int32,
        batch_idx: Int32,
        split_idx: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        sparse: Optional[tuple] = None,
    ):
        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
        if const_expr(not self.pack_gqa):
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
        if const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
        else:
            mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
            mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gV = cute.local_tile(mV_cur, (self.tile_hdimv, self.tile_n), (0, None))
        copy_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)
        copy_V, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V, 0, cute.make_layout(1), gV, sV)

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        smem_copy_atom_qk = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_thr_copy_q = utils.make_tiled_copy_A(smem_copy_atom_qk, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_k = utils.make_tiled_copy_B(smem_copy_atom_qk, tiled_mma_qk).get_slice(tidx)
        tCrQ = None
        if const_expr(self._q_in_regs_pipeline()):
            tCrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        smem_store_atom_p = utils.get_smem_store_atom(120, self.dtype)
        smem_thr_store_p = cute.make_tiled_copy_C(smem_store_atom_p, tiled_mma_qk).get_slice(tidx)
        tPsP_store = smem_thr_store_p.partition_D(sP)

        # The first KV block's K is issued before Q, so the two loads overlap at the tile's
        # start. Later K loads are not issued as soon as QK(n) releases the stage: that would
        # hold warp 0, which also issues V, behind the slowest QK warp.
        k_prefetched = const_expr(sparse is None)
        producer_state_k = PipelineState(self._num_k_stages(), Int32(0), Int32(0), Int32(1))
        if const_expr(k_prefetched):
            if tidx < cute.arch.WARP_SIZE:
                pipeline_k.producer_acquire(producer_state_k)
                copy_K(
                    src_idx=cutlass.max(n_block_max - 1, 0),
                    dst_idx=producer_state_k.index,
                    tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
                )
                pipeline_k.producer_commit(producer_state_k)
            producer_state_k.advance()
        gmem_thr_copy_q = gmem_tiled_copy_Q.get_slice(tidx)
        if const_expr(not self.pack_gqa):
            self.load_Q(
                gmem_thr_copy_q,
                gQ,
                sQ,
                m_block,
                seqlen=seqlen.seqlen_q,
                headdim=mQ.shape[1],
            )
        else:
            PackGQA(
                self.tile_m,
                self.tile_hdim,
                self.check_hdim_oob,
                self.qhead_per_kvhead,
            ).load_Q(
                mQ_cur,
                sQ,
                gmem_tiled_copy_Q,
                tidx,
                m_block,
                seqlen.seqlen_q,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_Q_load_threads,
        )
        if const_expr(self._q_in_regs_pipeline()):
            tCsQ = smem_thr_copy_q.partition_S(sQ)
            tCrQ_copy_view = smem_thr_copy_q.retile(tCrQ)
            cute.copy(smem_thr_copy_q, tCsQ, tCrQ_copy_view)
            # All QK warps must finish reading Q before its allocation becomes P.
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_Q_load_threads,
            )

        acc_shape_s = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_shape_s[0][0] * acc_shape_s[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        mask = AttentionMask(
            self.tile_m,
            self.tile_n,
            seqlen,
            block_info.window_size_left,
            block_info.window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            enable_r2p_optimization=not self.split_qk_n,
        )
        mask_fn = partial(
            mask.apply_mask,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            thr_mma=thr_mma_qk,
            mask_causal=self.is_causal,
            mask_local=self.is_local,
            aux_data=aux_data,
            fastdiv_mods=(fastdiv_mods if const_expr(self.mask_mod is not None) else None),
        )
        compute_one_n_block = partial(
            self.compute_one_n_block_qk_pipeline,
            thr_mma_qk=thr_mma_qk,
            sQ=sQ,
            tCrQ=tCrQ,
            sK=sK,
            tPsP_store=tPsP_store,
            smem_thr_copy_q=smem_thr_copy_q,
            smem_thr_copy_k=smem_thr_copy_k,
            smem_thr_store_p=smem_thr_store_p,
            tScS_mn=tScS_mn,
            copy_K=copy_K,
            copy_V=copy_V,
            tidx=tidx,
            sRowScale=sRowScale,
            softmax=softmax,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            pipeline_p=pipeline_p,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            seqlen=seqlen,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
            first_k_prefetched=k_prefetched,
        )
        role_state = (
            producer_state_k,
            PipelineState(self._num_v_stages(), Int32(0), Int32(0), Int32(1)),
            PipelineState(self._num_k_stages(), Int32(0), Int32(0), Int32(0)),
            PipelineState(self._num_p_stages(), Int32(0), Int32(0), Int32(1)),
        )
        producer_state_k, producer_state_v, k_state, p_state = self._run_n_block_schedule(
            compute_one_n_block,
            role_state,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            mask_fn,
            sparse=sparse,
        )

        sink_val = None
        if const_expr(learnable_sink is not None):
            if const_expr(not self.pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            else:
                sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                    row = m_block * self.tile_m + tScS_mn[r][0]
                    q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                    sink_val[r] = Float32(learnable_sink[q_head_idx])
        if const_expr(self.is_split_kv and learnable_sink is not None):
            if const_expr(not self.pack_gqa):
                sink_val = sink_val if split_idx == 0 else -Float32.inf
            elif split_idx != 0:
                sink_val.fill(-Float32.inf)
        row_scale = softmax.finalize(sink_val=sink_val)
        final_state = PipelineState(1, Int32(0), Int32(0), Int32(1))
        pipeline_final.producer_acquire(final_state)
        if tScS_mn[0][1] == 0:
            for r in cutlass.range(cute.size(row_scale), unroll_full=True):
                row = tScS_mn[r][0]
                sFinalScale[row] = row_scale[r]
                sLSE[row] = softmax.row_sum[r]
        cute.arch.fence_view_async_shared()
        pipeline_final.producer_commit(final_state)
        final_state.advance()

        pipeline_p.producer_tail(p_state)
        pipeline_final.producer_tail(final_state)
        if tidx < cute.arch.WARP_SIZE:
            pipeline_k.producer_tail(producer_state_k)
            pipeline_v.producer_tail(producer_state_v)

    @cute.jit
    def compute_one_n_block_qk_pipeline(
        self,
        n_block: Int32,
        role_state,
        thr_mma_qk: cute.TiledMma,
        sQ: cute.Tensor,
        tCrQ: Optional[cute.Tensor],
        sK: cute.Tensor,
        tPsP_store: cute.Tensor,
        smem_thr_copy_q: cute.TiledCopy,
        smem_thr_copy_k: cute.TiledCopy,
        smem_thr_store_p: cute.TiledCopy,
        tScS_mn: cute.Tensor,
        copy_K: Callable,
        copy_V: Callable,
        tidx: Int32,
        sRowScale: cute.Tensor,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        first_k_prefetched: cutlass.Constexpr = False,
    ):
        """With first_k_prefetched, the first block's K was issued before the tile's Q."""
        issue_k = const_expr(not (first_k_prefetched and is_first_n_block))
        producer_state_k, producer_state_v, k_state, p_state = role_state
        if tidx < cute.arch.WARP_SIZE:
            if const_expr(issue_k):
                pipeline_k.producer_acquire(producer_state_k)
                copy_K(
                    src_idx=n_block,
                    dst_idx=producer_state_k.index,
                    tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
                )
                pipeline_k.producer_commit(producer_state_k)
            pipeline_v.producer_acquire(producer_state_v)
            copy_V(
                src_idx=n_block,
                dst_idx=producer_state_v.index,
                tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state_v),
            )
            pipeline_v.producer_commit(producer_state_v)
        if const_expr(issue_k):
            producer_state_k.advance()
        producer_state_v.advance()

        pipeline_p.producer_acquire(p_state)

        k_wait_token = pipeline_k.consumer_try_wait(k_state)
        pipeline_k.consumer_wait(k_state, k_wait_token)

        acc_shape_s = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_s = cute.make_rmem_tensor(acc_shape_s, Float32)
        acc_s.fill(0.0)
        if const_expr(self._q_in_regs_pipeline()):
            self._gemm_qk_a_in_regs(
                thr_mma_qk,
                acc_s,
                tCrQ,
                sK[None, None, k_state.index],
                smem_thr_copy_k,
            )
        else:
            self._gemm_qk_phase_local(
                thr_mma_qk,
                acc_s,
                sQ,
                sK[None, None, k_state.index],
                smem_thr_copy_q,
                smem_thr_copy_k,
            )
        pipeline_k.consumer_release(k_state)
        k_state.advance()

        if const_expr(self.score_mod is not None):
            self.apply_score_mod(
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_s,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_s, n_block=n_block)
        row_scale = softmax.online_softmax(acc_s, is_first=is_first_n_block)
        rP = cute.make_fragment_like(acc_s, self.dtype)
        rP.store(acc_s.load().to(self.dtype))
        tOrP_qk = layout_utils.reshape_acc_to_frgA(rP)
        tPrP = smem_thr_store_p.retile(tOrP_qk)
        cute.copy(
            smem_thr_store_p,
            tPrP,
            tPsP_store[None, None, None, p_state.index],
        )
        self._publish_row_scale(row_scale, tScS_mn, sRowScale[p_state.index, None])
        cute.arch.fence_view_async_shared()
        pipeline_p.producer_commit(p_state)
        p_state.advance()
        return producer_state_k, producer_state_v, k_state, p_state

    @cute.jit
    def mma_pv_pipeline(
        self,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
        sparse: Optional[tuple] = None,
        single_split_out: Optional[tuple] = None,
    ):
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        acc_shape_o = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_o = cute.make_rmem_tensor(acc_shape_o, Float32)
        acc_o.fill(0.0)

        smem_copy_atom_p = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_v = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_p = utils.make_tiled_copy_A(smem_copy_atom_p, tiled_mma_pv).get_slice(tidx)
        smem_thr_copy_v = utils.make_tiled_copy_B(smem_copy_atom_v, tiled_mma_pv).get_slice(tidx)
        tPsP = smem_thr_copy_p.partition_S(sP)
        tOrP = thr_mma_pv.make_fragment_A(thr_mma_pv.partition_A(sP[None, None, 0]))
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        tOcO_mn = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
        compute_one_n_block = partial(
            self.compute_one_n_block_pv_pipeline,
            thr_mma_pv=thr_mma_pv,
            acc_o=acc_o,
            tOrP=tOrP,
            tPsP=tPsP,
            sV=sV,
            sRowScale=sRowScale,
            tOcO_mn=tOcO_mn,
            smem_thr_copy_p=smem_thr_copy_p,
            smem_thr_copy_v=smem_thr_copy_v,
            pipeline_v=pipeline_v,
            pipeline_p=pipeline_p,
        )
        role_state = (
            PipelineState(self._num_v_stages(), Int32(0), Int32(0), Int32(0)),
            PipelineState(self._num_p_stages(), Int32(0), Int32(0), Int32(0)),
        )
        v_state, p_state = self._run_n_block_schedule(
            compute_one_n_block,
            role_state,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            None,
            sparse=sparse,
        )

        final_state = PipelineState(1, Int32(0), Int32(0), Int32(0))
        pipeline_final.consumer_wait(final_state)
        num_rows_pv = acc_o.shape[0][0] * acc_o.shape[1]
        row_scale = cute.make_rmem_tensor(num_rows_pv, Float32)
        lse = cute.make_rmem_tensor(num_rows_pv, Float32)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            row = tOcO_mn[r, 0][0]
            row_scale[r] = sFinalScale[row]
            lse[r] = sLSE[row]
        pipeline_final.consumer_release(final_state)
        final_state.advance()
        self._rescale_O(acc_o, row_scale)
        self.epilogue(
            acc_o,
            lse,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
            single_split_out=single_split_out,
        )

    @cute.jit
    def compute_one_n_block_pv_pipeline(
        self,
        n_block: Int32,
        role_state,
        thr_mma_pv: cute.TiledMma,
        acc_o: cute.Tensor,
        tOrP: cute.Tensor,
        tPsP: cute.Tensor,
        sV: cute.Tensor,
        sRowScale: cute.Tensor,
        tOcO_mn: cute.Tensor,
        smem_thr_copy_p: cute.TiledCopy,
        smem_thr_copy_v: cute.TiledCopy,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        v_state, p_state = role_state
        p_wait_token = pipeline_p.consumer_try_wait(p_state)
        pipeline_p.consumer_wait(p_state, p_wait_token)
        num_rows_pv = acc_o.shape[0][0] * acc_o.shape[1]
        row_scale = cute.make_rmem_tensor(num_rows_pv, Float32)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            row_scale[r] = sRowScale[p_state.index, tOcO_mn[r, 0][0]]
        self._rescale_O(acc_o, row_scale)
        tOrP_copy_view = smem_thr_copy_p.retile(tOrP)
        cute.copy(
            smem_thr_copy_p,
            tPsP[None, None, None, p_state.index],
            tOrP_copy_view,
        )
        pipeline_p.consumer_release(p_state)
        p_state.advance()

        v_wait_token = pipeline_v.consumer_try_wait(v_state)
        pipeline_v.consumer_wait(v_state, v_wait_token)
        self._gemm_pv_phase_local(
            thr_mma_pv,
            acc_o,
            tOrP,
            sV[None, None, v_state.index],
            smem_thr_copy_v,
        )
        pipeline_v.consumer_release(v_state)
        v_state.advance()
        return v_state, p_state

    @cute.jit
    def mma_persistent(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sRowScale: Optional[cute.Tensor],
        sLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        is_qk_owner: cutlass.Constexpr[bool],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        fold_kv: Optional[tuple] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        single_split_out: Optional[tuple] = None,
    ):
        consumer_state = PipelineState(self.num_stages, Int32(0), Int32(0), Int32(0))
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        split_idx, num_splits = self._work_split(split_idx, tile_scheduler.params)
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]),
            seqlen_k_static=(
                mK.shape[0] if const_expr(mPageTable is None) else mK.shape[0] * mPageTable.shape[1]
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            **self._seqlen_sparse_kwargs(blocksparse_tensors),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = self._get_n_block_min_max(
            block_info,
            seqlen,
            m_block,
            split_idx,
            num_splits,
        )
        mma_fn = partial(
            self.mma,
            mQ,
            mO,
            mLSE,
            sQ,
            sK,
            sV,
            sO,
            sP,
            sRowScale,
            sLSE,
            learnable_sink,
            pipeline_k,
            pipeline_v,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tidx,
            softmax_scale_log2,
            softmax_scale,
            consumer_state,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
            is_qk_owner,
            aux_data,
            fastdiv_mods,
        )
        # Only block-sparse calls pass the lists (subclasses override mma without them).
        sparse_kwargs = {}
        if const_expr(blocksparse_tensors is not None):
            sparse_kwargs = {
                "sparse": self._sparse_kv_list(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    split_idx,
                    num_splits,
                )
            }
        if const_expr(single_split_out is not None):
            sparse_kwargs["single_split_out"] = (*single_split_out, num_splits)
        if const_expr(fold_kv is not None):
            mma_fn(fold_kv=fold_kv, **sparse_kwargs)
        else:
            mma_fn(**sparse_kwargs)

    @cute.jit
    def load_tma(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        producer_state_k: PipelineState,
        producer_state_v: PipelineState,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        sparse: Optional[tuple] = None,
    ):
        if const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, head_idx, batch_idx]
            mV_cur = mV[None, None, head_idx, batch_idx]
        else:
            mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx])
            mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx])
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gV = cute.local_tile(mV_cur, (self.tile_hdimv, self.tile_n), (0, None))
        copy_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)
        copy_V, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V, 0, cute.make_layout(1), gV, sV)
        if const_expr(sparse is not None):
            return self._load_tma_sparse(
                copy_K, copy_V, pipeline_k, pipeline_v, producer_state_k, producer_state_v, sparse
            )
        num_n_blocks = cutlass.max(n_block_max - n_block_min, 1)
        if const_expr(self._qk_pipelined()):
            # K runs one block ahead of V, as the consumer issues QK(j + 1) before PV(j).
            pipeline_k.producer_acquire(producer_state_k)
            copy_K(
                src_idx=cutlass.max(n_block_max - 1, n_block_min),
                dst_idx=producer_state_k.index,
                tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
            )
            pipeline_k.producer_commit(producer_state_k)
            producer_state_k.advance()
            for n_tile in cutlass.range(num_n_blocks, unroll=1):
                if n_tile + 1 < num_n_blocks:
                    pipeline_k.producer_acquire(producer_state_k)
                    copy_K(
                        src_idx=cutlass.max(n_block_max - 2 - n_tile, n_block_min),
                        dst_idx=producer_state_k.index,
                        tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
                    )
                    pipeline_k.producer_commit(producer_state_k)
                    producer_state_k.advance()
                pipeline_v.producer_acquire(producer_state_v)
                copy_V(
                    src_idx=cutlass.max(n_block_max - 1 - n_tile, n_block_min),
                    dst_idx=producer_state_v.index,
                    tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state_v),
                )
                pipeline_v.producer_commit(producer_state_v)
                producer_state_v.advance()
            return producer_state_k, producer_state_v
        for n_tile in cutlass.range(num_n_blocks, unroll=1):
            n_block = cutlass.max(n_block_max - 1 - n_tile, n_block_min)
            pipeline_k.producer_acquire(producer_state_k)
            copy_K(
                src_idx=n_block,
                dst_idx=producer_state_k.index,
                tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
            )
            pipeline_k.producer_commit(producer_state_k)
            producer_state_k.advance()

            pipeline_v.producer_acquire(producer_state_v)
            copy_V(
                src_idx=n_block,
                dst_idx=producer_state_v.index,
                tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state_v),
            )
            pipeline_v.producer_commit(producer_state_v)
            producer_state_v.advance()
        return producer_state_k, producer_state_v

    @cute.jit
    def mma(
        self,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sRowScale: Optional[cute.Tensor],
        sLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        consumer_state: PipelineState,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
        is_qk_owner: cutlass.Constexpr[bool],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        fold_kv: Optional[tuple] = None,
        sparse: Optional[tuple] = None,
        single_split_out: Optional[tuple] = None,
    ):
        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
        if const_expr(not self.pack_gqa):
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
        split_pv_warps = self._uses_split_pv_warps()
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
        if const_expr(split_pv_warps):
            tOrV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV[None, None, 0]))
            tOsV = smem_thr_copy_V.partition_S(sV)
            smem_thr_copy_P = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_pv).get_slice(
                tidx
            )
            tPsP = smem_thr_copy_P.partition_S(sP)
            tOrP = thr_mma_pv.make_fragment_A(thr_mma_pv.partition_A(sP[None, None, 0]))
        if const_expr(not split_pv_warps or is_qk_owner):
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(
                tidx
            )
            smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(
                tidx
            )
            if const_expr(split_pv_warps):
                tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
                tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
                tSsQ = smem_thr_copy_Q.partition_S(sQ)
                tSsK = smem_thr_copy_K.partition_S(sK)
                smem_store_atom_P = utils.get_smem_store_atom(
                    120,
                    self.dtype,
                )
                smem_thr_store_P = cute.make_tiled_copy_C(
                    smem_store_atom_P, tiled_mma_qk
                ).get_slice(tidx)
                tPsP_store = smem_thr_store_P.partition_D(sP)

        if const_expr(not split_pv_warps or is_qk_owner):
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            if const_expr(not self.pack_gqa):
                self.load_Q(
                    gmem_thr_copy_Q,
                    gQ,
                    sQ,
                    m_block,
                    seqlen=seqlen.seqlen_q,
                    headdim=mQ.shape[1],
                )
            else:
                PackGQA(
                    self.tile_m,
                    self.tile_hdim,
                    self.check_hdim_oob,
                    self.qhead_per_kvhead,
                ).load_Q(
                    mQ_cur,
                    sQ,
                    gmem_tiled_copy_Q,
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier(
                barrier_id=1,
                number_of_threads=self.num_Q_load_threads,
            )

        if const_expr(not split_pv_warps or is_qk_owner):
            if const_expr(self._uses_n_distributed_qk()):
                softmax = None
                mma_params = SimpleNamespace(
                    thr_mma_qk=thr_mma_qk,
                    thr_mma_pv=thr_mma_pv,
                    tSrQ=tSrQ,
                    tSrK=tSrK,
                    tOrV=tOrV,
                    acc_O=acc_O,
                    tOrP=tOrP,
                    tidx=tidx,
                )
                smem_copy_params = SimpleNamespace(
                    smem_thr_copy_Q=smem_thr_copy_Q,
                    smem_thr_copy_K=smem_thr_copy_K,
                    smem_thr_copy_V=smem_thr_copy_V,
                    tOsV=tOsV,
                    smem_thr_copy_P=smem_thr_copy_P,
                    tPsP=tPsP,
                    smem_thr_store_P=(smem_thr_store_P if const_expr(self.split_qk_n) else None),
                    tPsP_store=(tPsP_store if const_expr(self.split_qk_n) else None),
                    sQ=sQ,
                    sK=sK,
                    sP=sP,
                    sRowScale=sRowScale,
                    sLSE=sLSE,
                    softmax_scale_log2=softmax_scale_log2,
                    softmax_scale=softmax_scale,
                )
                if const_expr(self.split_qk_n):
                    mask = AttentionMask(
                        self.tile_m,
                        self.tile_n,
                        seqlen,
                        block_info.window_size_left,
                        block_info.window_size_right,
                        (self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
                        enable_r2p_optimization=False,
                    )
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        m_block=m_block,
                        thr_mma=thr_mma_qk,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        aux_data=aux_data,
                        fastdiv_mods=(
                            fastdiv_mods if const_expr(self.mask_mod is not None) else None
                        ),
                    )
                else:
                    mask_fn = None
            else:
                softmax = Softmax.create(
                    softmax_scale_log2,
                    num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                    softmax_scale=softmax_scale,
                )
                softmax.reset()
                if const_expr(split_pv_warps):
                    mma_params = SimpleNamespace(
                        thr_mma_qk=thr_mma_qk,
                        thr_mma_pv=thr_mma_pv,
                        tSrQ=tSrQ,
                        tSrK=tSrK,
                        tOrV=tOrV,
                        acc_O=acc_O,
                        tOrP=tOrP,
                    )
                    smem_copy_params = SimpleNamespace(
                        smem_thr_copy_Q=smem_thr_copy_Q,
                        smem_thr_copy_K=smem_thr_copy_K,
                        smem_thr_copy_V=smem_thr_copy_V,
                        tSsQ=tSsQ,
                        tSsK=tSsK,
                        tOsV=tOsV,
                        smem_thr_store_P=smem_thr_store_P,
                        tPsP_store=tPsP_store,
                        smem_thr_copy_P=smem_thr_copy_P,
                        tPsP=tPsP,
                        sRowScale=sRowScale,
                        sLSE=sLSE,
                    )
                else:
                    mma_params = SimpleNamespace(
                        thr_mma_qk=thr_mma_qk,
                        thr_mma_pv=thr_mma_pv,
                        acc_O=acc_O,
                    )
                    smem_copy_params = SimpleNamespace(
                        smem_thr_copy_Q=smem_thr_copy_Q,
                        smem_thr_copy_K=smem_thr_copy_K,
                        smem_thr_copy_V=smem_thr_copy_V,
                        sQ=sQ,
                        sK=sK,
                        sV=sV,
                    )
                mask = AttentionMask(
                    self.tile_m,
                    self.tile_n,
                    seqlen,
                    block_info.window_size_left,
                    block_info.window_size_right,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                )
                mask_fn = partial(
                    mask.apply_mask,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    aux_data=aux_data,
                    fastdiv_mods=(fastdiv_mods if const_expr(self.mask_mod is not None) else None),
                )
        else:
            softmax = None
            mma_params = SimpleNamespace(
                thr_mma_pv=thr_mma_pv,
                tOrV=tOrV,
                acc_O=acc_O,
                tOrP=tOrP,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_V=smem_thr_copy_V,
                tOsV=tOsV,
                smem_thr_copy_P=smem_thr_copy_P,
                tPsP=tPsP,
                sRowScale=sRowScale,
                sLSE=sLSE,
            )
            mask_fn = None
        if const_expr(split_pv_warps and not self._uses_n_distributed_qk()):
            compute_one_n_block = (
                self.compute_one_n_block_split_pv_owner
                if const_expr(is_qk_owner)
                else self.compute_one_n_block_split_pv_helper
            )
        elif const_expr(not self._uses_n_distributed_qk()):
            compute_one_n_block = self.compute_one_n_block
            if const_expr(self._fold_generic()):
                loader = self._fold_loader(fold_kv, sK, sV, seqlen, head_idx, batch_idx, tidx)
                compute_one_n_block = partial(
                    self.compute_one_n_block, loader=loader, n_block_min=n_block_min
                )
        if const_expr(not split_pv_warps or is_qk_owner):
            mask_fn_seqlen = partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True)
            mask_fn_no_seqlen = partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False)
            mask_fn_band = self._band_mask_fn(mask_fn)
        else:
            mask_fn_seqlen = None
            mask_fn_no_seqlen = None
            mask_fn_band = None
        n_block = cutlass.max(n_block_max - 1, 0)
        if const_expr(self._qk_pipelined() and sparse is not None):
            consumer_state = self._run_sparse_qk_pipelined(
                sparse,
                self._sparse_masks(mask_fn),
                consumer_state,
                mma_params,
                smem_copy_params,
                softmax,
                pipeline_k,
                pipeline_v,
                seqlen,
                m_block,
                batch_idx,
                head_idx,
                aux_data,
                fastdiv_mods,
                self._fold_loader(fold_kv, sK, sV, seqlen, head_idx, batch_idx, tidx),
            )
        elif const_expr(self._qk_pipelined()):
            consumer_state = self._run_n_blocks_qk_pipelined(
                n_block,
                n_block_min,
                n_block_max,
                consumer_state,
                mma_params,
                smem_copy_params,
                softmax,
                pipeline_k,
                pipeline_v,
                block_info,
                seqlen,
                m_block,
                batch_idx,
                head_idx,
                aux_data,
                fastdiv_mods,
                mask_fn_seqlen,
                mask_fn_no_seqlen,
                self._fold_loader(fold_kv, sK, sV, seqlen, head_idx, batch_idx, tidx),
            )
        elif const_expr(self._uses_n_distributed_qk()):
            consumer_state = self.compute_one_n_block_split_pv_distributed_qk(
                n_block,
                consumer_state,
                mma_params,
                smem_copy_params,
                None,
                pipeline_k,
                pipeline_v,
                score_mod=self.score_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                mask_fn=mask_fn_seqlen,
                is_first_n_block=True,
                is_last_n_block=n_block == n_block_min,
                learnable_sink=learnable_sink,
                split_idx=split_idx,
            )
            for n_tile in cutlass.range(n_block - n_block_min, unroll=1):
                consumer_state = self.compute_one_n_block_split_pv_distributed_qk(
                    n_block - n_tile - 1,
                    consumer_state,
                    mma_params,
                    smem_copy_params,
                    None,
                    pipeline_k,
                    pipeline_v,
                    score_mod=self.score_mod,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    seqlen=seqlen,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    mask_fn=mask_fn_no_seqlen,
                    is_last_n_block=n_block - n_tile - 1 == n_block_min,
                    learnable_sink=learnable_sink,
                    split_idx=split_idx,
                )
        elif const_expr(sparse is not None):
            masks = self._sparse_masks(mask_fn)
            if sparse[5] > 0:
                n_block, is_full, list_head = self._sparse_kv_block(sparse, 0)
                if const_expr(self._fold_generic()):
                    self._fold_issue(pipeline_k, loader.copy_K, n_block, consumer_state)
                    self._fold_issue(pipeline_v, loader.copy_V, n_block, consumer_state)
                consumer_state = compute_one_n_block(
                    n_block,
                    consumer_state,
                    mma_params,
                    smem_copy_params,
                    softmax,
                    pipeline_k,
                    pipeline_v,
                    score_mod=self.score_mod,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    seqlen=seqlen,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    mask_fn=partial(masks, is_full=is_full, list_head=list_head),
                    is_first_n_block=True,
                    **self._fold_next_kwargs(sparse, 1),
                )
                for it in cutlass.range(1, sparse[5], unroll=1):
                    n_block, is_full, list_head = self._sparse_kv_block(sparse, it)
                    consumer_state = compute_one_n_block(
                        n_block,
                        consumer_state,
                        mma_params,
                        smem_copy_params,
                        softmax,
                        pipeline_k,
                        pipeline_v,
                        score_mod=self.score_mod,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        m_block=m_block,
                        seqlen=seqlen,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                        mask_fn=partial(masks, is_full=is_full, list_head=list_head),
                        **self._fold_next_kwargs(sparse, it + 1),
                    )
        else:
            if const_expr(mask_fn_seqlen is not None):
                local_diag, local_left = self._local_edge_masks(
                    mask_fn, block_info, seqlen, m_block, n_block_min
                )
                if const_expr(local_diag is not None):
                    mask_fn_seqlen, mask_fn_no_seqlen = local_diag, local_left
            if const_expr(self._fold_generic()):
                self._fold_issue(pipeline_k, loader.copy_K, n_block, consumer_state)
                self._fold_issue(pipeline_v, loader.copy_V, n_block, consumer_state)
            consumer_state = compute_one_n_block(
                n_block,
                consumer_state,
                mma_params,
                smem_copy_params,
                softmax,
                pipeline_k,
                pipeline_v,
                score_mod=self.score_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                mask_fn=mask_fn_seqlen,
                is_first_n_block=True,
            )
            n_block_upper = n_block
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(
                    n_block_max - 1 - n_block_min_causal_local_mask, unroll=1
                ):
                    n_block = n_block_max - 2 - n_tile
                    consumer_state = compute_one_n_block(
                        n_block,
                        consumer_state,
                        mma_params,
                        smem_copy_params,
                        softmax,
                        pipeline_k,
                        pipeline_v,
                        score_mod=self.score_mod,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        m_block=m_block,
                        seqlen=seqlen,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                        mask_fn=mask_fn_seqlen,
                    )
                n_block_upper = cutlass.min(n_block_upper, n_block_min_causal_local_mask)
            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_upper - n_block_min_before_local_mask, unroll=1):
                consumer_state = compute_one_n_block(
                    n_block_upper - n_tile - 1,
                    consumer_state,
                    mma_params,
                    smem_copy_params,
                    softmax,
                    pipeline_k,
                    pipeline_v,
                    score_mod=self.score_mod,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    seqlen=seqlen,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    mask_fn=mask_fn_band,
                )
            if const_expr(self.is_local and block_info.window_size_left is not None):
                n_block_upper = cutlass.min(n_block_upper, n_block_min_before_local_mask)
                for n_tile in cutlass.range(n_block_upper - n_block_min, unroll=1):
                    consumer_state = compute_one_n_block(
                        n_block_upper - n_tile - 1,
                        consumer_state,
                        mma_params,
                        smem_copy_params,
                        softmax,
                        pipeline_k,
                        pipeline_v,
                        score_mod=self.score_mod,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        m_block=m_block,
                        seqlen=seqlen,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                        mask_fn=mask_fn_no_seqlen,
                    )

        if const_expr(self._uses_n_distributed_qk()):
            cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
            tOcO_mn_finalize = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
            num_rows_pv = acc_O.shape[0][0] * acc_O.shape[1]
            row_scale_pv = cute.make_rmem_tensor(num_rows_pv, Float32)
            lse = cute.make_rmem_tensor(num_rows_pv, Float32)
            for r in cutlass.range(cute.size(row_scale_pv), unroll_full=True):
                row = tOcO_mn_finalize[r, 0][0]
                row_scale_pv[r] = sRowScale[0, row]
                lse[r] = sLSE[row]
            self._rescale_O(acc_O, row_scale_pv)
        elif const_expr(split_pv_warps):
            if const_expr(is_qk_owner):
                sink_val = None
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                    else:
                        sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                        tScS_mn_finalize = layout_utils.reshape_acc_to_mn(
                            thr_mma_qk.partition_C(cS)
                        )
                        for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                            row = m_block * self.tile_m + tScS_mn_finalize[r][0]
                            q_head_idx = (
                                row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            )
                            sink_val[r] = Float32(learnable_sink[q_head_idx])
                if const_expr(self.is_split_kv and learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = sink_val if split_idx == 0 else -Float32.inf
                    elif split_idx != 0:
                        sink_val.fill(-Float32.inf)
                row_scale_qk = softmax.finalize(sink_val=sink_val)
                cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                tScS_mn_finalize = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                if tScS_mn_finalize[0][1] == 0:
                    for r in cutlass.range(cute.size(row_scale_qk), unroll_full=True):
                        row = tScS_mn_finalize[r][0]
                        sRowScale[2, row] = row_scale_qk[r]
                        sLSE[row] = softmax.row_sum[r]
                cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PFull),
                number_of_threads=self.num_mma_threads,
            )
            cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
            tOcO_mn_finalize = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
            num_rows_pv = acc_O.shape[0][0] * acc_O.shape[1]
            row_scale_pv = cute.make_rmem_tensor(num_rows_pv, Float32)
            lse = cute.make_rmem_tensor(num_rows_pv, Float32)
            for r in cutlass.range(cute.size(row_scale_pv), unroll_full=True):
                row = tOcO_mn_finalize[r, 0][0]
                row_scale_pv[r] = sRowScale[2, row]
                lse[r] = sLSE[row]
            self._rescale_O(acc_O, row_scale_pv)
        else:
            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn_finalize = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn_finalize[r][0]
                        q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
            if const_expr(self.is_split_kv and learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = sink_val if split_idx == 0 else -Float32.inf
                elif split_idx != 0:
                    sink_val.fill(-Float32.inf)
            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)
            lse = softmax.row_sum

        self.epilogue(
            acc_O,
            lse,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
            single_split_out=single_split_out,
        )
        return consumer_state

    @cute.jit
    def compute_one_n_block_split_pv_distributed_qk(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: None,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        score_mod: Callable | None,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        is_last_n_block: cutlass.Boolean = False,
        learnable_sink: Optional[cute.Tensor] = None,
        split_idx: Int32 = 0,
    ):
        """Run N-distributed QK with a shared online-softmax reduction.

        Four warps own disjoint K/V-column slices. Each warp publishes local
        row max/sum values; one lane per query row combines them with the
        running state and publishes the scale used for the vector P store.
        """
        p_stage = consumer_state.index
        num_qk_warps = const_expr(self.num_qk_threads // cute.arch.WARP_SIZE)
        local_sum_base = const_expr(num_qk_warps)
        global_max_row = const_expr(2 * num_qk_warps)
        global_sum_row = const_expr(global_max_row + 1)
        old_o_scale_row = const_expr(global_max_row + 2)
        warp_scale_base = const_expr(global_max_row + 3)

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        k_wait_token = pipeline_k.consumer_try_wait(consumer_state)
        pipeline_k.consumer_wait(consumer_state, k_wait_token)

        self._gemm_qk(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.smem_thr_copy_Q.partition_S(smem_copy_params.sQ),
            smem_copy_params.smem_thr_copy_K.partition_S(smem_copy_params.sK)[
                None, None, None, p_stage
            ],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
        )
        pipeline_k.consumer_release(consumer_state)

        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=smem_copy_params.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        num_rows = cute.size(acc_S_mn.shape[0])
        row_max_local = cute.make_rmem_tensor(num_rows, Float32)
        row_sum_local = cute.make_rmem_tensor(num_rows, Float32)
        for r in cutlass.range(num_rows, unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()
            row_max = utils.fmax_reduce(acc_S_row)
            row_max = cute.arch.warp_reduction_max(row_max, threads_in_group=4)
            row_max_safe = 0.0 if row_max == -Float32.inf else row_max
            acc_S_row_exp = cute.math.exp2(
                (acc_S_row - row_max_safe) * smem_copy_params.softmax_scale_log2,
                fastmath=True,
            )
            row_sum = utils.fadd_reduce(acc_S_row_exp)
            row_sum = utils.warp_reduce(row_sum, operator.add, width=4)
            row_max_local[r] = row_max
            row_sum_local[r] = row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(
            mma_params.thr_mma_qk.partition_C(cS),
        )
        row_coord = const_expr(0)
        col_coord = const_expr(1)
        warp_idx = mma_params.tidx // cute.arch.WARP_SIZE
        stat_writer_period = const_expr(8)
        if tScS_mn[0, 0][col_coord] % stat_writer_period == 0:
            for r in cutlass.range(num_rows, unroll_full=True):
                row = tScS_mn[r, 0][row_coord]
                smem_copy_params.sRowScale[warp_idx, row] = row_max_local[r]
                smem_copy_params.sRowScale[local_sum_base + warp_idx, row] = row_sum_local[r]
        cute.arch.fence_view_async_shared()

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )
        if mma_params.tidx < self.tile_m:
            row = mma_params.tidx
            row_max = smem_copy_params.sRowScale[0, row]
            for warp_idx_it in cutlass.range_constexpr(1, num_qk_warps):
                row_max = utils.fmax(
                    row_max,
                    smem_copy_params.sRowScale[warp_idx_it, row],
                )
            row_max_prev = (
                row_max
                if const_expr(is_first_n_block)
                else smem_copy_params.sRowScale[global_max_row, row]
            )
            row_max_new = (
                row_max if const_expr(is_first_n_block) else utils.fmax(row_max_prev, row_max)
            )
            row_max_new_safe = 0.0 if row_max_new == -Float32.inf else row_max_new
            old_o_scale = (
                1.0
                if const_expr(is_first_n_block)
                else cute.math.exp2(
                    (row_max_prev - row_max_new_safe) * smem_copy_params.softmax_scale_log2,
                    fastmath=True,
                )
            )
            row_sum_new = (
                0.0
                if const_expr(is_first_n_block)
                else smem_copy_params.sRowScale[global_sum_row, row] * old_o_scale
            )
            for warp_idx_it in cutlass.range_constexpr(num_qk_warps):
                warp_scale = cute.math.exp2(
                    (smem_copy_params.sRowScale[warp_idx_it, row] - row_max_new_safe)
                    * smem_copy_params.softmax_scale_log2,
                    fastmath=True,
                )
                smem_copy_params.sRowScale[warp_scale_base + warp_idx_it, row] = warp_scale
                row_sum_new += (
                    smem_copy_params.sRowScale[local_sum_base + warp_idx_it, row] * warp_scale
                )
            smem_copy_params.sRowScale[global_max_row, row] = row_max_new
            smem_copy_params.sRowScale[global_sum_row, row] = row_sum_new
            smem_copy_params.sRowScale[old_o_scale_row, row] = old_o_scale
            if is_last_n_block:
                row_max_final = row_max_new
                row_sum_final = row_sum_new
                if const_expr(learnable_sink is not None):
                    if split_idx == 0:
                        q_head_idx = (
                            row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            if const_expr(self.pack_gqa)
                            else head_idx
                        )
                        sink_val = Float32(learnable_sink[q_head_idx])
                        log2_e = math.log2(math.e)
                        if row_max_final == -Float32.inf:
                            row_max_final = sink_val * (
                                log2_e / smem_copy_params.softmax_scale_log2
                            )
                            row_sum_final = 1.0
                        else:
                            row_sum_final += cute.math.exp2(
                                sink_val * log2_e
                                - row_max_final * smem_copy_params.softmax_scale_log2,
                                fastmath=True,
                            )
                row_sum_is_zero_or_nan = row_sum_final == 0.0 or row_sum_final != row_sum_final
                smem_copy_params.sRowScale[0, row] = cute.arch.rcp_approx(
                    row_sum_final if not row_sum_is_zero_or_nan else 1.0
                )
                smem_copy_params.sLSE[row] = (
                    (
                        row_max_final * smem_copy_params.softmax_scale_log2
                        + cute.math.log2(row_sum_final, fastmath=True)
                    )
                    * math.log(2.0)
                    if not row_sum_is_zero_or_nan
                    else -Float32.inf
                )
        cute.arch.fence_view_async_shared()

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PEmpty),
            number_of_threads=self.num_mma_threads,
        )
        for r in cutlass.range(num_rows, unroll_full=True):
            row = tScS_mn[r, 0][row_coord]
            warp_scale = smem_copy_params.sRowScale[warp_scale_base + warp_idx, row]
            acc_S_mn[r, None].store(acc_S_mn[r, None].load() * warp_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP_qk = layout_utils.reshape_acc_to_frgA(rP)
        tPrP = smem_copy_params.smem_thr_store_P.retile(tOrP_qk)
        cute.copy(
            smem_copy_params.smem_thr_store_P,
            tPrP,
            smem_copy_params.tPsP_store[None, None, None, p_stage],
        )
        cute.arch.fence_view_async_shared()

        return self._compute_one_n_block_split_pv_common(
            consumer_state,
            mma_params,
            smem_copy_params,
            pipeline_k,
            pipeline_v,
            False,
            skip_p_empty=is_last_n_block,
        )

    @cute.jit
    def compute_one_n_block_split_pv_owner(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        score_mod: Callable | None,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Compute QK/softmax in four owner warps, then join eight PV warps."""
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        k_wait_token = pipeline_k.consumer_try_wait(consumer_state)
        pipeline_k.consumer_wait(consumer_state, k_wait_token)

        self._gemm_qk(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
        )

        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP_qk = layout_utils.reshape_acc_to_frgA(rP)
        tPrP = smem_copy_params.smem_thr_store_P.retile(tOrP_qk)
        cute.copy(
            smem_copy_params.smem_thr_store_P,
            tPrP,
            smem_copy_params.tPsP_store[None, None, None, consumer_state.index],
        )
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(mma_params.thr_mma_qk.partition_C(cS))
        self._publish_row_scale(
            row_scale,
            tScS_mn,
            smem_copy_params.sRowScale[consumer_state.index, None],
        )
        cute.arch.fence_view_async_shared()

        return self._compute_one_n_block_split_pv_common(
            consumer_state,
            mma_params,
            smem_copy_params,
            pipeline_k,
            pipeline_v,
            True,
        )

    @cute.jit
    def compute_one_n_block_split_pv_helper(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: None,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        score_mod: Callable | None,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Join the owner-published P tile using the four helper PV warps."""
        return self._compute_one_n_block_split_pv_common(
            consumer_state,
            mma_params,
            smem_copy_params,
            pipeline_k,
            pipeline_v,
            False,
        )

    @cute.jit
    def _compute_one_n_block_split_pv_common(
        self,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        release_k: cutlass.Constexpr[bool],
        skip_p_empty: cutlass.Boolean = False,
    ):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        tOcO_mn = layout_utils.reshape_acc_to_mn(mma_params.thr_mma_pv.partition_C(cO))
        num_rows_pv = mma_params.acc_O.shape[0][0] * mma_params.acc_O.shape[1]
        row_scale_pv = cute.make_rmem_tensor(num_rows_pv, Float32)
        for r in cutlass.range(cute.size(row_scale_pv), unroll_full=True):
            row = tOcO_mn[r, 0][0]
            row_scale_pv[r] = (
                smem_copy_params.sRowScale[
                    2 * (self.num_qk_threads // cute.arch.WARP_SIZE) + 2,
                    row,
                ]
                if const_expr(self._uses_n_distributed_qk())
                else smem_copy_params.sRowScale[consumer_state.index, row]
            )
        self._rescale_O(mma_params.acc_O, row_scale_pv)

        v_wait_token = pipeline_v.consumer_try_wait(consumer_state)
        pipeline_v.consumer_wait(consumer_state, v_wait_token)
        tOrP_copy_view = smem_copy_params.smem_thr_copy_P.retile(mma_params.tOrP)
        cute.copy(
            smem_copy_params.smem_thr_copy_P,
            smem_copy_params.tPsP[None, None, None, consumer_state.index],
            tOrP_copy_view,
        )
        self._gemm_pv(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            mma_params.tOrP,
            mma_params.tOrV,
            smem_copy_params.tOsV[None, None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_V,
        )
        pipeline_v.consumer_release(consumer_state)

        if not skip_p_empty:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PEmpty),
                number_of_threads=self.num_mma_threads,
            )
        if const_expr(release_k):
            pipeline_k.consumer_release(consumer_state)
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _publish_row_scale(
        self,
        row_scale: cute.Tensor,
        row_coords: cute.Tensor,
        sRowScale: cute.Tensor,
    ):
        """Have one QK lane group publish each row without escaping state."""
        if row_coords[0][1] == 0:
            for r in cutlass.range(cute.size(row_scale), unroll_full=True):
                sRowScale[row_coords[r][0]] = row_scale[r]

    @cute.jit
    def _rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor):
        """Apply a shared-published row scale without carrying softmax state."""
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        score_mod: Callable | None,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        loader: Optional[SimpleNamespace] = None,
        n_block_min: Optional[Int32] = None,
        n_next: Optional[Int32] = None,
    ):
        """One KV block. With fold_dma (loader), warp 0 loads the next block's K right after
        the QK release and its V after the PV release: n_next for block-sparse lists, else
        n_block - 1 down to n_block_min."""
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        k_wait_token = pipeline_k.consumer_try_wait(consumer_state)
        pipeline_k.consumer_wait(consumer_state, k_wait_token)
        self._gemm_qk_phase_local(
            mma_params.thr_mma_qk,
            acc_S,
            smem_copy_params.sQ,
            smem_copy_params.sK[None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
        )
        pipeline_k.consumer_release(consumer_state)
        if const_expr(loader is not None):
            next_state = consumer_state.clone()
            next_state.advance()
            self._fold_next(pipeline_k, loader.copy_K, next_state, n_block, n_block_min, n_next)

        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)

        v_wait_token = pipeline_v.consumer_try_wait(consumer_state)
        pipeline_v.consumer_wait(consumer_state, v_wait_token)
        self._gemm_pv_phase_local(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            smem_copy_params.sV[None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_V,
        )
        pipeline_v.consumer_release(consumer_state)
        consumer_state.advance()
        if const_expr(loader is not None):
            self._fold_next(pipeline_v, loader.copy_V, consumer_state, n_block, n_block_min, n_next)
        return consumer_state

    def _fold_next_kwargs(self, sparse, j):
        if const_expr(not self._fold_generic()):
            return {}
        return {"n_next": self._sparse_next_block(sparse, j)}

    @cute.jit
    def _fold_next(self, pipeline, copy, cons_state, n_block, n_block_min, n_next):
        if const_expr(n_next is None):
            if n_block > n_block_min:
                self._fold_issue(pipeline, copy, n_block - 1, cons_state)
        elif n_next >= 0:
            self._fold_issue(pipeline, copy, n_next, cons_state)

    @cute.jit
    def _sparse_qk(self, qk, sparse, masks, acc, it, k_state):
        n_block, is_full, list_head = self._sparse_kv_block(sparse, it)
        return qk(
            acc,
            n_block,
            k_state,
            n_next=self._sparse_next_block(sparse, it + 1),
            mask_fn=partial(masks, is_full=is_full, list_head=list_head),
        )

    @cute.jit
    def _sparse_sv(self, sv, sparse, acc, v_state, is_first: cutlass.Constexpr[bool], it):
        return sv(
            acc,
            v_state,
            is_first,
            self._sparse_kv_block(sparse, it)[0],
            n_next=self._sparse_next_block(sparse, it + 1),
        )

    @cute.jit
    def _run_sparse_qk_pipelined(
        self,
        sparse,
        masks: Callable,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        aux_data: AuxData,
        fastdiv_mods,
        loader: Optional[SimpleNamespace] = None,
    ):
        """_run_n_blocks_qk_pipelined over the block-sparse iterations: QK(it + 1) before the
        softmax and PV of iteration it; with fold_dma each K / V load is iteration it + 1's."""
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_A = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_B = cute.make_rmem_tensor(acc_shape_S, Float32)
        k_state = consumer_state.clone()
        v_state = consumer_state.clone()
        qk = partial(
            self._qk_into,
            use_seqlen_mask=False,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            pipeline_k=pipeline_k,
            softmax=softmax,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            seqlen=seqlen,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
            mask_fn_seqlen=None,
            mask_fn_no_seqlen=None,
            n_block_min=Int32(0),
            loader=loader,
        )
        sv = partial(
            self._sv_from,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            pipeline_v=pipeline_v,
            n_block_min=Int32(0),
            loader=loader,
        )

        num = sparse[5]
        if num > 0:
            if const_expr(loader is not None):
                n_first = self._sparse_kv_block(sparse, 0)[0]
                self._fold_issue(pipeline_k, loader.copy_K, n_first, k_state)
                self._fold_issue(pipeline_v, loader.copy_V, n_first, v_state)
            k_state = self._sparse_qk(qk, sparse, masks, acc_A, 0, k_state)
            if num == 1:
                v_state = self._sparse_sv(sv, sparse, acc_A, v_state, True, 0)
            else:
                k_state = self._sparse_qk(qk, sparse, masks, acc_B, 1, k_state)
                v_state = self._sparse_sv(sv, sparse, acc_A, v_state, True, 0)
                rem = num - 2
                for p in cutlass.range(rem // 2, unroll=1):
                    it = 2 + 2 * p
                    k_state = self._sparse_qk(qk, sparse, masks, acc_A, it, k_state)
                    v_state = self._sparse_sv(sv, sparse, acc_B, v_state, False, it - 1)
                    k_state = self._sparse_qk(qk, sparse, masks, acc_B, it + 1, k_state)
                    v_state = self._sparse_sv(sv, sparse, acc_A, v_state, False, it)
                if rem % 2 == 1:
                    k_state = self._sparse_qk(qk, sparse, masks, acc_A, num - 1, k_state)
                    v_state = self._sparse_sv(sv, sparse, acc_B, v_state, False, num - 2)
                    v_state = self._sparse_sv(sv, sparse, acc_A, v_state, False, num - 1)
                else:
                    v_state = self._sparse_sv(sv, sparse, acc_B, v_state, False, num - 1)
        return v_state

    @cute.jit
    def _run_n_blocks_qk_pipelined(
        self,
        n_first: Int32,
        n_block_min: Int32,
        n_block_max: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        aux_data: AuxData,
        fastdiv_mods,
        mask_fn_seqlen: Callable,
        mask_fn_no_seqlen: Callable,
        loader: Optional[SimpleNamespace] = None,
    ):
        """KV blocks n_first down to n_block_min with QK(j + 1) issued before the online
        softmax and PV of block j, alternating two S accumulators. Blocks from n_mask_min up
        (and the first) take the seqlen / causal / local mask, as in the segmented loop."""
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_A = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_B = cute.make_rmem_tensor(acc_shape_S, Float32)
        k_state = consumer_state.clone()
        v_state = consumer_state.clone()
        n_mask_min = n_block_max
        if const_expr(self.is_causal or self.is_local):
            n_mask_min = block_info.get_n_block_min_causal_local_mask(seqlen, m_block, n_block_min)
        qk = partial(
            self._qk_into,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            pipeline_k=pipeline_k,
            softmax=softmax,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            seqlen=seqlen,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
            mask_fn_seqlen=mask_fn_seqlen,
            mask_fn_no_seqlen=mask_fn_no_seqlen,
            n_block_min=n_block_min,
            loader=loader,
        )
        sv = partial(
            self._sv_from,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            pipeline_v=pipeline_v,
            n_block_min=n_block_min,
            loader=loader,
        )
        num = 1 + cutlass.max(n_first - n_block_min, 0)
        if const_expr(loader is not None):
            # fold_dma: the first K / V loads; later ones follow each stage's release.
            self._fold_issue(pipeline_k, loader.copy_K, n_first, k_state)
            self._fold_issue(pipeline_v, loader.copy_V, n_first, v_state)
        k_state = qk(acc_A, n_first, k_state, True)
        if num == 1:
            v_state = sv(acc_A, v_state, True, n_first)
        else:
            k_state = qk(acc_B, n_first - 1, k_state, n_first - 1 >= n_mask_min)
            v_state = sv(acc_A, v_state, True, n_first)
            rem = num - 2
            for p in cutlass.range(rem // 2, unroll=1):
                n = n_first - 2 - 2 * p
                k_state = qk(acc_A, n, k_state, n >= n_mask_min)
                v_state = sv(acc_B, v_state, False, n + 1)
                k_state = qk(acc_B, n - 1, k_state, n - 1 >= n_mask_min)
                v_state = sv(acc_A, v_state, False, n)
            if rem % 2 == 1:
                k_state = qk(acc_A, n_block_min, k_state, n_block_min >= n_mask_min)
                v_state = sv(acc_B, v_state, False, n_block_min + 1)
                v_state = sv(acc_A, v_state, False, n_block_min)
            else:
                v_state = sv(acc_B, v_state, False, n_block_min)
        return v_state

    @cute.jit
    def _qk_into(
        self,
        acc_S: cute.Tensor,
        n_block: Int32,
        k_state: PipelineState,
        use_seqlen_mask,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        pipeline_k: PipelineAsync,
        softmax: Softmax,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData,
        fastdiv_mods,
        mask_fn_seqlen: Callable,
        mask_fn_no_seqlen: Callable,
        n_block_min: Int32,
        loader: Optional[SimpleNamespace] = None,
        n_next: Optional[Int32] = None,
        mask_fn: Optional[Callable] = None,
    ):
        """acc_S = Q K(n_block)^T, score-modded and masked; returns the advanced K state.
        n_next (block sparsity): the KV tile fold_dma loads next, -1 for none; mask_fn
        overrides the seqlen / no-seqlen choice."""
        acc_S.fill(0.0)
        k_wait_token = pipeline_k.consumer_try_wait(k_state)
        pipeline_k.consumer_wait(k_state, k_wait_token)
        self._gemm_qk_phase_local(
            mma_params.thr_mma_qk,
            acc_S,
            smem_copy_params.sQ,
            smem_copy_params.sK[None, None, k_state.index],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
        )
        pipeline_k.consumer_release(k_state)
        k_state.advance()
        if const_expr(loader is not None):
            if const_expr(n_next is None):
                if n_block > n_block_min:
                    self._fold_issue(pipeline_k, loader.copy_K, n_block - 1, k_state)
            elif n_next >= 0:
                self._fold_issue(pipeline_k, loader.copy_K, n_next, k_state)
        if const_expr(self.score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        elif use_seqlen_mask:
            mask_fn_seqlen(acc_S, n_block=n_block)
        else:
            mask_fn_no_seqlen(acc_S, n_block=n_block)
        return k_state

    @cute.jit
    def _sv_from(
        self,
        acc_S: cute.Tensor,
        v_state: PipelineState,
        is_first: cutlass.Constexpr[bool],
        n_block: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        pipeline_v: PipelineAsync,
        n_block_min: Int32,
        loader: Optional[SimpleNamespace] = None,
        n_next: Optional[Int32] = None,
    ):
        """Online softmax of acc_S, O rescale and O += P V; returns the advanced V state."""
        row_scale = softmax.online_softmax(acc_S, is_first=is_first, check_inf=True)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        v_wait_token = pipeline_v.consumer_try_wait(v_state)
        pipeline_v.consumer_wait(v_state, v_wait_token)
        self._gemm_pv_phase_local(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            smem_copy_params.sV[None, None, v_state.index],
            smem_copy_params.smem_thr_copy_V,
        )
        pipeline_v.consumer_release(v_state)
        v_state.advance()
        if const_expr(loader is not None):
            if const_expr(n_next is None):
                if n_block > n_block_min:
                    self._fold_issue(pipeline_v, loader.copy_V, n_block - 1, v_state)
            elif n_next >= 0:
                self._fold_issue(pipeline_v, loader.copy_V, n_next, v_state)
        return v_state

    @cute.jit
    def _kv_iters(self, sparse, n_block_min, n_block_max):
        """KV iterations of a tile: its block range, or its share of the sparse lists."""
        if const_expr(sparse is None):
            return cutlass.max(n_block_max - n_block_min, 1)
        return sparse[5]

    @cute.jit
    def _kv_block_of(self, sparse, n_tile, n_block_min, n_block_max):
        """KV tile of iteration n_tile, from the top of the range or from the sparse lists."""
        if const_expr(sparse is None):
            return cutlass.max(n_block_max - 1 - n_tile, n_block_min)
        return self._sparse_kv_block(sparse, n_tile)[0]

    def _sparse_kv_list(
        self,
        blocksparse_tensors,
        batch_idx,
        head_idx,
        m_block,
        seqlen,
        split_idx=None,
        num_splits=None,
    ):
        """Block-sparse KV lists of Q tile m_block (None for dense grids): (partial list, full
        list, partial iterations, iterations, this split's first iteration, its iteration
        count). SplitKV splits the iterations into contiguous ceil-div ranges."""
        if const_expr(blocksparse_tensors is None):
            return None
        return self._sparse_kv_list_of(
            blocksparse_tensors, batch_idx, head_idx, m_block, seqlen, split_idx, num_splits
        )

    def _seqlen_sparse_kwargs(self, blocksparse_tensors):
        """Varlen block-sparse offsets for SeqlenInfoQK (the lists' per-batch M / index starts)."""
        if const_expr(blocksparse_tensors is None):
            return {}
        return dict(
            mCuTotalMBlocks=blocksparse_tensors.cu_total_m_blocks,
            mCuBlockIdxOffsets=blocksparse_tensors.cu_block_idx_offsets,
        )

    @cute.jit
    def _sparse_kv_list_of(
        self, blocksparse_tensors, batch_idx, head_idx, m_block, seqlen, split_idx, num_splits
    ):
        m_sparse = sparse_tensor_m_block(
            m_block,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            self.q_subtile_factor,
        )
        cnt, idx, full_cnt, full_idx = get_curr_blocksparse_tensors(
            batch_idx, head_idx, m_sparse, blocksparse_tensors, seqlen, self.kv_subtile_factor
        )
        n_tiles = cute.ceil_div(seqlen.seqlen_k, self.tile_n)
        n_partial = self._sparse_run_iters(cnt, idx, n_tiles)
        n_iter = n_partial
        if const_expr(full_idx is not None):
            n_iter = n_partial + self._sparse_run_iters(full_cnt, full_idx, n_tiles)
        it_begin, it_count = Int32(0), n_iter
        if const_expr(self.is_split_kv):
            per_split = cute.ceil_div(n_iter, num_splits)
            it_begin = cutlass.min(split_idx * per_split, n_iter)
            it_count = cutlass.min(per_split, n_iter - it_begin)
        return idx, full_idx, n_partial, n_iter, it_begin, it_count

    @cute.jit
    def _sparse_run_iters(self, cnt: Int32, idx: cute.Tensor, n_tiles: Int32) -> Int32:
        """KV tiles of a list: kv_subtile_factor per sparse block. The lists are ascending (as
        BlockMask builds them), so only the last block can reach past the n_tiles KV tiles."""
        f = self.kv_subtile_factor
        n = cnt
        if const_expr(f > 1):
            n = Int32(0)
            if cnt > 0:
                n = (cnt - 1) * f + cutlass.min(f, n_tiles - idx[cnt - 1] * f)
        return n

    @cute.jit
    def _sparse_kv_block(self, sparse, j):
        """(KV tile, is full block, first tile of its list) of this split's iteration j: the
        partial list, then the full list, each from its highest tile down. A list's first tile
        is the only one that can cross seqlen_k."""
        idx, full_idx, n_partial, n_iter, it_begin, _ = sparse
        it = it_begin + j
        f = self.kv_subtile_factor
        n_block, is_full, list_head = Int32(0), False, it == 0
        if const_expr(full_idx is None):
            j = n_partial - 1 - it
            n_block = idx[j // f] * f + j % f
        else:
            # Each list is read only within its own count (index tensors may be compact).
            if it < n_partial:
                j = n_partial - 1 - it
                n_block = idx[j // f] * f + j % f
            else:
                j = n_iter - 1 - it
                n_block = full_idx[j // f] * f + j % f
                is_full = True
                list_head = it == n_partial
        return n_block, is_full, list_head

    @cute.jit
    def _sparse_next_block(self, sparse, j):
        """KV tile of this split's iteration j, or -1 past its last one."""
        n_block = Int32(-1)
        if j < sparse[5]:
            n_block = self._sparse_kv_block(sparse, j)[0]
        return n_block

    @cute.jit
    def _sparse_mask(
        self,
        acc_S,
        n_block,
        is_full,
        list_head,
        mask_seqlen: Callable,
        mask_no_seqlen: Callable,
        mask_full_seqlen: Callable,
        mask_full: Callable,
    ):
        """Partial blocks take mask_mod, full blocks not; a list's first tile also the seqlen
        mask. Causal / local masks apply to both."""
        if is_full:
            if list_head:
                mask_full_seqlen(acc_S, n_block=n_block)
            else:
                mask_full(acc_S, n_block=n_block)
        else:
            if list_head:
                mask_seqlen(acc_S, n_block=n_block)
            else:
                mask_no_seqlen(acc_S, n_block=n_block)

    def _sparse_masks(self, mask_fn):
        """_sparse_mask with its four masks bound."""
        return partial(
            self._sparse_mask,
            mask_seqlen=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
            mask_no_seqlen=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
            mask_full_seqlen=partial(mask_fn, mask_mod=None, mask_seqlen=True),
            mask_full=partial(mask_fn, mask_mod=None, mask_seqlen=False),
        )

    @cute.jit
    def _load_tma_sparse(self, copy_K, copy_V, pipeline_k, pipeline_v, state_k, state_v, sparse):
        """The KV tiles of the sparse lists, in the consumer's order."""
        n_iter = sparse[5]
        for it in cutlass.range(n_iter, unroll=1):
            n_block = self._sparse_kv_block(sparse, it)[0]
            pipeline_k.producer_acquire(state_k)
            copy_K(
                src_idx=n_block,
                dst_idx=state_k.index,
                tma_bar_ptr=pipeline_k.producer_get_barrier(state_k),
            )
            pipeline_k.producer_commit(state_k)
            state_k.advance()
            pipeline_v.producer_acquire(state_v)
            copy_V(
                src_idx=n_block,
                dst_idx=state_v.index,
                tma_bar_ptr=pipeline_v.producer_get_barrier(state_v),
            )
            pipeline_v.producer_commit(state_v)
            state_v.advance()
        return state_k, state_v

    @cute.jit
    def _fold_issue(self, pipeline, copy, src_idx, cons_state):
        """fold_dma: warp 0 loads KV block src_idx into the stage cons_state consumes next
        (the producer's state for that block is the consumer's with the phase flipped)."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            state = PipelineState(
                cons_state.stages, cons_state.count, cons_state.index, cons_state.phase ^ 1
            )
            pipeline.producer_acquire(state)
            copy(
                src_idx=src_idx,
                dst_idx=state.index,
                tma_bar_ptr=pipeline.producer_get_barrier(state),
            )
            pipeline.producer_commit(state)

    def _fold_loader(self, fold_kv, sK, sV, seqlen, head_idx, batch_idx, tidx):
        """K / V TMA copies of this CTA's KV head, for fold_dma (None otherwise)."""
        if const_expr(fold_kv is None):
            return None
        mK, mV, tma_atom_K, tma_atom_V = fold_kv
        head_idx_kv = head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        if const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
        else:
            mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
            mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gV = cute.local_tile(mV_cur, (self.tile_hdimv, self.tile_n), (0, None))
        copy_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)
        copy_V, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V, 0, cute.make_layout(1), gV, sV)
        return SimpleNamespace(copy_K=copy_K, copy_V=copy_V)

    @cute.jit
    def _gemm_qk_a_in_regs(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrQ: cute.Tensor,
        sK: cute.Tensor,
        smem_thr_copy_K: cute.TiledCopy,
    ):
        """Issue QK with the full Q fragment resident across N blocks."""
        tCrK = tiled_mma.make_fragment_B(tiled_mma.partition_B(sK))
        tCsK = smem_thr_copy_K.partition_S(sK)
        tCrK_copy_view = smem_thr_copy_K.retile(tCrK)
        cute.copy(
            smem_thr_copy_K,
            tCsK[None, None, 0],
            tCrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tCsK.shape[2])):
            if k < cute.size(tCsK.shape[2]) - 1:
                cute.copy(
                    smem_thr_copy_K,
                    tCsK[None, None, k + 1],
                    tCrK_copy_view[None, None, k + 1],
                )
            cute.gemm(
                tiled_mma,
                acc,
                tCrQ[None, None, k],
                tCrK[None, None, k],
                acc,
            )

    @cute.jit
    def _gemm_qk_phase_local(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_K: cute.TiledCopy,
    ):
        """Allocate Q/K fragments only for the QK phase."""
        tCrQ = tiled_mma.make_fragment_A(tiled_mma.partition_A(sQ))
        tCrK = tiled_mma.make_fragment_B(tiled_mma.partition_B(sK))
        self._gemm_qk(
            tiled_mma,
            acc,
            tCrQ,
            tCrK,
            smem_thr_copy_Q.partition_S(sQ),
            smem_thr_copy_K.partition_S(sK),
            smem_thr_copy_Q,
            smem_thr_copy_K,
        )

    @cute.jit
    def _gemm_pv_phase_local(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrP: cute.Tensor,
        sV: cute.Tensor,
        smem_thr_copy_V: cute.TiledCopy,
    ):
        """Allocate the V fragment only for the PV phase."""
        tCrV = tiled_mma.make_fragment_B(tiled_mma.partition_B(sV))
        self._gemm_pv(
            tiled_mma,
            acc,
            tCrP,
            tCrV,
            smem_thr_copy_V.partition_S(sV),
            smem_thr_copy_V,
        )

    @cute.jit
    def _gemm_qk(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrQ: cute.Tensor,
        tCrK: cute.Tensor,
        tCsQ: cute.Tensor,
        tCsK: cute.Tensor,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_K: cute.TiledCopy,
    ):
        """Issue the SM120 QK warp-MMA mainloop."""
        tCrQ_copy_view = smem_thr_copy_Q.retile(tCrQ)
        tCrK_copy_view = smem_thr_copy_K.retile(tCrK)
        cute.copy(smem_thr_copy_Q, tCsQ[None, None, 0], tCrQ_copy_view[None, None, 0])
        cute.copy(smem_thr_copy_K, tCsK[None, None, 0], tCrK_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tCsQ.shape[2])):
            if k < cute.size(tCsQ.shape[2]) - 1:
                cute.copy(
                    smem_thr_copy_Q,
                    tCsQ[None, None, k + 1],
                    tCrQ_copy_view[None, None, k + 1],
                )
                cute.copy(
                    smem_thr_copy_K,
                    tCsK[None, None, k + 1],
                    tCrK_copy_view[None, None, k + 1],
                )
            cute.gemm(
                tiled_mma,
                acc,
                tCrQ[None, None, k],
                tCrK[None, None, k],
                acc,
            )

    @cute.jit
    def _gemm_pv(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrP: cute.Tensor,
        tCrV: cute.Tensor,
        tCsV: cute.Tensor,
        smem_thr_copy_V: cute.TiledCopy,
    ):
        """Issue the SM120 PV warp-MMA mainloop."""
        tCrV_copy_view = smem_thr_copy_V.retile(tCrV)
        for k in cutlass.range_constexpr(cute.size(tCrP.shape[2])):
            cute.copy(
                smem_thr_copy_V,
                tCsV[None, None, k],
                tCrV_copy_view[None, None, k],
            )
            cute.gemm(
                tiled_mma,
                acc,
                tCrP[None, None, k],
                tCrV[None, None, k],
                acc,
            )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)
        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
