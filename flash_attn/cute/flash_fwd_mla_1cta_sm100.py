# Copyright (c) 2026, Colfax International.
#
# 1CTA (cta_group::1) variant of the SM100 MLA absorbed forward kernel.
#
# Computes softmax(Q @ K^T + Qv @ V^T) @ V FA-style with mma tile M = 64, using
# weight-stationary tcgen05.mma.ws so the accumulators use the Layout E packed TMEM
# organization (a 64 x N accumulator stored as physical 128 lanes x N/2 columns) --
# the same per-CTA packing the 2CTA kernel (flash_fwd_mla_sm100.py) produces, which
# lets the softmax / correction / epilogue logic carry over nearly unchanged.
#
# Key structural differences from the 2CTA kernel:
#  - No cluster: every CTA owns a full 64-row m-tile and loads full K/V tiles.
#  - V latent is loaded ONCE per n_block: the P @ V GEMM reads the same sV bytes
#    through a byte-identical MN-major descriptor view (sVt). No Vt reloads.
#  - S is computed as: QvV(dv0)[zero_init] -> QK phase0 -> QvV(dv1) -> QK phase1.
#    The QK GEMM is split into two N=64 ws phases with an interleaved K token
#    gather (phase p holds tokens {p*32+[0,32)} u {64+p*32+[0,32)}) so the two
#    phase accumulators tile the plain N=128 Layout E S accumulator (phase p at
#    column offset 32*p). sK is a single (64, hdim) slot loaded twice per block
#    as 2x2 32-row TMA boxes (keeping the true seqlen extent for OOB zero-fill).
#    With `q_in_tmem` this collapses to QK[zero_init] -> QvV(dv0) -> QvV(dv1):
#    Q is staged into TMEM once per m-tile (tcgen05.cp, duplicated across lane
#    halves) and QK becomes ONE N=128 ws mma with its A operand in TMEM, reading a
#    unified (128, hdim) sK slot filled by a single TMA box per n_block. Faster on
#    every measured shape; see agent_space/NOTES_mla_1cta.md for the ablation and
#    for the two syncs that path depends on.
#  - Varlen: cu_seqlens / seqused on either side (independently). Q-side varlen uses
#    SingleTileVarlenScheduler; under cu_seqlens_q the O store falls back to the
#    per-row predicated path (a bulk TMA box would race on rows belonging to the next
#    sequence). See agent_space/NOTES_mla_1cta.md.
#  - Paged KV at any page size: TMA when page_size == tile_n (the page index is a
#    descriptor coordinate), otherwise a cp.async gather by a dedicated 4-warp load
#    warp group (warps 12-15) reusing PagedKVManager. Requires q_in_tmem when there is
#    a rope part.
#  - Not supported: topk/DSA gather, bitmask, P/rowmax emission, local.
#    pack_gqa is supported (any ratio up to 128).
#
# SMEM (226 KB): sQv 64K + sV 128K (one n_block) + sP 16K + stats, plus either
#   sQ 8K + sK 8K, or (q_in_tmem) a unified sK 16K whose first half doubles as the
#   Q staging buffer.
# TMEM (384/512, 416 with q_in_tmem): S 2 stages x 64 cols + O 2 dv-splits x 128
#   cols, Layout E packed, + 32 cols of duplicated Q.

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Int32, Boolean, const_expr
from cutlass.cute import FastDivmodDivisor
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils import ClcDynamicPersistentTileScheduler

from quack import copy_utils

from flash_attn.cute import utils as fa_utils
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors
import flash_attn.cute.blackwell_helpers as fa_sm100_utils
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.tile_scheduler import (
    ClcState,
    SchedulingMode,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
    ParamsBase,
)
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100_MLA2CTA


class FlashAttentionMLAForward1CtaSm100:
    # TMEM lane stride and datapath-half offset for hand-built Layout E layouts.
    # TMEM addresses: bits 16-31 = lane, bits 0-15 = column.
    TMEM_LANE_STRIDE = 1 << 16
    TMEM_HALF_STRIDE = 64 << 16

    def __init__(
        self,
        is_causal: bool = False,
        qhead_per_kvhead: int = 1,
        nheads_kv: int = 1,
        hdim: int = 64,
        hdimv: int = 512,
        use_clc_scheduler: bool = True,
        has_qk: bool = True,
        pack_gqa: bool = False,
        q_in_tmem: bool = False,
        has_seqused_q: bool = False,
        has_cu_seqlens_q: bool = False,
        use_cpasync_load_KV: bool = False,
        is_split_kv: bool = False,
        is_fp8: bool = False,
        num_stages_V: Optional[int] = None,
        num_stages_K: Optional[int] = None,
        num_stages_P: Optional[int] = None,
        _qk_issue_last: bool = False,
    ):
        self.is_causal = is_causal
        self.is_local = False
        self.qhead_per_kvhead = qhead_per_kvhead
        self.nheads_kv = nheads_kv
        self.has_qk = has_qk
        # q_in_tmem: stage the rope-part Q into TMEM (32 cols, duplicated across lane
        # halves by tcgen05.cp) and run QK as ONE N=128 weight-stationary TS mma against
        # a unified (128, hdim) sK slot -- instead of two N=64 SS phases against a
        # half-size sK reloaded with an interleaved token gather. The freed sQ (8 KB)
        # pays for the larger sK. Q's smem staging buffer aliases the first half of sK.
        self.q_in_tmem = q_in_tmem and has_qk
        # Paged KV with page_size != tile_n cannot be expressed as TMA boxes (a tile
        # spans several pages and its rows are non-contiguous), so it is gathered with
        # cp.async by a dedicated warp group. page_size == tile_n keeps pure TMA with
        # the page index as a descriptor coordinate.
        self.use_cpasync_load_KV = use_cpasync_load_KV
        self.use_tma_KV = not use_cpasync_load_KV
        # The cp.async gather fills one unified (tile_n, hdim) sK slot. The legacy
        # two-phase QK path instead loads K as interleaved 32-row token gathers, which
        # would need a second, quite different paged gather for a path that is slower on
        # every shape (kept only as a dense ablation baseline) -- so paged requires
        # q_in_tmem.
        assert self.q_in_tmem or not use_cpasync_load_KV or not has_qk, (
            "cp.async paged KV requires q_in_tmem=True (unified sK slot)"
        )
        # Q staging only exists (and only aliases sK) when there is a rope part staged
        # into TMEM; without it the cp.async group gathers V only and needs no handshake.
        self.cpasync_staging_sync = use_cpasync_load_KV and self.q_in_tmem
        # EXPERIMENT-ONLY knob (perf attribution): with q_in_tmem, issue the QK mma
        # AFTER the QvV splits instead of before, isolating the "QK overlaps the V TMA
        # tail" mechanism from the structural unified-sK/TS changes. Not for production.
        self._qk_issue_last = _qk_issue_last and self.q_in_tmem
        # pack_gqa folds qhead_per_kvhead into the row (m) dimension so one K/V tile
        # serves every q head of a kv group. Any ratio is supported: the packed row
        # index is m = h_in_group + qhead_per_kvhead * q_pos.
        # - Regular ratios (ratio | 64): the packed 64-row tile is a rectangle in
        #   (h_in_group, q_pos), so Q/Qv/O go through TMA on the packed layout directly
        #   (same recipe as the 2CTA MLA kernel). The packed gmem tensor
        #   ((ratio, s), d, h_kv, b) is exactly 5 flat dims -- at the TMA limit.
        # - Irregular ratios and ratio > 64 (splitting the ratio mode would need a 6th
        #   TMA dim): rows are gathered per-row instead (cp.async for Q/Qv, plain
        #   stores for O/LSE).
        self.pack_gqa = pack_gqa and qhead_per_kvhead > 1
        assert qhead_per_kvhead <= 128, "qhead_per_kvhead > 128 is not supported"
        # NB: 64 here is cta_tile_m, which is assigned further down in __init__.
        self.pack_gqa_tma = self.pack_gqa and 64 % qhead_per_kvhead == 0
        self.use_tma_QQv = not self.pack_gqa or self.pack_gqa_tma
        # O: a bulk TMA store writes whole 64-row boxes, which is unsafe under
        # cu_seqlens_q -- rows past this batch's seqlen_q belong to the NEXT sequence
        # and are written by another tile (write-write race), and the 64-row tile
        # straddles whenever (ratio *) seqlen_q % 64 != 0, i.e. routinely. Fall back to
        # the per-row predicated rmem->gmem store (the same path pack_gqa gather uses).
        # seqused_q keeps TMA: its layout is dense, so the rows past seqlen_q are that
        # batch's own padding and harmless to overwrite.
        # Under split-kv O_partial is fp32, which doubles sO past the sV slot it overlays;
        # the per-row predicated store needs no smem at all, and partial-O traffic
        # (64 x 512 x 4B per tile) is small next to the per-n_block KV stream.
        self.use_tma_O = (
            (not self.pack_gqa or self.pack_gqa_tma)
            and not has_cu_seqlens_q
            and not is_split_kv
        )

        # ==== split-kv info ====
        # Each (m_block, head, batch) tile is launched num_splits times; split s takes a
        # contiguous chunk of the tile's n_block range (BlockInfo does the arithmetic) and
        # writes a split-local softmax-normalized O_partial + LSE_partial, which the
        # separate combine kernel reduces. num_splits itself is a runtime value
        # (mO.shape[0]); only is_split_kv is compile-time.
        self.is_split_kv = is_split_kv

        # ==== varlen info ====
        # Q-side varlen changes the number of m-blocks per batch, so it needs the
        # varlen scheduler. K-side varlen only changes seqlen_k / the K,V base offsets
        # (both handled per-tile), so it keeps the dense schedulers.
        self.has_seqused_q = has_seqused_q
        self.has_cu_seqlens_q = has_cu_seqlens_q
        self.is_varlen_q = has_seqused_q or has_cu_seqlens_q

        # ==== tile scheduler ====
        self.is_persistent = False
        self.use_clc_scheduler = use_clc_scheduler
        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )
        if self.is_varlen_q:
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        # ==== thread info ====
        # With cp.async KV loads (paged KV, page_size != tile_n) a whole extra warp
        # group (warps 12-15) does the per-row gather: 128 threads = one page-table
        # entry and one token row each, per n_block. No relay warp is needed (unlike
        # the 2CTA kernel): with cluster size 1 the cp.async producers arrive directly
        # on the mma-visible mbarrier.
        self.num_softmax_threads = 128
        self.num_epilogue_threads = 128
        self.num_load_threads = 32
        self.num_mma_threads = 32
        self.num_empty_threads = 64
        self.num_cpasync_load_threads = 128 if self.use_cpasync_load_KV else 0
        self.num_threads = (
            self.num_softmax_threads
            + self.num_epilogue_threads
            + self.num_load_threads
            + self.num_mma_threads
            + self.num_empty_threads
            + self.num_cpasync_load_threads
        )
        self.num_warps = self.num_threads // 32
        assert self.num_warps == (16 if self.use_cpasync_load_KV else 12)
        self.softmax_warp_indices = (0, 1, 2, 3)
        self.epilogue_warp_indices = (4, 5, 6, 7)
        self.load_warp_id = 8
        self.mma_warp_id = 9
        self.clc_scheduler_warp_id = 10
        self.cpasync_load_warp_indices = (12, 13, 14, 15) if self.use_cpasync_load_KV else ()
        self.empty_warp_ids = tuple(
            w
            for w, active in [
                (11, True),
                (self.clc_scheduler_warp_id, not self.use_clc_scheduler),
            ]
            if active
        )

        # ==== register usage ====
        if const_expr(self.use_cpasync_load_KV):
            # 16 warps: 512 threads x 128 regs = the full 64K register file.
            self.num_regs_load = 112
            self.num_regs_mma = 112
            self.num_regs_softmax = 192
            self.num_regs_epilogue = 128
            self.num_regs_cpasync = 80
            self.num_regs_other = 48
            self.num_regs_per_thread = 128
            self.num_regs_total = 512
        else:
            self.num_regs_load = 168 - 40
            self.num_regs_mma = 168 - 40
            self.num_regs_softmax = 168 + 80
            self.num_regs_epilogue = 168 - 40
            self.num_regs_cpasync = 0
            self.num_regs_other = 48
            self.num_regs_per_thread = 168
            self.num_regs_total = 504
        assert (
            self.num_regs_mma
            + self.num_regs_softmax
            + self.num_regs_epilogue
            + self.num_regs_cpasync
            <= self.num_regs_total
        )

        # ==== 1cta info ====
        self.use_2cta_instrs = False
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (1, 1, 1)
        # A 64 x N Layout E accumulator packs its N columns into 2 datapath halves
        # (lanes 0-63 and 64-127). This is the same "2" as the per-CTA layout of the
        # 2CTA kernel, where it coincided with cta_group_size.
        self.num_acc_halves = 2

        # ==== problem shape info ====
        self.hdim = hdim
        self.hdimv = hdimv
        self.cta_tile_m = 64
        assert not self.pack_gqa_tma or self.cta_tile_m % self.qhead_per_kvhead == 0
        self.tile_n = 128
        self.num_hdimv_splits = 2  # split hdimv in half for our Qv @ V^T and P @ V mmas.
        assert hdimv % (2 * self.num_hdimv_splits) == 0
        self.epi_tile = (self.cta_tile_m, self.hdimv // self.num_hdimv_splits)
        self.tile_P = (self.cta_tile_m, self.tile_n)
        # q_in_tmem: QK is ONE N=128 ws mma over a unified (128, hdim) sK slot.
        # Otherwise QK runs as two N=64 ws phases with interleaved K token gather;
        # each phase loads 2 x 32 gathered rows into a single (64, hdim) sK slot.
        self.num_qk_phases = 1 if self.q_in_tmem else 2
        self.qk_phase_n = self.tile_n // self.num_qk_phases
        assert self.qk_phase_n == (128 if self.q_in_tmem else 64)

        # ==== MMA info ====
        self.mma_tiler_QK = (self.cta_tile_m, self.qk_phase_n, self.hdim)
        # helper tiler for the 32-row K TMA box loads (layout/TMA math only, no mma)
        self.mma_tiler_QK32 = (self.cta_tile_m, 32, self.hdim)
        self.mma_tiler_QvV = (self.cta_tile_m, self.tile_n, self.hdimv // self.num_hdimv_splits)
        self.mma_tiler_PVt = (self.cta_tile_m, self.hdimv // self.num_hdimv_splits, self.tile_n)
        self.major_mode_Q = tcgen05.OperandMajorMode.K
        self.major_mode_Qvi = tcgen05.OperandMajorMode.K
        self.major_mode_K = tcgen05.OperandMajorMode.K
        self.major_mode_Vi = tcgen05.OperandMajorMode.K
        self.major_mode_Vti = tcgen05.OperandMajorMode.MN
        self.major_mode_P = tcgen05.OperandMajorMode.K
        self.operand_source_A = tcgen05.OperandSource.SMEM
        # Only the QK mma can take its A operand (Q) from TMEM; Qv and P stay in SMEM
        # (Qv would need 256 TMEM columns -- see agent_space/NOTES_mla_1cta.md).
        self.operand_source_Q = (
            tcgen05.OperandSource.TMEM if self.q_in_tmem else tcgen05.OperandSource.SMEM
        )

        # ==== element width ====
        # Needed here rather than in __call__ (where the tensors arrive) because the TMEM
        # accounting below depends on it: the A-in-TMEM staging packs 32 // width
        # elements per 32-bit word. __call__ asserts the tensors match.
        self.is_fp8 = is_fp8
        self.dtype_ab_width = 8 if is_fp8 else 16
        # Note [Low Precision Scaling] (see flash_fwd_sm100.py): P in (0, 1] is cast to
        # the input dtype before P @ V, so for fp8 it is pre-scaled by 2^max_offset to
        # spend e4m3's unused upper code points on the probability tail. row_sum carries
        # the same factor, so O / row_sum cancels it -- only LSE must subtract it back
        # out. rescale_threshold (letting row_max go stale) must be 0 for fp8: it would
        # let P reach 2^(max_offset + threshold) and saturate.
        self.max_offset = 8 if is_fp8 else 0
        self.rescale_threshold = 0.0 if is_fp8 else 8.0
        # e4m3 max is 448 -> log2 = 8.807
        _log2_dtype_max = 8.807 if is_fp8 else 127.0
        assert self.max_offset + self.rescale_threshold < _log2_dtype_max, (
            f"max_offset {self.max_offset} + rescale_threshold {self.rescale_threshold} "
            f"exceeds the representable log2 range {_log2_dtype_max}"
        )

        # ==== pipeline info ====
        # fp8 halves every K/V/Q byte, freeing ~112 KB of the 227 KB SMEM budget. Measured
        # (agent_space/bench_mla_1cta_fp8.py --ablate): spending it on the V ring is worth
        # 17-29% on decode -- it lets the load warp run a whole n_block ahead instead of
        # stalling on single-block V residency (the known softmax-on-the-critical-path
        # limitation). Deeper K and P rings measured neutral or worse. V=6 does not fit
        # (32 KB/stage in fp8). Overridable per knob for ablation.
        self.num_stages_Q = 1
        # K stays single-slot even for fp8: the Q staging tile aliases the first half of
        # the unified sK slot, so a deeper K ring would need a separate staging buffer.
        # K is also only 8 KB in fp8 -- the headroom is worth far more on V.
        self.num_stages_K = 1 if num_stages_K is None else num_stages_K
        self.num_stages_Qv = 2  # the two dv splits of stationary Qv
        # the two dv splits of ONE n_block (bf16 default), or of two n_blocks (fp8)
        self.num_stages_V = (4 if is_fp8 else 2) if num_stages_V is None else num_stages_V
        self.num_stages_S = 2
        # P depth measured neutral-to-negative for fp8 on every shape (the P handoff was
        # never the bottleneck) -- see the ablation in agent_space/NOTES_mla_1cta.md.
        self.num_stages_P = 1 if num_stages_P is None else num_stages_P
        self.num_stages_Oi = 1
        self.num_stages_sm_stats = 2
        assert self.num_stages_S == 2, "mainloops expect 2 stages for S"
        assert self.num_stages_V % self.num_hdimv_splits == 0, (
            "V stages must be a whole number of n_blocks (num_hdimv_splits per block)"
        )
        assert not q_in_tmem or self.num_stages_K == 1, (
            "q_in_tmem aliases the Q staging tile onto the first half of the single sK "
            "slot; num_stages_K > 1 would need a separate Q staging buffer"
        )

        # ==== dtype info ====
        self.dtype_acc = Float32

        # ==== TMEM info ====
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        # Layout E packed: 64 x N occupies N // 2 columns of all 128 lanes.
        self.tmem_cols_S = self.tile_n // self.num_acc_halves
        self.tmem_cols_Oi = (self.hdimv // self.num_hdimv_splits) // self.num_acc_halves
        self.tmem_offset_S = [
            self.tmem_cols_S * stage for stage in range(self.num_stages_S)
        ]
        self.tmem_offset_O0 = self.tmem_cols_S * self.num_stages_S
        self.tmem_offset_O1 = self.tmem_offset_O0 + self.tmem_cols_Oi
        self.tmem_offsets_O = [self.tmem_offset_O0, self.tmem_offset_O1]
        self.total_tmem = self.tmem_offset_O1 + self.tmem_cols_Oi
        # Q operand region (q_in_tmem): a (64, hdim) 16-bit A operand for an M=64 ws mma
        # occupies hdim/2 columns, holding TWO copies of the tile (rows land at lanes m
        # and m + 64 -- the PTX Layout E A-operand organization, decoded empirically in
        # agent_space/ws_ts_mma_validation.py).
        self.tmem_offset_Q = self.total_tmem
        # 32 // width elements per 32-bit TMEM word (bf16: hdim/2 cols, fp8: hdim/4)
        self.tmem_cols_Q = (
            self.hdim * self.dtype_ab_width // 32 if self.q_in_tmem else 0
        )
        self.total_tmem += self.tmem_cols_Q
        assert self.total_tmem <= self.tmem_alloc_cols, (
            f"Total TMEM columns allocated {self.total_tmem} exceeds capacity {self.tmem_alloc_cols}"
        )

    def _packed_threads_per_row(self, width: int) -> int:
        """Threads cooperating on one row for the gathered (pack_gqa) Q/Qv loads.

        PackGQA.load_Q reads this back as `layout_tv_tiled.shape[0][0]`, which collapses
        to a plain int if the thread layout degenerates to a single row per pass. Keep at
        least 2 rows per pass so the TV layout stays 2-D. Halving is safe because widths
        and thread counts here are powers of two.
        """
        elems = 128 // self.dtype_Qv.width
        tpr = math.gcd(width // elems, self.num_load_threads)
        if tpr == self.num_load_threads:
            assert tpr % 2 == 0, f"cannot split {tpr} threads per row"
            tpr //= 2
        return tpr

    def _acc_tmem_layout(self, n: int) -> cute.Layout:
        """Layout E packed accumulator: logical (64, n) as 128 lanes x n/2 columns.

        Element (m, j + (n/2) * h) lives at lane m + 64*h, column j. Validated
        empirically against tcgen05.mma.ws in agent_space/ws_mma_validation.py.
        """
        n2 = n // self.num_acc_halves
        return cute.make_layout(
            (64, (n2, 2)),
            stride=(self.TMEM_LANE_STRIDE, (1, self.TMEM_HALF_STRIDE)),
        )

    def _acc_coord_view(self, n: int) -> cute.Tensor:
        """Identity tensor of logical (64, n) coordinates arranged to match
        _acc_tmem_layout's (64, (n/2, 2)) profile, for tmem-copy partitioning."""
        n2 = n // self.num_acc_halves
        cS = cute.make_identity_tensor((64, n))
        return cute.composition(
            cS, cute.make_layout((64, (n2, 2)), stride=(1, (64, 64 * n2)))
        )

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
                # q_in_tmem: Q's smem staging aliases the first (64, hdim) half of the
                # unified sK slot, so it needs no allocation of its own.
                (self.dtype_Q, self.sQ_layout_staged, not self.has_qk or self.q_in_tmem),
                (self.dtype_K, self.sK_layout_staged, not self.has_qk),
                (self.dtype_Qv, self.sQv_layout_staged, False),
                (self.dtype_V, self.sV_layout_staged, False),
                (self.dtype_P, self.sP_layout_staged, False),
            ]
        )
        sStats_struct = cute.struct.MemRange[Float32, cute.cosize(self.sStats_layout)]
        sScale_struct = cute.struct.MemRange[Float32, cute.cosize(self.sScale_layout)]

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
            ]
        )

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
            mbar_ptr_sm_stats: mbar_sm_stats_struct
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            # the CLC response is read with a 16-byte copy
            clc_response: cute.struct.Align[
                cute.struct.MemRange[Int32, clc_response_size], 16
            ]
            tmem_holding_buf: Int32
            sO_empty_mbar_ptr: cutlass.Int64
            # q_in_tmem: "Q staging consumed". The staging tile aliases the first half
            # of the unified sK slot, so the load warp must not issue this tile's K
            # loads until the mma warp's tcgen05.cp's have drained it. The mma warp
            # arrives with tcgen05.commit, which completes on cp completion.
            staging_mbar_ptr: cutlass.Int64
            # cp.async paged KV only: "this tile's K gathers have landed". The Q staging
            # tile aliases sK's first half, and there the K writer is a different warp
            # group than the Q writer, so the two directions need explicit ordering:
            # staging_mbar gates K-after-Q-consumed, this one gates Q-after-K-landed.
            sK_free_mbar_ptr: cutlass.Int64

            sRowMax: sStats_struct
            sRowSum: sStats_struct
            sScale: sScale_struct
            sQv: sQv_struct
            sQ: sQ_struct
            sK: sK_struct
            sV: sV_struct
            sP: sP_struct

        return SharedStorage

    # fmt: off
    @cute.jit
    def __call__(
        self,
        mQ: Optional[cute.Tensor],    # (b, s_q, h, d)   or (total_q, h, d)  if cu_seqlens_q
        mQv: cute.Tensor,             # (b, s_q, h, dv)  or (total_q, h, dv) if cu_seqlens_q
        mK: Optional[cute.Tensor],    # (b, s_k, h_k, d) or (total_k, h_k, d)  if cu_seqlens_k
        mV: cute.Tensor,              # (b, s_k, h_k, dv) or (total_k, h_k, dv) if cu_seqlens_k
        mO: cute.Tensor,              # (b, s_q, h, dv)  or (total_q, h, dv) if cu_seqlens_q
        mLSE: Optional[cute.Tensor],  # (b, s_q, h)      or (total_q, h)     if cu_seqlens_q
        softmax_scale: Float32,
        # The following are accepted for interface compatibility with the 2CTA MLA
        # kernel; the ones the 1CTA kernel does not support are asserted None.
        mP: Optional[cute.Tensor] = None,
        mRowMax: Optional[cute.Tensor] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,  # (b + 1)
        mCuSeqlensK: Optional[cute.Tensor] = None,  # (b + 1)
        mSeqUsedQ: Optional[cute.Tensor] = None,    # (b)
        mSeqUsedK: Optional[cute.Tensor] = None,    # (b)
        mIndexTopk: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        # fp8 per-(batch, kv head) dequantization scales; see _effective_descales.
        descale_tensors: Optional[DescaleTensors] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # fmt: on
        for name, t in [
            ("mP", mP), ("mRowMax", mRowMax), ("mIndexTopk", mIndexTopk),
            ("window_size_left", window_size_left),
            ("window_size_right", window_size_right),
        ]:
            assert t is None, f"{name} is not supported by the 1CTA MLA kernel (v1)"
        if const_expr(mPageTable is not None):
            # Paged KV: mK/mV are (num_pages, page_size, h_k, d) -- the KV transpose
            # below maps that to (page_size, d, h_k, num_pages) with exactly the same
            # permutation as the dense (b, s_k, h_k, d) case, so only the meaning of the
            # last mode changes (page instead of batch).
            assert mCuSeqlensK is None, "paged KV is not supported with cu_seqlens_k"
        else:
            assert not self.use_cpasync_load_KV, (
                "use_cpasync_load_KV requires a page table"
            )
        # Q-side varlen is a compile-time ctor flag (it selects the scheduler); K-side
        # is inferred from the tensors here, as in the 2CTA kernel.
        assert (mCuSeqlensQ is not None) == self.has_cu_seqlens_q, (
            "mCuSeqlensQ presence must match the has_cu_seqlens_q ctor flag"
        )
        assert (mSeqUsedQ is not None) == self.has_seqused_q, (
            "mSeqUsedQ presence must match the has_seqused_q ctor flag"
        )
        if const_expr(self.has_qk):
            assert mQ is not None and mK is not None, "has_qk requires mQ and mK"
        else:
            assert mQ is None and mK is None, "not has_qk disallows mQ and mK"

        # ==== dtype info ====
        self.dtype_Q = mQ.element_type if self.has_qk else cutlass.BFloat16
        if const_expr(descale_tensors is not None):
            assert self.is_fp8, "descales are only meaningful for fp8 inputs"
        assert mQv.element_type.width == self.dtype_ab_width, (
            f"is_fp8={self.is_fp8} implies {self.dtype_ab_width}-bit inputs, "
            f"got {mQv.element_type}"
        )
        self.dtype_K = mK.element_type if self.has_qk else cutlass.BFloat16
        self.dtype_Qv = mQv.element_type
        self.dtype_V = mV.element_type
        self.dtype_P = mV.element_type
        self.dtype_O = mO.element_type

        # ==== Prepare Tensors ====
        new_stride = lambda mX: (
            *(cute.assume(s, divby=128 // mX.element_type.width) for s in mX.stride[:-1]),
            mX.stride[-1],
        )
        mQ, mQv, mK, mV, mO = [
            cute.make_tensor(mX.iterator, cute.make_layout(mX.shape, stride=new_stride(mX)))
            if mX is not None
            else None
            for mX in (mQ, mQv, mK, mV, mO)
        ]

        # (b, s, h, d) -> (s, d, h, b), or packed (total, h, d) -> (total, d, h)
        # (Q and K sides are independent; seqused_* keeps the dense 4-D layout.)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        # Split-kv: O/LSE arrive with num_splits as mode 0 (fp32 partials) and keep it as
        # the trailing mode after the transpose, so per-tile slicing just appends
        # [..., split_idx]. num_splits is dynamic; only is_split_kv is compile-time.
        if const_expr(self.is_split_kv):
            O_split_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            num_splits = mO.shape[0]
        else:
            O_split_transpose = None
            num_splits = Int32(1)
        mQ, mQv = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            if mX is not None
            else None
            for mX in (mQ, mQv)
        ]
        mO = cute.make_tensor(
            mO.iterator,
            cute.select(
                mO.layout,
                mode=QO_layout_transpose
                if const_expr(not self.is_split_kv)
                else O_split_transpose,
            ),
        )
        mK, mV = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=KV_layout_transpose))
            if mX is not None
            else None
            for mX in (mK, mV)
        ]
        # (b, s_q, h) -> (s_q, h, b), or packed (total_q, h) unchanged; split-kv adds the
        # trailing split mode: (splits, b, s, h) -> (s, h, b, splits)
        if const_expr(self.is_split_kv):
            LSE_layout_transpose = (
                [2, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 2, 0]
            )
        else:
            LSE_layout_transpose = [1, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 1]
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if mLSE is not None
            else None
        )

        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.pack_gqa):
            # (s_q, d, h, b) -> ((qhead_per_kvhead, s_q), d, h_kv, b); mode 0 becomes the
            # packed row index m = h_in_group + qhead_per_kvhead * q_pos.
            mQ, mQv, mO = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                if mX is not None
                else None
                for mX in (mQ, mQv, mO)
            ]
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)

        # ==== Prepare MMAs ====
        # (local_var, dtype_a, major_a, major_b, mma_tiler)
        # fmt: off
        # tiled_mma_Qst is the SMEM-A twin of tiled_mma_QK: with q_in_tmem the QK mma
        # takes Q from TMEM, but the Q smem staging layout / TMA atom / load-side
        # partition_A still need an SMEM-A mma (the A-side tiler is N-independent).
        _mma_specs = [
            ("tiled_mma_QK",   self.dtype_Q,  self.major_mode_Q,   self.major_mode_K,   self.mma_tiler_QK,   self.operand_source_Q),
            ("tiled_mma_Qst",  self.dtype_Q,  self.major_mode_Q,   self.major_mode_K,   self.mma_tiler_QK,   self.operand_source_A),
            ("tiled_mma_QK32", self.dtype_Q,  self.major_mode_Q,   self.major_mode_K,   self.mma_tiler_QK32, self.operand_source_A),
            ("tiled_mma_QvV",  self.dtype_Qv, self.major_mode_Qvi, self.major_mode_Vi,  self.mma_tiler_QvV,  self.operand_source_A),
            ("tiled_mma_PVt",  self.dtype_P,  self.major_mode_P,   self.major_mode_Vti, self.mma_tiler_PVt,  self.operand_source_A),
        ]
        tiled_mma_QK, tiled_mma_Qst, tiled_mma_QK32, tiled_mma_QvV, tiled_mma_PVt = (
            sm100_utils.make_trivial_tiled_mma(
                dtype_a, major_a, major_b, self.dtype_acc, self.cta_group, mma_tiler[:2],
                a_source,
            )
            for _, dtype_a, major_a, major_b, mma_tiler, a_source in _mma_specs
        )
        # fmt: on

        # ==== Prepare SMEM layouts ====
        # (attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages)
        # fmt: off
        _smem_layout_specs = [
            ("sQ_layout",   sm100_utils.make_smem_layout_a, tiled_mma_Qst,  self.mma_tiler_QK,   self.dtype_Q,  self.num_stages_Q),
            ("sK_layout",   sm100_utils.make_smem_layout_b, tiled_mma_QK,   self.mma_tiler_QK,   self.dtype_K,  self.num_stages_K),
            # 32-row halves of the sK slot, used as the TMA destination view.
            ("sK32_layout", sm100_utils.make_smem_layout_b, tiled_mma_QK32, self.mma_tiler_QK32, self.dtype_K,  2 * self.num_stages_K),
            ("sP_layout",   sm100_utils.make_smem_layout_a, tiled_mma_PVt,  self.mma_tiler_PVt,  self.dtype_P,  self.num_stages_P),
            ("sQv_layout",  sm100_utils.make_smem_layout_a, tiled_mma_QvV,  self.mma_tiler_QvV,  self.dtype_Qv, self.num_stages_Qv),
            ("sV_layout",   sm100_utils.make_smem_layout_b, tiled_mma_QvV,  self.mma_tiler_QvV,  self.dtype_V,  self.num_stages_V),
            # MN-major descriptor view over the same sV bytes for the P @ V gemm
            # (byte-identical to sV_layout; V is loaded only once).
            ("sVt_layout",  sm100_utils.make_smem_layout_b, tiled_mma_PVt,  self.mma_tiler_PVt,  self.dtype_V,  self.num_stages_V),
        ]
        for attr, make_fn, tiled_mma, mma_tiler, dtype, num_stages in _smem_layout_specs:
            ab_kwarg = "a_dtype" if make_fn is sm100_utils.make_smem_layout_a else "b_dtype"
            staged = make_fn(
                tiled_mma=tiled_mma,
                mma_tiler_mnk=mma_tiler,
                num_stages=num_stages,
                **{ab_kwarg: dtype},
            )
            setattr(self, f"{attr}_staged", staged)
            setattr(self, attr, cute.select(staged, mode=[0, 1, 2]))
        # fmt: on
        if const_expr(self.has_qk and not self.q_in_tmem):
            assert cute.cosize(self.sK32_layout_staged) == cute.cosize(self.sK_layout_staged), (
                "stacked (32, hdim) K tiles must tile the (64, hdim) sK slot byte-identically"
            )
        self.sQst_layout = None
        if const_expr(self.q_in_tmem):
            # Q's staging buffer aliases the first (64, hdim) half of the unified
            # (128, hdim) sK slot.
            assert cute.cosize(self.sQ_layout_staged) * 2 == cute.cosize(self.sK_layout_staged), (
                "unified sK must be exactly twice the Q staging tile"
            )
            # Flat (cta_tile_m, hdim) view of the staging tile: cp_smem_to_tmem_dup_64
            # builds a UMMA descriptor from it, which needs the plain 2-D canonical
            # profile rather than the mma A-fragment profile of sQ_layout. Derived from
            # sQ_layout itself (regroup the A-fragment modes to (m, k), then coalesce),
            # so it addresses exactly the bytes the Q load writes, by construction.
            sQ_outer = self.sQ_layout.outer
            self.sQst_layout = cute.make_composed_layout(
                self.sQ_layout.inner,
                0,
                cute.coalesce(
                    cute.make_layout(
                        (sQ_outer.shape[0][0], (sQ_outer.shape[0][1], sQ_outer.shape[2])),
                        stride=(
                            sQ_outer.stride[0][0],
                            (sQ_outer.stride[0][1], sQ_outer.stride[2]),
                        ),
                    ),
                    target_profile=(1, 1),
                ),
            )
        assert cute.cosize(self.sVt_layout_staged) == cute.cosize(self.sV_layout_staged), (
            "sVt view must cover exactly the sV bytes"
        )

        self.sStats_layout = cute.make_layout((self.cta_tile_m, self.num_acc_halves))
        self.sScale_layout = cute.make_layout((self.cta_tile_m, self.num_stages_sm_stats))

        # fmt: off
        for attr, dtype, layout in [
            ("tma_copy_bytes_Q",   self.dtype_Q,  self.sQ_layout if self.has_qk else None),
            ("tma_copy_bytes_K",   self.dtype_K,  self.sK_layout if self.has_qk else None),
            ("tma_copy_bytes_Qvi", self.dtype_Qv, self.sQv_layout),
            ("tma_copy_bytes_Vi",  self.dtype_V,  self.sV_layout),
        ]:
            setattr(self, attr, cute.size_in_bytes(dtype, layout) if layout is not None else 0)
        # fmt: on

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_QvV.thr_id.shape,)
        )
        cta_shape = cta_layout_vmnk.shape

        def make_tma(make_fn, mX, smem_layout, mma_tiler, tiled_mma):
            return make_fn(tma_load_op, mX, smem_layout, mma_tiler, tiled_mma, cta_shape)

        A, B = cute.nvgpu.make_tiled_tma_atom_A, cute.nvgpu.make_tiled_tma_atom_B

        # (atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma)
        # fmt: off
        # K: with a unified sK slot one plain (tile_n, hdim) box per n_block suffices.
        # Otherwise it loads via 32-row boxes (interleaved gather); either way the
        # descriptor keeps the true seqlen extent, so partial tail blocks zero-fill.
        _K_tma = (
            (self.sK_layout, self.mma_tiler_QK, tiled_mma_QK)
            if self.q_in_tmem
            else (self.sK32_layout, self.mma_tiler_QK32, tiled_mma_QK32)
        )
        _tma_specs = [
            ("tma_atom_Q",  "tma_tensor_Q",  A, mQ,  self.sQ_layout,   self.mma_tiler_QK,   tiled_mma_Qst),
            ("tma_atom_Qv", "tma_tensor_Qv", A, mQv, self.sQv_layout,  self.mma_tiler_QvV,  tiled_mma_QvV),
            ("tma_atom_K",  "tma_tensor_K",  B, mK,  *_K_tma),
            ("tma_atom_V",  "tma_tensor_V",  B, mV,  self.sV_layout,   self.mma_tiler_QvV,  tiled_mma_QvV),
        ]
        _tmas = {}
        for atom_name, tensor_name, make_fn, m, smem_layout, mma_tiler, tiled_mma in _tma_specs:
            # Q/Qv are gathered with cp.async in the pack_gqa path (no TMA descriptor);
            # K/V are gathered per row for paged KV with page_size != tile_n.
            skip = (atom_name in ("tma_atom_Q", "tma_atom_Qv") and not self.use_tma_QQv) or (
                atom_name in ("tma_atom_K", "tma_atom_V") and not self.use_tma_KV
            )
            _tmas[atom_name], _tmas[tensor_name] = (
                make_tma(make_fn, m, smem_layout, mma_tiler, tiled_mma)
                if const_expr(m is not None and not skip)
                else (None, None)
            )

        (tma_atom_Q,  tma_tensor_Q,
         tma_atom_Qv, tma_tensor_Qv,
         tma_atom_K,  tma_tensor_K,
         tma_atom_V,  tma_tensor_V) = _tmas.values()
        # fmt: on
        if const_expr(not self.use_tma_KV):
            # gathered K/V: PagedKVManager issues one cp.async per (row, k-chunk)
            tma_tensor_K, tma_tensor_V = mK, mV
        if const_expr(not self.use_tma_QQv):
            # gathered Q/Qv: PackGQA.load_Q issues one cp.async per (row, k-chunk)
            tma_tensor_Q, tma_tensor_Qv = mQ, mQv
            async_copy_elems = 128 // self.dtype_Qv.width
            self.gmem_tiled_copy_Qv = copy_utils.tiled_copy_2d(
                self.dtype_Qv,
                self._packed_threads_per_row(self.hdimv // self.num_hdimv_splits),
                self.num_load_threads,
                async_copy_elems,
                is_async=True,
            )
            self.gmem_tiled_copy_Q = (
                copy_utils.tiled_copy_2d(
                    self.dtype_Q,
                    self._packed_threads_per_row(self.hdim),
                    self.num_load_threads,
                    async_copy_elems,
                    is_async=True,
                )
                if const_expr(self.has_qk)
                else None
            )

        # ==== Set up Oi smem -> gmem tma store ====
        # sO overlays sV (both sO slots fit in sV's first dv-split slot, which is
        # free once PVt(dv0) completes; guarded by sO_empty_mbar across tiles).
        self.overlap_sO_sV = True
        num_stages_sO = self.num_hdimv_splits
        sO_layout = sm100_utils.make_smem_layout_epi(
            self.dtype_O, self.o_layout, self.epi_tile, num_stages_sO
        )
        assert cute.cosize(sO_layout) <= cute.cosize(self.sV_layout_staged)

        if const_expr(self.use_tma_O):
            tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
                tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
        else:
            # packed rows are not contiguous in gmem, and varlen-q rows past seqlen_q
            # must not be written at all: store rmem -> gmem per row instead.
            tma_atom_O, tma_tensor_O = None, mO

        # ==== Set up Oi tmem -> rmem -> smem copy ====
        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype_O,
            num_bits_per_copy=universal_copy_bits,
        )
        # 128 threads as (64 rows, 2 datapath halves); each thread handles half the
        # columns of one row of the (64, hdimv/2) epi tile.
        thread_layout_O_r2g = cute.make_layout((64, 2), stride=(1, 64))
        value_layout_O_r2g = cute.make_layout(
            (1, self.hdimv // self.num_hdimv_splits // self.num_acc_halves)
        )
        tiled_copy_O_r2g = cute.make_tiled_copy_tv(
            atom=atom_universal_copy,
            thr_layout=thread_layout_O_r2g,
            val_layout=value_layout_O_r2g,
        )

        # ==== Allocate shared memory ====
        SharedStorage = self._get_shared_storage_cls()

        # ==== Tile scheduler ====
        TileScheduler = self.TileScheduler
        # Under cu_seqlens_q, mQv is packed (total_q, dv, h) -- no batch mode -- and the
        # varlen scheduler derives per-batch m-block counts from cu_seqlens/seqused
        # (scaling by qhead_per_kvhead_packgqa), so num_block is unused there.
        # NB cute.size(mQv.shape[0]) counts PACKED rows under pack_gqa (mode 0 is the
        # (ratio, s) tuple), which is what the varlen scheduler's grid bound expects --
        # it scales per-batch seqlens by qhead_per_kvhead_packgqa the same way.
        num_batch_sched = (
            cute.size(mCuSeqlensQ.shape[0] - 1)
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQv.shape[3])
        )
        total_q_sched = cute.size(mQv.shape[0]) * (
            1 if const_expr(mCuSeqlensQ is not None) else cute.size(mQv.shape[3])
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQv.shape[0]), self.cta_tile_m),
            num_head=cute.size(mQv.shape[2]),
            num_batch=num_batch_sched,
            num_splits=num_splits,
            seqlen_k=cute.size(mV.shape[0])
            if const_expr(mPageTable is None)
            else cute.size(mV.shape[0]) * cute.size(mPageTable.shape[1]),
            headdim=self.hdim,
            headdim_v=self.hdimv,
            total_q=total_q_sched,
            tile_shape_mn=(self.cta_tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype_V.width // 8,
            is_persistent=self.is_persistent,
            lpt=False,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=True,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # ==== Named Barrier ====
        self.softmax_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Softmax),
            num_threads=self.num_softmax_threads,
        )
        self.epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100_MLA2CTA.Epilogue),
            num_threads=self.num_epilogue_threads,
        )
        # softmax -> correction
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
            tma_tensor_Q,
            tma_tensor_Qv,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            descale_tensors,
            tma_atom_Q,
            tma_atom_Qv,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tiled_copy_O_r2g,
            self.gmem_tiled_copy_Q if const_expr(not self.use_tma_QQv) else None,
            self.gmem_tiled_copy_Qv if const_expr(not self.use_tma_QQv) else None,
            self.sQ_layout_staged,
            self.sQst_layout,
            self.sK_layout_staged,
            self.sK32_layout_staged,
            self.sQv_layout_staged,
            self.sV_layout_staged,
            self.sVt_layout_staged,
            self.sP_layout_staged,
            self.sStats_layout,
            self.sScale_layout,
            sO_layout,
            tiled_mma_QK,
            tiled_mma_Qst,
            tiled_mma_QK32,
            tiled_mma_QvV,
            tiled_mma_PVt,
            softmax_scale_log2,
            num_splits,
            tile_sched_params,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=(self.num_threads, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: Optional[cute.Tensor],
        mQv: cute.Tensor,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        descale_tensors,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_Qv: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tiled_copy_O_r2g: cute.TiledCopy,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_Qv: Optional[cute.TiledCopy],
        sQ_layout_staged: Optional[cute.ComposedLayout],
        sQst_layout: Optional[cute.ComposedLayout],
        sK_layout_staged: Optional[cute.ComposedLayout],
        sK32_layout_staged: Optional[cute.ComposedLayout],
        sQv_layout_staged: cute.ComposedLayout,
        sV_layout_staged: cute.ComposedLayout,
        sVt_layout_staged: cute.ComposedLayout,
        sP_layout_staged: cute.ComposedLayout,
        sStats_layout: cute.Layout,
        sScale_layout: cute.Layout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_Qst: cute.TiledMma,
        tiled_mma_QK32: cute.TiledMma,
        tiled_mma_QvV: cute.TiledMma,
        tiled_mma_PVt: cute.TiledMma,
        softmax_scale_log2: Float32,
        num_splits: Int32,
        tile_sched_params: ParamsBase,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_QvV.thr_id.shape,)
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
        )

        # ==== Prefetch TMA descriptors ====
        if warp_idx == self.load_warp_id:
            if const_expr(self.has_qk):
                if const_expr(self.use_tma_KV):
                    cpasync.prefetch_descriptor(tma_atom_K)
                if const_expr(self.use_tma_QQv):
                    cpasync.prefetch_descriptor(tma_atom_Q)
            if const_expr(self.use_tma_QQv):
                cpasync.prefetch_descriptor(tma_atom_Qv)
            if const_expr(self.use_tma_KV):
                cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(self.use_tma_O):
                cpasync.prefetch_descriptor(tma_atom_O)

        # ==== Construct pipelines ====
        tma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_load_threads)
        mma_warp = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        sm_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_threads)
        epi_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epilogue_threads)
        cpasync_threads = (
            pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cpasync_load_threads)
            if const_expr(self.use_cpasync_load_KV)
            else None
        )

        TmaUmma = pipeline.PipelineTmaUmma
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
        # Q/Qv: TMA-driven mbarrier normally; in the gathered pack_gqa path the load warp
        # signals completion of its cp.async group instead (PipelineAsyncUmma).
        QQv_cls, QQv_producer = (
            (TmaUmma, tma_warp) if const_expr(self.use_tma_QQv) else (AsyncUmma, load_threads)
        )
        tx_Q  = self.tma_copy_bytes_Q   if const_expr(self.use_tma_QQv) else None
        tx_Qv = self.tma_copy_bytes_Qvi if const_expr(self.use_tma_QQv) else None
        # K/V: TMA normally (incl. paged with page_size == tile_n, where the page index is
        # just a descriptor coordinate). For paged with page_size != tile_n the cp.async
        # warp group gathers per row and each of its threads registers a non-incrementing
        # arrival that lands when its cp.async group completes -- no relay warp needed,
        # since with cluster size 1 that arrival is already mma-visible.
        KV_cls, KV_producer = (
            (TmaUmma, tma_warp) if const_expr(self.use_tma_KV) else (AsyncUmma, cpasync_threads)
        )
        tx_K  = self.tma_copy_bytes_K   if const_expr(self.use_tma_KV) else None
        tx_Vi = self.tma_copy_bytes_Vi  if const_expr(self.use_tma_KV) else None
        pipeline_Q = pipeline_K = None
        if const_expr(self.has_qk):
            pipeline_Q    = make_pipeline(QQv_cls,   storage.mbar_ptr_Q,        self.num_stages_Q,        QQv_producer, mma_warp,  tx_Q)
            pipeline_K    = make_pipeline(KV_cls,    storage.mbar_ptr_K,        self.num_stages_K,        KV_producer, mma_warp,   tx_K)
        pipeline_Qv       = make_pipeline(QQv_cls,   storage.mbar_ptr_Qv,       self.num_stages_Qv,       QQv_producer, mma_warp,  tx_Qv)
        pipeline_V        = make_pipeline(KV_cls,    storage.mbar_ptr_V,        self.num_stages_V,        KV_producer, mma_warp,   tx_Vi)
        pipeline_S        = make_pipeline(UmmaAsync, storage.mbar_ptr_S,        self.num_stages_S,        mma_warp,   sm_threads)
        pipeline_P        = make_pipeline(AsyncUmma, storage.mbar_ptr_P,        self.num_stages_P,        sm_threads, mma_warp)
        pipeline_O0       = make_pipeline(UmmaAsync, storage.mbar_ptr_O0,       self.num_stages_Oi,       mma_warp,   epi_threads)
        pipeline_O1       = make_pipeline(UmmaAsync, storage.mbar_ptr_O1,       self.num_stages_Oi,       mma_warp,   epi_threads)
        pipeline_sm_stats = make_pipeline(Async,     storage.mbar_ptr_sm_stats, self.num_stages_sm_stats, sm_threads, epi_threads)
        # fmt: on

        sO_empty_mbar_ptr = storage.sO_empty_mbar_ptr
        staging_mbar_ptr = storage.staging_mbar_ptr
        sK_free_mbar_ptr = storage.sK_free_mbar_ptr
        if warp_idx == 0:
            cute.arch.mbarrier_init(sO_empty_mbar_ptr, 1)
            if const_expr(self.q_in_tmem):
                cute.arch.mbarrier_init(staging_mbar_ptr, 1)
            if const_expr(self.cpasync_staging_sync):
                # every cp.async load thread arrives once per tile
                cute.arch.mbarrier_init(sK_free_mbar_ptr, self.num_cpasync_load_threads)

        pipeline.pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # ==== Get SMEM tensors ====
        # fmt: off
        sQ, sK, sK32, sQv, sV, sVt, sP = (
            store.get_tensor(layout.outer, swizzle=layout.inner)
            if const_expr(store._size > 0) else None
            for store, layout in [
                # q_in_tmem: Q's staging buffer is the first (64, hdim) half of sK
                (storage.sQ if not self.q_in_tmem else storage.sK, sQ_layout_staged),
                (storage.sK,  sK_layout_staged),
                (storage.sK,  sK32_layout_staged),  # 32-row TMA destination view of sK
                (storage.sQv, sQv_layout_staged),
                (storage.sV,  sV_layout_staged),
                (storage.sV,  sVt_layout_staged),   # sVt reuses sV storage (MN-major view)
                (storage.sP,  sP_layout_staged),
            ]
        )
        # fmt: on
        # q_in_tmem: flat 2-D view of the Q staging tile (cp source), same bytes as sQ
        sQst = (
            storage.sK.get_tensor(sQst_layout.outer, swizzle=sQst_layout.inner)
            if const_expr(self.q_in_tmem)
            else None
        )
        sRowMax = storage.sRowMax.get_tensor(sStats_layout)
        sRowSum = storage.sRowSum.get_tensor(sStats_layout)
        sScale = storage.sScale.get_tensor(sScale_layout)

        # sO overlays sV (first dv-split slot region)
        sO = cute.make_tensor(
            cute.recast_ptr(sV.iterator, sO_layout.inner, self.dtype_O), sO_layout.outer
        )

        # With pack_gqa the m dimension is the packed row index, so BlockInfo /
        # masking convert it back to a q position via `row // qhead_per_kvhead`.
        block_info = BlockInfo(
            self.cta_tile_m,
            self.tile_n,
            is_causal=self.is_causal,
            is_split_kv=self.is_split_kv,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            # packed mQv mode 0 is (qhead_per_kvhead, seqlen_q); seqlen_q is the inner size
            seqlen_q_static=mQv.shape[0] if const_expr(not self.pack_gqa) else mQv.shape[0][1],
            seqlen_k_static=mV.shape[0]
            if const_expr(mPageTable is None)
            else mV.shape[0] * mPageTable.shape[1],
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
            clc = ClcState.create(
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
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
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
                if warp_idx == self.empty_warp_ids[i] and warp_idx != self.clc_scheduler_warp_id:
                    if const_expr(self.num_regs_other < self.num_regs_per_thread):
                        cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    if const_expr(self.num_regs_other < self.num_regs_per_thread):
                        cute.arch.setmaxregister_decrease(self.num_regs_other)

        if warp_idx == self.load_warp_id:
            if const_expr(self.num_regs_load < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                mQ,
                mK,
                mQv,
                mV,
                mPageTable,
                sQ,
                sK,
                sK32,
                sQv,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Qv,
                tma_atom_V,
                pipeline_Q,
                pipeline_K,
                pipeline_Qv,
                pipeline_V,
                sO_empty_mbar_ptr,
                staging_mbar_ptr,
                sK_free_mbar_ptr,
                tiled_mma_QK,
                tiled_mma_Qst,
                tiled_mma_QK32,
                tiled_mma_QvV,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_Qv,
                block_info,
                SeqlenInfoCls,
                num_splits,
                tile_scheduler=tile_scheduler,
            )

        if const_expr(self.use_cpasync_load_KV):
            if warp_idx in self.cpasync_load_warp_indices:
                if const_expr(self.num_regs_cpasync < self.num_regs_per_thread):
                    cute.arch.setmaxregister_decrease(self.num_regs_cpasync)
                self.load_cpasync(
                    mK,
                    mV,
                    mPageTable,
                    sK,
                    sV,
                    pipeline_K,
                    pipeline_V,
                    staging_mbar_ptr,
                    sK_free_mbar_ptr,
                    block_info,
                    SeqlenInfoCls,
                    num_splits,
                    tile_scheduler=tile_scheduler,
                )

        if warp_idx == self.mma_warp_id:
            if const_expr(self.num_regs_mma < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # ==== Allocate TMEM ====
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            self.mma(
                sQ,
                sQst,
                sK,
                sQv,
                sV,
                sVt,
                sP,
                tmem_ptr,
                staging_mbar_ptr,
                tiled_mma_QK,
                tiled_mma_QvV,
                tiled_mma_PVt,
                pipeline_Q,
                pipeline_K,
                pipeline_Qv,
                pipeline_V,
                pipeline_S,
                pipeline_P,
                pipeline_O0,
                pipeline_O1,
                block_info,
                SeqlenInfoCls,
                num_splits,
                tile_scheduler=tile_scheduler,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx in self.softmax_warp_indices:
            if const_expr(self.num_regs_softmax > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            self.softmax_loop(
                softmax_scale_log2,
                mLSE,
                sRowMax,
                sRowSum,
                sScale,
                sP,
                tmem_ptr,
                pipeline_S,
                pipeline_P,
                pipeline_sm_stats,
                block_info,
                SeqlenInfoCls,
                num_splits,
                descale_tensors,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

        if warp_idx in self.epilogue_warp_indices:
            if const_expr(self.num_regs_epilogue < self.num_regs_per_thread):
                cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            elif const_expr(self.num_regs_epilogue > self.num_regs_per_thread):
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.dtype_acc)
            self.correction_loop(
                softmax_scale_log2,
                mO,
                mLSE,
                tma_atom_O,
                sRowMax,
                sRowSum,
                sScale,
                sO,
                tmem_ptr,
                pipeline_O0,
                pipeline_O1,
                pipeline_sm_stats,
                sO_empty_mbar_ptr,
                tiled_copy_O_r2g,
                block_info,
                SeqlenInfoCls,
                num_splits,
                descale_tensors,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def load(
        self,
        mQ: Optional[cute.Tensor],
        mK: Optional[cute.Tensor],
        mQv: cute.Tensor,
        mV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        sQ: Optional[cute.Tensor],
        sK: Optional[cute.Tensor],
        sK32: Optional[cute.Tensor],
        sQv: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_Qv: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_Q: Optional[pipeline.PipelineAsync],
        pipeline_K: Optional[pipeline.PipelineAsync],
        pipeline_Qv: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: cute.Pointer,
        staging_mbar_ptr: cute.Pointer,
        sK_free_mbar_ptr: cute.Pointer,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_Qst: cute.TiledMma,
        tiled_mma_QK32: cute.TiledMma,
        tiled_mma_QvV: cute.TiledMma,
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_Qv: Optional[cute.TiledCopy],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        num_splits: Int32,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== Load warp ====
        # Description: loads tiles of Q, Qv, K (gathered phases), V0, V1 via TMA
        # produces: Q, Qv, K, V
        # consumes: -

        thr_mma_QK = tiled_mma_QK.get_slice(0)
        thr_mma_Qst = tiled_mma_Qst.get_slice(0)
        thr_mma_QK32 = tiled_mma_QK32.get_slice(0)
        thr_mma_QvV = tiled_mma_QvV.get_slice(0)

        Producer = pipeline.PipelineUserType.Producer
        if const_expr(self.has_qk):
            producer_state_Q = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Q)
            producer_state_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
        producer_state_Qv = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Qv)
        producer_state_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        producer_phase_O = Int32(1)
        staging_phase = Int32(0)
        # phase 1 first: like sO_empty, the first wait must pass (nothing to protect yet)
        sK_free_phase = Int32(1)

        tidx = cute.arch.thread_idx()[0] % self.num_load_threads

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            # with pack_gqa the scheduler's head index already is the kv head
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )

            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx=split_idx, num_splits=num_splits
            )
            num_n_blocks = n_block_max - n_block_min
            # Split-kv: this split's slice of the n_block range can be empty (the
            # per-split count is rounded up, so empties are the trailing splits).
            # Every warp role skips the whole tile body together, which keeps all
            # pipeline phases and raw-mbar phase counters consistent; the epilogue
            # still writes LSE = -inf so the combine kernel can ignore this split.
            has_work = const_expr(not self.is_split_kv) or n_block_min < n_block_max
            if has_work:

                if const_expr(self.cpasync_staging_sync):
                    # Q staging aliases sK's first half and K is written by the cp.async warp
                    # group: wait until its previous-tile gathers landed before overwriting.
                    cute.arch.mbarrier_wait(sK_free_mbar_ptr, phase=sK_free_phase)
                    sK_free_phase ^= 1

                # ==== Partition GMEM tensors ====
                if const_expr(self.has_qk):
                    mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                    if const_expr(self.use_tma_QQv):
                        gQ = cute.local_tile(
                            mQ_cur, (self.mma_tiler_QK[0], self.mma_tiler_QK[2]), (m_block, 0)
                        )
                        tSgQ = thr_mma_Qst.partition_A(gQ)
                        tQsQ, tQgQ = cpasync.tma_partition(
                            atom=tma_atom_Q,
                            cta_coord=0,
                            cta_layout=cute.make_layout(1),
                            smem_tensor=cute.group_modes(sQ, 0, 3),
                            gmem_tensor=cute.group_modes(tSgQ, 0, 3),
                        )
                    # Paged KV keeps the page mode as a free TMA coordinate (the page table
                    # selects the batch), so no batch offset is applied here.
                    mK_cur = (
                        seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                        if const_expr(mPageTable is None)
                        else mK[None, None, head_idx_kv, None]
                    )
                    if const_expr(not self.use_tma_KV):
                        # K is gathered by the cp.async warp group (no descriptor here)
                        pass
                    elif const_expr(self.q_in_tmem):
                        # unified sK: one plain (tile_n, hdim) box per n_block (or per page)
                        gK = cute.local_tile(
                            mK_cur,
                            (self.mma_tiler_QK[1], self.mma_tiler_QK[2]),
                            (None, 0) if const_expr(mPageTable is None) else (0, 0, None),
                        )
                        tSgK = thr_mma_QK.partition_B(gK)
                        tKsK, tKgK = cpasync.tma_partition(
                            atom=tma_atom_K,
                            cta_coord=0,
                            cta_layout=cute.make_layout(1),
                            smem_tensor=cute.group_modes(sK, 0, 3),
                            gmem_tensor=cute.group_modes(tSgK, 0, 3),
                        )
                    else:
                        # K: 32-row tiles; token = n_block*128 + p*32 + r + 64*h
                        # -> 32-row block index = 4*n_block + 2*h + p
                        gK32 = cute.local_tile(mK_cur, (32, self.mma_tiler_QK32[2]), (None, 0))
                        tSgK = thr_mma_QK32.partition_B(gK32)
                        tKsK, tKgK = cpasync.tma_partition(
                            atom=tma_atom_K,
                            cta_coord=0,
                            cta_layout=cute.make_layout(1),
                            smem_tensor=cute.group_modes(sK32, 0, 3),
                            gmem_tensor=cute.group_modes(tSgK, 0, 3),
                        )

                mQv_cur = seqlen.offset_batch_Q(mQv, batch_idx, dim=3)[None, None, head_idx]
                if const_expr(self.use_tma_QQv):
                    gQv = cute.local_tile(
                        mQv_cur, (self.mma_tiler_QvV[0], self.mma_tiler_QvV[2]), (m_block, None)
                    )
                    tSgQv = thr_mma_QvV.partition_A(gQv)
                    tQvsQv, tQvgQv = cpasync.tma_partition(
                        atom=tma_atom_Qv,
                        cta_coord=0,
                        cta_layout=cute.make_layout(1),
                        smem_tensor=cute.group_modes(sQv, 0, 3),
                        gmem_tensor=cute.group_modes(tSgQv, 0, 3),
                    )

                if const_expr(self.use_tma_KV):
                    mV_cur = (
                        seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                        if const_expr(mPageTable is None)
                        else mV[None, None, head_idx_kv, None]
                    )
                if const_expr(not self.use_tma_KV):
                    # V is gathered by the cp.async warp group
                    load_V = None
                elif const_expr(mPageTable is None):
                    # (tile_n, hdimv//2, num_n_blocks, num_d_blocks=2)
                    gV = cute.local_tile(
                        mV_cur, (self.mma_tiler_QvV[1], self.mma_tiler_QvV[2]), (None, None)
                    )
                    # (tile_n, hdimv//2, num_d_blocks=2, num_n_blocks)
                    gV = cute.make_tensor(gV.iterator, cute.select(gV.layout, mode=[0, 1, 3, 2]))
                else:
                    # paged: one page per tile -> (tile_n, hdimv//2, num_d_blocks=2, num_pages),
                    # already in (split, page) order
                    gV = cute.local_tile(
                        mV_cur, (self.mma_tiler_QvV[1], self.mma_tiler_QvV[2]), (0, None, None)
                    )
                if const_expr(self.use_tma_KV):
                    tSgV = thr_mma_QvV.partition_B(gV)
                    tVsV, tVgV = cpasync.tma_partition(
                        atom=tma_atom_V,
                        cta_coord=0,
                        cta_layout=cute.make_layout(1),
                        smem_tensor=cute.group_modes(sV, 0, 3),
                        gmem_tensor=cute.group_modes(tSgV, 0, 3),
                    )
                    load_V = partial(self.load_inner, tma_atom_V, tVgV, tVsV, pipeline_V)

                # ==== Load stationary operands ====
                if const_expr(self.use_tma_QQv):
                    if const_expr(self.has_qk):
                        producer_state_Q = self.load_inner(
                            tma_atom_Q, tQgQ, tQsQ, pipeline_Q, producer_state_Q
                        )
                    load_Qv = partial(self.load_inner, tma_atom_Qv, tQvgQv, tQvsQv, pipeline_Qv)
                    for dv_split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_Qv = load_Qv(producer_state_Qv, block=dv_split)
                else:
                    # pack_gqa: gather the packed rows with cp.async (see load_QQv_packed)
                    if const_expr(self.has_qk):
                        producer_state_Q = self.load_QQv_packed(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            pipeline_Q,
                            producer_state_Q,
                            self.hdim,
                            tidx,
                            m_block,
                            seqlen.seqlen_q,
                        )
                    for dv_split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_Qv = self.load_QQv_packed(
                            cute.domain_offset(
                                (0, dv_split * self.hdimv // self.num_hdimv_splits), mQv_cur
                            ),
                            sQv,
                            gmem_tiled_copy_Qv,
                            pipeline_Qv,
                            producer_state_Qv,
                            self.hdimv // self.num_hdimv_splits,
                            tidx,
                            m_block,
                            seqlen.seqlen_q,
                        )

                if const_expr(self.use_tma_O):
                    # sO (previous tile's epilogue) overlays sV: wait for it to drain before
                    # overwriting V slots. Not needed when O is stored straight from rmem.
                    cute.arch.mbarrier_wait(sO_empty_mbar_ptr, phase=producer_phase_O)
                    producer_phase_O ^= 1

                # ==== Main loop (descending n_block) ====
                # Issue order matches mma consumption: V0, K-phase0, V1, K-phase1.
                #
                # A fully-masked m-tile (causal with seqlen_q > seqlen_k) has num_n_blocks == 0,
                # but the mma warp always issues one S step and one P @ V step. Load one dummy
                # block (index 0) in that case so the K/V pipeline counts stay balanced --
                # otherwise the mma warp waits forever. Masking zeroes the dummy block's
                # contribution, so the result (O = 0, LSE = -inf) is still correct.
                num_n_blocks_load = cutlass.max(num_n_blocks, 1)
                n_block_first = n_block_max - 1 if num_n_blocks > 0 else 0
                # Paged KV (page_size == tile_n): the page table turns the logical n_block
                # into the physical page that indexes the K/V descriptors.
                mPageTable_cur = (
                    mPageTable[batch_idx, None] if const_expr(mPageTable is not None) else None
                )
                if const_expr(not self.use_tma_KV):
                    # K/V are gathered by the cp.async warp group; nothing to do here.
                    pass
                elif const_expr(self.q_in_tmem):
                    # The Q staging tile aliases the first half of the sK slot: wait until
                    # the mma warp's cp's have drained it before overwriting it with K.
                    cute.arch.mbarrier_wait(staging_mbar_ptr, phase=staging_phase)
                    staging_phase ^= 1
                    # Unified sK: K first (it gates the first mma of the step and is 8x
                    # smaller than a V split), then the two V dv splits.
                    load_K = partial(self.load_inner, tma_atom_K, tKgK, tKsK, pipeline_K)
                    for i in cutlass.range(num_n_blocks_load, unroll=1):
                        n_block = n_block_first - i
                        block = self._get_block_idx(n_block, mPageTable_cur)
                        producer_state_K = load_K(producer_state_K, block=block)
                        producer_state_V = load_V(producer_state_V, block=block, split=0)
                        producer_state_V = load_V(producer_state_V, block=block, split=1)
                else:
                    # Legacy two-phase QK path. Paged K would need a second 32-row gather, so
                    # paged requires q_in_tmem when there is a rope part; with has_qk=False
                    # only V is loaded and the page indirection applies to it as usual.
                    assert not self.has_qk or mPageTable is None, (
                        "paged KV with a rope part requires q_in_tmem=True"
                    )
                    for i in cutlass.range(num_n_blocks_load, unroll=1):
                        n_block = n_block_first - i
                        block = self._get_block_idx(n_block, mPageTable_cur)
                        producer_state_V = load_V(producer_state_V, block=block, split=0)
                        if const_expr(self.has_qk):
                            producer_state_K = self.load_K_phase(
                                tma_atom_K, tKgK, tKsK, pipeline_K, producer_state_K, n_block, 0
                            )
                        producer_state_V = load_V(producer_state_V, block=block, split=1)
                        if const_expr(self.has_qk):
                            producer_state_K = self.load_K_phase(
                                tma_atom_K, tKgK, tKsK, pipeline_K, producer_state_K, n_block, 1
                            )

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(self.has_qk):
            pipeline_Q.producer_tail(producer_state_Q)
        pipeline_Qv.producer_tail(producer_state_Qv)
        if const_expr(self.use_tma_KV):
            # K/V producer tails belong to whoever produced them (see load_cpasync).
            if const_expr(self.has_qk):
                pipeline_K.producer_tail(producer_state_K)
            pipeline_V.producer_tail(producer_state_V)

    def _get_block_idx(self, n_block: Int32, mPageTable_cur: Optional[cute.Tensor]) -> Int32:
        """Physical KV block for a logical n_block: the page id when paged."""
        return n_block if const_expr(mPageTable_cur is None) else mPageTable_cur[n_block]

    @cute.jit
    def load_inner(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        load_pipeline: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        block: Optional[Int32] = None,
        split: Optional[Int32] = None,
    ):
        if const_expr(split is not None):
            tXgX = tXgX[(None, split, None)]
        if const_expr(block is not None):
            tXgX = tXgX[(None, block)]
        if const_expr(cute.rank(tXsX) != 1):
            assert cute.rank(tXsX) == 2, f"wrong rank for tXsX, got {cute.rank(tXsX)}"
            stage = producer_state.index
            tXsX = tXsX[(None, stage)]
        load_pipeline.producer_acquire(producer_state)
        tma_bar_ptr = load_pipeline.producer_get_barrier(producer_state)
        cute.copy(tma_atom, tXgX, tXsX, tma_bar_ptr=tma_bar_ptr)
        producer_state.advance()
        return producer_state

    @cute.jit
    def cpasync_load_KV(
        self,
        paged_kv_manager: PagedKVManager,
        pipeline_X: pipeline.PipelineAsyncUmma,
        sX: cute.Tensor,  # staged MMA B-operand smem tile
        head_dim: cutlass.Constexpr[int],
        n_block: Int32,
        producer_state: pipeline.PipelineState,
        d_offset: int = 0,
    ):
        """Gather one (tile_n, head_dim) tile of paged K or V from gmem into smem.

        Each of the `num_cpasync_load_threads` threads owns one token row's page pointer
        (computed by `load_page_table`, which the CALLER must have run for this n_block --
        calling it across a @cute.jit boundary breaks MLIR SSA verification because its
        rmem writes live in a dynamic loop). Rows are issued one at a time, broadcasting
        the row's gmem base pointer to the lanes that cover it.

        With cta_group == 1 there is no cluster-scope commit to relay: each producer
        thread's `arrive_cp_async_mbarrier` lands when its own cp.async group completes,
        and that arrival is directly visible to the consuming mma warp.
        """
        stage = producer_state.index
        pipeline_X.producer_acquire(producer_state)

        tPrXPtr = paged_kv_manager.compute_X_ptr("K", d_offset)

        # The swizzle rides on sX's iterator, so composing a flat (tile_n, head_dim)
        # layout over the staged MMA-B tile yields the byte-identical image TMA writes.
        sX_nd = cute.composition(
            sX[None, None, None, stage],
            cute.make_ordered_layout((self.tile_n, head_dim), order=(0, 1)),
        )
        cX = cute.make_identity_tensor((self.tile_n, head_dim))
        gmem_thr_copy = paged_kv_manager.gmem_thr_copy_KV
        tXsX = gmem_thr_copy.partition_D(sX_nd)
        tXcX = gmem_thr_copy.partition_S(cX)
        tXc0X = gmem_thr_copy.get_slice(0).partition_S(cX)

        # Rows past seqlen_k are predicated off (cp.async leaves them zero), which is
        # what makes a partial trailing page safe; n_block < 0 marks a dummy tile.
        seqlenk_row_limit = (
            paged_kv_manager.seqlen_k - n_block * self.tile_n - tXcX[0][0]
            if n_block >= 0
            else 0
        )
        gtpr = paged_kv_manager.gmem_threads_per_row
        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_valid = tXc0X[0, m, 0][0] < seqlenk_row_limit
            should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)
            x_ptr_i64 = fa_utils.shuffle_sync(tPrXPtr[m // gtpr], m % gtpr, width=gtpr)
            x_gmem_ptr = cute.make_ptr(
                paged_kv_manager.mK_paged.element_type,
                x_ptr_i64,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            mX_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_cur_copy = cute.tiled_divide(mX_cur, (paged_kv_manager.async_copy_elems,))
            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, k][1] // paged_kv_manager.async_copy_elems
                tXsX_k = tXsX[None, m, k]
                mX_cur_copy_ki = cute.make_tensor(
                    mX_cur_copy[None, ki].iterator, tXsX_k.layout
                )
                cute.copy(
                    paged_kv_manager.gmem_tiled_copy_KV,
                    mX_cur_copy_ki,
                    tXsX_k,
                    pred=should_load,
                )

        cute.arch.cp_async_commit_group()
        pipeline_X.sync_object_full.arrive_cp_async_mbarrier(stage)
        producer_state.advance()
        return producer_state

    @cute.jit
    def load_cpasync(
        self,
        mK: Optional[cute.Tensor],
        mV: cute.Tensor,
        mPageTable: cute.Tensor,
        sK: Optional[cute.Tensor],
        sV: cute.Tensor,
        pipeline_K: Optional[pipeline.PipelineAsyncUmma],
        pipeline_V: pipeline.PipelineAsyncUmma,
        staging_mbar_ptr: cute.Pointer,
        sK_free_mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        num_splits: Int32,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== cp.async load warp group (paged KV, page_size != tile_n) ====
        # produces: K, V ; consumes: -
        Producer = pipeline.PipelineUserType.Producer
        if const_expr(self.has_qk):
            producer_state_K = pipeline.make_pipeline_state(Producer, stages=self.num_stages_K)
        producer_state_V = pipeline.make_pipeline_state(Producer, stages=self.num_stages_V)
        staging_phase = Int32(0)
        tidx = cute.arch.thread_idx()[0] % self.num_cpasync_load_threads
        dv_split = const_expr(self.hdimv // self.num_hdimv_splits)
        page_size_divmod = FastDivmodDivisor(cute.size(mV.shape[0]))

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            )
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx=split_idx, num_splits=num_splits
            )
            num_n_blocks = n_block_max - n_block_min
            # Split-kv: this split's slice of the n_block range can be empty (the
            # per-split count is rounded up, so empties are the trailing splits).
            # Every warp role skips the whole tile body together, which keeps all
            # pipeline phases and raw-mbar phase counters consistent; the epilogue
            # still writes LSE = -inf so the combine kernel can ignore this split.
            has_work = const_expr(not self.is_split_kv) or n_block_min < n_block_max
            if has_work:

                make_manager = partial(
                    PagedKVManager.create,
                    mPageTable,
                    page_size_divmod=page_size_divmod,
                    bidb=batch_idx,
                    bidh=head_idx_kv,
                    thread_idx=tidx,
                    seqlen_k=seqlen.seqlen_k,
                    leftpad_k=0,
                    n_block_size=self.tile_n,
                    num_threads=self.num_cpasync_load_threads,
                )
                # Both managers use mode "K" (non-transposed): the 1CTA kernel reads P@V
                # through an smem VIEW of the same sV bytes, so there is no separate Vt tile
                # to gather (unlike the CTA-split 2CTA kernel).
                if const_expr(self.has_qk):
                    paged_K = make_manager(
                        mK_paged=mK,
                        mV_paged=mK,
                        head_dim_padded=self.hdim,
                        head_dim_v_padded=self.hdim,
                        dtype=self.dtype_K,
                    )
                paged_V = make_manager(
                    mK_paged=mV,
                    mV_paged=mV,
                    head_dim_padded=dv_split,
                    head_dim_v_padded=dv_split,
                    dtype=self.dtype_V,
                )

                # See the load warp: a fully-masked m-tile still needs one balanced K/V set.
                num_n_blocks_load = cutlass.max(num_n_blocks, 1)
                n_block_first = n_block_max - 1 if num_n_blocks > 0 else 0
                if const_expr(self.cpasync_staging_sync):
                    # The Q staging tile aliases the first half of sK; wait until the mma
                    # warp's tcgen05.cp's have consumed it before gathering K over it.
                    cute.arch.mbarrier_wait(staging_mbar_ptr, phase=staging_phase)
                    staging_phase ^= 1
                for i in cutlass.range(num_n_blocks_load, unroll=1):
                    n_block = n_block_first - i
                    if const_expr(self.has_qk):
                        paged_K.load_page_table(n_block)
                        producer_state_K = self.cpasync_load_KV(
                            paged_K, pipeline_K, sK, self.hdim, n_block, producer_state_K
                        )
                    paged_V.load_page_table(n_block)
                    for split in cutlass.range_constexpr(self.num_hdimv_splits):
                        producer_state_V = self.cpasync_load_KV(
                            paged_V,
                            pipeline_V,
                            sV,
                            dv_split,
                            n_block,
                            producer_state_V,
                            d_offset=split * dv_split,
                        )

                if const_expr(self.cpasync_staging_sync):
                    # All of this tile's K writes have landed -> the load warp may stage the
                    # next tile's Q over sK's first half. cp.async completion is per thread,
                    # so each waits for its own groups and arrives (init count = all of them).
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.mbarrier_arrive(sK_free_mbar_ptr)

            work_tile = tile_scheduler.advance_to_next_work()

        if const_expr(self.has_qk):
            pipeline_K.producer_tail(producer_state_K)
        pipeline_V.producer_tail(producer_state_V)

    @cute.jit
    def load_QQv_packed(
        self,
        mX: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), hdim) for one kv head / batch
        sX: cute.Tensor,  # staged MMA A-operand smem tile
        gmem_tiled_copy: cute.TiledCopy,
        pipeline_X: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        hdim: cutlass.Constexpr[int],
        tidx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
    ):
        """Gather one 64-row packed tile of Q (or of a Qv dv split) with cp.async.

        The packed rows of a tile are generally not a rectangle in (h_in_group, q_pos),
        so TMA cannot express them; PackGQA.load_Q computes a per-row gmem pointer
        (q_pos = row // qhead_per_kvhead) and issues one cp.async per row segment.
        """
        stage = producer_state.index
        pipeline_X.producer_acquire(producer_state)
        pack_gqa = PackGQA(self.cta_tile_m, hdim, False, self.qhead_per_kvhead)
        # Present the staged A-operand tile as plain (m, k), keeping its swizzle.
        sX_stage = sX[None, None, None, stage]
        sX_mk = cute.make_tensor(
            sX_stage.iterator,
            cute.make_layout(
                (sX_stage.shape[0][0], (sX_stage.shape[0][1], sX_stage.shape[2])),
                stride=(sX_stage.stride[0][0], (sX_stage.stride[0][1], sX_stage.stride[2])),
            ),
        )
        pack_gqa.load_Q(mX, sX_mk, gmem_tiled_copy, tidx, m_block, seqlen_q)
        cute.arch.cp_async_commit_group()
        pipeline_X.sync_object_full.arrive_cp_async_mbarrier(stage)
        producer_state.advance()
        return producer_state

    @cute.jit
    def load_K_phase(
        self,
        tma_atom_K: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        pipeline_K: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        n_block: Int32,
        qk_phase: cutlass.Constexpr[int],
    ):
        """Load one gathered QK phase: tokens {p*32 + [0,32)} u {64 + p*32 + [0,32)}
        of block n as two 32-row TMA boxes into the two halves of the sK slot."""
        pipeline_K.producer_acquire(producer_state)
        tma_bar_ptr = pipeline_K.producer_get_barrier(producer_state)
        for h in cutlass.range_constexpr(2):
            row32_idx = 4 * n_block + 2 * h + qk_phase
            cute.copy(
                tma_atom_K,
                tKgK[(None, row32_idx)],
                tKsK[(None, h)],
                tma_bar_ptr=tma_bar_ptr,
            )
        producer_state.advance()
        return producer_state

    @cute.jit
    def mma(
        self,
        sQ: Optional[cute.Tensor],
        sQst: Optional[cute.Tensor],
        sK: Optional[cute.Tensor],
        sQv: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sP: cute.Tensor,
        tmem_ptr: cute.Pointer,
        staging_mbar_ptr: cute.Pointer,
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
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        num_splits: Int32,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== mma warp ====
        # Description: computes S = Q @ K^T + Qv @ V^T (ws, two gathered QK phases)
        #              and Oi += P @ Vi (ws, via the sVt view of sV)
        # Produces: S, O
        # Consumes: Q, K, Qv, V, P

        pipelines_O = [pipeline_O0, pipeline_O1]

        # Operands
        if const_expr(self.has_qk):
            if const_expr(self.q_in_tmem):
                # TS A operand: gemm_ptx_partial only reads the element type and the
                # k-group offsets off this tensor (recast to 32-bit TMEM column units);
                # the base address is passed separately as tA_addr. One k-group is the
                # mma's K-instruction (16 bf16 / 32 fp8 elements) and recasts to the
                # same 8 TMEM columns either way -- the validated ws A stepping.
                kinstr = const_expr(256 // self.dtype_ab_width)
                tSrQ = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + self.tmem_offset_Q, dtype=self.dtype_Q),
                    cute.make_layout((1, 1, self.hdim // kinstr), stride=(0, 0, kinstr)),
                )
            else:
                tSrQ = tiled_mma_QK.make_fragment_A(sQ)
            tSrK = tiled_mma_QK.make_fragment_B(sK)
        tSrQv = tiled_mma_QvV.make_fragment_A(sQv)
        tSrV = tiled_mma_QvV.make_fragment_B(sV)
        tOrP = tiled_mma_PVt.make_fragment_A(sP)
        tOrVt = tiled_mma_PVt.make_fragment_B(sVt)

        # GEMM functions (weight-stationary; Layout E packed accumulators).
        # S stage s occupies TMEM cols [64*s, 64*s + 64); QK phase p writes its
        # N=64 phase accumulator at column offset 32*p within the stage.
        gemm_QvV = [
            partial(
                fa_sm100_utils.gemm_ws_ptx_partial,
                tiled_mma_QvV.op,
                self.tmem_offset_S[stage],
                cta_group=1,
            )
            for stage in range(self.num_stages_S)
        ]
        if const_expr(self.has_qk):
            # q_in_tmem: ONE N=128 TS mma per stage, covering the full Layout E S
            # region -- so QK owns the zero-init and QvV accumulates on top (unless the
            # _qk_issue_last experiment reorders it after QvV, which then zero-inits).
            qk_extra = (
                {
                    "zero_init": not self._qk_issue_last,
                    "tA_addr": Int32(tmem_ptr.toint() + self.tmem_offset_Q),
                }
                if const_expr(self.q_in_tmem)
                else {"zero_init": False}
            )
            gemm_QK = [
                [
                    partial(
                        fa_sm100_utils.gemm_ws_ptx_partial,
                        tiled_mma_QK.op,
                        self.tmem_offset_S[stage] + self.qk_phase_n // 2 * phase,
                        cta_group=1,
                        **qk_extra,
                    )
                    for phase in range(self.num_qk_phases)
                ]
                for stage in range(self.num_stages_S)
            ]
        gemm_PVt = [
            partial(
                fa_sm100_utils.gemm_ws_ptx_partial,
                tiled_mma_PVt.op,
                self.tmem_offsets_O[split],
                cta_group=1,
            )
            for split in range(self.num_hdimv_splits)
        ]

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        # Q/K states are created unconditionally (unused when has_qk=False) so they can
        # be threaded through the mma step helpers uniformly.
        consumer_state_Q = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Q)
        consumer_state_K = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_K)
        consumer_state_Qv = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_Qv)
        # V slots are consumed twice: QvV waits (no release), PVt releases.
        consumer_state_V_wait = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        consumer_state_V_release = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_V)
        producer_state_S = pipeline.make_pipeline_state(Producer, stages=self.num_stages_S)
        consumer_state_P = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_P)
        producer_state_O0 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)
        producer_state_O1 = pipeline.make_pipeline_state(Producer, stages=self.num_stages_Oi)

        # Bind the per-block S and PVt steps (see mma_S_step / mma_PVt_step).
        mma_S = partial(
            self.mma_S_step,
            gemm_QvV,
            gemm_QK if const_expr(self.has_qk) else None,
            pipeline_S,
            pipeline_V,
            pipeline_K,
            tSrQ if const_expr(self.has_qk) else None,
            sQ,
            tSrK if const_expr(self.has_qk) else None,
            sK,
            tSrQv,
            sQv,
            tSrV,
            sV,
        )
        mma_PVt = partial(
            self.mma_PVt_step,
            gemm_PVt,
            pipelines_O,
            pipeline_P,
            pipeline_V,
            tOrP,
            sP,
            tOrVt,
            sVt,
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        O_should_accumulate = Boolean(False)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx=split_idx, num_splits=num_splits
            )
            num_n_blocks = n_block_max - n_block_min
            # Split-kv: this split's slice of the n_block range can be empty (the
            # per-split count is rounded up, so empties are the trailing splits).
            # Every warp role skips the whole tile body together, which keeps all
            # pipeline phases and raw-mbar phase counters consistent; the epilogue
            # still writes LSE = -inf so the combine kernel can ignore this split.
            has_work = const_expr(not self.is_split_kv) or n_block_min < n_block_max
            if has_work:
                even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
                num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

                if const_expr(self.has_qk):
                    pipeline_Q.consumer_wait(consumer_state_Q)
                if const_expr(self.q_in_tmem):
                    # Stage Q into TMEM, duplicated across lane halves (the ws M=64 A-operand
                    # layout). No wait is needed between these cp's and the QK mma below:
                    # tcgen05.cp -> tcgen05.mma from the same thread is a PTX-ordered pair.
                    #
                    # The reverse (mma -> cp) is NOT ordered, but the previous tile's QK
                    # mmas are already known to have retired here: this tile's Q TMA had to
                    # wait on pipeline_Q's empty barrier, which the mma warp signals at the
                    # end of a tile with a tcgen05.commit-backed release.
                    fa_sm100_utils.cp_smem_to_tmem_dup_64(
                        sQst, Int32(tmem_ptr.toint() + self.tmem_offset_Q)
                    )
                    # Tell the load warp the staging tile (aliased with sK) is free. The
                    # commit completes when the cp's do.
                    with cute.arch.elect_one():
                        tcgen05.commit(staging_mbar_ptr, None, self.cta_group)

                consumer_wait_state_Qv = consumer_state_Qv.clone()
                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_Qv.consumer_wait(consumer_wait_state_Qv)
                    consumer_wait_state_Qv.advance()

                # ==== Prologue ====
                producer_state_S, consumer_state_V_wait, consumer_state_K = mma_S(
                    producer_state_S, consumer_state_V_wait, consumer_state_K, stage=0
                )

                # ==== Mainloop ====
                # Single-block V residency forces PVt(cur) before S(next): S(next)
                # needs V(next), whose slots are freed only by PVt(cur).
                for _ in cutlass.range(num_n_block_groups - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        (
                            producer_state_O0,
                            producer_state_O1,
                            consumer_state_P,
                            consumer_state_V_release,
                            O_should_accumulate,
                        ) = mma_PVt(
                            producer_state_O0,
                            producer_state_O1,
                            consumer_state_P,
                            consumer_state_V_release,
                            O_should_accumulate,
                        )
                        producer_state_S, consumer_state_V_wait, consumer_state_K = mma_S(
                            producer_state_S,
                            consumer_state_V_wait,
                            consumer_state_K,
                            stage=const_expr((stage + 1) % self.num_stages_S),
                        )

                # ==== Epilogue ====
                num_final_n_blocks = self.num_stages_S if even_n_blocks else self.num_stages_S - 1
                for stage in cutlass.range_constexpr(self.num_stages_S):
                    n_block = num_final_n_blocks - 1 - stage
                    if n_block >= 0:
                        (
                            producer_state_O0,
                            producer_state_O1,
                            consumer_state_P,
                            consumer_state_V_release,
                            O_should_accumulate,
                        ) = mma_PVt(
                            producer_state_O0,
                            producer_state_O1,
                            consumer_state_P,
                            consumer_state_V_release,
                            O_should_accumulate,
                        )
                        if const_expr(stage == 0):
                            if n_block > 0:
                                producer_state_S, consumer_state_V_wait, consumer_state_K = mma_S(
                                    producer_state_S,
                                    consumer_state_V_wait,
                                    consumer_state_K,
                                    stage=1,
                                )

                if const_expr(self.has_qk):
                    # Released at the END of the tile on purpose when q_in_tmem: the release
                    # is a tcgen05.commit, so it fires only once every mma of this tile has
                    # retired -- which is what keeps the next tile's cp's from overwriting
                    # the Q TMEM region this tile's QK mmas are still reading.
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                for _ in cutlass.range_constexpr(self.num_hdimv_splits):
                    pipeline_Qv.consumer_release(consumer_state_Qv)
                    consumer_state_Qv.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            O_should_accumulate = Boolean(False)

        pipeline_S.producer_tail(producer_state_S)
        pipeline_O0.producer_tail(producer_state_O0)
        pipeline_O1.producer_tail(producer_state_O1)

    @cute.jit
    def mma_S_step(
        self,
        gemm_QvV,
        gemm_QK,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        pipeline_K: Optional[pipeline.PipelineAsync],
        tSrQ: Optional[cute.Tensor],
        sQ: Optional[cute.Tensor],
        tSrK: Optional[cute.Tensor],
        sK: Optional[cute.Tensor],
        tSrQv: cute.Tensor,
        sQv: cute.Tensor,
        tSrV: cute.Tensor,
        sV: cute.Tensor,
        producer_state_S: pipeline.PipelineState,
        consumer_state_V_wait: pipeline.PipelineState,
        consumer_state_K: pipeline.PipelineState,
        stage: cutlass.Constexpr[Int32],
    ):
        """S(stage) = Q @ K^T + Qv @ V^T for the current n_block.

        Order with q_in_tmem: QK[zero_init] -> QvV(dv0) -> QvV(dv1), one N=128 TS mma
        for QK against the unified sK slot.

        Otherwise: QvV(dv0)[zero_init] -> QK phase0 -> QvV(dv1) -> QK phase1, where
        QvV(dv0) zero-inits the whole N=128 Layout E S region in one instruction (the
        N=64 QK phase accumulators each cover only half of it) and the K-phase1 TMA
        load hides under the long QvV(dv1) mma.

        Either way V slots are only waited on here -- they are released by the P @ V
        gemm (mma_PVt_step).
        """
        pipeline_S.producer_acquire(producer_state_S)

        def qk_ts_step():
            # Single N=128 TS mma. Issued FIRST in production: K (16 KB) lands well
            # before the two 64 KB V splits, so S accumulation starts during the V TMA
            # tail, and it owns the zero-init of the full S region.
            pipeline_K.consumer_wait(consumer_state_K)
            gemm_QK[stage][0](
                tCrA=tSrQ,
                tCrB=tSrK[None, None, None, consumer_state_K.index],
                sA=None,
                sB=sK[None, None, None, consumer_state_K.index],
            )
            pipeline_K.consumer_release(consumer_state_K)
            consumer_state_K.advance()

        if const_expr(self.q_in_tmem and not self._qk_issue_last):
            qk_ts_step()
        for split in cutlass.range_constexpr(self.num_hdimv_splits):
            v_stage = consumer_state_V_wait.index
            pipeline_V.consumer_wait(consumer_state_V_wait)
            gemm_QvV[stage](
                tCrA=tSrQv[None, None, None, split],
                tCrB=tSrV[None, None, None, v_stage],
                sA=sQv[None, None, None, split],
                sB=sV[None, None, None, v_stage],
                zero_init=split == 0 and (not self.q_in_tmem or self._qk_issue_last),
            )
            consumer_state_V_wait.advance()
            if const_expr(self.has_qk and not self.q_in_tmem):
                pipeline_K.consumer_wait(consumer_state_K)
                gemm_QK[stage][split](
                    tCrA=tSrQ[None, None, None, 0],
                    tCrB=tSrK[None, None, None, consumer_state_K.index],
                    sA=sQ[None, None, None, 0],
                    sB=sK[None, None, None, consumer_state_K.index],
                )
                pipeline_K.consumer_release(consumer_state_K)
                consumer_state_K.advance()
        if const_expr(self._qk_issue_last):
            qk_ts_step()
        pipeline_S.producer_commit(producer_state_S)
        producer_state_S.advance()
        return producer_state_S, consumer_state_V_wait, consumer_state_K

    @cute.jit
    def mma_PVt_step(
        self,
        gemm_PVt,
        pipelines_O,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_V: pipeline.PipelineAsync,
        tOrP: cute.Tensor,
        sP: cute.Tensor,
        tOrVt: cute.Tensor,
        sVt: cute.Tensor,
        producer_state_O0: pipeline.PipelineState,
        producer_state_O1: pipeline.PipelineState,
        consumer_state_P: pipeline.PipelineState,
        consumer_state_V_release: pipeline.PipelineState,
        O_should_accumulate: Boolean,
    ):
        """Oi += P @ Vi for the current n_block, reading V through the MN-major sVt
        descriptor view; releases the V slots (the release is a tcgen05 commit, so it
        fires only once the mma actually completes)."""
        pipeline_P.consumer_wait(consumer_state_P)
        producer_states_O = [producer_state_O0, producer_state_O1]
        for split in cutlass.range_constexpr(self.num_hdimv_splits):
            producer_state_Oi = producer_states_O[split]
            pipelines_O[split].producer_acquire(producer_state_Oi)
            gemm_PVt[split](
                tCrA=tOrP[None, None, None, consumer_state_P.index],
                tCrB=tOrVt[None, None, None, consumer_state_V_release.index],
                sA=sP[None, None, None, consumer_state_P.index],
                sB=sVt[None, None, None, consumer_state_V_release.index],
                zero_init=~O_should_accumulate,
            )
            pipelines_O[split].producer_commit(producer_state_Oi)
            producer_state_Oi.advance()
            producer_states_O[split] = producer_state_Oi
            pipeline_V.consumer_release(consumer_state_V_release)
            consumer_state_V_release.advance()
        pipeline_P.consumer_release(consumer_state_P)
        consumer_state_P.advance()
        return (
            producer_states_O[0],
            producer_states_O[1],
            consumer_state_P,
            consumer_state_V_release,
            Boolean(True),
        )

    @cute.jit
    def apply_mask(
        self,
        tSrS: cute.Tensor,
        tScS_t2r: cute.Tensor,
        seqlen,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
    ):
        """Seqlen / causal masking on the per-thread S fragment.

        Per Layout E each thread owns one (row, datapath-half): the row coordinate is
        constant across the fragment and only columns vary.
        """
        row_idx = tScS_t2r[0][0] + m_block * self.cta_tile_m
        if const_expr(self.pack_gqa):
            # row_idx is a packed row: all q heads of a group share one q position
            row_idx = row_idx // self.qhead_per_kvhead
        if const_expr(mask_causal):
            # right-aligned causal: q row i attends k col j iff j <= i + seqlen_k - seqlen_q
            col_limit = row_idx + 1 + seqlen.seqlen_k - seqlen.seqlen_q
            if const_expr(mask_seqlen):
                col_limit = cutlass.min(col_limit, seqlen.seqlen_k)
        else:
            col_limit = seqlen.seqlen_k
        for i in cutlass.range_constexpr(cute.size(tSrS)):
            col_idx = tScS_t2r[i][1] + n_block * self.tile_n
            # col_idx < 0 only happens for the dummy block of a fully-masked m-tile
            # (num_n_blocks == 0 => n_block == -1 here); mask it entirely so the tile
            # produces O = 0 / LSE = -inf rather than the dummy block's values.
            if col_idx >= col_limit or col_idx < 0:
                tSrS[i] = -Float32.inf

    def _kv_head_idx(self, head_idx: Int32) -> Int32:
        """KV head for descale indexing (FA3 semantics: descales are per (b, h_kv))."""
        return (
            head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        )

    @cute.jit
    def _effective_descales(self, descale_tensors, batch_idx: Int32, head_idx: Int32):
        """(qk_descale, v_descale) for this tile; (1.0, 1.0) when there are no descales.

        MLA has TWO Q operands sharing q_descale, and BOTH S contributions land in one
        accumulator: S = qd*kd*(Q@K^T) + qd*vd*(Qv@V^T). Folding a single scale into
        softmax is therefore only valid when kd == vd, which the interface enforces
        (natural for MLA: the rope-K and latent-V halves live in one quantized cache).
        We fold qd*vd; v_descale is additionally applied to O in the epilogue.

        All three members must be present when the struct is: `DescaleTensors`
        round-trips through MLIR via `__new_from_mlir_values__`, which pads the *tail*
        with None -- so a partially-filled struct (e.g. q and v set, k None) arrives
        positionally shifted (v lands in the k slot) and silently loses v_descale. The
        interface normalizes to all-three-or-none for this path.
        """
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(descale_tensors is not None):
            assert (
                descale_tensors.q_descale is not None
                and descale_tensors.k_descale is not None
                and descale_tensors.v_descale is not None
            ), (
                "the MLA kernel needs all three descales present (see docstring: "
                "partially-filled DescaleTensors shift positionally across MLIR)"
            )
            head_idx_kv = self._kv_head_idx(head_idx)
            v_descale = Float32(descale_tensors.v_descale[batch_idx, head_idx_kv])
            # k_descale == v_descale is enforced host-side, so folding either is the same
            qk_descale = Float32(descale_tensors.q_descale[batch_idx, head_idx_kv]) * v_descale
        return qk_descale, v_descale

    @cute.jit
    def softmax_loop(
        self,
        softmax_scale_log2: Float32,
        mLSE: Optional[cute.Tensor],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sP: cute.Tensor,
        tmem_ptr: cute.Pointer,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        num_splits: Int32,
        descale_tensors,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== softmax warpgroup ====
        # Description: computes softmax on S and writes the result to P
        # Produces: P, softmax stats
        # Consumes: S

        tidx = cute.arch.thread_idx()[0] % self.num_softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_softmax_threads // 32
        )

        # Hand-built Layout E packed S accumulators (one per stage).
        acc_layout_S = self._acc_tmem_layout(self.tile_n)
        tSAcc_staged = [
            cute.make_tensor(tmem_ptr + self.tmem_offset_S[stage], acc_layout_S)
            for stage in range(self.num_stages_S)
        ]
        tSAcc = tSAcc_staged[0]

        # S tmem -> rmem copy objects
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.dtype_acc,
        )
        tmem_load_tiled = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc)
        tmem_load_thr = tmem_load_tiled.get_slice(tidx)
        tStS_t2r_staged = [
            tmem_load_thr.partition_S(tSAcc_staged[stage]) for stage in range(self.num_stages_S)
        ]
        # logical (row, col) coordinates matching the packed layout
        tScS_t2r = tmem_load_thr.partition_D(self._acc_coord_view(self.tile_n))
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.dtype_acc)

        # P rmem -> smem copy objects
        universal_copy_bits = 128
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype_P,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_store_tiled = cute.make_tiled_copy_D(smem_store_atom, tmem_load_tiled)
        smem_store_thr = smem_store_tiled.get_slice(tidx)
        # P rmem -> smem copy operands
        sP_mnp_layout = cute.make_ordered_layout(
            self.tile_P + (self.num_stages_P,), order=(0, 1, 2)
        )
        sP_mnp = cute.composition(sP, sP_mnp_layout)
        sP_smem_view = smem_store_thr.partition_D(sP_mnp)

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        consumer_state_S = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_S)
        producer_state_P = pipeline.make_pipeline_state(Producer, stages=self.num_stages_P)
        producer_state_sm_stats = pipeline.make_pipeline_state(
            Producer, stages=self.num_stages_sm_stats
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            # fp8: fold q_descale * v_descale into the softmax scale (see
            # _effective_descales); log2 folding is valid because the descale multiplies
            # S, and softmax_scale_log2 == softmax_scale * log2(e).
            qk_descale, _ = self._effective_descales(descale_tensors, batch_idx, head_idx)
            softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx=split_idx, num_splits=num_splits
            )
            num_n_blocks = n_block_max - n_block_min
            # Split-kv: this split's slice of the n_block range can be empty (the
            # per-split count is rounded up, so empties are the trailing splits).
            # Every warp role skips the whole tile body together, which keeps all
            # pipeline phases and raw-mbar phase counters consistent; the epilogue
            # still writes LSE = -inf so the combine kernel can ignore this split.
            has_work = const_expr(not self.is_split_kv) or n_block_min < n_block_max
            if has_work:
                even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
                num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

                mask_fn = partial(
                    self.apply_mask,
                    tScS_t2r=tScS_t2r,
                    seqlen=seqlen,
                    m_block=m_block,
                    mask_causal=self.is_causal,
                )

                softmax = SoftmaxSm100.create(
                    softmax_scale_log2_eff,
                    rescale_threshold=self.rescale_threshold,
                    max_offset=self.max_offset,
                )
                softmax.reset()

                softmax_step_fn = partial(
                    self.softmax_step,
                    softmax,
                    sRowMax,
                    sScale,
                    tStS_t2r_staged,
                    tSrS_t2r,
                    sP_smem_view,
                    tmem_load_thr,
                    smem_store_thr,
                    pipeline_S,
                    pipeline_P,
                    pipeline_sm_stats,
                    tidx,
                    warp_idx,
                )

                ### first iteration ###
                n_block = n_block_max - 1
                (
                    consumer_state_S,
                    producer_state_P,
                    producer_state_sm_stats,
                ) = softmax_step_fn(
                    consumer_state_S,
                    producer_state_P,
                    producer_state_sm_stats,
                    0,
                    n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    is_first=True,
                )
                n_block -= 1

                ### Separate iterations with causal masking
                # note: For square-ish tiles, at most ceil(tile_n / tile_m) extra blocks
                # need causal masking beyond the first.
                if const_expr(self.is_causal):
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    # Clamp at 0: with split-kv this split's range can lie entirely below
                    # the causal-mask boundary, making the difference negative. Left
                    # unclamped it would SUBTRACT from num_n_block_groups below, so
                    # softmax would wait on S stages the mma warp never produces.
                    num_masked_n_blocks = cutlass.max(
                        n_block_max - 1 - n_block_min_causal_local_mask, 0
                    )
                    num_masked_n_block_groups = cutlass.max(
                        min(
                            num_n_block_groups - 1,
                            cute.ceil_div(num_masked_n_blocks, self.num_stages_S),
                        ),
                        0,
                    )
                    num_n_block_groups -= num_masked_n_block_groups
                    for _ in cutlass.range(num_masked_n_block_groups, unroll=1):
                        for stage in cutlass.range_constexpr(self.num_stages_S):
                            (
                                consumer_state_S,
                                producer_state_P,
                                producer_state_sm_stats,
                            ) = softmax_step_fn(
                                consumer_state_S,
                                producer_state_P,
                                producer_state_sm_stats,
                                1 - stage,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                            n_block -= 1

                ### Mainloop ###
                for _ in cutlass.range(num_n_block_groups - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        (
                            consumer_state_S,
                            producer_state_P,
                            producer_state_sm_stats,
                        ) = softmax_step_fn(
                            consumer_state_S,
                            producer_state_P,
                            producer_state_sm_stats,
                            1 - stage,
                            n_block,
                            mask_fn=None,
                        )
                        n_block -= 1

                ### last iteration if even ###
                # always mask to simplify logic
                if even_n_blocks:
                    (
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                    ) = softmax_step_fn(
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                        1,
                        n_block,
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                    )
                    n_block -= 1

                # write row max and sum to smem
                sRowSum[tidx % self.cta_tile_m, warp_idx // self.num_acc_halves] = softmax.row_sum[0]
                if const_expr(mLSE is not None):
                    if tidx < self.cta_tile_m:
                        sRowMax[tidx, 0] = softmax.row_max[0]
                self.sm_stats_barrier_full.arrive()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            if has_work:
                # Paired with the epilogue's arrive inside ITS has_work branch: an empty
                # split must skip both halves of this 256-thread barrier.
                self.sm_stats_barrier_empty.arrive_and_wait()

        pipeline_P.producer_tail(producer_state_P)
        pipeline_sm_stats.producer_tail(producer_state_sm_stats)

    @cute.jit
    def softmax_step(
        self,
        softmax: SoftmaxSm100,
        sRowMax: cute.Tensor,
        sScale: cute.Tensor,
        tStS_t2r_staged: cute.Tensor,
        tSrS_t2r: cute.Tensor,
        sP_smem_view: cute.Tensor,
        tmem_load_thr: cute.CopyAtom,
        smem_store_thr: cute.CopyAtom,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        tidx: Int32,
        warp_idx: Int32,
        consumer_state_S: pipeline.PipelineState,
        producer_state_P: pipeline.PipelineState,
        producer_state_sm_stats: pipeline.PipelineState,
        stage: cutlass.Constexpr[Int32],
        n_block: Int32,
        mask_fn: Optional[Callable] = None,
        is_first: Boolean = False,
    ):
        tSrP = cute.make_rmem_tensor(tSrS_t2r.shape, self.dtype_P)
        rP_smem_view = smem_store_thr.retile(tSrP)

        pipeline_S.consumer_wait(consumer_state_S)
        cute.copy(tmem_load_thr, tStS_t2r_staged[stage], tSrS_t2r)
        cute.arch.fence_view_async_tmem_load()
        pipeline_S.consumer_release(consumer_state_S)

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)

        # compute threadwise row_max
        row_max = softmax.compute_row_max_local(tSrS_t2r.load(), is_first)
        # 2-thread reduce row_max through smem: each thread of a (row, half) pair
        # holds half the row's columns (Layout E datapath halves).
        assert self.cta_tile_m * self.num_acc_halves == 128
        sRowMax[tidx % self.cta_tile_m, warp_idx // self.num_acc_halves] = row_max
        self.softmax_barrier.arrive_and_wait()
        row_max0 = sRowMax[tidx % self.cta_tile_m, 0]
        row_max1 = sRowMax[tidx % self.cta_tile_m, 1]
        row_max = max(row_max0, row_max1)

        row_max, acc_scale = softmax.update_row_max_from_local(row_max, is_first)

        # note: acc_scales agree for paired threads
        pipeline_sm_stats.producer_acquire(producer_state_sm_stats)
        if warp_idx < self.num_acc_halves:
            sScale[tidx % self.cta_tile_m, producer_state_sm_stats.index] = acc_scale
        pipeline_sm_stats.producer_commit(producer_state_sm_stats)

        # x -> scale_log2*x-rowmax
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)

        # x -> exp2(x)
        softmax.apply_exp2_convert(tSrS_t2r, tSrP)

        pipeline_P.producer_acquire(producer_state_P)
        cute.copy(
            smem_store_thr, rP_smem_view, sP_smem_view[None, None, None, producer_state_P.index]
        )
        cute.arch.fence_view_async_shared()
        pipeline_P.producer_commit(producer_state_P)
        # unconditionally necessary for sRowMax read to complete before next iter's store
        self.softmax_barrier.arrive_and_wait()

        consumer_state_S.advance()
        producer_state_P.advance()
        producer_state_sm_stats.advance()

        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)

        return consumer_state_S, producer_state_P, producer_state_sm_stats

    @cute.jit
    def correction_loop(
        self,
        softmax_scale_log2: Float32,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_O: cute.CopyAtom,
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sO: cute.Tensor,
        tmem_ptr: cute.Pointer,
        pipeline_O0: pipeline.PipelineAsync,
        pipeline_O1: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sO_empty_mbar_ptr: cute.Pointer,
        tiled_copy_O_r2g: cute.TiledCopy,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        num_splits: Int32,
        descale_tensors,
        tile_scheduler: TileSchedulerProtocol,
    ):
        ### ==== correction/epilogue warpgroup ====
        # Correction: copy scale smem -> rmem, copy O tmem -> rmem, rescale O, store O rmem -> tmem
        # Epilogue:   copy O tmem -> rmem, do final scaling of O, store O rmem -> smem -> gmem,
        #             optionally store LSE
        # Produces: -
        # Consumes: O, softmax stats

        tidx = cute.arch.thread_idx()[0] % self.num_epilogue_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_epilogue_threads // 32
        )
        leader_warp = warp_idx == 0

        # Hand-built Layout E packed O accumulators: logical (64, hdimv/2) each.
        acc_layout_O = self._acc_tmem_layout(self.hdimv // self.num_hdimv_splits)
        tOtOs = [
            cute.make_tensor(tmem_ptr + self.tmem_offsets_O[split], acc_layout_O)
            for split in range(self.num_hdimv_splits)
        ]
        tOtO0 = tOtOs[0]

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
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            qk_descale, v_descale = self._effective_descales(
                descale_tensors, batch_idx, head_idx
            )
            # same effective scale the softmax warp used (needed for LSE)
            softmax_scale_log2_eff = softmax_scale_log2 * qk_descale

            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx=split_idx, num_splits=num_splits
            )
            num_n_blocks = n_block_max - n_block_min
            # Split-kv: this split's slice of the n_block range can be empty (the
            # per-split count is rounded up, so empties are the trailing splits).
            # Every warp role skips the whole tile body together, which keeps all
            # pipeline phases and raw-mbar phase counters consistent; the epilogue
            # still writes LSE = -inf so the combine kernel can ignore this split.
            has_work = const_expr(not self.is_split_kv) or n_block_min < n_block_max
            if has_work:

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

                # (seqlen_q, hdimv), or ((qhead_per_kvhead, seqlen_q), hdimv) when packed;
                # split-kv writes this split's fp32 partial (trailing split mode)
                mO_batch = seqlen.offset_batch_Q(mO, batch_idx, dim=3)
                mO_cur = (
                    mO_batch[None, None, head_idx]
                    if const_expr(not self.is_split_kv)
                    else mO_batch[None, None, head_idx, split_idx]
                )
                gO = None
                if const_expr(self.use_tma_O):
                    # (cta_tile_m, hdimv//2, 2)
                    gO = cute.local_tile(
                        mO_cur,
                        (self.cta_tile_m, self.hdimv // self.num_hdimv_splits),
                        (m_block, None),
                    )
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
                LN2 = math.log(2.0)
                acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                # fp8: dequantize O by v_descale here (never in the running rescale,
                # and never in LSE -- LSE is a function of the dequantized scores only).
                scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                scale = scale * v_descale

                row_max = 0.0
                if const_expr(mLSE is not None):
                    if tidx < self.cta_tile_m:
                        row_max = sRowMax[tidx, 0]

                self.sm_stats_barrier_empty.arrive()

                seqlen_q = seqlen.seqlen_q

                # compute and store lse to gmem
                if const_expr(mLSE is not None):
                    # row_sum carries the fp8 2^max_offset pre-scale; subtract it here
                    # (O needs no correction -- it cancels in O / row_sum).
                    lse = (
                        (
                            row_max * softmax_scale_log2_eff
                            + (cute.math.log2(row_sum, fastmath=True) - self.max_offset)
                        )
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    self.store_lse(mLSE, seqlen, head_idx, batch_idx, split_idx, m_block, tidx, lse)

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
                        # packed rows are scattered in gmem: store rmem -> gmem per row
                        self.store_O_packed(
                            mO_cur, tOrOs_r2g[split], tidx, m_block, split, seqlen_q
                        )
                    else:
                        # copy Oi rmem -> smem
                        cute.copy(
                            thr_tiled_copy_O_r2g,
                            tOrOs_r2g[split],
                            tOsO[None, None, None, split],
                        )
                        cute.arch.fence_view_async_shared()
                        self.epi_barrier.arrive_and_wait()
                        # tma store Oi smem -> gmem
                        if leader_warp:
                            store_O(src_idx=split, dst_idx=split)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(1 - split, read=True)
                            if const_expr(split == self.num_hdimv_splits - 1):
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive(sO_empty_mbar_ptr)

                consumer_state_O0, consumer_state_O1 = consumer_states_O

                cute.arch.fence_view_async_tmem_load()
                pipeline_O0.consumer_release(consumer_state_O0)
                pipeline_O1.consumer_release(consumer_state_O1)
                consumer_state_O0.advance()
                consumer_state_O1.advance()
            else:
                # Empty split: no O to reduce, but the combine kernel keys off LSE, so
                # this split must be marked dead. O_partial is left unwritten (combine
                # never reads a split whose LSE is -inf).
                if const_expr(mLSE is not None):
                    self.store_lse(
                        mLSE, seqlen, head_idx, batch_idx, split_idx, m_block, tidx,
                        -Float32.inf,
                    )

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def store_lse(
        self,
        mLSE: cute.Tensor,
        seqlen: SeqlenInfoQK,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
        m_block: Int32,
        tidx: Int32,
        lse: Float32,
    ):
        """Store one row's LSE. Offsets are hand-rolled (not offset_batch_Q): the seqlen
        mode is 0 but there is no d mode, and under pack_gqa the offset must land on the
        q_pos sub-mode of the packed tuple. Split-kv adds a trailing split mode."""
        if const_expr(not seqlen.has_cu_seqlens_q):
            mLSE_cur = (
                mLSE[None, head_idx, batch_idx]
                if const_expr(not self.is_split_kv)
                else mLSE[None, head_idx, batch_idx, split_idx]
            )
        else:
            lse_offset = (
                seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
            )
            mLSE_head = (
                mLSE[None, head_idx]
                if const_expr(not self.is_split_kv)
                else mLSE[None, head_idx, split_idx]
            )
            mLSE_cur = cute.domain_offset((lse_offset,), mLSE_head)
        if tidx < self.cta_tile_m:
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.cta_tile_m,), (m_block,))
                if tidx < seqlen.seqlen_q - m_block * self.cta_tile_m:
                    gLSE[tidx] = lse
            else:
                # mLSE_cur is ((qhead_per_kvhead, seqlen_q),): scatter per packed row
                packed_row = m_block * self.cta_tile_m + tidx
                q_pos = packed_row // self.qhead_per_kvhead
                h_in_group = packed_row - q_pos * self.qhead_per_kvhead
                if q_pos < seqlen.seqlen_q:
                    mLSE_cur[((h_in_group, q_pos),)] = lse

    @cute.jit
    def store_O_packed(
        self,
        mO_cur: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), hdimv) or (seqlen_q, hdimv)
        tOrO: cute.Tensor,  # this thread's O values, ((elems), 1, num_chunks)
        tidx: Int32,
        m_block: Int32,
        split: cutlass.Constexpr[int],
        seqlen_q: Int32,
    ):
        """Store one thread's slice of O to gmem, one row per thread.

        Used when a bulk TMA store is not applicable: packed pack_gqa rows (not
        contiguous in gmem) or varlen-q (rows past seqlen_q belong to the next sequence
        and must not be written).

        Under the Layout E epilogue partitioning each thread owns exactly one row
        (`tidx % cta_tile_m`) and a contiguous `cols_per_thread` run of that row, so a
        single per-thread row pointer suffices -- no cross-thread pointer shuffling.
        """
        dv_split = const_expr(self.hdimv // self.num_hdimv_splits)
        cols_per_thread = const_expr(dv_split // self.num_acc_halves)
        elems_per_copy = const_expr(cute.size(tOrO.shape[0]))
        assert cute.size(tOrO.shape[1]) == 1, "expected exactly one row per thread"

        row_in_tile = tidx % self.cta_tile_m
        packed_row = m_block * self.cta_tile_m + row_in_tile
        q_pos = packed_row // self.qhead_per_kvhead if const_expr(self.pack_gqa) else packed_row
        h_in_group = packed_row - q_pos * self.qhead_per_kvhead
        col_base = split * dv_split + cols_per_thread * (tidx // self.cta_tile_m)

        if q_pos < seqlen_q:
            # hdimv is contiguous and 16B-aligned in gmem, and col_base is a multiple of
            # cols_per_thread, so this thread's run is a 16B-aligned contiguous vector.
            row_crd = (h_in_group, q_pos) if const_expr(self.pack_gqa) else q_pos
            o_ptr = cute.make_ptr(
                self.dtype_O,
                fa_utils.elem_pointer(mO_cur, (row_crd, col_base)).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            gO_row = cute.make_tensor(o_ptr, cute.make_layout((cols_per_thread,)))
            gO_row_v = cute.tiled_divide(gO_row, (elems_per_copy,))
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype_O, num_bits_per_copy=128
            )
            for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                cute.copy(copy_atom, tOrO[None, 0, k], gO_row_v[None, k])

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
