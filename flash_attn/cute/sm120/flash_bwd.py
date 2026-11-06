# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SM120 (Blackwell GeForce / DGX Spark) backward pass.
#
# The dK/dV kernel owns one KV tile per CTA. Fused, it also adds each Q tile's dQ piece into an
# fp32 accumulator, which a postprocess converts; when the keys span one or two KV tiles, one
# CTA instead covers all of them for its Q range and stores dQ itself (direct_dq). Split, a
# separate dQ kernel owns one Q tile per CTA and recomputes S and dP (7 GEMMs instead of 5)
# without atomics, and every output is written once by its owner, which makes the result
# deterministic.

from types import SimpleNamespace
from typing import Callable, Optional
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisorV2
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op
from quack import copy_utils as quack_copy_utils
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import copy_utils, utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import apply_score_mod_bwd_inner, apply_score_mod_inner
from flash_attn.cute.utils import AuxData


@dsl_user_op
def red_add_v2_f32(gmem_ptr: cute.Pointer, a, b, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            gmem_ptr.llvm_ptr,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_v2_f32(gmem_ptr: cute.Pointer, a, b, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            gmem_ptr.llvm_ptr,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_relaxed_gpu_v2_f32(gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> tuple:
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [gmem_ptr.llvm_ptr],
        "ld.relaxed.gpu.global.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), res, [j], loc=loc, ip=ip)) for j in range(2)
    )


# tma_ws named barriers. 0 is the consumer barrier (256 threads). Consumers: A = warps 0-3
# (KV rows 0-63), B = warps 4-7; warp 8 is the TMA producer (warps 9-11 only give up registers).
_TW_QR_A, _TW_QR_B = 1, 3  # + stage: Q(i) + LSE landed (producer warp + 128)
_TW_DR_A, _TW_DR_B = 5, 6  # dO(i) + dPsum landed (producer warp + 128)
_TW_QE = 7  # + stage: both groups' dK(i) done with Q stage i % 2 (256 + producer warp)
_TW_DE = 9  # both groups' dV(i) done with dO (256 + producer warp)
_TW_HEAD = 10  # consumers done with a head (256 + producer warp)
_TW_A_S = 11  # A: S(i) GEMM done (and dQ(i - 1)) -> B starts S(i)
_TW_A_DS = 12  # A: dS(i) half stored -> B dQ(i)
_TW_B_DS = 13  # B: dS(i) half stored -> A dQ(i)
_TW_B_END = 14  # + i % 2, B: iteration i done (dS(i) reads) -> A stores dS(i + 1)
_TW_PRODUCER_REGS, _TW_CONSUMER_REGS = 24, 240


def _tma_smem_layout_atom(dtype, major_dim: int) -> cute.ComposedLayout:
    """K-major SMEM atom that both TMA and the warp-MMA LDSM views accept (as in the fwd)."""
    major_mode_bits = major_dim * dtype.width
    contiguous_bits, swizzle_bits = next(
        (c, b) for c, b in ((1024, 3), (512, 2), (256, 1), (128, 0)) if major_mode_bits % c == 0
    )
    contiguous_elems = contiguous_bits // dtype.width
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 4, 3),
        0,
        cute.make_layout((8, contiguous_elems), stride=(contiguous_elems, 1)),
    )


@dsl_user_op
def mbarrier_inval(mbar_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [mbar_ptr.toint(loc=loc, ip=ip).ir_value()],
        "mbarrier.inval.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class _ScoreModSm120:
    """score_mod / score_mod_bwd on an S tile with the tile's logical coordinates."""

    def _score_mod_coords(self, thr_mma, m_block, n_block, transposed: bool):
        tile = (
            (self.n_block_size, self.m_block_size)
            if transposed
            else (self.m_block_size, self.n_block_size)
        )
        origin = (
            (n_block * self.n_block_size, m_block * self.m_block_size)
            if transposed
            else (m_block * self.m_block_size, n_block * self.n_block_size)
        )
        return thr_mma.partition_C(cute.domain_offset(origin, cute.make_identity_tensor(tile)))

    def _aux_fastdiv_mods(self, aux_data: AuxData, seqlen_q, seqlen_k):
        """Divisors that wrap score_mod / mask_mod indices into aux tensors (as on SM100)."""
        if const_expr(aux_data.tensors is None):
            return (None, None)
        return (FastDivmodDivisorV2(seqlen_q), FastDivmodDivisorV2(seqlen_k))

    @cute.jit
    def apply_score_mod(
        self,
        acc_S,
        thr_mma,
        m_block,
        n_block,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen,
        transposed: cutlass.Constexpr[bool],
        aux_data: AuxData,
        fastdiv_mods,
    ):
        apply_score_mod_inner(
            acc_S,
            self._score_mod_coords(thr_mma, m_block, n_block, transposed),
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            1,
            Float32,
            aux_data,
            fastdiv_mods,
            seqlen,
            constant_q_idx=None,
            # Pack-GQA factor: Q heads are separate rows here, never packed.
            qhead_per_kvhead=1,
            transpose_indices=transposed,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad,
        score,
        thr_mma,
        m_block,
        n_block,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen,
        transposed: cutlass.Constexpr[bool],
        aux_data: AuxData,
        fastdiv_mods,
    ):
        apply_score_mod_bwd_inner(
            grad,
            score,
            self._score_mod_coords(thr_mma, m_block, n_block, transposed),
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            1,
            Float32,
            aux_data,
            fastdiv_mods,
            seqlen,
            constant_q_idx=None,
            # Pack-GQA factor: Q heads are separate rows here, never packed.
            qhead_per_kvhead=1,
            transpose_indices=transposed,
        )

    @cute.jit
    def _zero_masked_dS(self, dS, P):
        """dS = 0 where P = 0. score_mod_bwd scales dS there by values computed at masked
        positions, whose aux reads may fall past the end of the aux data; P = 0 makes the
        gradient 0 whatever those values are (NaN included)."""
        for i in cutlass.range(cute.size(dS), unroll_full=True):
            dS[i] = 0.0 if P[i] == 0.0 else dS[i]


class FlashAttentionBackwardDKDVSm120(_ScoreModSm120, FlashAttentionBackwardSm80):
    """dK/dV kernel (with fuse_dq also dQ): one CTA per KV tile, iterating over Q tiles.

    S^T = K Q^T and dP^T = V dO^T keep the KV tile on the MMA M axis, so their
    accumulators are directly the register A operands of dV += P^T dO and dK += dS^T Q
    (unless pds_in_smem); dS goes through shared memory only for the fused dQ GEMM.
    """

    def __init__(
        self,
        *args,
        fuse_dq: bool = False,
        head_split: int = 1,
        split_slots: bool = False,
        tail_split: int = 1,
        l2_group_bytes: int = 50 * 1024 * 1024,
        l2_group_ctas: int = 0,
        tma_ws: bool = False,
        direct_dq: bool = False,
        kv_sweep: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        accum_hdim_multiple: int = 64,
        mask_mod: cutlass.Constexpr | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_mod = mask_mod
        # Row stride of the fp32 accumulate tiles (dQ, split-group dK / dV), shared with the
        # postprocess, whose row-major reader needs it to tile its 16 B vectors evenly.
        self.dq_accum_hdim = -(-self.head_dim_padded // accum_hdim_multiple) * accum_hdim_multiple
        self.dv_accum_hdim = -(-self.head_dim_v_padded // accum_hdim_multiple) * accum_hdim_multiple
        # head_split > 1 splits each KV head's group of Q heads over that many CTAs (small
        # grids, e.g. MQA); their dK / dV partials go into row-major fp32 accumulators.
        assert self.qhead_per_kvhead % head_split == 0
        self.head_split = head_split
        # split_slots: each split adds into its own accumulator (slot head_slot of
        # h_kv * head_split), which the host sums after the kernel: deterministic.
        assert not split_slots or head_split > 1
        self.split_slots = split_slots
        # tail_split > 1: the last tail_tiles KV tiles of the grid order (the
        # partial last wave) each run as tail_split CTAs over equal shares of their Q tiles.
        # Without a head split their dK / dV partials meet in a per-tile fp32 scratch; the
        # last part to arrive (counted per tile) stores the sum.
        assert tail_split == 1 or not split_slots
        self.tail_split = tail_split
        # Dense grids launch head groups whose Q / dO / dQaccum fill this many L2 bytes.
        self.l2_group_bytes = l2_group_bytes
        self.l2_group_ctas = l2_group_ctas
        assert self.SdP_swapAB, "the SM120 dK/dV kernel computes S^T = K Q^T"
        # With every warp on the KV axis, P^T / dS^T are the register A operands of the dV / dK
        # GEMMs. Other layouts (e.g. splitting dK / dV over the head dim for hd 256) stage
        # them in shared memory.
        self.pds_in_smem = not self.Mma_dKV_is_RS
        # tma_ws: a producer warpgroup loads Q / LSE / dO / dPsum with TMA, and the
        # two KV-row halves of the consumers run out of phase (B's S after A's S), synced through
        # named barriers only.
        self.tma_ws = tma_ws
        # direct_dq (seqlen_k <= tile_n, dense, or kv_sweep): the CTA's KV tiles are every key,
        # so its dQ tiles are complete; mdQaccum is dQ itself, stored scaled in the input dtype
        # (no fp32 accumulator, its zero fill or the postprocess).
        self.direct_dq = direct_dq
        assert not direct_dq or fuse_dq
        # kv_sweep (with direct_dq, dense, non-causal, non-local): the keys span more than one KV
        # tile (the runtime takes two), which one CTA sweeps for its Q range. Its dQ rows are
        # then complete after the last sweep; between sweeps their fp32 partials wait in a
        # scratch that every thread reads back for its own elements (no zero fill, reductions
        # or postprocess).
        self.kv_sweep = kv_sweep
        assert not kv_sweep or direct_dq
        # Block sparsity (Q-direction lists per KV block): a sparse block covers
        # q_subtile_factor Q tiles and kv_subtile_factor KV tiles.
        self.q_subtile_factor = q_subtile_factor
        self.kv_subtile_factor = kv_subtile_factor
        assert not tma_ws or (
            fuse_dq
            and not self.pds_in_smem
            and self.num_stages_Q == 2
            and self.num_stages_dO == 1
            and self.num_threads == 256
        )
        assert (self.score_mod is None) == (self.score_mod_bwd is None)
        # fuse_dq: also compute dS K per Q tile and add it into a row-major fp32 dQaccum
        # tile, one head-dim piece at a time to bound the accumulator registers. A piece is
        # one swizzle atom of sK wide: a view starting inside an atom would apply the swizzle
        # to the wrong relative offsets.
        self.fuse_dq = fuse_dq
        # Same width rule as sm80_utils.get_smem_layout_atom.
        bytes_per_row = self.head_dim_padded * self.dtype.width // 8
        atom_bytes = next(w for w in (128, 64, 32, 16) if bytes_per_row % w == 0)
        self.dq_chunk = atom_bytes * 8 // self.dtype.width
        assert self.head_dim_padded % self.dq_chunk == 0

    def _get_tiled_mma(self):
        tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq = super()._get_tiled_mma()
        n_warps_dq = self.num_threads // 32 // self.AtomLayoutMdQ
        if const_expr(self.fuse_dq and self.dq_chunk < 16 * n_warps_dq):
            # A dQ piece narrower than the warps' 16-column default (hd 96: 32 columns over
            # 4 warps): one 16 x 8 atom per warp along the head dim.
            assert self.dq_chunk == 8 * n_warps_dq
            tiled_mma_dq = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                (self.AtomLayoutMdQ, n_warps_dq, 1),
                permutation_mnk=(self.AtomLayoutMdQ * 16, self.dq_chunk, 16),
            )
        return tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq

    def _cbar(self):
        """Barrier over every MMA thread (the whole CTA unless a producer warpgroup exists)."""
        if const_expr(self.tma_ws):
            cute.arch.barrier(barrier_id=0, number_of_threads=self.num_threads)
        else:
            cute.arch.barrier()

    def _setup_attributes(self):
        super()._setup_attributes()
        if const_expr(self.tma_ws):
            self.sQ_layout = cute.tile_to_shape(
                _tma_smem_layout_atom(self.dtype, self.head_dim_padded),
                (self.m_block_size, self.head_dim_padded, self.num_stages_Q),
                (0, 1, 2),
            )
            self.sdO_layout = cute.tile_to_shape(
                _tma_smem_layout_atom(self.dtype, self.head_dim_v_padded),
                (self.m_block_size, self.head_dim_v_padded, self.num_stages_dO),
                (0, 1, 2),
            )
        # dK / dV are written once per KV head in the input dtype, also for GQA.
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        hdim_max = max(self.head_dim_padded, self.head_dim_v_padded)
        atom_bytes = next(w for w in (128, 64, 32, 16) if hdim_max * self.dtype.width // 8 % w == 0)
        threads_per_row = atom_bytes * 8 // self.dtype.width // async_copy_elems
        self.gmem_tiled_copy_dKV_out = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits
            ),
            cute.make_ordered_layout(
                (self.num_threads // threads_per_row, threads_per_row), order=(1, 0)
            ),
            cute.make_layout((1, async_copy_elems)),
        )
        if const_expr(self.fuse_dq or self.pds_in_smem):
            # P^T / dS^T are stored as computed, (n, m); the dQ GEMM reads dS through its
            # transpose.
            self.sPdS_layout = cute.tile_to_shape(
                sm80_utils.get_smem_layout_atom(self.dtype, self.m_block_size),
                (self.n_block_size, self.m_block_size),
                (0, 1),
            )

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout, self.sdO_layout)
        ]
        sLSE_struct, sdPsum_struct = [
            cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(self.sLSE_layout)], 128]
            for _ in range(2)
        ]

        @cute.struct
        class SharedStorage:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct

        if const_expr(not self.fuse_dq and not self.pds_in_smem):
            return SharedStorage
        sdS_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 1024
        ]

        @cute.struct
        class SharedStorageStaged:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sdS: sdS_struct

        @cute.struct
        class SharedStorageTma:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sdS: sdS_struct
            # TMA transaction barriers: Q stage 0 / 1 (with LSE), dO (with dPsum).
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 8]

        if const_expr(self.tma_ws):
            return SharedStorageTma
        if const_expr(not self.pds_in_smem):
            return SharedStorageStaged

        @cute.struct
        class SharedStorageStagedP:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sdS: sdS_struct
            sP: sdS_struct

        return SharedStorageStagedP

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        tail_tiles: Int32 | int | None = None,
        mTailK: Optional[cute.Tensor] = None,
        mTailV: Optional[cute.Tensor] = None,
        mTailCount: Optional[cute.Tensor] = None,
        mdQpart: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream=None,
    ):
        from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
        from flash_attn.cute.tile_scheduler import (
            SingleTileLPTBwdScheduler,
            SingleTileVarlenScheduler,
            TileSchedulerArguments,
        )

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) if t is not None else None
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]
        assert (mdQaccum is not None) == self.fuse_dq
        self.varlen_q = mCuSeqlensQ is not None
        self.block_sparse = blocksparse_tensors is not None
        assert not self.block_sparse or (
            self.fuse_dq and self.tail_split == 1 and not self.direct_dq and not self.varlen_q
        )
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq = self._get_tiled_mma()
        varlen_k = const_expr(mCuSeqlensK is not None or mSeqUsedK is not None)
        num_head_kv = mK.shape[1] if const_expr(mCuSeqlensK is not None) else mK.shape[2]
        num_batch = (
            mCuSeqlensK.shape[0] - 1
            if const_expr(mCuSeqlensK is not None)
            else (mSeqUsedK.shape[0] if const_expr(mSeqUsedK is not None) else mK.shape[0])
        )
        seqlen_k = mK.shape[0] if const_expr(mCuSeqlensK is not None) else mK.shape[1]
        # Dense: KV tiles in ascending order across a group of heads that shares Q / dO in
        # L2; for causal masks the low KV tiles see the most Q tiles, so this is longest
        # first.
        TileScheduler = SingleTileVarlenScheduler if varlen_k else SingleTileLPTBwdScheduler
        seqlen_q = mQ.shape[0] if const_expr(mCuSeqlensQ is not None) else mQ.shape[1]
        # One CTA per (KV tile, KV head, head split, batch), with direct_dq per (KV head,
        # head split, batch) over every KV tile; its Q heads loop inside it.
        tile_sched_args = TileSchedulerArguments(
            num_block=1
            if const_expr(self.direct_dq)
            else cute.ceil_div(seqlen_k, self.n_block_size),
            num_head=num_head_kv * self.head_split,
            num_batch=num_batch,
            num_splits=1,
            # Q rows per (KV head, split): the L2 grouping's per-head footprint.
            seqlen_k=seqlen_q * self.qhead_per_kvhead // self.head_split,
            headdim=mK.shape[cute.rank(mK) - 1],
            headdim_v=mV.shape[cute.rank(mV) - 1],
            total_q=(
                mK.shape[0] if const_expr(mCuSeqlensK is not None) else mK.shape[0] * mK.shape[1]
            ),
            tile_shape_mn=(self.n_block_size, self.m_block_size),
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            element_size=self.dtype.width // 8,
            lpt=False,
            l2_group_bytes=self.l2_group_bytes,
            l2_group_max_ctas=self.l2_group_ctas,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        if const_expr(self.tail_split > 1):
            tail_tiles = Int32(tail_tiles)
            grid_dim = (grid_dim[0] + tail_tiles * (self.tail_split - 1), grid_dim[1], grid_dim[2])
        softmax_scale_log2, _ = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)
        fastdiv_mods = self._aux_fastdiv_mods(aux_data, seqlen_q, seqlen_k)
        tma = [None] * 8
        if const_expr(self.tma_ws):
            # Rows first: (s, d, h, b) / (s, h, b) views, or for varlen Q (total, d, h) /
            # (padded total, h). Varlen tiles start at a sequence's row offset; rows past its
            # end are the next sequence's (or TMA's zero fill), with LSE +inf and dPsum 0 in
            # the padded scratch, so P = dS = 0 there.
            g2s = cpasync.CopyBulkTensorTileG2SOp()
            m, d, dv = self.m_block_size, self.head_dim_padded, self.head_dim_v_padded
            rows, stats = (
                ([0, 2, 1], [1, 0]) if const_expr(self.varlen_q) else ([1, 3, 2, 0], [2, 1, 0])
            )
            sources = (
                (mQ, cute.select(self.sQ_layout, mode=[0, 1]), (m, d), rows),
                (mdO, cute.select(self.sdO_layout, mode=[0, 1]), (m, dv), rows),
                (mLSE, cute.select(self.sLSE_layout, mode=[0]), (m,), stats),
                (mdPsum, cute.select(self.sLSE_layout, mode=[0]), (m,), stats),
            )
            for i, (t, smem_layout, tiler, order) in enumerate(sources):
                t_tma = cute.make_tensor(t.iterator, cute.select(t.layout, mode=order))
                tma[2 * i], tma[2 * i + 1] = cpasync.make_tiled_tma_atom(
                    g2s, t_tma, smem_layout, tiler, 1
                )
        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            aux_data,
            fastdiv_mods,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sPdS_layout,
            self.sLSE_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_VdO,
            self.gmem_tiled_copy_dKV_out,
            self.gmem_tiled_copy_LSE,
            tiled_mma_sdp,
            tiled_mma_dkv,
            tiled_mma_dq,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            *tma,
            tail_tiles,
            mTailK,
            mTailV,
            mTailCount,
            mdQpart,
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads + (128 if const_expr(self.tma_ws) else 0), 1, 1],
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
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        aux_data: AuxData,
        fastdiv_mods,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dKV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        tma_atom_Q: Optional[cute.CopyAtom] = None,
        tma_Q: Optional[cute.Tensor] = None,
        tma_atom_dO: Optional[cute.CopyAtom] = None,
        tma_dO: Optional[cute.Tensor] = None,
        tma_atom_LSE: Optional[cute.CopyAtom] = None,
        tma_LSE: Optional[cute.Tensor] = None,
        tma_atom_dPsum: Optional[cute.CopyAtom] = None,
        tma_dPsum: Optional[cute.Tensor] = None,
        tail_tiles: Optional[Int32] = None,
        mTailK: Optional[cute.Tensor] = None,
        mTailV: Optional[cute.Tensor] = None,
        mTailCount: Optional[cute.Tensor] = None,
        mdQpart: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if const_expr(self.tail_split > 1):
            # Blocks past base are tail_split parts of the grid order's last tail_tiles tiles.
            bidx = cute.arch.block_idx()[0]
            base = tile_sched_params.total_blocks - tail_tiles
            is_tail = bidx >= base
            tile_lin, tail_part, tail_i = bidx, Int32(0), Int32(0)
            if is_tail:
                tail_i = (bidx - base) // self.tail_split
                tail_part = (bidx - base) - tail_i * self.tail_split
                tile_lin = base + tail_i
            tile_scheduler = TileScheduler(tile_sched_params, tile_lin)
        else:
            tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        n_block, head_slot, batch_idx, _ = work_tile.tile_idx
        head_idx_kv = head_slot // self.head_split
        split_idx = head_slot - head_idx_kv * self.head_split

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mQ.shape[1],
                mK.shape[1],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                tile_m=self.m_block_size,
                tile_n=self.n_block_size,
            )
            block_info = BlockInfo(
                self.m_block_size,
                self.n_block_size,
                self.is_causal,
                self.is_local,
                False,
                window_size_left,
                window_size_right,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            if const_expr(self.block_sparse):
                # Each head visits the Q tiles of its list; the causal / local masks still apply.
                m_block_min = Int32(0)
                m_block_max = cute.ceil_div(seqlen.seqlen_q, self.m_block_size)
            if const_expr(self.tail_split > 1):
                if is_tail:
                    m_count = m_block_max - m_block_min
                    m_block_max = m_block_min + (tail_part + 1) * m_count // self.tail_split
                    m_block_min = m_block_min + tail_part * m_count // self.tail_split
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur, mV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mK, mV)]
            else:
                mK_cur, mV_cur = [
                    cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None])
                    for t in (mK, mV)
                ]
            if const_expr(self.head_split == 1):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mdK_cur, mdV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mdK, mdV)]
                else:
                    mdK_cur, mdV_cur = [
                        cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None])
                        for t in (mdK, mdV)
                    ]
            else:
                # (b, h_kv, s * hdim) or (h_kv, padded total * hdim) fp32 accumulators; with
                # split_slots, h_kv * head_split of them.
                accum_head = head_slot if const_expr(self.split_slots) else head_idx_kv
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mdK_cur, mdV_cur = (
                        mdK[batch_idx, accum_head, None],
                        mdV[batch_idx, accum_head, None],
                    )
                else:
                    mdK_cur = cute.domain_offset(
                        (seqlen.padded_offset_k * self.dq_accum_hdim,), mdK[accum_head, None]
                    )
                    mdV_cur = cute.domain_offset(
                        (seqlen.padded_offset_k * self.dv_accum_hdim,), mdV[accum_head, None]
                    )
            gK = cute.local_tile(mK_cur, (self.n_block_size, self.head_dim_padded), (n_block, 0))
            gV = cute.local_tile(mV_cur, (self.n_block_size, self.head_dim_v_padded), (n_block, 0))

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            if const_expr(self.tma_ws):
                # TMA layouts: the swizzle acts on the byte address, as TMA writes it.
                sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
                sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
            else:
                sQ = storage.sQ.get_tensor(sQ_layout)
                sdO = storage.sdO.get_tensor(sdO_layout)
            sK = storage.sK.get_tensor(sK_layout)
            sV = storage.sV.get_tensor(sV_layout)
            sLSE = storage.sLSE.get_tensor(sLSE_layout)
            sdPsum = storage.sdPsum.get_tensor(sLSE_layout)
            sQt, sdOt = [layout_utils.transpose_view(t) for t in (sQ, sdO)]
            sdSt = (
                storage.sdS.get_tensor(sPdS_layout)
                if const_expr(self.fuse_dq or self.pds_in_smem)
                else None
            )
            sPt = storage.sP.get_tensor(sPdS_layout) if const_expr(self.pds_in_smem) else None
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if const_expr(self.tma_ws):
                if warp_idx >= self.num_threads // 32:
                    cute.arch.setmaxregister_decrease(_TW_PRODUCER_REGS)
                    if warp_idx == self.num_threads // 32 and m_block_min < m_block_max:
                        self.tma_producer(
                            storage.mbar.data_ptr(),
                            (tma_atom_Q, tma_Q, tma_atom_dO, tma_dO),
                            (tma_atom_LSE, tma_LSE, tma_atom_dPsum, tma_dPsum),
                            sQ,
                            sdO,
                            sLSE,
                            sdPsum,
                            batch_idx,
                            seqlen,
                            head_idx_kv,
                            split_idx,
                            m_block_min,
                            m_block_max,
                            n_block,
                            blocksparse_tensors,
                        )
            is_consumer = warp_idx < self.num_threads // 32 if const_expr(self.tma_ws) else True
            if is_consumer:
                if const_expr(self.tma_ws):
                    cute.arch.setmaxregister_increase(_TW_CONSUMER_REGS)

                gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
                gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
                tKgK, tKsK = gmem_thr_copy_QK.partition_S(gK), gmem_thr_copy_QK.partition_D(sK)
                tVgV, tVsV = gmem_thr_copy_VdO.partition_S(gV), gmem_thr_copy_VdO.partition_D(sV)

                thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
                thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
                acc_dK = cute.make_rmem_tensor(
                    thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_padded)),
                    Float32,
                )
                acc_dV = cute.make_rmem_tensor(
                    thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_v_padded)),
                    Float32,
                )
                acc_dK.fill(0.0)
                acc_dV.fill(0.0)

                # S^T = K Q^T, dP^T = V dO^T: A is the KV tile, B the Q tile.
                tSrK = utils.mma_make_fragment_A(sK, thr_mma_sdp)
                tSrQ = utils.mma_make_fragment_B(sQ[None, None, 0], thr_mma_sdp)
                tdPrV = utils.mma_make_fragment_A(sV, thr_mma_sdp)
                tdPrdO = utils.mma_make_fragment_B(sdO[None, None, 0], thr_mma_sdp)
                # dV += P^T dO, dK += dS^T Q: A from registers, B = dO^T / Q^T views.
                tdVrdO = utils.mma_make_fragment_B(sdOt[None, None, 0], thr_mma_dkv)
                tdKrQ = utils.mma_make_fragment_B(sQt[None, None, 0], thr_mma_dkv)
                # Per-thread LSE / dPsum for the Q rows (accumulator columns) this thread owns.
                tSsLSE = layout_utils.mma_partition_C_vec(
                    sLSE, thr_mma_sdp, expand_shape=self.n_block_size, is_colvec=False
                )
                tSsdPsum = layout_utils.mma_partition_C_vec(
                    sdPsum, thr_mma_sdp, expand_shape=self.n_block_size, is_colvec=False
                )

                smem_copy_atom = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
                )
                smem_copy_atom_transposed = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
                )
                smem_thr_copy_KV = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_sdp).get_slice(
                    tidx
                )
                smem_thr_copy_QdO = utils.make_tiled_copy_B(
                    smem_copy_atom, tiled_mma_sdp
                ).get_slice(tidx)
                smem_thr_copy_QdOt = utils.make_tiled_copy_B(
                    smem_copy_atom_transposed, tiled_mma_dkv
                ).get_slice(tidx)
                d_head = mQ.shape[cute.rank(mQ) - 1]
                d_head_v = mdO.shape[cute.rank(mdO) - 1]

                # TMA stages swizzle the byte address: swizzle each thread's base pointer once and
                # keep the stage and k-step offsets linear, so they fold into immediates instead
                # of re-swizzling every address per stage.
                partition_staged = (
                    quack_copy_utils.partition_S_position_independent
                    if const_expr(self.tma_ws)
                    else (lambda thr_copy, t: thr_copy.partition_S(t))
                )
                params = SimpleNamespace(
                    tiled_mma_sdp=tiled_mma_sdp,
                    tiled_mma_dkv=tiled_mma_dkv,
                    thr_mma_sdp=thr_mma_sdp,
                    tSrK=tSrK,
                    tSrQ=tSrQ,
                    tdPrV=tdPrV,
                    tdPrdO=tdPrdO,
                    tdVrdO=tdVrdO,
                    tdKrQ=tdKrQ,
                    tSsK=smem_thr_copy_KV.partition_S(sK),
                    tSsQ=partition_staged(smem_thr_copy_QdO, sQ),
                    tdPsV=smem_thr_copy_KV.partition_S(sV),
                    tdPsdO=partition_staged(smem_thr_copy_QdO, sdO),
                    tdVsdOt=partition_staged(smem_thr_copy_QdOt, sdOt),
                    tdKsQt=partition_staged(smem_thr_copy_QdOt, sQt),
                    tSsLSE=tSsLSE,
                    tSsdPsum=tSsdPsum,
                    smem_thr_copy_KV=smem_thr_copy_KV,
                    smem_thr_copy_QdO=smem_thr_copy_QdO,
                    smem_thr_copy_QdOt=smem_thr_copy_QdOt,
                    acc_dK=acc_dK,
                    acc_dV=acc_dV,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                )
                if const_expr(self.fuse_dq or self.pds_in_smem):
                    params.sdSt = sdSt
                    params.r2s_thr_copy_dS = cute.make_tiled_copy_C(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.dtype,
                            num_bits_per_copy=2 * self.dtype.width,
                        ),
                        tiled_mma_sdp,
                    ).get_slice(tidx)
                if const_expr(self.pds_in_smem):
                    params.sPt = sPt
                    smem_thr_copy_PdSt = utils.make_tiled_copy_A(
                        smem_copy_atom, tiled_mma_dkv
                    ).get_slice(tidx)
                    params.smem_thr_copy_PdSt = smem_thr_copy_PdSt
                    params.tdVrPt = utils.mma_make_fragment_A(sPt, thr_mma_dkv)
                    params.tdKrdSt = utils.mma_make_fragment_A(sdSt, thr_mma_dkv)
                    params.tdVsPt = smem_thr_copy_PdSt.partition_S(sPt)
                    params.tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt)
                if const_expr(self.fuse_dq):
                    params.tiled_mma_dq = tiled_mma_dq
                    params.thr_mma_dq = tiled_mma_dq.get_slice(tidx)
                    params.sdS = layout_utils.transpose_view(sdSt)
                    params.sKt = layout_utils.transpose_view(sK)
                    params.smem_thr_copy_dS = utils.make_tiled_copy_A(
                        smem_copy_atom_transposed, tiled_mma_dq
                    ).get_slice(tidx)
                    params.smem_thr_copy_Kt = utils.make_tiled_copy_B(
                        smem_copy_atom_transposed, tiled_mma_dq
                    ).get_slice(tidx)

                n_sweeps = 1
                if const_expr(self.kv_sweep):
                    n_sweeps = cute.ceil_div(seqlen.seqlen_k, self.n_block_size)

                # One KV tile: a fully unrolled single iteration (no loop in the kernel).
                for sweep in cutlass.range(
                    n_sweeps, unroll=1, unroll_full=const_expr(not self.kv_sweep)
                ):
                    kv_block = n_block + sweep
                    tKgK_s, tVgV_s = tKgK, tVgV
                    if const_expr(self.kv_sweep):
                        if sweep > 0:
                            acc_dK.fill(0.0)
                            acc_dV.fill(0.0)
                            # The previous KV tile's dK / dV store is done with sK / sV.
                            self._cbar()
                        tKgK_s = gmem_thr_copy_QK.partition_S(
                            cute.local_tile(
                                mK_cur, (self.n_block_size, self.head_dim_padded), (kv_block, 0)
                            )
                        )
                        tVgV_s = gmem_thr_copy_VdO.partition_S(
                            cute.local_tile(
                                mV_cur, (self.n_block_size, self.head_dim_v_padded), (kv_block, 0)
                            )
                        )
                    if m_block_min < m_block_max:
                        self.load_V(
                            gmem_thr_copy_VdO,
                            tVgV_s,
                            tVsV,
                            kv_block,
                            seqlen=seqlen.seqlen_k,
                            headdim=d_head_v,
                        )
                        self.load_K(
                            gmem_thr_copy_QK,
                            tKgK_s,
                            tKsK,
                            kv_block,
                            seqlen=seqlen.seqlen_k,
                            headdim=d_head,
                        )
                        cute.arch.cp_async_commit_group()
                        mask = AttentionMask(
                            self.m_block_size,
                            self.n_block_size,
                            seqlen,
                            window_size_left,
                            window_size_right,
                            swap_AB=True,
                        )
                        # A runtime trip count (Q heads over KV heads and splits): with a
                        # constant one, ptxas restructures the head-split variants' mainloop and
                        # spills the shared-memory addresses the GEMMs reload every k step.
                        q_heads = (
                            mQ.shape[1] if const_expr(mCuSeqlensQ is not None) else mQ.shape[2]
                        )
                        kv_heads = (
                            mK.shape[1] if const_expr(mCuSeqlensK is not None) else mK.shape[2]
                        )
                        heads_per_split = q_heads // (kv_heads * self.head_split)
                        for head_in_group in cutlass.range(heads_per_split, unroll=1):
                            head_idx = (
                                head_idx_kv * self.qhead_per_kvhead
                                + split_idx * heads_per_split
                                + head_in_group
                            )
                            if const_expr(self.block_sparse):
                                idx_p, idx_f, n_partial, n_iter = self._sparse_q_list(
                                    blocksparse_tensors, batch_idx, head_idx, kv_block, m_block_max
                                )
                                if n_iter > 0:
                                    self.dkdv_one_head(
                                        head_idx,
                                        batch_idx,
                                        kv_block,
                                        Int32(0),
                                        n_iter,
                                        seqlen,
                                        mask,
                                        mQ,
                                        mdO,
                                        mLSE,
                                        mdPsum,
                                        mdQaccum,
                                        sQ,
                                        sdO,
                                        sLSE,
                                        sdPsum,
                                        gmem_tiled_copy_QK,
                                        gmem_tiled_copy_VdO,
                                        gmem_tiled_copy_LSE,
                                        params,
                                        d_head,
                                        d_head_v,
                                        softmax_scale_log2,
                                        softmax_scale,
                                        sparse_q=(idx_p, idx_f, n_partial),
                                    )
                            else:
                                self.dkdv_one_head(
                                    head_idx,
                                    batch_idx,
                                    kv_block,
                                    m_block_min,
                                    m_block_max,
                                    seqlen,
                                    mask,
                                    mQ,
                                    mdO,
                                    mLSE,
                                    mdPsum,
                                    mdQaccum,
                                    sQ,
                                    sdO,
                                    sLSE,
                                    sdPsum,
                                    gmem_tiled_copy_QK,
                                    gmem_tiled_copy_VdO,
                                    gmem_tiled_copy_LSE,
                                    params,
                                    d_head,
                                    d_head_v,
                                    softmax_scale_log2,
                                    softmax_scale,
                                    mdQpart,
                                    sweep,
                                    n_sweeps,
                                    # A sweep's last head is reported after its dK / dV epilogue.
                                    head_in_group + 1 < heads_per_split or sweep + 1 == n_sweeps,
                                )
                        cute.arch.cp_async_wait_group(0)

                    if const_expr(self.head_split == 1):
                        store = Int32(1)
                        if const_expr(self.tail_split > 1):
                            if is_tail:
                                store = self._tail_fixup(
                                    acc_dK,
                                    acc_dV,
                                    thr_mma_dkv,
                                    mTailK[tail_i * n_sweeps + sweep, None],
                                    mTailV[tail_i * n_sweeps + sweep, None],
                                    mTailCount,
                                    tail_i * n_sweeps + sweep,
                                    tail_part,
                                    sLSE,
                                    tidx,
                                )
                        if store == 1:
                            self._store_dKV_scaled(
                                acc_dK,
                                acc_dV,
                                mdK_cur,
                                mdV_cur,
                                sK,
                                sV,
                                gmem_tiled_copy_dKV,
                                tiled_mma_dkv,
                                tidx,
                                kv_block,
                                seqlen,
                                d_head,
                                d_head_v,
                                softmax_scale,
                            )
                    elif m_block_min < m_block_max:
                        # The postprocess applies softmax_scale to dK.
                        self._reduce_dKV(
                            acc_dK,
                            mdK_cur,
                            thr_mma_dkv,
                            kv_block,
                            self.head_dim_padded,
                            self.dq_accum_hdim,
                        )
                        self._reduce_dKV(
                            acc_dV,
                            mdV_cur,
                            thr_mma_dkv,
                            kv_block,
                            self.head_dim_v_padded,
                            self.dv_accum_hdim,
                        )

                    if const_expr(self.kv_sweep and self.tma_ws):
                        if sweep + 1 < n_sweeps and m_block_min < m_block_max:
                            # The last head's _TW_HEAD arrival, deferred past the dK / dV fixup
                            # (which uses sLSE): the producer refills sQ / sLSE for the next KV
                            # tile only after it. One arrival per producer wait; the next one
                            # follows the producer's Q publication, i.e. its wait.
                            cute.arch.barrier_arrive(
                                barrier_id=_TW_HEAD, number_of_threads=self.num_threads + 32
                            )

    @cute.jit
    def dkdv_one_head(
        self,
        head_idx: Int32,
        batch_idx: Int32,
        n_block: Int32,
        m_block_min: Int32,
        m_block_max: Int32,
        seqlen: SeqlenInfoQK,
        mask: AttentionMask,
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        params: SimpleNamespace,
        d_head: Int32,
        d_head_v: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        mdQpart: Optional[cute.Tensor] = None,
        sweep: Int32 | int = 0,
        n_sweeps: Int32 | int = 1,
        notify_head: cutlass.Boolean | bool = True,
        sparse_q: Optional[tuple] = None,
    ):
        """Accumulate one Q head's contribution to this KV tile's dK / dV (and its dQ).

        sparse_q (block sparsity): (partial list, full list, partial iterations) of this head;
        m_block_min / m_block_max then bound the iteration index, not the Q tile.
        """
        tidx, _, _ = cute.arch.thread_idx()
        if const_expr(not seqlen.has_cu_seqlens_q):
            mQ_cur = mQ[batch_idx, None, head_idx, None]
            mLSE_cur = mLSE[batch_idx, head_idx, None]
            mdO_cur = mdO[batch_idx, None, head_idx, None]
            mdPsum_cur = mdPsum[batch_idx, head_idx, None]
        else:
            mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, head_idx, None])
            mLSE_cur = cute.domain_offset((seqlen.padded_offset_q,), mLSE[head_idx, None])
            mdO_cur = cute.domain_offset((seqlen.offset_q, 0), mdO[None, head_idx, None])
            mdPsum_cur = cute.domain_offset((seqlen.padded_offset_q,), mdPsum[head_idx, None])
        gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.head_dim_padded), (None, 0))
        gdO = cute.local_tile(mdO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
        gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (None,))
        gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (None,))
        if const_expr(self.fuse_dq):
            # Varlen: the sequence's padded offset (whole Q tiles) goes into the tile index; as an
            # element offset it adds address arithmetic to every dQ reduction.
            params.dq_tile_offset = Int32(0)
            if const_expr(self.direct_dq):
                params.gdQ = cute.local_tile(
                    mdQaccum[batch_idx, None, head_idx, None],
                    (self.m_block_size, self.head_dim_padded),
                    (None, 0),
                )
                params.dq_rows = seqlen.seqlen_q
                params.dq_scale = softmax_scale
                if const_expr(self.kv_sweep):
                    params.gdQpart = cute.local_tile(
                        mdQpart[batch_idx, head_idx, None],
                        (self.m_block_size * self.dq_accum_hdim,),
                        (None,),
                    )
                    params.dq_first = sweep == 0
                    params.dq_last = sweep + 1 == n_sweeps
            else:
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
                else:
                    mdQaccum_cur = mdQaccum[head_idx, None]
                    params.dq_tile_offset = seqlen.padded_offset_q // self.m_block_size
                params.gdQaccum = cute.local_tile(
                    mdQaccum_cur, (self.m_block_size * self.dq_accum_hdim,), (None,)
                )
        gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
        gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
        gmem_thr_copy_lse = gmem_tiled_copy_LSE.get_slice(tidx)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQcQ = gmem_thr_copy_QK.partition_S(cQ)
        t0QcQ = gmem_thr_copy_QK.get_slice(0).partition_S(cQ)
        if const_expr(self.head_dim_padded == self.head_dim_v_padded):
            tdOcdO, t0dOcdO = tQcQ, t0QcQ
        else:
            cdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tdOcdO = gmem_thr_copy_VdO.partition_S(cdO)
            t0dOcdO = gmem_thr_copy_VdO.get_slice(0).partition_S(cdO)
        tLSEcLSE = gmem_thr_copy_lse.partition_S(cute.make_identity_tensor((self.m_block_size,)))
        tQpQ = utils.predicate_k(tQcQ, limit=d_head)
        tdOpdO = (
            tQpQ if const_expr(self.same_hdim_kv) else utils.predicate_k(tdOcdO, limit=d_head_v)
        )
        load_Q_LSE = partial(
            self.load_Q_LSE,
            gmem_tiled_copy_QK,
            gmem_tiled_copy_LSE,
            gmem_thr_copy_QK.partition_S(gQ),
            gmem_thr_copy_QK.partition_D(sQ),
            tQcQ,
            t0QcQ,
            tQpQ,
            gmem_thr_copy_lse.partition_S(gLSE),
            gmem_thr_copy_lse.partition_D(sLSE),
            tLSEcLSE,
            seqlen=seqlen.seqlen_q,
        )
        load_dO_dPsum = partial(
            self.load_dO_dPsum,
            gmem_tiled_copy_VdO,
            gmem_tiled_copy_LSE,
            gmem_thr_copy_VdO.partition_S(gdO),
            gmem_thr_copy_VdO.partition_D(sdO),
            tdOcdO,
            t0dOcdO,
            tdOpdO,
            gmem_thr_copy_lse.partition_S(gdPsum),
            gmem_thr_copy_lse.partition_D(sdPsum),
            tLSEcLSE,
            seqlen=seqlen.seqlen_q,
        )
        mask_fn = partial(
            mask.apply_mask,
            n_block=n_block,
            thr_mma=params.thr_mma_sdp,
            batch_idx=batch_idx,
            head_idx=head_idx,
            mask_seqlen=True,
            mask_causal=self.is_causal,
            mask_local=self.is_local,
            mask_mod=self.mask_mod,
            aux_data=params.aux_data,
            fastdiv_mods=params.fastdiv_mods,
        )
        sparse_tile, mask_fn_full = None, None
        if const_expr(sparse_q is not None):
            sparse_tile = partial(self._sparse_tile, *sparse_q)
            load_Q_LSE = partial(self._load_sparse, load_Q_LSE, sparse_tile)
            load_dO_dPsum = partial(self._load_sparse, load_dO_dPsum, sparse_tile)
            # Full blocks: no mask_mod (the seqlen / causal / local masks still apply).
            mask_fn_full = partial(mask_fn, mask_mod=None)
        if const_expr(self.tma_ws):
            # The K / V copies (a KV tile's first head only) are this loop's only cp.asyncs.
            cute.arch.cp_async_wait_group(0)
        # The previous head's last Q tile is done with every stage before they are refilled.
        self._cbar()
        m_block = m_block_min
        is_a = cute.arch.make_warp_uniform(cute.arch.warp_idx()) < self.num_threads // 64
        for stage in cutlass.range_constexpr(0 if const_expr(self.tma_ws) else self.num_stages_Q):
            if const_expr(self.num_stages_Q == 1 or stage < self.num_stages_Q - 1):
                if stage == 0 or m_block + stage < m_block_max:
                    load_Q_LSE(m_block + stage, smem_pipe_write_q=stage)
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages_dO):
                if stage == 0 or m_block + stage < m_block_max:
                    load_dO_dPsum(m_block + stage, smem_pipe_write_q=stage)
                cute.arch.cp_async_commit_group()
        smem_pipe_read_q = Int32(0)
        smem_pipe_read_do = Int32(0)
        smem_pipe_write_q = Int32(self.num_stages_Q - 1)
        smem_pipe_write_do = Int32(0)
        score_ctx = (n_block, batch_idx, head_idx, seqlen)
        for m_tile in cutlass.range(m_block_min, m_block_max, unroll=1):
            self.dkdv_one_m_block(
                m_tile,
                smem_pipe_read_q,
                smem_pipe_read_do,
                smem_pipe_write_q,
                smem_pipe_write_do,
                params,
                load_Q_LSE,
                load_dO_dPsum,
                m_block_max,
                softmax_scale_log2,
                mask_fn,
                softmax_scale,
                score_ctx,
                m_block_min,
                is_a,
                sparse_tile,
                mask_fn_full,
            )
            smem_pipe_read_q = self.advance_pipeline(smem_pipe_read_q, self.num_stages_Q)
            smem_pipe_read_do = self.advance_pipeline(smem_pipe_read_do, self.num_stages_dO)
            smem_pipe_write_q = self.advance_pipeline(smem_pipe_write_q, self.num_stages_Q)
            smem_pipe_write_do = self.advance_pipeline(smem_pipe_write_do, self.num_stages_dO)
        if const_expr(self.tma_ws):
            if const_expr(self.kv_sweep):
                if notify_head:
                    cute.arch.barrier_arrive(
                        barrier_id=_TW_HEAD, number_of_threads=self.num_threads + 32
                    )
            else:
                cute.arch.barrier_arrive(
                    barrier_id=_TW_HEAD, number_of_threads=self.num_threads + 32
                )

    @cute.jit
    def tma_producer(
        self,
        mbar: cute.Pointer,
        tma_qdo: tuple,
        tma_stats: tuple,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        batch_idx: Int32,
        seqlen: SeqlenInfoQK,
        head_idx_kv: Int32,
        split_idx: Int32,
        m_block_min: Int32,
        m_block_max: Int32,
        n_block: Int32,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        """Producer warp: Q / LSE and dO / dPsum of every Q tile of every head, by TMA.

        Each tile is published on the reading group's named barrier once its transaction
        barrier completes; the consumers never wait on an mbarrier.
        """
        atom_Q, tma_Q, atom_dO, tma_dO = tma_qdo
        atom_LSE, tma_LSE, atom_dPsum, tma_dPsum = tma_stats
        m = self.m_block_size
        heads_per_split = self.qhead_per_kvhead // self.head_split
        n_sweeps = 1
        if const_expr(self.kv_sweep):
            n_sweeps = cute.ceil_div(seqlen.seqlen_k, self.n_block_size)
        # Block sparsity: whether a previous head initialized the transaction barriers (heads
        # with an empty Q list are skipped by producer and consumers alike).
        inited = Int32(0)
        for unit in cutlass.range(heads_per_split * n_sweeps, unroll=1):
            head_in_group = unit % heads_per_split if const_expr(self.kv_sweep) else unit
            head_idx = (
                head_idx_kv * self.qhead_per_kvhead + split_idx * heads_per_split + head_in_group
            )
            gQ, gdO = [
                cute.local_tile(
                    seqlen.offset_batch_Q(t, batch_idx, dim=3)[None, None, head_idx],
                    (m, hd),
                    (None, 0),
                )
                for t, hd in ((tma_Q, self.head_dim_padded), (tma_dO, self.head_dim_v_padded))
            ]
            gLSE, gdPsum = [
                cute.local_tile(
                    seqlen.offset_batch_Q(t, batch_idx, dim=2, padded=True)[None, head_idx],
                    (m,),
                    (None,),
                )
                for t in (tma_LSE, tma_dPsum)
            ]
            copy_Q, _, _ = copy_utils.tma_get_copy_fn(atom_Q, 0, cute.make_layout(1), gQ, sQ)
            copy_dO, _, _ = copy_utils.tma_get_copy_fn(atom_dO, 0, cute.make_layout(1), gdO, sdO)
            copy_LSE, _, _ = copy_utils.tma_get_copy_fn(
                atom_LSE, 0, cute.make_layout(1), gLSE, sLSE
            )
            copy_dPsum, _, _ = copy_utils.tma_get_copy_fn(
                atom_dPsum, 0, cute.make_layout(1), gdPsum, sdPsum
            )
            copies = (copy_Q, copy_LSE, copy_dO, copy_dPsum)
            if const_expr(not self.block_sparse):
                self._produce_head(
                    mbar, copies, m_block_min, m_block_max - m_block_min, None, unit > 0
                )
            else:
                idx_p, idx_f, n_partial, n_iter = self._sparse_q_list(
                    blocksparse_tensors, batch_idx, head_idx, n_block, m_block_max
                )
                if n_iter > 0:
                    sparse_tile = partial(self._sparse_tile, idx_p, idx_f, n_partial)
                    self._produce_head(mbar, copies, Int32(0), n_iter, sparse_tile, inited > 0)
                    inited = Int32(1)

    def _src_tile(self, m_block_min, sparse_tile, j):
        return m_block_min + j if const_expr(sparse_tile is None) else sparse_tile(j)[0]

    @staticmethod
    def _acquire_from_consumers(barrier_id, number_of_threads):
        """Wait until the consumers release a shared-memory stage, then order their reads of it
        (ldmatrix, generic proxy) before this warp's TMA refill (async proxy)."""
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=number_of_threads)
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def _produce_head(self, mbar, copies, m_block_min, n_iter, sparse_tile, reinit):
        """One head's n_iter Q / dO tiles (Q tile m_block_min + j, or sparse_tile(j))."""
        copy_Q, copy_LSE, copy_dO, copy_dPsum = copies
        m = self.m_block_size
        q_bytes = (m * self.head_dim_padded) * self.dtype.width // 8 + m * 4
        do_bytes = (m * self.head_dim_v_padded) * self.dtype.width // 8 + m * 4
        ready = self.num_threads // 2 + 32
        empty = self.num_threads + 32
        lane = cute.arch.lane_idx()
        # Fresh transaction barriers per (KV tile, head): every phase of the previous one
        # was waited.
        if lane == 0:
            if reinit:
                for b in cutlass.range_constexpr(3):
                    mbarrier_inval(mbar + b)
            for b in cutlass.range_constexpr(3):
                cute.arch.mbarrier_init(mbar + b, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_warp()
        # Q(0) / dO(0). The TMA copies elect their issuing thread themselves; they must be
        # called by the whole warp.
        src0 = self._src_tile(m_block_min, sparse_tile, Int32(0))
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar + 0, q_bytes)
        copy_Q(src_idx=src0, dst_idx=0, tma_bar_ptr=mbar + 0)
        copy_LSE(src_idx=src0, dst_idx=0, tma_bar_ptr=mbar + 0)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar + 2, do_bytes)
        copy_dO(src_idx=src0, dst_idx=0, tma_bar_ptr=mbar + 2)
        copy_dPsum(src_idx=src0, dst_idx=0, tma_bar_ptr=mbar + 2)
        cute.arch.mbarrier_wait(mbar + 0, 0)
        cute.arch.barrier_arrive(barrier_id=_TW_QR_A, number_of_threads=ready)
        cute.arch.barrier_arrive(barrier_id=_TW_QR_B, number_of_threads=ready)
        cute.arch.mbarrier_wait(mbar + 2, 0)
        cute.arch.barrier_arrive(barrier_id=_TW_DR_A, number_of_threads=ready)
        cute.arch.barrier_arrive(barrier_id=_TW_DR_B, number_of_threads=ready)
        for j in cutlass.range(1, n_iter, unroll=1):
            stage = j % 2
            src_j = self._src_tile(m_block_min, sparse_tile, j)
            if j >= 2:
                # Both groups' dK(j - 2) are done with this Q stage.
                self._acquire_from_consumers(_TW_QE + stage, empty)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + stage, q_bytes)
            copy_Q(src_idx=src_j, dst_idx=stage, tma_bar_ptr=mbar + stage)
            copy_LSE(src_idx=src_j, dst_idx=stage, tma_bar_ptr=mbar + stage)
            # Both groups' dV(j - 1) are done with dO.
            self._acquire_from_consumers(_TW_DE, empty)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + 2, do_bytes)
            copy_dO(src_idx=src_j, dst_idx=0, tma_bar_ptr=mbar + 2)
            copy_dPsum(src_idx=src_j, dst_idx=0, tma_bar_ptr=mbar + 2)
            cute.arch.mbarrier_wait(mbar + stage, (j // 2) % 2)
            cute.arch.barrier_arrive(barrier_id=_TW_QR_A + stage, number_of_threads=ready)
            cute.arch.barrier_arrive(barrier_id=_TW_QR_B + stage, number_of_threads=ready)
            cute.arch.mbarrier_wait(mbar + 2, j % 2)
            cute.arch.barrier_arrive(barrier_id=_TW_DR_A, number_of_threads=ready)
            cute.arch.barrier_arrive(barrier_id=_TW_DR_B, number_of_threads=ready)
        self._acquire_from_consumers(_TW_HEAD, empty)

    @cute.jit
    def _reduce_dKV(
        self,
        acc: cute.Tensor,
        mX_cur: cute.Tensor,
        thr_mma: cute.ThrMma,
        n_block: Int32,
        tile_cols: cutlass.Constexpr[int],
        row_stride: cutlass.Constexpr[int],
    ):
        """Add a (tile_n, hdim) accumulator into its row-major fp32 tile of mX_cur."""
        gX = cute.local_tile(mX_cur, (self.n_block_size * row_stride,), (n_block,))
        tXcX = thr_mma.partition_C(cute.make_identity_tensor((self.n_block_size, tile_cols)))
        # C fragments hold column pairs: values 2i and 2i + 1 are (row, col), (row, col + 1).
        for i in cutlass.range_constexpr(cute.size(acc) // 2):
            row, col = tXcX[2 * i]
            red_add_v2_f32(
                utils.elem_pointer(gX, (row * row_stride + col,)), acc[2 * i], acc[2 * i + 1]
            )

    @cute.jit
    def _store_dKV_scaled(
        self,
        acc_dK,
        acc_dV,
        mdK_cur,
        mdV_cur,
        sK,
        sV,
        gmem_tiled_copy_dKV,
        tiled_mma_dkv,
        tidx,
        n_block,
        seqlen,
        d_head,
        d_head_v,
        softmax_scale,
    ):
        acc_dK.store(acc_dK.load() * softmax_scale)
        self._store_dKV(
            acc_dK,
            mdK_cur,
            sK,
            gmem_tiled_copy_dKV,
            tiled_mma_dkv,
            tidx,
            n_block,
            seqlen,
            self.head_dim_padded,
            d_head,
            self.check_hdim_oob,
        )
        self._store_dKV(
            acc_dV,
            mdV_cur,
            sV,
            gmem_tiled_copy_dKV,
            tiled_mma_dkv,
            tidx,
            n_block,
            seqlen,
            self.head_dim_v_padded,
            d_head_v,
            self.check_hdim_v_oob,
        )

    @cute.jit
    def _tail_fixup(
        self,
        acc_dK,
        acc_dV,
        thr_mma,
        mTailK_i,
        mTailV_i,
        mTailCount,
        tail_i,
        tail_part,
        sLSE,
        tidx,
    ):
        """Store this part's dK / dV in its own fp32 slot of the tile's scratch; 1 for the last
        part to arrive, whose accumulators then hold the sum (unscaled). The sum runs over the
        slots in part order, whatever the arrival order, so the result is deterministic."""
        self._store_dKV_part(
            acc_dK, mTailK_i, thr_mma, tail_part, self.head_dim_padded, self.dq_accum_hdim
        )
        self._store_dKV_part(
            acc_dV, mTailV_i, thr_mma, tail_part, self.head_dim_v_padded, self.dv_accum_hdim
        )
        cute.arch.fence_acq_rel_gpu()
        self._cbar()
        if tidx == 0:
            old = cute.arch.atomic_add(
                utils.elem_pointer(mTailCount, (tail_i,)).llvm_ptr,
                Int32(1),
                sem="acq_rel",
                scope="gpu",
            )
            sLSE[0, 0] = Float32(old)
        self._cbar()
        last = Int32(0)
        if sLSE[0, 0] == Float32(self.tail_split - 1):
            last = Int32(1)
            cute.arch.fence_acq_rel_gpu()
            self._sum_dKV_parts(
                acc_dK, mTailK_i, thr_mma, self.head_dim_padded, self.dq_accum_hdim
            )
            self._sum_dKV_parts(
                acc_dV, mTailV_i, thr_mma, self.head_dim_v_padded, self.dv_accum_hdim
            )
        # sLSE is reused by the next head / tile only after the store's barriers.
        return last

    @cute.jit
    def _slot_offset(self, row, col, row_stride: cutlass.Constexpr[int]):
        """Offset of (row, col) in a tail scratch slot. The slot stores the tile as contiguous
        8 x 8 blocks, the span one warp's accumulator pairs cover, so each warp access is one
        contiguous run instead of eight row pieces."""
        return ((row // 8) * (row_stride // 8) + col // 8) * 64 + (row % 8) * 8 + col % 8

    @cute.jit
    def _store_dKV_part(
        self,
        acc,
        mX_cur,
        thr_mma,
        part: Int32,
        tile_cols: cutlass.Constexpr[int],
        row_stride: cutlass.Constexpr[int],
    ):
        """Store a (tile_n, hdim) accumulator into fp32 slot `part` of mX_cur."""
        gX = cute.local_tile(mX_cur, (self.n_block_size * row_stride,), (part,))
        tXcX = thr_mma.partition_C(cute.make_identity_tensor((self.n_block_size, tile_cols)))
        for i in cutlass.range_constexpr(cute.size(acc) // 2):
            row, col = tXcX[2 * i]
            ptr = utils.elem_pointer(gX, (self._slot_offset(row, col, row_stride),))
            st_v2_f32(ptr, acc[2 * i], acc[2 * i + 1])

    @cute.jit
    def _sum_dKV_parts(
        self,
        acc,
        mX_cur,
        thr_mma,
        tile_cols: cutlass.Constexpr[int],
        row_stride: cutlass.Constexpr[int],
    ):
        """acc = the sum of mX_cur's tail_split slots, added in slot order (coherent at gpu
        scope)."""
        tXcX = thr_mma.partition_C(cute.make_identity_tensor((self.n_block_size, tile_cols)))
        for p in cutlass.range_constexpr(self.tail_split):
            gX = cute.local_tile(mX_cur, (self.n_block_size * row_stride,), (p,))
            for i in cutlass.range_constexpr(cute.size(acc) // 2):
                row, col = tXcX[2 * i]
                ptr = utils.elem_pointer(gX, (self._slot_offset(row, col, row_stride),))
                v = ld_relaxed_gpu_v2_f32(ptr)
                for j in cutlass.range_constexpr(2):
                    if const_expr(p == 0):
                        acc[2 * i + j] = v[j]
                    else:
                        acc[2 * i + j] = acc[2 * i + j] + v[j]

    @cute.jit
    def _store_dKV(
        self,
        acc: cute.Tensor,
        mX_cur: cute.Tensor,
        sX: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        seqlen: SeqlenInfoQK,
        tile_cols: cutlass.Constexpr[int],
        col_limit: Int32,
        check_cols: cutlass.Constexpr[bool],
    ):
        """Scale-free bf16/fp16 store of a (tile_n, hdim) accumulator through its K/V buffer."""
        rX = cute.make_fragment_like(acc, self.dtype)
        rX.store(acc.load().to(self.dtype))
        # Every warp is done with the K / V tile (and the previous store) before reuse.
        self._cbar()
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
        )
        smem_thr_copy = cute.make_tiled_copy_C(smem_copy_atom, tiled_mma).get_slice(tidx)
        cute.copy(smem_copy_atom, smem_thr_copy.retile(rX), smem_thr_copy.partition_D(sX))
        self._cbar()
        gX = cute.local_tile(mX_cur, (self.n_block_size, tile_cols), (n_block, 0))
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        tXsX = gmem_thr_copy.partition_S(sX)
        tXgX = gmem_thr_copy.partition_D(gX)
        tXrX = cute.make_fragment_like(tXgX, self.dtype)
        cute.autovec_copy(tXsX, tXrX)
        cX = cute.make_identity_tensor((self.n_block_size, tile_cols))
        tXcX = gmem_thr_copy.partition_S(cX)
        tXpX = utils.predicate_k(tXcX, limit=col_limit)
        for m in cutlass.range_constexpr(cute.size(tXrX.shape[1])):
            if tXcX[0, m, 0][0] < seqlen.seqlen_k - n_block * self.n_block_size:
                cute.copy(
                    gmem_tiled_copy,
                    tXrX[None, m, None],
                    tXgX[None, m, None],
                    pred=tXpX[None, m, None] if const_expr(check_cols) else None,
                )

    @cute.jit
    def dkdv_one_m_block(
        self,
        m_block: Int32,
        smem_pipe_read_q: Int32,
        smem_pipe_read_do: Int32,
        smem_pipe_write_q: Int32,
        smem_pipe_write_do: Int32,
        params: SimpleNamespace,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: Int32,
        softmax_scale_log2: Float32,
        mask_fn: Callable,
        softmax_scale: Float32,
        score_ctx: tuple,
        m_block_min: Int32 | None = None,
        is_a: cutlass.Boolean | None = None,
        sparse_tile: Optional[Callable] = None,
        mask_fn_full: Optional[Callable] = None,
    ):
        # m_block is the iteration index; with block sparsity m_tile is the Q tile it maps to.
        m_tile, is_full = m_block, False
        if const_expr(sparse_tile is not None):
            m_tile, is_full = sparse_tile(m_block)
        stage_q = smem_pipe_read_q if const_expr(self.num_stages_Q > 1) else 0
        tw = const_expr(self.tma_ws)
        cta, ready, empty = self.num_threads, self.num_threads // 2 + 32, self.num_threads + 32
        if const_expr(tw):
            it = m_block - m_block_min
        stage_do = smem_pipe_read_do if const_expr(self.num_stages_dO > 1) else 0

        def load_Q_next():
            m_block_next = m_block + (
                self.num_stages_Q - 1 if const_expr(self.num_stages_Q > 1) else 1
            )
            if m_block_next < m_block_max:
                load_Q_LSE(m_block_next, smem_pipe_write_q)
            cute.arch.cp_async_commit_group()

        def load_dO_next():
            if m_block + self.num_stages_dO < m_block_max:
                load_dO_dPsum(m_block + self.num_stages_dO, smem_pipe_write_do)
            cute.arch.cp_async_commit_group()

        rpart = None
        if const_expr(self.kv_sweep):
            rpart = self.dq_part_load(m_block, params)
        acc_shape = params.thr_mma_sdp.partition_shape_C((self.n_block_size, self.m_block_size))
        # S^T = K Q^T
        acc_S = cute.make_rmem_tensor(acc_shape, Float32)
        acc_S.fill(0.0)
        if const_expr(tw):
            if is_a:
                cute.arch.barrier(barrier_id=_TW_QR_A + stage_q, number_of_threads=ready)
            else:
                cute.arch.barrier(barrier_id=_TW_QR_B + stage_q, number_of_threads=ready)
                # Turn: B's S(i) after A's; also A's dQ(i - 1) is done before B's dS(i) store.
                cute.arch.barrier(barrier_id=_TW_A_S, number_of_threads=cta)
        else:
            cute.arch.cp_async_wait_group(1 if const_expr(self.num_stages_Q > 1) else 0)
            self._cbar()
        sm80_utils.gemm(
            params.tiled_mma_sdp,
            acc_S,
            params.tSrK,
            params.tSrQ,
            params.tSsK,
            params.tSsQ[None, None, None, stage_q],
            params.smem_thr_copy_KV,
            params.smem_thr_copy_QdO,
        )
        if const_expr(tw):
            if is_a:
                cute.arch.barrier_arrive(barrier_id=_TW_A_S, number_of_threads=cta)
        n_block, batch_idx, head_idx, seqlen = score_ctx
        if const_expr(self.score_mod is not None):
            acc_S_pre = cute.make_fragment_like(acc_S)
            acc_S_pre.store(acc_S.load())
            self.apply_score_mod(
                acc_S,
                params.thr_mma_sdp,
                m_tile,
                n_block,
                batch_idx,
                head_idx,
                softmax_scale,
                seqlen,
                transposed=True,
                aux_data=params.aux_data,
                fastdiv_mods=params.fastdiv_mods,
            )
        if const_expr(mask_fn_full is not None):
            if is_full:
                mask_fn_full(acc_S, m_block=m_tile)
            else:
                mask_fn(acc_S, m_block=m_tile)
        else:
            mask_fn(acc_S, m_block=m_tile)
        # (m, n) view of the transposed accumulator: rows are Q rows.
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=True)
        tLSErLSE = cute.make_fragment_like(params.tSsLSE[None, 0])
        cute.autovec_copy(params.tSsLSE[None, stage_q], tLSErLSE)
        for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
            acc_S_mn[r, None].store(
                cute.math.exp2(
                    acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r], fastmath=True
                )
            )

        # dP^T = V dO^T
        acc_dP = cute.make_rmem_tensor(acc_shape, Float32)
        acc_dP.fill(0.0)
        if const_expr(tw):
            if is_a:
                cute.arch.barrier(barrier_id=_TW_DR_A, number_of_threads=ready)
            else:
                cute.arch.barrier(barrier_id=_TW_DR_B, number_of_threads=ready)
        else:
            cute.arch.cp_async_wait_group(1 if const_expr(self.num_stages_dO > 1) else 0)
            self._cbar()
        sm80_utils.gemm(
            params.tiled_mma_sdp,
            acc_dP,
            params.tdPrV,
            params.tdPrdO,
            params.tdPsV,
            params.tdPsdO[None, None, None, stage_do],
            params.smem_thr_copy_KV,
            params.smem_thr_copy_QdO,
            hook_fn=load_Q_next if const_expr(self.num_stages_Q > 1 and not tw) else None,
        )
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=True)
        tLSErdPsum = cute.make_fragment_like(params.tSsdPsum[None, 0])
        cute.autovec_copy(params.tSsdPsum[None, stage_do], tLSErdPsum)
        for r in cutlass.range(cute.size(acc_dP_mn, mode=[0]), unroll_full=True):
            acc_dP_mn[r, None].store(
                acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r])
            )
        if const_expr(self.score_mod_bwd is not None):
            self.apply_score_mod_bwd(
                acc_dP,
                acc_S_pre,
                params.thr_mma_sdp,
                m_tile,
                n_block,
                batch_idx,
                head_idx,
                softmax_scale,
                seqlen,
                transposed=True,
                aux_data=params.aux_data,
                fastdiv_mods=params.fastdiv_mods,
            )
            self._zero_masked_dS(acc_dP, acc_S)

        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        rdS = cute.make_fragment_like(acc_dP, self.dtype)
        rdS.store(acc_dP.load().to(self.dtype))
        if const_expr(self.fuse_dq or self.pds_in_smem):
            # dS^T (and P^T) in shared memory; the previous Q tile's GEMMs finished reading
            # them before the barrier at the top of this iteration.
            r2s = params.r2s_thr_copy_dS
            if const_expr(tw):
                if is_a:
                    if it >= 1:
                        # Every warp's dQ(i - 1) is done reading dS.
                        cute.arch.barrier(
                            barrier_id=_TW_B_END + (it - 1) % 2, number_of_threads=cta
                        )
            cute.copy(r2s, r2s.retile(rdS), r2s.partition_D(params.sdSt))
            if const_expr(tw):
                if is_a:
                    cute.arch.barrier_arrive(barrier_id=_TW_A_DS, number_of_threads=cta)
                else:
                    cute.arch.barrier_arrive(barrier_id=_TW_B_DS, number_of_threads=cta)
            if const_expr(self.pds_in_smem):
                cute.copy(r2s, r2s.retile(rP), r2s.partition_D(params.sPt))
                self._cbar()

        # dV += P^T dO
        if const_expr(self.pds_in_smem):
            sm80_utils.gemm(
                params.tiled_mma_dkv,
                params.acc_dV,
                params.tdVrPt,
                params.tdVrdO,
                params.tdVsPt,
                params.tdVsdOt[None, None, None, stage_do],
                params.smem_thr_copy_PdSt,
                params.smem_thr_copy_QdOt,
            )
        else:
            sm80_utils.gemm_rs(
                params.tiled_mma_dkv,
                params.acc_dV,
                layout_utils.reshape_acc_to_frgA(rP),
                params.tdVrdO,
                params.tdVsdOt[None, None, None, stage_do],
                params.smem_thr_copy_QdOt,
            )
        # Every warp is done with this dO stage before it is refilled.
        if const_expr(tw):
            if m_block + 1 < m_block_max:
                cute.arch.barrier_arrive(barrier_id=_TW_DE, number_of_threads=empty)
        else:
            self._cbar()
        # dK += dS^T Q
        if const_expr(self.pds_in_smem):
            sm80_utils.gemm(
                params.tiled_mma_dkv,
                params.acc_dK,
                params.tdKrdSt,
                params.tdKrQ,
                params.tdKsdSt,
                params.tdKsQt[None, None, None, stage_q],
                params.smem_thr_copy_PdSt,
                params.smem_thr_copy_QdOt,
                hook_fn=load_dO_next,
            )
        else:
            sm80_utils.gemm_rs(
                params.tiled_mma_dkv,
                params.acc_dK,
                layout_utils.reshape_acc_to_frgA(rdS),
                params.tdKrQ,
                params.tdKsQt[None, None, None, stage_q],
                params.smem_thr_copy_QdOt,
                hook_fn=load_dO_next if const_expr(not tw) else None,
            )
        if const_expr(tw):
            if m_block + 2 < m_block_max:
                cute.arch.barrier_arrive(barrier_id=_TW_QE + stage_q, number_of_threads=empty)
            # Both halves of dS(i) are stored.
            if is_a:
                cute.arch.barrier(barrier_id=_TW_B_DS, number_of_threads=cta)
            else:
                cute.arch.barrier(barrier_id=_TW_A_DS, number_of_threads=cta)
        if const_expr(self.fuse_dq):
            self.dq_one_m_block(m_tile, params, rpart)
        if const_expr(tw):
            if not is_a and m_block + 1 < m_block_max:
                cute.arch.barrier_arrive(barrier_id=_TW_B_END + it % 2, number_of_threads=cta)
        if const_expr(self.num_stages_Q == 1):
            self._cbar()
            load_Q_next()

    @cute.jit
    def _sparse_q_list(self, blocksparse_tensors, batch_idx, head_idx, n_block, m_tiles):
        """(partial list, full list, partial iterations, iterations) of one head for KV tile
        n_block. A sparse block is q_subtile_factor Q-tile iterations; the lists are
        ascending (as BlockMask builds them), so only the last block of a list can reach
        past the m_tiles Q tiles, and its iterations are cut there."""
        cnt_t, idx_t, full_cnt_t, full_idx_t, *_ = blocksparse_tensors
        sparse_n = n_block // self.kv_subtile_factor
        idx = idx_t[batch_idx, head_idx, sparse_n, None]
        n_partial = self._sparse_run_iters(cnt_t[batch_idx, head_idx, sparse_n], idx, m_tiles)
        n_iter = n_partial
        full_idx = None
        if const_expr(full_cnt_t is not None):
            full_idx = full_idx_t[batch_idx, head_idx, sparse_n, None]
            n_iter = n_partial + self._sparse_run_iters(
                full_cnt_t[batch_idx, head_idx, sparse_n], full_idx, m_tiles
            )
        return idx, full_idx, n_partial, n_iter

    @cute.jit
    def _sparse_run_iters(self, cnt: Int32, idx: cute.Tensor, m_tiles: Int32) -> Int32:
        q = self.q_subtile_factor
        n = cnt
        if const_expr(q > 1):
            n = Int32(0)
            if cnt > 0:
                n = (cnt - 1) * q + cutlass.min(q, m_tiles - idx[cnt - 1] * q)
        return n

    @cute.jit
    def _sparse_tile(self, idx, full_idx, n_partial, it):
        """(Q tile, is full block) of iteration it: the partial list's tiles, then the full
        list's."""
        q = self.q_subtile_factor
        tile, is_full = Int32(0), False
        if const_expr(full_idx is None):
            tile = idx[it // q] * q + it % q
        else:
            # Each list is read only within its own count (index tensors may be compact).
            if it < n_partial:
                tile = idx[it // q] * q + it % q
            else:
                j = it - n_partial
                tile = full_idx[j // q] * q + j % q
                is_full = True
        return tile, is_full

    def _load_sparse(self, load, sparse_tile, it, smem_pipe_write_q):
        load(sparse_tile(it)[0], smem_pipe_write_q=smem_pipe_write_q)

    @cute.jit
    def dq_part_load(self, m_block: Int32, params: SimpleNamespace):
        """This thread's fp32 dQ partials of Q tile m_block (zero in the first sweep), loaded
        at the top of the Q tile so the tile's GEMMs cover the load latency."""
        chunk_shape = (self.m_block_size, self.dq_chunk)
        tdQcdQ = params.thr_mma_dq.partition_C(cute.make_identity_tensor(chunk_shape))
        n_c = cute.size(tdQcdQ)
        rpart = cute.make_rmem_tensor((self.head_dim_padded // self.dq_chunk * n_c,), Float32)
        rpart.fill(0.0)
        if not params.dq_first:
            gdQpart = params.gdQpart[None, m_block]
            for c in cutlass.range_constexpr(self.head_dim_padded // self.dq_chunk):
                for i in cutlass.range_constexpr(n_c):
                    row, col = tdQcdQ[i]
                    rpart[c * n_c + i] = gdQpart[row * self.dq_accum_hdim + c * self.dq_chunk + col]
        return rpart

    @cute.jit
    def dq_one_m_block(self, m_block: Int32, params: SimpleNamespace, rpart=None):
        """dQaccum[m_block] += dS K (direct_dq: dQ[m_block] = scale * (partials + dS K) in the
        last sweep), one dq_chunk-column piece of the head dim at a time."""
        thr_mma_dq = params.thr_mma_dq
        tdQrdS = utils.mma_make_fragment_A(params.sdS, thr_mma_dq)
        tdQsdS = params.smem_thr_copy_dS.partition_S(params.sdS)
        if const_expr(self.direct_dq):
            gdQ = params.gdQ[None, None, m_block]
            if const_expr(self.kv_sweep):
                gdQpart = params.gdQpart[None, m_block]
            row_limit = params.dq_rows - m_block * self.m_block_size
        else:
            gdQaccum = params.gdQaccum[None, m_block + params.dq_tile_offset]
        chunk_shape = (self.m_block_size, self.dq_chunk)
        tdQcdQ = thr_mma_dq.partition_C(cute.make_identity_tensor(chunk_shape))
        for c in cutlass.range_constexpr(self.head_dim_padded // self.dq_chunk):
            sKt_c = cute.local_tile(params.sKt, (self.dq_chunk, self.n_block_size), (c, 0))
            tdQrKt = utils.mma_make_fragment_B(sKt_c, thr_mma_dq)
            tdQsKt = params.smem_thr_copy_Kt.partition_S(sKt_c)
            acc_dQ = cute.make_rmem_tensor(thr_mma_dq.partition_shape_C(chunk_shape), Float32)
            acc_dQ.fill(0.0)
            sm80_utils.gemm(
                params.tiled_mma_dq,
                acc_dQ,
                tdQrdS,
                tdQrKt,
                tdQsdS,
                tdQsKt,
                params.smem_thr_copy_dS,
                params.smem_thr_copy_Kt,
            )
            # C fragments hold column pairs: values 2i and 2i + 1 are (row, col), (row, col + 1).
            for i in cutlass.range_constexpr(cute.size(acc_dQ) // 2):
                row, col = tdQcdQ[2 * i]
                is_last = True
                if const_expr(self.kv_sweep):
                    # Each thread owns the same elements in every sweep: plain loads / stores.
                    n_c = cute.size(acc_dQ)
                    acc_dQ[2 * i] = acc_dQ[2 * i] + rpart[c * n_c + 2 * i]
                    acc_dQ[2 * i + 1] = acc_dQ[2 * i + 1] + rpart[c * n_c + 2 * i + 1]
                    is_last = params.dq_last
                    if not is_last:
                        off = row * self.dq_accum_hdim + c * self.dq_chunk + col
                        gdQpart[off] = acc_dQ[2 * i]
                        gdQpart[off + 1] = acc_dQ[2 * i + 1]
                if const_expr(self.direct_dq):
                    # Complete: scale and store; rows past seqlen_q are dropped.
                    if is_last and row < row_limit:
                        gdQ[row, c * self.dq_chunk + col] = (acc_dQ[2 * i] * params.dq_scale).to(
                            self.dtype
                        )
                        gdQ[row, c * self.dq_chunk + col + 1] = (
                            acc_dQ[2 * i + 1] * params.dq_scale
                        ).to(self.dtype)
                else:
                    red_add_v2_f32(
                        utils.elem_pointer(
                            gdQaccum, (row * self.dq_accum_hdim + c * self.dq_chunk + col,)
                        ),
                        acc_dQ[2 * i],
                        acc_dQ[2 * i + 1],
                    )


class FlashAttentionBackwardDQSm120(_ScoreModSm120, FlashAttentionBackwardSm80):
    """dQ half of the split backward: one CTA per Q tile, iterating over KV tiles.

    Q, dO, LSE and dPsum stay resident while K and V stream through num_stages_Q stages.
    dS is the register A operand of dQ += dS K, and dQ is written once, scaled.
    """

    def __init__(self, *args, mask_mod: cutlass.Constexpr | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_mod = mask_mod
        num_mma_warps = self.num_threads // cute.arch.WARP_SIZE
        assert not self.SdP_swapAB and self.AtomLayoutMSdP == num_mma_warps
        assert self.AtomLayoutMdQ == num_mma_warps and not self.dQ_swapAB, (
            "dS must be the register A operand of the dQ GEMM"
        )
        assert (self.score_mod is None) == (self.score_mod_bwd is None)
        self.num_stages_KV = self.num_stages_Q

    def _setup_attributes(self):
        super()._setup_attributes()
        sK_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_padded)
        sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_v_padded)
        self.sQ_layout = cute.tile_to_shape(
            sK_layout_atom, (self.m_block_size, self.head_dim_padded), (0, 1)
        )
        self.sdO_layout = cute.tile_to_shape(
            sV_layout_atom, (self.m_block_size, self.head_dim_v_padded), (0, 1)
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.n_block_size, self.head_dim_padded, self.num_stages_KV), (0, 1, 2)
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.n_block_size, self.head_dim_v_padded, self.num_stages_KV),
            (0, 1, 2),
        )
        self.sLSE_layout = cute.make_layout((self.m_block_size, 1))
        # dQ is a bf16/fp16 per-Q-head output even for GQA, where the parent switches the
        # dK/dV copies to fp32 accumulators.
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        tQ_shape_dim_1 = sK_layout_atom.outer.shape[1] // async_copy_elems
        self.gmem_tiled_copy_dQ = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits
            ),
            cute.make_ordered_layout(
                (self.num_threads // tQ_shape_dim_1, tQ_shape_dim_1), order=(1, 0)
            ),
            cute.make_layout((1, async_copy_elems)),
        )

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout, self.sdO_layout)
        ]
        sLSE_struct, sdPsum_struct = [
            cute.struct.Align[cute.struct.MemRange[Float32, self.m_block_size], 128]
            for _ in range(2)
        ]

        @cute.struct
        class SharedStorage:
            sQ: sQ_struct
            sdO: sdO_struct
            sK: sK_struct
            sV: sV_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        dq_head_group: Int32 | int = 0,
        aux_data: AuxData = AuxData(),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream=None,
    ):
        from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
        from flash_attn.cute.tile_scheduler import (
            SingleTileScheduler,
            SingleTileVarlenScheduler,
            TileSchedulerArguments,
        )

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQ = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQ)
        ]
        self.varlen_q = mCuSeqlensQ is not None
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        tiled_mma_sdp, _, tiled_mma_dq = self._get_tiled_mma()
        varlen_q = const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None)
        num_head = mQ.shape[1] if const_expr(mCuSeqlensQ is not None) else mQ.shape[2]
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else (mSeqUsedQ.shape[0] if const_expr(mSeqUsedQ is not None) else mQ.shape[0])
        )
        TileScheduler = SingleTileVarlenScheduler if varlen_q else SingleTileScheduler
        seqlen_q = mQ.shape[0] if const_expr(mCuSeqlensQ is not None) else mQ.shape[1]
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(seqlen_q, self.m_block_size),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mQ.shape[cute.rank(mQ) - 1],
            headdim_v=mV.shape[cute.rank(mV) - 1],
            total_q=(
                mQ.shape[0] if const_expr(mCuSeqlensQ is not None) else mQ.shape[0] * mQ.shape[1]
            ),
            tile_shape_mn=(self.m_block_size, self.n_block_size),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, _ = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)
        seqlen_k = mK.shape[0] if const_expr(mCuSeqlensK is not None) else mK.shape[1]
        fastdiv_mods = self._aux_fastdiv_mods(aux_data, seqlen_q, seqlen_k)
        self.dq_kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQ,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            Int32(dq_head_group),
            aux_data,
            fastdiv_mods,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_VdO,
            self.gmem_tiled_copy_dQ,
            self.gmem_tiled_copy_LSE,
            tiled_mma_sdp,
            tiled_mma_dq,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def dq_kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        dq_head_group: Int32,
        aux_data: AuxData,
        fastdiv_mods,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        if const_expr(
            self.is_causal
            and mCuSeqlensQ is None
            and mCuSeqlensK is None
            and mSeqUsedQ is None
            and mSeqUsedK is None
        ):
            if dq_head_group > 0:
                # Causal Q tiles see more KV tiles the later they are: issue them longest first
                # within groups of dq_head_group (head, batch) pairs whose K / V share L2.
                num_m, num_head, num_batch = cute.arch.grid_dim()
                num_hb = num_head * num_batch
                linear = m_block + num_m * (head_idx + num_head * batch_idx)
                group_start = linear // (dq_head_group * num_m) * dq_head_group
                group_size = cutlass.min(dq_head_group, num_hb - group_start)
                in_group = linear - group_start * num_m
                m_block = num_m - 1 - in_group // group_size
                hb = group_start + in_group % group_size
                head_idx = hb % num_head
                batch_idx = hb // num_head

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mQ.shape[1],
                mK.shape[1],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                tile_m=self.m_block_size,
                tile_n=self.n_block_size,
            )
            block_info = BlockInfo(
                self.m_block_size,
                self.n_block_size,
                self.is_causal,
                self.is_local,
                False,
                window_size_left,
                window_size_right,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            if const_expr(not seqlen.has_cu_seqlens_q):
                mQ_cur = mQ[batch_idx, None, head_idx, None]
                mdO_cur = mdO[batch_idx, None, head_idx, None]
                mdQ_cur = mdQ[batch_idx, None, head_idx, None]
                mLSE_cur = mLSE[batch_idx, head_idx, None]
                mdPsum_cur = mdPsum[batch_idx, head_idx, None]
            else:
                padded_offset_q = seqlen.padded_offset_q
                mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, head_idx, None])
                mdO_cur = cute.domain_offset((seqlen.offset_q, 0), mdO[None, head_idx, None])
                mdQ_cur = cute.domain_offset((seqlen.offset_q, 0), mdQ[None, head_idx, None])
                mLSE_cur = cute.domain_offset((padded_offset_q,), mLSE[head_idx, None])
                mdPsum_cur = cute.domain_offset((padded_offset_q,), mdPsum[head_idx, None])
            head_idx_kv = head_idx // self.qhead_per_kvhead
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur, mV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mK, mV)]
            else:
                mK_cur, mV_cur = [
                    cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None])
                    for t in (mK, mV)
                ]
            gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.head_dim_padded), (m_block, 0))
            gdO = cute.local_tile(
                mdO_cur, (self.m_block_size, self.head_dim_v_padded), (m_block, 0)
            )
            gdQ = cute.local_tile(mdQ_cur, (self.m_block_size, self.head_dim_padded), (m_block, 0))
            gK = cute.local_tile(mK_cur, (self.n_block_size, self.head_dim_padded), (None, 0))
            gV = cute.local_tile(mV_cur, (self.n_block_size, self.head_dim_v_padded), (None, 0))
            gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (m_block,))

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sdO = storage.sdO.get_tensor(sdO_layout)
            sK = storage.sK.get_tensor(sK_layout)
            sV = storage.sV.get_tensor(sV_layout)
            sLSE = storage.sLSE.get_tensor(cute.make_layout((self.m_block_size, 1)))
            sdPsum = storage.sdPsum.get_tensor(cute.make_layout((self.m_block_size, 1)))
            sKt = layout_utils.transpose_view(sK)

            gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
            gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
            tQgQ, tQsQ = gmem_thr_copy_QK.partition_S(gQ), gmem_thr_copy_QK.partition_D(sQ)
            tdOgdO, tdOsdO = gmem_thr_copy_VdO.partition_S(gdO), gmem_thr_copy_VdO.partition_D(sdO)
            tKgK, tKsK = gmem_thr_copy_QK.partition_S(gK), gmem_thr_copy_QK.partition_D(sK)
            tVgV, tVsV = gmem_thr_copy_VdO.partition_S(gV), gmem_thr_copy_VdO.partition_D(sV)
            d_head = mQ.shape[cute.rank(mQ) - 1]
            d_head_v = mdO.shape[cute.rank(mdO) - 1]

            # Q, dO, LSE, dPsum: loaded once, rows past seqlen_q zero-filled.
            self._load_rows(
                gmem_thr_copy_QK,
                tQgQ,
                tQsQ,
                self.m_block_size,
                self.head_dim_padded,
                seqlen.seqlen_q - m_block * self.m_block_size,
                d_head,
                self.check_hdim_oob,
            )
            self._load_rows(
                gmem_thr_copy_VdO,
                tdOgdO,
                tdOsdO,
                self.m_block_size,
                self.head_dim_v_padded,
                seqlen.seqlen_q - m_block * self.m_block_size,
                d_head_v,
                self.check_hdim_v_oob,
            )
            # LSE / dPsum are padded to tile_m by the preprocess kernel.
            for i in cutlass.range(tidx, self.m_block_size, self.num_threads, unroll=1):
                sLSE[i, 0] = gLSE[i]
                sdPsum[i, 0] = gdPsum[i]
            cute.arch.cp_async_commit_group()

            thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
            thr_mma_dq = tiled_mma_dq.get_slice(tidx)
            acc_dQ = cute.make_rmem_tensor(
                thr_mma_dq.partition_shape_C((self.m_block_size, self.head_dim_padded)), Float32
            )
            acc_dQ.fill(0.0)
            tSrQ = utils.mma_make_fragment_A(sQ, thr_mma_sdp)
            tSrK = utils.mma_make_fragment_B(sK[None, None, 0], thr_mma_sdp)
            tdPrdO = utils.mma_make_fragment_A(sdO, thr_mma_sdp)
            tdPrV = utils.mma_make_fragment_B(sV[None, None, 0], thr_mma_sdp)
            tdQrKt = utils.mma_make_fragment_B(sKt[None, None, 0], thr_mma_dq)

            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
            )
            smem_copy_atom_transposed = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
            )
            smem_thr_copy_QdO = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_sdp).get_slice(
                tidx
            )
            smem_thr_copy_KV = utils.make_tiled_copy_B(smem_copy_atom, tiled_mma_sdp).get_slice(
                tidx
            )
            smem_thr_copy_Kt = utils.make_tiled_copy_B(
                smem_copy_atom_transposed, tiled_mma_dq
            ).get_slice(tidx)
            tSsQ = smem_thr_copy_QdO.partition_S(sQ)
            tdPsdO = smem_thr_copy_QdO.partition_S(sdO)
            tSsK = smem_thr_copy_KV.partition_S(sK)
            tdPsV = smem_thr_copy_KV.partition_S(sV)
            tdQsKt = smem_thr_copy_Kt.partition_S(sKt)
            tSsLSE = layout_utils.mma_partition_C_vec(
                sLSE, thr_mma_sdp, expand_shape=self.n_block_size, is_colvec=True
            )
            tSsdPsum = layout_utils.mma_partition_C_vec(
                sdPsum, thr_mma_sdp, expand_shape=self.n_block_size, is_colvec=True
            )

            load_KV = partial(
                self._load_KV,
                gmem_thr_copy_QK,
                gmem_thr_copy_VdO,
                tKgK,
                tKsK,
                tVgV,
                tVsV,
                seqlen_k=seqlen.seqlen_k,
                d_head=d_head,
                d_head_v=d_head_v,
            )
            mask = AttentionMask(
                self.m_block_size,
                self.n_block_size,
                seqlen,
                window_size_left,
                window_size_right,
            )
            if n_block_min < n_block_max:
                for stage in cutlass.range_constexpr(self.num_stages_KV - 1):
                    if n_block_min + stage < n_block_max:
                        load_KV(n_block_min + stage, stage)
                    cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(max(self.num_stages_KV - 1, 0))
                cute.arch.barrier()
                tLSErLSE = cute.make_fragment_like(tSsLSE[None, 0])
                cute.autovec_copy(tSsLSE[None, 0], tLSErLSE)
                tLSErdPsum = cute.make_fragment_like(tSsdPsum[None, 0])
                cute.autovec_copy(tSsdPsum[None, 0], tLSErdPsum)

                stage = Int32(0)
                for n_block in cutlass.range(n_block_min, n_block_max, unroll=1):
                    if const_expr(self.num_stages_KV == 1):
                        # Single buffer: every warp is done with the previous K / V first.
                        cute.arch.barrier()
                        load_KV(n_block, 0)
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.barrier()
                    else:
                        # K / V of this stage are resident for every warp, and every warp
                        # is done with the stage the next prefetch overwrites.
                        cute.arch.cp_async_wait_group(self.num_stages_KV - 2)
                        cute.arch.barrier()
                        n_block_next = n_block + self.num_stages_KV - 1
                        stage_next = (stage + self.num_stages_KV - 1) % self.num_stages_KV
                        if n_block_next < n_block_max:
                            load_KV(n_block_next, stage_next)
                        cute.arch.cp_async_commit_group()

                    acc_shape = thr_mma_sdp.partition_shape_C(
                        (self.m_block_size, self.n_block_size)
                    )
                    acc_S = cute.make_rmem_tensor(acc_shape, Float32)
                    acc_S.fill(0.0)
                    sm80_utils.gemm(
                        tiled_mma_sdp,
                        acc_S,
                        tSrQ,
                        tSrK,
                        tSsQ,
                        tSsK[None, None, None, stage],
                        smem_thr_copy_QdO,
                        smem_thr_copy_KV,
                    )
                    if const_expr(self.score_mod is not None):
                        acc_S_pre = cute.make_fragment_like(acc_S)
                        acc_S_pre.store(acc_S.load())
                        self.apply_score_mod(
                            acc_S,
                            thr_mma_sdp,
                            m_block,
                            n_block,
                            batch_idx,
                            head_idx,
                            softmax_scale,
                            seqlen,
                            transposed=False,
                            aux_data=aux_data,
                            fastdiv_mods=fastdiv_mods,
                        )
                    mask.apply_mask(
                        acc_S,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        m_block=m_block,
                        n_block=n_block,
                        thr_mma=thr_mma_sdp,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        mask_mod=self.mask_mod,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                    )
                    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
                    for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
                        acc_S_mn[r, None].store(
                            cute.math.exp2(
                                acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r],
                                fastmath=True,
                            )
                        )
                    acc_dP = cute.make_rmem_tensor(acc_shape, Float32)
                    acc_dP.fill(0.0)
                    sm80_utils.gemm(
                        tiled_mma_sdp,
                        acc_dP,
                        tdPrdO,
                        tdPrV,
                        tdPsdO,
                        tdPsV[None, None, None, stage],
                        smem_thr_copy_QdO,
                        smem_thr_copy_KV,
                    )
                    acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
                    for r in cutlass.range(cute.size(acc_dP_mn, mode=[0]), unroll_full=True):
                        acc_dP_mn[r, None].store(
                            acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r])
                        )
                    if const_expr(self.score_mod_bwd is not None):
                        self.apply_score_mod_bwd(
                            acc_dP,
                            acc_S_pre,
                            thr_mma_sdp,
                            m_block,
                            n_block,
                            batch_idx,
                            head_idx,
                            softmax_scale,
                            seqlen,
                            transposed=False,
                            aux_data=aux_data,
                            fastdiv_mods=fastdiv_mods,
                        )
                        self._zero_masked_dS(acc_dP, acc_S)
                    rdS = cute.make_fragment_like(acc_dP, self.dtype)
                    rdS.store(acc_dP.load().to(self.dtype))
                    # dQ += dS K
                    sm80_utils.gemm_rs(
                        tiled_mma_dq,
                        acc_dQ,
                        layout_utils.reshape_acc_to_frgA(rdS),
                        tdQrKt,
                        tdQsKt[None, None, None, stage],
                        smem_thr_copy_Kt,
                    )
                    stage = stage + 1 if stage < self.num_stages_KV - 1 else 0
            # The output reuses sQ: the Q / dO copies issued above must land first, also for
            # Q tiles that see no KV tile and skip the loop (else a late copy overwrites dQ).
            cute.arch.cp_async_wait_group(0)

            acc_dQ.store(acc_dQ.load() * softmax_scale)
            self._store_dQ(
                acc_dQ,
                gdQ,
                sQ,
                gmem_tiled_copy_dQ,
                tiled_mma_dq,
                tidx,
                seqlen.seqlen_q - m_block * self.m_block_size,
                d_head,
            )

    @cute.jit
    def _load_KV(
        self,
        gmem_thr_copy_QK: cute.TiledCopy,
        gmem_thr_copy_VdO: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        n_block: Int32,
        stage: Int32,
        seqlen_k: Int32,
        d_head: Int32,
        d_head_v: Int32,
    ):
        limit = seqlen_k - n_block * self.n_block_size
        self._load_rows(
            gmem_thr_copy_QK,
            tKgK[None, None, None, n_block],
            tKsK[None, None, None, stage],
            self.n_block_size,
            self.head_dim_padded,
            limit,
            d_head,
            self.check_hdim_oob,
        )
        self._load_rows(
            gmem_thr_copy_VdO,
            tVgV[None, None, None, n_block],
            tVsV[None, None, None, stage],
            self.n_block_size,
            self.head_dim_v_padded,
            limit,
            d_head_v,
            self.check_hdim_v_oob,
        )

    @cute.jit
    def _load_rows(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        tile_rows: cutlass.Constexpr[int],
        tile_cols: cutlass.Constexpr[int],
        row_limit: Int32,
        col_limit: Int32,
        check_cols: cutlass.Constexpr[bool],
    ):
        """cp.async a (rows, cols) tile, zero-filling rows >= row_limit and cols >= col_limit."""
        cX = cute.make_identity_tensor((tile_rows, tile_cols))
        tXcX = gmem_thr_copy.partition_S(cX)
        tXpX = utils.predicate_k(tXcX, limit=col_limit)
        for m in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
            if tXcX[0, m, 0][0] < tile_rows:
                predicate_m = tXcX[0, m, 0][0] < row_limit
                predicate = cute.make_fragment_like(tXpX[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (
                            tXpX[i, m, k] if const_expr(check_cols) else True
                        ) and predicate_m
                cute.copy(gmem_thr_copy, tXgX[None, m, None], tXsX[None, m, None], pred=predicate)

    @cute.jit
    def _store_dQ(
        self,
        acc_dQ: cute.Tensor,
        gdQ: cute.Tensor,
        sdQ: cute.Tensor,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        row_limit: Int32,
        d_head: Int32,
    ):
        rdQ = cute.make_fragment_like(acc_dQ, self.dtype)
        rdQ.store(acc_dQ.load().to(self.dtype))
        # Every warp is done reading sQ before it is reused for the output.
        cute.arch.barrier()
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
        )
        smem_thr_copy = cute.make_tiled_copy_C(smem_copy_atom, tiled_mma).get_slice(tidx)
        cute.copy(smem_copy_atom, smem_thr_copy.retile(rdQ), smem_thr_copy.partition_D(sdQ))
        cute.arch.barrier()
        gmem_thr_copy = gmem_tiled_copy_dQ.get_slice(tidx)
        tdQsdQ = gmem_thr_copy.partition_S(sdQ)
        tdQgdQ = gmem_thr_copy.partition_D(gdQ)
        tdQrdQ = cute.make_fragment_like(tdQgdQ, self.dtype)
        cute.autovec_copy(tdQsdQ, tdQrdQ)
        cdQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tdQcdQ = gmem_thr_copy.partition_S(cdQ)
        tdQpdQ = utils.predicate_k(tdQcdQ, limit=d_head)
        for m in cutlass.range_constexpr(cute.size(tdQrdQ.shape[1])):
            if tdQcdQ[0, m, 0][0] < row_limit:
                cute.copy(
                    gmem_tiled_copy_dQ,
                    tdQrdQ[None, m, None],
                    tdQgdQ[None, m, None],
                    pred=tdQpdQ[None, m, None] if const_expr(self.check_hdim_oob) else None,
                )
