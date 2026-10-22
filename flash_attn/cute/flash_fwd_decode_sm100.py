# Native private core adapted from the reviewer ReusableSwapAB v3 prototype.
# Original prototype copied from flash_attn/cute/flash_fwd_decode_sm100.py
# at FlashAttention PR #2973, commit 10ec7dcb84f45c8c98da76e6da546e2a72234f76.
# Original kernel adapted from NVIDIA CUTLASS gqa_decode_opt:
# examples/python/CuTeDSL/cute_ext/blackwell/attention/gqa_decode_opt.py.
# The review checkout is unchanged. MMA, load, and softmax scheduling is retained;
# Shared score/mask/local helpers, head-dimension views, and normalized split outputs
# are integrated. The shared-memory
# atomic max in the online-softmax mainloop remains; no cluster/global atomic reduction,
# global max/sum buffers, custom combine, or programmatic dependent launch remains.
#
# Contract: Q is (B,H,Dqk), K is (B,S,Hkv,Dqk), V is (B,S,Hkv,Dv).
# O storage is (splits,B,H,Dv), or (B,H,Dv) when splits=1; optional LSE storage is
# (B,H,splits), fp32 natural log. Output views preserve actual strides. Empty splits
# write O=0 and, when requested, LSE=-inf. The caller launches the shared FA4 combine
# if splits>1 and guarantees fp32 O partials and an LSE buffer in that case.
# One CTA = 16 warps: 0 QK MMA, 1 PV MMA, 2 K/V load + final statistics,
# 3 Q load + O store, WGs 1/2 alternating softmax, WG3 O correction.

import math
from functools import partial
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import Float32, Int32

from cutlass import const_expr

from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors
from flash_attn.cute.pipeline import NamedBarrier
from flash_attn.cute.paged_kv import resolve_paged_token, copy_kv_tma
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import call_score_mod
from flash_attn.cute.mask import call_mask_mod
from flash_attn.cute.utils import AuxData, LOG2_E, scalar_to_ssa, ssa_to_scalar

try:  # nvidia-cutlass-dsl >= 4.8
    from cutlass.memory import get_smem_capacity_in_bytes
except ImportError:  # older releases
    from cutlass.utils import get_smem_capacity_in_bytes

exp2_fast = partial(cute.math.exp2, fastmath=True)
warp_reduce_max = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_atomic_max = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")

WARP_SIZE = 32
WARPS_PER_WG = 4
WG_SIZE = WARPS_PER_WG * WARP_SIZE
MMA_N_MIN = 8
NUM_SOFTMAX_WG = 2
NUM_WG = NUM_SOFTMAX_WG + 2


class _SwapABCore:
    def __init__(
        self,
        tile_hq,
        tile_n,
        head_dim,
        dtype,
        grid_shape,
        score_mod=None,
        mask_mod=None,
        is_local=False,
        head_dim_v=None,
    ):
        self.tile_hq = tile_hq
        self.tile_n = tile_n
        self.head_dim = head_dim
        self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
        self.dtype = dtype
        self.qk_acc_dtype = cutlass.Float32
        self.grid_shape = grid_shape
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.is_local = is_local
        self.check_empty_rows = score_mod is not None or mask_mod is not None or is_local

        assert head_dim > 0 and head_dim % 16 == 0
        assert self.head_dim_v > 0 and self.head_dim_v % 16 == 0
        assert tile_hq in (1, 2, 4, 8, 16, 32)
        assert tile_n > 0 and tile_n % 128 == 0 and tile_n <= 256

    def can_implement(self, problem_size):
        _b, num_heads, num_heads_kv, _s_k, hdim = problem_size
        if hdim != self.head_dim:
            raise ValueError(f"KV head dim ({hdim}) mismatch with {self.head_dim=}")
        if num_heads % num_heads_kv != 0:
            raise ValueError(
                f"Number of Q heads ({num_heads}) must be divisible by KV heads ({num_heads_kv})"
            )
        if hdim % 16 != 0:
            raise ValueError(f"QK head dim ({hdim}) must be divisible by 16")

    @cute.experimental.jit
    def __call__(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        mQ: cute.Tensor,  # (batch, num_heads, head_dim), head_dim contiguous
        mK: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim), head_dim contiguous
        mV: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim_v), last dim contiguous
        mO_partial: cute.Tensor,  # (num_splits, batch, num_heads, head_dim_v), or rank 3 for ns=1
        mLSE_partial: cute.Tensor | None,  # (batch, num_heads, num_splits), natural log
        mSeqUsedK: cute.Tensor | None,  # (batch,) int32: KV tokens actually used per sequence
        softmax_scale_log2: Float32,
        stream: cuda.CUstream,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
        aux_data: AuxData = AuxData(),
        mPageTable: cute.Tensor | None = None,
    ):
        block = (NUM_WG * WG_SIZE, 1, 1)
        num_splits = self.grid_shape[0]
        smem = cute.Int64(get_smem_capacity_in_bytes("sm_100"))

        hdim = self.head_dim
        hdim_v = self.head_dim_v
        batch, num_heads, num_heads_kv, _, _d = problem_size
        qhead_per_kvhead = num_heads // num_heads_kv
        tile_hq = self.tile_hq
        tile_n = self.tile_n

        # GEMM views over the FA layouts, built from the real strides (so e.g. a KV cache slice
        # works without a copy). The (kv head, batch) pair is one hierarchical mode, indexed by
        # bh_idx = batch_idx * num_heads_kv + kv_head_idx.
        # QK: K is operand A (M = tokens, K = head_dim), Q is operand B (N = query heads of this
        # KV head). PV: V^T is operand A (M = head_dim_v, K = tokens).
        cache_extent, cache_batch = mK.shape[1], mK.shape[0]
        q_sb, q_sh, _ = mQ.stride
        k_sb, k_ss, k_sh, _ = mK.stride
        v_sb, v_ss, v_sh, _ = mV.stride
        mK_qk = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (cache_extent, hdim, (num_heads_kv, cache_batch)), stride=(k_ss, 1, (k_sh, k_sb))
            ),
        )
        mQ_qk = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (qhead_per_kvhead, hdim, (num_heads_kv, batch)),
                stride=(q_sh, 1, (qhead_per_kvhead * q_sh, q_sb)),
            ),
        )
        mVt_pv = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (hdim_v, cache_extent, (num_heads_kv, cache_batch)), stride=(1, v_ss, (v_sh, v_sb))
            ),
        )
        # Normalized O can be fp16/bf16 for a direct, one-split output. With multiple
        # splits the caller passes fp32 storage and launches FlashAttentionForwardCombine.
        if cutlass.const_expr(len(mO_partial.shape) == 3):
            o_sb, o_sh, o_sd = mO_partial.stride
            o_ssplit = 0
        else:
            o_ssplit, o_sb, o_sh, o_sd = mO_partial.stride
        mO_partial_tma = cute.make_tensor(
            mO_partial.iterator,
            cute.make_layout(
                (hdim_v, qhead_per_kvhead, (num_heads_kv, batch), num_splits),
                stride=(o_sd, o_sh, (qhead_per_kvhead * o_sh, o_sb), o_ssplit),
            ),
        )
        mLSE_partial_fwd = None
        if cutlass.const_expr(mLSE_partial is not None):
            assert mLSE_partial.element_type == Float32
            l_sb, l_sh, l_ssplit = mLSE_partial.stride
            mLSE_partial_fwd = cute.make_tensor(
                mLSE_partial.iterator,
                cute.make_layout(
                    (num_splits, qhead_per_kvhead, (num_heads_kv, batch)),
                    stride=(l_ssplit, l_sh, (qhead_per_kvhead * l_sh, l_sb)),
                ),
            )

        # MMA tile: M is always 128 (even for head_dim 64) so S and O use the canonical TMEM
        # accumulator layout. The K-step is 64 for both GEMMs so that a K tile (128 tokens x 64 dims)
        # and a V^T tile (128 dims x 64 tokens) have the same size and can share one SMEM ring.
        tile_m = 128
        tile_hq_mma = max(MMA_N_MIN, tile_hq)
        tile_k = 64
        assert tile_hq_mma % tile_hq == 0
        num_s_subtiles = cute.ceil_div(tile_n, tile_m)
        num_k_tiles = cute.ceil_div(hdim, tile_k)
        num_d_tiles = cute.ceil_div(hdim_v, tile_m)
        num_pv_k_tiles = cute.ceil_div(tile_n, tile_k)
        self.mma_tiler = (tile_m, tile_hq_mma, tile_k)
        self.subtile_counts = (num_s_subtiles, num_k_tiles, num_d_tiles, num_pv_k_tiles)

        # Pipeline depths. S: 4 TMEM score buffers, P: 4 SMEM buffers, O: 2 TMEM slots (even / odd
        # blocks), L: one row-sum mailbox per softmax WG. Everything else is fixed-size, and the
        # SMEM that is left over becomes K/V ring stages (12 for head_dim 128, tile_hq 8).
        self.num_stages_q = num_k_tiles
        self.num_stages_p = 4
        self.num_stages_s = 4
        self.num_stages_l = NUM_SOFTMAX_WG
        self.num_stages_o = 2
        mbar_bits = 64
        pipe_bits_per_stage = mbar_bits * 2
        bits_mk_tile = tile_m * tile_k * self.dtype.width
        bits_nk_tile = tile_hq_mma * tile_k * self.dtype.width
        bits_mn_tile = tile_m * tile_hq_mma * self.dtype.width
        smem_bits_used = 0
        smem_bits_used += tile_hq * self.qk_acc_dtype.width
        smem_bits_used += 4 * tile_hq * self.qk_acc_dtype.width
        smem_bits_used += self.num_stages_q * bits_nk_tile + mbar_bits
        smem_bits_used += self.num_stages_p * (num_s_subtiles * bits_mn_tile + pipe_bits_per_stage)
        smem_bits_used += self.num_stages_s * pipe_bits_per_stage
        smem_bits_used += self.num_stages_o * pipe_bits_per_stage
        smem_bits_total = get_smem_capacity_in_bytes("sm_100") * 8
        smem_bits_align = 1024 - (smem_bits_used % 1024)
        smem_bits_free = smem_bits_total - smem_bits_used - smem_bits_align
        num_kv_stages = smem_bits_free // bits_mk_tile
        num_kv_stages -= 1 if num_kv_stages * pipe_bits_per_stage > smem_bits_align else 0
        assert num_kv_stages > 0
        self.num_stages_kv = num_kv_stages

        # K and Q are both contiguous along head_dim (the QK reduction dim) -> K-major.
        qk_a_major = OperandMajorMode.K
        qk_b_major = OperandMajorMode.K
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            qk_a_major,
            qk_b_major,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )

        # The PV reduction runs over tokens, which is not the contiguous dim of V nor of P -> MN-major.
        pv_a_major = OperandMajorMode.MN
        pv_b_major = OperandMajorMode.MN
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            pv_a_major,
            pv_b_major,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )

        sQ_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler, self.dtype, self.num_stages_q
        )
        sK_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler, self.dtype, self.num_stages_kv
        )
        sVt_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler, self.dtype, self.num_stages_kv
        )
        sO_layout_atom = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, mO_partial_tma.element_type
        )

        # O staging tile uses the output dtype and reuses the K/V ring SMEM at the end.
        O_shape = (num_d_tiles * tile_m, tile_hq_mma)
        sO_layout = cute.tile_to_shape(sO_layout_atom, O_shape, order=(1, 0))
        sO_layout = cute.flat_divide(sO_layout, (tile_m, tile_hq_mma))

        self.attn_kernel(
            problem_size,
            mQ_qk,
            mK_qk,
            mVt_pv,
            mO_partial_tma,
            mO_partial,
            mLSE_partial_fwd,
            mSeqUsedK,
            softmax_scale_log2,
            tiled_mma_qk,
            tiled_mma_pv,
            sQ_layout,
            sK_layout,
            sVt_layout,
            sO_layout,
            window_size_left,
            window_size_right,
            aux_data,
            mPageTable,
        ).launch(
            grid=self.grid_shape,
            block=block,
            cluster=(1, 1, 1),
            smem=smem,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.experimental.kernel
    def attn_kernel(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO_partial_tma: cute.Tensor,
        mO_partial_raw: cute.Tensor,
        mLSE_partial: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        softmax_scale_log2: Float32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sVt_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        aux_data: AuxData,
        mPageTable: cute.Tensor | None,
    ):
        # Warp roles (see header): WG0 = MMA + TMA warps, WG1 / WG2 = softmax for even / odd KV blocks,
        # WG3 = correction. Warp 2 publishes this split's LSE and reciprocal row sum.
        mma_load_wg_id = 0
        softmax_wg_ids = tuple(range(mma_load_wg_id + 1, mma_load_wg_id + 1 + NUM_SOFTMAX_WG))
        correction_wg_id = softmax_wg_ids[-1] + 1
        mma_qk_warp_id = mma_load_wg_id * WARPS_PER_WG + 0
        mma_pv_warp_id = mma_load_wg_id * WARPS_PER_WG + 1
        load_kv_warp_id = mma_load_wg_id * WARPS_PER_WG + 2
        load_q_store_o_warp_id = mma_load_wg_id * WARPS_PER_WG + 3
        split_stats_warp_id = load_kv_warp_id
        thread_id, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        wg_idx = warp_idx // WARPS_PER_WG
        wg_warp_idx = warp_idx % WARPS_PER_WG

        hdim = self.head_dim
        hdim_v = self.head_dim_v
        _b, num_heads, num_heads_kv, seqlen_k, _d = problem_size
        tile_n = self.tile_n
        tile_hq = self.tile_hq
        split_id, hq_block, bh_idx = cute.arch.block_idx()
        qhead_per_kvhead = num_heads // num_heads_kv
        seqlen_info = SeqlenInfoQK.create(
            batch_idx=bh_idx // num_heads_kv,
            seqlen_q_static=1,
            seqlen_k_static=seqlen_k,
            mSeqUsedK=mSeqUsedK,
            tile_m=1,
            tile_n=tile_n,
        )
        seqlen_k = seqlen_info.seqlen_k
        # BlockInfo supplies total bounds; cyclic split assignment stays explicit below.
        block_info = BlockInfo(
            tile_m=1,
            tile_n=tile_n,
            is_causal=False,
            is_local=self.is_local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen_info, m_block=0)
        num_n_blocks = n_block_max - n_block_min
        num_splits = cute.arch.grid_dim()[0]

        mma_tiler = self.mma_tiler
        tile_m, tile_hq_mma, tile_k = self.mma_tiler
        num_s_subtiles, num_k_tiles, num_d_tiles, num_pv_k_tiles = self.subtile_counts

        mma_kq_instr_shape_k = cute.size(tiled_mma_qk.shape_mnk, mode=[2])
        mma_vp_instr_shape_k = cute.size(tiled_mma_pv.shape_mnk, mode=[2])
        qk_k_blocks = tile_k // mma_kq_instr_shape_k
        pv_k_blocks = tile_k // mma_vp_instr_shape_k

        # With 32 heads per CTA the softmax and correction threads hold a lot of state; move registers
        # away from the MMA / TMA warps, which barely need any.
        setmaxreg = tile_hq > 16
        max_sw_regs_per_wg_thread = 256
        max_hw_regs_per_wg_thread = 64 * 1024 // WG_SIZE
        num_regs_mma_load = 64
        num_regs_softmax = 120
        num_regs_correction = min(
            max_sw_regs_per_wg_thread,
            max_hw_regs_per_wg_thread - num_regs_mma_load - num_regs_softmax * 2,
        )
        assert (
            num_regs_mma_load + num_regs_softmax * NUM_SOFTMAX_WG + num_regs_correction
        ) <= max_hw_regs_per_wg_thread

        is_empty_split = split_id >= num_n_blocks

        def smem_alloc(dtype, layout, alignment=1024):
            return cute_ext.allocate(dtype, cutlass.AddressSpace.smem, layout, alignment=alignment)

        def tmem_alloc(dtype, layout, alignment=16):
            return cute_ext.allocate(dtype, cutlass.AddressSpace.tmem, layout, alignment=alignment)

        def make_rmem(dtype, layout, alignment=32):
            return cute_ext.allocate(dtype, cutlass.AddressSpace.rmem, layout, alignment=alignment)

        sQ = smem_alloc(self.dtype, sQ_layout)

        # One SMEM ring for both K and V^T tiles (same bytes, two swizzled views).
        sKV = smem_alloc(self.dtype, sK_layout)
        sK = sKV
        sVt_iter = cute.recast_ptr(sKV.iterator, sVt_layout.inner, dtype=self.dtype)
        sVt = cute.make_tensor(sVt_iter, sVt_layout.outer)

        # The O staging tile is only written after the last PV MMA, so it can reuse the ring.
        sO_iter = cute.recast_ptr(sKV.iterator, sO_layout.inner, dtype=mO_partial_tma.element_type)
        sO = cute.make_tensor(sO_iter, sO_layout.outer)

        sP_tiler_nm = (None, tile_hq_mma, tile_n)
        sP_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, sP_tiler_nm, self.dtype, self.num_stages_p
        )
        sP = smem_alloc(self.dtype, sP_layout)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        sP_nk_tile = thr_mma_pv.partition_shape_B((tile_hq_mma, tile_k))
        sP_nk = cute.local_tile(sP, sP_nk_tile, (0, 0, None, None))

        # TMEM: S (4 stages x 2 sub-tiles of 128 tokens), L (one row-sum mailbox per softmax WG),
        # O (2 slots x ceil(head_dim_v / 128) tiles).
        tStS_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma_qk, mma_tiler, self.num_stages_s * num_s_subtiles
        )
        tLtL_layout = cute_ext.make_tmem_layout_acc(tiled_mma_qk, mma_tiler, self.num_stages_l)
        tOtO_layout = cute.make_layout(
            shape=((tile_m, tile_hq_mma), 1, 1, num_d_tiles, self.num_stages_o),
            stride=((65536, 1), 0, 0, tile_hq_mma, num_d_tiles * tile_hq_mma),
        )
        tStS = tmem_alloc(self.qk_acc_dtype, tStS_layout)
        tLtL = tmem_alloc(self.qk_acc_dtype, tLtL_layout)
        tOtO = tmem_alloc(self.qk_acc_dtype, tOtO_layout)
        tStS_staged_layout = cute.make_layout(
            shape=((tile_m, tile_hq_mma), 1, 1, num_s_subtiles, self.num_stages_s),
            stride=((65536, 1), 0, 0, tile_hq_mma, num_s_subtiles * tile_hq_mma),
        )
        tStS_staged = cute.make_tensor(tStS.iterator, tStS_staged_layout)

        tLtL_0 = tLtL[(None, None), 0, 0, 0]
        t2r_rep = tcgen05.Repetition(tile_hq)
        ld_op = tcgen05.Ld32x32bOp(t2r_rep, tcgen05.Pack.NONE)
        st_op = tcgen05.St32x32bOp(t2r_rep, tcgen05.Unpack.NONE)
        t2r_atom = cute.make_copy_atom(ld_op, self.qk_acc_dtype)
        r2t_atom = cute.make_copy_atom(st_op, self.qk_acc_dtype)
        r2s_atom_P = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        r2s_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mO_partial_tma.element_type)
        tiled_t2r = tcgen05.make_tmem_copy(t2r_atom, tLtL_0)
        tiled_r2t = tcgen05.make_tmem_copy(r2t_atom, tLtL_0)
        tiled_r2s_O = cute.make_tiled_copy_D(r2s_atom_O, tiled_t2r)
        thr_in_wg = thread_id % WG_SIZE
        thr_t2r = tiled_t2r.get_slice(thr_in_wg)
        thr_r2t = tiled_r2t.get_slice(thr_in_wg)
        thr_r2s_O = tiled_r2s_O.get_slice(thr_in_wg)
        vec_layout = thr_t2r.partition_D(tLtL_0).layout
        vec_size = cute.cosize(vec_layout)
        assert vec_size == tile_hq_mma
        assert vec_layout.shape == ((tile_hq, 1), 1, tile_hq_mma // tile_hq)

        # sRowMax: the running max per query head, shared by both softmax WGs and read by the
        # correction WG. sRowSumWarp: per-warp row sums from the correction WG for the split stats.
        sRowMax = smem_alloc(self.qk_acc_dtype, vec_layout)
        sRowSumWarp_layout = cute.make_layout(
            shape=(4, tile_hq),
            stride=(tile_hq, 1),
        )
        sRowSumWarp = smem_alloc(self.qk_acc_dtype, sRowSumWarp_layout)
        lane_owns_head = lane_idx < tile_hq
        if warp_idx == load_q_store_o_warp_id:
            if lane_owns_head:
                sRowMax[lane_idx] = -Float32.inf

        # The KV ring carries K two blocks ahead of V: K(j) is loaded together with V(j - 2).
        kv_lookahead = 2
        kv_lookahead_blocks = kv_lookahead * num_splits

        # Named barriers (hardware barriers that only count arriving threads):
        #   bar_max_full / bar_max_empty   softmax WG <-> correction WG hand-off of the running max
        #   bar_sum_full / bar_sum_empty   row-sum mailbox hand-off, one pair per softmax WG (+phase)
        #   bar_max_turn                   token passed between the two softmax WGs (see softmax)
        #   bar_qk_issued / bar_pv_issued  keep the two MMA warps in K/V ring order
        #   bar_max_done / bar_sum_done    final max / sum ready for the split-stats warp
        #   bar_o_done                     final O staged in SMEM, ready for the TMA store
        # Both roles wait at different instruction sites; use the shared unaligned helper.
        bar_max_full = NamedBarrier(2, 2 * WG_SIZE)
        bar_max_empty = pipeline.NamedBarrier(3, 2 * WG_SIZE)
        bar_sum_full = pipeline.NamedBarrier(4, 2 * WG_SIZE)
        bar_sum_empty = pipeline.NamedBarrier(6, 2 * WG_SIZE)
        bar_max_done = pipeline.NamedBarrier(8, WG_SIZE + WARP_SIZE)
        # Final-max notification and reciprocal-sum publication are different
        # rendezvous. Reusing ID 8 lets a fast correction warp arrive twice
        # before all warps have contributed to its first generation.
        bar_stats_ready = pipeline.NamedBarrier(15, WG_SIZE + WARP_SIZE)
        bar_sum_done = pipeline.NamedBarrier(10, WG_SIZE + WARP_SIZE)
        bar_qk_issued = pipeline.NamedBarrier(11, 2 * WARP_SIZE)
        bar_pv_issued = pipeline.NamedBarrier(12, 2 * WARP_SIZE)
        bar_max_turn = pipeline.NamedBarrier(13, NUM_SOFTMAX_WG * WG_SIZE)
        bar_o_done = pipeline.NamedBarrier(9, 5 * WARP_SIZE)

        # Per-WG barrier pairs use consecutive ids: id + 0 for WG1, id + 1 for WG2.
        def phase_of(nbar, phase):
            return pipeline.NamedBarrier(nbar.barrier_id + phase, nbar.num_threads)

        mbar_q_full = smem_alloc(cutlass.Int64, cute.make_layout(1), alignment=8).iterator
        if warp_idx == load_q_store_o_warp_id:
            q_tx_bytes = cute.size_in_bytes(self.dtype, sQ_layout)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_q_full, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_q_full, q_tx_bytes)
        # mbarrier pipelines: KV (TMA -> MMA), S (QK MMA -> softmax), P (softmax -> PV MMA),
        # O (PV MMA -> correction).
        pipeline_kv = cute_ext.TMAToUMMAPipeline.create(
            num_stages=self.num_stages_kv,
            mma_operation_type=cute_ext.OperationTypeEnum.SM100_MMA_1SM_SS,
        )
        pipeline_s = cute_ext.UMMAtoAsyncPipeline.create(
            num_stages=self.num_stages_s,
            mma_operation_type=cute_ext.OperationTypeEnum.SM100_MMA_1SM_SS,
            consumer=cute_ext.OperationTypeEnum.SM100_COPY_T2R,
            consumer_arv_count=WG_SIZE,
        )
        pipeline_p = cute_ext.AsyncToUMMAPipeline.create(
            num_stages=self.num_stages_p,
            producer=cute_ext.OperationTypeEnum.SM100_COPY_R2T,
            producer_arv_count=WG_SIZE,
            mma_operation_type=cute_ext.OperationTypeEnum.SM100_MMA_1SM_SS,
        )
        pipeline_o = cute_ext.UMMAtoAsyncPipeline.create(
            num_stages=self.num_stages_o,
            mma_operation_type=cute_ext.OperationTypeEnum.SM100_MMA_1SM_SS,
            consumer=cute_ext.OperationTypeEnum.SM100_COPY_T2R,
            consumer_arv_count=WG_SIZE,
        )

        cute.arch.sync_threads()

        # Every empty split must publish O=0 as well as LSE=-inf: the shared combine
        # may load its O even with a zero weight, and zero times an uninitialized NaN is NaN.
        if is_empty_split:
            if wg_idx == correction_wg_id:
                bar_max_done.arrive()
                if lane_idx < tile_hq:
                    sRowSumWarp[wg_warp_idx, lane_idx] = Float32(0.0)
                bar_sum_done.arrive()
            for o_idx in cutlass.range(thread_id, hdim_v * tile_hq, NUM_WG * WG_SIZE):
                head = hq_block * tile_hq + o_idx // hdim_v
                if head < qhead_per_kvhead:
                    batch_idx = bh_idx // num_heads_kv
                    global_head = (bh_idx % num_heads_kv) * qhead_per_kvhead + head
                    # Keep direct stores on a raw pointer argument, separate from
                    # the experimental compiler's automatically lowered TMA argument.
                    if cutlass.const_expr(len(mO_partial_raw.shape) == 3):
                        mO_partial_raw[batch_idx, global_head, o_idx % hdim_v] = (
                            mO_partial_raw.element_type(0.0)
                        )
                    else:
                        mO_partial_raw[split_id, batch_idx, global_head, o_idx % hdim_v] = (
                            mO_partial_raw.element_type(0.0)
                        )

        elif warp_idx == load_q_store_o_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            # Q: all query heads of this KV head, loaded once, one 64-dim chunk per stage.
            q_tiler = (tile_hq_mma, tile_k)
            q_coord = (hq_block, None, bh_idx)
            gQ_blk = cute.local_tile(mQ, q_tiler, q_coord)
            gQ = thr_mma_qk.partition_B(gQ_blk)
            for dk in cutlass.range_constexpr(num_k_tiles):
                gQ_cur = gQ[None, None, None, dk]
                cute_ext.tma_load(
                    gQ_cur,
                    sQ[None, None, None, dk],
                    mbar_q_full.value,
                    update_expect_tx=False,
                )

            o_partial_tiler = (tile_m, tile_hq)
            o_mma_tiler = (tile_m, tile_hq_mma)
            o_cta_v_map = cute_ext.get_cta_v_map_c(mO_partial_tma, o_mma_tiler)
            # Wait until the correction WG has staged the final O, then TMA-store this split's partial.
            bar_o_done.arrive_and_wait()
            for dm in cutlass.range_constexpr(num_d_tiles):
                o_blk_coord = (dm, hq_block, bh_idx, split_id)
                gO_blk = cute.local_tile(mO_partial_tma, o_partial_tiler, o_blk_coord)
                gO_mma = cute.flat_divide(gO_blk, o_mma_tiler)
                cute_ext.tma_store(
                    sO[None, None, dm, 0],
                    gO_mma[None, None, 0, 0],
                    cta_v_map=o_cta_v_map,
                )
            cute.arch.cp_async_bulk_commit_group()

        elif warp_idx == load_kv_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            # Producer for the shared ring. Loop iteration s pushes K(s) and V(s - lookahead), so the
            # stream is K0 K1 K2 | V0 K3 V1 ... and finally the trailing V tiles V(N-3) V(N-2) V(N-1).
            kv_tiler = (tile_m, tile_k)
            kv_token = cutlass.Boolean(True)
            for s in cutlass.range(split_id, kv_lookahead_blocks + num_n_blocks, num_splits):
                if s < num_n_blocks:
                    k_block, k_bh = s + n_block_min, bh_idx
                    if const_expr(mPageTable is not None):
                        page, offset = resolve_paged_token(
                            mPageTable[bh_idx // num_heads_kv, None],
                            k_block * tile_n,
                            mK.shape[0],
                        )
                        page = cute.arch.make_warp_uniform(page)
                        offset = cute.arch.make_warp_uniform(offset)
                        k_block = offset // tile_n
                        k_bh = page * num_heads_kv + bh_idx % num_heads_kv
                    k_block_coord = (k_block, 0, k_bh)
                    gK = cute.local_tile(mK, (tile_n, hdim), k_block_coord)
                    gK_blk = cute.zipped_divide(gK, kv_tiler)
                    for sm in cutlass.range_constexpr(num_s_subtiles):
                        gK_sub = gK_blk[(None, None), (sm, None)]
                        tKgK = thr_mma_qk.partition_A(gK_sub)
                        for dk in cutlass.range_constexpr(num_k_tiles):
                            gK_cur = tKgK[None, None, None, dk]
                            kv_token_stage, kv_stage_idx = (
                                pipeline_kv.producer_acquire_and_get_stage(token=kv_token)
                            )
                            kv_mbar = cute_ext.get_mbarrier(kv_token_stage)
                            sK_cur = sK[None, None, None, kv_stage_idx]
                            copy_kv_tma(gK_cur, sK_cur, kv_mbar)
                            pipeline_kv.producer_commit_and_advance()
                            kv_token = pipeline_kv.producer_try_acquire()

                if s >= kv_lookahead_blocks:
                    v_block, v_bh = s - kv_lookahead_blocks + n_block_min, bh_idx
                    if const_expr(mPageTable is not None):
                        page, offset = resolve_paged_token(
                            mPageTable[bh_idx // num_heads_kv, None],
                            v_block * tile_n,
                            mV.shape[1],
                        )
                        page = cute.arch.make_warp_uniform(page)
                        offset = cute.arch.make_warp_uniform(offset)
                        v_block = offset // tile_n
                        v_bh = page * num_heads_kv + bh_idx % num_heads_kv
                    v_block_coord = (0, v_block, v_bh)
                    gVt = cute.local_tile(mV, (hdim_v, tile_n), v_block_coord)
                    gVt_blk = cute.zipped_divide(gVt, kv_tiler)
                    for sk in cutlass.range_constexpr(num_pv_k_tiles):
                        gVt_sub = gVt_blk[(None, None), (None, sk)]
                        tVgV = thr_mma_pv.partition_A(gVt_sub)
                        for dm in cutlass.range_constexpr(num_d_tiles):
                            gVt_cur = tVgV[None, None, None, dm]
                            kv_token_stage, kv_stage_idx = (
                                pipeline_kv.producer_acquire_and_get_stage(token=kv_token)
                            )
                            kv_mbar = cute_ext.get_mbarrier(kv_token_stage)
                            sVt_cur = sVt[None, None, None, kv_stage_idx]
                            copy_kv_tma(gVt_cur, sVt_cur, kv_mbar)
                            pipeline_kv.producer_commit_and_advance()
                            kv_token = pipeline_kv.producer_try_acquire()

        elif warp_idx == mma_qk_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            pv_tiles_per_iter = num_d_tiles * num_pv_k_tiles
            mma_atom = cute.make_mma_atom(tiled_mma_qk.op)

            # QK MMA warp: S^T(j) = K(j) Q^T, two 128-token sub-tiles per 256-token block.
            cute.arch.mbarrier_wait(mbar_q_full, phase=0)

            for s in cutlass.range(split_id, num_n_blocks, num_splits):
                s_token = pipeline_s.producer_try_acquire()
                mma_atom.set(tcgen05.Field.ACCUMULATE, False)

                # Both MMA warps read the same ring but each keeps its own position in it. Skip the V(j-3)
                # stages (the PV warp consumes them) and wait until the PV warp has issued V(j-3); this keeps
                # the two cursors within one window so mbarrier phases are never misread.
                if s >= split_id + (kv_lookahead + 1) * num_splits:
                    for _ in cutlass.range_constexpr(pv_tiles_per_iter):
                        pipeline_kv.consumer_state = pipeline_kv.increment_state(
                            pipeline_kv.consumer_state
                        )
                bar_pv_issued.arrive_and_wait()
                k_token = pipeline_kv.consumer_try_wait()

                _, qk_stage_idx = pipeline_s.producer_acquire_and_get_stage(token=s_token)
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    s_acc_stage = qk_stage_idx * num_s_subtiles + sm
                    mma_atom.set(tcgen05.Field.ACCUMULATE, False)
                    tStS_slice = tStS[None, None, None, s_acc_stage]
                    for dk in cutlass.range_constexpr(num_k_tiles):
                        _, k_stage_idx = pipeline_kv.consumer_wait_and_get_stage(token=k_token)
                        for instr_idx in cutlass.range_constexpr(qk_k_blocks):
                            k_slice = sK[None, None, instr_idx, k_stage_idx]
                            q_slice = sQ[None, None, instr_idx, dk]
                            cute_ext.dot(mma_atom, k_slice, q_slice, tStS_slice)
                            mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                        pipeline_kv.consumer_release_and_advance()
                        if sm == num_s_subtiles - 1 and dk == num_k_tiles - 1:
                            bar_qk_issued.arrive()
                        else:
                            k_token = pipeline_kv.consumer_try_wait()
                pipeline_s.producer_commit_and_advance()

            for _ in cutlass.range_constexpr(kv_lookahead):
                bar_pv_issued.arrive_and_wait()
                bar_qk_issued.arrive()

        elif warp_idx == mma_pv_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            qk_tiles_per_iter = num_s_subtiles * num_k_tiles
            num_iters = cute.ceil_div(num_n_blocks - split_id, num_splits)
            mma_atom = cute.make_mma_atom(tiled_mma_pv.op)

            # PV MMA warp: O^T(slot j % 2) += V^T(j) P^T(j). Warm-up: skip K(0), K(1) and hand the QK warp
            # the tokens it waits for, so both warps start in ring order.
            bar_pv_issued.arrive()
            for prefetch_iter in cutlass.range_constexpr(kv_lookahead):
                if split_id + prefetch_iter * num_splits < num_n_blocks:
                    for _ in cutlass.range_constexpr(qk_tiles_per_iter):
                        pipeline_kv.consumer_state = pipeline_kv.increment_state(
                            pipeline_kv.consumer_state
                        )
                bar_qk_issued.arrive_and_wait()
                bar_pv_issued.arrive()

            for s in cutlass.range(split_id, num_n_blocks, num_splits):
                if s + kv_lookahead * num_splits < num_n_blocks:
                    for _ in cutlass.range_constexpr(qk_tiles_per_iter):
                        pipeline_kv.consumer_state = pipeline_kv.increment_state(
                            pipeline_kv.consumer_state
                        )
                bar_qk_issued.arrive_and_wait()

                p_token = pipeline_p.consumer_try_wait()
                v_token = pipeline_kv.consumer_try_wait()
                o_token = pipeline_o.producer_try_acquire()

                _, p_idx = pipeline_p.consumer_wait_and_get_stage(token=p_token)
                # The O slot for block j is free once the correction WG has rescaled what PV(j - 2) left there.
                _, o_idx = pipeline_o.producer_acquire_and_get_stage(token=o_token)
                for dk in cutlass.range_constexpr(num_pv_k_tiles):
                    for dm in cutlass.range_constexpr(num_d_tiles):
                        _, v_stage_idx = pipeline_kv.consumer_wait_and_get_stage(token=v_token)
                        tOtO_slot = tOtO[None, None, None, dm, o_idx]
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                        for instr_idx in cutlass.range_constexpr(pv_k_blocks):
                            v_slice = sVt[None, None, instr_idx, v_stage_idx]
                            p_slice = sP_nk[None, None, instr_idx, dk, p_idx]
                            cute_ext.dot(mma_atom, v_slice, p_slice, tOtO_slot)
                            mma_atom.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)
                        if dm == num_d_tiles - 1 and dk == num_pv_k_tiles - 1:
                            bar_pv_issued.arrive()
                        pipeline_kv.consumer_release_and_advance()
                        if dm != num_d_tiles - 1 or dk != num_pv_k_tiles - 1:
                            v_token = pipeline_kv.consumer_try_wait()
                pipeline_p.consumer_release_and_advance()
                pipeline_o.producer_commit_and_advance()
            # The correction tail always merges two slots; with a single block, commit an empty second one.
            if num_iters == 1:
                o_token = pipeline_o.producer_try_acquire()
                _, _o_idx = pipeline_o.producer_acquire_and_get_stage(token=o_token)
                pipeline_o.producer_commit_and_advance()
            pipeline_o.producer_tail()

        elif wg_idx in softmax_wg_ids:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_softmax)

            # Softmax WGs. Each thread owns one token row of each 128-token sub-tile, i.e. 2 rows of the
            # block, and tile_hq query-head columns.
            num_iters = cute.ceil_div(num_n_blocks - split_id, num_splits)

            tStS_0 = tStS_staged[(None, None), 0, 0, None, 0]
            t2r_rep_S = tcgen05.Repetition(tile_hq)
            if cutlass.const_expr(tile_hq_mma == tile_hq and num_s_subtiles > 1):
                t2r_rep_S = tcgen05.Repetition(tile_hq * num_s_subtiles)
            t2r_atom_S = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(t2r_rep_S, tcgen05.Pack.NONE),
                self.qk_acc_dtype,
            )
            tiled_t2r_S = tcgen05.make_tmem_copy(t2r_atom_S, tStS_0)
            tiled_r2s_P = cute.make_tiled_copy_D(r2s_atom_P, tiled_t2r_S)
            thr_t2r_S = tiled_t2r_S.get_slice(thr_in_wg)
            thr_r2s_P = tiled_r2s_P.get_slice(thr_in_wg)
            tSrS_layout = thr_t2r_S.partition_D(tStS_0).layout
            assert cute.cosize(tSrS_layout) == tile_hq_mma * num_s_subtiles

            tSrP_layout = cute.make_layout(
                (*vec_layout.shape, num_s_subtiles),
                stride=(*vec_layout.stride, vec_size),
            )
            scores_layout = cute.make_layout(
                shape=(tile_hq, num_s_subtiles),
                stride=(1, vec_size),
            )
            sP_view_layout = cute.make_layout(
                shape=(tile_m, tile_hq_mma, num_s_subtiles),
                stride=(tile_hq_mma, 1, tile_m * tile_hq_mma),
            )

            tSrS = make_rmem(self.qk_acc_dtype, tSrS_layout)
            tSrP = make_rmem(self.dtype, tSrP_layout)
            tLrL = make_rmem(self.qk_acc_dtype, vec_layout)
            row_max_r = make_rmem(self.qk_acc_dtype, vec_layout)
            scores_r = cute.make_tensor(tSrS.iterator, scores_layout)
            probs_r = cute.make_tensor(tSrP.iterator, scores_layout)
            row_sum_valid = tLrL[(None, 0), 0, 0]
            row_max_valid = row_max_r[(None, 0), 0, 0]

            phase = wg_idx - softmax_wg_ids[0]
            bar_turn_acquire = phase_of(bar_max_turn, phase)
            bar_turn_release = phase_of(bar_max_turn, phase ^ 1)
            bar_sum_empty_cur = phase_of(bar_sum_empty, phase)
            bar_sum_full_cur = phase_of(bar_sum_full, phase)
            tLtL_cur = tLtL[(None, None), 0, 0, phase]

            # WG2 handles the odd blocks: shift its pipeline cursors by one block and give WG1 the first
            # turn on the shared running max.
            if phase == 1:
                pipeline_s.consumer_state = pipeline_s.increment_state(pipeline_s.consumer_state)
                pipeline_p.producer_state = pipeline_p.increment_state(pipeline_p.producer_state)
                bar_turn_release.arrive()
                if num_iters == 1:
                    bar_sum_full_cur.arrive()

            for n_iter in cutlass.range(phase, num_iters, NUM_SOFTMAX_WG):
                s = split_id + n_iter * num_splits
                s_token = pipeline_s.consumer_try_wait()
                p_token = pipeline_p.producer_try_acquire()

                _, s_idx = pipeline_s.consumer_wait_and_get_stage(token=s_token)
                row_max_r.fill(-Float32.inf)
                tStS_cur = tStS_staged[(None, None), 0, 0, None, s_idx]
                cute_ext.partition_and_copy(thr_t2r_S, tStS_cur, tSrS)
                cute.arch.fence_view_async_tmem_load()
                pipeline_s.consumer_release_and_advance()

                # Mask the token rows past seqlen_k in the last block.
                valid_rows = seqlen_k - (s + n_block_min) * tile_n
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    if thr_in_wg + sm * tile_m >= valid_rows:
                        scores_r[None, sm].fill(-Float32.inf)

                if cutlass.const_expr(self.check_empty_rows):
                    batch_ssa = scalar_to_ssa(bh_idx // num_heads_kv, Int32)
                    q_ssa = scalar_to_ssa(Int32(0), Int32)
                    for sm in cutlass.range_constexpr(num_s_subtiles):
                        kv_idx = (s + n_block_min) * tile_n + thr_in_wg + sm * tile_m
                        for gi in cutlass.range_constexpr(tile_hq):
                            head = hq_block * tile_hq + gi
                            value = -Float32.inf
                            if head < qhead_per_kvhead and kv_idx < seqlen_k:
                                global_head = (bh_idx % num_heads_kv) * qhead_per_kvhead + head
                                head_ssa = scalar_to_ssa(global_head, Int32)
                                kv_ssa = scalar_to_ssa(kv_idx, Int32)
                                value = scores_r[gi, sm]
                                if cutlass.const_expr(self.score_mod is not None):
                                    value = ssa_to_scalar(
                                        call_score_mod(
                                            self.score_mod,
                                            scalar_to_ssa(
                                                value * (softmax_scale_log2 / LOG2_E), Float32
                                            ),
                                            batch_ssa,
                                            head_ssa,
                                            q_ssa,
                                            kv_ssa,
                                            seqlen_info,
                                            aux_data,
                                        )
                                    )
                                keep = cutlass.Boolean(True)
                                if cutlass.const_expr(self.is_local):
                                    if cutlass.const_expr(window_size_left is not None):
                                        keep &= kv_idx >= seqlen_k - 1 - window_size_left
                                    if cutlass.const_expr(window_size_right is not None):
                                        keep &= kv_idx <= seqlen_k - 1 + window_size_right
                                if cutlass.const_expr(self.mask_mod is not None):
                                    keep &= cutlass.Boolean(
                                        ssa_to_scalar(
                                            call_mask_mod(
                                                self.mask_mod,
                                                batch_ssa,
                                                head_ssa,
                                                q_ssa,
                                                kv_ssa,
                                                seqlen_info,
                                                aux_data,
                                            )
                                        )
                                    )
                                value = value if keep else -Float32.inf
                            scores_r[gi, sm] = value

                score_scale_log2 = (
                    LOG2_E if cutlass.const_expr(self.score_mod is not None) else softmax_scale_log2
                )
                scores = scores_r.load()
                row_max_valid.store(
                    scores.reduce(
                        cute.ReductionOp.MAX,
                        -Float32.inf,
                        (None, 0),
                    ).reshape((tile_hq,))
                )

                # Block max per head: reduce over this warp's rows; lane h then holds the max of head h.
                lane_max = -Float32.inf
                for j in cutlass.range_constexpr(tile_hq):
                    row_max_r[j] = warp_reduce_max(row_max_r[j])
                    if j == lane_idx:
                        lane_max = row_max_r[j]
                lane_max *= score_scale_log2

                # --- critical section: the running max ---
                # sRowMax holds ONE chain: M(j) = max(M(j-1), blockmax(j)), with WG1 producing the even j and
                # WG2 the odd j. Named barriers only count arrivals, so without bar_max_turn both softmax WGs
                # could satisfy bar_max_empty / bar_max_full together and the correction WG would miss an M(j).
                # bar_max_turn is a token: take it, wait until the correction WG has read M(j-1), fold in our
                # block max with a shared-memory atomic max, publish M(j), then pass the token on.
                bar_turn_acquire.arrive_and_wait()
                bar_max_empty.arrive_and_wait()
                if lane_owns_head:
                    sRowMax_ptr = sRowMax.iterator + sRowMax.layout(lane_idx)
                    smem_atomic_max(sRowMax_ptr, lane_max)

                bar_max_full.arrive_and_wait_unaligned()
                cute.autovec_copy(sRowMax, row_max_r)
                bar_turn_release.arrive()

                # P = exp2(S * scale * log2(e) - M(j)) in bf16 / fp16 into SMEM for the PV MMA.
                _, p_idx = pipeline_p.producer_acquire_and_get_stage(token=p_token)

                if cutlass.const_expr(self.check_empty_rows):
                    # Only the local exponent bias is zeroed; shared final maxima stay -inf.
                    for gi in cutlass.range_constexpr(tile_hq):
                        if row_max_valid[gi] == -Float32.inf:
                            row_max_valid[gi] = Float32(0.0)
                row_max_tile = row_max_valid.load().reshape((tile_hq, 1))
                probs = exp2_fast(score_scale_log2 * scores - row_max_tile)

                if cutlass.const_expr(tile_hq < tile_hq_mma):
                    tSrP.fill(self.dtype(0))

                probs_r.store(probs.to(self.dtype))
                sP_view = cute.make_tensor(sP[None, None, None, p_idx].iterator, sP_view_layout)
                cute_ext.partition_and_copy(thr_r2s_P, tSrP, sP_view)
                cute.arch.fence_view_async_shared()
                pipeline_p.producer_commit_and_advance()

                # Only this thread's partial row sums (its 2 rows, per head) go to the correction WG through
                # this WG's TMEM mailbox; the full P is never copied. The cross-thread sum happens once at
                # the very end.
                row_sum_valid.store(
                    probs.reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        (None, 0),
                    ).reshape((tile_hq,))
                )

                bar_sum_empty_cur.arrive_and_wait()
                cute_ext.partition_and_copy(thr_r2t, tLrL, tLtL_cur)
                cute.arch.fence_view_async_tmem_store()
                bar_sum_full_cur.arrive()

                # Skip the block that the other softmax WG handles.
                pipeline_s.consumer_state = pipeline_s.increment_state(pipeline_s.consumer_state)
                pipeline_p.producer_state = pipeline_p.increment_state(pipeline_p.producer_state)

        elif wg_idx == correction_wg_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_increase(num_regs_correction)

            # Correction WG. Thread t owns TMEM lane t (one head_dim_v row of O, all heads) and keeps one
            # row-sum accumulator per O slot.
            l_slot_shape = (*vec_layout.shape, self.num_stages_o)
            l_slot_stride = (*vec_layout.stride, vec_size)
            row_sum_slot_layout = cute.make_layout(l_slot_shape, stride=l_slot_stride)
            tLrL = make_rmem(self.qk_acc_dtype, vec_layout)
            row_sum_final = make_rmem(self.qk_acc_dtype, vec_layout)
            acc_scale = make_rmem(self.qk_acc_dtype, vec_layout)
            row_sum_acc = make_rmem(self.qk_acc_dtype, row_sum_slot_layout)

            row_sum_acc.fill(Float32(0.0))
            num_iters = cute.ceil_div(num_n_blocks - split_id, num_splits)

            zeros_r = make_rmem(self.qk_acc_dtype, vec_layout)
            zeros_r.fill(Float32(0.0))
            for phase in cutlass.range_constexpr(self.num_stages_o):
                for dm in cutlass.range_constexpr(num_d_tiles):
                    tOtO_slot = tOtO[(None, None), 0, 0, dm, phase]
                    cute_ext.partition_and_copy(thr_r2t, zeros_r, tOtO_slot)
            tLtL_1 = tLtL[(None, None), 0, 0, 1]
            cute_ext.partition_and_copy(thr_r2t, zeros_r, tLtL_1)
            cute.arch.fence_view_async_tmem_store()

            # Initial tokens: the running max and both row-sum mailboxes start out empty.
            bar_max_empty.arrive()
            for phase in cutlass.range_constexpr(NUM_SOFTMAX_WG):
                phase_of(bar_sum_empty, phase).arrive()

            # Read M(0) and M(1) up front: rescaling slot j needs M(j) and M(j + 2).
            row_max_prev2, row_max_prev = -Float32.inf, -Float32.inf
            for preload_idx in cutlass.range_constexpr(self.num_stages_o):
                row_max_prev2 = row_max_prev
                if not (preload_idx == 1 and num_iters == 1):
                    bar_max_full.arrive_and_wait_unaligned()
                    row_max_prev = Float32(0.0)
                    if lane_owns_head:
                        row_max_prev = sRowMax[lane_idx]
                    bar_max_empty.arrive()

            # --- main loop, block s uses O slot s % 2 ---
            # 1. collect the partial row sums of block s
            # 2. read M(s + 2); the last time round tell the split-stats warp the max is final
            # 3. once PV(s) has landed, scale the slot by exp2(M(s) - M(s + 2)) and release it, so
            #    PV(s + 2) can accumulate on top. Two slots keep this off the PV MMA's critical path.
            phase = 0
            for s in cutlass.range(num_iters - self.num_stages_o, unroll=self.num_stages_o):
                phase_of(bar_sum_full, phase).arrive_and_wait()
                tLtL_cur = tLtL[(None, None), 0, 0, phase]
                cute_ext.partition_and_copy(thr_t2r, tLtL_cur, tLrL)
                cute.arch.fence_view_async_tmem_load()
                phase_of(bar_sum_empty, phase).arrive()

                bar_max_full.arrive_and_wait_unaligned()
                if s == num_iters - self.num_stages_o - 1:
                    bar_max_done.arrive()
                row_max_cur = Float32(0.0)
                if lane_owns_head:
                    row_max_cur = sRowMax[lane_idx]
                bar_max_empty.arrive()

                o_token = pipeline_o.consumer_try_wait()
                _, _o_idx = pipeline_o.consumer_wait_and_get_stage(token=o_token)

                acc_scale_lane = exp2_fast(row_max_prev2 - row_max_cur)
                if cutlass.const_expr(self.check_empty_rows):
                    if row_max_prev2 == -Float32.inf and row_max_cur == -Float32.inf:
                        acc_scale_lane = Float32(1.0)
                for gi in cutlass.range_constexpr(tile_hq):
                    acc_scale[gi] = cute.arch.shuffle_sync(acc_scale_lane, gi)

                for dm in cutlass.range_constexpr(num_d_tiles):
                    tOtO_slot = tOtO[(None, None), 0, 0, dm, phase]
                    tOrO = make_rmem(self.qk_acc_dtype, vec_layout)
                    cute_ext.partition_and_copy(thr_t2r, tOtO_slot, tOrO)
                    cute.arch.fence_view_async_tmem_load()
                    tOrO.store(tOrO.load() * acc_scale.load())
                    cute_ext.partition_and_copy(thr_r2t, tOrO, tOtO_slot)
                cute.arch.fence_view_async_tmem_store()
                pipeline_o.consumer_release_and_advance()

                row_sum_acc_cur = row_sum_acc[None, None, None, phase]
                tLrL.store(tLrL.load() * acc_scale.load())
                row_sum_acc_cur.store(row_sum_acc_cur.load() * acc_scale.load() + tLrL.load())

                row_max_prev2, row_max_prev = row_max_prev, row_max_cur
                phase ^= 1

            if num_iters <= self.num_stages_o:
                bar_max_done.arrive()

            # --- tail: merge the two slots ---
            # The last two blocks n-2, n-1 are relative to M(n-2) and M(n-1). Bring slot (n-2) up to the
            # final max with exp2(M(n-2) - M(n-1)) and add the slots: O = O[n-1] + c * O[n-2], and the
            # row sums the same way.
            acc_scale_lane = exp2_fast(row_max_prev2 - row_max_prev)
            if cutlass.const_expr(self.check_empty_rows):
                if row_max_prev2 == -Float32.inf and row_max_prev == -Float32.inf:
                    acc_scale_lane = Float32(1.0)
            for gi in cutlass.range_constexpr(tile_hq):
                acc_scale[gi] = cute.arch.shuffle_sync(acc_scale_lane, gi)

            tail_phase = num_iters % self.num_stages_o
            for phase in cutlass.range_constexpr(self.num_stages_o):
                if tail_phase == phase:
                    final_phase = phase ^ 1

                    phase_of(bar_sum_full, phase).arrive_and_wait()
                    tLtL_cur = tLtL[(None, None), 0, 0, phase]
                    cute_ext.partition_and_copy(thr_t2r, tLtL_cur, tLrL)
                    cute.arch.fence_view_async_tmem_load()
                    phase_of(bar_sum_empty, phase).arrive()

                    phase_of(bar_sum_full, final_phase).arrive_and_wait()
                    tLtL_last = tLtL[(None, None), 0, 0, final_phase]
                    cute_ext.partition_and_copy(thr_t2r, tLtL_last, row_sum_final)
                    cute.arch.fence_view_async_tmem_load()
                    phase_of(bar_sum_empty, final_phase).arrive()

                    tLrL.store(tLrL.load() + row_sum_acc[None, None, None, phase].load())
                    row_sum_final.store(
                        row_sum_final.load()
                        + row_sum_acc[None, None, None, final_phase].load()
                        + acc_scale.load() * tLrL.load()
                    )

            # Sum the row sums across the 32 lanes, then per warp into SMEM for the split-stats warp.
            for gi in cutlass.range_constexpr(tile_hq):
                row_sum_final[gi] = cute.arch.warp_reduction_sum(row_sum_final[gi])
                if lane_idx == 0:
                    sRowSumWarp[wg_warp_idx, gi] = row_sum_final[gi]
            bar_sum_done.arrive()

            # The stats warp publishes 1/L only after consuming the final max
            # and all four warp sums. This rendezvous must not alias the earlier
            # nonblocking final-max notification.
            bar_stats_ready.arrive_and_wait()
            cute.autovec_copy(sRowMax, tLrL)

            o_slot_shape = (*vec_layout.shape, num_d_tiles, self.num_stages_o)
            o_slot_stride = (
                *vec_layout.stride,
                vec_size,
                num_d_tiles * vec_size,
            )
            tOrO_slot_layout = cute.make_layout(o_slot_shape, stride=o_slot_stride)
            tOrO_tail = make_rmem(self.qk_acc_dtype, tOrO_slot_layout)
            for s in cutlass.range_constexpr(self.num_stages_o):
                phase = tail_phase ^ s
                o_token = pipeline_o.consumer_try_wait()
                _, _o_idx = pipeline_o.consumer_wait_and_get_stage(token=o_token)
                cute_ext.partition_and_copy(
                    thr_t2r,
                    tOtO[(None, None), 0, 0, None, phase],
                    tOrO_tail[(None, None), None, None, None, s],
                )
                cute.arch.fence_view_async_tmem_load()
                pipeline_o.consumer_release_and_advance()

            # Normalize the merged O in fp32, then stage it for warp 3's ordinary TMA store.
            tOrO_tail0 = tOrO_tail[None, None, None, None, 0].load()
            tOrO_tail1 = tOrO_tail[None, None, None, None, 1].load()
            tOrO_tail0 = tOrO_tail0.reshape((vec_size, num_d_tiles))
            tOrO_tail1 = tOrO_tail1.reshape((vec_size, num_d_tiles))
            tOrO_final = tOrO_tail1 + acc_scale.load().reshape((vec_size, 1)) * tOrO_tail0
            tOrO_final *= tLrL.load().reshape((vec_size, 1))
            if cutlass.const_expr(self.check_empty_rows):
                # Select real zeros for fully masked heads, never rely on 0 * NaN.
                normalized = make_rmem(self.qk_acc_dtype, cute.make_layout((vec_size, num_d_tiles)))
                normalized.store(tOrO_final)
                for gi in cutlass.range_constexpr(tile_hq):
                    if tLrL[gi] == Float32(0.0):
                        for dm in cutlass.range_constexpr(num_d_tiles):
                            normalized[gi, dm] = Float32(0.0)
                tOrO_final = normalized.load()
            tOsO = cute_ext.partition(
                sO[None, None, None, 0],
                thr_r2s_O.thr_idx,
                layout_tv=thr_r2s_O.layout_dst_tv_tiled,
                tiler=cute.core._pack_tile(thr_r2s_O.tiler_mn),
            )
            tOsO.store(tOrO_final.to(mO_partial_tma.element_type).reshape(tOsO.shape))

            cute.arch.fence_view_async_shared()
            bar_o_done.arrive()

        # After its loads, warp 2 normalizes this split's statistics, not other CTAs'.
        if warp_idx == split_stats_warp_id:
            self.store_split_stats(
                tile_hq,
                (split_id, hq_block, bh_idx),
                lane_idx,
                is_empty_split,
                bar_max_done,
                bar_sum_done,
                bar_stats_ready,
                sRowMax,
                sRowSumWarp,
                mLSE_partial,
            )

    @staticmethod
    @cute.jit
    def store_split_stats(
        tile_hq: int,
        block_idx: Tuple[Int32, Int32, Int32],
        lane_idx: Int32,
        is_empty_split: cutlass.Boolean,
        bar_max_done,
        bar_sum_done,
        bar_stats_ready,
        sRowMax: cute.Tensor,
        sRowSumWarp: cute.Tensor,
        mLSE_partial: cute.Tensor | None,
    ):
        split_id, hq_block, bh_idx = block_idx
        lane_owns_head = lane_idx < tile_hq
        head = hq_block * tile_hq + lane_idx
        row_max_lane = -Float32.inf

        cute.arch.fence_acq_rel_cta()
        bar_max_done.arrive_and_wait()
        if lane_owns_head:
            row_max_lane = sRowMax[lane_idx]

        bar_sum_done.arrive_and_wait()
        if lane_owns_head:
            row_sum_w01 = sRowSumWarp[0, lane_idx] + sRowSumWarp[1, lane_idx]
            row_sum_w23 = sRowSumWarp[2, lane_idx] + sRowSumWarp[3, lane_idx]
            row_sum = row_sum_w01 + row_sum_w23
            inv_sum = Float32(0.0)
            if row_sum > Float32(0.0):
                inv_sum = cute.arch.rcp_approx(row_sum)
            sRowMax[lane_idx] = inv_sum
            if cutlass.const_expr(mLSE_partial is not None):
                if head < mLSE_partial.shape[1]:
                    lse = -Float32.inf
                    if row_sum > Float32(0.0):
                        lse = (row_max_lane + cute.math.log2(row_sum, fastmath=True)) * math.log(
                            2.0
                        )
                    num_heads_kv = mLSE_partial.shape[2][0]
                    bh_coord = (bh_idx % num_heads_kv, bh_idx // num_heads_kv)
                    mLSE_partial[split_id, head, bh_coord] = lse

        # Empty CTAs skip correction's final wait: do not leave an unmatched arrival.
        if not is_empty_split:
            bar_stats_ready.arrive()


class FlashAttentionForwardSm100SwapAB:
    """Standard forward ABI for single-query inference; interface owns allocation and combine.

    The cute_ext mainloop is private. Split outputs are normalized FP32 O and natural-log
    LSE; the single-split case writes directly to the input-dtype output tensor.
    """

    ptxas_options = ""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        m_block_size: int = 1,
        n_block_size: int = 256,
        q_stage: int = 1,
        is_static_persistent: bool = False,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
        has_tile_count_semaphore: bool = False,
        seqlen_k_per_split: Optional[int] = None,
        use_s_ping_pong: bool = False,
    ):
        if head_dim not in (64, 96, 128) or head_dim_v not in (None, 64, 96, 128):
            raise ValueError("SwapAB supports QK and V head dimensions 64, 96, 128")
        if not 1 <= qhead_per_kvhead <= 32:
            raise ValueError("SwapAB requires 1 <= qhead_per_kvhead <= 32")
        if m_block_size not in (1, 2, 4, 8, 16, 32) or n_block_size != 256:
            raise ValueError(
                "SwapAB requires a power-of-two head tile <= 32 and a 256-token KV tile"
            )
        # These are genuinely different schedules or semantics, not ignored tuning knobs.
        for name, enabled in (
            ("paged_kv_non_tma", paged_kv_non_tma),
            ("is_varlen_q", is_varlen_q),
            ("use_2cta_instrs", use_2cta_instrs),
            ("use_clc_scheduler", use_clc_scheduler),
            ("has_tile_count_semaphore", has_tile_count_semaphore),
            ("is_static_persistent", is_static_persistent),
            ("use_s_ping_pong", use_s_ping_pong),
        ):
            if enabled:
                raise ValueError(f"SwapAB does not support {name}")
        if q_stage != 1 or q_subtile_factor != 1 or kv_subtile_factor != 1:
            raise ValueError("SwapAB requires q_stage=q_subtile_factor=kv_subtile_factor=1")
        if seqlen_k_per_split is not None:
            raise ValueError("SwapAB uses cyclic splits, not seqlen_k_per_split")
        # For one query at the end of seqused_k, causal and noncausal bounds agree.
        # pack_gqa does not alter the external tensor ABI: the core packs heads itself.
        self.is_causal = is_causal
        self.pack_gqa = pack_gqa
        self.head_dim = head_dim
        self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.is_local = is_local
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.tile_hq = m_block_size
        self.num_hq_blocks = (qhead_per_kvhead + self.tile_hq - 1) // self.tile_hq

    @cute.experimental.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, 1, h, d)
        mK: cute.Tensor,  # (b, s_k, h_k, d)
        mV: cute.Tensor,  # (b, s_k, h_k, d)
        mO: cute.Tensor,  # split: out_partial (ns, b, 1, h, d) fp32 ; else out (b, 1, h, d)
        mLSE: Optional[cute.Tensor],  # split: lse_partial (ns, b, h, 1) ; else lse (b, h, 1) | None
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
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        tile_count_semaphore: Optional[cute.Tensor] = None,
        virtual_batch_idx_ptr: Optional[cute.Tensor] = None,
        num_nheads_in_l2_ptr: Optional[cute.Tensor] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None,
        mBlocksToBatchIdx: Optional[cute.Tensor] = None,
        max_seqlen_q: Int32 | int | None = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Compile-time guards (Python None / static rank / dtype).
        for value in (
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            learnable_sink,
            descale_tensors,
            blocksparse_tensors,
            num_splits_dynamic_ptr,
            tile_count_semaphore,
            virtual_batch_idx_ptr,
            num_nheads_in_l2_ptr,
            mCuTotalMBlocks,
            mCuTotalSplitsMBlocks,
            mBlocksToBatchIdx,
        ):
            assert value is None, "swap-AB adapter: unsupported argument"
        assert len(mQ.shape) == 4 and len(mK.shape) == 4 and len(mV.shape) == 4
        assert mQ.element_type in (cutlass.Float16, cutlass.BFloat16)
        assert mQ.element_type == mK.element_type == mV.element_type
        if const_expr(not self.is_split_kv):
            assert mO.element_type == mQ.element_type
        if const_expr(mLSE is not None):
            assert mLSE.element_type == Float32

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        batch, num_heads = mQ.shape[0], mQ.shape[2]
        num_heads_kv, seqlen_k = mK.shape[2], mK.shape[1]
        if const_expr(mPageTable is not None):
            seqlen_k = mPageTable.shape[1] * mK.shape[1]
        # (b, 1, h, d) -> (b, h, d): the dropped seqlen_q mode has extent 1 (host-checked)
        mQ3 = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[0, 2, 3]))
        if const_expr(self.is_split_kv):
            assert len(mO.shape) == 5 and mO.element_type == Float32 and mLSE is not None
            num_splits = mO.shape[0]
            # out_partial (ns, b, 1, h, d) -> (ns, b, h, d); consumed by the stock combine
            mO_core = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[0, 1, 3, 4]))
            # lse_partial (ns, b, h, 1) -> (b, h, ns), real strides (h, 1, b*h); the combine
            # reads lse_partial.transpose(-1, -2), so natural-log LSE lands where it expects it
            mLSE_core = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[1, 2, 0]))
        else:
            num_splits = Int32(1)
            # out (b, 1, h, d) in the q dtype -> (b, h, d): the core writes normalized O directly
            mO_core = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=[0, 2, 3]))
            mLSE_core = mLSE  # (b, h, 1) == (b, h, ns=1), or None
        grid = (num_splits, self.num_hq_blocks, batch * num_heads_kv)
        core = _SwapABCore(
            self.tile_hq,
            256,
            self.head_dim,
            mQ.element_type,
            grid,
            score_mod=self.score_mod,
            mask_mod=self.mask_mod,
            is_local=self.is_local,
            head_dim_v=self.head_dim_v,
        )
        problem_size = (batch, num_heads, num_heads_kv, seqlen_k, self.head_dim)
        # Stock conventions carried through: window sizes as resolved by
        # _resolve_causal_local_window (dynamic Int32 or None), AuxData(tensors, scalars) as
        # converted by the interface (to_cute_aux_tensor / aux_scalar_metadata in the key).
        core(
            problem_size,
            mQ3,
            mK,
            mV,
            mO_core,
            mLSE_core,
            mSeqUsedK,
            softmax_scale * LOG2_E,
            stream,
            window_size_left,
            window_size_right,
            aux_data,
            mPageTable,
        )
