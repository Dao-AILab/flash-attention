# FlashAttention decode for Blackwell (SM100): one query token per sequence, MHA / GQA / MQA.
# Adapted from NVIDIA CUTLASS gqa_decode_opt
# (examples/python/CuTeDSL/cute_ext/blackwell/attention/gqa_decode_opt.py).
#
# How it works
#   * Swap-AB. Decode has only a handful of query heads per KV head, so instead of padding them up
#     to a 128-row MMA tile (what the regular FA4 forward does), we compute S^T = K Q^T and
#     O^T = V^T P^T: the KV sequence is the 128-row M side and the query heads are the small N side
#     (8..32 columns). No tensor-core work is wasted on padding.
#   * Split-KV. The sequence is cut into 256-token blocks; CTA `split_id` handles blocks
#     split_id, split_id + num_splits, ... and produces a partial (O, row max, row sum).
#   * Two ways to merge the splits (reduction_mode):
#       "kernel"  each split writes fp32 partials; a small combine kernel merges them.
#                 Deterministic. With PDL the combine kernel launches early and overlaps the tail.
#       "atomic"  the splits of one (batch, KV head) form a thread-block cluster, butterfly-reduce
#                 the row max / row sum through DSMEM and TMA-reduce-add normalized O into the
#                 (zeroed) output. No second kernel; not bitwise deterministic; <= 16 splits.
#   * One CTA = 16 warps:
#       WG0  warp 0 QK MMA | warp 1 PV MMA | warp 2 K/V TMA loads (+ split stats at the end)
#            | warp 3 Q TMA load (+ O partial TMA store at the end)
#       WG1, WG2  softmax, alternating blocks (WG1 even, WG2 odd), so one WG computes while the
#                 other waits on TMEM / barriers
#       WG3  correction: rescales the O accumulator and accumulates the row sums
#   * K and V tiles stream through ONE shared SMEM ring in the order K(j), V(j-2): the QK MMA can
#     run two blocks ahead of the PV MMA.
#   * O lives in two TMEM slots (even / odd blocks); the correction rescales a slot two blocks
#     later, so it never stalls the PV MMA.
#
# What this version changes (relative to gqa_decode_opt)
#   * No memset in kernel mode. The original kernel reduction kept a global running max (gpu-scope
#     atomicMax into an M_final buffer), which had to be reset to -inf before every call (an extra
#     memset node, ~1.6-1.9 us on B200). Here each split just writes its own max; the combine
#     kernel takes the max over the split maxima it already stages in SMEM. No buffer, no atomic,
#     no reset: workspaces can be plain torch.empty. Same output bit for bit; 1.02-1.23x faster.
#   * Both reduction modes and PDL are kept; "auto" picks the mode from the split / CTA count,
#     and PDL defaults to on for kernel mode and off for atomic mode (measured, see
#     flash_attn_decode_func). With PDL the decode kernel also waits (griddepcontrol.wait) before
#     reading Q, K / V or seqused_k, so it can follow a PDL-launched producer kernel safely.
#   * FlashAttention layouts and features: q (batch, 1, num_heads, head_dim) and k / v
#     (batch, seqlen_k, num_heads_kv, head_dim) read through their real strides (a slice of a KV
#     cache works without a copy), per-sequence `seqused_k`, optional LSE output (both modes),
#     and a split heuristic fitted for bandwidth (see get_decode_config).
#   * Python entry point: flash_attn_decode_func, with a compile cache

import math
from functools import partial
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32

try:  # nvidia-cutlass-dsl >= 4.8
    from cutlass.memory import get_smem_capacity_in_bytes
except ImportError:  # older releases
    from cutlass.utils import get_smem_capacity_in_bytes

LOG2_E = math.log2(math.e)

exp2_fast = partial(cute.math.exp2, fastmath=True)
warp_reduce_max = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_atomic_max = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")

_TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

WARP_SIZE = 32
WARPS_PER_WG = 4
WG_SIZE = WARPS_PER_WG * WARP_SIZE
MMA_N_MIN = 8
NUM_SOFTMAX_WG = 2
NUM_WG = NUM_SOFTMAX_WG + 2
# Atomic reduction: splits form one thread-block cluster; log2(16) butterfly steps at most.
MAX_CLUSTER_REDUCE_STEPS = 4


# tile_hq: query heads per CTA (MMA N, padded to >= 8); tile_n: KV tokens per block (256);
# grid_shape: (num_splits, num_hq_blocks, batch * num_heads_kv).
class FlashAttentionDecodeSm100:
    def __init__(
        self,
        tile_hq,
        tile_n,
        head_dim,
        dtype,
        grid_shape,
        reduction_mode="kernel",
        use_pdl=True,
    ):
        # reduction_mode "kernel": deterministic, splits write fp32 partials and a combine kernel
        #   merges them (no workspace initialization needed).
        # reduction_mode "atomic": the splits of one (batch, KV head) form a cluster, exchange their
        #   row max / row sum through DSMEM and TMA-reduce-add normalized O straight into the
        #   output. No second kernel, but the output must be zeroed first and the bf16 / fp16
        #   additions are not deterministic.
        assert reduction_mode in ("kernel", "atomic")
        self.use_atomic_reduction = reduction_mode == "atomic"
        # PDL: the combine kernel (and the next kernel in the stream) may launch while this one
        # drains; they wait in griddepcontrol.wait before touching our outputs.
        self.use_pdl = use_pdl
        self.tile_hq = tile_hq
        self.tile_n = tile_n
        self.head_dim = head_dim
        self.dtype = dtype
        self.qk_acc_dtype = cutlass.Float32
        self.grid_shape = grid_shape

        assert head_dim > 0 and head_dim % 64 == 0
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
        if hdim % 64 != 0:
            raise ValueError(f"KV head dim ({hdim}) must be divisible by 64")
        if self.use_atomic_reduction and self.grid_shape[0] not in (1, 2, 4, 8, 16):
            raise ValueError(
                f"atomic reduction needs num_splits in (1, 2, 4, 8, 16), got {self.grid_shape[0]}"
            )

    @staticmethod
    def _view(tensor_in: cute.Tensor, shape: tuple, order: tuple):
        return cute.make_tensor(
            tensor_in.iterator,
            cute.make_ordered_layout(shape=shape, order=order),
        )

    @cute.experimental.jit
    def __call__(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        mQ: cute.Tensor,  # (batch, num_heads, head_dim), head_dim contiguous
        mK: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim), head_dim contiguous
        mV: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim), head_dim contiguous
        mO_partial: Optional[cute.Tensor],
        mMax_partial: Optional[cute.Tensor],
        mSum_partial: Optional[cute.Tensor],
        mO: cute.Tensor,  # (batch, num_heads, head_dim)
        mLSE: Optional[cute.Tensor],  # (batch, num_heads) fp32, natural log
        mSeqUsedK: Optional[cute.Tensor],  # (batch,) int32: KV tokens actually used per sequence
        softmax_scale_log2: Float32,
        stream: cuda.CUstream,
    ):
        block = (NUM_WG * WG_SIZE, 1, 1)
        num_splits = self.grid_shape[0]
        cluster = (num_splits, 1, 1) if cutlass.const_expr(self.use_atomic_reduction) else (1, 1, 1)
        smem = cute.Int64(get_smem_capacity_in_bytes("sm_100"))

        hdim = self.head_dim
        batch, num_heads, num_heads_kv, seqlen_k, _d = problem_size
        qhead_per_kvhead = num_heads // num_heads_kv
        num_bh = batch * num_heads_kv
        tile_hq = self.tile_hq
        tile_n = self.tile_n

        # GEMM views over the FA layouts, built from the real strides (so e.g. a KV cache slice
        # works without a copy). The (kv head, batch) pair is one hierarchical mode, indexed by
        # bh_idx = batch_idx * num_heads_kv + kv_head_idx.
        # QK: K is operand A (M = tokens, K = head_dim), Q is operand B (N = query heads of this
        # KV head). PV: V^T is operand A (M = head_dim, K = tokens).
        q_sb, q_sh, _ = mQ.stride
        k_sb, k_ss, k_sh, _ = mK.stride
        v_sb, v_ss, v_sh, _ = mV.stride
        mK_qk = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (seqlen_k, hdim, (num_heads_kv, batch)), stride=(k_ss, 1, (k_sh, k_sb))
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
                (hdim, seqlen_k, (num_heads_kv, batch)), stride=(1, v_ss, (v_sh, v_sb))
            ),
        )
        mLSE_fwd = None
        if cutlass.const_expr(not self.use_atomic_reduction):
            assert mO_partial is not None
            assert mMax_partial is not None
            assert mSum_partial is not None
            # Per-split outputs: unnormalized fp32 O, and the row max / row sum it is relative to.
            mO_partial_tma = self._view(
                mO_partial,
                (hdim, qhead_per_kvhead, num_bh, num_splits),
                (0, 1, 2, 3),
            )
            mMax_partial_fwd = self._view(
                mMax_partial, (num_splits, qhead_per_kvhead, num_bh), (0, 1, 2)
            )
            mSum_partial_fwd = self._view(
                mSum_partial, (num_splits, qhead_per_kvhead, num_bh), (0, 1, 2)
            )
        else:
            # Atomic reduction stores straight into the output. A size-1 "split" mode keeps the
            # TMA store tile rank identical to the kernel-reduction path.
            o_sb, o_sh, _ = mO.stride
            mO_partial_tma = cute.make_tensor(
                mO.iterator,
                cute.make_layout(
                    (hdim, qhead_per_kvhead, (num_heads_kv, batch), 1),
                    stride=(1, o_sh, (qhead_per_kvhead * o_sh, o_sb), 0),
                ),
            )
            mMax_partial_fwd = None
            mSum_partial_fwd = None
            if cutlass.const_expr(mLSE is not None):
                l_sb, l_sh = mLSE.stride
                mLSE_fwd = cute.make_tensor(
                    mLSE.iterator,
                    cute.make_layout(
                        (qhead_per_kvhead, (num_heads_kv, batch)),
                        stride=(l_sh, (qhead_per_kvhead * l_sh, l_sb)),
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
        num_d_tiles = cute.ceil_div(hdim, tile_m)
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
        if cutlass.const_expr(self.use_atomic_reduction):
            # DSMEM scratch (max and sum per butterfly step) + one mbarrier per step and value
            smem_bits_used += MAX_CLUSTER_REDUCE_STEPS * tile_hq * self.qk_acc_dtype.width * 2
            smem_bits_used += MAX_CLUSTER_REDUCE_STEPS * mbar_bits * 2
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

        # fp32 staging tile for the O partial TMA store (it reuses the K/V ring SMEM at the end).
        O_shape = (max(self.head_dim, tile_m), tile_hq_mma)
        sO_layout = cute.tile_to_shape(sO_layout_atom, O_shape, order=(1, 0))
        sO_layout = cute.flat_divide(sO_layout, (tile_m, tile_hq_mma))

        self.attn_kernel(
            problem_size,
            mQ_qk,
            mK_qk,
            mVt_pv,
            mO_partial_tma,
            mMax_partial_fwd,
            mSum_partial_fwd,
            mSeqUsedK,
            mLSE_fwd,
            softmax_scale_log2,
            tiled_mma_qk,
            tiled_mma_pv,
            sQ_layout,
            sK_layout,
            sVt_layout,
            sO_layout,
        ).launch(
            grid=self.grid_shape,
            block=block,
            cluster=cluster,
            smem=smem,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

        if cutlass.const_expr(not self.use_atomic_reduction):
            hdim_per_cta = hdim
            hdim_per_thread = 32 // self.dtype.width
            threads_per_cta = hdim_per_cta // hdim_per_thread
            num_hdim_blocks = cute.ceil_div(hdim, hdim_per_cta)
            num_n_blocks = cute.ceil_div(seqlen_k, self.tile_n)
            # Splits beyond the last KV block never write O, so the combine kernel only reads the first
            # num_valid_splits entries.
            num_valid_splits = min(num_splits, num_n_blocks)
            combine_smem_bytes = num_valid_splits * 2 * Float32.width // 8
            assert mO_partial is not None
            assert mMax_partial is not None
            assert mSum_partial is not None
            mO_partial_cmb = self._view(
                mO_partial, (hdim, num_heads, batch, num_splits), (0, 1, 2, 3)
            )
            mMax_partial_cmb = self._view(mMax_partial, (num_splits, num_heads, batch), (0, 1, 2))
            mSum_partial_cmb = self._view(mSum_partial, (num_splits, num_heads, batch), (0, 1, 2))
            mO_cmb = cute.make_tensor(
                mO.iterator,
                cute.make_layout((hdim, num_heads, batch), stride=(1, mO.stride[1], mO.stride[0])),
            )
            mLSE_cmb = None
            if cutlass.const_expr(mLSE is not None):
                mLSE_cmb = cute.make_tensor(
                    mLSE.iterator,
                    cute.make_layout((num_heads, batch), stride=(mLSE.stride[1], mLSE.stride[0])),
                )
            self.combine_kernel(
                mO_partial_cmb,
                mMax_partial_cmb,
                mSum_partial_cmb,
                mO_cmb,
                mLSE_cmb,
                num_valid_splits,
            ).launch(
                grid=(num_hdim_blocks, num_heads, batch),
                block=(threads_per_cta, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
                smem=combine_smem_bytes,
                min_blocks_per_mp=1,
                use_pdl=self.use_pdl,
            )

    @cute.experimental.kernel
    def attn_kernel(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32],
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO_partial_tma: cute.Tensor,
        mMax_partial: Optional[cute.Tensor],
        mSum_partial: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sVt_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
    ):
        # Warp roles (see header): WG0 = MMA + TMA warps, WG1 / WG2 = softmax for even / odd KV blocks,
        # WG3 = correction. Warp 2 also writes this split's (row max, row sum) at the end.
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
        _b, num_heads, num_heads_kv, seqlen_k, _d = problem_size
        tile_n = self.tile_n
        tile_hq = self.tile_hq
        split_id, hq_block, bh_idx = cute.arch.block_idx()
        qhead_per_kvhead = num_heads // num_heads_kv
        if cutlass.const_expr(mSeqUsedK is not None):
            # seqused_k may be written by the previous kernel in the stream: with PDL, wait for it.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            # Per-sequence KV length: everything below (block count, empty splits, the tail mask)
            # derives from this value, so ragged batches need no other change.
            seqlen_k = mSeqUsedK[bh_idx // num_heads_kv]
        num_n_blocks = cute.ceil_div(seqlen_k, tile_n)
        num_splits = self.grid_shape[0]

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
        # O (2 slots x head_dim / 128 tiles).
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
        sReduce = None
        if cutlass.const_expr(self.use_atomic_reduction):
            # DSMEM landing buffers for the cluster butterfly: [head, step, (max | sum)]
            sReduce = smem_alloc(
                self.qk_acc_dtype, cute.make_layout((tile_hq, MAX_CLUSTER_REDUCE_STEPS, 2))
            )

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
        bar_max_full = pipeline.NamedBarrier(2, 2 * WG_SIZE)
        bar_max_empty = pipeline.NamedBarrier(3, 2 * WG_SIZE)
        bar_sum_full = pipeline.NamedBarrier(4, 2 * WG_SIZE)
        bar_sum_empty = pipeline.NamedBarrier(6, 2 * WG_SIZE)
        bar_max_done = pipeline.NamedBarrier(8, WG_SIZE + WARP_SIZE)
        bar_sum_done = pipeline.NamedBarrier(10, WG_SIZE + WARP_SIZE)
        bar_qk_issued = pipeline.NamedBarrier(11, 2 * WARP_SIZE)
        bar_pv_issued = pipeline.NamedBarrier(12, 2 * WARP_SIZE)
        bar_max_turn = pipeline.NamedBarrier(13, NUM_SOFTMAX_WG * WG_SIZE)
        bar_o_done = pipeline.NamedBarrier(9, 5 * WARP_SIZE)

        # Per-WG barrier pairs use consecutive ids: id + 0 for WG1, id + 1 for WG2.
        def phase_of(nbar, phase):
            return pipeline.NamedBarrier(nbar.barrier_id + phase, nbar.num_threads)

        mbar_reduce_ptr = None
        if cutlass.const_expr(self.use_atomic_reduction):
            # One single-use mbarrier per butterfly step and value (max, sum): the peer CTA's DSMEM
            # store completes the transaction on it.
            mbar_reduce = smem_alloc(
                cutlass.Int64, cute.make_layout((MAX_CLUSTER_REDUCE_STEPS * 2,)), alignment=8
            )
            mbar_reduce_ptr = mbar_reduce.iterator
            if warp_idx == split_stats_warp_id and lane_idx < MAX_CLUSTER_REDUCE_STEPS * 2:
                local_mbar = mbar_reduce_ptr + lane_idx
                cute.arch.mbarrier_init(local_mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(
                    local_mbar, tile_hq * self.qk_acc_dtype.width // 8
                )
            cute.arch.cluster_arrive_relaxed()

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

        if cutlass.const_expr(self.use_atomic_reduction):
            # Peers must have initialized their mbarriers before anyone stores into their DSMEM.
            cute.arch.cluster_wait()
        cute.arch.sync_threads()

        # More splits than KV blocks: this CTA has no work. It still publishes max = -inf and
        # sum = 0 so the split-stats warp finishes, but writes no O (the combine kernel skips it).
        if is_empty_split:
            if warp_idx == split_stats_warp_id:
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
            if wg_idx == correction_wg_id:
                bar_max_done.arrive()
                if lane_idx < tile_hq:
                    sRowSumWarp[wg_warp_idx, lane_idx] = Float32(0.0)
                bar_sum_done.arrive()

        elif warp_idx == load_q_store_o_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            # Q: all query heads of this KV head, loaded once, one 64-dim chunk per stage.
            q_tiler = (tile_hq_mma, tile_k)
            q_coord = (hq_block, None, bh_idx)
            gQ_blk = cute.local_tile(mQ, q_tiler, q_coord)
            gQ = thr_mma_qk.partition_B(gQ_blk)
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
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
                split_store_idx = 0 if cutlass.const_expr(self.use_atomic_reduction) else split_id
                o_blk_coord = (dm, hq_block, bh_idx, split_store_idx)
                gO_blk = cute.local_tile(mO_partial_tma, o_partial_tiler, o_blk_coord)
                gO_mma = cute.flat_divide(gO_blk, o_mma_tiler)
                if cutlass.const_expr(self.use_atomic_reduction):
                    # O was already scaled by exp2(M_split - M) / L in the correction WG; the TMA
                    # reduce-add accumulates the splits directly in the output.
                    cute_ext.tma_reduce_store(
                        sO[None, None, dm, 0],
                        gO_mma[None, None, 0, 0],
                        kind=cute.ReductionKind.ADD,
                        cta_v_map=o_cta_v_map,
                    )
                else:
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
            # stream is K0 K1 K2 V0 K3 V1 ... and finally the trailing V tiles.
            kv_tiler = (tile_m, tile_k)
            if cutlass.const_expr(self.use_pdl):
                # The KV cache (new token appended) may come from the previous kernel.
                cute.arch.griddepcontrol_wait()
            kv_token = cutlass.Boolean(True)
            for s in cutlass.range(split_id, kv_lookahead_blocks + num_n_blocks, num_splits):
                if s < num_n_blocks:
                    k_block_coord = (s, 0, bh_idx)
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
                            cute_ext.tma_load(gK_cur, sK_cur, kv_mbar)
                            pipeline_kv.producer_commit_and_advance()
                            kv_token = pipeline_kv.producer_try_acquire()

                if s >= kv_lookahead_blocks:
                    v_block_coord = (0, s - kv_lookahead_blocks, bh_idx)
                    gVt = cute.local_tile(mV, (hdim, tile_n), v_block_coord)
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
                            cute_ext.tma_load(gVt_cur, sVt_cur, kv_mbar)
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
                valid_rows = seqlen_k - s * tile_n
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    if thr_in_wg + sm * tile_m >= valid_rows:
                        scores_r[None, sm].fill(-Float32.inf)

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
                lane_max *= softmax_scale_log2

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

                bar_max_full.arrive_and_wait()
                cute.autovec_copy(sRowMax, row_max_r)
                bar_turn_release.arrive()

                # P = exp2(S * scale * log2(e) - M(j)) in bf16 / fp16 into SMEM for the PV MMA.
                _, p_idx = pipeline_p.producer_acquire_and_get_stage(token=p_token)

                row_max_tile = row_max_valid.load().reshape((tile_hq, 1))
                probs = exp2_fast(softmax_scale_log2 * scores - row_max_tile)

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

            # Correction WG. Thread t owns TMEM lane t (one head_dim row of O, all heads) and keeps one
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
                    bar_max_full.arrive_and_wait()
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

                bar_max_full.arrive_and_wait()
                if s == num_iters - self.num_stages_o - 1:
                    bar_max_done.arrive()
                row_max_cur = Float32(0.0)
                if lane_owns_head:
                    row_max_cur = sRowMax[lane_idx]
                bar_max_empty.arrive()

                o_token = pipeline_o.consumer_try_wait()
                _, _o_idx = pipeline_o.consumer_wait_and_get_stage(token=o_token)

                acc_scale_lane = exp2_fast(row_max_prev2 - row_max_cur)
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

            if cutlass.const_expr(self.use_atomic_reduction):
                # The split-stats warp publishes exp2(M_split - M) / L in sRowMax after the
                # cluster reduction; apply it before the TMA reduce-add of O.
                bar_max_done.arrive_and_wait()
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

            # Merged, still unnormalized O goes to SMEM; warp 3 TMA-stores it as this split's partial.
            tOrO_tail0 = tOrO_tail[None, None, None, None, 0].load()
            tOrO_tail1 = tOrO_tail[None, None, None, None, 1].load()
            tOrO_tail0 = tOrO_tail0.reshape((vec_size, num_d_tiles))
            tOrO_tail1 = tOrO_tail1.reshape((vec_size, num_d_tiles))
            tOrO_final = tOrO_tail1 + acc_scale.load().reshape((vec_size, 1)) * tOrO_tail0
            if cutlass.const_expr(self.use_atomic_reduction):
                tOrO_final *= tLrL.load().reshape((vec_size, 1))
            tOsO = cute_ext.partition(
                sO[None, None, None, 0],
                thr_r2s_O.thr_idx,
                layout_tv=thr_r2s_O.layout_dst_tv_tiled,
                tiler=cute.core._pack_tile(thr_r2s_O.tiler_mn),
            )
            tOsO.store(tOrO_final.to(mO_partial_tma.element_type).reshape(tOsO.shape))

            cute.arch.fence_view_async_shared()
            bar_o_done.arrive()

        # After its loads, warp 2 writes this split's per-head row max and row sum.
        if warp_idx == split_stats_warp_id:
            if cutlass.const_expr(not self.use_atomic_reduction):
                assert mMax_partial is not None
                assert mSum_partial is not None
                self.store_split_stats(
                    tile_hq,
                    (split_id, hq_block, bh_idx),
                    lane_idx,
                    bar_max_done,
                    bar_sum_done,
                    sRowMax,
                    sRowSumWarp,
                    mMax_partial,
                    mSum_partial,
                )
            else:
                sMax_cluster = cute.make_tensor(
                    sRowMax.iterator, cute.make_layout(shape=(tile_hq,), stride=(1,))
                )
                sSum_cluster = cute.make_tensor(
                    sRowSumWarp.iterator,
                    cute.make_layout(shape=(tile_hq, WARPS_PER_WG), stride=(1, tile_hq)),
                )
                self.reduce_splits_cluster(
                    tile_hq,
                    num_splits,
                    split_id,
                    lane_idx,
                    bar_max_done,
                    bar_sum_done,
                    mbar_reduce_ptr,
                    sMax_cluster,
                    sSum_cluster,
                    sReduce,
                    mLSE,
                    hq_block,
                    bh_idx,
                    qhead_per_kvhead,
                )
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

    @staticmethod
    @cute.jit
    def store_split_stats(
        tile_hq: int,
        block_idx: Tuple[Int32, Int32, Int32],
        lane_idx: Int32,
        bar_max_done,
        bar_sum_done,
        sRowMax: cute.Tensor,
        sRowSumWarp: cute.Tensor,
        mMax_partial: cute.Tensor,
        mSum_partial: cute.Tensor,
    ):
        split_id, hq_block, bh_idx = block_idx

        stats_tiler = (1, tile_hq)
        stats_coord = (split_id, hq_block, bh_idx)
        gMax_partial = cute.local_tile(mMax_partial, stats_tiler, stats_coord)
        gSum_partial = cute.local_tile(mSum_partial, stats_tiler, stats_coord)

        lane_owns_head = lane_idx < tile_hq
        lane_owns_head &= hq_block * tile_hq + lane_idx < mMax_partial.shape[1]

        cute.arch.fence_acq_rel_cta()
        # Only a plain store per split: no global atomic, so nothing to reset between calls.
        bar_max_done.arrive_and_wait()
        if lane_owns_head:
            row_max_lane = sRowMax[lane_idx]
            gMax_partial[0, lane_idx] = row_max_lane

        bar_sum_done.arrive_and_wait()
        if lane_owns_head:
            row_sum_w01 = sRowSumWarp[0, lane_idx] + sRowSumWarp[1, lane_idx]
            row_sum_w23 = sRowSumWarp[2, lane_idx] + sRowSumWarp[3, lane_idx]
            gSum_partial[0, lane_idx] = row_sum_w01 + row_sum_w23

    # Atomic mode: cluster-wide reduction of the per-split row max and row sum.
    #   1. butterfly max over the splits through DSMEM -> global max M on every split
    #   2. local row sum (4 warps) rescaled by exp2(M_split - M), butterfly sum -> global L
    #   3. publish this split's O scale exp2(M_split - M) / L in sRowMax for the correction WG
    # Each step stores to the peer CTA (split_id ^ 2^step) and waits on its own single-use mbarrier.
    @staticmethod
    @cute.jit
    def reduce_splits_cluster(
        tile_hq: int,
        num_splits: Int32,
        split_id: Int32,
        lane_idx: Int32,
        bar_max_done,
        bar_sum_done,
        mbar_reduce_ptr,
        sMax: cute.Tensor,
        sSum: cute.Tensor,
        sReduce: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        hq_block: Int32,
        bh_idx: Int32,
        qhead_per_kvhead: Int32,
    ):
        vec_bits = tile_hq * Float32.width
        copy_bits = min(vec_bits, 128)
        store_threads = vec_bits // copy_bits
        store_values = copy_bits // Float32.width
        dsmem_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyDsmemStoreOp(), Float32, num_bits_per_copy=copy_bits
        )
        dsmem_copy = cute.make_tiled_copy(
            dsmem_atom,
            cute.make_ordered_layout((store_threads, store_values), order=(1, 0)),
            (tile_hq,),
        )
        thr_copy = dsmem_copy.get_slice(lane_idx)
        tMs = thr_copy.partition_S(sMax)
        tSs = thr_copy.partition_S(sSum)
        tRs = thr_copy.partition_S(sReduce)
        tHc = thr_copy.partition_S(cute.make_identity_tensor((tile_hq,)))
        vec_shape = thr_copy.partition_D(sMax).shape
        row_max = cute.make_rmem_tensor(vec_shape, Float32)
        row_max_split = cute.make_rmem_tensor(vec_shape, Float32)

        cute.arch.fence_acq_rel_cta()
        bar_max_done.arrive_and_wait()
        is_reduce_lane = lane_idx < store_threads
        if is_reduce_lane:
            row_max_split.store(tMs.load())
            row_max.store(row_max_split.load())
            for step in cutlass.range_constexpr(MAX_CLUSTER_REDUCE_STEPS):
                xor_mask = 1 << step
                if xor_mask < num_splits:
                    peer = split_id ^ xor_mask
                    tRs_local = tRs[None, None, step, 0]
                    tRs_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRs_local.iterator, peer), tRs_local.layout
                    )
                    local_mbar = mbar_reduce_ptr + step
                    cute.copy(
                        dsmem_atom,
                        row_max,
                        tRs_peer,
                        mbar_ptr=cute.arch.map_dsmem_ptr(local_mbar, peer),
                    )
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    peer_max = tRs_local.load()
                    for i in cutlass.range_constexpr(cute.size(row_max)):
                        row_max[i] = cute.arch.fmax(row_max[i], peer_max[i])

        bar_sum_done.arrive_and_wait()
        if is_reduce_lane:
            row_sum = tSs[None, None, 0].load()
            for w in cutlass.range_constexpr(1, WARPS_PER_WG):
                row_sum += tSs[None, None, w].load()
            scale = exp2_fast(row_max_split.load() - row_max.load()).reshape(row_sum.shape)
            row_sum *= scale
            for step in cutlass.range_constexpr(MAX_CLUSTER_REDUCE_STEPS):
                xor_mask = 1 << step
                if xor_mask < num_splits:
                    peer = split_id ^ xor_mask
                    row_sum_r = cute.make_rmem_tensor(vec_shape, Float32)
                    row_sum_r.store(row_sum)
                    tRs_local = tRs[None, None, step, 1]
                    tRs_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRs_local.iterator, peer), tRs_local.layout
                    )
                    local_mbar = mbar_reduce_ptr + MAX_CLUSTER_REDUCE_STEPS + step
                    cute.copy(
                        dsmem_atom,
                        row_sum_r,
                        tRs_peer,
                        mbar_ptr=cute.arch.map_dsmem_ptr(local_mbar, peer),
                    )
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    row_sum += tRs_local.load()

            inv_sum = cute.make_rmem_tensor(row_sum.shape, Float32)
            for i in cutlass.range(cute.size(row_sum.shape)):
                inv_sum[i] = cute.arch.rcp_approx(row_sum[i])
            tMs.store(scale * inv_sum.load())

            if cutlass.const_expr(mLSE is not None):
                # Every split now holds the global M and L; split 0 writes lse = ln2 * (M + log2 L).
                if split_id == 0:
                    row_sum_r = cute.make_rmem_tensor(vec_shape, Float32)
                    row_sum_r.store(row_sum)
                    for i in cutlass.range_constexpr(cute.size(row_max)):
                        head = hq_block * tile_hq + tHc[i][0]
                        if head < qhead_per_kvhead:
                            lse = -Float32.inf
                            if row_sum_r[i] > Float32(0.0):
                                lse = (
                                    row_max[i] + cute.math.log2(row_sum_r[i], fastmath=True)
                                ) * math.log(2.0)
                            mLSE[head, bh_idx] = lse

        bar_max_done.arrive()

    # Combine kernel: one CTA per (head, batch). out = sum_i w_i O_i / sum_i w_i L_i with
    # w_i = exp2(M_i - M), M = max_i M_i.
    @cute.experimental.kernel
    def combine_kernel(
        self,
        mO_partial: cute.Tensor,
        mMax_partial: cute.Tensor,
        mSum_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        num_valid_splits: Int32,
    ):
        hdim_blk, head_idx, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        hdim_per_cta = self.head_dim
        hdim_per_thread = 32 // self.dtype.width
        threads_per_cta = hdim_per_cta // hdim_per_thread
        hdim, _h, _b, _splits = mO_partial.shape
        hdim_in_bounds = True
        if hdim % hdim_per_cta != 0:
            hdim_in_bounds = hdim_blk * hdim_per_cta + tidx * hdim_per_thread < hdim

        hdim_tiler = (hdim_per_cta,)
        o_coord = (hdim_blk, head_idx, batch_idx, None)
        gO = cute.local_tile(mO, hdim_tiler, o_coord[:3])
        gO_partial = cute.local_tile(mO_partial, hdim_tiler, o_coord)

        stats_coord_cmb = (None, head_idx, batch_idx)
        gMax_partial = cute.local_tile(mMax_partial, (1,), stats_coord_cmb)
        gSum_partial = cute.local_tile(mSum_partial, (1,), stats_coord_cmb)

        smem_ptr = cute.arch.get_dyn_smem(Float32)
        split_layout = cute.make_layout((1, num_valid_splits))
        offset = num_valid_splits
        sMax_partial = cute.make_tensor(smem_ptr, split_layout)
        sSum_partial = cute.make_tensor(smem_ptr + offset, split_layout)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        tv_layout = cute.make_ordered_layout((threads_per_cta, hdim_per_thread), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (hdim_per_cta,))
        thr_copy = tiled_copy.get_slice(tidx)

        tOgO_partial = thr_copy.partition_S(gO_partial)
        tOgO_partial = tOgO_partial[None, 0, None]
        tOgO = thr_copy.partition_D(gO)
        tOgO = tOgO[None, 0]
        tOrO_final = cute.zeros_like(tOgO, Float32)

        if cutlass.const_expr(self.use_pdl):
            # The decode kernel may still be running: wait until its partials are visible.
            cute.arch.fence_acq_rel_cta()
            cute.arch.griddepcontrol_wait()

        if tidx < num_valid_splits:
            cute_ext.simt_auto_vec_copy(
                gSum_partial[None, tidx], sSum_partial[None, tidx], async_op=True
            )
            cute_ext.simt_auto_vec_copy(
                gMax_partial[None, tidx], sMax_partial[None, tidx], async_op=True
            )

        for split_idx in cutlass.range(threads_per_cta + tidx, num_valid_splits, threads_per_cta):
            cute_ext.simt_auto_vec_copy(
                gSum_partial[None, split_idx],
                sSum_partial[None, split_idx],
                async_op=True,
            )
            cute_ext.simt_auto_vec_copy(
                gMax_partial[None, split_idx],
                sMax_partial[None, split_idx],
                async_op=True,
            )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # The no-memset change: compute the global max here from the split maxima already staged in
        # SMEM, instead of reading a global max that every decode CTA had to atomically update.
        row_max = -Float32.inf
        for split_idx in cutlass.range(num_valid_splits, unroll=8):
            row_max = cute.arch.fmax(row_max, sMax_partial[0, split_idx])
        row_sum = Float32(0)
        if row_max > -Float32.inf and hdim_in_bounds:
            for split_idx in cutlass.range(num_valid_splits, unroll=8):
                row_max_split = sMax_partial[0, split_idx]
                if row_max_split > -Float32.inf:
                    acc_scale_split = exp2_fast(row_max_split - row_max)
                    row_sum += acc_scale_split * sSum_partial[0, split_idx]
                    tOrO_final += acc_scale_split * tOgO_partial[None, split_idx].load()
            tOrO_final *= cute.arch.rcp_approx(row_sum)

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if hdim_in_bounds:
            tOgO.store(tOrO_final.to(mO.element_type))

        if cutlass.const_expr(mLSE is not None):
            # Split maxima are in units of log2 (scores were scaled by softmax_scale * log2(e)),
            # so lse = ln(2) * (max + log2(sum)). An empty sequence gives -inf.
            if tidx == 0 and hdim_blk == 0:
                lse = -Float32.inf
                if row_sum > Float32(0.0):
                    lse = (row_max + cute.math.log2(row_sum, fastmath=True)) * math.log(2.0)
                mLSE[head_idx, batch_idx] = lse


def _next_pow2(x):
    if x <= 0:
        raise ValueError(f"x must be positive, got {x}")
    return 1 << (x - 1).bit_length()


# Default split count. Decode is bandwidth bound, so what matters is enough bytes in flight: each
# CTA keeps ~192 KB of K/V in its SMEM ring, and DRAM saturates with roughly `target_ctas` CTAs
# (fewer KV bytes per tile at head_dim 64 -> more CTAs). More splits than that only add per-CTA
# fixed cost and combine work. The split count is rounded down to a power of two (uneven splits
# leave some CTAs with an extra block), capped at 32, and each split keeps >= 512 tokens.
# Fitted on B200: within 2% of the best split on average, 8.5% worst case, over a 22-shape sweep.
DECODE_TARGET_CTAS = {64: 128}
DECODE_MAX_SPLITS = 32
DECODE_MIN_TOKENS_PER_SPLIT = 512


def get_decode_config(batch, num_heads, num_heads_kv, seqlen_k, head_dim=128, target_ctas=None):
    qhead_per_kvhead = num_heads // num_heads_kv
    tile_hq = min(_next_pow2(qhead_per_kvhead), 32)
    num_hq_blocks = (qhead_per_kvhead + tile_hq - 1) // tile_hq
    num_bh = batch * num_heads_kv
    if target_ctas is None:
        target_ctas = DECODE_TARGET_CTAS.get(head_dim, 64)
    num_splits = max(1, target_ctas // (num_hq_blocks * num_bh))
    num_splits = 1 << (num_splits.bit_length() - 1)
    num_splits = min(num_splits, DECODE_MAX_SPLITS, max(1, seqlen_k // DECODE_MIN_TOKENS_PER_SPLIT))
    return (num_splits, tile_hq, num_hq_blocks, num_bh)


_compile_cache = {}


def _check_tensor(name, t, ndim, dtype=None):
    if t.dim() != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape {tuple(t.shape)}")
    if dtype is not None and t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    if t.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in the last dimension")
    elem = t.element_size()
    if t.data_ptr() % 16 != 0 or any((s * elem) % 16 != 0 for s in t.stride()[:-1]):
        raise ValueError(f"{name} must be 16-byte aligned (pointer and all but the last stride)")


def flash_attn_decode_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    seqused_k: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    reduction_mode: str = "kernel",
    use_pdl: Optional[bool] = None,
):
    """Single-token decode attention for SM100 (swap-AB, split-KV).

    q: (batch, 1, num_heads, head_dim)
    k, v: (batch, seqlen_k, num_heads_kv, head_dim); only the last dim needs to be contiguous.
    seqused_k: optional (batch,) int32, number of valid KV tokens per sequence (<= seqlen_k).
    num_splits: 0 picks a heuristic.
    reduction_mode: how the KV splits are merged.
        "kernel" (default): deterministic. Splits write fp32 partials, a combine kernel merges them.
            Workspaces are plain torch.empty: no initialization is needed between calls.
        "atomic": the splits of one (batch, KV head) form a thread-block cluster, reduce their
            row max / row sum through distributed shared memory and TMA-reduce-add normalized O
            into `out`. No second kernel, but `out` is zeroed first and the low-precision
            additions are not deterministic. num_splits is rounded down to a power of two <= 16.
        "auto": "atomic" when num_splits <= 4, or num_splits == 8 with at most 64 CTAs (clusters
            of 8 beyond that do not all fit at once); else "kernel". Measured on B200 over 60
            shapes, this is within ~1% of the faster mode on average.
    use_pdl: programmatic dependent launch: the combine kernel launches early and overlaps the
        decode tail, and the decode kernel waits for its inputs with griddepcontrol.wait.
        None (default) enables it in kernel mode (B200: -0.8 us median) and disables it in atomic
        mode, where there is no combine kernel to overlap and it measured +0.5 us.
    Returns out (batch, 1, num_heads, head_dim), and lse (batch, num_heads, 1) fp32 if return_lse.
    """
    if torch.cuda.get_device_capability(q.device)[0] != 10:
        raise RuntimeError("flash_attn_decode_func requires an SM100-family GPU")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must all be float16 or all bfloat16")
    _check_tensor("q", q, 4)
    _check_tensor("k", k, 4)
    _check_tensor("v", v, 4)
    batch, seqlen_q, num_heads, hdim = q.shape
    _, seqlen_k, num_heads_kv, _ = k.shape
    if seqlen_q != 1:
        raise ValueError("flash_attn_decode_func only supports seqlen_q == 1")
    if k.shape != v.shape or k.shape[0] != batch or k.shape[3] != hdim:
        raise ValueError(
            f"shape mismatch: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}"
        )
    if num_heads % num_heads_kv != 0:
        raise ValueError("num_heads must be divisible by num_heads_kv")
    if hdim % 64 != 0 or hdim > 256:
        raise ValueError(f"head_dim must be a multiple of 64 and <= 256, got {hdim}")
    if seqused_k is not None:
        _check_tensor("seqused_k", seqused_k, 1, torch.int32)
        if seqused_k.shape[0] != batch:
            raise ValueError("seqused_k must have shape (batch,)")

    default_splits, tile_hq, num_hq_blocks, num_bh = get_decode_config(
        batch, num_heads, num_heads_kv, seqlen_k, hdim
    )
    num_splits = default_splits if num_splits < 1 else min(num_splits, math.ceil(seqlen_k / 256))
    if reduction_mode not in ("kernel", "atomic", "auto"):
        raise ValueError(
            f"reduction_mode must be 'kernel', 'atomic' or 'auto', got {reduction_mode}"
        )
    if reduction_mode == "auto":
        num_ctas = num_splits * num_hq_blocks * num_bh
        use_cluster = num_splits <= 4 or (num_splits == 8 and num_ctas <= 64)
        reduction_mode = "atomic" if use_cluster else "kernel"
    use_atomic = reduction_mode == "atomic"
    if use_pdl is None:
        use_pdl = not use_atomic
    if use_atomic:
        # one cluster per (batch, KV head): power-of-two split count, at most 16
        num_splits = min(1 << (num_splits.bit_length() - 1), 16)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(hdim)
    if out is None:
        out = torch.zeros_like(q) if use_atomic else torch.empty_like(q)
    elif use_atomic:
        out.zero_()
    o_partial = max_partial = sum_partial = None
    if not use_atomic:
        o_partial = torch.empty(
            (num_splits, batch, num_heads, hdim), device=q.device, dtype=torch.float32
        )
        max_partial = torch.empty(
            (batch, num_heads, num_splits), device=q.device, dtype=torch.float32
        )
        sum_partial = torch.empty_like(max_partial)
    lse = (
        torch.empty((batch, num_heads, 1), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )

    q3, out3 = q[:, 0], out[:, 0]
    to_cute = lambda t: None if t is None else from_dlpack(t.detach(), assumed_align=16)  # noqa: E731
    tensors = [
        to_cute(t)
        for t in (
            q3,
            k,
            v,
            o_partial,
            max_partial,
            sum_partial,
            out3,
            None if lse is None else lse[:, :, 0],
            seqused_k,
        )
    ]
    problem_size = (batch, num_heads, num_heads_kv, seqlen_k, hdim)
    dtype = _TORCH_TO_CUTE_DTYPE[q.dtype]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    softmax_scale_log2 = softmax_scale * LOG2_E
    key = (
        problem_size,
        dtype,
        num_splits,
        q3.stride(),
        k.stride(),
        v.stride(),
        out3.stride(),
        return_lse,
        seqused_k is not None,
        reduction_mode,
        use_pdl,
    )
    if key not in _compile_cache:
        fa_decode = FlashAttentionDecodeSm100(
            tile_hq,
            256,
            hdim,
            dtype,
            (num_splits, num_hq_blocks, num_bh),
            reduction_mode=reduction_mode,
            use_pdl=use_pdl,
        )
        fa_decode.can_implement(problem_size)
        compiled = cute_ext.compile(fa_decode, problem_size, *tensors, softmax_scale_log2, stream)
        compiled.engine.initialize()
        _compile_cache[key] = compiled
    _compile_cache[key](problem_size, *tensors, softmax_scale_log2, stream)
    return (out, lse) if return_lse else out
