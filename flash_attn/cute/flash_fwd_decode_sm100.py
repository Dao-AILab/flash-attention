# SM100 swap-AB single-token decode, adapted from NVIDIA CUTLASS gqa_decode_opt
# (examples/python/CuTeDSL/cute_ext/blackwell/attention/gqa_decode_opt.py).
# It computes S^T = K Q^T and O^T = V^T P^T, so KV tokens fill the 128-row MMA M dimension
# and the query heads of one KV head fill the small N dimension.
#
# Core contract: Q is (B,H,Dqk), K is (B,S,Hkv,Dqk), V is (B,S,Hkv,Dv), O is (splits,B,H,Dv),
# optional LSE is (splits,B,H) fp32 natural log. Empty splits write O=0 and, when requested, LSE=-inf.
# The caller launches the shared FA4 combine if splits>1. With per-sequence split counts (from the
# prepare-scheduler kernel), split CTAs past their sequence's count exit without writing.
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
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.typing import Float32, Int32

from cutlass import const_expr
from quack import copy_utils

from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.config import FwdConfig
from flash_attn.cute.compat import SmemAllocator, TmemAllocator, get_smem_capacity_in_bytes
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100SwapAB
from flash_attn.cute.pipeline import NamedBarrier
from flash_attn.cute.tile_scheduler import SingleTileScheduler, TileSchedulerArguments
from flash_attn.cute.pack_gqa import pack_gqa_layout, select_modes
from flash_attn.cute.paged_kv import resolve_paged_token
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tensormap_utils import acquire_tensormap, bound_rows, cta_tensormap_slot
from flash_attn.cute.softmax import apply_score_mod_inner
from flash_attn.cute.mask import call_mask_mod
from flash_attn.cute import utils
from flash_attn.cute.utils import AuxData, scalar_to_ssa, ssa_to_scalar


exp2_fast = partial(cute.math.exp2, fastmath=True)
warp_reduce_max = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_atomic_max = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")

WARP_SIZE = 32
WARPS_PER_WG = 4
WG_SIZE = WARPS_PER_WG * WARP_SIZE
MMA_N_MIN = 8
NUM_SOFTMAX_WG = 2
NUM_WG = NUM_SOFTMAX_WG + 2


@cute.jit
def mask_mod_keep(
    mask_mod: cutlass.Constexpr,
    vec_size: cutlass.Constexpr[int],
    batch_ssa,
    head_ssa,
    q_ssa,
    kv_idx: Int32,
    seqlen_k: Int32,
    seqlen_info,
    aux_data: AuxData,
) -> cutlass.Boolean:
    """Keep bit of one score under a scalar or vectorized (__vec_size__ > 1) mask_mod.

    A vectorized mask_mod returns bit-packed Uint32 words for vec_size adjacent KV
    indices. Evaluate the aligned chunk that contains kv_idx and extract its bit.
    Like the standard SM100 path, chunk indices past seqlen_k wrap when aux tensors
    are present so the callback never reads out of bounds.
    """
    kv_ssa = scalar_to_ssa(kv_idx, Int32)
    if const_expr(vec_size == 1):
        keep = cutlass.Boolean(
            ssa_to_scalar(
                call_mask_mod(mask_mod, batch_ssa, head_ssa, q_ssa, kv_ssa, seqlen_info, aux_data)
            )
        )
    else:
        offset = kv_idx % vec_size
        chunk_start = kv_idx - offset
        kv_vec = cute.make_rmem_tensor(vec_size, Int32)
        for j in cutlass.range_constexpr(vec_size):
            kv_j = chunk_start + j
            if const_expr(aux_data.tensors is not None):
                kv_j = kv_j if kv_j < seqlen_k else kv_j % seqlen_k
            kv_vec[j] = kv_j
        words = cute.make_rmem_tensor(max(1, vec_size // 32), cutlass.Uint32)
        words.store(
            call_mask_mod(
                mask_mod,
                batch_ssa.broadcast_to((vec_size,)),
                head_ssa.broadcast_to((vec_size,)),
                q_ssa.broadcast_to((vec_size,)),
                kv_vec.load(),
                seqlen_info,
                aux_data,
            )
        )
        word = words[offset // 32]
        keep = cutlass.Boolean(utils.shr_u32(word, cutlass.Uint32(offset % 32)) & cutlass.Uint32(1))
    return keep


@cute.jit
def kv_cache_unit(
    unit: Int32,
    bh_idx: Int32,
    num_heads_kv: Int32,
    unit_tokens: cutlass.Constexpr[int],
    mPageTable: cute.Tensor | None,
    page_size: Int32,
    seqlen_k: Int32,
    num_pages: Int32,
) -> Tuple[Int32, Int32]:
    """Coordinates (unit, (kv head, batch) index) of logical KV unit `unit` in the K/V views.

    A unit is one 128-token MMA subtile. With a page table the unit lies in one physical page
    (page_size % unit_tokens == 0): the unit index is the offset within the page, and the page
    replaces the batch in the (kv head, batch) mode. Resolving per unit lets a 256-token KV
    block span two pages. A unit wholly past seqlen_k owns no page: it gets an out-of-range page
    so its TMA loads zero-fill (still crediting the full transaction bytes) instead of reading
    an unowned page, which may hold NaN, or a page-table entry past the end of the row. A unit
    that straddles seqlen_k loads its whole page rows, so tokens past seqlen_k inside a used page
    must be finite (the paged contract of test_flash_attn_paged_padding_contract).
    """
    block, bh = unit, bh_idx
    if const_expr(mPageTable is not None):
        block, bh = Int32(0), num_pages * num_heads_kv + bh_idx % num_heads_kv
        if unit * unit_tokens < seqlen_k:
            page, offset = resolve_paged_token(
                mPageTable[bh_idx // num_heads_kv, None], unit * unit_tokens, page_size
            )
            page = cute.arch.make_warp_uniform(page)
            offset = cute.arch.make_warp_uniform(offset)
            block = offset // unit_tokens
            bh = page * num_heads_kv + bh_idx % num_heads_kv
    return block, bh


@cute.jit
def kv_cache_block(
    block: Int32,
    num_units: cutlass.Constexpr[int],
    bh_idx: Int32,
    num_heads_kv: Int32,
    unit_tokens: cutlass.Constexpr[int],
    mPageTable: cute.Tensor | None,
    page_size: Int32,
    seqlen_k: Int32,
    num_pages: Int32,
):
    """kv_cache_unit coordinates of the num_units units of KV block `block`.

    A unit continues its predecessor's page unless that unit ends the page or the unit starts
    past seqlen_k, so pages of 256+ tokens take one page-table read per block.
    """
    unit0 = block * num_units
    unit, bh = kv_cache_unit(
        unit0, bh_idx, num_heads_kv, unit_tokens, mPageTable, page_size, seqlen_k, num_pages
    )
    coords = [(unit, bh)]
    for i in cutlass.range_constexpr(1, num_units):
        next_unit, next_bh = unit + i, bh
        if const_expr(mPageTable is not None):
            if (unit + i) * unit_tokens >= page_size or (unit0 + i) * unit_tokens >= seqlen_k:
                next_unit, next_bh = kv_cache_unit(
                    unit0 + i,
                    bh_idx,
                    num_heads_kv,
                    unit_tokens,
                    mPageTable,
                    page_size,
                    seqlen_k,
                    num_pages,
                )
        coords.append((next_unit, next_bh))
    return coords


@cute.jit
def load_kv_stage(
    pipeline_kv: pipeline.PipelineTmaUmma,
    kv_state: pipeline.PipelineState,
    kv_token: cutlass.Boolean,
    tma_atom: cute.CopyAtom,
    tXgX: cute.Tensor,
    tXsX: cute.Tensor,
    tma_desc_ptr: cute.Pointer | None = None,
) -> Tuple[pipeline.PipelineState, cutlass.Boolean]:
    """TMA one K or V^T tile tXgX into the next ring stage of tXsX (staged by its last mode).

    Acquires the stage (skipping the wait if kv_token says it is already free), issues the copy
    on its full barrier, and returns the advanced producer state with the next stage's token.
    tma_desc_ptr replaces the atom's descriptor (a per-CTA bounded copy).
    """
    pipeline_kv.producer_acquire(kv_state, kv_token)
    barrier = pipeline_kv.producer_get_barrier(kv_state)
    desc_kwargs = {"tma_desc_ptr": tma_desc_ptr} if const_expr(tma_desc_ptr is not None) else {}
    cute.copy(tma_atom, tXgX, tXsX[None, kv_state.index], tma_bar_ptr=barrier, **desc_kwargs)
    pipeline_kv.producer_commit(kv_state)
    kv_state.advance()
    return kv_state, pipeline_kv.producer_try_acquire(kv_state)


def gemm_view(mX, order, head_idx=None, qhead_per_kvhead=None, num_heads_kv=None):
    """mX with its modes permuted to order and the (kv head, batch) modes 2 and 3 grouped.

    With head_idx, the head mode is first split into (qhead_per_kvhead, num_heads_kv) as in
    pack_gqa_layout, so order indexes (qhead_per_kvhead, mode 0, ..., num_heads_kv, ...).
    """
    if head_idx is not None:
        mX = cute.flatten(pack_gqa_layout(mX, qhead_per_kvhead, num_heads_kv, head_idx))
    return cute.group_modes(select_modes(mX, order), 2, 4)


class _SwapABCore:
    def __init__(
        self,
        tile_hq,
        tile_n,
        head_dim,
        dtype,
        score_mod=None,
        mask_mod=None,
        is_local=False,
        *,
        head_dim_v,
    ):
        self.tile_hq = tile_hq
        self.tile_n = tile_n
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.dtype = dtype
        self.qk_acc_dtype = cutlass.Float32
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.mask_vec_size = getattr(mask_mod, "__vec_size__", 1)
        assert self.mask_vec_size % 32 == 0 or 32 % self.mask_vec_size == 0, (
            "mask_mod __vec_size__ must divide 32 or be a multiple of 32"
        )
        self.is_local = is_local
        self.check_empty_rows = score_mod is not None or mask_mod is not None or is_local

        assert head_dim > 0 and head_dim % 16 == 0
        assert self.head_dim_v > 0 and self.head_dim_v % 16 == 0
        assert tile_hq in (1, 2, 4, 8, 16, 32)
        assert tile_n > 0 and tile_n % 128 == 0 and tile_n <= 256

    def make_shared_storage(self, sQ_layout, sK_layout, sP_layout):
        tile_hq_mma = self.mma_tiler[1]

        @cute.struct
        class SharedStorage:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 1]
            mbar_kv: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_kv]
            mbar_s: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_s]
            mbar_p: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_p]
            mbar_o: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages_o]
            tmem_holding_buf: Int32
            # Per-head running max for softmax/correction; split stats later publish 1/L here.
            sRowMax: cute.struct.Align[cute.struct.MemRange[Float32, tile_hq_mma], 128]
            # Per-warp row sums from the correction WG for the split statistics.
            sRowSumWarp: cute.struct.Align[
                cute.struct.MemRange[Float32, WARPS_PER_WG * self.tile_hq], 128
            ]
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sP_layout)], 1024]
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 1024]
            # One ring for both K and V^T tiles (same bytes, two swizzled views), reused for O.
            sKV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 1024]

        return SharedStorage

    def plan_kv_ring(self, tiled_mma_qk, sQ_layout, sP_layout):
        """Size the K/V ring one stage below the deepest that fits; sets the stages and storage.

        The deepest ring measured up to 4% slower on B200 for head_dim 64 MHA decode (13 vs 12
        stages), and the decode heuristics were tuned at this depth.
        """

        def storage_for(num_stages_kv):
            self.num_stages_kv = num_stages_kv
            sK_layout = sm100_utils.make_smem_layout_a(
                tiled_mma_qk, self.mma_tiler, self.dtype, num_stages_kv
            )
            self.shared_storage = self.make_shared_storage(sQ_layout, sK_layout, sP_layout)
            return sK_layout

        smem_capacity = get_smem_capacity_in_bytes("sm_100")
        tile_m, _, tile_k = self.mma_tiler
        num_stages_kv = smem_capacity // (tile_m * tile_k * self.dtype.width // 8)
        storage_for(num_stages_kv)
        while self.shared_storage.size_in_bytes() > smem_capacity:
            num_stages_kv -= 1
            storage_for(num_stages_kv)
        assert num_stages_kv > 1
        return storage_for(num_stages_kv - 1)

    @cute.jit
    def __call__(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32],
        mQ: cute.Tensor,  # (batch, num_heads, head_dim), head_dim contiguous
        mK: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim), head_dim contiguous
        mV: cute.Tensor,  # (batch, seqlen_k, num_heads_kv, head_dim_v), last dim contiguous
        mO_partial: cute.Tensor,  # (num_splits, batch, num_heads, head_dim_v)
        mLSE_partial: cute.Tensor | None,  # (num_splits, batch, num_heads), natural log
        mSeqUsedK: cute.Tensor | None,  # (batch,) int32: KV tokens actually used per sequence
        softmax_scale: Float32,
        stream: cuda.CUstream,
        window_size_left: Int32 | None = None,
        window_size_right: Int32 | None = None,
        aux_data: AuxData = AuxData(),
        mPageTable: cute.Tensor | None = None,
        mCuSeqlensQ: cute.Tensor | None = None,
        mNumSplitsDynamic: cute.Tensor | None = None,  # (batch,) int32: splits per sequence
        tensormap_workspace: cute.Tensor | None = None,  # (num_ctas, 16) Int64 (ragged V only)
    ):
        hdim = self.head_dim
        hdim_v = self.head_dim_v
        batch, num_heads, num_heads_kv, _ = problem_size
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        qhead_per_kvhead = num_heads // num_heads_kv
        tile_hq = self.tile_hq
        tile_n = self.tile_n

        # GEMM views over the FA layouts with the real strides (so e.g. a KV cache slice works
        # without a copy). The (kv head, batch) pair is one hierarchical mode, indexed by
        # bh_idx = batch_idx * num_heads_kv + kv_head_idx.
        # QK: K is operand A (M = tokens, K = head_dim), Q is operand B (N = query heads of this
        # KV head). PV: V^T is operand A (M = head_dim_v, K = tokens).
        mK_qk = gemm_view(mK, [1, 3, 2, 0])  # (seqlen_k, d, (h_kv, b))
        mVt_pv = gemm_view(mV, [3, 1, 2, 0])  # (d_v, seqlen_k, (h_kv, b))
        gqa = (qhead_per_kvhead, num_heads_kv)
        mQ_qk = gemm_view(mQ, [0, 3, 2, 1], 1, *gqa)  # (qpk, d, (h_kv, b))
        # (d_v, qpk, (h_kv, b), ns)
        mO_partial_tma = gemm_view(mO_partial, [4, 0, 3, 2, 1], 2, *gqa)
        mLSE_partial_fwd = None
        if cutlass.const_expr(mLSE_partial is not None):
            # (ns, qpk, (h_kv, b))
            mLSE_partial_fwd = gemm_view(mLSE_partial, [1, 0, 3, 2], 2, *gqa)

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
        # blocks), L: one row-sum mailbox per softmax WG. The K/V ring gets the SMEM left over.
        self.num_stages_q = num_k_tiles
        self.num_stages_p = 4
        self.num_stages_s = 4
        self.num_stages_l = NUM_SOFTMAX_WG
        self.num_stages_o = 2

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
        sP_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, (None, tile_hq_mma, tile_n), self.dtype, self.num_stages_p
        )
        sO_layout_atom = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, mO_partial_tma.element_type
        )

        # O staging tile uses the output dtype and reuses the K/V ring SMEM at the end.
        O_shape = (num_d_tiles * tile_m, tile_hq_mma)
        sO_layout = cute.tile_to_shape(sO_layout_atom, O_shape, order=(1, 0))
        sO_layout = cute.flat_divide(sO_layout, (tile_m, tile_hq_mma))

        sK_layout = self.plan_kv_ring(tiled_mma_qk, sQ_layout, sP_layout)
        sVt_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler, self.dtype, self.num_stages_kv
        )
        # The V^T view and the O staging tile alias the K ring. A V^T stage must span exactly the
        # bytes of a K stage: the KV pipeline arms one transaction count for both.
        ring_bytes = cute.size_in_bytes(self.dtype, sK_layout)
        k_stage_bytes = cute.size_in_bytes(self.dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        vt_stage_bytes = cute.size_in_bytes(self.dtype, cute.select(sVt_layout, mode=[0, 1, 2]))
        assert vt_stage_bytes == k_stage_bytes
        assert cute.size_in_bytes(self.dtype, sVt_layout) <= ring_bytes
        o_bytes = cute.size_in_bytes(mO_partial_tma.element_type, sO_layout)
        assert o_bytes <= ring_bytes

        # TMA atoms; the returned coordinate tensors replace the GEMM views in the kernel.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        cluster_shape_vmnk = (1, 1, 1, 1)
        tma_atom_Q, mQ_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_qk,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_qk,
            cluster_shape_vmnk,
        )
        tma_atom_K, mK_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK_qk,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_qk,
            cluster_shape_vmnk,
        )
        tma_atom_V, mVt_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mVt_pv,
            cute.select(sVt_layout, mode=[0, 1, 2]),
            self.mma_tiler,
            tiled_mma_pv,
            cluster_shape_vmnk,
        )
        # Bound dense V loads by seqused_k, as FlashAttentionForwardSm100 does: the load warp
        # rewrites its CTA's copy of the V^T descriptor so tokens at or past the sequence end
        # arrive zero-filled. Masking P alone cannot keep unused NaN cache rows out of the PV
        # MMA (0 * NaN). Pages past seqlen_k already load from an out-of-range page.
        self.ragged_tma_V = (
            mSeqUsedK is not None
            and mPageTable is None
            and not (isinstance(mVt_pv.stride[1], int) and mVt_pv.stride[1] == 0)
        )
        if const_expr(self.ragged_tma_V):
            assert tensormap_workspace is not None, "ragged V TMA loads need a tensormap workspace"
        tma_atom_O, mO_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mO_partial_tma,
            cute.select(sO_layout, mode=[0, 1]),
            (tile_m, tile_hq_mma),
        )

        # Shared tile scheduler: one tile per (split, KV head, sequence); tile_hq covers every
        # query head of a KV head (adapter-asserted). With per-sequence split counts it packs
        # the sequence's count into the split index and launches split-major.
        self.num_splits_dynamic = mNumSplitsDynamic is not None
        tile_sched_args = TileSchedulerArguments(
            1,
            num_heads_kv,
            batch,
            mO_partial.shape[0],
            problem_size[3],
            hdim,
            hdim_v,
            mQ.shape[0],
            tile_shape_mn=(tile_hq, tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            qhead_per_kvhead_packgqa=qhead_per_kvhead,
            element_size=self.dtype.width // 8,
            is_split_kv=True,
            num_splits_dynamic_ptr=mNumSplitsDynamic,
            split_major=self.num_splits_dynamic,
        )
        tile_sched_params = SingleTileScheduler.to_underlying_arguments(tile_sched_args)
        grid = SingleTileScheduler.get_grid_shape(tile_sched_params)
        self.attn_kernel(
            problem_size,
            tma_atom_Q,
            mQ_tma,
            tma_atom_K,
            mK_tma,
            tma_atom_V,
            mVt_tma,
            tma_atom_O,
            mO_tma,
            mO_partial,
            mLSE_partial_fwd,
            mSeqUsedK,
            softmax_scale_log2,
            softmax_scale,
            tiled_mma_qk,
            tiled_mma_pv,
            sQ_layout,
            sK_layout,
            sVt_layout,
            sO_layout,
            sP_layout,
            window_size_left,
            window_size_right,
            aux_data,
            mPageTable,
            mCuSeqlensQ,
            tile_sched_params,
            mVt_pv if const_expr(self.ragged_tma_V) else None,
            tensormap_workspace if const_expr(self.ragged_tma_V) else None,
        ).launch(
            grid=grid,
            block=(NUM_WG * WG_SIZE, 1, 1),
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            # Overlap this launch with the prepare-scheduler kernel that writes the split counts.
            use_pdl=self.num_splits_dynamic,
        )

    @cute.kernel
    def attn_kernel(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32],
        tma_atom_Q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        mV: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        mO_partial_tma: cute.Tensor,
        mO_partial_raw: cute.Tensor,  # direct stores; mO_partial_tma holds TMA coordinates
        mLSE_partial: cute.Tensor | None,
        mSeqUsedK: cute.Tensor | None,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sVt_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        aux_data: AuxData,
        mPageTable: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        tile_sched_params: SingleTileScheduler.Params,
        mV_gmem: cute.Tensor | None,
        tensormap_workspace: cute.Tensor | None,
    ):
        if const_expr(self.num_splits_dynamic):
            cute.arch.griddepcontrol_wait()  # the split counts
        tile_scheduler = SingleTileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        _, kv_head_idx, batch_idx, split_idx = work_tile.tile_idx
        is_valid = work_tile.is_valid_tile
        if const_expr(self.num_splits_dynamic):
            # Splits past the sequence's count exit before any SMEM, TMEM or barrier work and
            # write nothing: the combine reads only the first count partials. Running them as
            # empty splits instead measured 15.8 vs 9.5 us (B300, b8 s1024 paged).
            num_splits_seq = cute.arch.make_warp_uniform(split_idx >> 16)
            is_valid = is_valid & ((split_idx & 0xFFFF) < num_splits_seq)
        _b, _h, num_heads_kv, seqlen_k = problem_size
        bh_idx = batch_idx * num_heads_kv + kv_head_idx
        seqlen_info = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=1,
            seqlen_k_static=seqlen_k,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=1,
            tile_n=self.tile_n,
        )
        if is_valid:
            self.attn_cta(
                problem_size,
                tma_atom_Q,
                mQ,
                tma_atom_K,
                mK,
                tma_atom_V,
                mV,
                tma_atom_O,
                mO_partial_tma,
                mO_partial_raw,
                mLSE_partial,
                softmax_scale_log2,
                softmax_scale,
                tiled_mma_qk,
                tiled_mma_pv,
                sQ_layout,
                sK_layout,
                sVt_layout,
                sO_layout,
                sP_layout,
                window_size_left,
                window_size_right,
                aux_data,
                mPageTable,
                seqlen_info,
                split_idx,
                bh_idx,
                tile_sched_params.num_splits,
                mV_gmem,
                tensormap_workspace,
            )

    @cute.jit
    def attn_cta(
        self,
        problem_size: Tuple[Int32, Int32, Int32, Int32],
        tma_atom_Q: cute.CopyAtom,
        mQ: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        mK: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        mV: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        mO_partial_tma: cute.Tensor,
        mO_partial_raw: cute.Tensor,
        mLSE_partial: cute.Tensor | None,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sVt_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        window_size_left: Int32 | None,
        window_size_right: Int32 | None,
        aux_data: AuxData,
        mPageTable: cute.Tensor | None,
        seqlen_info: SeqlenInfoQK,
        split_idx: Int32,
        bh_idx: Int32,
        num_splits: Int32,
        mV_gmem: cute.Tensor | None,
        tensormap_workspace: cute.Tensor | None,
    ):
        """One split CTA: its contiguous KV block range (BlockInfo) of sequence bh_idx."""
        # Warp roles (see header): WG0 = MMA + TMA warps, WG1 / WG2 = softmax for even / odd KV blocks,
        # WG3 = correction. Warp 2 publishes this split's LSE and reciprocal row sum.
        mma_qk_warp_id, mma_pv_warp_id, load_kv_warp_id, load_q_store_o_warp_id = 0, 1, 2, 3
        softmax_wg_ids = tuple(range(1, 1 + NUM_SOFTMAX_WG))
        correction_wg_id = 1 + NUM_SOFTMAX_WG
        split_stats_warp_id = load_kv_warp_id
        thread_id, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        wg_idx = warp_idx // WARPS_PER_WG
        wg_warp_idx = warp_idx % WARPS_PER_WG

        hdim = self.head_dim
        hdim_v = self.head_dim_v
        _b, num_heads, num_heads_kv, _ = problem_size
        tile_n = self.tile_n
        tile_hq = self.tile_hq
        qhead_per_kvhead = num_heads // num_heads_kv
        seqlen_k = seqlen_info.seqlen_k
        # This split's KV blocks are [n_block_min, n_block_max). With per-sequence split counts
        # the scheduler packed the count into split_idx's high 16 bits.
        packed_splits = self.num_splits_dynamic
        block_info = BlockInfo(
            tile_m=1,
            tile_n=tile_n,
            is_causal=False,
            is_local=self.is_local,
            is_split_kv=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            num_splits=num_splits,
            pack_split_idx=packed_splits,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen_info, 0, split_idx, num_splits
        )
        num_n_blocks = cutlass.max(n_block_max - n_block_min, 0)
        split_id = split_idx
        if const_expr(packed_splits):
            split_id = split_idx & 0xFFFF
        # Q, O and LSE rows: the sequence index, or with varlen Q the token of the sequence
        # (max_seqlen_q == 1, host-checked). A sequence without a query token takes the empty-split
        # path, and its out-of-range row fails the direct-store guards.
        num_q_rows = mO_partial_raw.shape[1]
        q_row = bh_idx // num_heads_kv
        if const_expr(seqlen_info.has_cu_seqlens_q):
            has_q = seqlen_info.seqlen_q > 0
            q_row = seqlen_info.offset_q if has_q else num_q_rows
            num_n_blocks = num_n_blocks if has_q else Int32(0)
        q_bh_idx = q_row * num_heads_kv + bh_idx % num_heads_kv

        mma_tiler = self.mma_tiler
        tile_m, tile_hq_mma, tile_k = self.mma_tiler
        num_s_subtiles, num_k_tiles, num_d_tiles, num_pv_k_tiles = self.subtile_counts

        mma_kq_instr_shape_k = cute.size(tiled_mma_qk.shape_mnk, mode=[2])
        mma_vp_instr_shape_k = cute.size(tiled_mma_pv.shape_mnk, mode=[2])
        qk_k_blocks = tile_k // mma_kq_instr_shape_k
        pv_k_blocks = tile_k // mma_vp_instr_shape_k

        # The 32-head tile needs redistribution for its softmax / correction row state.
        # O-tail size alone does not capture the row-vector state or compiler liveness.
        # Keep sufficient registers in the MMA / load warps as well.
        setmaxreg = tile_hq > 16
        # Per-thread setmaxnreg targets: 64 + 2 * 120 + 208 = 512 = 64K registers / 128 threads.
        num_regs_mma_load, num_regs_softmax, num_regs_correction = 64, 120, 208

        is_empty_split = num_n_blocks == 0

        if warp_idx == load_kv_warp_id:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        storage = SmemAllocator().allocate(self.shared_storage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)

        sK = storage.sKV.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sVt_iter = cute.recast_ptr(sK.iterator, sVt_layout.inner, dtype=self.dtype)
        sVt = cute.make_tensor(sVt_iter, sVt_layout.outer)

        # The O staging tile is only written after the last PV MMA, so it can reuse the ring.
        o_dtype = mO_partial_raw.element_type
        sO_iter = cute.recast_ptr(sK.iterator, sO_layout.inner, dtype=o_dtype)
        sO = cute.make_tensor(sO_iter, sO_layout.outer)

        thr_mma_pv = tiled_mma_pv.get_slice(0)
        thr_mma_qk = tiled_mma_qk.get_slice(0)

        # TMEM columns: S (4 stages x 2 sub-tiles of 128 tokens) from 0, L (one row-sum mailbox per
        # softmax WG), then O (2 slots x ceil(head_dim_v / 128) tiles). These views address TMEM from
        # column 0 instead of the allocator's pointer, which holds only because the kernel allocates
        # every column: keep num_tmem_cols at the maximum.
        acc_shape = thr_mma_qk.partition_shape_C(mma_tiler[:2])
        num_s_cols = self.num_stages_s * num_s_subtiles * tile_hq_mma
        num_l_cols = self.num_stages_l * tile_hq_mma
        num_o_cols = self.num_stages_o * num_d_tiles * tile_hq_mma
        num_tmem_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        assert num_s_cols + num_l_cols + num_o_cols <= num_tmem_cols
        # ((tile_m, tile_hq_mma), 1, 1, subtile, stage)
        tStS = thr_mma_qk.make_fragment_C(
            cute.append(cute.append(acc_shape, num_s_subtiles), self.num_stages_s)
        )
        tLtL = thr_mma_qk.make_fragment_C(cute.append(acc_shape, self.num_stages_l))
        tLtL = cute.make_tensor(tStS.iterator + num_s_cols, tLtL.layout)
        # ((tile_m, tile_hq_mma), 1, 1, d tile, slot)
        tOtO = thr_mma_pv.make_fragment_C(
            cute.append(cute.append(acc_shape, num_d_tiles), self.num_stages_o)
        )
        tOtO = cute.make_tensor(tStS.iterator + num_s_cols + num_l_cols, tOtO.layout)

        tLtL_0 = tLtL[(None, None), 0, 0, 0]
        t2r_rep = tcgen05.Repetition(tile_hq)
        t2r_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(t2r_rep, tcgen05.Pack.NONE), self.qk_acc_dtype
        )
        r2t_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(t2r_rep, tcgen05.Unpack.NONE), self.qk_acc_dtype
        )
        r2s_atom_P = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        r2s_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), o_dtype)
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

        sRowMax = storage.sRowMax.get_tensor(vec_layout)
        sRowSumWarp = storage.sRowSumWarp.get_tensor(
            cute.make_layout(shape=(WARPS_PER_WG, tile_hq), stride=(tile_hq, 1))
        )
        lane_owns_head = lane_idx < tile_hq
        if warp_idx == load_q_store_o_warp_id:
            if lane_owns_head:
                sRowMax[lane_idx] = -Float32.inf

        # The KV ring carries K two blocks ahead of V: K(j) is loaded together with V(j - 2).
        # Both MMA warps read the ring with their own cursors: each skips the other's stages and
        # syncs on bar_pv_issued, so the cursors stay within one window and mbarrier phases are
        # never misread.
        kv_lookahead = 2

        # Named barriers (hardware barriers that only count arriving threads):
        #   bar_max_full / bar_max_empty   softmax WG <-> correction WG hand-off of the running max
        #   bar_sum_full / bar_sum_empty   row-sum mailbox hand-off, one pair per softmax WG (+phase)
        #   bar_max_turn                   token passed between the two softmax WGs (see softmax)
        #   bar_qk_issued / bar_pv_issued  keep the two MMA warps in K/V ring order
        #   bar_max_done / bar_sum_done    final max / sum ready for the split-stats warp
        #   bar_o_done                     final O staged in SMEM, ready for the TMA store
        B = NamedBarrierFwdSm100SwapAB
        # Softmax and correction wait at different sites: unaligned wait.
        bar_max_full = NamedBarrier(int(B.MaxFull), 2 * WG_SIZE)
        bar_max_empty = NamedBarrier(int(B.MaxEmpty), 2 * WG_SIZE)
        bar_sum_full = NamedBarrier(int(B.SumFullWG1), 2 * WG_SIZE)
        bar_sum_empty = NamedBarrier(int(B.SumEmptyWG1), 2 * WG_SIZE)
        bar_max_done = NamedBarrier(int(B.MaxDone), WG_SIZE + WARP_SIZE)
        bar_stats_ready = NamedBarrier(int(B.StatsReady), WG_SIZE + WARP_SIZE)
        bar_sum_done = NamedBarrier(int(B.SumDone), WG_SIZE + WARP_SIZE)
        bar_qk_issued = NamedBarrier(int(B.QKIssued), 2 * WARP_SIZE)
        bar_pv_issued = NamedBarrier(int(B.PVIssued), 2 * WARP_SIZE)
        bar_max_turn = NamedBarrier(int(B.MaxTurnWG1), NUM_SOFTMAX_WG * WG_SIZE)
        bar_o_done = NamedBarrier(int(B.ODone), WG_SIZE + WARP_SIZE)

        mbar_q_full = storage.mbar_q.data_ptr()
        if warp_idx == load_q_store_o_warp_id:
            q_tx_bytes = cute.size_in_bytes(self.dtype, sQ_layout)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_q_full, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_q_full, q_tx_bytes)
            # Q: all query heads of this KV head, loaded once, one 64-dim chunk per stage. Only
            # warp 3 initializes and produces this barrier, so the load starts before the
            # CTA-wide prologue barrier; warp 0 waits on it only after that barrier.
            if not is_empty_split:
                gQ = cute.local_tile(mQ, (tile_hq_mma, tile_k), (0, None, q_bh_idx))
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), thr_mma_qk.partition_B(gQ), sQ
                )
                for dk in cutlass.range_constexpr(num_k_tiles):
                    load_Q(src_idx=dk, dst_idx=dk, tma_bar_ptr=mbar_q_full)
        # mbarrier pipelines: KV (TMA -> MMA), S (QK MMA -> softmax), P (softmax -> PV MMA),
        # O (PV MMA -> correction). Arrival counts: one elected TMA arrive or tcgen05.commit per
        # MMA/TMA side, one arrive per thread on the softmax / correction side.
        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        wg_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, WG_SIZE)
        kv_tx_bytes = cute.size_in_bytes(self.dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        pipeline_kv = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_kv.data_ptr(),
            num_stages=self.num_stages_kv,
            producer_group=one_thread,
            consumer_group=one_thread,
            tx_count=kv_tx_bytes,
            defer_sync=True,
        )
        pipeline_s = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s.data_ptr(),
            num_stages=self.num_stages_s,
            producer_group=one_thread,
            consumer_group=wg_threads,
            defer_sync=True,
        )
        pipeline_p = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p.data_ptr(),
            num_stages=self.num_stages_p,
            producer_group=wg_threads,
            consumer_group=one_thread,
            defer_sync=True,
        )
        pipeline_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_o.data_ptr(),
            num_stages=self.num_stages_o,
            producer_group=one_thread,
            consumer_group=wg_threads,
            defer_sync=True,
        )
        Producer, Consumer = pipeline.PipelineUserType.Producer, pipeline.PipelineUserType.Consumer

        # The pipelines initialize their mbarriers from warp 0, so warp 1 (PV MMA) allocates TMEM
        # meanwhile (allocating from warp 0 measured ~1-2% slower on short decodes). The TmemPtr
        # barrier publishes both the allocation and the mbarrier inits to the CTA.
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=NamedBarrier(int(B.TmemPtr), NUM_WG * WG_SIZE),
            allocator_warp_id=mma_pv_warp_id,
        )
        tmem.allocate(num_tmem_cols)
        tmem.relinquish_alloc_permit()
        cute.arch.mbarrier_init_fence()
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

        # Every empty split must publish O=0 as well as LSE=-inf: the shared combine
        # may load its O even with a zero weight, and zero times an uninitialized NaN is NaN.
        if is_empty_split:
            if wg_idx == correction_wg_id:
                bar_max_done.arrive()
                if lane_idx < tile_hq:
                    sRowSumWarp[wg_warp_idx, lane_idx] = Float32(0.0)
                bar_sum_done.arrive()
            for o_idx in cutlass.range(thread_id, hdim_v * tile_hq, NUM_WG * WG_SIZE):
                head = o_idx // hdim_v
                if head < qhead_per_kvhead and q_row < num_q_rows:
                    global_head = (bh_idx % num_heads_kv) * qhead_per_kvhead + head
                    mO_partial_raw[split_id, q_row, global_head, o_idx % hdim_v] = (
                        mO_partial_raw.element_type(0.0)
                    )

        elif warp_idx == load_q_store_o_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            # Wait until the correction WG has staged the final O, then TMA-store this split's partial.
            bar_o_done.arrive_and_wait()
            for dm in cutlass.range_constexpr(num_d_tiles):
                gO = cute.local_tile(
                    mO_partial_tma, (tile_m, tile_hq_mma), (dm, 0, q_bh_idx, split_id)
                )
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1), sO[None, None, dm, 0], gO, single_stage=True
                )
                store_O()
            cute.arch.cp_async_bulk_commit_group()
            # The ring SMEM must stay valid until the bulk store has read it.
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        elif warp_idx == load_kv_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            # Producer for the shared ring. Loop iteration s pushes K(s) and V(s - lookahead), so the
            # stream is K0 K1 K2 | V0 K3 V1 ... and finally the trailing V tiles V(N-3) V(N-2) V(N-1).
            kv_tiler = (tile_m, tile_k)
            kv_state = pipeline.make_pipeline_state(Producer, self.num_stages_kv)
            kv_token = cutlass.Boolean(True)
            page_size = mK.shape[0]
            num_pages = mK.shape[2][1]
            tensormap_manager = None
            tensormap_V_gmem = None
            v_desc_ptr = None
            # Only the sequence's last block can hold rows past seqlen_k, so only a CTA ending
            # in a partial block loads that block through a bounded descriptor: this CTA's slot,
            # with the V^T rows extent shrunk to the sequence end. The acquire fence waits for
            # that block's first V load, which hides the rewrite behind the earlier loads.
            bound_last_v = num_n_blocks > 0 and n_block_max * tile_n > seqlen_k
            if const_expr(self.ragged_tma_V):
                tensormap_manager, tensormap_V_gmem, v_desc_ptr = cta_tensormap_slot(
                    tensormap_workspace
                )
                if bound_last_v:
                    tensormap_manager.init_tensormap_from_atom(
                        tma_atom_V, tensormap_V_gmem, load_kv_warp_id
                    )
                    tensormap_manager.fence_tensormap_initialization()
                    tensormap_manager.update_tensormap(
                        (bound_rows(mV_gmem, seqlen_k),),
                        (tma_atom_V,),
                        (tensormap_V_gmem,),
                        load_kv_warp_id,
                        (None,),
                    )
            for s in cutlass.range(kv_lookahead + num_n_blocks):
                if s < num_n_blocks:
                    k_units = kv_cache_block(
                        s + n_block_min,
                        num_s_subtiles,
                        bh_idx,
                        num_heads_kv,
                        tile_m,
                        mPageTable,
                        page_size,
                        seqlen_k,
                        num_pages,
                    )
                    for sm in cutlass.range_constexpr(num_s_subtiles):
                        k_unit, k_bh = k_units[sm]
                        gK = cute.local_tile(mK, (tile_m, hdim), (k_unit, 0, k_bh))
                        gK_sub = cute.zipped_divide(gK, kv_tiler)[(None, None), (0, None)]
                        tKsK, tKgK = cpasync.tma_partition(
                            tma_atom_K,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(thr_mma_qk.partition_A(gK_sub), 0, 3),
                        )
                        for dk in cutlass.range_constexpr(num_k_tiles):
                            kv_state, kv_token = load_kv_stage(
                                pipeline_kv, kv_state, kv_token, tma_atom_K, tKgK[None, dk], tKsK
                            )

                if s >= kv_lookahead:
                    v_block = s - kv_lookahead + n_block_min
                    bounded_v = None
                    if const_expr(self.ragged_tma_V):
                        bounded_v = bound_last_v and v_block == n_block_max - 1
                        if bounded_v:
                            acquire_tensormap(tensormap_manager, tensormap_V_gmem)
                    v_units = kv_cache_block(
                        v_block,
                        num_s_subtiles,
                        bh_idx,
                        num_heads_kv,
                        tile_m,
                        mPageTable,
                        page_size,
                        seqlen_k,
                        num_pages,
                    )
                    for sk in cutlass.range_constexpr(num_pv_k_tiles):
                        # Consecutive 64-token V^T tiles; each 128-token unit has its own page.
                        if cutlass.const_expr(sk % (tile_m // tile_k) == 0):
                            v_unit, v_bh = v_units[sk // (tile_m // tile_k)]
                            gVt_blk = cute.zipped_divide(
                                cute.local_tile(mV, (hdim_v, tile_m), (0, v_unit, v_bh)), kv_tiler
                            )
                        gVt_sub = gVt_blk[(None, None), (None, sk % (tile_m // tile_k))]
                        tVsV, tVgV = cpasync.tma_partition(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sVt, 0, 3),
                            cute.group_modes(thr_mma_pv.partition_A(gVt_sub), 0, 3),
                        )
                        for dm in cutlass.range_constexpr(num_d_tiles):
                            load_v = partial(
                                load_kv_stage,
                                pipeline_kv,
                                tma_atom=tma_atom_V,
                                tXgX=tVgV[None, dm],
                                tXsX=tVsV,
                            )
                            if const_expr(self.ragged_tma_V):
                                if bounded_v:
                                    kv_state, kv_token = load_v(
                                        kv_state=kv_state,
                                        kv_token=kv_token,
                                        tma_desc_ptr=v_desc_ptr,
                                    )
                                else:
                                    kv_state, kv_token = load_v(
                                        kv_state=kv_state, kv_token=kv_token
                                    )
                            else:
                                kv_state, kv_token = load_v(kv_state=kv_state, kv_token=kv_token)
            # No producer_tail: the cluster is one CTA, and each MMA warp issues its ring releases
            # before the S / O commits that the softmax / correction WGs wait on before exit.

        elif warp_idx == mma_qk_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            pv_tiles_per_iter = num_d_tiles * num_pv_k_tiles
            mma_atom = cute.make_mma_atom(tiled_mma_qk.op)
            tCrK = tiled_mma_qk.make_fragment_A(sK)
            tCrQ = tiled_mma_qk.make_fragment_B(sQ)

            # QK MMA warp: S^T(j) = K(j) Q^T, two 128-token sub-tiles per 256-token block.
            cute.arch.mbarrier_wait(mbar_q_full, phase=0)

            kv_state = pipeline.make_pipeline_state(Consumer, self.num_stages_kv)
            s_state = pipeline.make_pipeline_state(Producer, self.num_stages_s)
            for s in cutlass.range(num_n_blocks):
                s_token = pipeline_s.producer_try_acquire(s_state)

                # Skip the PV warp's V(j-3) stages and wait until it has issued them (see kv_lookahead).
                if s >= kv_lookahead + 1:
                    for _ in cutlass.range_constexpr(pv_tiles_per_iter):
                        kv_state.advance()
                bar_pv_issued.arrive_and_wait()
                k_token = pipeline_kv.consumer_try_wait(kv_state)

                pipeline_s.producer_acquire(s_state, s_token)
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    mma_atom.set(tcgen05.Field.ACCUMULATE, False)
                    tStS_slice = tStS[None, None, None, sm, s_state.index]
                    for dk in cutlass.range_constexpr(num_k_tiles):
                        pipeline_kv.consumer_wait(kv_state, k_token)
                        k_stage_idx = kv_state.index
                        for instr_idx in cutlass.range_constexpr(qk_k_blocks):
                            k_slice = tCrK[None, None, instr_idx, k_stage_idx]
                            q_slice = tCrQ[None, None, instr_idx, dk]
                            cute.gemm(mma_atom, tStS_slice, k_slice, q_slice, tStS_slice)
                            mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                        pipeline_kv.consumer_release(kv_state)
                        kv_state.advance()
                        if sm == num_s_subtiles - 1 and dk == num_k_tiles - 1:
                            bar_qk_issued.arrive()
                        else:
                            k_token = pipeline_kv.consumer_try_wait(kv_state)
                pipeline_s.producer_commit(s_state)
                s_state.advance()

            for _ in cutlass.range_constexpr(kv_lookahead):
                bar_pv_issued.arrive_and_wait()
                bar_qk_issued.arrive()

        elif warp_idx == mma_pv_warp_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_mma_load)

            qk_tiles_per_iter = num_s_subtiles * num_k_tiles
            mma_atom = cute.make_mma_atom(tiled_mma_pv.op)
            # The correction WG zero-fills both O slots, so every PV MMA accumulates.
            mma_atom.set(tcgen05.Field.ACCUMULATE, True)
            tCrVt = tiled_mma_pv.make_fragment_A(sVt)
            tCrP = tiled_mma_pv.make_fragment_B(sP)

            # PV MMA warp: O^T(slot j % 2) += V^T(j) P^T(j). Warm-up skips K(0), K(1) and releases the
            # QK warp's first bar_pv_issued waits (see kv_lookahead).
            kv_state = pipeline.make_pipeline_state(Consumer, self.num_stages_kv)
            p_state = pipeline.make_pipeline_state(Consumer, self.num_stages_p)
            o_state = pipeline.make_pipeline_state(Producer, self.num_stages_o)
            bar_pv_issued.arrive()
            for prefetch_iter in cutlass.range_constexpr(kv_lookahead):
                if prefetch_iter < num_n_blocks:
                    for _ in cutlass.range_constexpr(qk_tiles_per_iter):
                        kv_state.advance()
                bar_qk_issued.arrive_and_wait()
                bar_pv_issued.arrive()

            for s in cutlass.range(num_n_blocks):
                if s + kv_lookahead < num_n_blocks:
                    for _ in cutlass.range_constexpr(qk_tiles_per_iter):
                        kv_state.advance()
                bar_qk_issued.arrive_and_wait()

                p_token = pipeline_p.consumer_try_wait(p_state)
                v_token = pipeline_kv.consumer_try_wait(kv_state)
                o_token = pipeline_o.producer_try_acquire(o_state)

                pipeline_p.consumer_wait(p_state, p_token)
                p_idx = p_state.index
                # The O slot for block j is free once the correction WG has rescaled what PV(j - 2) left there.
                pipeline_o.producer_acquire(o_state, o_token)
                o_idx = o_state.index
                for dk in cutlass.range_constexpr(num_pv_k_tiles):
                    for dm in cutlass.range_constexpr(num_d_tiles):
                        pipeline_kv.consumer_wait(kv_state, v_token)
                        v_stage_idx = kv_state.index
                        tOtO_slot = tOtO[None, None, None, dm, o_idx]
                        for instr_idx in cutlass.range_constexpr(pv_k_blocks):
                            v_slice = tCrVt[None, None, instr_idx, v_stage_idx]
                            p_slice = tCrP[None, None, dk * pv_k_blocks + instr_idx, p_idx]
                            cute.gemm(mma_atom, tOtO_slot, v_slice, p_slice, tOtO_slot)
                        if dm == num_d_tiles - 1 and dk == num_pv_k_tiles - 1:
                            bar_pv_issued.arrive()
                        pipeline_kv.consumer_release(kv_state)
                        kv_state.advance()
                        if dm != num_d_tiles - 1 or dk != num_pv_k_tiles - 1:
                            v_token = pipeline_kv.consumer_try_wait(kv_state)
                pipeline_p.consumer_release(p_state)
                p_state.advance()
                pipeline_o.producer_commit(o_state)
                o_state.advance()
            # The correction tail always merges two slots; with a single block, commit an empty second one.
            if num_n_blocks == 1:
                pipeline_o.producer_acquire(o_state)
                pipeline_o.producer_commit(o_state)
                o_state.advance()
            pipeline_o.producer_tail(o_state)

        elif wg_idx in softmax_wg_ids:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_decrease(num_regs_softmax)

            # Softmax WGs. Each thread owns one token row of each 128-token sub-tile, i.e. 2 rows of the
            # block, and tile_hq query-head columns.

            tStS_0 = tStS[(None, None), 0, 0, None, 0]
            t2r_rep_S = tcgen05.Repetition(tile_hq)
            if cutlass.const_expr(tile_hq_mma == tile_hq):
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
            # (token row, head, subtile) -> one P stage's logical (head, token) index, head + N * token.
            sP_view_layout = cute.make_layout(
                shape=(tile_m, tile_hq_mma, num_s_subtiles),
                stride=(tile_hq_mma, 1, tile_m * tile_hq_mma),
            )

            tSrS = cute.make_rmem_tensor(tSrS_layout, self.qk_acc_dtype)
            tSrP = cute.make_rmem_tensor(tSrP_layout, self.dtype)
            tLrL = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            row_max_r = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            scores_r = cute.make_tensor(tSrS.iterator, scores_layout)
            probs_r = cute.make_tensor(tSrP.iterator, scores_layout)
            row_sum_valid = tLrL[(None, 0), 0, 0]
            row_max_valid = row_max_r[(None, 0), 0, 0]

            phase = wg_idx - softmax_wg_ids[0]
            tLtL_cur = tLtL[(None, None), 0, 0, phase]

            # WG2 handles the odd blocks: shift its pipeline cursors by one block and give WG1 the first
            # turn on the shared running max.
            s_state = pipeline.make_pipeline_state(Consumer, self.num_stages_s)
            p_state = pipeline.make_pipeline_state(Producer, self.num_stages_p)
            if phase == 1:
                s_state.advance()
                p_state.advance()
                bar_max_turn.arrive_w_index(phase ^ 1)
                if num_n_blocks == 1:
                    bar_sum_full.arrive_w_index(phase)

            for n_iter in cutlass.range(phase, num_n_blocks, NUM_SOFTMAX_WG):
                s = n_iter
                s_token = pipeline_s.consumer_try_wait(s_state)
                p_token = pipeline_p.producer_try_acquire(p_state)

                pipeline_s.consumer_wait(s_state, s_token)
                row_max_r.fill(-Float32.inf)
                tStS_cur = tStS[(None, None), 0, 0, None, s_state.index]
                cute.copy(tiled_t2r_S, thr_t2r_S.partition_S(tStS_cur), tSrS)
                cute.arch.fence_view_async_tmem_load()
                pipeline_s.consumer_release(s_state)
                s_state.advance()

                # Mask the token rows past seqlen_k in the last block.
                valid_rows = seqlen_k - (s + n_block_min) * tile_n
                for sm in cutlass.range_constexpr(num_s_subtiles):
                    if thr_in_wg + sm * tile_m >= valid_rows:
                        scores_r[None, sm].fill(-Float32.inf)

                if cutlass.const_expr(self.check_empty_rows):
                    batch_idx = bh_idx // num_heads_kv
                    # score_mod runs on one score per call: a register column holds one KV token
                    # across tile_hq heads, not the consecutive tokens of one head that the
                    # shared helper's vector chunks assume.
                    score_tiles = cute.zipped_divide(scores_r, (1, 1))
                    for sm in cutlass.range_constexpr(num_s_subtiles):
                        kv_idx = (s + n_block_min) * tile_n + thr_in_wg + sm * tile_m
                        # (q, kv) coordinate of this thread's token; the decode query is q = 0.
                        tScS = cute.domain_offset((0, kv_idx), cute.make_identity_tensor((1, 1)))
                        for gi in cutlass.range_constexpr(tile_hq):
                            value = -Float32.inf
                            if gi < qhead_per_kvhead and kv_idx < seqlen_k:
                                global_head = (bh_idx % num_heads_kv) * qhead_per_kvhead + gi
                                if cutlass.const_expr(self.score_mod is not None):
                                    apply_score_mod_inner(
                                        score_tiles[None, (gi, sm)],
                                        tScS,
                                        self.score_mod,
                                        batch_idx,
                                        global_head,
                                        softmax_scale,
                                        1,
                                        self.qk_acc_dtype,
                                        aux_data,
                                        None,  # the guard keeps indices in bounds: no wrapping
                                        seqlen_info,
                                        constant_q_idx=0,
                                    )
                                value = scores_r[gi, sm]
                                keep = cutlass.Boolean(True)
                                if cutlass.const_expr(self.is_local):
                                    if cutlass.const_expr(window_size_left is not None):
                                        keep &= kv_idx >= seqlen_k - 1 - window_size_left
                                    if cutlass.const_expr(window_size_right is not None):
                                        keep &= kv_idx <= seqlen_k - 1 + window_size_right
                                if cutlass.const_expr(self.mask_mod is not None):
                                    # Build fresh SSA values per score: a callback that indexes
                                    # (b[0]) an SSA value shared across dynamic regions fails IR
                                    # verification.
                                    keep &= mask_mod_keep(
                                        self.mask_mod,
                                        self.mask_vec_size,
                                        scalar_to_ssa(batch_idx, Int32),
                                        scalar_to_ssa(global_head, Int32),
                                        scalar_to_ssa(Int32(0), Int32),
                                        kv_idx,
                                        seqlen_k,
                                        seqlen_info,
                                        aux_data,
                                    )
                                value = value if keep else -Float32.inf
                            scores_r[gi, sm] = value

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

                # M(j) = max(M(j-1), blockmax(j)); WG1/WG2 own even/odd j.
                # Arrival counts alone can admit both WGs and make correction miss M(j).
                # bar_max_turn: take -> wait for correction to read M(j-1) -> atomic max -> publish -> pass.
                bar_max_turn.arrive_and_wait_w_index(phase)
                bar_max_empty.arrive_and_wait()
                if lane_owns_head:
                    sRowMax_ptr = sRowMax.iterator + sRowMax.layout(lane_idx)
                    smem_atomic_max(sRowMax_ptr, lane_max)

                bar_max_full.arrive_and_wait_unaligned()
                cute.autovec_copy(sRowMax, row_max_r)
                bar_max_turn.arrive_w_index(phase ^ 1)

                # P = exp2(S * scale * log2(e) - M(j)) in bf16 / fp16 into SMEM for the PV MMA.
                pipeline_p.producer_acquire(p_state, p_token)

                if cutlass.const_expr(self.check_empty_rows):
                    # Only the local exponent bias is zeroed; shared final maxima stay -inf.
                    for gi in cutlass.range_constexpr(tile_hq):
                        if row_max_valid[gi] == -Float32.inf:
                            row_max_valid[gi] = Float32(0.0)
                row_max_tile = row_max_valid.load().reshape((tile_hq, 1))
                probs = exp2_fast(softmax_scale_log2 * scores - row_max_tile)

                if cutlass.const_expr(tile_hq < tile_hq_mma):
                    tSrP.fill(self.dtype(0))

                probs_r.store(probs.to(self.dtype))
                sP_view = cute.composition(sP[None, None, None, p_state.index], sP_view_layout)
                cute.autovec_copy(tSrP, thr_r2s_P.partition_D(sP_view))
                cute.arch.fence_view_async_shared()
                pipeline_p.producer_commit(p_state)
                p_state.advance()

                # This WG's TMEM mailbox carries this thread's two-token sums per head, not P.
                # Cross-thread reduction runs once in the correction tail.
                row_sum_valid.store(
                    probs.reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        (None, 0),
                    ).reshape((tile_hq,))
                )

                bar_sum_empty.arrive_and_wait_w_index(phase)
                cute.copy(tiled_r2t, tLrL, thr_r2t.partition_D(tLtL_cur))
                cute.arch.fence_view_async_tmem_store()
                bar_sum_full.arrive_w_index(phase)

                # Skip the block that the other softmax WG handles.
                s_state.advance()
                p_state.advance()

        elif wg_idx == correction_wg_id:
            if cutlass.const_expr(setmaxreg):
                cute.arch.setmaxregister_increase(num_regs_correction)

            # Correction WG. Thread t owns TMEM lane t (one head_dim_v row of O, all heads) and keeps one
            # row-sum accumulator per O slot.
            l_slot_shape = (*vec_layout.shape, self.num_stages_o)
            l_slot_stride = (*vec_layout.stride, vec_size)
            row_sum_slot_layout = cute.make_layout(l_slot_shape, stride=l_slot_stride)
            tLrL = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            row_sum_final = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            acc_scale = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            row_sum_acc = cute.make_rmem_tensor(row_sum_slot_layout, self.qk_acc_dtype)

            row_sum_acc.fill(Float32(0.0))

            zeros_r = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
            zeros_r.fill(Float32(0.0))
            for phase in cutlass.range_constexpr(self.num_stages_o):
                for dm in cutlass.range_constexpr(num_d_tiles):
                    tOtO_slot = tOtO[(None, None), 0, 0, dm, phase]
                    cute.copy(tiled_r2t, zeros_r, thr_r2t.partition_D(tOtO_slot))
            tLtL_1 = tLtL[(None, None), 0, 0, 1]
            cute.copy(tiled_r2t, zeros_r, thr_r2t.partition_D(tLtL_1))
            cute.arch.fence_view_async_tmem_store()

            # Initial tokens: the running max and both row-sum mailboxes start out empty.
            bar_max_empty.arrive()
            for phase in cutlass.range_constexpr(NUM_SOFTMAX_WG):
                bar_sum_empty.arrive_w_index(phase)

            # Read M(0) and M(1) up front: rescaling slot j needs M(j) and M(j + 2).
            row_max_prev2, row_max_prev = -Float32.inf, -Float32.inf
            for preload_idx in cutlass.range_constexpr(self.num_stages_o):
                row_max_prev2 = row_max_prev
                if not (preload_idx == 1 and num_n_blocks == 1):
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
            o_state = pipeline.make_pipeline_state(Consumer, self.num_stages_o)
            phase = 0
            for s in cutlass.range(num_n_blocks - self.num_stages_o, unroll=self.num_stages_o):
                bar_sum_full.arrive_and_wait_w_index(phase)
                tLtL_cur = tLtL[(None, None), 0, 0, phase]
                cute.copy(tiled_t2r, thr_t2r.partition_S(tLtL_cur), tLrL)
                cute.arch.fence_view_async_tmem_load()
                bar_sum_empty.arrive_w_index(phase)

                bar_max_full.arrive_and_wait_unaligned()
                if s == num_n_blocks - self.num_stages_o - 1:
                    bar_max_done.arrive()
                row_max_cur = Float32(0.0)
                if lane_owns_head:
                    row_max_cur = sRowMax[lane_idx]
                bar_max_empty.arrive()

                pipeline_o.consumer_wait(o_state, pipeline_o.consumer_try_wait(o_state))

                acc_scale_lane = exp2_fast(row_max_prev2 - row_max_cur)
                if cutlass.const_expr(self.check_empty_rows):
                    if row_max_prev2 == -Float32.inf and row_max_cur == -Float32.inf:
                        acc_scale_lane = Float32(1.0)
                for gi in cutlass.range_constexpr(tile_hq):
                    acc_scale[gi] = cute.arch.shuffle_sync(acc_scale_lane, gi)

                for dm in cutlass.range_constexpr(num_d_tiles):
                    tOtO_slot = tOtO[(None, None), 0, 0, dm, phase]
                    tOrO = cute.make_rmem_tensor(vec_layout, self.qk_acc_dtype)
                    cute.copy(tiled_t2r, thr_t2r.partition_S(tOtO_slot), tOrO)
                    cute.arch.fence_view_async_tmem_load()
                    tOrO.store(tOrO.load() * acc_scale.load())
                    cute.copy(tiled_r2t, tOrO, thr_r2t.partition_D(tOtO_slot))
                cute.arch.fence_view_async_tmem_store()
                pipeline_o.consumer_release(o_state)
                o_state.advance()

                row_sum_acc_cur = row_sum_acc[None, None, None, phase]
                tLrL.store(tLrL.load() * acc_scale.load())
                row_sum_acc_cur.store(row_sum_acc_cur.load() * acc_scale.load() + tLrL.load())

                row_max_prev2, row_max_prev = row_max_prev, row_max_cur
                phase ^= 1

            if num_n_blocks <= self.num_stages_o:
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

            tail_phase = num_n_blocks % self.num_stages_o
            for phase in cutlass.range_constexpr(self.num_stages_o):
                if tail_phase == phase:
                    final_phase = phase ^ 1

                    bar_sum_full.arrive_and_wait_w_index(phase)
                    tLtL_cur = tLtL[(None, None), 0, 0, phase]
                    cute.copy(tiled_t2r, thr_t2r.partition_S(tLtL_cur), tLrL)
                    cute.arch.fence_view_async_tmem_load()
                    bar_sum_empty.arrive_w_index(phase)

                    bar_sum_full.arrive_and_wait_w_index(final_phase)
                    tLtL_last = tLtL[(None, None), 0, 0, final_phase]
                    cute.copy(tiled_t2r, thr_t2r.partition_S(tLtL_last), row_sum_final)
                    cute.arch.fence_view_async_tmem_load()
                    bar_sum_empty.arrive_w_index(final_phase)

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

            # StatsReady publishes 1/L after the final max and four warp sums are consumed.
            bar_stats_ready.arrive_and_wait()
            cute.autovec_copy(sRowMax, tLrL)

            o_slot_shape = (*vec_layout.shape, num_d_tiles, self.num_stages_o)
            o_slot_stride = (
                *vec_layout.stride,
                vec_size,
                num_d_tiles * vec_size,
            )
            tOrO_slot_layout = cute.make_layout(o_slot_shape, stride=o_slot_stride)
            tOrO_tail = cute.make_rmem_tensor(tOrO_slot_layout, self.qk_acc_dtype)
            for s in cutlass.range_constexpr(self.num_stages_o):
                phase = tail_phase ^ s
                pipeline_o.consumer_wait(o_state, pipeline_o.consumer_try_wait(o_state))
                cute.copy(
                    tiled_t2r,
                    thr_t2r.partition_S(tOtO[(None, None), 0, 0, None, phase]),
                    tOrO_tail[None, None, None, None, s],
                )
                cute.arch.fence_view_async_tmem_load()
                pipeline_o.consumer_release(o_state)
                o_state.advance()

            # Normalize the merged O in fp32, then stage it for warp 3's ordinary TMA store.
            tOrO_tail0 = tOrO_tail[None, None, None, None, 0].load()
            tOrO_tail1 = tOrO_tail[None, None, None, None, 1].load()
            tOrO_tail0 = tOrO_tail0.reshape((vec_size, num_d_tiles))
            tOrO_tail1 = tOrO_tail1.reshape((vec_size, num_d_tiles))
            tOrO_final = tOrO_tail1 + acc_scale.load().reshape((vec_size, 1)) * tOrO_tail0
            tOrO_final *= tLrL.load().reshape((vec_size, 1))
            if cutlass.const_expr(self.check_empty_rows):
                # Select real zeros for fully masked heads, never rely on 0 * NaN.
                normalized = cute.make_rmem_tensor(
                    cute.make_layout((vec_size, num_d_tiles)), self.qk_acc_dtype
                )
                normalized.store(tOrO_final)
                for gi in cutlass.range_constexpr(tile_hq):
                    if tLrL[gi] == Float32(0.0):
                        for dm in cutlass.range_constexpr(num_d_tiles):
                            normalized[gi, dm] = Float32(0.0)
                tOrO_final = normalized.load()
            tOsO = thr_r2s_O.partition_D(sO[None, None, None, 0])
            tOsO.store(tOrO_final.to(o_dtype).reshape(tOsO.shape))

            cute.arch.fence_view_async_shared()
            bar_o_done.arrive()

        # After its loads, warp 2 normalizes this split's statistics, not other CTAs'.
        if warp_idx == split_stats_warp_id:
            self.store_split_stats(
                tile_hq,
                (split_id, q_bh_idx),
                lane_idx,
                is_empty_split,
                bar_max_done,
                bar_sum_done,
                bar_stats_ready,
                sRowMax,
                sRowSumWarp,
                mLSE_partial,
            )

        # Free TMEM once every role is done with it.
        cute.arch.sync_threads()
        tmem.free(tmem_ptr)

    @staticmethod
    @cute.jit
    def store_split_stats(
        tile_hq: int,
        block_idx: Tuple[Int32, Int32],
        lane_idx: Int32,
        is_empty_split: cutlass.Boolean,
        bar_max_done,
        bar_sum_done,
        bar_stats_ready,
        sRowMax: cute.Tensor,
        sRowSumWarp: cute.Tensor,
        mLSE_partial: cute.Tensor | None,
    ):
        split_id, bh_idx = block_idx
        lane_owns_head = lane_idx < tile_hq
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
                if lane_idx < mLSE_partial.shape[1]:
                    lse = -Float32.inf
                    if row_sum > Float32(0.0):
                        lse = (row_max_lane + cute.math.log2(row_sum, fastmath=True)) * math.log(
                            2.0
                        )
                    num_heads_kv = mLSE_partial.shape[2][0]
                    bh_coord = (bh_idx % num_heads_kv, bh_idx // num_heads_kv)
                    if bh_coord[1] < mLSE_partial.shape[2][1]:
                        mLSE_partial[split_id, lane_idx, bh_coord] = lse

        # Empty CTAs skip StatsReady, as correction skips its wait there. bar_pv_issued,
        # bar_max_turn, bar_max_empty and bar_sum_empty do end with unmatched arrivals; that is
        # harmless only because each CTA runs one tile and exits. A persistent version must
        # balance them.
        if not is_empty_split:
            bar_stats_ready.arrive()


class FlashAttentionForwardSm100SwapAB:
    """Standard forward ABI for single-query inference; interface owns allocation and combine.

    The swap-AB mainloop (_SwapABCore) is private. Split outputs are normalized FP32 O and natural-log
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
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        has_tile_count_semaphore: bool = False,
        seqlen_k_per_split: Optional[int] = None,
        *,
        config: FwdConfig,
    ):
        # validate_fwd_config fixed the shape and schedule (config.SWAPAB.config); the call
        # site's other facts must not ask for features this kernel does not implement.
        assert config.use_swapab
        assert not paged_kv_non_tma and not has_tile_count_semaphore
        assert seqlen_k_per_split is None and q_subtile_factor == kv_subtile_factor == 1
        self.head_dim = head_dim
        self.head_dim_v = head_dim if head_dim_v is None else head_dim_v
        # For one query at the end of seqused_k, causal and noncausal bounds agree.
        # pack_gqa does not alter the external tensor ABI: the core packs heads itself.
        self.is_split_kv = config.num_splits > 1
        self.is_varlen_q = is_varlen_q
        self.is_local = is_local
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.tile_hq = config.tile_m
        self.tile_n = config.tile_n

    @cute.jit
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
        tensormap_workspace: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Compile-time guards (Python None / static rank / dtype).
        assert (mCuSeqlensQ is not None) == self.is_varlen_q
        for value in (
            mCuSeqlensK,
            mSeqUsedQ,
            learnable_sink,
            descale_tensors,
            blocksparse_tensors,
            tile_count_semaphore,
            virtual_batch_idx_ptr,
            num_nheads_in_l2_ptr,
            mCuTotalMBlocks,
            mCuTotalSplitsMBlocks,
            mBlocksToBatchIdx,
        ):
            assert value is None, "swap-AB adapter: unsupported argument"
        assert len(mQ.shape) == (3 if self.is_varlen_q else 4)
        assert len(mK.shape) == 4 and len(mV.shape) == 4
        assert mQ.element_type in (cutlass.Float16, cutlass.BFloat16)
        assert mQ.element_type == mK.element_type == mV.element_type
        if const_expr(not self.is_split_kv):
            assert mO.element_type == mQ.element_type
            assert num_splits_dynamic_ptr is None
        if const_expr(mLSE is not None):
            assert mLSE.element_type == Float32

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        num_heads = mQ.shape[-2]
        batch = mCuSeqlensQ.shape[0] - 1 if const_expr(self.is_varlen_q) else mQ.shape[0]
        num_heads_kv, seqlen_k = mK.shape[2], mK.shape[1]
        if const_expr(mPageTable is not None):
            seqlen_k = mPageTable.shape[1] * mK.shape[1]
        # The core takes Q as (rows, h, d), O as (ns, rows, h, d) and LSE as (ns, rows, h), where
        # rows are sequences, or tokens with varlen Q. A missing split mode (one split, or the
        # dense seqlen_q mode of extent 1, host-checked) becomes a split mode of extent 1.
        mLSE_core = None
        if const_expr(self.is_varlen_q):
            mQ_core = mQ  # (total_q, h, d)
            if const_expr(self.is_split_kv):
                mO_core = mO  # out_partial (ns, total_q, h, d)
                mLSE_core = select_modes(mLSE, [0, 2, 1])  # lse_partial (ns, h, total_q)
            else:
                mO_core = cute.prepend_ones(mO, up_to_rank=4)  # out (total_q, h, d)
                if const_expr(mLSE is not None):
                    mLSE_core = cute.prepend_ones(select_modes(mLSE, [1, 0]), up_to_rank=3)
        else:
            mQ_core = mQ[None, 0, None, None]  # (b, h, d)
            if const_expr(self.is_split_kv):
                mO_core = mO[None, None, 0, None, None]  # out_partial (ns, b, 1, h, d)
                mLSE_core = mLSE[None, None, None, 0]  # lse_partial (ns, b, h, 1)
            else:
                mO_core = select_modes(mO, [1, 0, 2, 3])  # out (b, 1, h, d)
                if const_expr(mLSE is not None):
                    mLSE_core = select_modes(mLSE, [2, 0, 1])  # lse (b, h, 1)
        if const_expr(self.is_split_kv):
            assert mO_core.element_type == Float32 and mLSE_core is not None
        core = _SwapABCore(
            self.tile_hq,
            self.tile_n,
            self.head_dim,
            mQ.element_type,
            score_mod=self.score_mod,
            mask_mod=self.mask_mod,
            is_local=self.is_local,
            head_dim_v=self.head_dim_v,
        )
        problem_size = (batch, num_heads, num_heads_kv, seqlen_k)
        # Stock conventions carried through: window sizes as resolved by
        # _resolve_causal_local_window (dynamic Int32 or None), AuxData(tensors, scalars) as
        # converted by the interface (to_cute_aux_tensor / aux_scalar_metadata in the key).
        core(
            problem_size,
            mQ_core,
            mK,
            mV,
            mO_core,
            mLSE_core,
            mSeqUsedK,
            softmax_scale,
            stream,
            window_size_left,
            window_size_right,
            aux_data,
            mPageTable,
            mCuSeqlensQ,
            num_splits_dynamic_ptr,
            tensormap_workspace,
        )
