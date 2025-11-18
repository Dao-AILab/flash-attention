# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_prepare_scheduler.cu
# from Cutlass C++ to Cute-DSL.

from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr, Constexpr

from flash_attn.cute.fast_math import FastDivmod


class FlashPrepareScheduler:
    def __init__(
        self,
        packgqa: bool = False,
        sort: bool = False,
        num_batch: int = 1,
    ):
        self.packgqa = packgqa
        self.sort = sort
        self.num_batch = num_batch
        self.num_threads_per_warp = 32
        self.k_num_batch_per_warp = 31
        self.num_warps = min((num_batch + 30) // 31, 32)
        self.num_ctas = (num_batch + (31 * 32 - 1)) // (31 * 32)

    @cute.jit
    def __call__(
        self,
        seqlen_q_static: int,
        seqlen_k_static: int,
        seqlen_k_new_static: int,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensKNew: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mLeftPadK: Optional[cute.Tensor],
        nheads: int,
        nheads_kv: int,
        num_sm: int,
        num_splits_static: int,
        tile_m: int,
        tile_n: int,
        tile_count_semaphore: Optional[cute.Tensor],
        mPrepareSeqlenQ: Optional[cute.Tensor],
        mNumSplitsDynamic: Optional[cute.Tensor],
        mVarlenBatchIdx: Optional[cute.Tensor],
        mNumNheadsInL2: Optional[cute.Tensor],
        enable_pdl: bool,
        is_causal: bool,
        d: int,
        dv: int,
        stream: cuda.CUstream,
    ):
        """
        Execute the prepare scheduler kernel.
        """
        # Store as Python ints for grid calculation
        self.nheads = nheads
        self.nheads_kv = nheads_kv
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.d = d
        self.dv = dv
        self.enable_pdl = enable_pdl
        self.is_causal = is_causal
        self.num_splits_static = num_splits_static
        self.qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv

        # L2 cache calculations
        qhead_per_khead = self.qhead_per_khead
        self.size_l2_divisor = (
            1
            if qhead_per_khead == 1
            else (
                2
                if qhead_per_khead <= 2
                else (4 if qhead_per_khead <= 4 else (8 if qhead_per_khead <= 8 else 16))
            )
        )
        self.size_l2 = (32 * 1024 * 1024) // self.size_l2_divisor
        element_size = 2
        self.size_one_kvblock = self.tile_n * (self.d + self.dv) * element_size
        self.max_kvblocks_in_l2 = (
            self.size_l2 + self.size_one_kvblock - 1
        ) // self.size_one_kvblock

        self.num_head_computed = self.nheads if not self.packgqa else self.nheads_kv

        # Create FastDivmod objects
        tile_m_divmod = FastDivmod.create(Int32(tile_m))
        tile_n_divmod = FastDivmod.create(Int32(tile_n))

        qhead_per_khead_int32 = Int32(self.qhead_per_khead)

        grid = (self.num_ctas, 1, 1)
        block = (32 * self.num_warps, 1, 1)

        # shared memory for total block updates
        self.k_smem_size = 1

        @cute.struct
        class SharedStorage:
            total_blocks_smem: cute.struct.MemRange[Int32, self.k_smem_size]

        self.shared_storage = SharedStorage

        self.kernel(
            seqlen_q_static,
            seqlen_k_static,
            seqlen_k_new_static,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensKNew,
            mSeqUsedQ,
            mSeqUsedK,
            mLeftPadK,
            self.num_head_computed,
            qhead_per_khead_int32,
            num_sm,
            num_splits_static,
            tile_m_divmod,
            tile_n_divmod,
            tile_count_semaphore,
            mPrepareSeqlenQ,
            mNumSplitsDynamic,
            mVarlenBatchIdx,
            mNumNheadsInL2,
            enable_pdl,
            is_causal,
            self.max_kvblocks_in_l2,
        ).launch(
            grid=grid,
            block=block,
            stream=stream,
            smem=self.shared_storage.size_in_bytes(),
        )

    @cute.kernel
    def kernel(
        self,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        seqlen_k_new_static: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensKNew: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mLeftPadK: Optional[cute.Tensor],
        num_head: Int32,
        qhead_per_khead: Int32,
        num_sm: Int32,
        num_splits_static: Int32,
        tile_m_divmod: FastDivmod,
        tile_n_divmod: FastDivmod,
        tile_count_semaphore: Optional[cute.Tensor],
        mPrepareSeqlenQ: Optional[cute.Tensor],
        mNumSplitsDynamic: Optional[cute.Tensor],
        mVarlenBatchIdx: Optional[cute.Tensor],
        mNumNheadsInL2: Optional[cute.Tensor],
        enable_pdl: Boolean,
        is_causal: Boolean,
        max_kvblocks_in_l2: Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        grid_dimx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        total_blocks_smem = storage.total_blocks_smem.get_tensor((1,))

        if tidx == 0:
            total_blocks_smem[0] = Int32(0)
        cute.arch.sync_threads()

        if tidx == 0 and const_expr(tile_count_semaphore is not None):
            tile_count_semaphore[0] = Int32(0)

        batch_cta_idx_offset = bidx * 992
        bidb_start = batch_cta_idx_offset + self.k_num_batch_per_warp * warp_idx
        batch_idx = lane_idx + bidb_start

        num_m_blocks, seqlen_q = self.get_num_m_blocks_and_seqlen(
            lane_idx,
            batch_idx,
            mSeqUsedQ,
            mCuSeqlensQ,
            seqlen_q_static,
            tile_m_divmod,
            qhead_per_khead,
        )

        num_n_blocks = self.get_num_n_blocks(
            lane_idx,
            batch_idx,
            mSeqUsedK,
            mCuSeqlensK,
            mCuSeqlensKNew,
            seqlen_k_static,
            seqlen_k_new_static,
            mLeftPadK,
            tile_n_divmod,
        )

        num_splits_dynamic = Int32(0)
        if num_splits_static == 1:
            # Static scheduling - no splits needed, regardless of grid dimensions
            num_splits_dynamic = Int32(1)
        else:
            # Compute dynamic splits based on work distribution
            # This should run whenever we have flexibility (num_splits_static > 1)
            total_blocks = num_m_blocks * num_n_blocks
            # Warp reduction
            for i in range(self.num_threads_per_warp // 2, 0, -1):
                total_blocks += cute.arch.shuffle_sync_down(total_blocks, offset=i)
            if lane_idx == 0:
                total_blocks_smem.store(total_blocks_smem.load() + total_blocks)
            cute.arch.sync_threads()

            total_blocks = total_blocks_smem[0]
            blocks_per_sm = cute.ceil_div(total_blocks * 110 // 100 * num_head, num_sm)
            num_splits_dynamic = cutlass.max(
                cutlass.min(cute.ceil_div(num_n_blocks, blocks_per_sm), num_splits_static), Int32(1)
            )
            num_n_blocks = cute.ceil_div(num_n_blocks, num_splits_dynamic)

        if const_expr(self.sort):
            # TODO: Implement sort logic
            pass
        else:
            if batch_idx < self.num_batch and lane_idx < Int32(self.k_num_batch_per_warp):
                if const_expr(mPrepareSeqlenQ is not None):
                    if const_expr(self.packgqa):
                        mPrepareSeqlenQ[batch_idx] = seqlen_q * qhead_per_khead
                    else:
                        mPrepareSeqlenQ[batch_idx] = seqlen_q
                if const_expr(mNumSplitsDynamic is not None):
                    mNumSplitsDynamic[batch_idx] = num_splits_dynamic
                if const_expr(mNumNheadsInL2 is not None):
                    mNumNheadsInL2[batch_idx] = self.get_num_nheads_in_l2(
                        cutlass.max(num_n_blocks, Int32(1)),
                        num_head,
                        max_kvblocks_in_l2,
                        qhead_per_khead,
                    )

    @cute.jit
    def get_num_m_blocks_and_seqlen(
        self,
        lane_idx: Int32,
        batch_idx: Int32,
        mSeqUsedQ: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        seqlen_q_static: Int32,
        tile_m_divmod: FastDivmod,
        qhead_per_khead: Int32,
    ):
        seqlen = Int32(0)
        if const_expr(mSeqUsedQ is not None):
            seqlen = mSeqUsedQ[batch_idx] if batch_idx < self.num_batch else Int32(0)
        elif const_expr(mCuSeqlensQ is not None):
            cur_cu_seqlen = Int32(0)
            if batch_idx <= self.num_batch:
                cur_cu_seqlen = mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_q_static

        seqlen_for_blocks = seqlen
        if const_expr(self.packgqa):
            seqlen_for_blocks = seqlen * qhead_per_khead
        num_m_blocks = (
            tile_m_divmod.div(seqlen_for_blocks + tile_m_divmod.divisor - 1)
            if batch_idx < self.num_batch and lane_idx < Int32(self.k_num_batch_per_warp)
            else Int32(0)
        )
        return (num_m_blocks, seqlen)

    @cute.jit
    def get_num_n_blocks(
        self,
        lane_idx: Int32,
        batch_idx: Int32,
        mSeqUsedK: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensKNew: Optional[cute.Tensor],
        seqlen_k_static: Int32,
        seqlen_k_new_static: Int32,
        mLeftPadK: Optional[cute.Tensor],
        tile_n_divmod: FastDivmod,
    ):
        leftpad_k = (
            mLeftPadK[batch_idx]
            if const_expr(mLeftPadK is not None) and batch_idx < self.num_batch
            else Int32(0)
        )
        seqlen = Int32(0)
        if const_expr(mSeqUsedK is not None):
            seqlen = mSeqUsedK[batch_idx] if batch_idx < self.num_batch else Int32(0)
        elif const_expr(mCuSeqlensK is not None):
            cur_cu_seqlen = Int32(0)
            if batch_idx <= self.num_batch:
                cur_cu_seqlen = mCuSeqlensK[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_k_static

        seqlen_new = Int32(0)
        if const_expr(mCuSeqlensKNew is not None):
            cur_cu_seqlen_new = Int32(0)
            if batch_idx <= self.num_batch:
                cur_cu_seqlen_new = mCuSeqlensKNew[batch_idx]
            next_cu_seqlen_new = cute.arch.shuffle_sync_down(cur_cu_seqlen_new, offset=1)
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new
        else:
            seqlen_new = seqlen_k_new_static
        seqlen = seqlen - leftpad_k + seqlen_new
        ret = Int32(0)
        if batch_idx < self.num_batch and lane_idx < Int32(self.k_num_batch_per_warp):
            ret = tile_n_divmod.div(seqlen + tile_n_divmod.divisor - 1)
        return ret

    @cute.jit
    def get_num_nheads_in_l2(
        self,
        num_n_blocks: Int32,
        num_head: Int32,
        max_kvblocks_in_l2: Int32,
        qhead_per_khead: Int32,
    ):
        nheads_in_l2 = (
            Int32(16)
            if num_n_blocks * Int32(16) <= max_kvblocks_in_l2
            else (
                Int32(8)
                if num_n_blocks * Int32(8) <= max_kvblocks_in_l2
                else (
                    Int32(4)
                    if num_n_blocks * Int32(4) <= max_kvblocks_in_l2
                    else (Int32(2) if num_n_blocks * Int32(2) <= max_kvblocks_in_l2 else Int32(1))
                )
            )
        )
        if const_expr(not self.packgqa):
            nheads_in_l2 *= qhead_per_khead
        return cutlass.min(nheads_in_l2, num_head)


def prepare_varlen_num_blocks(
    num_batch: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    nheads_k: int,
    headdim: int,
    headdim_v: int,
    num_splits: int,
    tile_m: int,
    tile_n: int,
    num_sm: int,
    packgqa: bool,
    is_causal: bool,
    enable_pdl: bool,
    sort: bool,
    seqlen_k_new: int,
    stream: cuda.CUstream,
    mCuSeqlensQ: Optional[cute.Tensor] = None,
    mCuSeqlensK: Optional[cute.Tensor] = None,
    mCuSeqlensKNew: Optional[cute.Tensor] = None,
    mSeqUsedQ: Optional[cute.Tensor] = None,
    mSeqUsedK: Optional[cute.Tensor] = None,
    mLeftPadK: Optional[cute.Tensor] = None,
    mPrepareSeqlenQ: Optional[cute.Tensor] = None,
    mNumSplitsDynamic: Optional[cute.Tensor] = None,
    mVarlenBatchIdx: Optional[cute.Tensor] = None,
    mNumNheadsInL2: Optional[cute.Tensor] = None,
    tile_count_semaphore: Optional[cute.Tensor] = None,
):
    """
    Prepare scheduler metadata for varlen sequences with dynamic num splits.

    This function computes metadata needed for variable-length sequence scheduling,
    including dynamic number of splits per batch, prepared sequence lengths, and
    number of heads that fit in L2 cache.

    Args:
        num_batch: Number of batches
        seqlen_q: Static sequence length for Q (used if mCuSeqlensQ is None)
        seqlen_k: Static sequence length for K (used if mCuSeqlensK is None)
        nheads: Number of query heads
        nheads_k: Number of key/value heads
        headdim: Head dimension
        headdim_v: Value head dimension
        num_splits: Static maximum number of splits
        tile_m: M tile size
        tile_n: N tile size
        num_sm: Number of SMs on the device
        packgqa: Whether to use packed GQA
        is_causal: Whether attention is causal
        enable_pdl: Whether to enable PDL (persistent data layout)
        sort: Whether to sort batches
        seqlen_k_new: Static new sequence length for K
        mCuSeqlensQ: Cumulative sequence lengths for Q (CuTe tensor, shape: [batch_size + 1])
        mCuSeqlensK: Cumulative sequence lengths for K (CuTe tensor, shape: [batch_size + 1])
        mCuSeqlensKNew: Cumulative sequence lengths for new K (CuTe tensor, shape: [batch_size + 1])
        mSeqUsedQ: Used sequence lengths for Q (CuTe tensor, shape: [batch_size])
        mSeqUsedK: Used sequence lengths for K (CuTe tensor, shape: [batch_size])
        mLeftPadK: Left padding for K (CuTe tensor, shape: [batch_size])
        mPrepareSeqlenQ: Output tensor for prepared Q sequence lengths (CuTe tensor, shape: [batch_size])
        mNumSplitsDynamic: Output tensor for dynamic number of splits (CuTe tensor, shape: [batch_size])
        mVarlenBatchIdx: Output tensor for varlen batch indices (CuTe tensor, shape: [batch_size])
        mNumNheadsInL2: Output tensor for number of heads in L2 (CuTe tensor, shape: [batch_size])
        tile_count_semaphore: Semaphore for tile counting (CuTe tensor, shape: [1])
        stream: CUDA stream to use
    """

    # Create cache key for compilation
    cache_key = (
        packgqa,
        sort,
        num_batch,
        enable_pdl,
        is_causal,
    )

    # Compile if not cached
    if cache_key not in prepare_varlen_num_blocks.compile_cache:
        # Create scheduler instance
        prepare_scheduler = FlashPrepareScheduler(packgqa=packgqa, sort=sort, num_batch=num_batch)

        prepare_varlen_num_blocks.compile_cache[cache_key] = cute.compile(
            prepare_scheduler,
            seqlen_q,
            seqlen_k,
            seqlen_k_new,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensKNew,
            mSeqUsedQ,
            mSeqUsedK,
            mLeftPadK,
            nheads,
            nheads_k,
            num_sm,
            num_splits,
            tile_m,
            tile_n,
            tile_count_semaphore,
            mPrepareSeqlenQ,
            mNumSplitsDynamic,
            mVarlenBatchIdx,
            mNumNheadsInL2,
            enable_pdl,
            is_causal,
            headdim,
            headdim_v,
            stream,
        )

    # Launch the compiled kernel
    prepare_varlen_num_blocks.compile_cache[cache_key](
        seqlen_q,
        seqlen_k,
        seqlen_k_new,
        mCuSeqlensQ,
        mCuSeqlensK,
        mCuSeqlensKNew,
        mSeqUsedQ,
        mSeqUsedK,
        mLeftPadK,
        nheads,
        nheads_k,
        num_sm,
        num_splits,
        tile_m,
        tile_n,
        tile_count_semaphore,
        mPrepareSeqlenQ,
        mNumSplitsDynamic,
        mVarlenBatchIdx,
        mNumNheadsInL2,
        enable_pdl,
        is_causal,
        headdim,
        headdim_v,
        stream,
    )


# Initialize compile cache
prepare_varlen_num_blocks.compile_cache = {}
