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
        sort: bool = False,
    ):
        self.sort = sort
        self.num_threads_per_warp = 32
        self.k_num_batch_per_warp = 31

    def get_grid_and_block_shape(
        self,
        num_batch: int,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        num_warps = (num_batch + 30) // 31
        num_ctas = (num_batch + (31 * 32 - 1)) // (31 * 32)
        if num_ctas > 1:
            num_warps = 32
        return ((num_ctas, 1, 1), (32 * num_warps, 1, 1))

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
        num_batch: Constexpr[int],
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
        packgqa: Constexpr[bool],
        is_e4m3: Constexpr[bool],
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
        self.num_batch = num_batch  # Python int
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.d = d
        self.dv = dv
        self.is_e4m3 = is_e4m3
        self.enable_pdl = enable_pdl
        self.is_causal = is_causal
        self.packgqa = packgqa
        self.num_splits_static = num_splits_static
        self.qhead_per_khead = (nheads + nheads_kv - 1) // nheads_kv
        self.num_warps = (num_batch + 30) // 31
        self.num_ctas = (num_batch + (31 * 32 - 1)) // (31 * 32)

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
        element_size = 1 if self.is_e4m3 else 2
        self.size_one_kvblock = self.tile_n * (self.d + self.dv) * element_size
        self.max_kvblocks_in_l2 = (
            self.size_l2 + self.size_one_kvblock - 1
        ) // self.size_one_kvblock

        self.num_head_computed = self.nheads if not self.packgqa else self.nheads_kv

        # Create FastDivmod objects
        tile_m_divmod = FastDivmod.create(Int32(tile_m))
        tile_n_divmod = FastDivmod.create(Int32(tile_n))

        qhead_per_khead_int32 = Int32(self.qhead_per_khead)

        grid, block = self.get_grid_and_block_shape(num_batch)

        # shared memory for total block updates
        self.k_smem_size = 1

        @cute.struct
        class SharedStorage:
            total_blocks_smem: cute.struct.MemRange[Int32, self.k_smem_size]

        self.shared_storage = SharedStorage

        self.kernel(
            seqlen_q_static,
            Int32(seqlen_k_static),
            Int32(seqlen_k_new_static),
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensKNew,
            mSeqUsedQ,
            mSeqUsedK,
            mLeftPadK,
            Int32(num_batch),
            Int32(self.num_head_computed),
            qhead_per_khead_int32,
            Int32(num_sm),
            Int32(num_splits_static),
            tile_m_divmod,
            tile_n_divmod,
            tile_count_semaphore,
            mPrepareSeqlenQ,
            mNumSplitsDynamic,
            mVarlenBatchIdx,
            mNumNheadsInL2,
            enable_pdl,
            is_causal,
            packgqa,
            Int32(self.max_kvblocks_in_l2),
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
        num_batch: Int32,  # Int32 in kernel
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
        packgqa: Constexpr[bool],
        max_kvblocks_in_l2: Int32,
    ):
        k_num_batch_per_warp = self.k_num_batch_per_warp
        bdimx, _, _ = cute.arch.block_dim()
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
        bidb_start = batch_cta_idx_offset + k_num_batch_per_warp * warp_idx
        batch_idx = lane_idx + bidb_start

        num_m_blocks, seqlen_q = self.get_num_m_blocks_and_seqlen(
            lane_idx,
            batch_idx,
            Int32(k_num_batch_per_warp),  # Convert Python int to Int32
            mSeqUsedQ,
            mCuSeqlensQ,
            seqlen_q_static,
            tile_m_divmod,
            num_batch,
            qhead_per_khead,
        )

        num_n_blocks = self.get_num_n_blocks(
            lane_idx,
            batch_idx,
            Int32(k_num_batch_per_warp),
            mSeqUsedK,
            mCuSeqlensK,
            mCuSeqlensKNew,
            seqlen_k_static,
            seqlen_k_new_static,
            mLeftPadK,
            tile_n_divmod,
            num_batch,
        )
        num_splits_dynamic = Int32(0)
        if grid_dimx > 1 or num_splits_static == 1:
            num_splits_dynamic = Int32(1)
        else:
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
            if batch_idx < num_batch and lane_idx < Int32(k_num_batch_per_warp):
                if const_expr(mPrepareSeqlenQ is not None):
                    if const_expr(packgqa):
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
        k_num_batch_per_warp: Int32,
        mSeqUsedQ: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        seqlen_q_static: Int32,
        tile_m_divmod: FastDivmod,
        num_batch: Int32,
        qhead_per_khead: Int32,
    ):
        seqlen = Int32(0)
        if const_expr(mSeqUsedQ is not None):
            seqlen = mSeqUsedQ[batch_idx] if batch_idx < num_batch else Int32(0)
        elif const_expr(mCuSeqlensQ is not None):
            cur_cu_seqlen = mCuSeqlensQ[batch_idx] if batch_idx <= num_batch else Int32(0)
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_q_static

        seqlen_for_blocks = seqlen
        if const_expr(self.packgqa):
            seqlen_for_blocks = seqlen * qhead_per_khead
        num_m_blocks = (
            tile_m_divmod.div(seqlen_for_blocks + tile_m_divmod.divisor - 1)
            if batch_idx < num_batch and lane_idx < k_num_batch_per_warp
            else Int32(0)
        )
        return (num_m_blocks, seqlen)

    @cute.jit
    def get_num_n_blocks(
        self,
        lane_idx: Int32,
        batch_idx: Int32,
        k_num_batch_per_warp: Int32,
        mSeqUsedK: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensKNew: Optional[cute.Tensor],
        seqlen_k_static: Int32,
        seqlen_k_new_static: Int32,
        mLeftPadK: Optional[cute.Tensor],
        tile_n_divmod: FastDivmod,
        num_batch: Int32,
    ):
        leftpad_k = (
            mLeftPadK[batch_idx]
            if const_expr(mLeftPadK is not None) and batch_idx < num_batch
            else Int32(0)
        )
        seqlen = Int32(0)
        if const_expr(mSeqUsedK is not None):
            seqlen = mSeqUsedK[batch_idx] if batch_idx < num_batch else Int32(0)
        elif const_expr(mCuSeqlensK is not None):
            cur_cu_seqlen = mCuSeqlensK[batch_idx] if batch_idx <= num_batch else Int32(0)
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        else:
            seqlen = seqlen_k_static

        seqlen_new = Int32(0)
        if const_expr(mCuSeqlensKNew is not None):
            cur_cu_seqlen_new = mCuSeqlensKNew[batch_idx] if batch_idx <= num_batch else Int32(0)
            next_cu_seqlen_new = cute.arch.shuffle_sync_down(cur_cu_seqlen_new, offset=1)
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new
        else:
            seqlen_new = seqlen_k_new_static
        seqlen = seqlen - leftpad_k + seqlen_new
        return (
            tile_n_divmod.div(seqlen + tile_n_divmod.divisor - 1)
            if batch_idx < num_batch and lane_idx < k_num_batch_per_warp
            else Int32(0)
        )

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
