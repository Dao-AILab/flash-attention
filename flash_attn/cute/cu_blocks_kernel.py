from typing import Callable, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


class CuSeqlensToBlocksKernel:
    def __init__(
        self,
        tile: int = 128,
        num_threads: int = 1024,
        seqlen_q_multiplier: int = 1,
    ):
        self.tile = tile
        self.num_threads = num_threads
        assert num_threads % 32 == 0
        self.num_warps = num_threads // cute.arch.WARP_SIZE
        self.seqlen_q_multiplier = seqlen_q_multiplier

    @cute.jit
    def __call__(
        self,
        mCuBlocks: cute.Tensor,
        mCuSplitsBlocks: Optional[cute.Tensor],
        mCuSeqlens: Optional[cute.Tensor],
        mSeqUsed: Optional[cute.Tensor] = None,
        mNumSplits: Optional[cute.Tensor] = None,
        mVirtualBatchIdx: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        assert const_expr((mNumSplits is None) == (mCuSplitsBlocks is None))
        assert const_expr(mCuSeqlens is not None or mSeqUsed is not None)

        @cute.struct
        class SharedStorage:
            warp_block_count: cute.struct.MemRange[Int32, self.num_warps]
            warp_split_count: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mCuBlocks,
            mCuSplitsBlocks,
            mCuSeqlens,
            mSeqUsed,
            mNumSplits,
            mVirtualBatchIdx,
            SharedStorage,
        ).launch(
            grid=[1, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCuBlocks: cute.Tensor,
        mCuSplitsBlocks: Optional[cute.Tensor],
        mCuSeqlens: Optional[cute.Tensor],
        mSeqUsed: Optional[cute.Tensor],
        mNumSplits: Optional[cute.Tensor],
        mVirtualBatchIdx: Optional[cute.Tensor],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        has_splits = mNumSplits is not None
        batch_size = mCuBlocks.shape[0] - 1
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_block_count = storage.warp_block_count.get_tensor(cute.make_layout(self.num_warps))
        warp_split_count = storage.warp_split_count.get_tensor(cute.make_layout(self.num_warps))

        if tidx == 0:
            mCuBlocks[0] = 0
            if const_expr(has_splits):
                mCuSplitsBlocks[0] = 0

        # Process the batch in chunks of num_threads, carrying the running totals in the bases.
        base = Int32(0)
        base_splits = Int32(0)
        num_chunks = (batch_size + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            batch_idx = chunk * self.num_threads + tidx

            seqlen = Int32(0)
            batch_splits = Int32(0)
            if batch_idx < batch_size:
                if const_expr(mVirtualBatchIdx is not None):
                    batch_idx = Int32(mVirtualBatchIdx[batch_idx])
                if const_expr(mSeqUsed is not None):
                    seqlen = mSeqUsed[batch_idx]
                else:
                    seqlen = mCuSeqlens[batch_idx + 1] - mCuSeqlens[batch_idx]
                if const_expr(has_splits):
                    batch_splits = mNumSplits[batch_idx]
            seqlen *= self.seqlen_q_multiplier
            num_blocks = (seqlen + self.tile - 1) // self.tile
            num_split_blocks = num_blocks * batch_splits

            total_blocks_for_batch = num_blocks
            total_split_blocks_for_batch = num_split_blocks
            
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(total_blocks_for_batch, delta, mask_and_clamp=0)
                if const_expr(has_splits):
                    other_splits = cute.arch.shuffle_sync_up(
                        total_split_blocks_for_batch, delta, mask_and_clamp=0
                    )
                    if lane_idx >= delta:
                        total_split_blocks_for_batch += other_splits
                if lane_idx >= delta:
                    total_blocks_for_batch += other

            if lane_idx == 31:
                warp_block_count[warp_idx] = total_blocks_for_batch
                if const_expr(has_splits):
                    warp_split_count[warp_idx] = total_split_blocks_for_batch

            cute.arch.sync_threads()

            total_blocks_for_batch += base
            total_split_blocks_for_batch += base_splits
            
            for idx in cutlass.range(warp_idx):
                total_blocks_for_batch += warp_block_count[idx]
                if const_expr(has_splits):
                    total_split_blocks_for_batch += warp_split_count[idx]

            if batch_idx < batch_size:
                if const_expr(has_splits):
                    mCuSplitsBlocks[chunk * self.num_threads + tidx + 1] = (
                        total_split_blocks_for_batch
                    )

            for idx in cutlass.range(self.num_warps):
                base += warp_block_count[idx]
                if const_expr(has_splits):
                    base_splits += warp_split_count[idx]
            # warp_block_count / warp_split_count are reused by the next chunk.
            cute.arch.sync_threads()
