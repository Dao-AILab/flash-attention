from typing import Type, Optional
from dataclasses import dataclass
import operator

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass import Int32, Uint32, const_expr, Boolean

from flash_attn.cute import utils
from flash_attn.cute.utils import warp_reduce
from quack.cute_dsl_utils import ParamsBase

import math


@dataclass
class CpasyncGatherKVManager(ParamsBase):
    mIndexTopk: cute.Tensor
    sBitmask: Optional[cute.Tensor]

    cta_rank_in_cluster: Int32
    thread_idx: Int32
    warp_idx: Int32

    topk_length: Int32
    seqlen_k_limit: Int32
    tile_n: Int32
    num_threads: cutlass.Constexpr[Int32]
    hdim: cutlass.Constexpr[Int32]
    hdim_v: cutlass.Constexpr[Int32]
    num_hdimv_splits: cutlass.Constexpr[Int32]
    cta_group_size: cutlass.Constexpr[Int32]

    gmem_threads_per_row: cutlass.Constexpr[Int32]
    topk_indices_per_thread: Int32
    async_copy_elems: Int32

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy

    rTopk: cute.Tensor
    rTopkHalf: cute.Tensor
    # for bitmask
    rTopk_NonInterleaved: cute.Tensor

    pipeline_bitmask: Optional[pipeline.PipelineAsync]
    cpasync_barrier: Optional[pipeline.NamedBarrier]

    disable_bitmask: cutlass.Constexpr[Boolean]

    @staticmethod
    def create(
        mIndexTopk: cute.Tensor,
        cta_rank_in_cluster: Int32,
        thread_idx: Int32,
        warp_idx: Int32,
        topk_length: Int32,
        seqlen_k_limit: Int32,
        tile_n: cutlass.Constexpr[Int32],
        hdim: cutlass.Constexpr[Int32],
        hdim_v: cutlass.Constexpr[Int32],
        num_hdimv_splits: cutlass.Constexpr[Int32],
        num_threads: cutlass.Constexpr[Int32],
        dtype: Type[cutlass.Numeric],
        cta_group_size: cutlass.Constexpr[Int32],
        cpasync_barrier: Optional[pipeline.NamedBarrier] = None,
        disable_bitmask: cutlass.Constexpr[Boolean] = False,
        sBitmask: Optional[cute.Tensor] = None,
        pipeline_bitmask: Optional[pipeline.PipelineAsync] = None,
    ):
        assert tile_n % num_threads == 0
        assert num_threads == 128
        assert hdim % 64 == 0
        assert (hdim_v // num_hdimv_splits // cta_group_size) % 64 == 0
        assert num_threads % cute.arch.WARP_SIZE == 0
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        dtype_bytes = dtype.width // 8
        # assumes hdim is never part of transposed operand
        gmem_k_block_size = math.gcd(
            hdim,
            hdim_v // num_hdimv_splits // cta_group_size,
            128 // dtype_bytes,
        )
        assert gmem_k_block_size % async_copy_elems == 0
        gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        assert cute.arch.WARP_SIZE % gmem_threads_per_row == 0
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)
        topk_indices_per_thread = tile_n // num_threads

        rTopk = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)
        rTopkHalf = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)
        rTopk_NonInterleaved = cute.make_rmem_tensor((topk_indices_per_thread,), Int32)

        return CpasyncGatherKVManager(
            mIndexTopk,
            sBitmask,
            cta_rank_in_cluster,
            thread_idx,
            warp_idx,
            topk_length,
            seqlen_k_limit,
            tile_n,
            num_threads,
            hdim,
            hdim_v,
            num_hdimv_splits,
            cta_group_size,
            gmem_threads_per_row,
            topk_indices_per_thread,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            rTopk,
            rTopkHalf,
            rTopk_NonInterleaved,
            pipeline_bitmask,
            cpasync_barrier,
            disable_bitmask,
        )

    @cute.jit
    def load_index_topk(
        self,
        n_block: Int32,
        transpose: bool,
    ):
        entries_per_thread = self.topk_indices_per_thread
        rTopk = self.rTopk if const_expr(transpose) else self.rTopkHalf

        for i in cutlass.range_constexpr(entries_per_thread):
            row = (
                i * self.num_threads
                + (self.thread_idx % self.gmem_threads_per_row)
                * (self.num_threads // self.gmem_threads_per_row)
                + (self.thread_idx // self.gmem_threads_per_row)
            )
            # need this if not offset in load_X
            # if const_expr(not transpose):
            #     row += self.cta_rank_in_cluster * (self.tile_n//self.cta_group_size)
            #     row = row % self.tile_n
            row_idx = n_block * self.tile_n + row
            rTopk[i] = self.mIndexTopk[row_idx]

            if const_expr(not transpose and not self.disable_bitmask):
                row_non_interleaved = i * self.num_threads + self.thread_idx
                row_idx_non_interleaved = n_block * self.tile_n + row_non_interleaved
                self.rTopk_NonInterleaved[0] = self.mIndexTopk[row_idx_non_interleaved]

    @cute.jit
    def compute_bitmask(
        self,
        producer_state_bitmask,
    ):
        assert self.pipeline_bitmask is not None, "pipeline_bitmask not provided"
        assert self.cpasync_barrier is not None, "cpasync barrier not provided"

        lane_idx = cute.arch.lane_idx()
        assert cute.size(self.rTopk_NonInterleaved) == 1
        bitmask = Uint32(0)

        # Step 1. Construct per-thread bitmask
        topk_idx = self.rTopk_NonInterleaved[0]
        is_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
        if is_valid:
            bitmask = Uint32(1 << lane_idx)

        # Step 2. Warp shuffle bitwise OR = add since indices are exclusive.
        bitmask = warp_reduce(bitmask, operator.add)

        self.pipeline_bitmask.producer_acquire(producer_state_bitmask)
        # store to smem and sync threads
        if lane_idx == 0:
            self.sBitmask[self.warp_idx, producer_state_bitmask.index] = bitmask
        self.cpasync_barrier.arrive_and_wait()

        self.pipeline_bitmask.producer_commit(producer_state_bitmask)
        producer_state_bitmask.advance()
        return producer_state_bitmask

    @cute.jit
    def compute_X_ptr(
        self,
        mX: cute.Tensor,
        transpose: bool,
        d_offset: int = 0,
    ):
        entries_per_thread = self.topk_indices_per_thread
        tPrXPtr = cute.make_rmem_tensor((entries_per_thread,), cutlass.Int64)
        tPrRowValid = cute.make_rmem_tensor((entries_per_thread,), cutlass.Int32)
        rTopk = self.rTopk if const_expr(transpose) else self.rTopkHalf

        for i in cutlass.range_constexpr(entries_per_thread):
            topk_idx = rTopk[i]
            if const_expr(not self.disable_bitmask):
                row_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
                tPrRowValid[i] = row_valid
            if const_expr(not transpose):
                tPrXPtr[i] = utils.elem_pointer(mX, (topk_idx, d_offset)).toint()
            else:
                tPrXPtr[i] = utils.elem_pointer(mX, (d_offset, topk_idx)).toint()

        return tPrXPtr, tPrRowValid

    @cute.jit
    def load_X(
        self,
        mX: cute.Tensor,
        sX: cute.Tensor,
        transpose: bool,
        K_or_V: str,
        d_offset: int = 0,
    ):
        assert K_or_V in ("K", "V")
        cta_tile_n = self.tile_n if const_expr(transpose) else self.tile_n // self.cta_group_size
        head_dim = self.hdim if const_expr(K_or_V == "K") else self.hdim_v // self.num_hdimv_splits
        if const_expr(transpose):
            head_dim = head_dim // self.cta_group_size
        order = (1, 0) if const_expr(transpose) else (0, 1)

        sX_nd_layout = cute.make_ordered_layout((cta_tile_n, head_dim), order=order)
        sX_nd = cute.composition(sX, sX_nd_layout)

        cX = cute.make_identity_tensor((cta_tile_n, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_nd)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)

        tPrXPtr, tPrRowValid = self.compute_X_ptr(mX, transpose, d_offset)

        if const_expr(not transpose):
            offset = self.cta_rank_in_cluster * (self.gmem_threads_per_row // self.cta_group_size)
        else:
            offset = 0

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            if const_expr(not self.disable_bitmask):
                row_valid = utils.shuffle_sync(
                    tPrRowValid[m // self.gmem_threads_per_row],
                    (m + offset) % self.gmem_threads_per_row,
                    width=self.gmem_threads_per_row,
                )
                should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], Boolean)
                should_load.fill(Boolean(row_valid))
            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // self.gmem_threads_per_row],
                (m + offset) % self.gmem_threads_per_row,
                width=self.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                mX.element_type, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mX_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_cur_copy = cute.tiled_divide(mX_cur, (self.async_copy_elems,))

            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, k][1] // self.async_copy_elems
                mX_cur_copy_ki = mX_cur_copy[None, ki]
                tXsX_k = tXsX[None, m, k]
                mX_cur_copy_ki = cute.make_tensor(mX_cur_copy_ki.iterator, tXsX_k.layout)
                cute.copy(
                    self.gmem_tiled_copy_KV,
                    mX_cur_copy_ki,
                    tXsX_k,
                    pred=should_load if const_expr(not self.disable_bitmask) else None,
                )


@dataclass
class CpasyncGatherKVManagerH64(ParamsBase):
    """Gather producer of the native 64-head sparse-MLA forward (see AI/SPARSE_MLA_64H.md).

    One stage = 64 top-k keys x whole rows: the ``hdim_v`` latent row and the ``hdim`` rope row of
    the same key (two tables, one index) as 16-B ``cp.async.cg`` copies from 128 threads, 8 threads
    per 128-B chunk, a warp covering 4 rows x 128 B per instruction: 4 rows x (8 latent + 1 rope) =
    36 copies per thread per stage. Index ownership follows ``load_X``'s shuffle source: lane ``m``
    of each 8-thread group (``m < 4``) holds the index of row ``16 * m + t // 8``; lanes 4-7 load the
    same rows again (a duplicate 4-B read instead of a predicate). Indices are loaded two blocks
    ahead into two register sets (``buf`` 0 / 1), for both the interleaved order (copies) and the
    natural order (bitmask), so a block's issue never waits on its own index load. Rows whose index
    is -1 or >= ``seqlen_k_limit`` are zero-filled (predicated ``cp.async``) and cleared in the
    2-word validity bitmask (warps 0-1, bit = lane = key within the 32-key half).
    """

    mIndexTopk: cute.Tensor
    sBitmask: Optional[cute.Tensor]

    thread_idx: Int32
    warp_idx: Int32

    seqlen_k_limit: Int32
    tile_n: cutlass.Constexpr[int]
    num_threads: cutlass.Constexpr[int]
    hdim: cutlass.Constexpr[int]
    hdim_v: cutlass.Constexpr[int]
    gmem_threads_per_row: cutlass.Constexpr[int]
    async_copy_elems: cutlass.Constexpr[int]

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy

    # two register sets each (index of block n+2 loaded while block n is in flight)
    rTopk: cute.Tensor  # interleaved ownership: the row this thread's 8-group copies
    rTopk_NonInterleaved: cute.Tensor  # natural ownership: row = thread_idx % tile_n (bitmask)

    pipeline_bitmask: Optional[pipeline.PipelineAsync]
    cpasync_barrier: Optional[pipeline.NamedBarrier]

    disable_bitmask: cutlass.Constexpr[Boolean]

    @staticmethod
    def create(
        mIndexTopk: cute.Tensor,
        thread_idx: Int32,
        warp_idx: Int32,
        seqlen_k_limit: Int32,
        tile_n: cutlass.Constexpr[int],
        hdim: cutlass.Constexpr[int],
        hdim_v: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
        dtype: Type[cutlass.Numeric],
        cpasync_barrier: Optional[pipeline.NamedBarrier] = None,
        disable_bitmask: cutlass.Constexpr[Boolean] = False,
        sBitmask: Optional[cute.Tensor] = None,
        pipeline_bitmask: Optional[pipeline.PipelineAsync] = None,
    ):
        assert num_threads == 128, "H64 gather: 128 producer threads"
        assert tile_n == 64, "H64 gather: 64-key stages"
        assert hdim % 64 == 0 and hdim_v % 64 == 0, "rows are whole 128-B chunks"
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        gmem_k_block_size = 128 // (dtype.width // 8)  # one 128-B swizzle row of the SW128 tile
        gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        rows_per_copy = num_threads // gmem_threads_per_row
        assert tile_n % rows_per_copy == 0
        # load_X shuffles row 16*m + t//8's pointer from lane m of the 8-thread group
        assert tile_n // rows_per_copy <= gmem_threads_per_row
        assert tile_n % cute.arch.WARP_SIZE == 0
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (rows_per_copy, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)

        rTopk = cute.make_rmem_tensor((2,), Int32)
        rTopk_NonInterleaved = cute.make_rmem_tensor((2,), Int32)

        return CpasyncGatherKVManagerH64(
            mIndexTopk,
            sBitmask,
            thread_idx,
            warp_idx,
            seqlen_k_limit,
            tile_n,
            num_threads,
            hdim,
            hdim_v,
            gmem_threads_per_row,
            async_copy_elems,
            gmem_tiled_copy_KV,
            gmem_thr_copy_KV,
            rTopk,
            rTopk_NonInterleaved,
            pipeline_bitmask,
            cpasync_barrier,
            disable_bitmask,
        )

    @cute.jit
    def load_index_topk(self, n_block: Int32, buf: cutlass.Constexpr[int]):
        """Load this thread's two indices of block ``n_block`` into register set ``buf``."""
        rows_per_copy = self.num_threads // self.gmem_threads_per_row
        row_groups = self.tile_n // rows_per_copy
        lane_in_group = self.thread_idx % self.gmem_threads_per_row
        row = (
            lane_in_group % row_groups
        ) * rows_per_copy + self.thread_idx // self.gmem_threads_per_row
        self.rTopk[buf] = self.mIndexTopk[n_block * self.tile_n + row]
        if const_expr(not self.disable_bitmask):
            row_natural = self.thread_idx % self.tile_n
            self.rTopk_NonInterleaved[buf] = self.mIndexTopk[n_block * self.tile_n + row_natural]

    @cute.jit
    def compute_bitmask(self, producer_state_bitmask, buf: cutlass.Constexpr[int]):
        """One validity word per 32 keys, written by warps 0 .. tile_n // 32 - 1 (bit = lane)."""
        assert self.pipeline_bitmask is not None, "pipeline_bitmask not provided"
        assert self.cpasync_barrier is not None, "cpasync barrier not provided"
        lane_idx = cute.arch.lane_idx()
        topk_idx = self.rTopk_NonInterleaved[buf]
        is_valid = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
        bitmask = Uint32(0)
        if is_valid:
            bitmask = Uint32(1 << lane_idx)
        # indices within a block are exclusive -> OR == add
        bitmask = warp_reduce(bitmask, operator.add)

        self.pipeline_bitmask.producer_acquire(producer_state_bitmask)
        if lane_idx == 0 and self.warp_idx < self.tile_n // cute.arch.WARP_SIZE:
            self.sBitmask[self.warp_idx, producer_state_bitmask.index] = bitmask
        self.cpasync_barrier.arrive_and_wait()
        self.pipeline_bitmask.producer_commit(producer_state_bitmask)
        producer_state_bitmask.advance()
        return producer_state_bitmask

    @cute.jit
    def load_X(
        self,
        mX: cute.Tensor,
        sX: cute.Tensor,
        K_or_V: str,
        buf: cutlass.Constexpr[int],
        col_blocks: Optional[tuple] = None,
        identity_rows: cutlass.Constexpr[bool] = False,
    ):
        """Issue the cp.async copies of one 64-key stage of ``mX`` (``(seqlen_k, head_dim)``) into the
        K-major swizzled stage tensor ``sX`` (the MMA layout of a 64 x head_dim tile; the swizzle is
        the tensor's), whole rows, or only the 128-B column blocks ``col_blocks = (c0, c1)`` of every
        row (the caller lands a stage in parts, each followed by its own
        ``cp.async.mbarrier.arrive.noinc``). With ``identity_rows`` the stage is the 64 rows of ``mX``
        itself (row ``r`` of the tile <- ``mX[r]``, no top-k index, no validity predicate): the 64-head
        forward stages the token's Q tile through the KV ring this way (see AI/SPARSE_MLA_64H.md)."""
        assert K_or_V in ("K", "V")
        head_dim = self.hdim if const_expr(K_or_V == "K") else self.hdim_v
        sX_nd_layout = cute.make_ordered_layout((self.tile_n, head_dim), order=(0, 1))
        sX_nd = cute.composition(sX, sX_nd_layout)

        cX = cute.make_identity_tensor((self.tile_n, head_dim))
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_nd)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)

        use_pred = const_expr(not self.disable_bitmask and not identity_rows)
        tPrXPtr = cute.make_rmem_tensor((1,), cutlass.Int64)
        tPrRowValid = cute.make_rmem_tensor((1,), cutlass.Int32)
        if const_expr(not identity_rows):
            topk_idx = self.rTopk[buf]
            tPrXPtr[0] = utils.elem_pointer(mX, (topk_idx, 0)).toint()
            if const_expr(use_pred):
                tPrRowValid[0] = topk_idx >= 0 and topk_idx < self.seqlen_k_limit
        rows_per_copy = self.num_threads // self.gmem_threads_per_row

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            # row 16*m + t//8: its index sits in lane m of this thread's 8-thread group
            if const_expr(use_pred):
                row_valid = utils.shuffle_sync(tPrRowValid[0], m, width=self.gmem_threads_per_row)
                should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], Boolean)
                should_load.fill(Boolean(row_valid))
            if const_expr(identity_rows):
                x_ptr_i64 = utils.elem_pointer(
                    mX, (rows_per_copy * m + self.thread_idx // self.gmem_threads_per_row, 0)
                ).toint()
            else:
                x_ptr_i64 = utils.shuffle_sync(tPrXPtr[0], m, width=self.gmem_threads_per_row)
            x_gmem_ptr = cute.make_ptr(
                mX.element_type, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            mX_cur = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_cur_copy = cute.tiled_divide(mX_cur, (self.async_copy_elems,))

            num_col_blocks = cute.size(tXsX, mode=[2])
            ks = list(range(num_col_blocks)) if col_blocks is None else list(range(*col_blocks))
            assert len(ks) > 0 and ks[0] >= 0 and ks[-1] < num_col_blocks, (
                f"bad col_blocks {col_blocks}"
            )
            for k in ks:
                ki = tXcX[0, 0, k][1] // self.async_copy_elems
                mX_cur_copy_ki = mX_cur_copy[None, ki]
                tXsX_k = tXsX[None, m, k]
                mX_cur_copy_ki = cute.make_tensor(mX_cur_copy_ki.iterator, tXsX_k.layout)
                cute.copy(
                    self.gmem_tiled_copy_KV,
                    mX_cur_copy_ki,
                    tXsX_k,
                    pred=should_load if const_expr(use_pred) else None,
                )
