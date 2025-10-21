# Copyright (c) 2025, Tri Dao.

from typing import Optional, Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

import flash_attn.cute.utils as utils

@cute.jit
def mask_r2p(X: cute.Tensor, col_limit: Int32, arch: int = 90, rank1: bool = False) -> None:
    # Bit manipulation, compiles down to the R2P instruction
    # For sm100: we know that tScS_t2r[i][1] == i, for the particular tmem copy atom we're using.
    # For sm90: instead of comparing limit to 0, 1, 8, 9, 16, 17, ...,
    # we compare a transformed version of limit to 0, 1, 2, 3, 4, 5, ...
    if const_expr(arch == 90):
        col_limit_transformed = col_limit // 8 * 2 + min(col_limit % 8, 2)
    else:
        col_limit_transformed = col_limit
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # Ideally we'd move by 32 instead of 24, but mask >> i isn't correct for i == 31
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # Don't need to clamp to 32 since the shr.u32 instruction does that already
        col_limit_right_s = max(col_limit_transformed - s * 24, 0)
        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
        mask = (1 << col_limit_right_s) - 1
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
                # This is the equivalent of:
                # X[s * 24 + i] = X[s * 24 + i] if col_limit_right_s <= i else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf

@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_q: Int32
    seqlen_k: Int32
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA
    swap_AB: cutlass.Constexpr[bool] = False

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        buffers: Optional[list[cute.Tensor]] = None,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS), transpose=self.swap_AB)
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = utils.make_acc_tensor_mn_view(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local and mask_mod is None):
            if const_expr(mask_seqlen):
                # The compiler now choses not to use R2P
                r2p = const_expr(False and not self.swap_AB)
                if const_expr(not r2p):
                    # traverse column index.
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                else:
                    mask_r2p(acc_S_mn, seqlenk_col_limit, arch=90)
                                
        elif const_expr(not mask_causal and not mask_local and mask_mod is not None): # FlexAttention mask mod
            nrow = const_expr(cute.size(tScS_mn.shape[0]))
            ncol = const_expr(cute.size(tScS_mn.shape[1]))
            thr_col_offset = tScS_mn[0, 0][1]
            
            for r in cutlass.range_constexpr(nrow):
                global_row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                
                for col in cutlass.range_constexpr(ncol):
                    col_idx_local = t0ScS_mn[0, col][1]
                    # Convert to absolute column index
                    global_col_idx = thr_col_offset + col_idx_local + n_block * self.tile_n
                    
                    cond = cutlass.Boolean(
                        mask_mod(
                            batch_idx,
                            head_idx,
                            tScS_mn[r, 0][0] + m_block * self.tile_m,
                            thr_col_offset + t0ScS_mn[0, col][1] + n_block * self.tile_n,
                            self.seqlen_q,
                            self.seqlen_k,
                            buffers,
                        )
                    )
                    if const_expr(mask_seqlen):
                        out_of_bounds = (global_row_idx >= self.seqlen_q) or (
                            global_col_idx >= self.seqlen_k
                        )
                        if out_of_bounds:
                            acc_S_mn[r, col] = -cutlass.Float32.inf
                        else:
                            acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf
                    else:
                        acc_S_mn[r, col] = acc_S_mn[r, col] if cond else -cutlass.Float32.inf


        else:  # Causal or local
            if const_expr(not self.swap_AB):
                # If PackGQA, we split the work of compute divmod among threads in the same row
                threads_per_row = thr_mma.tv_layout_C.shape[0][0]
                mma_m_idx = None
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
                    assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                        "threads_per_row must divide WARP_SIZE"
                    )
                    assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                    tidx = thr_mma.thr_idx
                    mma_m_idx = (
                        m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                    ) // self.qhead_per_kvhead_packgqa
                causal_row_offset = (
                    1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
                )
                if const_expr(mask_causal):
                    r2p = const_expr(not self.swap_AB)  # R2P trick, see apply_mask_sm100
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        if const_expr(not r2p):
                            # traverse column index.
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if t0ScS_mn[0, c][1] >= col_limit_right
                                    else acc_S_mn[r, c]
                                )
                        else:
                            mask_r2p(acc_S_mn[r, None], col_limit_right, arch=90, rank1=True)
                else:  # Local
                    local_row_offset_right = (
                        causal_row_offset + self.window_size_right
                        if const_expr(self.window_size_right is not None)
                        else None
                    )
                    local_row_offset_left = (
                        causal_row_offset - 1 - self.window_size_left
                        if const_expr(self.window_size_left is not None)
                        else None
                    )
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        if const_expr(self.window_size_right is not None):
                            col_limit_right = row_idx + local_row_offset_right
                            if const_expr(mask_seqlen):
                                col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        else:
                            col_limit_right = self.tile_n
                        col_limit_left = (
                            row_idx + local_row_offset_left
                            if const_expr(self.window_size_left is not None)
                            else 0
                        )
                        # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block = {}, r = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", n_block, r, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                        # traverse column index.
                        for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                            col_idx = t0ScS_mn[0, c][1]
                            # only consider the column index, so the row index sets to 0.
                            if col_idx >= col_limit_right or col_idx < col_limit_left:
                                acc_S_mn[r, c] = -Float32.inf
            else:  # swap_AB
                assert self.qhead_per_kvhead_packgqa == 1
                thr_row_offset = tScS_mn[0][ROW]
                causal_row_offset = (
                    seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
                )
                if const_expr(mask_causal):
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m if col0 >= seqlenk_col_limit else col0 - causal_row_offset
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if t0ScS_mn[r, 0][ROW] < row_limit_top
                                else acc_S_mn[r, c]
                            )
                else:
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit
                            else col0 - causal_row_offset - self.window_size_right
                        )
                        # TODO: do we need col_limit_sink?
                        row_limit_bot = col0 - causal_row_offset + self.window_size_left
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            row_idx = t0ScS_mn[r, 0][ROW]
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if row_idx < row_limit_top or row_idx > row_limit_bot
                                else acc_S_mn[r, c]
                            )

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n
        r2p = True
        if const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -Float32.inf
                        # For some reason the 2 lines above generate really bad SASS
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= seqlenk_col_limit else acc_S[i]
                else:
                    mask_r2p(acc_S, seqlenk_col_limit, arch=100, rank1=True)
        else:  # Causal or local
            causal_row_offset = 1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # if cute.arch.thread_idx()[0] % 32 == 0:
                #     cute.printf("tidx = %d, tidx tmem = %d, row_idx = %d, col_limit_right = %d, causal_row_offset = %d\n", cute.arch.thread_idx()[0], thr_tmem_load.thr_idx, row_idx, col_limit_right, causal_row_offset)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                else:
                    mask_r2p(acc_S, col_limit_right, arch=100, rank1=True)
            else:
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                    if const_expr(mask_seqlen):
                        col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                else:
                    col_limit_right = self.tile_n
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                # if cute.arch.thread_idx()[0] == 0 or cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", m_block, n_block, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                    col_idx = tScS_t2r[i][1]
                    acc_S[i] = (
                        -Float32.inf
                        if col_idx >= col_limit_right or col_idx < col_limit_left
                        else acc_S[i]
                    )


    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r : cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[cutlass.Int32],
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
    ) -> None:
        '''
        Backward pass: mask S = K @ Q.T where n_block tiles seqlen_k and m_block tiles seqlen_q.
        '''
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"

        tidx = cute.arch.thread_idx()[0] % 128

        seqlenk_row_limit = self.seqlen_k - n_block * self.tile_n
        if const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if tScS_t2r[0][0] >= seqlenk_row_limit:
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = -cutlass.Float32.inf
        else:  # Causal or local
            causal_row_offset = (self.seqlen_q - self.seqlen_k - 1) - m_block * self.tile_m
            row_idx = tScS_t2r[0][0] + n_block * self.tile_n
            
            if const_expr(mask_causal):
                col_limit_left = row_idx + causal_row_offset
                ncol = const_expr(cute.size(tScS_t2r.shape))
                # if tidx == 32 and wg_idx == 1:
                #     cute.printf("row idx = {}, causal_row_offset = {}, col_limit_left = {}, first column = {}, last column = {} ", row_idx, causal_row_offset, col_limit_left, tScS_t2r[0][1], tScS_t2r[ncol - 1][1])
                if const_expr(mask_seqlen):
                    if tScS_t2r[0][0] >= seqlenk_row_limit:
                        col_limit_left = self.tile_m
                for i in cutlass.range(ncol, unroll_full=True):
                    acc_S[i] = (
                        -cutlass.Float32.inf if tScS_t2r[i][1] <= col_limit_left else acc_S[i]
                    )
            # TODO: local