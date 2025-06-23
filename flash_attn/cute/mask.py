# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute

import flash_attn.cute.utils as utils


class AttentionMask:

    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        n_block_size: cutlass.Constexpr[int],
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1,  # only pass in if we're doing PackGQA
    ):
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.qhead_per_kvhead_packgqa = qhead_per_kvhead_packgqa

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
    ) -> None:
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        tScS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = utils.make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        seqlenk_col_limit = self.seqlen_k - n_block * self.n_block_size - thr_col_offset
        if not mask_causal:
            if mask_seqlen:
                # traverse column index.
                for c in range(cute.size(tScS_mn.shape[1])):
                    if t0ScS_mn[0, c][1] >= seqlenk_col_limit:
                        acc_S_mn[None, c].fill(-cutlass.Float32.inf)
        else:  # Causal
            # If PackGQA, we split the work of compute divmod among threads in the same row
            threads_per_row = thr_mma.tv_layout_C.shape[0][0]
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
                assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                tidx = thr_mma.thr_idx
                mma_m_idx = (m_block * self.m_block_size + tScS_mn[tidx % threads_per_row, 0][0]) // self.qhead_per_kvhead_packgqa
            causal_row_offset = 1 + self.seqlen_k - n_block * self.n_block_size - self.seqlen_q - thr_col_offset
            for r in range(cute.size(tScS_mn.shape[0])):
                # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                if cutlass.const_expr(self.qhead_per_kvhead_packgqa == 1):
                    row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
                else:
                    row_idx = utils.shuffle_sync(mma_m_idx, r % threads_per_row, width=threads_per_row)
                col_limit_right = row_idx + causal_row_offset
                if cutlass.const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # traverse column index.
                for c in range(cute.size(tScS_mn.shape[1])):
                    # only consider the column index, so the row index sets to 0.
                    if t0ScS_mn[0, c][1] >= col_limit_right:
                        acc_S_mn[r, c] = -cutlass.Float32.inf
