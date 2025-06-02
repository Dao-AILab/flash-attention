# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute

from flash_attn.cute.utils import make_acc_tensor_mn_view


class AttentionMask:

    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        n_block_size: cutlass.Constexpr[int],
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        *,
        loc=None,
        ip=None
    ):
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.m_block_size, self.n_block_size, self.seqlen_q, self.seqlen_k]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.m_block_size, self.n_block_size, self.seqlen_q, self.seqlen_k], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return AttentionMask(*(tuple(obj_list)), loc=self._loc)

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
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        tScS_mn = make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        seqlenk_col_limit = self.seqlen_k - n_block * self.n_block_size - thr_col_offset
        if not mask_causal:
            if mask_seqlen:
                # traverse column index.
                for c in range(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(seqlenk_col_limit, t0ScS_mn[0, c][1] + 1):
                        acc_S_mn[None, c].fill(-cutlass.Float32.inf)
        else:  # Causal
            causal_row_offset = 1 + self.seqlen_k - n_block * self.n_block_size - self.seqlen_q - thr_col_offset
            for r in range(cute.size(tScS_mn.shape[0])):
                # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
                col_limit_right = row_idx + causal_row_offset
                if cutlass.const_expr(mask_seqlen):
                    col_idx = cutlass.min(col_limit_right, seqlenk_col_limit)
                # traverse column index.
                for c in range(cute.size(tScS_mn.shape[1])):
                    # only consider the column index, so the row index sets to 0.
                    if cute.elem_less(col_limit_right, t0ScS_mn[0, c][1] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
