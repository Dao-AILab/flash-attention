# Copyright (c) 2025, Tri Dao.

import math
import operator

import cutlass
import cutlass.cute as cute

from flash_attn.cute.utils import warp_reduce, make_acc_tensor_mn_view, exp2f, log2f


class Softmax:

    def __init__(self, softmax_scale_log2: cutlass.Float32, *, loc=None, ip=None):
        self.softmax_scale_log2 = softmax_scale_log2
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.softmax_scale_log2]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.softmax_scale_log2], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return Softmax(*(tuple(obj_list)), loc=self._loc)

    @cute.jit
    def online_softmax_rescale_O(
        self,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        is_first_n_block: cutlass.Constexpr[bool],
        check_inf: cutlass.Constexpr[bool],
    ) -> None:
        """Apply online softmax and rescale acc_O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param acc_O: acc_O tensor
        :type acc_O: cute.Tensor
        :param is_first_n_block: is first n_block
        :type is_first_n_block: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        acc_O_mn = make_acc_tensor_mn_view(acc_O)
        # Each iteration processes one row of acc_S
        for r in range(cute.size(row_max)):
            # (n_block_size)
            acc_S_row = acc_S_mn[r, None].load()
            # row_max_cur_row => f32
            row_max_cur_row = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            # quad reduction for row_max
            row_max_cur_row = warp_reduce(row_max_cur_row, cute.arch.fmax, width=4)
            row_max_prev_row = -cutlass.Float32.inf
            if not is_first_n_block:
                row_max_prev_row = row_max[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if check_inf:
                row_max_cur_row = 0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
            rescale = 1.0
            if not is_first_n_block:
                max_diff = (row_max_prev_row - row_max_cur_row) * self.softmax_scale_log2
                rescale = exp2f(max_diff)
            # compute exp(x - max) using exp2(x * log_2(e) - max * log_2(e))
            row_max_cur_row_scaled = row_max_cur_row * self.softmax_scale_log2
            acc_S_row_exp = exp2f(acc_S_row * self.softmax_scale_log2 - row_max_cur_row_scaled)
            # acc_S_row_sum => f32
            acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            if not is_first_n_block:
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * rescale
                acc_S_row_sum = acc_S_row_sum + row_sum[r] * rescale
            row_max[r] = row_max_cur_row
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize(
        self,
        acc_O: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        final_scale: cute.Float32 = 1.0
    ) -> None:
        """Normalize acc_O by row_sum.

        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_sum: row_sum tensor
        :type row_sum: cute.Tensor
        """
        # do quad reduction for row_sum.
        acc_O_mn = make_acc_tensor_mn_view(acc_O)
        for r in range(cute.size(row_sum)):
            row_sum[r] = warp_reduce(row_sum[r], operator.add, width=4)
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = ((row_max[r] * self.softmax_scale_log2 + log2f(row_sum_cur)) * LN2
                          if not acc_O_mn_row_is_zero_or_nan else -cutlass.Float32.inf)
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale
