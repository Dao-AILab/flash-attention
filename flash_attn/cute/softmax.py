# Copyright (c) 2025, Tri Dao.

import math
import operator

import cutlass
import cutlass.cute as cute

from flash_attn.cute.utils import warp_reduce, make_acc_tensor_mn_view, exp2f, log2f


class Softmax:

    def __init__(
        self,
        softmax_scale_log2: cutlass.Float32,
        *,
        loc=None, ip=None
    ):
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
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        is_first_n_block: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first_n_block: is first n_block
        :type is_first_n_block: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        row_scale = cute.make_fragment_like(row_max, cutlass.Float32)
        # Each iteration processes one row of acc_S
        for r in range(cute.size(row_max)):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)
            row_max_cur_row = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            row_max_cur_row = warp_reduce(row_max_cur_row, cute.arch.fmax, width=4)
            if cutlass.const_expr(is_first_n_block):
                if check_inf:
                    row_max_cur_row = 0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                row_max_cur_row_scaled = row_max_cur_row * self.softmax_scale_log2
                acc_S_row_exp = exp2f(acc_S_row * self.softmax_scale_log2 - row_max_cur_row_scaled)
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                row_scale[r] = 1.0
            else:
                row_max_prev_row = row_max[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
                if check_inf:
                    row_max_cur_row = 0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                row_max_cur_row_scaled = row_max_cur_row * self.softmax_scale_log2
                acc_S_row_exp = exp2f(acc_S_row * self.softmax_scale_log2 - row_max_cur_row_scaled)
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                # rescale = exp2f(row_max_prev_row * self.softmax_scale_log2 - row_max_cur_row_scaled)
                row_scale[r] = exp2f((row_max_prev_row - row_max_cur_row) * self.softmax_scale_log2)
                acc_S_row_sum = acc_S_row_sum + row_sum[r] * row_scale[r]
            row_max[r] = row_max_cur_row
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale

    @cute.jit
    def finalize(
        self,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        final_scale: cute.Float32 = 1.0
    ) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp.
        :param row_sum: row_sum tensor
        :type row_sum: cute.Tensor
        """
        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, cutlass.Float32)
        for r in range(cute.size(row_sum)):
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = ((row_max[r] * self.softmax_scale_log2 + log2f(row_sum_cur)) * LN2
                          if not acc_O_mn_row_is_zero_or_nan else -cutlass.Float32.inf)
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = make_acc_tensor_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in range(cute.size(row_scale)):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
