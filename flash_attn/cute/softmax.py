# Copyright (c) 2025, Tri Dao.

import math
import operator

import cutlass
import cutlass.cute as cute

import flash_attn.cute.utils as utils


class Softmax:

    def __init__(self, scale_log2: cutlass.Float32, num_rows: cutlass.Constexpr[int]):
        self.scale_log2 = scale_log2
        self.row_max = cute.make_fragment(num_rows, cutlass.Float32)
        self.row_sum = cute.make_fragment_like(self.row_max)

    def reset(self) -> None:
        self.row_max.fill(-cutlass.Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, cutlass.Float32)
        # Each iteration processes one row of acc_S
        for r in range(cute.size(self.row_max)):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)
            row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            row_max_cur = utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)
            if cutlass.const_expr(is_first):
                if check_inf:
                    row_max_cur = 0.0 if row_max_cur == -cutlass.Float32.inf else row_max_cur
                row_max_cur_scaled = row_max_cur * self.scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * self.scale_log2 - row_max_cur_scaled)
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                row_scale[r] = 1.0
            else:
                row_max_prev = self.row_max[r]
                row_max_cur = cute.arch.fmax(row_max_prev, row_max_cur)
                if check_inf:
                    row_max_cur = 0.0 if row_max_cur == -cutlass.Float32.inf else row_max_cur
                row_max_cur_scaled = row_max_cur * self.scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * self.scale_log2 - row_max_cur_scaled)
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                # row_scale[r] = utils.exp2f(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = utils.exp2f((row_max_prev - row_max_cur) * self.scale_log2)
                acc_S_row_sum = acc_S_row_sum + self.row_sum[r] * row_scale[r]
            self.row_max[r] = row_max_cur
            self.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale

    @cute.jit
    def finalize(self, final_scale: cute.Float32 = 1.0) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp.
        """
        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        self.row_sum.store(utils.warp_reduce(self.row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(self.row_max, cutlass.Float32)
        for r in range(cute.size(self.row_sum)):
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = self.row_sum[r] == 0.0 or self.row_sum[r] != self.row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(self.row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = self.row_sum[r]
            LN2 = math.log(2.0)
            self.row_sum[r] = (
                (self.row_max[r] * self.scale_log2 + utils.log2f(row_sum_cur)) * LN2
                if not acc_O_mn_row_is_zero_or_nan else -cutlass.Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = utils.make_acc_tensor_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in range(cute.size(row_scale)):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
