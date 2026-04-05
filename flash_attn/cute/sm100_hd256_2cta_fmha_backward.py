# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.


"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* mma_tiler_mn must be 64,64
* Batch size must be the same for Q, K, and V tensors
"""

import argparse
import enum
import math
import random
import time
from typing import Type, Tuple

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32

try:
    from flash_attn.cute.utils import make_cotiled_copy, warp_reduction_sum
    from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel import (
        BlackwellFusedMultiHeadAttentionBackwardDQKernel,
    )
    from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
        BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
    )
    from flash_attn.cute.mask import Sm100MaskEnum as MaskEnum
except ImportError:
    # Allow direct execution as a script (no package context).
    from flash_attn.cute.utils import make_cotiled_copy, warp_reduction_sum
    from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel import (
        BlackwellFusedMultiHeadAttentionBackwardDQKernel,
    )
    from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
        BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
    )
    from flash_attn.cute.mask import Sm100MaskEnum as MaskEnum


SM100_TMEM_CAPACITY_COLUMNS = 512
LAYOUT_RANK_CONSTANT = 3


@cute.jit
def split_wg(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """Split warp group."""
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == LAYOUT_RANK_CONSTANT):
        p = cute.composition(
            t,
            cute.make_layout((
                t.shape[0],
                t.shape[1],
                (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
            )),
        )
        ret = p[None, None, (wg_idx, None)]
    else:
        p = cute.composition(
            t,
            cute.make_layout((
                t.shape[0],
                t.shape[1],
                t.shape[2],
                (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
            )),
        )
        ret = p[None, None, None, (wg_idx, None)]
    return ret


def Tmemory_offset(lane, col):
    """Tensor memory offset."""
    return (lane << 16) + col


permute_order = (0, 1, 2, 3, 4)

class BlackwellFusedMultiHeadAttentionBackward:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        element_dtype: type[cutlass.Numeric],
        acc_dtype: type[cutlass.Numeric],
        mma_tiler: tuple[int, int, int],
        dkdv_mma_tiler: tuple[int, int, int],
        varlen: bool,
        is_causal: bool,
        mask_type: MaskEnum,
        window_size_left: int | None,
        window_size_right: int | None,
        split_head: bool,
        use_clc_dynamic_scheduler: bool = False,
    ):
        """Initialization."""
        self.element_dtype = element_dtype
        self.acc_dtype = acc_dtype
        self.varlen = varlen
        self.is_causal = is_causal
        self.mask_type = mask_type
        self.window_size_left = None if window_size_left < 0 else window_size_left
        self.window_size_right = None if window_size_right < 0 else window_size_right


        print(f'self.is_causal = {self.is_causal}', flush=True)
        print(f'window_size_left = {window_size_left}', flush=True)
        print(f'window_size_right = {window_size_right}', flush=True)
        print(f'mask_type = {mask_type}', flush=True)

        # =================== Sum OdO ================================
        self.sum_OdO_max_threads_per_block = 128
        self.sum_OdO_block_q = 16
        self.sum_OdO_num_threads_d = 8
        self.sum_OdO_num_threads_q = self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
        self.sum_OdO_elem_per_load = 2

        # Keep the original (known-good) mask selection for dQ kernel.
        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            element_dtype,
            acc_dtype,
            mma_tiler,
            varlen,
            is_causal,
            self.mask_type,
            window_size_left,
            window_size_right,
            False,
            split_head,
            use_clc_dynamic_scheduler=use_clc_dynamic_scheduler,
        )

        dkdv_cta_mma_tiler = (dkdv_mma_tiler[0], dkdv_mma_tiler[1], 256)

        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            element_dtype,
            acc_dtype,
            dkdv_cta_mma_tiler,
            varlen,
            self.is_causal,
            self.mask_type,
            window_size_left,
            window_size_right,
            use_clc_dynamic_scheduler=use_clc_dynamic_scheduler,
        )

    @cute.jit
    def __call__(
        self,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        LSE: cute.Tensor,
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        scale_softmax: cutlass.Float32,
        workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Host function to launch CuTeDSL kernel."""
        _, _, _, hb = problem_shape
        h, _ = hb
        h_r, h_k = h
        # (b, s, h_k, h_r, d) -> (s, d, ((h_r, h_k), b))
        mQ = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[4], hb),
                stride=(
                    Q.stride[1],
                    Q.stride[4],
                    (
                        (Q.shape[4], Q.shape[4] * Q.shape[3]),
                        (0 if self.varlen else cute.assume(Q.shape[1] * Q.shape[4] * h_r * h_k, divby=64)),
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        mK = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[1], K.shape[4], hb),
                stride=(
                    K.stride[1],
                    K.stride[4],
                    (
                        (0, K.shape[4]),
                        (0 if self.varlen else cute.assume(K.shape[1] * K.shape[4] * 1 * h_k, divby=64)),
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        mV = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[1], V.shape[4], hb),
                stride=(
                    V.stride[1],
                    V.stride[4],
                    (
                        (0, V.shape[4]),
                        (0 if self.varlen else cute.assume(V.shape[1] * V.shape[4] * 1 * h_k, divby=64)),
                    ),
                ),
            ),
        )
        mO = cute.make_tensor(O.iterator, mQ.layout)

        mdQ = cute.make_tensor(dQ.iterator, mQ.layout)
        mdK = cute.make_tensor(dK.iterator, mK.layout)
        mdV = cute.make_tensor(dV.iterator, mV.layout)
        mdO = cute.make_tensor(dO.iterator, mO.layout)

        # (b, h_k, h_r, s) -> (s, ((h_r, h_k), b))
        LSE = cute.make_tensor(
            LSE.iterator,
            cute.make_layout(
                (LSE.shape[3], hb),
                stride=(
                    LSE.stride[3],
                    (
                        (LSE.shape[3], LSE.shape[3] * LSE.shape[2]),
                        (0 if LSE.shape[0] == 1 else LSE.shape[1] * LSE.shape[2] * LSE.shape[3]),
                    ),
                ),
            ),
        )

        # =============================== Sum OdO ===============================
        sum_OdO_scale = cutlass.Float32(-1.0)
        LSE_scale = cutlass.Float32(-math.log2(math.e))
        sum_OdO, scaled_LSE, dQ_acc = self.get_workspace_tensor(problem_shape, workspace, self.acc_dtype, self.varlen)
        sum_OdO_grid = self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q)

        self.sum_OdO(
            mO,
            mdO,
            sum_OdO,
            LSE,
            scaled_LSE,
            cumulative_s_q,
            sum_OdO_scale,
            LSE_scale,
            problem_shape,
        ).launch(
            grid=sum_OdO_grid,
            block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        # Keep original order: dQ first, then dKdV.
        self.dq_kernel(
            problem_shape,
            Q,
            K,
            V,
            O,
            dQ,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            workspace,
            stream,
        )
        self.dkdv_kernel(
            problem_shape,
            Q,
            K,
            V,
            O,
            dK,
            dV,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            workspace,
            stream,
        )

    @cute.kernel
    def sum_OdO(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        scaled_lse: cute.Tensor,
        cumulative_s_q: cute.Tensor | None,
        sum_OdO_scale: cutlass.Float32,
        lse_scale: cutlass.Float32,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
    ):
        """CuTeDSL kernel for sum(dot(O, dO))."""
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, tidy, _ = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]
        offset = 0
        if cutlass.const_expr(self.varlen):
            assert isinstance(cumulative_s_q, cute.Tensor)
            offset = cumulative_s_q[bidz]
            seqlen_q = cumulative_s_q[bidz + 1] - offset

        for idx_q_t in cutlass.range(tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True):
            idx_q = idx_q_t + self.sum_OdO_block_q * bidx
            if idx_q < seqlen_q:
                O_bhq = O[idx_q + offset, None, (bidy, bidz)]
                O_bhq = cute.logical_divide(O_bhq, cute.make_layout(self.sum_OdO_elem_per_load))
                dO_bhq = dO[idx_q + offset, None, (bidy, bidz)]
                dO_bhq = cute.logical_divide(dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load))

                idx_d_start = tidx
                idx_d_step = self.sum_OdO_num_threads_d
                acc = 0.0
                for idx_d in cutlass.range(idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step):
                    O_frag = O_bhq[None, idx_d].load().to(self.acc_dtype)
                    dO_frag = dO_bhq[None, idx_d].load().to(self.acc_dtype)
                    prod_frag = O_frag * dO_frag
                    acc += prod_frag.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

                acc = warp_reduction_sum(acc, threads_in_group=self.sum_OdO_num_threads_d)

                if tidx == 0:
                    lse_bhq = lse[idx_q + offset, (bidy, bidz)]
                    sum_OdO[idx_q + offset, (bidy, bidz)] = sum_OdO_scale * acc
                    scaled_lse[idx_q + offset, (bidy, bidz)] = lse_scale * lse_bhq

    @staticmethod
    def get_workspace_size(s_q: int, d: int, h: int, b: int, acc_dtype: type[cutlass.Numeric]):
        """Get workspace size."""
        d = (d + 7) // 8 * 8  # round up to 8
        s_q = (s_q + 7) // 8 * 8  # round up to 8
        workspace_bytes = 0
        # OdO vector
        workspace_bytes += acc_dtype.width // 8
        # scaled LSE vector
        workspace_bytes += acc_dtype.width // 8
        # FP32 versions of outputs that are churned (start off with Q only)
        workspace_bytes += d * acc_dtype.width // 8
        return (b, s_q, h, workspace_bytes)

    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        block_q: int,
    ) -> tuple[Int32, Int32, Int32]:
        """Compute grid shape for sum_OdO kernel."""
        return (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )        

    def get_workspace_tensor(
        self,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        workspace: cute.Tensor,
        acc_dtype: type[cutlass.Numeric],
        varlen: bool,
    ) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        """Get workspace tensor."""
        D = problem_shape[2]
        H, B = cute.size(problem_shape[3][0]), problem_shape[3][1]
        H_r, H_k = problem_shape[3][0]
        D = cute.round_up(D, 8)

        # b = 1 for varlen, else batch_size
        b = workspace.shape[0]
        # s_q_sum for varlen, else s_q_max, already rounded to 8
        S_Q = workspace.shape[1]

        acc_bytes = acc_dtype.width // 8
        sum_OdO_bytes = cute.assume(b * H * S_Q * acc_bytes, divby=acc_bytes)
        scaled_lse_bytes = cute.assume(b * H * S_Q * acc_bytes, divby=acc_bytes)

        sum_OdO_iter = workspace.iterator
        scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes
        dQ_acc_iter = scaled_lse_iter + scaled_lse_bytes

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)
        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype)

        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout(
                (S_Q, ((H_r, H_k), B)),
                stride=(1, ((S_Q, S_Q * H_r), 0 if varlen else S_Q * H)),
            ),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout(
                (S_Q, ((H_r, H_k), B)),
                stride=(1, ((S_Q, S_Q * H_r), 0 if varlen else S_Q * H)),
            ),
        )
        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (S_Q, D, ((H_r, H_k), B)),
                stride=(D, 1, ((D * S_Q, D * S_Q * H_r), 0 if varlen else D * S_Q * H)),
            ),
        )

        return sum_OdO, scaled_lse, dQ_acc

def run(
    s_q_max: int,
    s_k_max: int,
    h_q: int,
    h_k: int,
    d: int,
    b: int,
    varlen: bool,
    is_causal: bool,
    element_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    dkdv_mma_tiler_mn: Tuple[int, int],
    scale_softmax: float,
    window_size: Tuple[int, int],
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    split_head: bool = False,
    use_clc_dynamic_scheduler: bool = False,
    **kwargs,
):
    print(f"Running Blackwell SM100 FMHA bwd test with:")
    print(f"  s_q_max: {s_q_max}")
    print(f"  s_k_max: {s_k_max}")
    print(f"  h_q: {h_q}")
    print(f"  h_k: {h_k}")
    print(f"  d: {d}")
    print(f"  b: {b}")
    print(f"  varlen: {varlen}")
    print(f"  is_causal: {is_causal}")
    print(f"  element_dtype: {element_dtype}")
    print(f"  acc_dtype: {acc_dtype}")
    print(f"  mma_tiler_mn: {mma_tiler_mn}")
    print(f"  dkdv_mma_tiler_mn: {dkdv_mma_tiler_mn}")
    print(f"  scale_softmax: {scale_softmax}")
    print(f"  window_size: {window_size}")
    print(f"  warmup_iterations: {warmup_iterations}")
    print(f"  iterations: {iterations}")
    print(f"  skip_ref_check: {skip_ref_check}")
    print(f"  split_head: {split_head}")
    print(f"  use_clc_dynamic_scheduler: {use_clc_dynamic_scheduler}")

    torch.manual_seed(42)
    random.seed(123)

    if d not in {64, 128, 256}:
        raise ValueError("head dimension must be 64, 128, or 256")

    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")

    if element_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16}:
        raise ValueError("in_dtype must be Float8E4M3FN or Float16 or BFloat16")

    if acc_dtype not in {cutlass.Float32}:
        raise ValueError("acc_dtype must be Float32")

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    h_r = h_q // h_k
    orig_b = b

    if scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    def create_and_permute_tensor(
        shape,
        permute_order,
        dtype,
        min_val=-2,
        max_val=2,
        is_dynamic_layout=True,
        zeros: bool = False,
    ):
        # (b, s, h_k, h_r, d)
        if zeros:
            ref_tensor = torch.zeros(*shape, dtype=torch.float32).permute(permute_order)
        else:
            ref_tensor = (
                torch.empty(*shape, dtype=torch.float32)
                .random_(min_val, max_val)
                .permute(permute_order)
            )

        torch_dtype = cutlass_torch.dtype(dtype)

        dst_tensor = ref_tensor.to(dtype=torch_dtype).cuda()
        cute_tensor = from_dlpack(dst_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=4
            ).mark_compact_shape_dynamic(
                mode=4, stride_order=(0, 1, 2, 3, 4), divisibility=64
            )

        return ref_tensor, cute_tensor, dst_tensor

    # create sequence lengths for variable length inputs
    cumulative_s_q = [0]
    cumulative_s_k = [0]
    if varlen:
        s_q_list = [random.randint(s_q_max // 2, s_q_max) for _ in range(b)]
        # TODO: We only support s_q == s_k for sliding window at this point
        s_k_list = s_q_list
        # Generate cumulative sequence lengths for variable length inputs

        for i in range(b):
            cumulative_s_q.append(cumulative_s_q[-1] + s_q_list[i])
            cumulative_s_k.append(cumulative_s_k[-1] + s_k_list[i])
        s_q = sum(s_q_list)
        s_k = sum(s_k_list)
        s_q_max = max(s_q_list)
        s_k_max = max(s_k_list)
        b = 1
    else:
        s_q = s_q_max
        s_k = s_k_max

    mask_type = MaskEnum.WINDOW_MASK_INFERENCE
    if not is_causal and (varlen or s_q % mma_tiler_mn[0] != 0):
        mask_type = MaskEnum.RESIDUAL_MASK

    window_size_left, window_size_right = window_size
    if is_causal:
        window_size_right = 0
    if window_size_left >= s_k_max - 1:
        raise ValueError("window_size_left must be less than s_k_max - 1")
    if window_size_right >= s_q_max - 1:
        raise ValueError("window_size_right must be less than s_q_max - 1")

    problem_shape = (s_q_max, s_k_max, d, ((h_r, h_k), orig_b))
    print (f'problem_shape = {problem_shape}', flush=True)
    cumulative_s_q_torch_tensor = (
        torch.tensor(cumulative_s_q, dtype=torch.int32).cuda() if varlen else None
    )
    cumulative_s_k_torch_tensor = (
        torch.tensor(cumulative_s_k, dtype=torch.int32).cuda() if varlen else None
    )
    cumulative_s_q_cute_tensor = (
        from_dlpack(cumulative_s_q_torch_tensor).mark_layout_dynamic()
        if varlen
        else None
    )
    cumulative_s_k_cute_tensor = (
        from_dlpack(cumulative_s_k_torch_tensor).mark_layout_dynamic()
        if varlen
        else None
    )

    q_ref, q_tensor, q_torch = create_and_permute_tensor(
        (b, s_q, h_k, h_r, d), permute_order, element_dtype, is_dynamic_layout=True
    )
    # dQ/dK/dV are outputs of the backward kernels. dKdV kernel uses reduction stores,
    # so these buffers must be zero-initialized.
    dq_ref, dq_tensor, dq_torch = create_and_permute_tensor(
        (b, s_q, h_k, h_r, d), permute_order, element_dtype, is_dynamic_layout=True, zeros=True,
    )
    k_ref, k_tensor, k_torch = create_and_permute_tensor(
        (b, s_k, h_k, 1, d), permute_order, element_dtype, is_dynamic_layout=True,
    )
    dk_ref, dk_tensor, dk_torch = create_and_permute_tensor(
        (b, s_k, h_k, 1, d), permute_order, element_dtype, is_dynamic_layout=True, zeros=True
    )
    v_ref, v_tensor, v_torch = create_and_permute_tensor(
        (b, s_k, h_k, 1, d), permute_order, element_dtype, is_dynamic_layout=True
    )
    dv_ref, dv_tensor, dv_torch = create_and_permute_tensor(
        (b, s_k, h_k, 1, d), permute_order, element_dtype, is_dynamic_layout=True, zeros=True
    )
    do_ref, do_tensor, do_torch = create_and_permute_tensor(
        (b, s_q, h_k, h_r, d), permute_order, element_dtype, is_dynamic_layout=True
    )
    o_ref, o_tensor, o_torch = create_and_permute_tensor(
        (b, s_q, h_k, h_r, d), permute_order, element_dtype, is_dynamic_layout=True
    )

    lse_ref = cutlass_torch.create_and_permute_torch_tensor(
        (b, h_k, h_r, s_q),
        cutlass.torch.dtype(acc_dtype),
        permute_order=(0, 1, 2, 3),
        init_type=cutlass.torch.TensorInitType.RANDOM,
        init_config=cutlass.torch.RandomInitConfig(min_val=10, max_val=11),
    )
    lse_torch = lse_ref.cuda()
    lse_tensor = from_dlpack(lse_torch, assumed_align=16)
    lse_tensor = lse_tensor.mark_layout_dynamic(leading_dim=3)

    mma_tiler = (*mma_tiler_mn, d)

    fmha_bwd = BlackwellFusedMultiHeadAttentionBackward(
        element_dtype,
        acc_dtype,
        mma_tiler,
        dkdv_mma_tiler_mn,
        varlen,
        is_causal,
        mask_type,
        window_size_left,
        window_size_right,
        split_head,
        use_clc_dynamic_scheduler=use_clc_dynamic_scheduler,
    )

    workspace_size = BlackwellFusedMultiHeadAttentionBackward.get_workspace_size(
        s_q, d, h_q, b, acc_dtype,
    )
    workspace_torch = torch.zeros(workspace_size, dtype=torch.uint8).cuda()
    workspace = from_dlpack(workspace_torch, assumed_align=16).mark_layout_dynamic(leading_dim=3)

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_fmha_bwd = cute.compile(
        fmha_bwd,
        problem_shape,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        do_tensor,
        lse_tensor,
        cumulative_s_q_cute_tensor,
        cumulative_s_k_cute_tensor,
        scale_softmax,
        workspace,
        current_stream,
        options = "--generate-line-info"
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    # for _ in range(warmup_iterations):
    #     compiled_fmha_bwd(
    #         problem_shape,
    #         q_tensor,
    #         k_tensor,
    #         v_tensor,
    #         o_tensor,
    #         dq_tensor,
    #         dk_tensor,
    #         dv_tensor,
    #         do_tensor,
    #         lse_tensor,
    #         cumulative_s_q_cute_tensor,
    #         cumulative_s_k_cute_tensor,
    #         scale_softmax,
    #         workspace,
    #         current_stream,
    #     )

    # for _ in range(iterations):
    #     compiled_fmha_bwd(
    #         problem_shape,
    #         q_tensor,
    #         k_tensor,
    #         v_tensor,
    #         o_tensor,
    #         dq_tensor,
    #         dk_tensor,
    #         dv_tensor,
    #         do_tensor,
    #         lse_tensor,
    #         cumulative_s_q_cute_tensor,
    #         cumulative_s_k_cute_tensor,
    #         scale_softmax,
    #         workspace,
    #         current_stream,
    #     )

    if not skip_ref_check:
        workspace_torch.fill_(0)
        compiled_fmha_bwd(
            problem_shape,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            dq_tensor,
            dk_tensor,
            dv_tensor,
            do_tensor,
            lse_tensor,
            cumulative_s_q_cute_tensor,
            cumulative_s_k_cute_tensor,
            scale_softmax,
            workspace,
            current_stream,
        )
        torch.cuda.synchronize()
        print("Verifying results...")


        q_ref = q_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        k_ref = k_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        v_ref = v_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        o_ref = o_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        do_ref = do_ref.cuda().to(cutlass.torch.dtype(element_dtype))
        dv = dv_torch.to(dtype=torch.float32)
        dk = dk_torch.to(dtype=torch.float32)
        dq = dq_torch.to(dtype=torch.float32)

        dv_ref, dk_ref, dq_ref = fmha_bwd_reference(
            problem_shape,
            q_ref,
            k_ref,
            v_ref,
            do_ref,
            o_ref,
            lse_torch,
            cumulative_s_q_torch_tensor,
            cumulative_s_k_torch_tensor,
            is_causal,
            window_size,
        )
        dv_pt, dk_pt, dq_pt = fmha_bwd_reference(
            problem_shape,
            q_ref,
            k_ref,
            v_ref,
            do_ref,
            o_ref,
            lse_torch,
            cumulative_s_q_torch_tensor,
            cumulative_s_k_torch_tensor,
            is_causal,
            window_size,
            upcast=False,
            reorder_ops=True,
        )

        # Below codes refer to the flash-attention's implementation
        # see: https://github.com/Dao-AILab/flash-attention/blob/7321879fde54f09ed94f7f6ce9377e2f4cf1fac0/hopper/test_flash_attn.py#L105
        rtol = 2
        dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item()
        dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item()
        dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
        print(f"Pytorch dv max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"Pytorch dv mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
        print(f"Pytorch dk max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"Pytorch dk mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"Pytorch dq max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"Pytorch dq mean diff: {(dq_pt - dq_ref).abs().mean().item()}")

        print(f"dv max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dv mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dk max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dk mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dq max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dq mean diff: {(dq - dq_ref).abs().mean().item()}")

        assert (dv - dv_ref).abs().max().item() <= rtol * (
            dv_pt - dv_ref
        ).abs().max().item() + dv_atol
        assert (dk - dk_ref).abs().max().item() <= rtol * (
            dk_pt - dk_ref
        ).abs().max().item() + dk_atol

        assert (dq - dq_ref).abs().max().item() <= rtol * (
            dq_pt - dq_ref
        ).abs().max().item() + dq_atol

        print("Results verified successfully!")

    def generate_tensors():
        _, q_tensor_new, _ = create_and_permute_tensor(
            (b, s_q, h_k, h_r, d),
            permute_order,
            element_dtype,
            is_dynamic_layout=True,
        )
        _, dq_tensor_new, _ = create_and_permute_tensor(
            (b, s_q, h_k, h_r, d),
            permute_order,
            element_dtype,
            is_dynamic_layout=True,
            zeros=True,
        )
        _, k_tensor_new, _ = create_and_permute_tensor(
            (b, s_k, h_k, 1, d),
            permute_order,
            element_dtype,
            is_dynamic_layout=True,
        )
        _, dk_tensor_new, _ = create_and_permute_tensor(
            (b, s_k, h_k, 1, d), permute_order, element_dtype, is_dynamic_layout=True
        )
        _, v_tensor_new, _ = create_and_permute_tensor(
            (b, s_k, h_k, 1, d), permute_order, element_dtype, is_dynamic_layout=True
        )
        _, dv_tensor_new, _ = create_and_permute_tensor(
            (b, s_k, h_k, 1, d), permute_order, element_dtype, is_dynamic_layout=True
        )
        _, do_tensor_new, _ = create_and_permute_tensor(
            (b, s_q, h_k, h_r, d),
            permute_order,
            element_dtype,
            is_dynamic_layout=True,
        )
        _, o_tensor_new, _ = create_and_permute_tensor(
            (b, s_q, h_k, h_r, d),
            permute_order,
            element_dtype,
            is_dynamic_layout=True,
        )

        lse_ref_new = cutlass_torch.create_and_permute_torch_tensor(
            (b, h_k, h_r, s_q),
            cutlass.torch.dtype(acc_dtype),
            permute_order=(0, 1, 2, 3),
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(min_val=10, max_val=11),
        )
        lse_torch_new = lse_ref_new.cuda()
        lse_tensor_new = from_dlpack(lse_torch_new, assumed_align=16)
        lse_tensor_new = lse_tensor_new.mark_layout_dynamic(leading_dim=3)

        return testing.JitArguments(
            problem_shape,
            q_tensor_new,
            k_tensor_new,
            v_tensor_new,
            o_tensor_new,
            dq_tensor_new,
            dk_tensor_new,
            dv_tensor_new,
            do_tensor_new,
            lse_tensor_new,
            cumulative_s_q_cute_tensor,
            cumulative_s_k_cute_tensor,
            scale_softmax,
            workspace,
            current_stream,
        )

    # workspace_count = 1
    # if use_cold_l2:
    #     one_workspace_bytes = (
    #         q_torch.numel() * q_torch.element_size()
    #         + dq_torch.numel() * dq_torch.element_size()
    #         + k_torch.numel() * k_torch.element_size()
    #         + dk_torch.numel() * dk_torch.element_size()
    #         + v_torch.numel() * v_torch.element_size()
    #         + dv_torch.numel() * dv_torch.element_size()
    #         + do_torch.numel() * do_torch.element_size()
    #         + o_torch.numel() * o_torch.element_size()
    #         + lse_torch.numel() * lse_torch.element_size()
    #     )
    #     workspace_count = testing.get_workspace_count(
    #         one_workspace_bytes, warmup_iterations, iterations
    #     )

    # exec_time = testing.benchmark(
    #     compiled_fmha_bwd,
    #     workspace_generator=generate_tensors,
    #     workspace_count=workspace_count,
    #     stream=current_stream,
    #     warmup_iterations=warmup_iterations,
    #     iterations=iterations,
    # )

    #print(f"Execution time: {exec_time} microseconds")

    #return exec_time  # Return execution time in microseconds

def fmha_bwd_reference(
    problem_shape: Tuple[int, int, int, Tuple[Tuple[int, int], int]],
    Q: torch.Tensor,    # [B, Q, H_K, H_R, D]
    K: torch.Tensor,    # [B, K, H_K, 1,   D]
    V: torch.Tensor,    # [B, K, H_K, 1,   D]
    dO: torch.Tensor,   # [B, Q, H_K, H_R, D]
    O: torch.Tensor,    # [B, Q, H_K, H_R, D]
    LSE: torch.Tensor,  # [B, Q, H_K, H_R]
    cumulative_s_q: torch.Tensor | None,
    cumulative_s_k: torch.Tensor | None,
    is_causal: bool,
    window_size: Tuple[int, int],
    upcast=True,
    reorder_ops=False,
):
    s_q_max, s_k_max, d, hb = problem_shape
    h, orig_b = hb
    h_r, h_k = h
    is_gqa = h_r != 1

    if upcast:
        Q = Q.to(dtype=torch.float32)
        K = K.to(dtype=torch.float32)
        V = V.to(dtype=torch.float32)
        dO = dO.to(dtype=torch.float32)
        O = O.to(dtype=torch.float32)
        LSE = LSE.to(dtype=torch.float32)

    softmax_scale = 1.0 / math.sqrt(problem_shape[2])
    dV = torch.zeros_like(V)
    dK = torch.zeros_like(K)
    dQ = torch.zeros_like(Q)

    for b in range(orig_b):
        q_offset = cumulative_s_q[b] if cumulative_s_q is not None else 0
        k_offset = cumulative_s_k[b] if cumulative_s_k is not None else 0
        s_q = (
            cumulative_s_q[b + 1] - cumulative_s_q[b]
            if cumulative_s_q is not None
            else s_q_max
        )
        s_k = (
            cumulative_s_k[b + 1] - cumulative_s_k[b]
            if cumulative_s_k is not None
            else s_k_max
        )
        for h_k_idx in range(h_k):
            b_idx = b if cumulative_s_k is None else 0
            cur_K = K[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :]
            cur_V = V[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :]
            for h_r_idx in range(h_r):
                cur_Q =  Q[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :]
                cur_dO = dO[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :]
                cur_O =  O[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :]
                cur_LSE = LSE[b_idx, h_k_idx, h_r_idx, q_offset : q_offset + s_q]
                if not reorder_ops:
                    cur_S = torch.einsum("qd,kd->qk", cur_Q, cur_K)
                else:
                    cur_S = torch.einsum("qd,kd->qk", cur_Q, cur_K)
                cur_S = cur_S * softmax_scale
                window_size_left, window_size_right = window_size
                if is_causal:
                    window_size_right = 0
                if window_size_left >= 0 or window_size_right >= 0:
                    q_coords = torch.arange(0, s_q).cuda().view(-1, 1)
                    k_coords = torch.arange(0, s_k).cuda().view(1, -1)
                    if window_size_left < 0:
                        mask = k_coords > q_coords + s_k - s_q + window_size_right
                    else:
                        mask = (k_coords > q_coords + s_k - s_q + window_size_right) | (
                            k_coords < q_coords + s_k - s_q - window_size_left
                        )
                    cur_S = cur_S.masked_fill(mask, -torch.inf)
                #print(f'cur_LSE.reshape(cur_LSE.shape[0], 1) = {cur_LSE.reshape(cur_LSE.shape[0], 1)}')
                scaled_lse = cur_LSE * math.log2(math.exp(1.0)) * (-1)
                #print(f'scaled_lse.reshape(cur_LSE.shape[0], 1) = {scaled_lse.reshape(scaled_lse.shape[0], 1)}')
                cur_P = torch.exp2(cur_S * math.log2(math.exp(1.0)) + scaled_lse.reshape(scaled_lse.shape[0], 1))
                #print(f'cur_P = {cur_P}', flush=True)

                cur_P = torch.exp(cur_S - cur_LSE.reshape(cur_LSE.shape[0], 1))
                #print(f'cur_P(origin) = {cur_P}', flush=True)
                cur_PT = cur_P.transpose(1, 0).to(dtype=Q.dtype)
                cur_dV = torch.einsum("kq,qd->kd", [cur_PT, cur_dO])

                cur_dP = torch.einsum("qd,kd->qk", cur_dO, cur_V)
                #print(f'cur_dP = {cur_dP}', flush=True)
                cur_D = torch.einsum("qd,qd->qd", cur_O, cur_dO)
                cur_D = cur_D.sum(dim=1)
                cur_D = -cur_D.reshape(cur_D.shape[0], 1)
                #print(f'cur_dP + cur_D = {cur_dP + cur_D}', flush=True)
                cur_dS = cur_P * (cur_dP + cur_D) * softmax_scale
                # for i in range(cur_dS.shape[0]):
                #     print(f'index = {i}, cur_dS = {cur_dS[i]}', flush=True)
                #print(f'cur_P * (cur_dP + cur_D) = {cur_P * (cur_dP + cur_D)}', flush=True)
                torch.set_printoptions(threshold=torch.inf)  # 或者一个足够大的整数
                #print(f'scaled_lse = {scaled_lse}')
                cur_dS = cur_dS.to(dtype=Q.dtype)
                cur_dST = cur_dS.transpose(1, 0)
                cur_dK = torch.einsum("kq,qd->kd", cur_dST, cur_Q)
                cur_dQ = torch.einsum("qk,kd->qd", cur_dS, cur_K)


                dQ[b_idx, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :] = cur_dQ
                dV[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :] += cur_dV
                dK[b_idx, k_offset : k_offset + s_k, h_k_idx, 0, :] += cur_dK

    dV = dV.to(dtype=torch.float32)
    dK = dK.to(dtype=torch.float32)
    dQ = dQ.to(dtype=torch.float32)

    return dV, dK, dQ

if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(description="Example of bwd FMHA on Blackwell.")

    parser.add_argument(
        "--element_dtype",
        type=cutlass.dtype,
        default=cutlass.Float16,
        help="Input data type",
    )

    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        help="accumulator data type",
    )

    parser.add_argument(
        "--mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tile shape (M, N)",
    )

    parser.add_argument(
        "--dkdv_mma_tiler_mn",
        type=parse_comma_separated_ints,
        default=(128, 64),
        help="MMA tile shape (M, N)",
    )    

    parser.add_argument(
        "--is_causal",
        action="store_true",
        help="Whether to use causal mask",
    )

    parser.add_argument(
        "--s_q_max",
        type=int,
        default=1024,
        help="max sequence length of Q",
    )

    parser.add_argument(
        "--s_k_max",
        type=int,
        default=1024,
        help="max sequence length of K",
    )

    parser.add_argument(
        "--d",
        type=int,
        default=256,
        help="head dimension",
    )

    parser.add_argument(
        "--h_q",
        type=int,
        default=16,
        help="number of heads of Q",
    )

    parser.add_argument(
        "--h_k",
        type=int,
        default=1,
        help="number of heads of K",
    )

    parser.add_argument(
        "--b",
        type=int,
        default=4,
        help="batch size",
    )

    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Whether to use variable length inputs",
    )

    parser.add_argument(
        "--scale_softmax",
        type=float,
        default=0.0,
        help="Scaling factor to scale S (i.e. Q*K); if zero, defaults to 1/sqrt(D)",
    )

    parser.add_argument(
        "--window_size",
        type=parse_comma_separated_ints,
        default=(-1, -1),
        help="Sliding window size",
    )

    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of iterations for warmup",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations after warmup",
    )

    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference check",
    )

    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    parser.add_argument(
        "--split_head",
        action="store_true",
        default=False,
        help="Dq kernel use split head",
    )

    parser.add_argument(
        "--use_clc_dynamic_scheduler",
        action="store_true",
        default=False,
        help="Use CLC dynamic tile scheduling for dQ kernel",
    )

    args = parser.parse_args()

    run(
        args.s_q_max,
        args.s_k_max,
        args.h_q,
        args.h_k,
        args.d,
        args.b,
        args.varlen,
        args.is_causal,
        args.element_dtype,
        args.acc_dtype,
        args.mma_tiler_mn,
        args.dkdv_mma_tiler_mn,
        args.scale_softmax,
        args.window_size,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.split_head,
        args.use_clc_dynamic_scheduler,
    )

    print("PASS")

