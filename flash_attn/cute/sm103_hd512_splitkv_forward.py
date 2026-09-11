# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import BFloat16, Float32, Int32, Int64, Pointer

from cutlass._mlir.dialects import llvm

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
cpy_dice = (None,) + mma_dice  # (CPY, #CPY_MMA, #CPY_M, #CPY_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
use_tensor_ssa_math = False  # experimental
fadd2 = partial(cute.arch.add_packed_f32x2, ftz=False, rnd="rn")
fmul2 = partial(cute.arch.mul_packed_f32x2, ftz=False, rnd="rn")
ffma2 = partial(cute.arch.fma_packed_f32x2, ftz=False, rnd="rn")
exp2 = partial(cute.math.exp2, fastmath=True)


class BlackwellHd512SplitKVFusedMultiHeadAttentionForward:
    """SM103 Q1 paged decode producing FA4 SplitKV partials.

    This kernel accepts the tensor layouts used by
    ``flash_attn_varlen_func`` for paged decode and writes FP32 partial output
    and partial LSE tensors for :class:`FlashAttentionForwardCombine`.

    Supported contract: BF16 Q/K/V, Q32/KV4/D512, GQA ratio 8, one packed query
    token per batch item, page size 64 or 128, and more than one KV split.
    """

    def __init__(
        self,
        headdim,
        grouped_head_tile,
        page_size,
        kv_layout="NHD",
    ):
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.page_size = page_size
        self.kv_layout = kv_layout

        assert headdim == 512
        assert grouped_head_tile == 8
        assert page_size in (64, 128)
        assert kv_layout in ("NHD", "HND")

        warpgroup_id = 0

        self.softmax_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        # One warpgroup owns the two MMA issue warps and the two TMA warps.
        self.mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        self.mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        self.tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        self.tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        self.mma_tma_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        self.threads_per_cta = warpgroup_id * warpgroup_threads

        self.use_reg_reconfig = False
        self.sp_stages = 2
        self.o_stages = 1
        # Deep independent K/V rings provide producer/consumer lookahead while
        # preserving distinct K and V ownership.
        self.k_stages = 6
        self.v_stages = 6

    def can_implement(
        self,
        problem_shape,
        kv_splits,
        q_dtype,
        kv_dtype,
        partial_dtype,
    ):
        b, h_q, h_k, s_k, d = problem_shape

        if kv_splits <= 1 or kv_splits > 256:
            raise ValueError("SM103 hd512 SplitKV requires 2 to 256 splits")
        if self.page_size == 64 and self.sp_stages != 2:
            raise ValueError("page64 two-slot ID ring requires sp_stages=2")
        if b <= 0 or s_k <= 0 or s_k % self.page_size != 0:
            raise ValueError("batch and paged KV capacity must be positive and page aligned")
        if (h_q, h_k, d) != (32, 4, 512):
            raise ValueError("SM103 hd512 SplitKV requires Q32/KV4/D512")

        if q_dtype is not cutlass.BFloat16 or kv_dtype is not cutlass.BFloat16:
            raise TypeError("SM103 hd512 SplitKV requires native BF16 Q/K/V")
        if partial_dtype is not cutlass.Float32:
            raise TypeError("SM103 hd512 SplitKV requires FP32 partial output")

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mOPartial: cute.Tensor,
        mLSEPartial: cute.Tensor,
        softmax_scale: Float32,
        mSeqUsedK: cute.Tensor,
        mPageTable: cute.Tensor,
        mNumSplitsDynamic: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        """Launch the Q1 paged decode partial kernel.

        Tensor layouts are the public FA4 SplitKV layouts:

        * ``mQ``: ``(total_q, 32, 512)`` where ``total_q == batch``;
        * ``mK`` / ``mV``: ``(num_pages, page_size, 4, 512)`` for NHD or
          ``(num_pages, 4, page_size, 512)`` for HND;
        * ``mOPartial``: ``(num_splits, total_q, 32, 512)`` FP32;
        * ``mLSEPartial``: ``(num_splits, total_q, 32)`` FP32;
        * ``mSeqUsedK``: ``(batch,)`` int32;
        * ``mPageTable``: ``(batch, page_table_width)`` int32.
        * ``mNumSplitsDynamic``: ``(batch,)`` int32 output for the combine.

        Each partial output is already normalized by its split-local softmax;
        ``mLSEPartial`` holds the corresponding natural-log LSE.  These are the
        inputs expected by FA4's production forward-combine kernel.
        """
        if cutlass.const_expr(len(mQ.shape) != 3):
            raise ValueError("SM103 hd512 SplitKV requires packed rank-3 Q")
        if cutlass.const_expr(len(mK.shape) != 4 or len(mV.shape) != 4):
            raise ValueError("SM103 hd512 SplitKV requires rank-4 paged K/V")
        if cutlass.const_expr(len(mOPartial.shape) != 4):
            raise ValueError("O partial must have shape (splits, total_q, heads, dim)")
        if cutlass.const_expr(len(mLSEPartial.shape) != 3):
            raise ValueError("LSE partial must have shape (splits, total_q, heads)")
        if cutlass.const_expr(
            len(mPageTable.shape) != 2
            or len(mSeqUsedK.shape) != 1
            or len(mNumSplitsDynamic.shape) != 1
        ):
            raise ValueError("page table must be rank 2 and seqused_k must be rank 1")

        total_q, h_q, d = mQ.shape
        if cutlass.const_expr(self.kv_layout == "NHD"):
            num_pages, page_size, h_k, d_k = mK.shape
        else:
            num_pages, h_k, page_size, d_k = mK.shape
        kv_splits = mOPartial.shape[0]
        batch = mPageTable.shape[0]
        page_table_width = mPageTable.shape[1]

        if cutlass.const_expr(mQ.element_type is not BFloat16):
            raise TypeError("SM103 hd512 SplitKV requires BF16 Q")
        if cutlass.const_expr(mK.element_type is not BFloat16 or mV.element_type is not BFloat16):
            raise TypeError("SM103 hd512 SplitKV requires BF16 K/V")
        if cutlass.const_expr(
            mOPartial.element_type is not Float32 or mLSEPartial.element_type is not Float32
        ):
            raise TypeError("SM103 hd512 SplitKV requires FP32 O/LSE partials")
        if cutlass.const_expr(
            mPageTable.element_type is not Int32
            or mSeqUsedK.element_type is not Int32
            or mNumSplitsDynamic.element_type is not Int32
        ):
            raise TypeError("page table, seqused_k, and split metadata must be int32")

        if cutlass.const_expr(
            h_q != 32 or h_k != 4 or d != 512 or d_k != 512 or mV.shape != mK.shape
        ):
            raise ValueError("SM103 hd512 SplitKV requires Q32/KV4/D512 and matching K/V")
        if cutlass.const_expr(page_size != self.page_size):
            raise ValueError("K/V page size does not match the compiled specialization")
        if cutlass.const_expr(
            total_q != batch
            or mSeqUsedK.shape[0] != batch
            or mNumSplitsDynamic.shape[0] != batch
        ):
            raise ValueError("Q1 decode requires exactly one packed query per batch item")
        if cutlass.const_expr(
            mOPartial.shape != (kv_splits, total_q, h_q, d)
            or mLSEPartial.shape != (kv_splits, total_q, h_q)
        ):
            raise ValueError("partial tensors do not match Q and split dimensions")
        if cutlass.const_expr(kv_splits <= 1 or kv_splits > 256):
            raise ValueError("SM103 hd512 SplitKV requires 2 to 256 splits")

        # The pointer-derived descriptors require contiguous tensors. FA4's
        # Python boundary validates that contract before converting tensors to
        # dynamic-stride CuTe layouts; only the statically-known innermost
        # stride remains checkable here.
        if cutlass.const_expr(
            mQ.stride[-1] != 1
            or mK.stride[-1] != 1
            or mV.stride[-1] != 1
            or mOPartial.stride[-1] != 1
            or mLSEPartial.stride[-2] != 1
            or mPageTable.stride[-1] != 1
            or mSeqUsedK.stride[-1] != 1
            or mNumSplitsDynamic.stride[-1] != 1
        ):
            raise ValueError("SM103 hd512 SplitKV requires contiguous inner dimensions")
        if cutlass.const_expr(mK.stride != mV.stride):
            raise ValueError("SM103 hd512 SplitKV requires matching K/V strides")

        problem_shape = (
            batch,
            h_q,
            h_k,
            page_table_width * page_size,
            d,
        )
        self._launch_from_pointers(
            problem_shape,
            kv_splits,
            mQ.iterator,
            mK.iterator,
            mV.iterator,
            mOPartial.iterator,
            mLSEPartial.iterator,
            softmax_scale,
            Float32(1.0),
            stream,
            mPageTable.iterator,
            mSeqUsedK.iterator,
            mNumSplitsDynamic.iterator,
            page_table_width,
            num_pages,
            mK.stride[0],
            mK.stride[2]
            if cutlass.const_expr(self.kv_layout == "NHD")
            else mK.stride[1],
            mK.stride[1]
            if cutlass.const_expr(self.kv_layout == "NHD")
            else mK.stride[2],
        )

    @cute.jit
    def _launch_from_pointers(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Int32, Int32],  # b, h_q, h_k, s_k, d
        kv_splits: Int32,  # threadblocks per sequence
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_partial_iter: cute.Pointer,  # normalized partial O per kv split
        m_partial_iter: cute.Pointer,  # partial LSE per kv split
        scale_qs: Float32,
        scale_o: Float32,
        stream: cuda.CUstream,
        page_table_iter: Optional[cute.Pointer] = None,
        seqused_k_iter: Optional[cute.Pointer] = None,
        num_splits_dynamic_iter: Optional[cute.Pointer] = None,
        page_table_width: Int32 = 0,
        num_pages: Int32 = 0,
        kv_page_stride: Int32 = 0,
        kv_head_stride: Int32 = 0,
        kv_token_stride: Int32 = 0,
    ):
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_iter.dtype
        acc_dtype = o_partial_iter.dtype
        assert acc_dtype is Float32

        # Block tile sets the granularity at which threadblocks consume work
        blk_tile_s = 128
        blk_tile_h = self.grouped_head_tile
        blk_tile_d = self.headdim
        blk_tile_shd = (blk_tile_s, blk_tile_h, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are issued
        mma_tile_m = 128
        mma_tile_n = self.grouped_head_tile
        # Native BF16 direct-SMEM UMMA K uses a 64-element reduction tile.
        mma_tile_k = 128 * 8 // mma_dtype.width
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)
        assert self.headdim % mma_tile_k == 0

        # GEMM1: (S_K, H_R, D, (H_K, B))
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            tcgen05.OperandMajorMode.K,  # K
            tcgen05.OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

        # GEMM2: (D, H_R, S_K, (H_K, B))
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(  #
            mma_dtype,
            tcgen05.OperandMajorMode.MN,  # V.T is contiguous in MMA M
            tcgen05.OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

        # Calculate Q stages
        self.q_stages = blk_tile_d // mma_tile_k

        # Compact TMEM: two score stages (2 * Q8 columns) and four dV128
        # output slices (4 * Q8 columns).  There are no K/V TMEM fragments.
        tmem_alloc_cols = mma_tile_n * self.sp_stages
        tmem_alloc_cols += mma_tile_n * self.o_stages * (blk_tile_d // mma_tile_m)
        self.tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))  # Tmem alloc must be PO2
        assert self.tmem_alloc_cols == 64

        ##############################
        # TMA creation
        ##############################
        b, h_q, h_k, s_k, d = problem_shape
        h_r = h_q // h_k

        q = cute.make_tensor(
            q_iter,
            cute.make_ordered_layout(shape=(h_r, d, (h_k, b)), order=(1, 0, (2, 3))),
        )

        if cutlass.const_expr(self.page_size in (64, 128)):
            assert (
                page_table_iter is not None
                and seqused_k_iter is not None
                and num_splits_dynamic_iter is not None
            )
            assert (
                page_table_iter.dtype is Int32
                and seqused_k_iter.dtype is Int32
                and num_splits_dynamic_iter.dtype is Int32
            )

            # Expose the page ID as the final descriptor coordinate so a
            # runtime int32 page-table entry can select the source page without
            # rebuilding the TMA descriptor. K is token-major for QK; V is
            # transposed to d-major for PV. NHD is FA4's public cache layout;
            # HND is vLLM's FlashInfer-compatible physical cache layout.
            # Use the source tensor's concrete compile-time strides. vLLM's
            # packed HND cache exposes zero-copy K/V views whose token and
            # page strides include the interleaved K/V storage dimension.
            physical_page_stride = kv_page_stride
            head_stride = kv_head_stride
            token_stride = kv_token_stride
            k = cute.make_tensor(
                k_iter,
                cute.make_layout(
                    (self.page_size, d, (h_k, num_pages)),
                    stride=(token_stride, 1, (head_stride, physical_page_stride)),
                ),
            )
            v = cute.make_tensor(
                v_iter,
                cute.make_layout(
                    (d, self.page_size, (h_k, num_pages)),
                    stride=(1, token_stride, (head_stride, physical_page_stride)),
                ),
            )
            page_table = cute.make_tensor(
                page_table_iter,
                cute.make_layout((b, page_table_width), stride=(page_table_width, 1)),
            )
            seqused_k = cute.make_tensor(seqused_k_iter, cute.make_layout((b,), stride=(1,)))
            num_splits_dynamic = cute.make_tensor(
                num_splits_dynamic_iter, cute.make_layout((b,), stride=(1,))
            )
        else:
            assert (
                page_table_iter is None
                and seqused_k_iter is None
                and num_splits_dynamic_iter is None
            )
            k = cute.make_tensor(
                k_iter,
                cute.make_ordered_layout(shape=(s_k, d, (h_k, b)), order=(1, 0, (2, 3))),
            )
            v = cute.make_tensor(
                v_iter,
                cute.make_ordered_layout(shape=(d, s_k, (h_k, b)), order=(0, 1, (2, 3))),
            )
            page_table = None
            seqused_k = None
            num_splits_dynamic = None
        assert k_iter.dtype is q_iter.dtype
        assert v_iter.dtype is k_iter.dtype

        # Every decode CTA writes an FP32 partial output and row statistics;
        # the production combine kernel reduces the split dimension.
        o_partial = cute.make_tensor(
            o_partial_iter,
            cute.make_ordered_layout(shape=(d, h_r, (h_k, b), kv_splits), order=(0, 1, (2, 3), 4)),
        )

        # Only the row extent is consumed from this view.  Reuse the LSE
        # partial allocation instead of carrying dead
        # final-output buffers through the product ABI.
        m = cute.make_tensor(
            m_partial_iter,
            cute.make_ordered_layout(
                shape=(h_r, (h_k, b)),
                order=(0, (1, 2)),
            ),
        )

        m_partial = cute.make_tensor(
            m_partial_iter,
            cute.make_ordered_layout(
                shape=(h_r, (h_k, b), kv_splits),
                # Production combine consumes logical [split, batch, head]
                # through storage physically ordered [split, head, batch].
                order=(1, (2, 0), 3),
            ),
        )
        assert m_partial_iter.dtype is acc_dtype

        # (MMA, MMA_M/N, MMA_K, Stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, q_iter.dtype, self.q_stages
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk, k_iter.dtype, self.k_stages
        )
        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp, mma_tile_mnk, v_iter.dtype, self.v_stages, is_k_major=False
        )  # V is always headdim-major (GEMM2 M-major) in gmem+smem

        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_partial_iter.dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (blk_tile_d, blk_tile_h), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            q,
            cute.select(smem_layout_q, mma_modes),
            mma_tile_mnk,
            tiled_mma_kq,
        )
        if cutlass.const_expr(self.page_size == 64):
            # A logical N128 tile spans two physical page64 entries.  Build
            # CTA-local generic descriptors for the independently addressable
            # K 64x64 halves and V d128x64 slices.  Both K halves later share
            # one existing 16-KiB PipelineTmaUmma completion barrier.
            tma_load_op_cta1 = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            # Explicit dense (64,64) boxes match the MMA read mapping over the
            # whole domain. The +4096-element half offset and +8192-element
            # stage offset commute with the S<3,4,3> swizzle. Direct layouts
            # also avoid a CuTeDSL lowering failure on composition-derived
            # descriptor layouts.
            # V's box is one d64 half-slab; each stage takes two V copies.
            k_tma_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout((64, mma_tile_k), stride=(mma_tile_k, 1)),
            )
            v_tma_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout((64, 64), stride=(1, 64)),
            )
            tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op_cta1,
                k,
                k_tma_smem_layout,
                (64, mma_tile_k),
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op_cta1,
                v,
                v_tma_smem_layout,
                (64, 64),
            )
        else:
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                k,
                cute.select(smem_layout_k, mma_modes),
                mma_tile_mnk,
                tiled_mma_kq,
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                v,
                cute.select(smem_layout_v, mma_modes),
                mma_tile_mnk,
                tiled_mma_vp,
            )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o_partial,
            cute.select(smem_layout_o, mode=[0, 1]),
            mma_tile_mnk[:2],
        )

        ##############################
        # Decode Kernel launch
        ##############################
        scale_qs_log2_e = scale_qs * log2_e

        n_tiles = cute.ceil_div(h_r, blk_tile_h)
        l_tiles = b * h_k
        grid = (kv_splits, n_tiles, l_tiles)

        self.decode(
            blk_tile_shd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            q_iter.dtype,
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            k_iter.dtype,
            smem_layout_k,
            tma_atom_k,
            tma_tensor_k,
            v_iter.dtype,
            smem_layout_v,
            tma_atom_v,
            tma_tensor_v,
            o_partial_iter.dtype,
            smem_layout_o,
            tma_atom_o,
            tma_tensor_o,
            m,
            m_partial,
            scale_qs,
            scale_qs_log2_e,
            scale_o,
            page_table,
            seqused_k,
            num_splits_dynamic,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        # Stop after producing the production split-output contract. The caller
        # launches FlashAttentionForwardCombine on the same stream.

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        # Q
        q_dtype: Type[cutlass.Numeric],
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # K
        k_dtype: Type[cutlass.Numeric],
        smem_layout_k: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        # V
        v_dtype: Type[cutlass.Numeric],
        smem_layout_v: cute.ComposedLayout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        # O
        o_dtype: Type[cutlass.Numeric],
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: cute.CopyAtom,
        mO: cute.Tensor,
        # Rest
        mM: cute.Tensor,
        mMPartial: cute.Tensor,
        scale_qs: Float32,
        scale_qs_log2_e: Float32,
        scale_o: Float32,
        mPageTable: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mNumSplitsDynamic: Optional[cute.Tensor],
    ):
        # Read special registers
        kv_splits, tiles_hr, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hr, coord_hb = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 0

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        mma_dtype = q_dtype
        acc_dtype = Float32
        assert k_dtype is mma_dtype and v_dtype is mma_dtype

        # Shapes for MMA tile indexing (Read TMA partition for example)
        blk_tile_s, blk_tile_h, blk_tile_d = blk_tile_shd
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_dm, tiles_sk = cute.ceil_div((blk_tile_d, blk_tile_s), (mma_tile_m, mma_tile_k))
        tiles_dk, tiles_sm = cute.ceil_div((blk_tile_d, blk_tile_s), (mma_tile_k, mma_tile_m))
        h_k = mQ.shape[2][0]
        batch_coord = coord_hb // h_k
        head_kv_coord = coord_hb % h_k
        seqlen_k = mK.shape[0]
        if cutlass.const_expr(mSeqUsedK is not None):
            seqlen_k = mSeqUsedK[batch_coord]
        tiles_s = cute.ceil_div(seqlen_k, blk_tile_s)
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        prefetch_iters = self.sp_stages - 1
        if iters_s < prefetch_iters:
            prefetch_iters = iters_s
        assert tiles_sm == 1

        # Runtime checks
        exit_early = kv_split_idx >= tiles_s
        lane_store_max = mma_tile_n == warp_threads or lane_idx < mma_tile_n

        # The following combine launch needs the number of splits that wrote
        # valid partials for each request. Produce it in the decode kernel so
        # CUDA graphs do not need a host fill or an extra metadata kernel.
        if (
            cutlass.const_expr(mNumSplitsDynamic is not None)
            and kv_split_idx == 0
            and coord_hr == 0
            and head_kv_coord == 0
            and tidx == 0
        ):
            valid_splits = kv_splits
            if tiles_s < valid_splits:
                valid_splits = tiles_s
            mNumSplitsDynamic[batch_coord] = valid_splits

        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        # Allocate Mbarriers
        q_pipeline_ptr = smem.allocate_array(Int64, self.q_stages * 2)
        k_pipeline_ptr = smem.allocate_array(Int64, self.k_stages * 2)
        v_pipeline_ptr = smem.allocate_array(Int64, self.v_stages * 2)
        s_pipeline_ptr = smem.allocate_array(Int64, self.sp_stages * 2)
        p_pipeline_ptr = smem.allocate_array(Int64, self.sp_stages * 2)
        o_pipeline_ptr = smem.allocate_array(Int64, self.o_stages * 2)
        page_ids_smem = None
        if cutlass.const_expr(self.page_size == 64):
            # Two N128 slots are sufficient for the one-tile K->V prefetch
            # distance.  Each slot retains the two physical page64 IDs so V
            # reuses K's mapping instead of reading the page table again.
            page_ids_smem = smem.allocate_array(Int32, 4)

        # Declare named barriers
        softmax_nbar_id = 1
        mma_kq_nbar_id = 2
        mma_vp_nbar_id = 3

        # Alias thread cooperatives
        elect_one_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        warpgroup_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread, warpgroup_threads)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        softmax_group = warpgroup_cooperative

        # Initialize pipelines
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stages,
            producer_group=tma_group,
            consumer_group=softmax_group,  # Reuse Q consumer mbarriers to sync O store
            tx_count=cute.size_in_bytes(q_dtype, cute.select(smem_layout_q, mma_modes)),
            barrier_storage=q_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=cute.size_in_bytes(k_dtype, cute.select(smem_layout_k, mma_modes)),
            barrier_storage=k_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        v_producer, v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=cute.size_in_bytes(v_dtype, cute.select(smem_layout_v, mma_modes)),
            barrier_storage=v_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.sp_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.sp_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.o_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()

        ##############################
        # MMA Partition + Allocate
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # M - colmax
        sM_layout = cute.make_layout(shape=(mma_tile_m, mma_tile_n), stride=(0, 1))
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        tCsM = thrblk_mma_kq.partition_C(sM)

        # L - colsum
        sL_layout = cute.make_layout(
            shape=(mma_tile_m, mma_tile_n, warpgroup_warps), stride=(0, 1, mma_tile_n)
        )
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        tCsL = thrblk_mma_kq.partition_C(sL)

        # Q
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)

        # K
        tAsK = smem.allocate_tensor(
            k_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, k_stages)

        # V has a separate backing allocation.  Do not alias it with K: the
        # independent pipelines are required by this native-BF16 path.
        tAsV = smem.allocate_tensor(
            v_dtype, smem_layout_v.outer, stensor_align, smem_layout_v.inner
        )  # (MMA, #MMA_M, #MMA_K, v_stages)

        # S
        tCtS_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, self.sp_stages))
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)  # (MMA_MN, #MMA_M=1, #MMA_N=1, sp_stages)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages)
        mma_tile_nm = (None, mma_tile_n, mma_tile_m)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, mma_tile_nm, mma_dtype, self.sp_stages
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        # Tile for NK B tile iteration
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_K/MMA_K, #TILES_SK=MMA_TILE_M/MMA_TILE_K, sp_stages)
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B((mma_tile_n, mma_tile_k))
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, sp_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O
        sO_iterator = cute.recast_ptr(
            tBsQ.iterator, smem_layout_o.inner, dtype=o_dtype
        )  # Reuse QKV smem for O TMA store
        sO_mma = cute.make_tensor(
            sO_iterator, smem_layout_o.outer
        )  # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
        tCsO = thrblk_mma_vp.partition_C(sO_mma)  # (MMA, #MMA_M, #MMA_N, #TILE_DM, #TILE_HN)
        tCtO = thrblk_mma_vp.make_fragment_C(tCsO.shape)

        # Tmem tensor allocation
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        assert tmem_offset <= self.tmem_alloc_cols

        ##############################
        # Exit early
        ##############################
        if exit_early:
            noop = None  # early return not supported

        ##############################
        # TMA KV Dispatch
        ##############################
        elif warp_idx == self.tma_kv_warp_id:
            # Apply block tiler and slice
            if cutlass.const_expr(mPageTable is None):
                gK = cute.local_tile(
                    mK, tiler=(blk_tile_s, blk_tile_d), coord=(None, 0, coord_hb)
                )  # (TILE_S, TILE_D, #TILE_S)
                gV = cute.local_tile(
                    mV, tiler=(blk_tile_d, blk_tile_s), coord=(0, None, coord_hb)
                )  # (TILE_D, TILE_S, #TILE_S)
            elif cutlass.const_expr(self.page_size == 128):
                # Select the logical KV head but retain the physical-page mode.
                # A page-table value becomes the final TMA source coordinate.
                mK_head = mK[None, None, (head_kv_coord, None)]
                mV_head = mV[None, None, (head_kv_coord, None)]
                gK = cute.local_tile(
                    mK_head, tiler=(blk_tile_s, blk_tile_d), coord=(0, 0, None)
                )  # (TILE_S, TILE_D, #PHYSICAL_PAGE)
                gV = cute.local_tile(
                    mV_head, tiler=(blk_tile_d, blk_tile_s), coord=(0, 0, None)
                )  # (TILE_D, TILE_S, #PHYSICAL_PAGE)
            else:
                # Page64 uses dense (64,64) boxes matching the host-side TMA
                # atoms. Halves and stages are pure pointer offsets. No
                # composition-derived layout is passed to descriptor lowering.
                mK_page64 = mK[None, None, (head_kv_coord, None)]
                mV_page64 = mV[None, None, (head_kv_coord, None)]

                # Device convention (same as the dense path's tAsK/tAsV): the
                # smem iterator already carries the S<3,4,3> swizzle from
                # allocate_tensor, so the box tensors use PLAIN layouts -- the
                # pre-swizzle offsets proven equal to the MMA mapping.  The
                # host-side TMA atoms keep the swizzled layout (no pointer
                # exists there), mirroring make_tiled_tma_atom_A usage.
                sK_box = cute.make_tensor(
                    tAsK.iterator,
                    cute.make_layout((64, mma_tile_k), stride=(mma_tile_k, 1)),
                )
                sV_box = cute.make_tensor(
                    tAsV.iterator,
                    cute.make_layout((64, 64), stride=(1, 64)),
                )

                gK_page64 = cute.local_tile(
                    mK_page64,
                    (64, 64),
                    (0, None, None),
                )
                tKsK_page64_template, tKgK_page64 = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_box, 0, 2),
                    cute.group_modes(gK_page64, 0, 2),
                )

                gV_page64 = cute.local_tile(
                    mV_page64,
                    (64, 64),
                    (None, 0, None),
                )
                tVsV_page64_template, tVgV_page64 = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV_box, 0, 2),
                    cute.group_modes(gV_page64, 0, 2),
                )

                # Static element offsets: stage stride and page-half stride.
                k_stage_elems = cute.size(cute.select(smem_layout_k, mma_modes))
                k_half_elems = k_stage_elems // 2
                v_stage_elems = cute.size(cute.select(smem_layout_v, mma_modes))
                v_half_elems = v_stage_elems // 2

            # #TILE_SM=TILE_S/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_DK=TILE_D/MMA_TILE_K
            # #TILE_DM=TILE_D/MMA_TILE_M, #TILE_HN=TILE_H/MMA_TILE_N, #TILE_SK=TILE_S/MMA_TILE_K
            #
            # Example with TILE_S=MMA_TILE_M=128, TILE_H=MMA_TILE_N=8, MMA_TILE_K=64, TILE_D=512
            # BMM1: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4, #TILE_SM=1, #TILE_HN=1, #TILE_DK=8, #TILE_S=S/128
            # BMM2: MMA=128x8x16, #MMA_M=1, #MMA_N=1, #MMA_K=4, #TILE_DM=4, #TILE_HN=1, #TILE_SK=2, #TILE_S=S/128

            if cutlass.const_expr(self.page_size != 64):
                # Apply MMA tiler and MMA partition.  Dense and page128 retain
                # the proven MMA-aware descriptor path unchanged.
                gK_mma = cute.flat_divide(
                    gK, (mma_tile_m, mma_tile_k)
                )  # (MMA_TILE_M, MMA_TILE_K, #TILE_SM, #TILE_DK, #TILE_S)
                gV_mma = cute.flat_divide(
                    gV, (mma_tile_m, mma_tile_k)
                )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DM, #TILE_SK, #TILE_S)
                tAgK = thrblk_mma_kq.partition_A(
                    gK_mma
                )  # (MMA, #MMA_M, #MMA_K, #TILE_SM, #TILE_DK, #TILE_S)
                tAgV = thrblk_mma_vp.partition_A(
                    gV_mma
                )  # (MMA, #MMA_M, #MMA_K, #TILE_DM, #TILE_SK, #TILE_S)

                # (MMA, #MMA_M, #MMA_K, Rest...) -> (TMA, Rest...)
                tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsK, 0, 3),
                    gmem_tensor=cute.group_modes(tAgK, 0, 3),
                )

                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsV, 0, 3),
                    gmem_tensor=cute.group_modes(tAgV, 0, 3),
                )

            #
            # Sequence loop
            #
            prefetch_tiles = prefetch_iters * kv_splits
            for s in cutlass.range(kv_split_idx, prefetch_tiles + tiles_s, kv_splits):
                # Load K
                if s < tiles_s:
                    if cutlass.const_expr(self.page_size != 64):
                        source_coord = s
                        if cutlass.const_expr(mPageTable is not None):
                            source_coord = mPageTable[batch_coord, s]
                        tGSgK_s = tGSgK[None, None, None, source_coord]
                    else:
                        # The page mapping is invariant across all eight d64
                        # K and eight V descriptor issues for this logical N128
                        # tile.  One elected lane resolves the pair, then a
                        # two-slot SMEM ring retains it across the one-tile
                        # prefetch distance until V consumes it.
                        local_iter = (s - kv_split_idx) // kv_splits
                        page_cache_offset = (local_iter % 2) * 2
                        with cute.arch.elect_one():
                            semantic_pages = cute.ceil_div(seqlen_k, 64)
                            lower_slot = s * 2
                            upper_slot = lower_slot + 1
                            lower_page_idx = Int32(mK_page64.shape[2])
                            upper_page_idx = Int32(mK_page64.shape[2])
                            if lower_slot < semantic_pages:
                                if lower_slot < mPageTable.shape[1]:
                                    lower_page_idx = mPageTable[batch_coord, lower_slot]
                            if upper_slot < semantic_pages:
                                if upper_slot < mPageTable.shape[1]:
                                    upper_page_idx = mPageTable[batch_coord, upper_slot]
                            page_ids_smem[page_cache_offset] = lower_page_idx
                            page_ids_smem[page_cache_offset + 1] = upper_page_idx
                        cute.arch.sync_warp()
                        lower_page_idx = page_ids_smem[page_cache_offset]
                        upper_page_idx = page_ids_smem[page_cache_offset + 1]
                    for dk in cutlass.range_constexpr(tiles_dk):
                        k_handle = k_producer.acquire_and_advance()
                        if cutlass.const_expr(self.page_size == 64):
                            # Dense halves are selected by pointer arithmetic;
                            # no per-copy layout algebra.
                            sK_stage_iter = (
                                tKsK_page64_template.iterator + k_handle.index * k_stage_elems
                            )
                            for page_half in cutlass.range_constexpr(2):
                                tKsK_page64 = cute.make_tensor(
                                    sK_stage_iter + page_half * k_half_elems,
                                    tKsK_page64_template.layout,
                                )
                                page_idx = lower_page_idx
                                if cutlass.const_expr(page_half == 1):
                                    page_idx = upper_page_idx
                                cute.copy(
                                    tma_atom_k,
                                    tKgK_page64[None, dk, page_idx],
                                    tKsK_page64,
                                    tma_bar_ptr=k_handle.barrier,
                                )
                        else:
                            cute.copy(
                                tma_atom_k,
                                tGSgK_s[None, 0, dk],
                                tGSsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )

                # Load V
                if s >= prefetch_tiles:
                    logical_tile = s - prefetch_tiles
                    if cutlass.const_expr(self.page_size != 64):
                        source_coord = logical_tile
                        if cutlass.const_expr(mPageTable is not None):
                            source_coord = mPageTable[batch_coord, source_coord]
                        tGSgV_s = tGSgV[None, None, None, source_coord]
                    else:
                        # The K prefetch for this logical tile resolved and
                        # retained both IDs.  Reuse them for all eight V
                        # descriptor issues; do not touch the page table here.
                        local_iter = (logical_tile - kv_split_idx) // kv_splits
                        page_cache_offset = (local_iter % 2) * 2
                        lower_page_idx = page_ids_smem[page_cache_offset]
                        upper_page_idx = page_ids_smem[page_cache_offset + 1]
                    for sk in cutlass.range_constexpr(tiles_sk):
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = v_producer.acquire_and_advance()
                            if cutlass.const_expr(self.page_size == 64):
                                # Two dense d64 half-slab copies per V
                                # stage; both arrive on the same stage barrier
                                # so tx accounting is unchanged (the K path
                                # already runs two arrivals per barrier).
                                page_idx = lower_page_idx
                                if cutlass.const_expr(sk == 1):
                                    page_idx = upper_page_idx
                                sV_stage_iter = (
                                    tVsV_page64_template.iterator + v_handle.index * v_stage_elems
                                )
                                for v_dh in cutlass.range_constexpr(2):
                                    tVsV_page64 = cute.make_tensor(
                                        sV_stage_iter + v_dh * v_half_elems,
                                        tVsV_page64_template.layout,
                                    )
                                    cute.copy(
                                        tma_atom_v,
                                        tVgV_page64[None, dm * 2 + v_dh, page_idx],
                                        tVsV_page64,
                                        tma_bar_ptr=v_handle.barrier,
                                    )
                            else:
                                cute.copy(
                                    tma_atom_v,
                                    tGSgV_s[None, dm, sk],
                                    tGSsV[None, v_handle.index],
                                    tma_bar_ptr=v_handle.barrier,
                                )

        ##############################
        # TMA QO Dispatch
        ##############################
        elif warp_idx == self.tma_qo_warp_id:
            # Apply block tiler and slice
            gQ = cute.local_tile(
                mQ, tiler=(blk_tile_h, blk_tile_d), coord=(coord_hr, 0, coord_hb)
            )  # (TILE_H, TILE_D)
            gO = cute.local_tile(
                mO,
                tiler=(blk_tile_d, blk_tile_h),
                coord=(0, coord_hr, coord_hb, kv_split_idx),
            )  # (TILE_D, TILE_H)
            # Apply MMA tiler and MMA partition
            gQ_mma = cute.flat_divide(
                gQ, (mma_tile_n, mma_tile_k)
            )  # (MMA_TILE_N, MMA_TILE_K, #TILE_HN, #TILE_DK)
            gO_mma = cute.flat_divide(
                gO, (mma_tile_m, mma_tile_n)
            )  # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)  # (MMA, #MMA_N, #MMA_K, #TILE_HN, #TILE_DK)

            # TMA partition
            tGSsQ, tGSgQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )

            tSGsO, tSGgO = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sO_mma, 0, 2),
                gmem_tensor=cute.group_modes(gO_mma, 0, 2),
            )

            # Load Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tGSgQ[None, 0, dk],  # stages_q == tiles_dk by construction
                    tGSsQ[None, dk],
                    tma_bar_ptr=q_handle.barrier,
                )

            # Store O
            for dm in cutlass.range_constexpr(tiles_dm):
                q_producer.acquire_and_advance()  # Reuse Q load barriers to sync O store
                cute.copy(tma_atom_o, tSGsO[None, dm, 0], tSGgO[None, dm, 0])
            # The output SMEM aliases Q storage.  Complete all S2G transactions
            # before the CTA exits or any aliased storage can be reclaimed.
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        ##############################
        # MMA KQ Dispatch
        ##############################
        elif warp_idx == self.mma_kq_warp_id:
            # Setup mma descriptors
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            # Wait for Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_consumer.wait_and_advance()

            # Sequence loop
            s_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # BMM1
                k_token = k_consumer.try_wait()
                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                s_handle = s_producer.acquire_and_advance(s_token)
                for dk in cutlass.range_constexpr(tiles_dk):
                    is_last_iter = dk == tiles_dk - 1
                    k_handle = k_consumer.wait_and_advance(k_token)
                    # Signal BMM2 to start
                    if is_last_iter:
                        cute.arch.barrier_arrive(barrier_id=mma_kq_nbar_id, number_of_threads=64)
                    num_kphases = cute.size(tAsK_desc, mode=[2])
                    for mma_k in cutlass.range(num_kphases, unroll_full=True):
                        cute.gemm(
                            tiled_mma_kq,
                            tCtS[mma_dice + (s_handle.index,)],
                            tAsK_desc[None, None, mma_k, k_handle.index],
                            tBsQ_desc[None, None, mma_k, dk],
                            tCtS[mma_dice + (s_handle.index,)],
                        )
                        if dk == 0 and mma_k == 0:
                            tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                    k_handle.release()
                    if not is_last_iter:
                        k_token = k_consumer.try_wait()
                s_handle.commit()

                # Advance and wait for BMM 2
                if s > 0:
                    cute.arch.barrier(barrier_id=mma_vp_nbar_id, number_of_threads=64)
                    s_token = s_producer.try_acquire()

        ##############################
        # MMA VP Dispatch
        ##############################
        elif warp_idx == self.mma_vp_warp_id:
            # Setup mma descriptors
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tAsV_desc = thrblk_mma_vp.make_fragment_A(tAsV)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)

            # Wait for BMM1
            cute.arch.barrier(barrier_id=mma_kq_nbar_id, number_of_threads=64)

            # Sequence loop
            p_token = False
            o_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # Advance and wait for BMM1
                if s < iters_s - 1:
                    cute.arch.barrier(barrier_id=mma_kq_nbar_id, number_of_threads=64)
                    p_token = p_consumer.try_wait()

                # BMM2
                v_token = v_consumer.try_wait()
                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        v_handle = v_consumer.wait_and_advance(v_token)
                        # Signal BMM1 to start
                        if is_last_iter:
                            cute.arch.barrier_arrive(
                                barrier_id=mma_vp_nbar_id, number_of_threads=64
                            )
                        num_kphases = cute.size(tAsV_desc, mode=[2])
                        for mma_k in cutlass.range(num_kphases, unroll_full=True):
                            cute.gemm(
                                tiled_mma_vp,
                                tCtO[mma_dice + (dm, 0)],
                                tAsV_desc[None, None, mma_k, v_handle.index],
                                tBsP_desc[None, None, mma_k, sk, p_handle.index],
                                tCtO[mma_dice + (dm, 0)],
                            )
                        v_handle.release()
                        if not is_last_iter:
                            v_token = v_consumer.try_wait()
                p_handle.release()
                o_handle.commit()
                o_token = o_producer.try_acquire()

            # Wait for signal to dealloc tmem, then dealloc
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

        ##############################
        # Softmax + Correction Dispatch
        ##############################
        elif warpgroup_idx == self.softmax_warpgroup_id:
            # Construct tiled copies
            tmem_op_width = 32
            tmem_op_repeat = tcgen05.Repetition(mma_tile_n * acc_dtype.width // tmem_op_width)
            tmem_load_atom_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tmem_op_repeat), acc_dtype)
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS[mma_dice + (0,)])
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)

            tmem_store_atom_o = cute.make_copy_atom(tcgen05.St32x32bOp(tmem_op_repeat), acc_dtype)
            tmem_store_o = tcgen05.make_tmem_copy(tmem_store_atom_o, tCtO[mma_dice + (0, 0)])
            thr_store_o = tmem_store_o.get_slice(warpgroup_tidx)

            # Partition S and P
            tStS = thr_load_s.partition_S(tCtS)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)
            tSsP = thr_load_s.partition_D(tCsP)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)

            # Residual-K coordinates. Scores are transposed S = K @ Q.T,
            # so C's M coordinate is the local KV-row coordinate.  Partition
            # the identity tensor through the exact copy used for tSrS to keep
            # the coordinate and score register fragments elementwise aligned.
            tScS = thrblk_mma_kq.partition_C(cute.make_identity_tensor((mma_tile_m, mma_tile_n)))
            tScS_t2r = thr_load_s.partition_D(tScS)
            assert cute.size(tScS_t2r) == cute.size(tSsP.shape[:-1])

            # Partition O
            tStO = thr_load_s.partition_S(
                tCtO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSsO = thr_load_s.partition_D(
                tCsO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSrO = cute.make_rmem_tensor(tSsO.shape, acc_dtype)

            # Partition colmax and initialize in RF
            tSsM = thr_load_s.partition_D(tCsM)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
            tSrM_prev = cute.make_rmem_tensor_like(tSsM)
            tSrM_prev.fill(-Float32.inf)

            # Partition colsum and initialize in RF
            # Each thread maintains a local colsum in RF, smem reduction happens after loop
            tSsL = thr_load_s.partition_D(tCsL)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, WARPS)
            tSrL = cute.make_rmem_tensor_like(tSsL[cpy_dice + (0,)])
            tSrL.fill(Float32(0))

            assert warp_threads >= cute.size(tSsM)

            # get gmem colmax + colsum to store to
            inbound_hr = coord_hr * blk_tile_h + lane_idx < mM.shape[0]
            gM_partial = cute.local_tile(
                mMPartial,
                tiler=(mma_tile_n, 1),
                coord=(coord_hr, coord_hb, kv_split_idx),
            )
            # Initialize O
            tSrO.fill(Float32(0))
            cute.copy(thr_store_o, tSrO, tStO)

            # Initialize colsum and colmax in smem and wait
            if warpgroup_widx == 0 and lane_store_max:
                tSsM[lane_idx] = -Float32.inf
            if warpgroup_widx == 1 and lane_store_max:
                tSsL[lane_idx] = Float32(0)
            cute.arch.barrier(barrier_id=softmax_nbar_id, number_of_threads=warpgroup_threads)

            #
            # Sequence loop
            #
            for s in cutlass.range(iters_s):
                # Load S from tmem
                s_handle = s_consumer.wait_and_advance()
                tSrS = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                cute.copy(tmem_load_s, tStS[cpy_dice + (s_handle.index,)], tSrS)
                cute.arch.fence_view_async_tmem_load()
                s_handle.release()

                # TMA boundary fill is a valid memory-access mechanism, not a
                # softmax mask.  Exclude physical rows beyond the true K extent
                # before either max or exp sees them.  Full K128 tiles skip this
                # uniform residual branch entirely.
                tile_s_idx = kv_split_idx + s * kv_splits
                if tile_s_idx == tiles_s - 1 and seqlen_k % blk_tile_s != 0:
                    valid_rows = seqlen_k - tile_s_idx * blk_tile_s
                    for i in cutlass.range_constexpr(cute.size(tSrS)):
                        tSrS[i] = -Float32.inf if tScS_t2r[i][0] >= valid_rows else tSrS[i]

                # Reduce colmax in warp RF
                tSrM = cute.make_rmem_tensor_like(tSsM)
                tSrM_lane = Float32(0)  # Avoid dynamic register indexing
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    tSrM[i] = cute.arch.warp_redux_sync(tSrS[i], kind="fmax", nan=True)
                    if i == lane_idx:
                        tSrM_lane = tSrM[i]

                # Reduce colmax in smem
                if lane_store_max:
                    self.smem_fmax(tSsM.iterator + tSsM.layout(lane_idx), tSrM_lane)

                # Wait for colmax then load
                cute.arch.barrier(barrier_id=softmax_nbar_id, number_of_threads=warpgroup_threads)
                cute.autovec_copy(tSsM, tSrM)

                # Compute online softmax
                tSrP = cute.make_rmem_tensor(tSsP.shape[:-1], mma_dtype)
                if cutlass.const_expr(use_tensor_ssa_math):
                    tSrP_f32 = exp2(scale_qs_log2_e * (tSrS.load() - tSrM.load()))
                    tSrP.store(tSrP_f32.to(mma_dtype))  # convert
                else:
                    tSrP_f32 = cute.make_rmem_tensor(tSrS.shape, acc_dtype)
                    for i in cutlass.range_constexpr(0, cute.size(tSrS), 2):
                        p_f32x2 = fadd2((tSrS[i], tSrS[i + 1]), (-tSrM[i], -tSrM[i + 1]))
                        p_f32x2 = fmul2(p_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                        tSrP_f32[i] = exp2(p_f32x2[0])
                        tSrP_f32[i + 1] = exp2(p_f32x2[1])
                    tSrP.store(tSrP_f32.load().to(mma_dtype))

                # Store P to smem
                p_handle = p_producer.acquire_and_advance()
                cute.autovec_copy(tSrP, tSsP[cpy_dice + (p_handle.index,)])
                cute.arch.fence_view_async_shared()
                p_handle.commit()

                # Compute correction and correct colsum
                if cutlass.const_expr(use_tensor_ssa_math):
                    correction = exp2(scale_qs_log2_e * (tSrM_prev.load() - tSrM.load()))
                    tSrL.store(tSrL.load() * correction + tSrP_f32)
                else:
                    correction = cute.make_rmem_tensor_like(tSrM)
                    for i in cutlass.range_constexpr(0, cute.size(tSrM), 2):
                        c_f32x2 = fadd2((tSrM_prev[i], tSrM_prev[i + 1]), (-tSrM[i], -tSrM[i + 1]))
                        c_f32x2 = fmul2(c_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                        c_f32x2 = (exp2(c_f32x2[0]), exp2(c_f32x2[1]))
                        correction[i] = c_f32x2[0]
                        correction[i + 1] = c_f32x2[1]
                        l_f32x2 = ffma2(
                            c_f32x2,
                            (tSrL[i], tSrL[i + 1]),
                            (tSrP_f32[i], tSrP_f32[i + 1]),
                        )
                        tSrL[i] = l_f32x2[0]
                        tSrL[i + 1] = l_f32x2[1]

                # Correct O
                if s > 0:
                    # Wait for O
                    o_handle = o_consumer.wait_and_advance()

                    # Apply correction
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tSrO_dm = cute.make_rmem_tensor(tSsO[cpy_dice + (0, 0)].shape, acc_dtype)
                        cute.copy(thr_load_s, tStO[cpy_dice + (dm, 0)], tSrO_dm)

                        for i in cutlass.range_constexpr(0, cute.size(tSrO_dm), 2):
                            o_f32x2 = fmul2(
                                (tSrO_dm[i], tSrO_dm[i + 1]),
                                (correction[i], correction[i + 1]),
                            )
                            tSrO_dm[i] = o_f32x2[0]
                            tSrO_dm[i + 1] = o_f32x2[1]

                        cute.copy(thr_store_o, tSrO_dm, tStO[cpy_dice + (dm, 0)])

                    # Notify MMA
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                # Update colmax
                tSrM_prev.store(tSrM.load())

            #
            # Softmax Epilogue
            #

            # Reduce colsum in warp RF
            tSrL_lane = Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(tSrL)):
                tSrL[i] = cute.arch.warp_reduction_sum(tSrL[i])
                if i == lane_idx:
                    tSrL_lane = tSrL[i]

            # Store partial colsum in smem
            if lane_store_max:
                tSsL[cpy_dice + (warpgroup_widx,)][lane_idx] = tSrL_lane

            # Wait for colsum
            cute.arch.barrier(barrier_id=softmax_nbar_id, number_of_threads=warpgroup_threads)

            if warpgroup_widx == 0 and lane_store_max and inbound_hr:
                # Load colsum and colmax
                sL_lane_wg = sL[0, lane_idx, None]
                sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
                sM_lane = sM[0, lane_idx]

                # Publish the same normalized-partial/LSE contract consumed by
                # FA4's production combine.  Reuse the shared max vector to
                # broadcast scale_o / L to every output-fragment owner.
                sM_lane = sM_lane * scale_qs
                row_empty = sL_lane == 0.0 or sL_lane != sL_lane
                gM_partial[lane_idx] = (
                    -Float32.inf if row_empty else sM_lane + cute.math.log(sL_lane, fastmath=False)
                )
                sM[0, lane_idx] = 0.0 if row_empty else scale_o / sL_lane

            cute.arch.barrier(barrier_id=softmax_nbar_id, number_of_threads=warpgroup_threads)

            # Previous-max registers are dead after the sequence loop.  Reload
            # the per-row normalization scale into their exact fragment layout.
            cute.autovec_copy(tSsM, tSrM_prev)

            o_handle = o_consumer.wait_and_advance()
            cute.copy(thr_load_s, tStO, tSrO)
            cute.arch.fence_view_async_tmem_load()
            o_handle.release()  # Final release signals tmem dealloc

            # Store O to smem
            for dm in cutlass.range_constexpr(tiles_dm):
                tOrO_dm = tSrO[cpy_dice + (dm, 0)]
                tOsO_dm = tSsO[cpy_dice + (dm, 0)]

                for i in cutlass.range_constexpr(cute.size(tOrO_dm)):
                    tOrO_dm[i] = tOrO_dm[i] * tSrM_prev[i]

                cute.autovec_copy(tOrO_dm, tOsO_dm)
                cute.arch.fence_view_async_shared()

                # Reuse Q consumer barriers to notify O TMA store
                q_consumer.release()
                q_consumer.advance()

        return

    @staticmethod
    @cute.jit
    def smem_fmax(ptr: Pointer, val: Float32):
        # https://stackoverflow.com/a/72461459
        # Works with canonical NaN which warp_redux_sync(kind="fmax") should return
        llvm.inline_asm(
            None,
            [ptr.llvm_ptr, val.ir_value()],
            """{\n\t
                .reg .pred p;\n\t
                setp.lt.s32 p, $1, 0x0;
            @p  red.relaxed.shared::cta.min.u32 [$0], $1;\n\t
            @!p red.relaxed.shared::cta.max.s32 [$0], $1;\n\t
            }\n\t""",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


# Increment whenever host-side specialization assumptions or the tensor ABI
# changes. Every shape and stride below participates in CuTeDSL specialization.
_ADAPTER_ABI_VERSION = 2
_COMPILE_OPTIONS = "--opt-level 2"


@dataclass(frozen=True)
class Sm103Hd512SplitKVCompileKey:
    adapter_abi_version: int
    device_index: int
    device_capability: tuple[int, int]
    kv_layout: str
    page_size: int
    q_spec: tuple[object, ...]
    k_spec: tuple[object, ...]
    v_spec: tuple[object, ...]
    out_partial_spec: tuple[object, ...]
    lse_partial_spec: tuple[object, ...]
    seqused_k_spec: tuple[object, ...]
    page_table_spec: tuple[object, ...]
    valid_splits_spec: tuple[object, ...]
    compile_options: str


_compiled_decode_cache: dict[Sm103Hd512SplitKVCompileKey, object] = {}


def _torch_tensor_spec(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
    )


def _as_cute_tensor(
    tensor: torch.Tensor,
    element_type: Type,
    assumed_align: int,
) -> cute.Tensor:
    result = from_dlpack(tensor, assumed_align=assumed_align)
    result.element_type = element_type
    return result


def _validate_sm103_hd512_decode_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    valid_splits: torch.Tensor,
    *,
    kv_layout: str,
) -> int:
    if not q.is_cuda:
        raise ValueError("SM103 hd512 SplitKV requires CUDA tensors")
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (10, 3):
        raise ValueError(
            "SM103 hd512 SplitKV requires compute capability 10.3, "
            f"got {capability}"
        )
    tensors = (
        k,
        v,
        out_partial,
        lse_partial,
        seqused_k,
        page_table,
        valid_splits,
    )
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("SM103 hd512 SplitKV tensors must share one CUDA device")
    if kv_layout != "HND":
        raise ValueError("the decode adapter supports HND KV cache only")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("SM103 hd512 SplitKV requires BF16 Q/K/V")
    if out_partial.dtype != torch.float32 or lse_partial.dtype != torch.float32:
        raise TypeError("SM103 hd512 SplitKV requires FP32 partial O/LSE")
    if any(
        tensor.dtype != torch.int32
        for tensor in (seqused_k, page_table, valid_splits)
    ):
        raise TypeError("seqused_k, page table, and valid splits must be int32")
    if q.ndim != 3 or tuple(q.shape[1:]) != (32, 512):
        raise ValueError(f"expected packed Q shape (batch, 32, 512), got {tuple(q.shape)}")
    batch = int(q.shape[0])
    if k.ndim != 4 or tuple(k.shape[1:]) != (4, 64, 512):
        raise ValueError(
            "expected page64 HND K shape (num_pages, 4, 64, 512), "
            f"got {tuple(k.shape)}"
        )
    if v.shape != k.shape or v.stride() != k.stride():
        raise ValueError("K and V must have identical HND shapes and strides")
    if tuple(out_partial.shape) != (32, batch, 32, 512):
        raise ValueError("partial output must have shape (32, batch, 32, 512)")
    if tuple(lse_partial.shape) != (32, batch, 32):
        raise ValueError("partial LSE must have shape (32, batch, 32)")
    if tuple(seqused_k.shape) != (batch,) or tuple(valid_splits.shape) != (batch,):
        raise ValueError("seqused_k and valid splits must have one entry per request")
    if page_table.ndim != 2 or int(page_table.shape[0]) != batch:
        raise ValueError("page table must have one row per request")
    if int(page_table.shape[1]) <= 0:
        raise ValueError("page table width must be positive")
    if not q.is_contiguous() or not out_partial.is_contiguous():
        raise ValueError("packed Q and partial output must be contiguous")
    expected_lse_stride = (batch * 32, 1, batch)
    if tuple(lse_partial.stride()) != expected_lse_stride:
        raise ValueError(
            "logical LSE [split, batch, head] must use contiguous "
            "[split, head, batch] backing; expected strides "
            f"{expected_lse_stride}, got {lse_partial.stride()}"
        )
    if not seqused_k.is_contiguous() or not page_table.is_contiguous():
        raise ValueError("seqused_k and page table must be contiguous")
    if not valid_splits.is_contiguous():
        raise ValueError("valid splits must be contiguous")
    # K/V are zero-copy views of vLLM's packed HND cache and deliberately need
    # not be contiguous, but their innermost dimension and all strides must be
    # positive so the pointer-derived TMA descriptors are valid.
    if k.stride(-1) != 1 or min(k.stride()) <= 0:
        raise ValueError(f"unsupported HND K/V strides: {k.stride()}")
    return device_index


def _get_compiled_sm103_hd512_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    softmax_scale: float,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    valid_splits: torch.Tensor,
    *,
    kv_layout: str,
):
    device_index = _validate_sm103_hd512_decode_tensors(
        q,
        k,
        v,
        out_partial,
        lse_partial,
        seqused_k,
        page_table,
        valid_splits,
        kv_layout=kv_layout,
    )
    capability = torch.cuda.get_device_capability(device_index)
    key = Sm103Hd512SplitKVCompileKey(
        adapter_abi_version=_ADAPTER_ABI_VERSION,
        device_index=device_index,
        device_capability=capability,
        kv_layout=kv_layout,
        page_size=int(k.shape[-2]),
        q_spec=_torch_tensor_spec(q),
        k_spec=_torch_tensor_spec(k),
        v_spec=_torch_tensor_spec(v),
        out_partial_spec=_torch_tensor_spec(out_partial),
        lse_partial_spec=_torch_tensor_spec(lse_partial),
        seqused_k_spec=_torch_tensor_spec(seqused_k),
        page_table_spec=_torch_tensor_spec(page_table),
        valid_splits_spec=_torch_tensor_spec(valid_splits),
        compile_options=_COMPILE_OPTIONS,
    )
    compiled = _compiled_decode_cache.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SM103 hd512 SplitKV specialization was not warmed before "
            f"CUDA graph capture: {key}"
        )

    q_cute = _as_cute_tensor(q, BFloat16, 16)
    k_cute = _as_cute_tensor(k, BFloat16, 16)
    v_cute = _as_cute_tensor(v, BFloat16, 16)
    out_partial_cute = _as_cute_tensor(out_partial, Float32, 16)
    lse_partial_cute = _as_cute_tensor(lse_partial, Float32, 4)
    seqused_k_cute = _as_cute_tensor(seqused_k, Int32, 4)
    page_table_cute = _as_cute_tensor(page_table, Int32, 4)
    valid_splits_cute = _as_cute_tensor(valid_splits, Int32, 4)
    fmha = BlackwellHd512SplitKVFusedMultiHeadAttentionForward(
        headdim=512,
        grouped_head_tile=8,
        page_size=64,
        kv_layout=kv_layout,
    )
    compiled = cute.compile(
        fmha,
        q_cute,
        k_cute,
        v_cute,
        out_partial_cute,
        lse_partial_cute,
        softmax_scale,
        seqused_k_cute,
        page_table_cute,
        valid_splits_cute,
        cutlass_torch.default_stream(),
        options=_COMPILE_OPTIONS,
    )
    _compiled_decode_cache[key] = compiled
    return compiled


def run_sm103_hd512_splitkv_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    softmax_scale: float,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    valid_splits: torch.Tensor,
) -> None:
    """Run the exact HND Q1 specialization and its warp-cooperative combine."""
    if out.dtype != torch.bfloat16 or out.shape != q.shape or not out.is_contiguous():
        raise ValueError("SM103 hd512 SplitKV output must be contiguous BF16 and match Q")

    compiled = _get_compiled_sm103_hd512_decode(
        q,
        k,
        v,
        out_partial,
        lse_partial,
        softmax_scale,
        seqused_k,
        page_table,
        valid_splits,
        kv_layout="HND",
    )
    q_cute = _as_cute_tensor(q, BFloat16, 16)
    k_cute = _as_cute_tensor(k, BFloat16, 16)
    v_cute = _as_cute_tensor(v, BFloat16, 16)
    out_partial_cute = _as_cute_tensor(out_partial, Float32, 16)
    lse_partial_cute = _as_cute_tensor(lse_partial, Float32, 4)
    seqused_k_cute = _as_cute_tensor(seqused_k, Int32, 4)
    page_table_cute = _as_cute_tensor(page_table, Int32, 4)
    valid_splits_cute = _as_cute_tensor(valid_splits, Int32, 4)
    compiled(
        q_cute,
        k_cute,
        v_cute,
        out_partial_cute,
        lse_partial_cute,
        softmax_scale,
        seqused_k_cute,
        page_table_cute,
        valid_splits_cute,
        cutlass_torch.current_stream(),
    )

    from .sm103_hd512_splitkv_combine import (
        run_sm103_hd512_splitkv_combine,
    )

    run_sm103_hd512_splitkv_combine(
        out_partial,
        lse_partial,
        out,
        valid_splits,
    )


__all__ = [
    "BlackwellHd512SplitKVFusedMultiHeadAttentionForward",
    "Sm103Hd512SplitKVCompileKey",
    "run_sm103_hd512_splitkv_decode",
]
