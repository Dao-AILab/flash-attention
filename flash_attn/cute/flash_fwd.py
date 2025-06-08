# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm80.h
# from Cutlass C++ to Cute-DSL.
# Built on Cute-DSL example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.ampere_helpers as sm80_utils_basic
import cutlass.utils.hopper_helpers as sm90_utils_basic

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import hopper_helpers as sm90_utils
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.seqlen_info import SeqlenInfo
from flash_attn.cute.pipeline import PipelineTmaAsyncNoCluster


class FlashAttentionForwardBase:

    arch: int = 80

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        is_causal: bool = False,
        has_softcap: bool = False,
        Q_in_regs: bool = False,
    ):
        """Initializes the configuration for a flash attention kernel.

        All contiguous dimensions must be at least 16 bytes aligned, which means that the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        """
        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads
        self.is_causal = is_causal
        self.has_softcap = has_softcap
        self.num_stages = num_stages
        self.Q_in_regs = Q_in_regs

    @staticmethod
    def can_implement(
        dtype, head_dim, head_dim_v, m_block_size, n_block_size, num_stages, num_threads, is_causal,
        Q_in_regs=False
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if n_block_size % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage_Q = m_block_size * head_dim * 2
        smem_usage_K = n_block_size * head_dim * num_stages * 2
        smem_usage_V = n_block_size * head_dim_v * num_stages * 2
        smem_usage_QV = (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        smem_usage = smem_usage_QV + smem_usage_K
        # TODO: sm86 and sm89
        smem_capacity = sm80_utils_basic.SMEM_CAPACITY["sm80"]
        if smem_usage > smem_capacity:
            return False
        # Check if twice the block size is divisible by the number of threads
        if (m_block_size * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric] | None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mLSE_type is not None and mLSE_type not in [cutlass.Float32]):
            raise TypeError("LSE tensor must be Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom = self._get_smem_layout_atom()
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.m_block_size, self.head_dim_padded), (0, 1),
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.n_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom, (self.n_block_size, self.head_dim_v_padded, self.num_stages), (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom, (self.m_block_size, self.head_dim_v_padded), (0, 1),
        )
        if cutlass.const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom, (self.m_block_size, self.n_block_size), (0, 1),
            )
        else:
            self.sP_layout = None

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits,
        )
        # tQK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_producer_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        tQK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.m_block_size % tQK_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0),
        )
        # TODO: need a different layout for O if O dtype is not the same as V dtype
        # tO_layout: thread layout for O store
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.m_block_size % tO_layout.shape[0] == 0

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        # gmem_tiled_copy_QK: tiled copy for QK load
        self.gmem_tiled_copy_QK = cute.make_tiled_copy_tv(atom_async_copy, tQK_layout, vQKV_layout)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy, tV_layout, vQKV_layout)
        # gmem_tiled_copy_O: tiled copy for O store
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

    def _get_smem_layout_atom(self):
        raise NotImplementedError()

    def _get_tiled_mma(self):
        raise NotImplementedError()

    def _get_shared_storage_cls(self):
        raise NotImplementedError()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softcap: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        raise NotImplementedError()

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        m_block: cutlass.Int32,
        num_head: cutlass.Int32,
        batch_size: cutlass.Int32,
    ):
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        # Make sure all threads have finished reading V
        cute.arch.barrier(barrier_id=5, number_of_threads=self.num_mma_threads)
        smem_copy_atom_O = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_O = utils.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))

        # Write LSE from rmem -> gmem
        if cutlass.const_expr(mLSE is not None):
            gLSE = cute.local_tile(mLSE[None, num_head, batch_size], (self.m_block_size,), (m_block,))
            gLSE_expanded_layout = cute.append(
                gLSE.layout,
                cute.make_layout((self.head_dim_v_padded,), stride=(0,))
            )
            gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
            thr_mma = tiled_mma.get_slice(tidx)
            taccOgLSE = utils.make_acc_tensor_mn_view(thr_mma.partition_C(gLSE_expanded))
            assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
            taccOcO = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
            t0accOcO = utils.make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cO))
            # Only the thread corresponding to column 0 writes out the lse to gmem
            if taccOcO[0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                    if cute.elem_less(t0accOcO[m, 0][0], mO.shape[0] - m_block * self.m_block_size - taccOcO[0][0]):
                        taccOgLSE[m, 0] = lse[m]

        gO = cute.local_tile(
            mO[None, None, num_head, batch_size],
            (self.m_block_size, self.head_dim_v_padded),
            (m_block, 0),
        )
        # thr_mma = tiled_mma.get_slice(tidx)
        # taccOgO = thr_mma.partition_C(gO)
        # cute.autovec_copy(rO, taccOgO)
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self.dtype)
        # sync before all smem stores are done.
        cute.arch.barrier(barrier_id=5, number_of_threads=self.num_mma_threads)
        # load acc O from smem to rmem for wider vectorization
        cute.autovec_copy(tOsO, tOrO)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        # copy acc O from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            # if cute.elem_less(tOcO[0, rest_m, 0][0], mO.shape[1] - m_block * self.m_block_size):
            if cute.elem_less(t0OcO[0, rest_m, 0][0], mO.shape[0] - m_block * self.m_block_size - tOcO[0][0]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None] if self.check_hdim_v_oob else None,
                )

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ):
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=headdim)
        for m in range(cute.size(tQsQ.shape[1])):
            # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
            # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            if cute.elem_less(t0QcQ[0, m, 0][0], seqlen - block * self.m_block_size - tQcQ[0][0]):
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if self.check_hdim_oob else None,
                )
            # We don't need to clear the sQ smem tiles since we'll only write out the valid outputs

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write: cutlass.Int32,
        seqlen: cutlass.Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load K?
        is_even_n_smem_k = self.n_block_size % gmem_tiled_copy.tiler_mn[0].shape == 0
        if cutlass.const_expr(need_predicates or not is_even_n_smem_k):
            # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
            # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
            if cutlass.const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.n_block_size
            else:
                if cutlass.const_expr(not need_predicates):
                    seqlen_limit = self.n_block_size
                else:
                    seqlen_limit = cutlass.min(seqlen - block * self.n_block_size, self.n_block_size)
            seqlen_limit -= tKcK[0][0]
            for n in range(cute.size(tKsK.shape[1])):
                if cute.elem_less(t0KcK[0, n, 0][0], seqlen_limit):
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[None, n, None, smem_pipe_write if self.num_stages > 1 else 0],
                        pred=tKpK[None, n, None] if self.check_hdim_oob else None,
                    )
                # We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        else:
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[None, None, None, smem_pipe_write if self.num_stages > 1 else 0],
                pred=tKpK if self.check_hdim_oob else None,
            )

    @cute.jit
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write: cutlass.Int32,
        seqlen: cutlass.Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load V?
        is_even_n_smem_v = self.n_block_size % gmem_tiled_copy.tiler_mn[0].shape == 0
        if cutlass.const_expr(need_predicates or not is_even_n_smem_v):
            for n in range(cute.size(tVsV.shape[1])):
                # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
                if is_even_n_smem_v or n < cute.size(tVsV.shape[1]) - 1 or cute.elem_less(tVcV[0, n, 0][0], self.n_block_size):
                    predicate = tVpV[None, n, None] if self.check_hdim_v_oob else None
                    if cutlass.const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.n_block_size - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in range(cute.size(predicate.shape[1])):
                            for i in range(cute.size(predicate.shape[0])):
                                predicate[i, k] = (tVpV[i, n, k] if self.check_hdim_v_oob else True) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[None, n, None, smem_pipe_write if self.num_stages > 1 else 0],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[None, None, None, smem_pipe_write if self.num_stages > 1 else 0],
                pred=tVpV if self.check_hdim_v_oob else None,
            )


class FlashAttentionForwardSm80(FlashAttentionForwardBase):

    def _get_smem_layout_atom(self):
        sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_padded)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_v_padded)
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = utils.max_constexpr(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]

        @cute.struct
        class SharedStorageQKV:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            sQ: sQV_struct
            sK: sK_struct

        return SharedStorageQKV if cutlass.const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softcap: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        self._check_type(*(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE)))
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0])) for t in (mQ, mK, mV, mO)]
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mQ.shape[0], self.m_block_size),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
        )
        # If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        # Right after this, we multiply by log2(e) before applying exp2.
        # To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        # (assigning it to softcap_val) and pre-multiply softcap_val * log2(e)
        # (assigning it to softmax_scale_log2).
        LOG2_E = math.log2(math.e)
        if cutlass.const_expr(not self.has_softcap):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = cutlass.Float32(0.0)
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = softmax_scale / softcap
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale_log2,
            softcap_val,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale_log2: cutlass.Float32,
        softcap_val: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[0], self.n_block_size)
        if self.is_causal:
            n_block_max = min(
                cute.ceil_div((m_block + 1) * self.m_block_size + mK.shape[0] - mQ.shape[0], self.n_block_size),
                n_block_max,
            )
        # TODO: return early if n_block_max == 0
        # if self.is_causal:
        #     if n_block_max <= 0:
        #         return
        n_block = n_block_max - 1

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.m_block_size, self.head_dim_padded)
        blkK_shape = (self.n_block_size, self.head_dim_padded)
        blkV_shape = (self.n_block_size, self.head_dim_v_padded)
        # (m_block_size, head_dim)
        gQ = cute.local_tile(mQ[None, None, num_head, batch_size], blkQ_shape, (m_block, 0))
        # (n_block_size, head_dim, n_block)
        num_head_kv = num_head // self.qhead_per_kvhead
        gK = cute.local_tile(mK[None, None, num_head_kv, batch_size], blkK_shape, (None, 0))
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(mV[None, None, num_head_kv, batch_size], blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if cutlass.const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        # Transpose view of V to tensor with layout (head_dim_v, n_block_size) for tiled mma
        sVt = utils.transpose_view(sV)

        gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K)
        tQgQ = gmem_thr_copy_QK.partition_S(gQ)
        tQsQ = gmem_thr_copy_QK.partition_D(sQ)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKgK = gmem_thr_copy_QK.partition_S(gK)
        tKsK = gmem_thr_copy_QK.partition_D(sK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVgV = gmem_thr_copy_V.partition_S(gV)
        tVsV = gmem_thr_copy_V.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.m_block_size, self.head_dim_v_padded))
        acc_O = cute.make_fragment(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype,
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
        cK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
        tKcK = gmem_thr_copy_QK.partition_S(cK)
        t0KcK = gmem_thr_copy_QK.get_slice(0).partition_S(cK)
        if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
        # use "if" on the mn dimension.
        # This is to reduce register pressure and gets 2-3% performance gain.
        tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
        if cutlass.const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

        # ///////////////////////////////////////////////////////////////////////////////
        # Softmax intermediate result: row_max and row_sum
        # ///////////////////////////////////////////////////////////////////////////////
        # shape: (atom_v_m * rest_m)
        row_max = cute.make_fragment(acc_O.shape[0][0] * acc_O.shape[1], cutlass.Float32)
        row_sum = cute.make_fragment_like(row_max)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)
        softmax = Softmax(softmax_scale_log2)

        # group parameters for compute_one_n_block
        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk, thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ, tSsK=tSsK, tOsVt=tOsVt,
        )
        softmax_params = SimpleNamespace(softmax=softmax, row_max=row_max, row_sum=row_sum)
        seqlen = SeqlenInfo(seqlen_q=mQ.shape[0], seqlen_k=mK.shape[0])
        load_K = partial(self.load_K, gmem_tiled_copy_QK, tKgK, tKsK, tKcK, t0KcK, tKpK,
                         seqlen=seqlen.seqlen_k)
        load_V = partial(self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                         seqlen=seqlen.seqlen_k)
        # Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
        # -inf to e.g. -50.0, which can affect the attention softmax.
        def scoremod_premask_fn(acc_S):
            if cutlass.const_expr(self.has_softcap):
                acc_S.store(cute.math.tanh(acc_S.load() * softcap_val, fastmath=True))

        compute_one_n_block = partial(
            self.compute_one_n_block, mma_params=mma_params, smem_copy_params=smem_copy_params,
            softmax_params=softmax_params, load_K=load_K, load_V=load_V,
            scoremod_premask_fn=scoremod_premask_fn,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        self.load_Q(gmem_thr_copy_QK, tQgQ, tQsQ, m_block, seqlen=seqlen.seqlen_q,
                    headdim=mQ.shape[1])
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
            if cutlass.const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        # read from smem_q to registers, then load V.
        # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
        if cutlass.const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

        for stage in range(self.num_stages):
            if cutlass.const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage==0)
                cute.arch.cp_async_commit_group()
            if stage < self.num_stages - 1:
                if stage == 0 or n_block - stage >= 0:
                    load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage==0)
                cute.arch.cp_async_commit_group()
        if cutlass.const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # ///////////////////////////////////////////////////////////////////////////////
        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
        # We also need masking on S if it's causal, for the last several blocks.
        mask = AttentionMask(self.m_block_size, self.n_block_size, seqlen.seqlen_q, seqlen.seqlen_k)
        mask_fn = partial(
            mask.apply_mask, m_block=m_block, thr_mma=thr_mma_qk, mask_causal=self.is_causal
        )

        # First iteration with seqlen masking
        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(self.num_stages - 1)
        compute_one_n_block(n_block, smem_pipe_read, smem_pipe_write, is_first_n_block=True,
                            check_inf=True, mask_fn=partial(mask_fn, mask_seqlen=True))
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # Next couple of iterations with causal masking
        if self.is_causal:
            m_idx_min = m_block * self.m_block_size
            n_idx_right = m_idx_min + seqlen.seqlen_k - seqlen.seqlen_q
            n_block_min_causal_local_mask = cutlass.max(0, n_idx_right // self.n_block_size)
            # Currently we can't do loop with negative step
            # https://github.com/NVIDIA/cutlass/issues/2326
            for n_tile in cutlass.range_dynamic(n_block_min_causal_local_mask, n_block_max - 1, unroll=1):
                n_block = n_block_max - 2 - n_tile + n_block_min_causal_local_mask
                compute_one_n_block(n_block, smem_pipe_read, smem_pipe_write, check_inf=True,
                                    mask_fn=partial(mask_fn, mask_seqlen=False))
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # The remaining iterations have no masking
        for n_tile in cutlass.range_dynamic(n_block, unroll=1):
            compute_one_n_block(n_block - n_tile - 1, smem_pipe_read, smem_pipe_write, check_inf=False)
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        # normalize acc_O by row_sum and calculate the lse
        softmax.normalize(acc_O, row_max, row_sum)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O, row_sum, mO, mLSE, sO,
            gmem_tiled_copy_O, tiled_mma_pv, tidx, m_block, num_head, batch_size
        )

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: cutlass.Int32,
        smem_pipe_read: cutlass.Int32,
        smem_pipe_write: cutlass.Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        load_K: Callable,
        load_V: Callable,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = False,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """
        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size))
        acc_S = cute.make_fragment(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)
        # wait for smem tile QK before mma calculation for S
        sync()
        # need predicates for the first tile
        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(n_block - self.num_stages + 1, smem_pipe_write,
                       need_predicates=is_first_n_block and self.num_stages == 1)
            cute.arch.cp_async_commit_group()
        load_V_next()
        sm80_utils.gemm(
            mma_params.thr_mma_qk, acc_S, mma_params.tSrQ, mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, smem_pipe_read if self.num_stages > 1 else 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
            # hook_fn=load_V_next,
            A_in_regs=self.Q_in_regs,
        )
        scoremod_premask_fn(acc_S)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(n_block - self.num_stages, smem_pipe_write, need_predicates=False)
            cute.arch.cp_async_commit_group()
        # wait for smem tile V for O
        if cutlass.const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        softmax_params.softmax.online_softmax_rescale_O(
            acc_S, mma_params.acc_O, softmax_params.row_max, softmax_params.row_sum,
            is_first_n_block=is_first_n_block, check_inf=check_inf,
        )
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = cute.make_tensor(rP.iterator, utils.convert_layout_acc_frgA(rP.layout))
        if cutlass.const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_O, tOrP, mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, smem_pipe_read if self.num_stages > 1 else 0],
            smem_copy_params.smem_thr_copy_V,
            # hook_fn=load_K_next,
        )
        # if cutlass.const_expr(self.num_stages > 1):
        #     load_K_next()


class FlashAttentionForwardSm90(FlashAttentionForwardBase):

    arch = 90

    def __init__(self, *args, intra_wg_overlap: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_padded
            ),
            self.dtype
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_v_padded
            ),
            self.dtype
        )
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.n_block_size
            ),
            self.dtype
        )
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            cutlass.Float32,
            atom_layout_mnk=(self.m_block_size // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.n_block_size),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            cutlass.Float32,
            atom_layout_mnk=(self.m_block_size // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.head_dim_v_padded),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = utils.max_constexpr(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if self.sP_layout is not None else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 for Q, 1 for O, self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_QO_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV if cutlass.const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softcap: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        self._check_type(*(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE)))
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_mma_warp_groups = self.num_mma_threads // self.num_threads_per_warp_group
        self.num_producer_threads = 32
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs = 240
        self.num_producer_regs = 24
        self.use_scheduler_barrier = (self.num_mma_warp_groups >= 2 and self.head_dim <= 128) if self.intra_wg_overlap else (self.num_mma_warp_groups == 2)
        # TODO: rescale_O_before_gemm
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0])) for t in (mQ, mK, mV, mO)]
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        self.tma_copy_q_bytes = cute.size_in_bytes(mQ.element_type, self.sQ_layout)
        self.tma_copy_k_bytes = cute.size_in_bytes(mK.element_type, cute.select(self.sK_layout, mode=[0, 1]))
        self.tma_copy_v_bytes = cute.size_in_bytes(mV.element_type, cute.select(self.sV_layout, mode=[0, 1]))
        tma_atom_Q, tma_tensor_Q = cpasync.make_tma_tile_atom(
            gmem_tiled_copy_Q, mQ, self.sQ_layout, (self.m_block_size, self.head_dim_padded), 1  # No mcast
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tma_tile_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
            1  # No mcast for now
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tma_tile_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_v_padded),
            1  # No mcast for now
        )
        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mQ.shape[0], self.m_block_size),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
        )
        # If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        # Right after this, we multiply by log2(e) before applying exp2.
        # To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        # (assigning it to softcap_val) and pre-multiply softcap_val * log2(e)
        # (assigning it to softmax_scale_log2).
        LOG2_E = math.log2(math.e)
        if cutlass.const_expr(not self.has_softcap):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = cutlass.Float32(0.0)
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = softmax_scale / softcap
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            softmax_scale_log2,
            softcap_val,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            # the compiler is unhappy about us using tiled_mma_qk/pv and setting the ACCUMULATE
            # field inside a for loop, so we work around by creating multiple copies of the
            # tiled_mma_qk/pv.
            *((tiled_mma_qk, tiled_mma_pv) * 3),
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        softmax_scale_log2: cutlass.Float32,
        softcap_val: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_qk_copy: cute.TiledMma,
        tiled_mma_pv_copy: cute.TiledMma,
        tiled_mma_qk_copy1: cute.TiledMma,
        tiled_mma_pv_copy1: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)

        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier init
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 0:
            # if tidx < 2:
            #     # barrierO num threads should be self.num_mma_threads
            #     cute.arch.mbarrier_init_arrive_cnt(mbar_ptr_Q + tidx, 1 if tidx == 0 else self.num_mma_threads)
            cute.arch.mbarrier_init_arrive_cnt(mbar_ptr_Q, 1)
            # cute.arch.mbarrier_init_arrive_cnt(mbar_ptr_Q + 1, self.num_mma_threads)
        # We rely on pipeline_k and pipeline_v to initialize the mbarrier fence and sync
        # cute.arch.mbarrier_init_fence()
        # # TODO: if cluster: need cluster arrive here
        # # We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
        # cute.arch.barrier()
        pipeline_kv_producer_group = cutlass.utils.CooperativeGroup(cutlass.utils.Agent.Thread)
        pipeline_kv_consumer_group = cutlass.utils.CooperativeGroup(cutlass.utils.Agent.Thread, self.num_mma_threads // self.num_threads_per_warp_group)
        pipeline_k = PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_k_bytes,
            init_wait=False,
        )
        pipeline_v = PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_v_bytes,
        )

        n_block_max = cute.ceil_div(mK.shape[0], self.n_block_size)
        if self.is_causal:
            n_block_max = min(
                cute.ceil_div((m_block + 1) * self.m_block_size + mK.shape[0] - mQ.shape[0], self.n_block_size),
                n_block_max,
            )
        # TODO: return early if n_block_max == 0
        # if self.is_causal:
        #     if n_block_max <= 0:
        #         return

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.m_block_size, self.head_dim_padded)
        blkK_shape = (self.n_block_size, self.head_dim_padded)
        blkV_shape = (self.n_block_size, self.head_dim_v_padded)
        # (m_block_size, head_dim)
        gQ = cute.local_tile(mQ[None, None, num_head, batch_size], blkQ_shape, (m_block, 0))
        # (n_block_size, head_dim, n_block)
        num_head_kv = num_head // self.qhead_per_kvhead
        gK = cute.local_tile(mK[None, None, num_head_kv, batch_size], blkK_shape, (None, 0))
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(mV[None, None, num_head_kv, batch_size], blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if cutlass.const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type)
        if cutlass.const_expr(sP_layout is not None):
            # sP_pi = storage.sP.get_tensor(sP_layout)
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
            sP_pi = cute.make_tensor(sP.iterator, sP_layout)
        else:
            sP, sP_pi = None
        # Transpose view of V to tensor with layout (head_dim_v, n_block_size) for tiled mma
        sVt = utils.transpose_view(sV)

        if warp_idx < 4:  # Producer
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 2),
                cute.group_modes(gQ, 0, 2),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 2),
                cute.group_modes(gK, 0, 2),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gV, 0, 2),
            )
            smem_pipe_write = cutlass.utils.make_pipeline_state(
                cutlass.utils.PipelineUserType.Producer, self.num_stages
            )
            load_K = partial(self.load_K, tma_atom_K, tKgK, tKsK, pipeline_k)
            load_V = partial(self.load_K, tma_atom_V, tVgV, tVsV, pipeline_v)
            if warp_idx == 0:  # Producer
                # load_Q
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init_tx_bytes(mbar_ptr_Q, self.tma_copy_q_bytes)
                cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=mbar_ptr_Q)
                for n_tile in cutlass.range_dynamic(n_block_max, unroll=2):
                    n_block = n_block_max - n_tile - 1
                    load_K(n_block, smem_pipe_write=smem_pipe_write)
                    load_V(n_block, smem_pipe_write=smem_pipe_write)
                    smem_pipe_write.advance()

        else:  # Consumer
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx = tidx - 128
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            warp_group_thread_layout = cute.make_layout(
                self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
            wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
            tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
            tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
            tOrP = tiled_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sP)) if cutlass.const_expr(sP is not None) else None
            tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))
            acc_shape_O = tiled_mma_pv.partition_shape_C((self.m_block_size, self.head_dim_v_padded))
            acc_O = cute.make_fragment(acc_shape_O, cutlass.Float32)

            # ///////////////////////////////////////////////////////////////////////////////
            # Smem copy atom tiling
            # ///////////////////////////////////////////////////////////////////////////////
            smem_copy_atom_P = utils.get_smem_store_atom(self.arch, self.dtype)
            smem_thr_copy_P = utils.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
            # tPsP = smem_thr_copy_P.partition_D(sP_pi) if cutlass.const_expr(sP_pi is not None) else None
            tPsP = smem_thr_copy_P.partition_D(sP) if cutlass.const_expr(sP is not None) else None
            # if cute.arch.thread_idx()[0] == 0:
            #     cute.printf(sP_pi.layout, sP_pi.iterator)
            #     cute.printf(sP.layout, sP.iterator)
            #     cute.printf(tPsP.layout, tPsP.iterator)

            self.mma_init()

            # ///////////////////////////////////////////////////////////////////////////////
            # Softmax intermediate result: row_max and row_sum
            # ///////////////////////////////////////////////////////////////////////////////
            # shape: (atom_v_m * rest_m)
            row_max = cute.make_fragment(acc_O.shape[0][0] * acc_O.shape[1], cutlass.Float32)
            row_sum = cute.make_fragment_like(row_max)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)
            softmax = Softmax(softmax_scale_log2)

            # group parameters for compute_one_n_block
            mma_params = SimpleNamespace(
                tSrQ=tSrQ, tSrK=tSrK, tOrP=tOrP, tOrVt=tOrVt, acc_O=acc_O,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP,
            )
            softmax_params = SimpleNamespace(softmax=softmax, row_max=row_max, row_sum=row_sum)
            seqlen = SeqlenInfo(seqlen_q=mQ.shape[0], seqlen_k=mK.shape[0])
            # Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
            # -inf to e.g. -50.0, which can affect the attention softmax.
            def scoremod_premask_fn(acc_S):
                if cutlass.const_expr(self.has_softcap):
                    acc_S.store(cute.math.tanh(acc_S.load() * softcap_val, fastmath=True))

            compute_one_n_block = partial(
                self.compute_one_n_block, pipeline_k=pipeline_k, pipeline_v=pipeline_v,
                mma_params=mma_params, smem_copy_params=smem_copy_params,
                softmax_params=softmax_params, scoremod_premask_fn=scoremod_premask_fn,
            )

            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
            # We also need masking on S if it's causal, for the last several blocks.
            mask = AttentionMask(
                self.m_block_size, self.n_block_size, seqlen.seqlen_q, seqlen.seqlen_k
            )
            mask_fn = partial(
                mask.apply_mask, m_block=m_block, thr_mma=thr_mma_qk, mask_causal=self.is_causal
            )
            cute.arch.mbarrier_wait(mbar_ptr_Q, phase=0)
            n_block = n_block_max - 1
            smem_pipe_read = cutlass.utils.make_pipeline_state(
                cutlass.utils.PipelineUserType.Consumer, self.num_stages
            )
            self.warp_scheduler_barrier_wait()
            # First iteration with seqlen masking
            compute_one_n_block(
                n_block, smem_pipe_read, tiled_mma_qk, tiled_mma_pv,
                is_first_n_block=True, check_inf=True, mask_fn=partial(mask_fn, mask_seqlen=True)
            )
            smem_pipe_read.advance()
            # Next couple of iterations with causal masking
            if self.is_causal:
                m_idx_min = m_block * self.m_block_size
                n_idx_right = m_idx_min + seqlen.seqlen_k - seqlen.seqlen_q
                n_block_min_causal_local_mask = cutlass.max(0, n_idx_right // self.n_block_size)
                # Currently we can't do loop with negative step
                # https://github.com/NVIDIA/cutlass/issues/2326
                for n_tile in cutlass.range_dynamic(n_block_min_causal_local_mask, n_block_max - 1, unroll=1):
                    n_block = n_block_max - 2 - n_tile + n_block_min_causal_local_mask
                    compute_one_n_block(
                        n_block, smem_pipe_read, tiled_mma_qk_copy, tiled_mma_pv_copy,
                        check_inf=True, mask_fn=partial(mask_fn, mask_seqlen=False)
                    )
                    smem_pipe_read.advance()
            # The remaining iterations have no masking
            for n_tile in cutlass.range_dynamic(n_block, unroll=1):
                compute_one_n_block(
                    n_block - n_tile - 1, smem_pipe_read, tiled_mma_qk_copy1, tiled_mma_pv_copy1,
                    check_inf=False,
                )
                smem_pipe_read.advance()
            self.warp_scheduler_barrier_arrive()

            # normalize acc_O by row_sum and calculate the lse
            softmax.normalize(acc_O, row_max, row_sum)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            # reuse sQ's data iterator
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            # sO = cute.make_tensor(cute.recast_ptr(sO.iterator, sO_layout.inner, dtype=sO.element_type), sO_layout.outer)
            self.epilogue(
                acc_O, row_sum, mO, mLSE, sO,
                gmem_tiled_copy_O, tiled_mma_pv, tidx, m_block, num_head, batch_size
            )

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: cutlass.Int32,
        smem_pipe_read: cutlass.utils.PipelineState,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        pipeline_k: cutlass.utils.PipelineAsync,
        pipeline_v: cutlass.utils.PipelineAsync,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = False,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size)), cutlass.Float32
        )
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        sm90_utils.gemm(
            tiled_mma_qk, acc_S, mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1
        )
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)
        scoremod_premask_fn(acc_S)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        softmax_params.softmax.online_softmax_rescale_O(
            acc_S, mma_params.acc_O, softmax_params.row_max, softmax_params.row_sum,
            is_first_n_block=is_first_n_block, check_inf=check_inf,
        )
        # if cute.arch.thread_idx()[0] == 0:
        #     cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        # tOrP = cute.make_tensor(rP.iterator, utils.convert_layout_acc_frgA(rP.layout))
        tPrP = smem_copy_params.smem_thr_copy_P.retile(rP)
        cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        # Fence and barrier to make sure smem store is visible to WGMMA
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
        cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_wait()
        sm90_utils.gemm(
            tiled_mma_pv, mma_params.acc_O, mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read.index],
            zero_init=is_first_n_block, wg_wait=0
        )
        pipeline_v.consumer_release(smem_pipe_read)

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if cutlass.const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                utils.barrier_arrive(
                    barrier_id=1 + 0, number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_wait(self):
        if cutlass.const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=1 - 1 + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group
            )

    def warp_scheduler_barrier_arrive(self):
        if cutlass.const_expr(self.use_scheduler_barrier):
            assert self.num_mma_warp_groups in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            next_wg = 1 - cur_wg if self.num_mma_warp_groups == 2 else (cur_wg + 1 if cur_wg < self.num_mma_warp_groups - 1 else 0)
            utils.barrier_arrive(
                barrier_id=1 + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    # @cute.jit
    def load_K(
        self,
        tma_atom: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        pipeline: cutlass.utils.PipelineAsync,
        block: cutlass.Int32,
        smem_pipe_write: cutlass.utils.PipelineState,
    ):
        # TODO: mcast
        # TODO check warp_idx if we have 128 producer threads
        pipeline.producer_acquire(smem_pipe_write)
        cute.copy(
            tma_atom,
            tKgK[None, block],
            tKsK[None, smem_pipe_write.index],
            tma_bar_ptr=pipeline.producer_get_barrier(smem_pipe_write)
        )
