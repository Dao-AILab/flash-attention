# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of
# https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm80.h
# and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h
# from Cutlass C++ to Cute-DSL.
# Built on Cute-DSL example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils as utils_basic
import cutlass.utils.hopper_helpers as sm90_utils_basic

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import hopper_helpers as sm90_utils
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from flash_attn.cute.pack_gqa import PackGQA
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, SingleTileLPTScheduler, SingleTileVarlenScheduler, ParamsBase


class FlashAttentionForwardBase:

    arch: int = 80

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
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
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads
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
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_80")
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
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        mSeqUsedQ_type: Type[cutlass.Numeric] | None,
        mSeqUsedK_type: Type[cutlass.Numeric] | None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")
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
        if const_expr(sP_layout_atom is not None):
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
        # tQ_layout and tK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        assert self.num_producer_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.m_block_size % tQ_layout.shape[0] == 0
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

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, tQ_layout, vQKV_layout)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy, tK_layout, vQKV_layout)
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
        softmax_scale: Float32,
        softcap: Float32,
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
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        # Make sure all threads have finished reading V
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads)
        smem_copy_atom_O = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        pack_gqa = PackGQA(self.m_block_size, self.head_dim_v_padded, self.check_hdim_v_oob, self.qhead_per_kvhead)

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            if const_expr(not seqlen.has_cu_seqlens_q):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.head_dim_v_padded,), stride=(0,))
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
                        if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * self.m_block_size - taccOcO[0][0]:
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        if const_expr(not seqlen.has_cu_seqlens_q):
            mO_cur = mO[None, None, head_idx, batch_idx]
        else:
            offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
            mO_cur = cute.domain_offset((offset, 0), mO[None, None, head_idx])
        # thr_mma = tiled_mma.get_slice(tidx)
        # taccOgO = thr_mma.partition_C(gO)
        # cute.autovec_copy(rO, taccOgO)
        # sync to make sure all smem stores are done
        if const_expr(self.use_tma_O):
            # ensure smem writes are visible to TMA
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier_arrive(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (m_block, 0))
            tOsO, tOgO = cpasync.tma_partition(
                tma_atom_O,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.group_modes(gO, 0, 2),
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:
                cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)
                cute.copy(tma_atom_O, tOsO, tOgO)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads)
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOrO = cute.make_fragment_like(tOsO, self.dtype)
            # load acc O from smem to rmem for wider vectorization
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (m_block, 0))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                # copy acc O from rmem to gmem
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if t0OcO[0, rest_m, 0][0] < seqlen.seqlen_q - m_block * self.m_block_size - tOcO[0][0]:
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                        )
            else:
                pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q)

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
    ):
        tQsQ, tQgQ = gmem_thr_copy.partition_D(sQ), gmem_thr_copy.partition_S(gQ)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=headdim)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
            # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            if t0QcQ[0, m, 0][0] < seqlen - block * self.m_block_size - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if const_expr(self.check_hdim_oob) else None,
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
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load K?
        is_even_n_smem_k = self.n_block_size % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_k):
            # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
            # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
            if const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.n_block_size
            else:
                if const_expr(not need_predicates):
                    seqlen_limit = self.n_block_size
                else:
                    seqlen_limit = cutlass.min(seqlen - block * self.n_block_size, self.n_block_size)
            seqlen_limit -= tKcK[0][0]
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                        pred=tKpK[None, n, None] if const_expr(self.check_hdim_oob) else None,
                    )
                # We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        else:
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
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
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load V?
        is_even_n_smem_v = self.n_block_size % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
                if is_even_n_smem_v or n < cute.size(tVsV.shape[1]) - 1 or tVcV[0, n, 0][0] < self.n_block_size:
                    predicate = tVpV[None, n, None] if const_expr(self.check_hdim_v_oob) else None
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.n_block_size - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                                predicate[i, k] = (tVpV[i, n, k] if const_expr(self.check_hdim_v_oob) else True) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
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
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
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

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        stream: cuda.CUstream,
        softmax_scale: Optional[Float32] = None,
        softcap: Optional[Float32] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        learnable_sink: Optional[cute.Tensor] = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        assert learnable_sink is None, "Learnable sink is not supported in this kernel"
        self._check_type(*(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE)))
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        # self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None
        self.use_tma_O = self.arch >= 90
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (*(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]), t.stride[-1])
        mQ, mK, mV, mO = [cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t))) for t in (mQ, mK, mV, mO)]
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
        if const_expr(softcap is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = None
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = Float32(softmax_scale / softcap)
            
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale_log2,
            softcap_val,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
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
        softmax_scale_log2: Float32,
        softcap_val: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        block_info = BlockInfo(
            self.m_block_size, self.n_block_size, self.is_causal, self.is_local,
            window_size_left, window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        seqlen = SeqlenInfoQK(seqlen_q_static=mQ.shape[0], seqlen_k_static=mK.shape[0])
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
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
        gQ = cute.local_tile(mQ[None, None, num_head, batch_size], blkQ_shape, (m_block, 0))
        num_head_kv = num_head // self.qhead_per_kvhead
        gK = cute.local_tile(mK[None, None, num_head_kv, batch_size], blkK_shape, (None, 0))
        gV = cute.local_tile(mV[None, None, num_head_kv, batch_size], blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        # Transpose view of V to tensor with layout (head_dim_v, n_block_size) for tiled mma
        sVt = utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.m_block_size, self.head_dim_v_padded))
        acc_O = cute.make_fragment(acc_shape_O, Float32)
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
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.head_dim_padded == self.head_dim_v_padded):
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
        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

        # shape: (atom_v_m * rest_m)
        softmax = Softmax(softmax_scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1])
        softmax.reset()

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
        load_K = partial(self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
                         seqlen=seqlen.seqlen_k)
        load_V = partial(self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                         seqlen=seqlen.seqlen_k)
        # Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
        # -inf to e.g. -50.0, which can affect the attention softmax.
        def scoremod_premask_fn(acc_S):
            if const_expr(softcap_val is not None):
                acc_S.store(cute.math.tanh(acc_S.load() * softcap_val, fastmath=True))

        compute_one_n_block = partial(
            self.compute_one_n_block, mma_params=mma_params, smem_copy_params=smem_copy_params,
            softmax=softmax, load_K=load_K, load_V=load_V, scoremod_premask_fn=scoremod_premask_fn,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        # read from smem_q to registers, then load V.
        # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

        for stage in cutlass.range_constexpr(self.num_stages):
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage==0)
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage==0)
                cute.arch.cp_async_commit_group()
        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # ///////////////////////////////////////////////////////////////////////////////
        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
        # We also need masking on S if it's causal, for the last several blocks.
        mask = AttentionMask(
            self.m_block_size, self.n_block_size, seqlen.seqlen_q, seqlen.seqlen_k,
            window_size_left, window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        mask_fn = partial(
            mask.apply_mask, m_block=m_block, thr_mma=thr_mma_qk,
            mask_causal=self.is_causal, mask_local=self.is_local,
        )

        # First iteration with seqlen masking
        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(self.num_stages - 1)
        compute_one_n_block(n_block, smem_pipe_read, smem_pipe_write, is_first_n_block=True,
                            check_inf=True, mask_fn=partial(mask_fn, mask_seqlen=True))
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # Next couple of iterations with causal masking
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_local_mask, unroll=1):
                n_block = n_block_max - 2 - n_tile
                compute_one_n_block(n_block, smem_pipe_read, smem_pipe_write, check_inf=True,
                                    mask_fn=partial(mask_fn, mask_seqlen=False))
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # The remaining iterations have no masking
        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(n_block - n_tile - 1, smem_pipe_read, smem_pipe_write, check_inf=True)
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # TODO: local

        # normalize acc_O by row_sum and calculate the lse
        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O, softmax.row_sum, mO, mLSE, sO, seqlen,
            gmem_tiled_copy_O, None, tiled_mma_pv, tidx, m_block, num_head, batch_size
        )

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """
        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size))
        acc_S = cute.make_fragment(acc_shape_S, Float32)
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
            smem_copy_params.tSsK[None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0],
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
        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = cute.make_tensor(rP.iterator, utils.convert_layout_acc_frgA(rP.layout))
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_O, tOrP, mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0],
            smem_copy_params.smem_thr_copy_V,
            # hook_fn=load_K_next,
        )
        # if const_expr(self.num_stages > 1):
        #     load_K_next()


class FlashAttentionForwardSm90(FlashAttentionForwardBase):

    arch = 90

    def __init__(self, *args, intra_wg_overlap: bool = True, mma_pv_is_rs: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs

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
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.n_block_size
                ),
                self.dtype
            )
        else:
            sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.m_block_size // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.n_block_size),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.m_block_size // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.head_dim_v_padded),
            a_source=warpgroup.OperandSource.RMEM if self.mma_pv_is_rs else warpgroup.OperandSource.SMEM,
        )
        tiled_mma_pv_rs = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.m_block_size // 64, 1, 1),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.head_dim_v_padded),
            a_source=warpgroup.OperandSource.RMEM
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs

    def _get_shared_storage_cls(self):
        # If we use cp.async to load Q, we want sQ to align to 1024 bytes
        sQ_alignment = 128 if const_expr(self.use_tma_Q) else 1024
        sK_alignment = 128
        sV_alignment = 128
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], alignment]
            for layout, alignment in zip(
                    (self.sQ_layout, self.sK_layout, self.sV_layout),
                    (sQ_alignment, sK_alignment, sV_alignment)
            )
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
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

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        self._check_type(
            *(t.element_type if t is not None else None
              for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK))
        )
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (*(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]), t.stride[-1])
        mQ, mK, mV, mO = [cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t))) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=QO_layout_transpose))
            for t in (mQ, mO)
        ]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)) if const_expr(mLSE is not None) else None
        tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_mma_warp_groups = self.num_mma_threads // self.num_threads_per_warp_group
        self.num_threads = self.num_threads_per_warp_group * (self.num_mma_warp_groups + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_mma_threads  # If not TMA_Q, MMA threads load Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs = (
            256
            if self.num_mma_warp_groups == 1
            else (240 if self.num_mma_warp_groups == 2 else 160)
        )
        self.num_producer_regs = (
            56 if self.num_mma_warp_groups == 1 else (24 if self.num_mma_warp_groups == 2 else 32)
        )
        # self.num_mma_regs = 232
        # self.num_producer_regs = 40
        self.use_scheduler_barrier = (self.num_mma_warp_groups >= 2 and self.head_dim_padded <= 128) if const_expr(self.intra_wg_overlap) else (self.num_mma_warp_groups == 2)
        self.use_tma_Q = self.arch >= 90 and not (self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0)
        self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None and mSeqUsedQ is None and not self.pack_gqa
        # TODO: rescale_O_before_gemm
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        if const_expr(self.pack_gqa):
            shape_Q_packed = ((self.qhead_per_kvhead, mQ.shape[0]), mQ.shape[1], mK.shape[2], *mQ.shape[3:])
            stride_Q_packed = ((mQ.stride[2], mQ.stride[0]), mQ.stride[1], mQ.stride[2] * self.qhead_per_kvhead, *mQ.stride[3:])
            mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))
            shape_O_packed = ((self.qhead_per_kvhead, mO.shape[0]), mK.shape[1], mK.shape[2], *mO.shape[3:])
            stride_O_packed = ((mO.stride[2], mO.stride[0]), mO.stride[1], mO.stride[2] * self.qhead_per_kvhead, *mO.stride[3:])
            mO = cute.make_tensor(mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed))
            if const_expr(mLSE is not None):
                shape_LSE_packed = ((self.qhead_per_kvhead, mLSE.shape[0]), mK.shape[2], *mLSE.shape[2:])
                stride_LSE_packed = ((mLSE.stride[1], mLSE.stride[0]), mLSE.stride[1] * self.qhead_per_kvhead, *mLSE.stride[2:])
                mLSE = cute.make_tensor(mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed))

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_q_bytes = cute.size_in_bytes(mQ.element_type, cute.select(self.sQ_layout, mode=[0, 1]))
        self.tma_copy_k_bytes = cute.size_in_bytes(mK.element_type, cute.select(self.sK_layout, mode=[0, 1]))
        self.tma_copy_v_bytes = cute.size_in_bytes(mV.element_type, cute.select(self.sV_layout, mode=[0, 1]))
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_Q, mQ, self.sQ_layout, (self.m_block_size, self.head_dim_padded), # No mcast
            )
        else:
            tma_atom_Q, tma_tensor_Q = None, None
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
            1  # No mcast for now
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_v_padded),
            1  # No mcast for now
        )
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_O, mO, self.sO_layout, (self.m_block_size, self.head_dim_v_padded), # No mcast
            )
        else:
            tma_atom_O = None
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler if const_expr(not self.is_causal or self.is_local) else SingleTileLPTScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.m_block_size),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) if const_expr(mCuSeqlensQ is not None) else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.m_block_size, self.n_block_size),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        # If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        # Right after this, we multiply by log2(e) before applying exp2.
        # To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        # (assigning it to softcap_val) and pre-multiply softcap_val * log2(e)
        # (assigning it to softmax_scale_log2).
        LOG2_E = math.log2(math.e)
        if const_expr(softcap is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = None
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = Float32(softmax_scale / softcap)
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)
        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softcap_val,
            window_size_left,
            window_size_right,
            learnable_sink,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_pv_rs,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softcap_val: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            if const_expr(tma_atom_Q is not None):
                cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(self.use_tma_O):
                cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier init
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            # if tidx < 2:
            #     # barrierO num threads should be self.num_mma_threads
            #     cute.arch.mbarrier_init(mbar_ptr_Q + tidx, 1 if tidx == 0 else self.num_mma_threads)
            cute.arch.mbarrier_init(mbar_ptr_Q, 1 if const_expr(self.use_tma_Q) else self.num_Q_load_threads)
            # cute.arch.mbarrier_init(mbar_ptr_Q + 1, self.num_mma_threads)
        # We rely on pipeline_k and pipeline_v to initialize the mbarrier fence and sync
        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // self.num_threads_per_warp_group
        )
        pipeline_k = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_k_bytes,
            init_wait=False,
        )
        pipeline_v = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_v_bytes,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        # TODO: how to get sQ_pi for cp.async if pack_gqa?
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type)
        # Transpose view of V to tensor with layout (head_dim_v, n_block_size) for tiled mma
        sVt = utils.transpose_view(sV)
        if const_expr(sP_layout is not None):
            sP_pi = storage.sP.get_tensor(sP_layout)
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        else:
            sP, sP_pi = None, None
        # reuse sQ's data iterator
        sO_pi = storage.sQ.get_tensor(sO_layout)
        # TODO: idk why not using sO_pi is faster
        sO = cute.make_tensor(cute.recast_ptr(sO_pi.iterator, sO_layout.inner, dtype=sO_pi.element_type), sO_layout.outer)

        block_info = BlockInfo(
            self.m_block_size, self.n_block_size, self.is_causal, self.is_local,
            window_size_left, window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK, seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ, mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask, self.m_block_size, self.n_block_size,
            window_size_left=window_size_left, window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:  # Producer
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        else:  # Consumer
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_mma_pv_rs,
                mQ,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softcap_val,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQ_cur = mQ[None, None, head_idx, batch_idx]
                else:
                    offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    mQ_cur = cute.domain_offset((offset, 0), mQ[None, None, head_idx])
                head_idx_kv = head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                else:
                    mK_cur, mV_cur = [cute.domain_offset((seqlen.offset_k, 0), t[None, None, head_idx_kv]) for t in (mK, mV)]
                gK = cute.local_tile(mK_cur, (self.n_block_size, self.head_dim_padded), (None, 0))
                gV = cute.local_tile(mV_cur, (self.n_block_size, self.head_dim_v_padded), (None, 0))
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.head_dim_padded), (m_block, 0))
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
                load_K = partial(self.load_K, tma_atom_K, tKgK, tKsK, pipeline_k)
                load_V = partial(self.load_K, tma_atom_V, tVgV, tVsV, pipeline_v)
                # load_Q
                if const_expr(self.use_tma_Q):
                    # TODO: wait for Q to be empty
                    q_producer_phase ^= 1
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_Q, self.tma_copy_q_bytes)
                    cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=mbar_ptr_Q)
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
                # if cute.arch.thread_idx()[0] == 0:
                #     cute.printf("m_block = %d, n_block_min: %d, n_block_max: %d", m_block, n_block_min, n_block_max)
                for i in cutlass.range(n_block_max - n_block_min, unroll=2):
                    n_block = n_block_max - i - 1
                    load_K(n_block, producer_state=kv_producer_state)
                    load_V(n_block, producer_state=kv_producer_state)
                    kv_producer_state.advance()
                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop


    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        # softmax: Softmax,
        # acc_O: cute.Tensor,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softcap_val: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        if const_expr(self.mma_pv_is_rs):
            acc_S_shape = tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size))
            tOrP = cute.make_fragment(
                utils.convert_layout_acc_frgA(cute.make_layout(acc_S_shape)), self.dtype
            )
        else:
            tOrP = tiled_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sP))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
        # tPsP = smem_thr_copy_P.partition_D(sP_pi) if const_expr(sP_pi is not None) else None
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        # if cute.arch.thread_idx()[0] == 0:
        #     cute.printf(sP_pi.layout, sP_pi.iterator)
        #     cute.printf(sP.layout, sP.iterator)
        #     cute.printf(tPsP.layout, tPsP.iterator)

        self.mma_init()

        acc_shape_O = tiled_mma_pv.partition_shape_C((self.m_block_size, self.head_dim_v_padded))
        acc_O = cute.make_fragment(acc_shape_O, Float32)
        # group parameters for mma_one_n_block
        mma_params = SimpleNamespace(tSrQ=tSrQ, tSrK=tSrK, tOrP=tOrP, tOrVt=tOrVt, acc_O=acc_O)
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap if const_expr(self.intra_wg_overlap) else self.mma_one_n_block,
            tiled_mma_qk=tiled_mma_qk, tiled_mma_pv=tiled_mma_pv, tiled_mma_pv_rs=tiled_mma_pv_rs,
            pipeline_k=pipeline_k, pipeline_v=pipeline_v,
            mma_params=mma_params, smem_copy_params=smem_copy_params,
            check_inf=True,
        )

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
        # if work_tile.is_valid_tile:
            # Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
            # -inf to e.g. -50.0, which can affect the attention softmax.
            def scoremod_premask_fn(acc_S):
                if const_expr(softcap_val is not None):
                    acc_S.store(cute.math.tanh(acc_S.load() * softcap_val, fastmath=True))

            # shape: (atom_v_m * rest_m)
            softmax = Softmax(softmax_scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1])
            mma_one_n_block = partial(
                mma_one_n_block_all, softmax=softmax, scoremod_premask_fn=scoremod_premask_fn
            )

            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask, m_block=m_block, thr_mma=thr_mma_qk,
                mask_causal=self.is_causal, mask_local=self.is_local,
            )
            softmax.reset()
            # Load Q if not TMA_Q
            if const_expr(not self.use_tma_Q):
                pack_gqa = PackGQA(self.m_block_size, self.head_dim_padded, self.check_hdim_oob, self.qhead_per_kvhead)
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQ_cur = mQ[None, None, head_idx, batch_idx]
                else:
                    offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                    mQ_cur = cute.domain_offset((offset, 0), mQ[None, None, head_idx])
                # gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
                # gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.head_dim_padded), (m_block, 0))
                # self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q,
                #             headdim=mQ.shape[1])
                pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ptr_Q)

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
            q_consumer_phase ^= 1
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
            # We also need masking on S if it's causal, for the last several blocks.
            O_should_accumulate = False
            # First iteration with seqlen masking
            if const_expr(self.intra_wg_overlap):
                acc_S = cute.make_fragment(
                    tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size)), Float32
                )
                pipeline_k.consumer_wait(kv_consumer_state)
                sm90_utils.gemm(
                    tiled_mma_qk, acc_S, tSrQ, tSrK[None, None, None, kv_consumer_state.index],
                    zero_init=True, wg_wait=0
                )
                pipeline_k.consumer_release(kv_consumer_state)
                scoremod_premask_fn(acc_S)
                # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
                mask_fn(acc_S, n_block=n_block_max - 1, mask_seqlen=True)
                # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
                softmax.online_softmax(acc_S, is_first=True)
                tOrP_acc = cute.make_tensor(acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout))
                tOrP = mma_params.tOrP if const_expr(self.mma_pv_is_rs) else cute.make_fragment_like(tOrP_acc, self.dtype)
                # tOrP.store(tOrP_acc.load().to(self.dtype))
                # the "to(self.dtype)" conversion fails to vectorize for block sizes other
                # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
                # 2 elements. So we just call ptx directly.
                utils.cvt_f16(tOrP_acc, tOrP)
                if const_expr(not self.mma_pv_is_rs):
                    tPrP = smem_thr_copy_P.retile(tOrP)
                    cute.copy(smem_thr_copy_P, tPrP, tPsP)
                    # Fence and barrier to make sure smem store is visible to WGMMA
                    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                    cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
                # Need to initialize tOrO in the case of RescaleOBeforeGemm where we will scale tOrO even in the 1st iter
                # acc_O.fill(0.0)
            else:
                self.warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    n_block_max - 1, kv_consumer_state,
                    is_first_n_block=True, mask_fn=partial(mask_fn, mask_seqlen=True),
                    O_should_accumulate=False
                )
                O_should_accumulate = True
            # if cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block_max = {}, n_block_min = {}", m_block, n_block_max, n_block_min)
            n_block_max -= 1
            # Next couple of iterations with causal masking
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_causal_local_mask = {}", n_block_min_causal_local_mask)
                for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    kv_consumer_state = mma_one_n_block(
                        n_block, kv_consumer_state, mask_fn=partial(mask_fn, mask_seqlen=False),
                        O_should_accumulate=O_should_accumulate
                    )
                    O_should_accumulate = True
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
            # The remaining iterations have no masking
            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
            for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                n_block = n_block_max - 1 - n_tile
                kv_consumer_state = mma_one_n_block(n_block, kv_consumer_state, check_inf=True, O_should_accumulate=O_should_accumulate)
                O_should_accumulate = True
            # Separate iterations with local masking on the left
            if const_expr(self.is_local and block_info.window_size_left is not None):
                n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    kv_consumer_state = mma_one_n_block(
                        n_block, kv_consumer_state,
                        check_inf=True, mask_fn=partial(mask_fn, mask_seqlen=False),
                        O_should_accumulate=O_should_accumulate
                    )
                    O_should_accumulate = True
            # Last "half" iteration
            if const_expr(self.intra_wg_overlap):
                pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
                sm90_utils.gemm(
                    tiled_mma_pv, mma_params.acc_O, mma_params.tOrP,
                    mma_params.tOrVt[None, None, None, kv_consumer_state.index],
                    zero_init=not O_should_accumulate, wg_wait=-1
                )
                warpgroup.wait_group(0)
                pipeline_v.consumer_release(kv_consumer_state)
                kv_consumer_state.advance()
            else:
                self.warp_scheduler_barrier_arrive()

            # normalize acc_O by row_sum and calculate the lse
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_fragment_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
                    tScS_mn = utils.make_acc_tensor_mn_view(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.m_block_size + tScS_mn[r][0]
                        q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
            else:
                sink_val = None

            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue(
                acc_O, softmax.row_sum, mO, mLSE, sO, seqlen,
                gmem_tiled_copy_O, tma_atom_O, tiled_mma_pv, tidx, m_block, head_idx, batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        O_should_accumulate: cutlass.Boolean = True,
    ):
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size)), Float32
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
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
        tOrP_acc = cute.make_tensor(acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout))
        tOrP = mma_params.tOrP if const_expr(self.mma_pv_is_rs) else cute.make_fragment_like(tOrP_acc, self.dtype)
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        utils.cvt_f16(tOrP_acc, tOrP)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        sm90_utils.gemm(
            tiled_mma_pv, mma_params.acc_O, mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read.index],
            zero_init=not O_should_accumulate, wg_wait=0
        )
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        n_block: Int32,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        O_should_accumulate: cutlass.Boolean = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size)), Float32
        )
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        sm90_utils.gemm(
            tiled_mma_qk, acc_S, mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1
        )
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        sm90_utils.gemm(
            tiled_mma_pv, mma_params.acc_O, mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read_v.index],
            zero_init=not O_should_accumulate, wg_wait=-1
        )
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)
        scoremod_premask_fn(acc_S)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = cute.make_tensor(acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout))
        tOrP = mma_params.tOrP if const_expr(self.mma_pv_is_rs) else cute.make_fragment_like(tOrP_acc, self.dtype)
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        utils.cvt_f16(tOrP_acc, tOrP)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) - 1 + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_mma_warp_groups in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_mma_warp_groups == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_mma_warp_groups
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    # @cute.jit
    def load_K(
        self,
        tma_atom: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineAsync,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
    ):
        # TODO: mcast
        # TODO check warp_idx if we have 128 producer threads
        pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tKgK[None, block],
            tKsK[None, producer_state.index],
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state)
        )
