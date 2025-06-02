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
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils.ampere_helpers as sm80_utils

from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax
from flash_attn.cute.seqlen_info import SeqlenInfo


class FlashAttentionForwardSm80:
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
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
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
        smem_capacity = sm80_utils.SMEM_CAPACITY["sm80"]
        if smem_usage > smem_capacity:
            return False
        # Check if twice the block size is divisible by the number of threads
        if (m_block_size * 2) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        sQ_layout_atom = utils.smem_layout_atom_sm80(self.head_dim_padded, self.dtype)
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.m_block_size, self.head_dim_padded), (0, 1),
        )
        sK_layout_atom = sQ_layout_atom
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.n_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2),
        )
        sV_layout_atom = utils.smem_layout_atom_sm80(self.head_dim_v_padded, self.dtype)
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom, (self.n_block_size, self.head_dim_v_padded, self.num_stages), (0, 1, 2),
        )
        sO_layout_atom = sV_layout_atom
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom, (self.m_block_size, self.head_dim_v_padded), (0, 1),
        )

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
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        tQK_layout = cute.make_ordered_layout(
            (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.m_block_size % tQK_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0),
        )
        # TODO: need a different layout for O if O dtype is not the same as V dtype
        # tO_layout: thread layout for O store
        tO_layout = tV_layout
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
        """Configures and launches the flash attention v2 kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(seqlen_q * num_head * head_dim, num_head * head_dim, head_dim, 1)

        Prepares the shared memory layout, tiled copy atoms, tiled mma and shared memory storage.
        Then launches the kernel function with the prepared parameters.
        """
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(mQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mLSE is not None and mLSE.element_type not in [cutlass.Float32]):
            raise TypeError("LSE tensor must be Float32")
        assert mQ.element_type == self.dtype

        self._setup_attributes()

        @cute.struct
        class SharedStorageQKV:
            sV: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)], 1024
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)], 1024
            ]

        cosize_sQV = utils.max_constexpr(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))

        @cute.struct
        class SharedStorageSharedQV:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cosize_sQV], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)], 1024
            ]

        SharedStorage = SharedStorageQKV
        if cutlass.const_expr(self.Q_in_regs):
            SharedStorage = SharedStorageSharedQV

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled mma
        # ///////////////////////////////////////////////////////////////////////////////
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

        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self.m_block_size),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[0]),
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

        n_block_max = cute.ceil_div(mK.shape[1], self.n_block_size)
        if self.is_causal:
            n_block_max = min(
                cute.ceil_div((m_block + 1) * self.m_block_size + mK.shape[1] - mQ.shape[1], self.n_block_size),
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
        gQ = cute.local_tile(mQ[batch_size, None, num_head, None], blkQ_shape, (m_block, 0))
        # (n_block_size, head_dim, n_block)
        num_head_kv = num_head // self.qhead_per_kvhead
        gK = cute.local_tile(mK[batch_size, None, num_head_kv, None], blkK_shape, (None, 0))
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(mV[batch_size, None, num_head_kv, None], blkV_shape, (None, 0))

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
        sVt = cute.composition(
            sV,
            cute.make_ordered_layout((self.head_dim_v_padded, self.n_block_size, self.num_stages), order=(1, 0, 2)),
        )

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
        tKpK = utils.predicate_k(tKcK, limit=mK.shape[3])
        if cutlass.const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = utils.predicate_k(tVcV, limit=mV.shape[3])

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
        seqlen = SeqlenInfo(seqlen_q=mQ.shape[1], seqlen_k=mK.shape[1])
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
                    headdim=mQ.shape[3])
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
            compute_one_n_block(n_block - n_tile - 1, smem_pipe_read, smem_pipe_write, check_inf=True)
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
        utils.gemm_sm80(
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
        tOrS = cute.make_tensor(rP.iterator, utils.convert_layout_acc_frgA(rP.layout))
        if cutlass.const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        utils.gemm_sm80_rs(
            mma_params.thr_mma_pv, mma_params.acc_O, tOrS, mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, smem_pipe_read if self.num_stages > 1 else 0],
            smem_copy_params.smem_thr_copy_V,
            # hook_fn=load_K_next,
        )
        # if cutlass.const_expr(self.num_stages > 1):
        #     load_K_next()

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
        cute.arch.barrier()  # make sure all threads have finished reading V
        # smem copy atom for O
        smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        smem_thr_copy_O = utils.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))

        # Write LSE from rmem -> gmem
        if cutlass.const_expr(mLSE is not None):
            gLSE = cute.local_tile(mLSE[batch_size, num_head, None], (self.m_block_size,), (m_block,))
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
            if taccOcO[0, 0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                    if cute.elem_less(t0accOcO[m, 0][0], mO.shape[1] - m_block * self.m_block_size - taccOcO[0][0]):
                        taccOgLSE[m, 0] = lse[m]

        gO = cute.local_tile(
            mO[batch_size, None, num_head, None],
            (self.m_block_size, self.head_dim_v_padded),
            (m_block, 0),
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self.dtype)
        # sync before all smem stores are done.
        cute.arch.barrier()
        # load acc O from smem to rmem for wider vectorization
        cute.autovec_copy(tOsO, tOrO)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[3])
        # copy acc O from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            # if cute.elem_less(tOcO[0, rest_m, 0][0], mO.shape[1] - m_block * self.m_block_size):
            if cute.elem_less(t0OcO[0, rest_m, 0][0], mO.shape[1] - m_block * self.m_block_size - tOcO[0][0]):
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
