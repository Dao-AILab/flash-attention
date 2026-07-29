# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of
# https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm80.h
# and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h
# from Cutlass C++ to Cute-DSL.
# Built on Cute-DSL example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils as utils_basic
from cutlass.cutlass_dsl import BaseDSL
from cutlass.base_dsl.arch import Arch

from quack import copy_utils
from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax, apply_score_mod_inner
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.block_sparsity import BlockSparseTensors
from cutlass.cute import FastDivmodDivisor
from flash_attn.cute.block_sparse_utils import (
    run_block_sparse_mainloop_sm80,
    get_curr_blocksparse_tensors,
    sparse_tensor_m_block,
)
from flash_attn.cute.tile_scheduler import SingleTileScheduler, SingleTileVarlenScheduler, TileSchedulerArguments
from flash_attn.cute.utils import AuxData


class FlashAttentionForwardBase:

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        q_subtile_factor: int | None = None,
        pack_gqa_all_rows_valid: bool = False,
        pack_gqa_fast_valid_rows: bool = False,
        skip_dense_seqlen_mask: bool = False,
        hook_load_k: bool = False,
        hook_load_v: bool = False,
        static_causal_blocks: bool = False,
        is_split_kv: bool = False,
        num_splits: int = 1,
    ):
        """Initializes the configuration for a flash attention kernel.

        All contiguous dimensions must be at least 16 bytes aligned, which means that the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param tile_n: n block size
        :type tile_n: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :param score_mod: A callable that takes the attention scores and applies a modification.
            Callable signature: ``score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, aux_tensors) -> Any``
        :param mask_mod: A callable that takes the attention scores and returns a boolean representing whether that score should be masked.
            Callable signature: ``mask_mod(batch_idx, head_idx, q_idx, kv_idx, aux_tensors) -> Boolean``
        """
        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.pack_gqa_all_rows_valid = pack_gqa_all_rows_valid
        self.pack_gqa_fast_valid_rows = pack_gqa_fast_valid_rows
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.q_subtile_factor = q_subtile_factor
        self.Q_in_regs = Q_in_regs
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.skip_dense_seqlen_mask = skip_dense_seqlen_mask
        self.hook_load_k = hook_load_k
        self.hook_load_v = hook_load_v
        self.static_causal_blocks = static_causal_blocks
        self.is_split_kv = is_split_kv
        self.num_splits = num_splits
        self.qk_acc_dtype = Float32
        self.score_vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        if self.score_vec_size > 2:
            raise ValueError(
                f"score_mod vec_size {self.score_vec_size} not supported on Sm80/90/120 "
                "due to accumulator thread ownership pattern."
            )
        self.mask_vec_size: cutlass.Constexpr = getattr(mask_mod, "__vec_size__", 1)
        if self.mask_vec_size > 1:
            raise ValueError(
                f"mask_mod vec_size {self.mask_vec_size} not supported on Sm80/90/120 "
                "due to accumulator thread ownership pattern."
            )
        self.arch = BaseDSL._get_dsl().get_arch_enum()

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param tile_n: n block size
        :type tile_n: int
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
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage_Q = tile_m * head_dim * 2
        smem_usage_K = tile_n * head_dim * num_stages * 2
        smem_usage_V = tile_n * head_dim_v * num_stages * 2
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
        # TODO: sm86 and sm89
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False
        # Check if twice the block size is divisible by the number of threads
        if (tile_m * 2) % num_threads != 0:
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
        # Get the data type and check if it is fp16 or bf16.  SplitKV writes a
        # float32 partial output (out_partial), so mO is allowed to be fp32
        # while Q/K/V remain fp16/bf16.
        if const_expr(self.is_split_kv):
            if const_expr(not (mQ_type == mK_type == mV_type)):
                raise TypeError("Q/K/V must have the same data type")
            if const_expr(mO_type != Float32):
                raise TypeError("SplitKV partial output must be Float32")
        elif const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
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
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom = (
            self._get_smem_layout_atom()
        )
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self.tile_m, self.tile_hdim),
            (0, 1),
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            (self.tile_n, self.tile_hdim, self.num_stages),
            (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.tile_n, self.tile_hdimv, self.num_stages),
            (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom,
            (self.tile_m, self.tile_hdimv),
            (0, 1),
        )
        if const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom,
                (self.tile_m, self.tile_n),
                (0, 1),
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
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQ_layout and tK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0, (
            "num_threads must be divisible by tQK_shape_dim_1"
        )
        assert self.num_producer_threads % tQK_shape_dim_1 == 0, (
            "num_threads must be divisible by tQK_shape_dim_1"
        )
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.tile_m % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        # TODO: need a different layout for O if O dtype is not the same as V dtype
        # tO_layout: thread layout for O store
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.tile_m % tO_layout.shape[0] == 0

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
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
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
        split_idx: Int32 = 0,
    ):
        # SplitKV writes the fp32 partial output (out_partial) directly from
        # registers to gmem, bypassing the bf16-sized smem O buffer (which is
        # aliased onto sQ and could not hold fp32 without doubling smem).  The
        # smem roundtrip below is only for the packed dtype (fp16/bf16) output.
        if const_expr(not self.is_split_kv):
            # store acc_O
            rO = cute.make_fragment_like(acc_O, self.dtype)
            rO.store(acc_O.load().to(self.dtype))
            # Make sure all threads have finished reading V
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads
            )
            # SM80/SM120 use SM80 MMA (m16n8k16) whose register layout is incompatible
            # with the SM90 stmatrix path get_smem_store_atom picks for arch >= 90;
            # force universal copy for them. SM90 keeps stmatrix (matches WGMMA layout).
            arch_int = self.arch.major * 10 + self.arch.minor
            store_atom_arch = 80 if arch_int // 10 in [8, 12] else arch_int
            smem_copy_atom_O = utils.get_smem_store_atom(store_atom_arch, self.dtype)
            smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            # taccOsO = copy_utils.partition_D_position_independent(smem_thr_copy_O, sO)
            # copy acc O from rmem to smem with the smem copy atom
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            # SplitKV: mLSE is (s, h, b, split) [non-varlen] or
            # (total_q, h, split) [varlen]; index batch (via offset_batch_Q),
            # then head and split.  Non-split: (s, h, b) -> select head.
            if const_expr(self.is_split_kv):
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx, split_idx]
            else:
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
            if const_expr(self.is_split_kv and self.pack_gqa):
                # SplitKV partial LSE: mLSE_cur keeps composite mode 0
                # (qhead_per_kvhead, seqlen_q); scatter packed rows to their
                # physical (h_idx, m_idx) slots so the (unpacked-layout) combine
                # reads them correctly.
                pack_gqa.store_LSE_partial(
                    mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q
                )
            elif const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
                # Only the thread corresponding to column 0 writes out the lse to gmem
                if taccOcO[0][1] == 0:
                    for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                if const_expr(self.pack_gqa_all_rows_valid):
                    if const_expr(self.pack_gqa_fast_valid_rows):
                        pack_gqa.store_LSE(
                            mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q,
                            all_rows_valid=True
                        )
                    else:
                        pack_gqa.store_LSE_all_rows_valid(
                            mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q
                        )
                else:
                    pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        ragged = self.use_tma_O and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
        # SplitKV: mO is (s, d, h, b, split) [non-varlen] or (total_q, d, h, split)
        # [varlen]; index batch, then head and split.  Non-split: (s, d, h, b).
        if const_expr(self.is_split_kv):
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[None, None, head_idx, split_idx]
        else:
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[None, None, head_idx]
        # thr_mma = tiled_mma.get_slice(tidx)
        # taccOgO = thr_mma.partition_C(gO)
        # cute.autovec_copy(rO, taccOgO)
        # sync to make sure all smem stores are done
        if const_expr(self.is_split_kv):
            # Direct fp32 register -> gmem store of the partial output, using
            # the MMA accumulator's partition_C layout (same as acc_O) so no
            # smem roundtrip / type conversion is needed.  reshape_acc_to_mn
            # gives a 2D (M, N) view; predicate rows by seqlen_q and columns by
            # head_dim_v via the identity-tensor coordinates (matches the LSE
            # write above and the SM100 split epilogue).
            acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
            if const_expr(self.pack_gqa):
                # SplitKV partial O under pack_gqa: mO_cur keeps composite mode 0
                # (qhead_per_kvhead, seqlen_q).  cute.local_tile cannot decompose
                # the packed row to its physical (h_idx, m_idx) slot, so scatter
                # the fp32 MMA accumulator directly via the composite stride
                # (same mapping as store_O/compute_ptr).  No smem roundtrip.
                pack_gqa.store_O_partial(
                    mO_cur, acc_O_mn, tiled_mma, tidx, m_block, seqlen.seqlen_q, mO.shape[1]
                )
            else:
                gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgO_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gO))
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
                for m in cutlass.range_constexpr(cute.size(taccOgO_mn.shape[0])):
                    if (
                        t0accOcO[m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                    ):
                        for n in cutlass.range_constexpr(cute.size(taccOgO_mn.shape[1])):
                            if const_expr(not self.check_hdim_v_oob):
                                taccOgO_mn[m, n] = acc_O_mn[m, n]
                            elif taccOcO[0, n][1] < mO.shape[1]:
                                taccOgO_mn[m, n] = acc_O_mn[m, n]
        elif const_expr(self.use_tma_O):
            # ensure smem writes are visible to TMA
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOrO = cute.make_fragment_like(tOsO, self.dtype)
            # load acc O from smem to rmem for wider vectorization
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                # copy acc O from rmem to gmem
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                if const_expr(self.pack_gqa_all_rows_valid):
                    if const_expr(self.pack_gqa_fast_valid_rows):
                        pack_gqa.store_O(
                            mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q,
                            all_rows_valid=True
                        )
                    else:
                        pack_gqa.store_O_all_rows_valid(
                            mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q
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
        cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=headdim)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
            # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            if t0QcQ[0, m, 0][0] < seqlen - block * self.tile_m - tQcQ[0][0]:
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
        is_even_n_smem_k = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_k):
            # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
            # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
            if const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.tile_n
            else:
                if const_expr(not need_predicates):
                    seqlen_limit = self.tile_n
                else:
                    seqlen_limit = cutlass.min(seqlen - block * self.tile_n, self.tile_n)
            seqlen_limit -= tKcK[0][0]
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[
                            None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0
                        ],
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
        is_even_n_smem_v = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
                if (
                    is_even_n_smem_v
                    or n < cute.size(tVsV.shape[1]) - 1
                    or tVcV[0, n, 0][0] < self.tile_n
                ):
                    predicate = tVpV[None, n, None] if const_expr(self.check_hdim_v_oob) else None
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.tile_n - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                                predicate[i, k] = (
                                    tVpV[i, n, k] if const_expr(self.check_hdim_v_oob) else True
                                ) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[
                            None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0
                        ],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
            )

    # Paged-KV variants. Same call signature as non-paged load_K/load_V above
    # so the mainloop is unchanged. load_K refreshes the page-table register
    # fragment for n_block; load_V on the same n_block reuses those indices.
    # need_predicates is ignored — PagedKVManager.load_KV bounds reads internally.


class FlashAttentionForwardSm80(FlashAttentionForwardBase):
    def _get_smem_layout_atom(self):
        sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdimv)
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
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        # Only the sm_120 specialization (FlashAttentionForwardSm120 /
        # ...Sm120Tma) supports a learnable sink in this SM80-base kernel. Real
        # SM80 rejects it exactly as main did, which also keeps the softmax
        # row_max_safe sink path unreachable on SM80. NOTE: the sm120 forward
        # forces self.arch = Arch.sm_80, so the backward's `arch == 120` idiom
        # does not work here; gate on the is_sm120 marker instead.
        assert (
            learnable_sink is None or getattr(self, "is_sm120", False)
        ), "Learnable sink is not supported in this kernel"
        self._check_type(
            *(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK))
        )
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        # self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None
        # The SM80 base class never constructs tma_atom_O (it passes None to
        # self.epilogue). Restrict TMA-O to sm_90..sm_119; on SM120 (self.arch
        # is forced to sm_80 in FlashAttentionForwardSm120.__init__, and the real
        # DSL arch is sm_12x) this evaluates False, falling back to the register
        # -> gmem O store the CpAsync SM120 kernel expects. (upstream #2656)
        self.use_tma_O = Arch.sm_90 <= self.arch < Arch.sm_120
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Layout permutation: 4D non-varlen vs 3D varlen
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=QO_layout_transpose))
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        # SplitKV: mO is the 5D out_partial (num_splits, b, s, h, d) and mLSE
        # the 4D lse_partial (num_splits, b, s, h) [or (num_splits, h, total_q)
        # for varlen].  Reorder so seqlen leads, head and split are selectable.
        # Mirrors flash_fwd_sm100.py: O select [2,4,3,1,0] -> (s, d, h, b, split),
        # LSE select [3,2,1,0] -> (s, h, b, split).
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
        else:
            O_layout_transpose = QO_layout_transpose
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        if const_expr(mLSE is not None):
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
        # Fold qhead_per_kvhead into the seqlen mode of mQ/mO/mLSE so the
        # mainloop iterates over KV heads with packed Q rows. Required for
        # the epilogue's pack_gqa.store_O strides to make sense.
        # sm120-only: gate on the is_sm120 marker (self.arch is forced to
        # Arch.sm_80 on the SM120 forward, so `arch == 120` does not work here).
        # Real SM80 keeps the unfolded layout and its own pack_gqa path.
        if const_expr(self.pack_gqa and getattr(self, "is_sm120", False)):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
        # TileScheduler for varlen, simple grid for non-varlen
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[3])
        )
        # When pack_gqa is True, mQ.shape[0] is a composite (qhead_per_kvhead,
        # seqlen_q) mode, so we use cute.size() to get the flat number of
        # packed rows; mQ.shape[2] is nheads_kv (not nheads_q).  Mirrors the
        # SM90 dispatch in flash_fwd_sm90.py:322-336.
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            # SplitKV: the SingleTileScheduler multiplies the head axis by
            # num_splits in get_grid_shape and divmods head_idx back into
            # (head_idx, split_idx) in get_current_work.  num_splits is a
            # compile-time Python int here (self.num_splits) so the grid shape
            # and the FastDivmodDivisor are static for this kernel variant.
            num_splits=self.num_splits if const_expr(self.is_split_kv) else 1,
            seqlen_k=cute.size(mK.shape[0]),
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors)

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            softmax_scale_log2,
            softmax_scale,
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
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.jit
    def compute_sink_val(
        self,
        learnable_sink: Optional[cute.Tensor],
        softmax: Softmax,
        m_block: Int32,
        head_idx: Int32,
        thr_mma_qk,
        split_idx: Int32 = Int32(0),
    ):
        """Per-row learnable-sink logit for softmax.finalize (mirrors SM90).

        Non-pack: head_idx is the query head -> a single scalar. Pack-GQA:
        head_idx is the KV head and each packed row maps to a different query
        head, so produce a per-row fragment shaped like softmax.row_max.

        SplitKV: the sink is a single virtual logit shared by every column, so
        it must be folded into the LSE/denominator EXACTLY ONCE across splits.
        Each split would otherwise add exp(sink) to its own row_sum, and the
        combine kernel reconstructs the final denominator as sum_s exp(LSE_s),
        which would count the sink num_splits times. We therefore apply it only
        in split 0 and suppress it (logit -> -inf, so exp2(-inf) == 0 in
        finalize) in every other split. With a single split this is a no-op.
        """
        if const_expr(learnable_sink is None):
            return None
        # Only split 0 carries the sink; suppress it in every other SplitKV split
        # by adding a runtime bias of 0 (split 0) or -inf (split>0). Adding (not
        # selecting) keeps the result Float32 and lets the -inf collapse the
        # exp2() term in softmax.finalize to 0. With a single split this is a
        # no-op. split_idx is a runtime value, so the choice is made at runtime.
        if const_expr(self.is_split_kv):
            suppress_bias = Float32(0.0) if split_idx == Int32(0) else -Float32.inf
        else:
            suppress_bias = Float32(0.0)
        if const_expr(not self.pack_gqa):
            sink_logit = Float32(learnable_sink[head_idx])
            return sink_logit + suppress_bias
        sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        for r in cutlass.range(cute.size(sink_val), unroll_full=True):
            row = m_block * self.tile_m + tScS_mn[r][0]
            q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
            sink_val[r] = Float32(learnable_sink[q_head_idx]) + suppress_bias
        return sink_val

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
        mPageTable: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
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
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, num_head, batch_size, split_idx = work_tile.tile_idx

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        # When pack_gqa is True, mQ.shape[0] is a composite (qhead_per_kvhead,
        # seqlen_q) mode produced by pack_gqa_layout in __call__.  The static
        # seqlen_q is the second sub-mode (shape[0][1]); shape[0] itself would
        # be the packed total qhead_per_kvhead * seqlen_q.
        seqlen_q_static = (
            mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
        )
        # When paged KV is enabled, mK has shape (page_size, d, h_k, num_pages)
        # after the KV_layout_transpose in __call__. The logical seqlen_k upper
        # bound is page_size * max_pages_per_seq, not page_size; mSeqUsedK gives
        # the true per-batch length and (if present) overrides this static value.
        seqlen_k_static = (
            mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1]
        )
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_size,
            seqlen_q_static=seqlen_q_static,
            seqlen_k_static=seqlen_k_static,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        if const_expr(self.static_causal_blocks and not self.is_split_kv):
            n_block_min, n_block_max = Int32(0), m_block + 1
        else:
            # SplitKV: get_n_block_min_max partitions [0, n_block_full_max) into
            # num_splits contiguous block ranges by split_idx; empty splits
            # (n_block_min >= n_block_max) run zero mainloop iterations so the
            # epilogue writes O=0 and LSE=-inf (softmax.finalize handles the
            # row_sum==0 case), which the combine kernel then drops.
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                m_block,
                split_idx,
                self.num_splits if const_expr(self.is_split_kv) else 1,
            )
        # For varlen, wasted grid tiles (where batch_idx >= num_batch) will have
        # seqlen_q=seqlen_k=0 and n_block_max=0.  Clamp to 0 so we don't use a
        # negative block index for K/V loads; the load/store predicates already
        # guard all memory accesses when seqlen is 0.
        n_block = cutlass.max(n_block_max - 1, 0)
        # SplitKV: a split with no assigned KV blocks must skip all compute and
        # fall straight through to the epilogue (which writes O=0, LSE=-inf so
        # the combine drops it).  For the non-split path has_work is always
        # True (a valid tile always has >= 1 block).
        if const_expr(self.is_split_kv):
            has_work = n_block_max > n_block_min
        else:
            has_work = True

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        if const_expr(getattr(self, "is_sm120", False)):
            # With pack_gqa, num_head iterates over KV heads (mQ.shape[2] is
            # nheads_kv) and equals head_idx_kv directly; without pack_gqa,
            # num_head iterates over all Q heads and we divide to get the KV head.
            num_head_kv = (
                num_head // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else num_head
            )
        else:
            num_head_kv = num_head // self.qhead_per_kvhead
        if const_expr(not seqlen.has_cu_seqlens_q):
            mQ_cur = mQ[None, None, num_head, batch_size]
        else:
            if const_expr(getattr(self, "is_sm120", False)):
                # Under pack_gqa, mode 0 of mQ is the composite (qhead_per_kvhead,
                # seqlen_q). A scalar token offset_q against that composite is
                # decomposed colexicographically by crd2idx (offset_q % qpkv,
                # offset_q // qpkv), which advances the base pointer by the wrong
                # amount for qpkv>1 and batch>0 -> garbage for varlen GQA seq>=1.
                # Offset the seqlen sub-mode only (matches the O/LSE offset_batch_Q
                # epilogue). MHA (qpkv=1) and batch 0 are unaffected.
                q_offset = (
                    ((None, seqlen.offset_q), 0)
                    if const_expr(self.pack_gqa)
                    else (seqlen.offset_q, 0)
                )
            else:
                q_offset = (seqlen.offset_q, 0)
            mQ_cur = cute.domain_offset(q_offset, mQ[None, None, num_head])
        # gK/gV are only used by the contiguous (non-paged) load path. For paged KV
        # the PagedKVManager indexes mK/mV directly via the page table.
        if const_expr(mPageTable is None):
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur = mK[None, None, num_head_kv, batch_size]
                mV_cur = mV[None, None, num_head_kv, batch_size]
            else:
                mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, num_head_kv])
                mV_cur = cute.domain_offset((seqlen.offset_k, 0), mV[None, None, num_head_kv])
            gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
            gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))
        else:
            gK = None
            gV = None
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))

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
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKsK = gmem_thr_copy_K.partition_D(sK)
        tVsV = gmem_thr_copy_V.partition_D(sV)
        if const_expr(mPageTable is None):
            tKgK = gmem_thr_copy_K.partition_S(gK)
            tVgV = gmem_thr_copy_V.partition_S(gV)
        else:
            tKgK = None
            tVgV = None

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
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
        cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.tile_hdim == self.tile_hdimv):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
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
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()

        # group parameters for compute_one_n_block
        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tOsVt=tOsVt,
        )
        if const_expr(mPageTable is None):
            load_K = partial(
                self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
                seqlen=seqlen.seqlen_k,
            )
            load_V = partial(
                self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                seqlen=seqlen.seqlen_k,
            )

            compute_one_n_block = partial(
                self.compute_one_n_block,
                mma_params=mma_params,
                smem_copy_params=smem_copy_params,
                softmax=softmax,
                load_K=load_K,
                load_V=load_V,
                score_mod=self.score_mod,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

        if const_expr(blocksparse_tensors is not None):
            # ///////////////////////////////////////////////////////////////////////////////
            # Block-sparse mainloop (SM80/SM120)
            # ///////////////////////////////////////////////////////////////////////////////
            qkv_factor = self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            subtile = self.q_subtile_factor if self.q_subtile_factor is not None else 1
            # SM80/SM120 don't support split-kv for block-sparse — sum mask + full
            # block counts directly. Calling get_total_block_count (which routes
            # through split_block_range -> cute.ceil_div with Int32 constants)
            # trips a DSL ICE: "cute.derefine ... explicitly marked illegal".
            # Mirror the simpler inline path used by run_block_sparse_mainloop_sm80
            # itself (block_sparse_utils.py:741+).
            bs_m_block = sparse_tensor_m_block(m_block, qkv_factor, subtile)
            (
                curr_mask_block_cnt,
                _,
                curr_full_block_cnt,
                _,
            ) = get_curr_blocksparse_tensors(
                batch_size, num_head, bs_m_block, blocksparse_tensors, seqlen,
            )
            total_block_cnt = curr_mask_block_cnt + curr_full_block_cnt

            bs_mask = AttentionMask(
                self.tile_m, self.tile_n, seqlen, window_size_left, window_size_right, qkv_factor
            )
            bs_mask_fn = partial(
                bs_mask.apply_mask,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=False,
                mask_local=False,
                aux_data=aux_data,
            )

            if total_block_cnt > 0:
                gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
                if const_expr(self.pack_gqa):
                    # See note above on pack_gqa_helper in the non-blocksparse path.
                    pack_gqa_helper = PackGQA(
                        self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                    )
                    if const_expr(self.pack_gqa_all_rows_valid):
                        if const_expr(self.pack_gqa_fast_valid_rows):
                            pack_gqa_helper.load_Q(
                                mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q,
                                all_rows_valid=True
                            )
                        else:
                            pack_gqa_helper.load_Q_all_rows_valid(
                                mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                            )
                    else:
                        pack_gqa_helper.load_Q(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            tidx,
                            m_block,
                            seqlen.seqlen_q,
                        )
                else:
                    self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block,
                                seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
                cute.arch.cp_async_commit_group()
                if const_expr(self.Q_in_regs):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                    tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                    cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)
                    cute.arch.barrier()
                else:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                mma_one_n_block = partial(
                    self.mma_one_n_block_bs,
                    mma_params=mma_params,
                    smem_copy_params=smem_copy_params,
                    softmax=softmax,
                    load_K=load_K,
                    load_V=load_V,
                    score_mod=self.score_mod,
                    batch_idx=batch_size,
                    head_idx=num_head,
                    m_block=m_block,
                    seqlen=seqlen,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                )

                run_block_sparse_mainloop_sm80(
                    blocksparse_tensors,
                    batch_size,
                    num_head,
                    m_block,
                    mma_one_n_block,
                    bs_mask_fn,
                    self.mask_mod,
                    fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                    qkv_factor,
                    subtile,
                )

            row_scale = softmax.finalize(
                sink_val=self.compute_sink_val(
                    learnable_sink, softmax, m_block, num_head, thr_mma_qk, split_idx
                ),
                is_sm120=getattr(self, "is_sm120", False),
            )
            softmax.rescale_O(acc_O, row_scale)
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            self.epilogue(
                acc_O, softmax.row_sum, mO, mLSE, sO, seqlen, gmem_tiled_copy_O,
                None, tiled_mma_pv, tidx, m_block, num_head, batch_size, split_idx,
            )

        if const_expr(blocksparse_tensors is None and mPageTable is None):
            # ///////////////////////////////////////////////////////////////////////////////
            # Prologue
            # ///////////////////////////////////////////////////////////////////////////////
            # Start async loads of the last mn-tile, where we take care of the mn residue
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            if const_expr(self.pack_gqa):
                # pack_gqa.load_Q computes per-row gmem pointers from the
                # packed mQ's composite (qhead_per_kvhead, seqlen) stride,
                # which the plain cp_async self.load_Q cannot do correctly
                # because cute.local_tile collapses adjacent qhead rows that
                # actually live at non-adjacent strides (qhead stride 64 vs
                # seqlen stride num_head*head_dim).
                pack_gqa_helper = PackGQA(
                    self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                )
                if const_expr(self.pack_gqa_all_rows_valid):
                    if const_expr(self.pack_gqa_fast_valid_rows):
                        pack_gqa_helper.load_Q(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q,
                            all_rows_valid=True
                        )
                    else:
                        pack_gqa_helper.load_Q_all_rows_valid(
                            mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                        )
                else:
                    pack_gqa_helper.load_Q(
                        mQ_cur,
                        sQ,
                        gmem_tiled_copy_Q,
                        tidx,
                        m_block,
                        seqlen.seqlen_q,
                    )
            else:
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
                        load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                    cute.arch.cp_async_commit_group()
                if const_expr(stage < self.num_stages - 1):
                    if stage == 0 or n_block - stage >= 0:
                        load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                    cute.arch.cp_async_commit_group()
            if const_expr(not self.Q_in_regs):
                preprocess_Q()

            # ///////////////////////////////////////////////////////////////////////////////
            # Mainloop
            # ///////////////////////////////////////////////////////////////////////////////
            # Start processing of the first n-block.
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
                window_size_left,
                window_size_right,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
            )

            # First iteration with seqlen masking, unless dense static noncausal
            # dispatch proved there is no K tail tile.
            smem_pipe_read = Int32(0)
            smem_pipe_write = Int32(self.num_stages - 1)
            # SplitKV: skip the unconditional first-block compute for an empty
            # split (no assigned blocks).  The masked/unmasked loops below are
            # already bounded by ranges that clamp to 0 trips for empty splits.
            if has_work:
                if const_expr(self.skip_dense_seqlen_mask):
                    compute_one_n_block(
                        n_block,
                        smem_pipe_read,
                        smem_pipe_write,
                        is_first_n_block=True,
                        seqlen=seqlen,
                    )
                else:
                    compute_one_n_block(
                        n_block,
                        smem_pipe_read,
                        smem_pipe_write,
                        is_first_n_block=True,
                        seqlen=seqlen,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                    )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            # Next couple of iterations with causal masking
            unmasked_n_block_start = n_block
            if const_expr(self.is_causal or self.is_local):
                if const_expr(self.static_causal_blocks):
                    n_block_min_causal_local_mask = m_block
                else:
                    n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                # The first n_block (n_block_max - 1) was already processed above
                # (is_first_n_block). For non-causal local with a right window that
                # reaches the seqlen boundary, get_n_block_min_causal_local_mask can
                # return a value >= n_block_max, which would make the unmasked loop
                # below re-process the first block (double-count) or read an
                # out-of-range block (NaN). Clamp the unmasked start to n_block_max-1.
                # (No-op for causal/causal-local, where it is already <= n_block_max-1;
                # cf. flash_fwd_sm90.py which caps n_block_max for the same reason.)
                unmasked_n_block_start = cutlass.min(n_block_min_causal_local_mask, n_block_max - 1)
                for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_local_mask, unroll=1):
                    n_block = n_block_max - 2 - n_tile
                    compute_one_n_block(
                        n_block,
                        smem_pipe_read,
                        smem_pipe_write,
                        seqlen=seqlen,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                    )
                    smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                    smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            # The remaining iterations have no masking
            unmasked_n_block_stop = n_block_min
            if const_expr(self.is_local):
                unmasked_n_block_stop = cutlass.min(
                    unmasked_n_block_start,
                    block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    ),
                )
            for n_tile in cutlass.range(unmasked_n_block_start - unmasked_n_block_stop, unroll=1):
                if const_expr(self.mask_mod is None):
                    compute_one_n_block(
                        unmasked_n_block_start - n_tile - 1,
                        smem_pipe_read,
                        smem_pipe_write,
                        seqlen=seqlen,
                        is_first_n_block=False,
                        check_inf=self.score_mod is not None,
                    )
                else:
                    compute_one_n_block(
                        unmasked_n_block_start - n_tile - 1,
                        smem_pipe_read,
                        smem_pipe_write,
                        seqlen=seqlen,
                        is_first_n_block=False,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                    )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            if const_expr(self.is_local):
                for n_tile in cutlass.range(unmasked_n_block_stop - n_block_min, unroll=1):
                    compute_one_n_block(
                        unmasked_n_block_stop - n_tile - 1,
                        smem_pipe_read,
                        smem_pipe_write,
                        seqlen=seqlen,
                        is_first_n_block=False,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                    )
                    smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                    smem_pipe_write = self.advance_pipeline(smem_pipe_write)

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax.finalize(
                sink_val=self.compute_sink_val(
                    learnable_sink, softmax, m_block, num_head, thr_mma_qk, split_idx
                ),
                is_sm120=getattr(self, "is_sm120", False),
            )
            softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            # reuse sQ's data iterator
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                num_head,
                batch_size,
                split_idx,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Paged-KV mainloop (inline). Mirrors the dense path's prologue ->
        # masked iteration -> unmasked iterations -> epilogue structure, but
        # routes every K/V load through PagedKVManager so per-n_block reads
        # follow the page table. We keep this fully inline rather than
        # reusing compute_one_n_block: the latter would require passing
        # paged_kv_manager through a @cute.jit boundary, which the CuTe DSL
        # verifier rejects ("operand does not dominate this use") when the
        # manager's mutable register fragments are referenced from inside
        # nested scf.if / scf.for regions inside compute_one_n_block.
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(blocksparse_tensors is None and mPageTable is not None):
            # Paged-KV mainloop: build the PagedKVManager and delegate to
            # _paged_kv_mainloop. That method is @cute.jit and takes
            # paged_kv_manager as an explicit argument so the manager's
            # mutable register fragments dominate every use inside.
            #
            # PagedKVManager allocates ceil(tile_n / num_threads) page-table
            # slots per producer thread, so SM120 D192/D256 can use tile_n=64
            # and still stay under the 99 KB SMEM cap.
            # CRITICAL: skip wasted varlen grid tiles. SingleTileVarlenScheduler
            # rounds the grid up so blockIdx may correspond to batch_idx >=
            # num_batch; for those, work_tile.is_valid_tile is False and the
            # tile_idx components (batch_idx in particular) are garbage. The
            # dense path tolerates this because its loads are page-table-free
            # and predicated by seqlen_q/seqlen_k (which OOB-read to 0/garbage
            # and then short-circuit). The paged path actively dereferences
            # mPageTable[batch_idx, ...] -> mK[..., page] before any
            # predicate, which dereferences garbage page indices and faults.
            paged_kv_manager = PagedKVManager.create(
                mPageTable,
                mK,
                mV,
                FastDivmodDivisor(mK.shape[0]),
                batch_size,
                num_head_kv,
                tidx,
                seqlen.seqlen_k,
                0,  # leftpad_k
                self.tile_n,
                self.tile_hdim,
                self.tile_hdimv,
                self.num_producer_threads,
                mK.element_type,
                arch=90,  # SM90 layout convention: V matches K, no gmem transpose
            )
            # Skip wasted varlen grid tiles (batch_idx >= num_batch). For
            # these, batch_idx is garbage and mPageTable[garbage, ...] would
            # dereference unmapped pages and fault.
            if work_tile.is_valid_tile:
                self._paged_kv_mainloop(
                    paged_kv_manager,
                    mO,
                    mLSE,
                    mQ,
                    acc_O,
                    softmax,
                    sQ,
                    sK,
                    sV,
                    sVt,
                    sO_layout,
                    gmem_tiled_copy_Q,
                    gmem_tiled_copy_O,
                    tiled_mma_pv,
                    thr_mma_qk,
                    thr_mma_pv,
                    tSrQ,
                    tSrK,
                    tOrVt,
                    tSsQ,
                    tSsK,
                    tOsVt,
                    smem_thr_copy_Q,
                    smem_thr_copy_K,
                    smem_thr_copy_V,
                    n_block,
                    n_block_min,
                    n_block_max,
                    block_info,
                    seqlen,
                    m_block,
                    batch_size,
                    num_head,
                    window_size_left,
                    window_size_right,
                    learnable_sink,
                    gQ,
                    tidx,
                    mQ_cur,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    split_idx=split_idx if const_expr(self.is_split_kv) else Int32(0),
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
        score_mod: Callable | None,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
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

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        # wait for smem tile QK before mma calculation for S
        sync()

        # need predicates for the first tile
        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(
                    n_block - self.num_stages + 1,
                    smem_pipe_write,
                    need_predicates=is_first_n_block and self.num_stages == 1,
                )
            cute.arch.cp_async_commit_group()

        if const_expr(not self.hook_load_v):
            load_V_next()
        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[
                None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0
            ],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            hook_fn=load_V_next if const_expr(self.hook_load_v) else None,
            A_in_regs=self.Q_in_regs,
        )
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(n_block - self.num_stages, smem_pipe_write, need_predicates=False)
            cute.arch.cp_async_commit_group()

        # wait for smem tile V for O
        if const_expr(self.num_stages == 1):
            sync()
            if const_expr(not self.hook_load_k):
                load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[
                None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0
            ],
            smem_copy_params.smem_thr_copy_V,
            hook_fn=load_K_next if const_expr(self.num_stages == 1 and self.hook_load_k) else None,
        )
        # if const_expr(self.num_stages > 1):
        #     load_K_next()
    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    @cute.jit
    def mma_one_n_block_bs(
        self,
        n_block: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        """Process one KV block for block-sparse attention (load, GEMM QK, mask, softmax, PV GEMM).

        Unlike compute_one_n_block, this does not overlap loads with the next block since the
        next block address is not known ahead of time in the block-sparse case.
        """
        acc_S = cute.make_rmem_tensor(
            mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_S.fill(0.0)

        # WAR hazard guard (sm120): this block reuses the single-stage smem K/V
        # buffers (smem_pipe_write=0). Before overwriting sK/sV with the next
        # block's cp.async loads we must ensure the *previous* block's QK/PV MMAs
        # have finished reading those same buffers. Without this, multi-mask-block
        # tiles (e.g. block-sparse + a within-tile mask_mod such as mini_causal at
        # seqlen >= 1024) race the load against the prior block's PV GEMM and
        # produce nondeterministic wrong output once enough heads/CTAs are in
        # flight. The first block has no predecessor within this tile, and its
        # prologue already synchronizes after load_Q. Gated to sm120; the SM80
        # base path is fixed separately.
        if const_expr(not is_first_n_block and getattr(self, "is_sm120", False)):
            cute.arch.barrier()

        load_K(n_block, smem_pipe_write=0, need_predicates=True)
        cute.arch.cp_async_commit_group()
        load_V(n_block, smem_pipe_write=0, need_predicates=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()

        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            A_in_regs=self.Q_in_regs,
        )
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=True)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, 0],
            smem_copy_params.smem_thr_copy_V,
        )

    @cute.jit
    def _paged_kv_mainloop(
        self,
        paged_kv_manager: PagedKVManager,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mQ: cute.Tensor,
        acc_O: cute.Tensor,
        softmax: Softmax,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_pv: cute.TiledMma,
        thr_mma_qk,
        thr_mma_pv,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrVt: cute.Tensor,
        tSsQ: cute.Tensor,
        tSsK: cute.Tensor,
        tOsVt: cute.Tensor,
        smem_thr_copy_Q,
        smem_thr_copy_K,
        smem_thr_copy_V,
        n_block: Int32,
        n_block_min: Int32,
        n_block_max: Int32,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        gQ: cute.Tensor,
        tidx: Int32,
        mQ_cur: cute.Tensor,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        split_idx: Int32 = Int32(0),
    ):
        """Inline mainloop for paged-KV (cp.async, num_stages=1) on SM80/SM120.

        This mirrors the structure of the dense path
        (prologue -> first masked iter -> causal/local-masked iters ->
        unmasked iters -> epilogue) but performs every K/V load through
        the supplied PagedKVManager. Because we are @cute.jit, the manager
        is reconstructed once at function entry and its SSA values
        dominate every nested scf.if / scf.for region inside.

        We support only num_stages == 1 here: that matches the configuration
        the SM80 base kernel ships with on consumer Blackwell, and matches
        SM90's paged_kv_non_tma path (which also runs single-stage when
        page_size != tile_n).
        """
        assert self.num_stages == 1, (
            "Paged-KV mainloop currently supports num_stages=1 only."
        )

        # Prologue: Q load, first K load.
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        if const_expr(self.pack_gqa):
            # Mirror the dense / block-sparse prologue: the plain cp_async
            # self.load_Q cannot address the packed composite
            # (qhead_per_kvhead, seqlen) Q layout (cute.local_tile collapses
            # adjacent qhead rows that live at non-adjacent strides), so use
            # the PackGQA per-row pointer loader instead.
            pack_gqa_helper = PackGQA(
                self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
            )
            if const_expr(self.pack_gqa_all_rows_valid):
                if const_expr(self.pack_gqa_fast_valid_rows):
                    pack_gqa_helper.load_Q(
                        mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q,
                        all_rows_valid=True,
                    )
                else:
                    pack_gqa_helper.load_Q_all_rows_valid(
                        mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q,
                    )
            else:
                pack_gqa_helper.load_Q(
                    mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q,
                )
        else:
            self.load_Q(
                gmem_thr_copy_Q, gQ, sQ, m_block,
                seqlen=seqlen.seqlen_q, headdim=mQ.shape[1],
            )
        cute.arch.cp_async_commit_group()

        paged_kv_manager.load_page_table(n_block)
        paged_kv_manager.load_KV(n_block, sK[None, None, 0], "K")
        cute.arch.cp_async_commit_group()

        if const_expr(self.Q_in_regs):
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)
            cute.arch.barrier()
        else:
            # Wait for Q so we can use sQ in GEMM_QK below.
            cute.arch.cp_async_wait_group(1)

        mask = AttentionMask(
            self.tile_m,
            self.tile_n,
            seqlen,
            window_size_left,
            window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

        # ---- One n-block iteration, inlined. Sequence (single-stage):
        #   1. wait for K(nb)
        #   2. issue V(nb) load (cp.async)
        #   3. GEMM_QK -> acc_S
        #   4. wait for V(nb) and (if nb > n_block_min) issue K(nb-1) load
        #   5. mask, softmax, rP
        #   6. GEMM_PV
        # We cannot factor this into a Python helper (closures over
        # paged_kv_manager are rejected in dynamic control flow), so the
        # body is open-coded per iteration site below.
        nb = n_block

        # SplitKV: a split with no assigned KV blocks (n_block_min == n_block_max)
        # must skip ALL compute and fall straight through to the finalize/epilogue,
        # which then writes the clean empty-split sentinel (O=0, LSE=-inf) the
        # combine kernel drops.  Without this guard the unconditional first
        # iteration below would process block max(n_block_max-1, 0) — a block that
        # actually belongs to a lower split — and emit a finite garbage partial
        # that the combine double-counts.  Mirrors the dense path's has_work guard.
        # For the non-split path has_work is always True (a valid tile always has
        # >= 1 block), so this is a no-op there.
        has_work = (
            n_block_max > n_block_min
            if const_expr(self.is_split_kv)
            else cutlass.Boolean(True)
        )

        # ---- First (masked) iteration ----
        if has_work:
            acc_S = cute.make_rmem_tensor(
                thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
            )
            acc_S.fill(0.0)
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            # Issue V(nb)
            paged_kv_manager.load_KV(nb, sV[None, None, 0], "V")
            cute.arch.cp_async_commit_group()
            sm80_utils.gemm(
                thr_mma_qk,
                acc_S,
                tSrQ,
                tSrK,
                tSsQ,
                tSsK[None, None, None, 0],
                smem_thr_copy_Q,
                smem_thr_copy_K,
                A_in_regs=self.Q_in_regs,
            )
            if const_expr(self.score_mod is not None):
                self.apply_score_mod(
                    thr_mma_qk, batch_idx, head_idx, m_block, acc_S, nb,
                    softmax_scale=softmax.softmax_scale, seqlen=seqlen,
                    aux_data=aux_data, fastdiv_mods=fastdiv_mods,
                )
            # Wait for V; issue K(nb-1) if any remaining
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if nb - 1 >= n_block_min:
                paged_kv_manager.load_page_table(nb - 1)
                paged_kv_manager.load_KV(nb - 1, sK[None, None, 0], "K")
            cute.arch.cp_async_commit_group()
            mask.apply_mask(
                acc_S, n_block=nb,
                batch_idx=batch_idx, head_idx=head_idx, m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal, mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                mask_mod=self.mask_mod,
                mask_seqlen=True,
            )
            row_scale = softmax.online_softmax(acc_S, is_first=True, check_inf=True)
            softmax.rescale_O(acc_O, row_scale)
            rP = cute.make_fragment_like(acc_S, self.dtype)
            rP.store(acc_S.load().to(self.dtype))
            tOrP = layout_utils.reshape_acc_to_frgA(rP)
            sm80_utils.gemm_rs(
                thr_mma_pv,
                acc_O,
                tOrP,
                tOrVt,
                tOsVt[None, None, None, 0],
                smem_thr_copy_V,
            )

        # ---- Causal/local masked iterations ----
        # After this block, `unmasked_n_block_start` is the n_block from
        # which the unmasked loop should iterate downward (exclusive).
        # For non-causal, that's n_block (= n_block_max - 1).
        unmasked_n_block_start = n_block
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen, m_block, n_block_min
            )
            # Mirror the dense path: a non-causal local window whose right bound
            # reaches the seqlen boundary can make get_n_block_min_causal_local_mask
            # return >= n_block_max, which would make the unmasked loop below
            # reprocess the first block or step past the valid range.
            unmasked_n_block_start = cutlass.min(
                n_block_min_causal_local_mask, n_block_max - 1
            )
            for n_tile in cutlass.range(
                n_block_max - 1 - n_block_min_causal_local_mask, unroll=1
            ):
                nb = n_block_max - 2 - n_tile
                acc_S = cute.make_rmem_tensor(
                    thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
                )
                acc_S.fill(0.0)
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                paged_kv_manager.load_KV(nb, sV[None, None, 0], "V")
                cute.arch.cp_async_commit_group()
                sm80_utils.gemm(
                    thr_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK,
                    tSsQ,
                    tSsK[None, None, None, 0],
                    smem_thr_copy_Q,
                    smem_thr_copy_K,
                    A_in_regs=self.Q_in_regs,
                )
                if const_expr(self.score_mod is not None):
                    self.apply_score_mod(
                        thr_mma_qk, batch_idx, head_idx, m_block, acc_S, nb,
                        softmax_scale=softmax.softmax_scale, seqlen=seqlen,
                        aux_data=aux_data, fastdiv_mods=fastdiv_mods,
                    )
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                if nb - 1 >= n_block_min:
                    paged_kv_manager.load_page_table(nb - 1)
                    paged_kv_manager.load_KV(nb - 1, sK[None, None, 0], "K")
                cute.arch.cp_async_commit_group()
                mask.apply_mask(
                    acc_S, n_block=nb,
                    batch_idx=batch_idx, head_idx=head_idx, m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal, mask_local=self.is_local,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                    mask_mod=self.mask_mod,
                    mask_seqlen=True,
                )
                row_scale = softmax.online_softmax(acc_S, is_first=False, check_inf=True)
                softmax.rescale_O(acc_O, row_scale)
                rP = cute.make_fragment_like(acc_S, self.dtype)
                rP.store(acc_S.load().to(self.dtype))
                tOrP = layout_utils.reshape_acc_to_frgA(rP)
                sm80_utils.gemm_rs(
                    thr_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt,
                    tOsVt[None, None, None, 0],
                    smem_thr_copy_V,
                )

        # ---- Unmasked iterations ----
        unmasked_n_block_stop = n_block_min
        if const_expr(self.is_local):
            unmasked_n_block_stop = cutlass.min(
                unmasked_n_block_start,
                block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                ),
            )
        for n_tile in cutlass.range(
            unmasked_n_block_start - unmasked_n_block_stop, unroll=1
        ):
            nb = unmasked_n_block_start - n_tile - 1
            acc_S = cute.make_rmem_tensor(
                thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
            )
            acc_S.fill(0.0)
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            paged_kv_manager.load_KV(nb, sV[None, None, 0], "V")
            cute.arch.cp_async_commit_group()
            sm80_utils.gemm(
                thr_mma_qk,
                acc_S,
                tSrQ,
                tSrK,
                tSsQ,
                tSsK[None, None, None, 0],
                smem_thr_copy_Q,
                smem_thr_copy_K,
                A_in_regs=self.Q_in_regs,
            )
            if const_expr(self.score_mod is not None):
                self.apply_score_mod(
                    thr_mma_qk, batch_idx, head_idx, m_block, acc_S, nb,
                    softmax_scale=softmax.softmax_scale, seqlen=seqlen,
                    aux_data=aux_data, fastdiv_mods=fastdiv_mods,
                )
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if nb - 1 >= n_block_min:
                paged_kv_manager.load_page_table(nb - 1)
                paged_kv_manager.load_KV(nb - 1, sK[None, None, 0], "K")
            cute.arch.cp_async_commit_group()
            mask.apply_mask(
                acc_S, n_block=nb,
                batch_idx=batch_idx, head_idx=head_idx, m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal, mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                mask_mod=self.mask_mod,
                mask_seqlen=False,
            )
            row_scale = softmax.online_softmax(acc_S, is_first=False, check_inf=True)
            softmax.rescale_O(acc_O, row_scale)
            rP = cute.make_fragment_like(acc_S, self.dtype)
            rP.store(acc_S.load().to(self.dtype))
            tOrP = layout_utils.reshape_acc_to_frgA(rP)
            sm80_utils.gemm_rs(
                thr_mma_pv,
                acc_O,
                tOrP,
                tOrVt,
                tOsVt[None, None, None, 0],
                smem_thr_copy_V,
            )

        # ---- Local-attention tail iterations ----
        if const_expr(self.is_local):
            for n_tile in cutlass.range(unmasked_n_block_stop - n_block_min, unroll=1):
                nb = unmasked_n_block_stop - n_tile - 1
                acc_S = cute.make_rmem_tensor(
                    thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
                )
                acc_S.fill(0.0)
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                paged_kv_manager.load_KV(nb, sV[None, None, 0], "V")
                cute.arch.cp_async_commit_group()
                sm80_utils.gemm(
                    thr_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK,
                    tSsQ,
                    tSsK[None, None, None, 0],
                    smem_thr_copy_Q,
                    smem_thr_copy_K,
                    A_in_regs=self.Q_in_regs,
                )
                if const_expr(self.score_mod is not None):
                    self.apply_score_mod(
                        thr_mma_qk, batch_idx, head_idx, m_block, acc_S, nb,
                        softmax_scale=softmax.softmax_scale, seqlen=seqlen,
                        aux_data=aux_data, fastdiv_mods=fastdiv_mods,
                    )
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                if nb - 1 >= n_block_min:
                    paged_kv_manager.load_page_table(nb - 1)
                    paged_kv_manager.load_KV(nb - 1, sK[None, None, 0], "K")
                cute.arch.cp_async_commit_group()
                mask.apply_mask(
                    acc_S, n_block=nb,
                    batch_idx=batch_idx, head_idx=head_idx, m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal, mask_local=self.is_local,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
                    mask_mod=self.mask_mod,
                    mask_seqlen=True,
                )
                row_scale = softmax.online_softmax(acc_S, is_first=False, check_inf=True)
                softmax.rescale_O(acc_O, row_scale)
                rP = cute.make_fragment_like(acc_S, self.dtype)
                rP.store(acc_S.load().to(self.dtype))
                tOrP = layout_utils.reshape_acc_to_frgA(rP)
                sm80_utils.gemm_rs(
                    thr_mma_pv,
                    acc_O,
                    tOrP,
                    tOrVt,
                    tOsVt[None, None, None, 0],
                    smem_thr_copy_V,
                )

        # ---- Finalize + epilogue ----
        # Drain any outstanding cp.async groups (e.g. trailing empty commits
        # we emit when n_block_min < 0 in the last iteration's "next K"
        # branch). Without this drain, later kernels reusing the same gmem
        # slots can race with our completion fence.
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        row_scale = softmax.finalize(
            sink_val=self.compute_sink_val(
                learnable_sink, softmax, m_block, head_idx, thr_mma_qk, split_idx
            ),
            is_sm120=getattr(self, "is_sm120", False),
        )
        softmax.rescale_O(acc_O, row_scale)
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O,
            softmax.row_sum,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
        )


# SM90 forward pass moved to flash_fwd_sm90.py; re-export for backward compatibility
def __getattr__(name):
    if name == "FlashAttentionForwardSm90":
        from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
        return FlashAttentionForwardSm90
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
