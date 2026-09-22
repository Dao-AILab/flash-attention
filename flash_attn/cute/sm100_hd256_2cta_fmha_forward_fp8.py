# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

import math
from typing import Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Int64, Float32
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
from cutlass.cutlass_dsl import min as dsl_min

from flash_attn.cute.tile_scheduler import (
    SM100_TMEM_CAPACITY_COLUMNS,
    sm100_fmha_block_coord,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
)
from flash_attn.cute.mask import Sm100FusedMask as FusedMask
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors, _TUNING_CONFIG
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.utils import (
    ex2_emulation_2,
    fmax as fmax3,
    as_bshkrd_tensor,
    AuxData,
)


class BlackwellFusedMultiHeadAttentionForwardFP8:
    """Dense FP8 (E4M3) forward for head_dim = head_dim_v = 256 on SM100 / SM103.

    A 2-CTA variant of ``BlackwellFusedMultiHeadAttentionForward`` tuned for the FP8
    tensor-core rate, where per-tile setup and the softmax / TMEM round trips of the
    generic kernel become the bottleneck:

    * persistent static scheduling; causal tiles are ordered longest-first with
      alternate waves reversed so trip counts balance across the grid;
    * separate three-deep K and V TMA rings, one full-D QK^T MMA per KV block, and
      QK^T issued one KV block ahead of PV;
    * scores are read from TMEM with ``tcgen05.ld.red`` (SM103) so the row max comes
      back with the load; SM100 uses a plain load and a software max;
    * the correction factor is passed to the correction warps through SMEM, and the
      accumulator rescale is skipped when every factor in the warp is exactly 1.0.

    FP32 accumulation, the softmax reduction order, EX2 emulation and the FP8 P
    conversion are unchanged from the generic kernel, so the BF16 output is bitwise
    identical to it. Dense layouts only: varlen, paged KV, local windows, descales,
    score/mask modifiers and aux tensors keep using the generic kernel.
    """

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        kv_stage: int = 3,
        use_ldred_rowmax: Optional[bool] = None,
        is_static_persistent: bool = True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
        has_tile_count_semaphore: bool = False,
        seqlen_k_per_split: Optional[int] = None,
    ):
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 FP8 hd256 kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert score_mod is None, "SM100 FP8 hd256 kernel does not support score_mod"
        assert mask_mod is None, "SM100 FP8 hd256 kernel does not support mask_mod"
        assert not has_aux_tensors, "SM100 FP8 hd256 kernel does not support aux tensors"
        assert not paged_kv_non_tma, "SM100 FP8 hd256 kernel does not support paged KV"
        assert not is_varlen_q, "SM100 FP8 hd256 kernel does not support varlen"
        assert not is_local, "SM100 FP8 hd256 kernel does not support local attention"
        assert not pack_gqa, "SM100 FP8 hd256 kernel does not support pack_gqa"
        assert not is_split_kv, "SM100 FP8 hd256 kernel does not support SplitKV"
        assert q_subtile_factor == 1 and kv_subtile_factor == 1, (
            "SM100 FP8 hd256 kernel does not support subtiling"
        )
        assert m_block_size == 128 and n_block_size == 128, (
            "SM100 FP8 hd256 kernel only supports tile_m=128 and tile_n=128"
        )
        # q_stage / persistence / scheduler knobs are accepted for interface parity only;
        # kv_stage is fixed at three KV blocks in flight (measured best of 2 / 3).

        self.qk_acc_dtype = cutlass.Float32
        self.pv_acc_dtype = cutlass.Float32
        self.qhead_per_kvhead = qhead_per_kvhead
        self.mma_tiler = (128, 128, head_dim)
        self.cta_tiler = self.mma_tiler
        # One QK^T MMA covers a 256 x 128 score tile over the full head dim; one PV
        # MMA covers the full 256-wide output tile over one 128-key block.
        self.qk_mma_tiler = (2 * self.mma_tiler[0], self.mma_tiler[1], self.cta_tiler[2])
        self.pv_mma_tiler = (2 * self.mma_tiler[0], self.cta_tiler[2], self.mma_tiler[1])
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // 2,
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        # The MMA warp takes one K and one V try-wait token per KV block.
        assert self.iterations_qk == 1 and self.iterations_pv == 1
        self.cluster_shape_mn = (2, 1)
        self.tmem_warp_shape_mn = (4, 1)
        self.kv_blocks = 3
        self.is_causal = is_causal
        self.is_local = is_local
        self.use_semantic_trip_range = is_causal or is_local

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (10, 11)
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_id,
            )
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

        self.tmem_s_offset = 0
        self.tmem_o_offset = 256
        self.tmem_p_offset = self.tmem_s_offset

        # Reuse the generic hd256 2-CTA register and EX2-emulation tuning.
        _tune = _TUNING_CONFIG.get((True, is_causal, 256, False), {})
        self.num_regs_softmax = _tune.get("num_regs_softmax", 256)
        self.num_regs_correction = _tune.get("num_regs_correction", 160)
        self.num_regs_other = 32
        self.ex2_emu_freq = _tune.get("ex2_emu_freq", 4)
        self.ex2_emu_res = _tune.get("ex2_emu_res", 3)
        self.ex2_emu_start_frg = _tune.get("ex2_emu_start_frg", 0)

        if use_ldred_rowmax is None:
            use_ldred_rowmax = BaseDSL._get_dsl().get_arch_enum().is_family_of(Arch.sm_103f)
        # tcgen05.ld.red (SM103 family) returns the per-row max with the score load.
        # It is only used on unmasked iterations; masked ones reduce in software.
        self.use_ldred_rowmax = use_ldred_rowmax

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = self.iterations_qk
        self.k_stage = self.kv_blocks * self.iterations_qk
        self.v_stage = self.kv_blocks
        self.qk_acc_stage = 2
        self.mma_corr_stage = 1

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
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        max_seqlen_q: Optional[Int32] = None,
        stream: cuda.CUstream = None,
    ):
        # Signature kept in parity with FlashAttentionForwardSm100.__call__; the
        # interface only routes dense E4M3 hd256 calls here.
        assert mCuSeqlensQ is None and mCuSeqlensK is None, "FP8 hd256 kernel is dense only"
        assert mSeqUsedQ is None and mSeqUsedK is None, "FP8 hd256 kernel is dense only"
        assert mPageTable is None, "FP8 hd256 kernel does not support paged KV"
        assert window_size_left is None and window_size_right is None, (
            "FP8 hd256 kernel does not support local attention"
        )
        assert learnable_sink is None, "FP8 hd256 kernel does not support learnable_sink"
        assert descale_tensors is None, "FP8 hd256 kernel does not support descale_tensors"
        assert blocksparse_tensors is None, "FP8 hd256 kernel does not support block sparsity"
        assert aux_data.tensors is None and aux_data.scalars is None, (
            "FP8 hd256 kernel does not support aux tensors"
        )

        o_tensor = assume_tensor_aligned(mO, canonicalize_singletons=True)

        # Accept legacy 5D (b, s, h_k, h_r, d) or standard 4D (b, s, h, d) tensors.
        q_rank = len(mQ.shape)
        k_rank = len(mK.shape)
        if cutlass.const_expr(q_rank == 5):
            s_q = mQ.shape[1]
            h_q = mQ.shape[2] * mQ.shape[3]
            d = mQ.shape[4]
        elif cutlass.const_expr(q_rank == 4):
            s_q = mQ.shape[1]
            h_q = mQ.shape[2]
            d = mQ.shape[3]
        else:
            raise RuntimeError(f"hd256 forward expects q rank 4 or 5, got rank {q_rank}")
        if cutlass.const_expr(k_rank == 5 or k_rank == 4):
            s_k = mK.shape[1]
            h_k = mK.shape[2]
        else:
            raise RuntimeError(f"hd256 forward expects k rank 4 or 5, got rank {k_rank}")
        b = mQ.shape[0]

        scale_softmax = softmax_scale
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
        scale_output = 1.0
        h_r = h_q // h_k
        s_q64 = Int64(s_q)
        s_k64 = Int64(s_k)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        b64 = Int64(b)

        q_norm = as_bshkrd_tensor(mQ, h_k, h_r, False)
        o_norm = as_bshkrd_tensor(o_tensor, h_k, h_r, False)
        k_norm = as_bshkrd_tensor(mK, h_k, 1, False)
        v_norm = as_bshkrd_tensor(mV, h_k, 1, False)

        # Kernel layouts: Q/O/K as (s, d, ((h_r, h_k), b)), V as (d, s, ((h_r, h_k), b)).
        # K/V broadcast over the grouped query heads with a 0 stride on h_r.
        q = cute.make_tensor(
            q_norm.iterator,
            cute.make_layout(
                (s_q64, d, ((h_r, h_k), b)),
                stride=(
                    q_norm.stride[1],
                    q_norm.stride[4],
                    ((q_norm.stride[3], q_norm.stride[2]), q_norm.stride[0]),
                ),
            ),
        )
        k = cute.make_tensor(
            k_norm.iterator,
            cute.make_layout(
                (s_k64, d, ((h_r, h_k), b)),
                stride=(
                    k_norm.stride[1],
                    k_norm.stride[4],
                    ((0, k_norm.stride[2]), k_norm.stride[0]),
                ),
            ),
        )
        v = cute.make_tensor(
            v_norm.iterator,
            cute.make_layout(
                (d, s_k64, ((h_r, h_k), b)),
                stride=(
                    v_norm.stride[4],
                    v_norm.stride[1],
                    ((0, v_norm.stride[2]), v_norm.stride[0]),
                ),
            ),
        )
        o = cute.make_tensor(
            o_norm.iterator,
            cute.make_layout(
                (s_q64, d, ((h_r, h_k), b)),
                stride=(
                    o_norm.stride[1],
                    o_norm.stride[4],
                    ((o_norm.stride[3], o_norm.stride[2]), o_norm.stride[0]),
                ),
            ),
        )
        if cutlass.const_expr(mLSE is not None):
            # (s, ((h_r, h_k), b))
            lse = cute.make_tensor(
                mLSE.iterator,
                cute.make_layout(
                    (s_q64, ((h_r, h_k), b64)),
                    stride=(1, ((s_q64, h_r64 * s_q64), h_r64 * h_k64 * s_q64)),
                ),
            )
        else:
            lse = None

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width

        # Persistent static scheduler over (m_ctas, h, b). m_ctas is rounded up to
        # the cluster size so both CTAs of a pair own the two halves of one 256-row
        # tile; the padded half is masked exactly like the non-persistent grid.
        m_ctas = cute.ceil_div(cute.size(o.shape[0]), self.cta_tiler[0])
        m_ctas = cute.round_up(m_ctas, self.cluster_shape_mn[0])
        self.tile_sched_params = FmhaStaticTileSchedulerParams(
            True,
            (m_ctas, cute.size(o.shape[2][0]), cute.size(o.shape[2][1])),
        )
        grid = FmhaStaticTileScheduler.get_grid_shape(self.tile_sched_params)
        if cutlass.const_expr(not self.is_causal):
            # Non-causal tiles have equal KV trip counts: keep the smallest number of
            # resident CTA pairs that still needs the same number of persistent
            # rounds, so the tail round is as full as possible.
            cluster_size = self.cluster_shape_mn[0]
            total_pairs = cute.size(self.tile_sched_params.problem_shape_mbh) // cluster_size
            resident_pairs = cute.size(grid) // cluster_size
            persistent_rounds = cute.ceil_div(total_pairs, resident_pairs)
            balanced_pairs = cute.ceil_div(total_pairs, persistent_rounds)
            grid = (balanced_pairs * cluster_size, 1, 1)

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != cute.nvgpu.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != cute.nvgpu.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != cute.nvgpu.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        # The unmasked softmax path relies on scores being finite, which holds for
        # E4M3 (no inf encoding) with TMA zero fill; E5M2 keeps the generic kernel.
        if cutlass.const_expr(self.q_dtype != cutlass.Float8E4M3FN):
            raise TypeError(f"FP8 hd256 kernel requires float8_e4m3fn inputs, got {self.q_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # P is consumed from TMEM as the K-major A operand of the PV MMA.
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = cute.nvgpu.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_k_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_v_bytes = v_copy_size * cute.size(pv_tiled_mma.thr_id.shape)

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]
            # MMA -> softmax (S ready) and softmax -> MMA (P ready)
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            # softmax -> correction: rescale factor ready
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            # MMA -> correction: O accumulator ownership
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        # Per-stage correction factor, softmax -> correction, guarded by s_corr.
        sScale = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.qk_acc_stage
            ),
            byte_alignment=128,
        )
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)

        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        blk_idx = cute.arch.block_idx()
        tile_sched = FmhaStaticTileScheduler(
            tile_sched_params, blk_idx[0], blk_idx, cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = self._remap_block_coord(
                    sm100_fmha_block_coord(work_tile, False), tile_sched_params
                )  # (q_tile_idx, 0, (head_idx, batch_idx))
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                q_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                )
                # (bM, bK, loopM, loopK, loopL)
                gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_q,
                    block_in_cluster_coord_vmnk[2],
                    q_cta_layout,
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tSgQ_qdl, 0, 3),
                )
                kv_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                )
                gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    block_in_cluster_coord_vmnk[1],
                    kv_cta_layout,
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK_kdl, 0, 3),
                )
                gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
                tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    block_in_cluster_coord_vmnk[1],
                    kv_cta_layout,
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tSgV_dkl, 0, 3),
                )
                # ((atom_v, rest_v), RestN, RestK)
                tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                # ((atom_v, rest_v), RestK)
                tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]

                kv_loop_start, kv_loop_steps = FusedMask.get_trip_start_count_via_block_info(
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    self.is_causal,
                    self.is_local,
                    None,
                    None,
                )
                kv_loop_end = kv_loop_start + kv_loop_steps
                # Q
                for iter in cutlass.range(self.iterations_qk, unroll=1):
                    q_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, iter],
                        tQsQ[None, q_handle.index],
                        tma_bar_ptr=q_handle.barrier,
                    )
                # K0
                for iter in cutlass.range(self.iterations_qk, unroll=1):
                    k_handle = load_k_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, kv_loop_start, iter],
                        tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                for i in cutlass.range(1, kv_loop_steps, 1, unroll=1):
                    kv_coord = kv_loop_start + i
                    # Ki
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        k_handle = load_k_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord, iter],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                    # Vi-1
                    for iter in cutlass.range(self.iterations_pv, unroll=1):
                        v_handle = load_v_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, iter, kv_coord - 1],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )
                # Vend
                for iter in cutlass.range(self.iterations_pv, unroll=1):
                    v_handle = load_v_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, iter, kv_loop_end - 1],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )

                work_tile = tile_sched.advance_to_next_work()
            load_k_producer.tail()
            load_v_producer.tail()
            load_q_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_cta = cta_rank_in_cluster % 2 == 0

            while work_tile.is_valid_tile:
                curr_block_coord = self._remap_block_coord(
                    sm100_fmha_block_coord(work_tile, False), tile_sched_params
                )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                _, kv_loop_steps = FusedMask.get_trip_start_count_via_block_info(
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    self.is_causal,
                    self.is_local,
                    None,
                    None,
                )

                load_q_releaser = load_q_consumer.clone()
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                if kv_loop_steps > 1:
                    if is_leader_cta:
                        # QK0
                        s_handle = mma_s_producer.acquire_and_advance()
                        tStS_slice = tStS[None, None, None, s_handle.index]
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            load_q_consumer.wait_and_advance()
                            tSrQ_slice = tSrQ[None, None, None, iter]
                            k_handle = load_k_consumer.wait_and_advance()
                            tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                            num_kphases = cute.size(tSrQ_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    qk_tiled_mma,
                                    tStS_slice,
                                    tSrQ_slice[kphase_coord],
                                    tSrK_trans_slice[kphase_coord],
                                    tStS_slice,
                                )
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                        s_handle.commit()
                        # QK1: QK runs one KV block ahead of PV so the blocking O-slot
                        # acquire before each PV always has an issued QK behind it.
                        s_handle = mma_s_producer.acquire_and_advance()
                        tStS_slice = tStS[None, None, None, s_handle.index]
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            tSrQ_slice = tSrQ[None, None, None, iter]
                            k_handle = load_k_consumer.wait_and_advance()
                            tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                            num_kphases = cute.size(tSrQ_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    qk_tiled_mma,
                                    tStS_slice,
                                    tSrQ_slice[kphase_coord],
                                    tSrK_trans_slice[kphase_coord],
                                    tStS_slice,
                                )
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                        s_handle.commit()
                    for i in cutlass.range(2, kv_loop_steps, 1, unroll=1):
                        if is_leader_cta:
                            p_try_token = p_mma_consumer.try_wait()
                            o_try_token = mma_corr_producer.try_acquire()
                            v_try_token = load_v_consumer.try_wait()
                            # PVi-2
                            p_handle = p_mma_consumer.wait_and_advance(p_try_token)
                            o_handle = mma_corr_producer.acquire_and_advance(o_try_token)
                            pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_pv, unroll=1):
                                v_handle = load_v_consumer.wait_and_advance(v_try_token)
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                tOtO_slice = tOtO_staged[None, None, None, iter]
                                tStS_slice = tStS[None, None, None, p_handle.index]
                                tP = cute.make_tensor(
                                    tStS_slice.iterator, p_tmem_layout_staged.outer
                                )
                                tOrP = pv_thr_mma.make_fragment_A(tP)
                                tOrP_slice = cute.make_tensor(
                                    cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                    tOrP.layout,
                                )
                                tOrV_slice = tOrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tOrV_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        pv_tiled_mma,
                                        tOtO_slice,
                                        tOrP_slice[kphase_coord],
                                        tOrV_slice[kphase_coord],
                                        tOtO_slice,
                                    )
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                            o_handle.commit()
                            p_handle.release()

                            # QKi
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_k_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            s_handle.commit()
                    if is_leader_cta:
                        # All QKs are issued; release Q for the next tile.
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            load_q_releaser.release()
                            load_q_releaser.advance()
                        # PVend-1
                        p_try_token = p_mma_consumer.try_wait()
                        o_try_token = mma_corr_producer.try_acquire()
                        v_try_token = load_v_consumer.try_wait()
                        p_handle = p_mma_consumer.wait_and_advance(p_try_token)
                        o_handle = mma_corr_producer.acquire_and_advance(o_try_token)
                        pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_v_consumer.wait_and_advance(v_try_token)
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            tOtO_slice = tOtO_staged[None, None, None, iter]
                            tStS_slice = tStS[None, None, None, p_handle.index]
                            tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout_staged.outer)
                            tOrP = pv_thr_mma.make_fragment_A(tP)
                            tOrP_slice = cute.make_tensor(
                                cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                tOrP.layout,
                            )
                            tOrV_slice = tOrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tOrV_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    pv_tiled_mma,
                                    tOtO_slice,
                                    tOrP_slice[kphase_coord],
                                    tOrV_slice[kphase_coord],
                                    tOtO_slice,
                                )
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                        o_handle.commit()
                        p_handle.release()
                else:
                    if is_leader_cta:
                        # QK0
                        s_handle = mma_s_producer.acquire_and_advance()
                        tStS_slice = tStS[None, None, None, s_handle.index]
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            load_q_consumer.wait_and_advance()
                            tSrQ_slice = tSrQ[None, None, None, iter]
                            k_handle = load_k_consumer.wait_and_advance()
                            tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                            num_kphases = cute.size(tSrQ_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    qk_tiled_mma,
                                    tStS_slice,
                                    tSrQ_slice[kphase_coord],
                                    tSrK_trans_slice[kphase_coord],
                                    tStS_slice,
                                )
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                            load_q_releaser.release()
                            load_q_releaser.advance()
                        s_handle.commit()

                if is_leader_cta:
                    # PVend
                    p_handle = p_mma_consumer.wait_and_advance()
                    o_handle = mma_corr_producer.acquire_and_advance()
                    pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                    for iter in cutlass.range(self.iterations_pv, unroll=1):
                        v_handle = load_v_consumer.wait_and_advance()
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                        tOtO_slice = tOtO_staged[None, None, None, iter]
                        tStS_slice = tStS[None, None, None, p_handle.index]
                        tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout_staged.outer)
                        tOrP = pv_thr_mma.make_fragment_A(tP)
                        tOrP_slice = cute.make_tensor(
                            cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                            tOrP.layout,
                        )
                        tOrV_slice = tOrV[None, None, None, v_handle.index]
                        num_kphases = cute.size(tOrV_slice, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO_slice,
                                tOrP_slice[kphase_coord],
                                tOrV_slice[kphase_coord],
                                tOtO_slice,
                            )
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        v_handle.release()
                    o_handle.commit()
                    p_handle.release()
                work_tile = tile_sched.advance_to_next_work()
            mma_s_producer.tail()
            mma_corr_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            while work_tile.is_valid_tile:
                curr_block_coord = self._remap_block_coord(
                    sm100_fmha_block_coord(work_tile, False), tile_sched_params
                )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )

                row_max = -Float32.inf
                row_max_prev = -Float32.inf
                row_sum = 0.0

                start_count, trip_count = FusedMask.get_trip_start_count_via_block_info(
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    self.is_causal,
                    self.is_local,
                    None,
                    None,
                )
                end_count = start_count + trip_count
                # A tile with no valid keys still runs one fully-masked softmax step.
                zero_trip = end_count <= start_count
                if end_count <= start_count:
                    start_count = 0
                    end_count = 1
                # The K-tail mask is a no-op when seqlen_k is a multiple of the K tile.
                k_tail_unaligned = (seqlen_k % self.qk_mma_tiler[1]) != 0
                if cutlass.const_expr(self.use_semantic_trip_range):
                    n_block_min_causal_local_mask, n_block_min_before_local_mask = (
                        FusedMask.get_trip_mask_bounds_via_block_info(
                            mma_block_coord,
                            self.qk_mma_tiler,
                            seqlen_q,
                            seqlen_k,
                            self.is_causal,
                            self.is_local,
                            None,
                            None,
                        )
                    )
                cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)

                for step in cutlass.range(start_count, end_count, 1, unroll=1):
                    cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                    tScS_iter = qk_thr_mma.partition_C(cS_iter)
                    if cutlass.const_expr(self.use_semantic_trip_range):
                        need_apply_mask = (
                            step >= n_block_min_causal_local_mask
                            or step < n_block_min_before_local_mask
                            or (step == end_count - 1 and k_tail_unaligned)
                            or zero_trip
                        )
                    else:
                        need_apply_mask = (step == end_count - 1 and k_tail_unaligned) or zero_trip
                    # One staged loop; the masked and unmasked bodies are specialized.
                    if need_apply_mask:
                        (
                            row_max,
                            row_sum,
                            mma_s_consumer,
                            p_mma_producer,
                            s_corr_producer,
                        ) = self.softmax_step(
                            True,
                            (row_max_prev, row_sum, seqlen_q, seqlen_k, scale_softmax_log2),
                            (tStS, tScS_iter, sScale),
                            (mma_s_consumer, p_mma_producer, s_corr_producer),
                        )
                    else:
                        (
                            row_max,
                            row_sum,
                            mma_s_consumer,
                            p_mma_producer,
                            s_corr_producer,
                        ) = self.softmax_step(
                            False,
                            (row_max_prev, row_sum, seqlen_q, seqlen_k, scale_softmax_log2),
                            (tStS, tScS_iter, sScale),
                            (mma_s_consumer, p_mma_producer, s_corr_producer),
                        )
                    row_max_prev = row_max
                sum_producer = self.store_sum_max(
                    row_max,
                    mLSE,
                    row_sum,
                    sSum,
                    sum_producer,
                    curr_block_coord,
                    seqlen_q,
                    scale_softmax,
                )
                work_tile = tile_sched.advance_to_next_work()
            p_mma_producer.tail()
            s_corr_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)

            while work_tile.is_valid_tile:
                curr_block_coord = self._remap_block_coord(
                    sm100_fmha_block_coord(work_tile, False), tile_sched_params
                )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )

                # (bM, bN, loopM, loopN, loopL)
                gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_block_tiler, mode=[0, 1]))
                cO_qdl = cute.flat_divide(
                    cute.make_identity_tensor(mO_qdl.shape),
                    cute.select(self.pv_block_tiler, mode=[0, 1]),
                )

                _, kv_loop_steps = FusedMask.get_trip_start_count_via_block_info(
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    self.is_causal,
                    self.is_local,
                    None,
                    None,
                )
                gO_staged = gO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]

                # The first step needs no correction; consume its factor and move on.
                stats_handle = s_corr_consumer.wait_and_advance()
                stats_handle.release()
                for step in cutlass.range(1, kv_loop_steps, 1, unroll=1):
                    # Oi-1 -> Oi
                    mma_corr_consumer, s_corr_consumer = self.correction_rescale(
                        (s_corr_consumer, sScale),
                        (mma_corr_consumer, tOtO_staged, cO_staged),
                        self.epi_tile,
                    )
                # O_partial -> O_final
                mma_corr_consumer, sum_consumer = self.correction_epilog(
                    (seqlen_q, scale_output),
                    (sum_consumer, sSum),
                    (mma_corr_consumer, gO_staged, cO_staged, tOtO_staged),
                    self.epi_tile,
                )
                work_tile = tile_sched.advance_to_next_work()

        if warp_idx > self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cooperative TMEM Deallocation (2CTA)
        # ///////////////////////////////////////////////////////////////////////////////
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

        return

    @cute.jit
    def _remap_block_coord(self, curr_block_coord, tile_sched_params):
        # Causal load balancing: order clusters longest-first over (m, head, batch)
        # and reverse every other persistent wave so the triangular trip counts
        # cancel across the grid-stride walk. A bijection; the two CTAs of a pair
        # keep their rank and land on the two halves of the same 256-row tile.
        if cutlass.const_expr(self.is_causal):
            num_m_ctas = cute.size(tile_sched_params.problem_shape_mbh[0])
            hb = curr_block_coord[2]
            num_heads = cute.size(tile_sched_params.problem_shape_mbh[1])
            cluster_size = self.cluster_shape_mn[0]
            num_m = num_m_ctas // cluster_size
            num_batches = cute.size(tile_sched_params.problem_shape_mbh[2])
            num_groups = num_heads * num_batches
            cta_rank = curr_block_coord[0] % cluster_size
            original_m = curr_block_coord[0] // cluster_size
            linear = original_m + num_m * (hb[0] + num_heads * hb[1])
            workers = cute.arch.grid_dim()[0] // cluster_size
            wave = linear // workers
            worker = linear - wave * workers
            remaining = num_m * num_groups - wave * workers
            wave_size = dsl_min(remaining, workers)
            reverse_wave = (wave % 2) == 1
            position = wave_size - 1 - worker if reverse_wave else worker
            ordered = wave * workers + position
            mapped_m = num_m - 1 - ordered // num_groups
            mapped_group = ordered % num_groups
            return (
                mapped_m * cluster_size + cta_rank,
                curr_block_coord[1],
                (mapped_group % num_heads, mapped_group // num_heads),
            )
        else:
            return curr_block_coord

    @cute.jit
    def softmax_step(
        self,
        need_apply_mask: bool,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        tStS, tScS, sScale = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        # Only warps 0..3 run softmax, so tidx is the 0..127 TMEM lane index.
        thread_idx = tidx
        stats_try_token = s_corr_producer.try_acquire()
        p_try_token = p_mma_producer.try_acquire()
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        if cutlass.const_expr(self.use_ldred_rowmax):
            # One x128 load-reduce covers the 128-column score row and returns the
            # row max alongside the full FP32 fragment.
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(tcgen05.copy.Repetition(128)), self.qk_acc_dtype
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
            )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        if cutlass.const_expr(self.use_ldred_rowmax):
            tTMEM_LOADrS_red = cute.make_rmem_tensor(
                ((1, 1), *tTMEM_LOADrS.shape[1:]), self.qk_acc_dtype
            )
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, (tTMEM_LOADrS, tTMEM_LOADrS_red))
        else:
            tTMEM_LOADrS_red = None
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)

        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        old_row_max = row_max
        row_max_safe = row_max
        if need_apply_mask:
            if cutlass.const_expr(
                self.is_causal and self.qk_mma_tiler[1] == 128 and cute.size(tTMEM_LOADrS) == 128
            ):
                # Bottom-right aligned causal mask on the thread's contiguous 128-key
                # row: keep the prefix of keys k < min(seqlen_k, q + seqlen_k - seqlen_q + 1).
                q0, k0 = tTMEM_LOADcS[0]
                n = cute.size(tTMEM_LOADrS)
                valid_prefix = seqlen_k - k0
                causal_prefix = q0 + (seqlen_k - seqlen_q) + 1 - k0
                valid_prefix = dsl_min(valid_prefix, causal_prefix)
                valid_prefix = Int32(dsl_min(n, cutlass.max(0, valid_prefix)))
                if q0 >= seqlen_q:
                    valid_prefix = Int32(0)
                flat_scores = cute.make_tensor(tTMEM_LOADrS.iterator, cute.make_layout(n))
                keep = cutlass.vector.create_mask((n,), [valid_prefix])
                neg_inf = cutlass.vector.full((n,), -Float32.inf, Float32)
                masked = cutlass.vector.where(keep, flat_scores.load().to_vector(), neg_inf)
                flat_scores.store(cute.TensorSSA.from_vector(masked.ir_value(), dtype=Float32))
            else:
                FusedMask.apply_mask_via_causal_local(
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    self.use_semantic_trip_range,
                    self.is_causal,
                    self.is_local,
                    None,
                    None,
                )
            # Masked iterations reduce over the post-mask values in software.
            row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
            row_max_safe = row_max
            if row_max == -cutlass.Float32.inf:
                row_max_safe = 0.0
        else:
            if cutlass.const_expr(self.use_ldred_rowmax and cute.size(tTMEM_LOADrS_red.shape) == 1):
                row_max = cute.arch.fmax(row_max, tTMEM_LOADrS_red[0])
            elif cutlass.const_expr(
                self.use_ldred_rowmax and cute.size(tTMEM_LOADrS_red.shape) == 4
            ):
                row_max = fmax3(row_max, tTMEM_LOADrS_red[0], tTMEM_LOADrS_red[1])
                row_max = fmax3(row_max, tTMEM_LOADrS_red[2], tTMEM_LOADrS_red[3])
            elif cutlass.const_expr(self.use_ldred_rowmax):
                for _red_i in cutlass.range_constexpr(cute.size(tTMEM_LOADrS_red.shape)):
                    row_max = cute.arch.fmax(row_max, tTMEM_LOADrS_red[_red_i])
            else:
                row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
            # Unmasked dot products of finite FP8 values are finite (out-of-range Q
            # rows are zero-filled by TMA), so no -inf fallback is needed.
            row_max_safe = row_max

        # Correction factor exp2(scale * (old_max - new_max)): handed to the
        # correction warps through SMEM and reused for the row_sum update below.
        scale = scale_softmax_log2
        acc_scale_ = scale * (old_row_max - row_max_safe)
        corr_scale = cute.math.exp2(acc_scale_, fastmath=True)
        stats_handle = s_corr_producer.acquire_and_advance(stats_try_token)
        sScale[
            thread_idx + stats_handle.index * (self.threads_per_warp * len(self.softmax_warp_ids))
        ] = corr_scale
        cute.arch.fence_view_async_shared()
        stats_handle.commit()

        minus_row_max_scale = (0.0 - row_max_safe) * scale
        # Acquire the P slot early so a pipeline stall overlaps the exp2 work.
        p_handle = p_mma_producer.acquire_and_advance(p_try_token)
        # Four packed FP32 chains, each fed as its EX2 fragment becomes ready. This
        # is the same per-chain and final combination order as the generic kernel.
        acc_scale = corr_scale * 0.5
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)
        # Fragment-based FMA + exp2 + FP8 probability conversion; EX2 emulation
        # trades SFU for FMA on a fraction of the elements.
        ex2_frg_tile = 32
        ex2_frg_cnt = cute.size(tTMEM_LOADrS) // ex2_frg_tile
        tTMEM_LOADrS_ex2 = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(ex2_frg_tile))
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        tTMEM_STORErP_ex2 = cute.logical_divide(tTMEM_STORErP, cute.make_layout(ex2_frg_tile))
        for j in cutlass.range_constexpr(ex2_frg_cnt):
            for k in cutlass.range_constexpr(0, ex2_frg_tile, 2):
                tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = cute.arch.fma_packed_f32x2(
                    (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
                if cutlass.const_expr(self.ex2_emu_freq == 0):
                    tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
                    tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                        tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                    )
                else:
                    if cutlass.const_expr(
                        k % self.ex2_emu_freq < self.ex2_emu_freq - self.ex2_emu_res
                        or j >= ex2_frg_cnt - 1
                        or j < self.ex2_emu_start_frg
                    ):
                        tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k, j], fastmath=True
                        )
                        tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                        )
                    else:
                        tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = ex2_emulation_2(
                            tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]
                        )
                exp_pair = (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j])
                if cutlass.const_expr(j == 0):
                    local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, exp_pair)
                elif cutlass.const_expr(j == 1):
                    local_row_sum_1 = cute.arch.add_packed_f32x2(local_row_sum_1, exp_pair)
                elif cutlass.const_expr(j == 2):
                    local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, exp_pair)
                else:
                    local_row_sum_3 = cute.arch.add_packed_f32x2(local_row_sum_3, exp_pair)
            tTMEM_STORErP_ex2[None, j].store(tTMEM_LOADrS_ex2[None, j].load().to(self.q_dtype))
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.q_dtype.width
        tStS_P_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32))
        )
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype),
            tTMEM_STOREcS.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        # Retire the FP32 sum tree while the asynchronous P store drains, then
        # fence and publish the P stage.
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        cute.arch.fence_view_async_tmem_store()
        p_handle.commit()
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_rescale(
        self,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        (s_corr_consumer, sScale) = stats_args
        (mma_o_consumer, tOtO_staged, cO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        scale = sScale[
            thread_idx + stats_handle.index * (self.threads_per_warp * len(self.softmax_warp_ids))
        ]
        cute.arch.fence_view_async_shared()
        stats_handle.release()
        # Skip the accumulator round trip only when every lane's factor is exactly
        # 1.0; multiplying by 1.0 is the identity for every FP32 value.
        need_rescale = cute.arch.vote_any_sync(scale != 1.0)
        o_handle = mma_o_consumer.wait_and_advance()
        if need_rescale:
            self._correction_rescale_body(scale, tOtO_staged, cO_staged, epi_tile, thread_idx)
            cute.arch.fence_view_async_tmem_store()
        o_handle.release()
        return mma_o_consumer, s_corr_consumer

    @cute.jit
    def _correction_rescale_body(
        self,
        scale: Float32,
        tOtO_staged: cute.Tensor,
        cO_staged: cute.Tensor,
        epi_tile: cute.Tile,
        thread_idx: Int32,
    ):
        for iter in cutlass.range(self.iterations_pv, unroll_full=True):
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            cO = cO_staged[None, None, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
            thr_load = tmem_tiled_load.get_slice(thread_idx)
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
            thr_store = tmem_store_atom.get_slice(thread_idx)
            tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_load.partition_D(cO_epi)
            tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
            iter_num = cute.size(tTMEM_LOADtO, mode=[1])
            # A ring of `depth` register fragments keeps depth-1 TMEM loads in
            # flight ahead of the multiply/store stream. Every element is still
            # multiplied by the same scale exactly once.
            depth = 4 if cutlass.const_expr(iter_num >= 4) else 2
            tTMrO = cute.make_rmem_tensor_like(
                cute.append(
                    cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                    cute.make_layout(depth, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                ),
                self.pv_acc_dtype,
            )
            for d in cutlass.range_constexpr(depth - 1):
                cute.copy(tmem_tiled_load, tTMEM_LOADtO[None, d, 0], tTMrO[None, d])
            for i in cutlass.range(depth - 1, iter_num, unroll_full=True):
                cute.copy(tmem_tiled_load, tTMEM_LOADtO[None, i, 0], tTMrO[None, i % depth])
                prev = i - (depth - 1)
                for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                    tTMrO[j, prev % depth], tTMrO[j + 1, prev % depth] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j, prev % depth], tTMrO[j + 1, prev % depth]),
                        (scale, scale),
                    )
                cute.copy(tmem_store_atom, tTMrO[None, prev % depth], tTMEM_STOREtO[None, prev, 0])
            for i in cutlass.range(iter_num - depth + 1, iter_num, unroll_full=True):
                for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                    tTMrO[j, i % depth], tTMrO[j + 1, i % depth] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j, i % depth], tTMrO[j + 1, i % depth]),
                        (scale, scale),
                    )
                cute.copy(tmem_store_atom, tTMrO[None, i % depth], tTMEM_STOREtO[None, i, 0])

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (seqlen_q, scale_output) = value_args
        (sum_consumer, sSum) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
        scale = scale_output / row_sum if not row_sum_is_zero_or_nan else 0.0
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv):
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale, scale),
                    )
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if cute.elem_less(tTMEM_LOADcO_i[0][0], seqlen_q):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
        o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def store_sum_max(
        self,
        row_max,
        mLSE,
        row_sum,
        sSum,
        sum_producer,
        current_block_coord,
        seqlen_q,
        scale_softmax,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_producer.acquire_and_advance()
        sSum[thread_idx] = row_sum
        cute.arch.fence_view_async_shared()
        sum_handle.commit()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum

        if cutlass.const_expr(mLSE is not None):
            q_idx = current_block_coord[0] * self.cta_tiler[0] + tidx
            lse_value = (
                scale_softmax * row_max + cute.math.log(row_sum, fastmath=True)
                if not row_sum_is_zero_or_nan
                else -Float32.inf
            )
            if cute.elem_less(q_idx, seqlen_q):
                mLSE[q_idx, current_block_coord[2]] = lse_value
        return sum_producer
