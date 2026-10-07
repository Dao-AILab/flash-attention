# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# SM120 uses warp-level mma.sync: m16n8k16 for FP16/BF16 and m16n8k32 for E4M3.
# Its 99-KiB shared-memory limit also bounds the FP8 P-layout conversion and BF16
# epilogue. The mainloop remains in the SM80-derived forward class.

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils as utils_basic
from cutlass.base_dsl.arch import Arch
import cuda.bindings.driver as cuda

from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned


class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    supports_fp8 = True

    def __init__(self, *args, **kwargs):
        """Force SM80 code paths while the DSL still targets the resident SM120 GPU."""
        super().__init__(*args, **kwargs)
        self.arch = Arch.sm_80

    def _get_tiled_mma(self):
        if not self.is_fp8:
            return super()._get_tiled_mma()
        def make_mma():
            return cute.make_tiled_mma(
                warp.MmaFP8Op(self.dtype, Float32, (16, 8, 32)),
                (self.num_threads // 32, 1, 1),
                permutation_mnk=(self.num_threads // 32 * 16, 16, 32),
            )
        return make_mma(), make_mma()

    def _get_smem_layout_atom(self):
        q, k, v, o, p = super()._get_smem_layout_atom()
        if self.is_fp8:
            p = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_n)
        return q, k, v, o, p

    def _setup_attributes(self):
        super()._setup_attributes()
        if self.is_fp8:
            # Packed V is K-major for the PV instruction: (D_v, sequence).
            vt = cute.tile_to_shape(
                sm80_utils.get_smem_layout_atom(self.dtype, self.tile_n),
                (self.tile_hdimv, self.tile_n, 1), (0, 1, 2),
            )
            self.sV_layout = cute.make_composed_layout(
                vt.inner, vt.offset, cute.select(vt.outer, mode=[1, 0, 2])
            )
            self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=64),
                cute.make_ordered_layout((self.tile_n // 8, self.num_threads // (self.tile_n // 8)), order=(0, 1)),
                cute.make_layout((8, 1)),
            )

    def _get_shared_storage_cls(self):
        if not self.is_fp8:
            return super()._get_shared_storage_cls()
        # O reuses the whole mainloop workspace only after its final barrier.
        # Its BF16 width must not be charged as an extra live Q allocation.
        align = lambda n: (n + 1023) // 1024 * 1024
        self.sK_offset = align(cute.cosize(self.sQ_layout))
        self.sV_offset = self.sK_offset + align(cute.cosize(self.sK_layout))
        self.sP_offset = self.sV_offset + align(cute.cosize(self.sV_layout))
        size = max(self.sP_offset + align(cute.cosize(self.sP_layout)),
                   align(cute.cosize(self.sO_layout) * 2))
        assert size <= utils_basic.get_smem_capacity_in_bytes("sm_120")

        @cute.struct
        class SharedStorage:
            buffer: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, size], 1024]

        return SharedStorage

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
        """Check if the kernel can be implemented on SM120.

        Same logic as SM80 but uses SM120's shared memory capacity (99 KB).
        """
        if dtype == cutlass.Float8E4M3FN:
            if (min(head_dim, head_dim_v, tile_m, tile_n, num_threads) <= 0
                    or head_dim % 16 or head_dim_v % 16
                    or max(head_dim, head_dim_v) > 256
                    or tile_m % 16 or tile_n % 32 or num_threads % 32 or num_threads > 1024
                    or num_stages != 1 or Q_in_regs
                    or (tile_m * 2) % num_threads
                    or tile_n // 8 > num_threads or num_threads % (tile_n // 8)):
                return False
            d, dv = ((value + 31) // 32 * 32 for value in (head_dim, head_dim_v))
            align = lambda n: (n + 1023) // 1024 * 1024
            live = sum(align(n) for n in (tile_m*d, tile_n*d, tile_n*dv, tile_m*tile_n))
            return max(live, align(tile_m*dv*2)) <= utils_basic.get_smem_capacity_in_bytes("sm_120")
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
        # Shared memory usage: Q tile + (K tile + V tile)
        smem_usage_Q = tile_m * head_dim * 2
        smem_usage_K = tile_n * head_dim * num_stages * 2
        smem_usage_V = tile_n * head_dim_v * num_stages * 2
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
        # SM120 has 99 KB shared memory (vs 163 KB on SM80)
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True


class PackVSm120:
    """Byte-preserving 64x32 transpose; packed rows are padded and zero-filled.

    One packed buffer per KV head (not per query head), with each varlen batch
    independently based at a 64-byte sequence boundary for aligned PV loads.
    """

    @cute.jit
    def __call__(self, mV: cute.Tensor, mPacked: cute.Tensor,
                 mCuSeqlens: cute.Tensor | None, mSeqUsed: cute.Tensor | None,
                 stream: cuda.CUstream = None):
        mV = assume_tensor_aligned(mV)
        mPacked = cute.make_tensor(mPacked.iterator, cute.make_layout(mPacked.shape, stride=tuple(
            s if i == 1 or isinstance(s, int) else cute.assume(s, divby=64)
            for i, s in enumerate(mPacked.stride))))
        layout = cute.tile_to_shape(sm80_utils.get_smem_layout_atom(cutlass.Uint8, 32),
                                    (64, 32), (0, 1))
        load = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Uint8, num_bits_per_copy=128),
            cute.make_ordered_layout((64, 2), order=(1, 0)), cute.make_layout((1, 16)),
        )
        store = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64),
            cute.make_ordered_layout((8, 16), order=(0, 1)), cute.make_layout((8, 1)),
        )
        self.kernel(mV, mPacked, mCuSeqlens, mSeqUsed, layout, load, store).launch(
            grid=(cute.ceil_div(mPacked.shape[1], 64), cute.ceil_div(mPacked.shape[3], 32),
                  mPacked.shape[0] * mPacked.shape[2]),
            block=(128, 1, 1), smem=cute.cosize(layout), stream=stream,
        )

    @cute.kernel
    def kernel(self, mV: cute.Tensor, mPacked: cute.Tensor,
               mCuSeqlens: cute.Tensor | None, mSeqUsed: cute.Tensor | None,
               layout: cute.ComposedLayout, load: cute.TiledCopy, store: cute.TiledCopy):
        tid, _, _ = cute.arch.thread_idx()
        seq_tile, dim_tile, bh = cute.arch.block_idx()
        batch, head = bh // mPacked.shape[2], bh % mPacked.shape[2]
        if const_expr(mCuSeqlens is None):
            source = mV[batch, None, head, None]
            length = Int32(mV.shape[1])
        else:
            begin = mCuSeqlens[batch]
            length = mCuSeqlens[batch + 1] - begin
            source = cute.domain_offset((begin, 0), mV[None, head, None])
        if const_expr(mSeqUsed is not None):
            length = mSeqUsed[batch]
        target = mPacked[batch, None, head, None]
        gSrc = cute.local_tile(source, (64, 32), (seq_tile, dim_tile))
        gDst = cute.local_tile(target, (64, 32), (seq_tile, dim_tile))
        smem = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Uint8, layout.outer, byte_alignment=128, swizzle=layout.inner)
        loader, storer = load.get_slice(tid), store.get_slice(tid)
        tSrc, tShared = loader.partition_S(gSrc), loader.partition_D(smem)
        coords = loader.partition_S(cute.make_identity_tensor((64, 32)))
        pred = cute.make_rmem_tensor((1, coords.shape[1], coords.shape[2]), cutlass.Boolean)
        for m in cutlass.range(cute.size(pred.shape[1]), unroll_full=True):
            for n in cutlass.range(cute.size(pred.shape[2]), unroll_full=True):
                row, dim = coords[0, m, n]
                pred[0, m, n] = row + seq_tile*64 < length and dim + dim_tile*32 < mPacked.shape[3]
        zeros = cute.make_rmem_tensor(tShared.shape, cutlass.Uint8)
        zeros.fill(0)
        cute.autovec_copy(zeros, tShared)
        cute.arch.barrier()
        cute.copy(load, tSrc, tShared, pred=pred)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        tRead, tDst = storer.partition_S(smem), storer.partition_D(gDst)
        values = cute.make_rmem_tensor(tRead.shape, cutlass.Uint8)
        cute.autovec_copy(tRead, values)
        out_coords = storer.partition_D(cute.make_identity_tensor((64, 32)))
        out_pred = cute.make_rmem_tensor((1, out_coords.shape[1], out_coords.shape[2]), cutlass.Boolean)
        for m in cutlass.range(cute.size(out_pred.shape[1]), unroll_full=True):
            for n in cutlass.range(cute.size(out_pred.shape[2]), unroll_full=True):
                out_pred[0, m, n] = out_coords[0, m, n][1] + dim_tile*32 < mPacked.shape[3]
        cute.copy(store, values, tDst, pred=out_pred)
