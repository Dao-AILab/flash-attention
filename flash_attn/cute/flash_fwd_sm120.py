# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# SM120 uses warp-level mma.sync: m16n8k16 for FP16/BF16 and m16n8k32 for E4M3.
# Its 99-KiB shared-memory limit also bounds the FP8 P-layout conversion and BF16
# epilogue. The mainloop remains in the SM80-derived forward class.

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.nvgpu import warp
import cutlass.utils as utils_basic
from cutlass.base_dsl.arch import Arch

from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute import ampere_helpers as sm80_utils


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

    @cute.jit
    def gemm_pv_fp8(self, tiled_mma, acc, tCrA, tCrB, tCsA, tCsB,
                    smem_thr_copy_A, smem_thr_copy_B):
        tCrA_copy = smem_thr_copy_A.retile(tCrA)
        # ldmatrix.b8 returns (byte, n8, k16), while MMA B owns
        # (byte, k16, n8). A register view selects the destinations directly.
        paired = cute.logical_divide(tCrB, (None, 2, None))
        b, s = paired.shape, paired.stride
        tCrB_copy = cute.make_tensor(tCrB.iterator, cute.make_layout(
            (((4, 2, 2), 1), b[1][1], b[2]),
            stride=(((s[0][0], s[1][0], s[0][1]), 0), s[1][1], s[2]),
        ))
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy[None, None, 0])
        cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
            if k < cute.size(tCsA.shape[2]) - 1:
                cute.copy(smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy[None, None, k + 1])
                cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy[None, None, k + 1])
            cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

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
