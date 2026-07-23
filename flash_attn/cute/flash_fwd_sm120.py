# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has
# a smaller shared memory capacity (99 KB vs 163 KB on SM80). This module subclasses
# FlashAttentionForwardSm80 and overrides the SMEM capacity check accordingly.

import cutlass
import cutlass.utils as utils_basic
from cutlass.base_dsl.arch import Arch

from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80


class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    # Marker for arch-gated logic inside the SM80-shared forward body. self.arch
    # is forced to Arch.sm_80 below (so the SM80 epilogue/MMA paths are used), so
    # the backward's `arch == 120` idiom does not work in the forward; gate sm120-
    # only forward behavior on this flag instead. Base class defaults False.
    is_sm120: bool = True

    def __init__(self, *args, **kwargs):
        """Force SM80 code paths while the DSL still targets the resident SM120 GPU."""
        super().__init__(*args, **kwargs)
        # Override arch to sm_80 so that __call__ uses CpAsync (not TMA) for the O epilogue.
        # BaseDSL._get_dsl().get_arch_enum() returns the real GPU arch (sm_121a on DGX Spark),
        # but SM120 must use the SM80 epilogue path (no TMA-O support in this kernel variant).
        self.arch = Arch.sm_80

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
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        # NOTE: head_dim > head_dim_v works fine on this SM80-base non-TMA
        # path. The previous Bug E hang lives in FlashAttentionForwardSm120Tma
        # (which still rejects head_dim > head_dim_v in its can_implement);
        # the dispatcher falls through to this non-TMA path when the TMA
        # path refuses, so d > dv shapes are handled here.
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
