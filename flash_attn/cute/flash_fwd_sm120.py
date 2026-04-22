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
    # Class-level arch = 80 is overwritten by the base __init__ which
    # assigns self.arch = BaseDSL._get_dsl().get_arch_enum() (returns
    # sm_121a on SM120 hardware). The override in __init__ below takes
    # care of the instance attribute.
    arch = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Base Sm80 __init__ sets self.arch to the real GPU arch enum
        # (e.g. sm_121a on SM121a). Sm120 must use CpAsync output path
        # (no TMA-O on SM12x), which is gated on `self.arch >= Arch.sm_90`
        # inside Sm80.__call__. Force arch back to sm_80 so use_tma_O
        # resolves False.
        self.arch = Arch.sm_80
        # Sm80 base __init__ does not set `self.is_split_kv` (only Sm90
        # and Sm100 do). The shared Sm80 __call__ path references
        # `self.is_split_kv`, so it must exist on the instance. SM120 does
        # not yet support split-KV; default to False.
        self.is_split_kv = False
        # PackGQA on the Sm80 code path is broken: it is missing the
        # pack_gqa_layout transforms that Sm90.__call__ (L273) and
        # Sm100.__call__ (L504) apply before handing tensors to PackGQA
        # methods, and simply porting those three lines is not sufficient
        # on Sm80 because the Sm80 mainloop's tile sizing does not handle
        # the packed (qhead_per_kvhead, seqlen) layout (tile division
        # fails for qhead_per_kvhead that does not divide tile_m). A
        # proper Sm80 pack-GQA fix needs a follow-up touching the tile
        # scheduler. Disabling pack_gqa on Sm120 routes through the
        # non-packed path, which is functionally correct.
        self.pack_gqa = False

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
