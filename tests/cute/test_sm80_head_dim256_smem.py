# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Regression tests for the SM80-family (sm_80/sm_86/sm_87/sm_89) head-dim-aware forward
# tile heuristic (`_tile_size_fwd_sm80`) and the arch-aware `can_implement` SMEM
# capacity check (`smem_capacity_arch`).
#
# GPU-free: only exercises pure-Python tile-size selection and the `can_implement`
# static predicate directly -- no CUDA device or JIT compile involved.

import cutlass
import pytest

from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.interface import FwdConfig, _tile_size_fwd_sm80

FP16 = cutlass.Float16

# sm_86 / sm_89 (RTX 30xx/40xx) static SMEM budget in bytes; see SMEM_CAPACITY_MAP in
# cutlass.utils. sm_80 (A100) has a larger 166912 B budget.
SM86_SM89_SMEM_CAPACITY = 101376


def _smem_usage(tile_m, tile_n, head_dim, head_dim_v, num_stages=1):
    """Mirrors the SMEM formula in `FlashAttentionForwardBase.can_implement`."""
    smem_q = tile_m * head_dim * 2
    smem_k = tile_n * head_dim * num_stages * 2
    smem_v = tile_n * head_dim_v * num_stages * 2
    return smem_q + smem_k + smem_v


@pytest.mark.parametrize(
    "head_dim", [64, 96, 128, 160, 192, 200, 208, 224, 232, 240, 248, 256]
)
def test_tile_size_fwd_sm80_fits_sm86_sm89_budget(head_dim):
    """Regression test for the head_dim=256 causal/bf16 SMEM overflow on sm_86/sm_89.

    Before this fix, `_flash_attn_fwd` used a single hardcoded FwdConfig(128, 64, ...)
    for every head_dim on SM80, which needs 131072 B of SMEM at head_dim=256 -- over
    the 101376 B sm_86/sm_89 budget -- and crashed at kernel launch with a raw
    `cudaErrorInvalidValue: Allocated: 131072 bytes. Max: 101376 bytes.`.
    """
    cfg = _tile_size_fwd_sm80(head_dim, head_dim)
    usage = _smem_usage(cfg.m_block_size, cfg.n_block_size, head_dim, head_dim)
    assert usage <= SM86_SM89_SMEM_CAPACITY, (
        f"head_dim={head_dim}: {cfg} needs {usage} B, exceeds sm_86/sm_89's "
        f"{SM86_SM89_SMEM_CAPACITY} B budget"
    )


def test_old_hardcoded_config_reproduces_the_reported_crash_numbers():
    """Documents the exact crash this fix addresses: the old unconditional
    FwdConfig(128, 64, ...) needed exactly 131072 B at head_dim=256, matching the
    `cudaErrorInvalidValue` allocation size in the original bug report."""
    old_cfg = FwdConfig(128, 64, True, True)
    assert _smem_usage(old_cfg.m_block_size, old_cfg.n_block_size, 256, 256) == 131072


def test_can_implement_default_arch_is_sm_80_for_backward_compatibility():
    """Existing callers that don't pass `smem_capacity_arch` keep the historical sm_80
    (A100, 166912 B) behavior: the old head_dim=256/tile_n=64 config still passes
    against the default, but must fail once checked against the real sm_86/sm_89
    budget -- this is the gap the fix closes."""
    assert FlashAttentionForwardSm80.can_implement(FP16, 256, 256, 128, 64, 1, 128, True)
    assert not FlashAttentionForwardSm80.can_implement(
        FP16, 256, 256, 128, 64, 1, 128, True, smem_capacity_arch="sm_89"
    )
    assert not FlashAttentionForwardSm80.can_implement(
        FP16, 256, 256, 128, 64, 1, 128, True, smem_capacity_arch="sm_86"
    )


def test_can_implement_accepts_new_sm80_head_dim_256_tile_config_on_sm89():
    """The new SM80 tile heuristic's head_dim=256 config passes `can_implement` even
    when checked against the tighter sm_89 budget."""
    cfg = _tile_size_fwd_sm80(256, 256)
    assert FlashAttentionForwardSm80.can_implement(
        FP16, 256, 256, cfg.m_block_size, cfg.n_block_size, 1, 128, True,
        smem_capacity_arch="sm_89",
    )
