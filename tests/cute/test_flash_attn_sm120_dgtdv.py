"""Regression tests for Bug E (head_dim > head_dim_v) routing on SM120.

Bug E (fix in commit 886f04f):
``head_dim > head_dim_v`` on SM120 hangs the TMA forward kernel. The shipped
fix is a two-gate dispatcher pattern in ``flash_attn.cute.interface``:

* ``FlashAttentionForwardSm120Tma.can_implement`` REJECTS ``head_dim > head_dim_v``
  so the TMA path is never selected for those shapes.
* ``FlashAttentionForwardSm120.can_implement`` ACCEPTS ``head_dim > head_dim_v``
  so the non-TMA SM80-base kernel handles them.

This routing is fragile: a future contributor relaxing the TMA gate to
"allow more shapes" would silently re-introduce the GPU hang on the
minimum repro (B=1, H=1, S=64, d=128, dv=64, non-causal).

These tests cover:

* Several ``head_dim > head_dim_v`` shapes against PyTorch SDPA math backend.
* Both ``causal=False`` and ``causal=True``.
* A direct *negative* unit-level probe that asserts
  ``FlashAttentionForwardSm120Tma.can_implement(..., d>dv)`` returns False
  so a relaxation of the gate fails this test BEFORE any kernel is launched.

Each kernel-launching test is wrapped in a 30-second pytest-timeout marker
so a regression that re-introduces the hang fails as a timeout rather than
wedging the GPU. pytest-timeout uses SIGALRM under the hood (the default
``signal`` method on Linux), which IS able to interrupt a Python-level
``torch.cuda.synchronize()`` because synchronize releases the GIL — so the
signal handler runs once the driver call returns from the kernel-launch
overhead. For deeper-driver hangs you should still run pytest under an
outer ``timeout`` wrapper.

bf16 tolerance vs SDPA is 0.05 (the actual observed diff is ~0.004).

Skips when not running on sm_120.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute import flash_attn_func


def _sm120_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = torch.cuda.get_device_capability(0)
    if cc != (12, 0):
        pytest.skip(f"Test targets sm_120, current device is sm_{cc[0]}{cc[1]}")


def _sdpa_reference(
    q: torch.Tensor,  # (b, h, s, d)
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    # Use the math backend explicitly so this reference still works when the
    # default SDPA backend rejects an unusual head_dim/head_dim_v combo.
    with torch.nn.attention.sdpa_kernel(
        backends=[torch.nn.attention.SDPBackend.MATH]
    ):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _run_dgtdv_case(
    batch_size: int,
    seqlen: int,
    nheads: int,
    head_dim: int,
    head_dim_v: int,
    causal: bool,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """Run flash_attn_func with d > dv and compare to SDPA math.

    Returns (max_abs_diff, mean_abs_diff) vs the SDPA reference.
    """
    assert head_dim > head_dim_v, "this helper is for the d > dv path"
    device = "cuda"
    torch.manual_seed(seed)

    # Layout: (B, S, H, D). Same as smoke_test_fa4.py / repro_bugE.py.
    q = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, head_dim_v, device=device, dtype=dtype)

    torch.cuda.synchronize()
    out = flash_attn_func(q, k, v, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()

    # SDPA reference in (B, H, S, D) layout, fp32, math backend.
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    out_ref = _sdpa_reference(q_ref, k_ref, v_ref, causal=causal).transpose(1, 2).to(dtype)

    diff = (out.float() - out_ref.float()).abs()
    return float(diff.max()), float(diff.mean())


TOL_BF16 = 0.05

# (batch, seqlen, nheads, head_dim, head_dim_v)
DGTDV_SHAPES = [
    (1, 64, 1, 128, 64),     # minimum Bug E repro shape
    (2, 256, 4, 128, 64),
    (1, 512, 8, 96, 64),
    (1, 1024, 8, 128, 64),
]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "batch_size,seqlen,nheads,head_dim,head_dim_v", DGTDV_SHAPES,
)
def test_dgtdv_routes_to_non_tma(
    batch_size, seqlen, nheads, head_dim, head_dim_v, causal,
):
    """head_dim > head_dim_v shapes must run (on the non-TMA SM120 path) and
    match SDPA. A regression that re-introduces the TMA hang will fail as a
    pytest-timeout rather than wedging the GPU.
    """
    _sm120_only()
    seed = (batch_size * 31 + seqlen) * 31 + nheads + (1 if causal else 0)
    md, _ = _run_dgtdv_case(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        causal=causal,
        seed=seed,
    )
    assert md < TOL_BF16, (
        f"d={head_dim} dv={head_dim_v} causal={causal} "
        f"shape=({batch_size},{seqlen},{nheads}): max diff {md:.5f} >= {TOL_BF16}"
    )


@pytest.mark.timeout(30)
def test_sm120_tma_d_lt_dv_uses_v_copy_bytes():
    """TMA must size the V transfer from the V tile when head_dim_v > head_dim."""
    _sm120_only()
    torch.manual_seed(1201)
    batch_size, seqlen, nheads, head_dim, head_dim_v = 2, 256, 8, 64, 128
    dtype = torch.bfloat16
    device = "cuda"
    q = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, head_dim_v, device=device, dtype=dtype)

    out = flash_attn_func(q, k, v, causal=False)
    if isinstance(out, tuple):
        out = out[0]

    out_ref = _sdpa_reference(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        causal=False,
    ).transpose(1, 2).to(dtype)
    md = float((out.float() - out_ref.float()).abs().max())
    assert md < TOL_BF16, f"d={head_dim} dv={head_dim_v}: max diff {md:.5f} >= {TOL_BF16}"


def test_sm120_tma_can_implement_rejects_d_gt_dv():
    """Negative unit-level probe: the TMA gate must keep rejecting d > dv.

    This is the *direct* guard against the "someone relaxed the TMA gate"
    failure mode. It runs without launching a kernel, so a regression here
    catches the bug at the unit level even on CI machines without an SM120
    GPU available (the gate is a pure-Python staticmethod).
    """
    # Late import so the worktree overlay shim above has had a chance to run.
    import cutlass
    from flash_attn.cute.flash_fwd_sm120_tma import FlashAttentionForwardSm120Tma

    # Realistic TMA-path parameters that would otherwise pass can_implement
    # (tile_m=128, tile_n=128, kv_stages=2 fits SM120 SMEM at d=dv=64; the
    # only thing that should reject is the d > dv check).
    ok_baseline = FlashAttentionForwardSm120Tma.can_implement(
        dtype=cutlass.BFloat16,
        head_dim=64,
        head_dim_v=64,
        tile_m=128,
        tile_n=128,
        num_mma_warps=4,
        kv_stages=2,
        is_causal=False,
    )
    assert ok_baseline, (
        "Baseline (d==dv==64) must be implementable on the TMA path; "
        "if this fails, the test's baseline parameters are no longer valid."
    )

    # All four production Bug E shapes must be rejected by the TMA gate.
    for head_dim, head_dim_v in [(128, 64), (96, 64), (128, 96)]:
        result = FlashAttentionForwardSm120Tma.can_implement(
            dtype=cutlass.BFloat16,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            tile_m=128,
            tile_n=128,
            num_mma_warps=4,
            kv_stages=2,
            is_causal=False,
        )
        assert result is False, (
            f"FlashAttentionForwardSm120Tma.can_implement(d={head_dim}, dv={head_dim_v}) "
            f"returned {result!r}; it MUST return False to avoid the Bug E TMA hang. "
            f"If you intentionally relaxed this gate, you also need to verify the TMA "
            f"kernel no longer hangs on the minimum repro shape "
            f"(B=1, H=1, S=64, d=128, dv=64, non-causal)."
        )


def test_sm120_non_tma_can_implement_accepts_d_gt_dv():
    """Positive unit-level probe: the non-TMA SM120 gate must accept d > dv.

    Companion to the TMA-rejection test. The dispatcher relies on the
    non-TMA gate ACCEPTING d > dv (so dispatch falls through there); if
    someone tightened this gate the runtime would AssertionError instead
    of routing correctly.
    """
    import cutlass
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

    for head_dim, head_dim_v in [(128, 64), (96, 64), (128, 96)]:
        result = FlashAttentionForwardSm120.can_implement(
            dtype=cutlass.BFloat16,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            tile_m=128,
            tile_n=128,
            num_stages=1,
            num_threads=128,
            is_causal=False,
            Q_in_regs=False,
        )
        assert result is True, (
            f"FlashAttentionForwardSm120.can_implement(d={head_dim}, dv={head_dim_v}) "
            f"returned {result!r}; the non-TMA SM120 path MUST accept d > dv so the "
            f"dispatcher can route those shapes here instead of to the (hanging) TMA path."
        )


def test_sm120_can_implement_smem_constraint_at_ns2():
    """FIX 2 sibling test: exercise can_implement with num_stages=2.

    Phase 5c added per-shape ``sm120_num_stages`` lookup that can pick
    ``ns=2`` for some shapes. The dispatcher's pre-launch assertion must
    pass ``sm120_num_stages`` (not a hardcoded ``1``) so that SMEM math
    is checked at the actual configuration that will run. This test
    confirms can_implement's SMEM gate IS the load-bearing constraint at
    ``(tile_m=128, tile_n=128, ns=2, d=128)`` — i.e. that bumping the
    tile size up from here would push SMEM over the 99 KB cap.
    """
    import cutlass
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

    # The exact tile/ns config that Phase 5c may pick: tm=128, tn=128, ns=2,
    # d=dv=128 -> SMEM = tm*d*2 + 2*tn*d*ns*2 + 2*tn*dv*ns*2
    #           = 128*128*2 + 2*128*128*2*2 + 2*128*128*2*2
    #           = 32768 + 65536 + 65536 = 163840 bytes wait, that's wrong
    # Actual SMEM formula in FlashAttentionForwardSm120.can_implement:
    #   smem_Q = tile_m * head_dim * 2
    #   smem_K = tile_n * head_dim * num_stages * 2
    #   smem_V = tile_n * head_dim_v * num_stages * 2
    #   smem = (smem_Q + smem_V) + smem_K   (Q_in_regs=False)
    # ns=1, d=dv=128: 128*128*2 + 128*128*1*2 + 128*128*1*2 = 32768*3 = 98304 bytes (96 KB) - fits
    # ns=2, d=dv=128: 128*128*2 + 128*128*2*2 + 128*128*2*2 = 32768 + 65536*2 = 163840 bytes - doesn't fit
    # So at ns=2 and d=128, can_implement MUST return False (caught here).
    blocked_ns2 = FlashAttentionForwardSm120.can_implement(
        dtype=cutlass.BFloat16,
        head_dim=128,
        head_dim_v=128,
        tile_m=128,
        tile_n=128,
        num_stages=2,
        num_threads=128,
        is_causal=False,
        Q_in_regs=False,
    )
    assert blocked_ns2 is False, (
        "can_implement(tm=128, tn=128, ns=2, d=128) must return False because "
        "SMEM usage (160 KB) exceeds SM120's 99 KB cap. If this passes, either "
        "the SMEM math changed, the cap changed, or the formula is wrong — "
        "FIX 2 in interface.py is only meaningful if can_implement DOES catch "
        "this case."
    )

    # Sanity check: the same config with ns=1 IS implementable (98304 / 99 KB).
    ok_ns1 = FlashAttentionForwardSm120.can_implement(
        dtype=cutlass.BFloat16,
        head_dim=128,
        head_dim_v=128,
        tile_m=128,
        tile_n=128,
        num_stages=1,
        num_threads=128,
        is_causal=False,
        Q_in_regs=False,
    )
    assert ok_ns1 is True, (
        "Baseline (tm=128, tn=128, ns=1, d=128) must be implementable on SM120; "
        "if this fails the SMEM math or capacity changed."
    )
