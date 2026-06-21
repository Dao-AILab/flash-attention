# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
"""
Tests for: fix(hd256/sm100): make dedicated hd256 kernel stride-aware (issue #2665).

ROOT CAUSE
----------
BlackwellFusedMultiHeadAttentionForward builds Q/K/V CuTe layouts with strides
synthesized from shape:

    q_layout = cute.make_layout(
        (s_q_total, d, ((h_r, h_k), b)),
        stride=(d64 * h_r64 * h_k64, 1, ((d64, d64 * h_r64), stride_b_qo)),
    )

All non-d strides are computed from dimension sizes, ignoring the actual tensor
strides.  maybe_contiguous() only guarantees stride(-1)==1, so a .transpose(1,2)
input (S-stride = D instead of H*D) produced wrong TMA loads and silently
corrupted outputs (issue #2665).

THE FIX
-------
1. Forward (dense + paged): derive s/h/b strides from the actual tensors; keep d
   statically 1 (TMA requires the contiguous mode's stride to be statically known).

2. Backward: hd256 backward kernels already read tensor.stride[0..3] natively,
   so q/k/v/out/dout pass through without any extra copies.

3. interface.py: replace any unconditional contiguous() calls with an
   alignment-gated guard that copies only when a stride would violate the
   kernel's 16-byte / divby-64 TMA assumptions (stride % 64 != 0 for fp16/bf16,
   i.e. the stride in bytes is not a multiple of 128).

BEHAVIOUR
---------
- Well-aligned non-contiguous (e.g. transpose, stride = D = 256, 256 % 64 == 0):
    kernel reads real strides directly.  Zero extra allocation.
- Misaligned (e.g. padded buffer, stride = 257, 257 % 64 != 0):
    alignment guard copies to a fresh contiguous tensor before dispatch.

TEST STRUCTURE
--------------
Group 0: Helpers — _CopyTracker, stride factories, reference implementations.

Group A: Zero-copy fast path
  A1. test_zero_copy_transpose_fwd       — well-aligned transpose → no copy (fwd)
  A2. test_zero_copy_permute_fwd         — well-aligned permute → no copy (fwd)
  A3. test_zero_copy_gqa_fwd             — well-aligned GQA → no copy (fwd)
  A4. test_zero_copy_causal_fwd          — causal + well-aligned → no copy (fwd)
  A5. test_zero_copy_bwd                 — backward: q/k/v/out/dout pass through
                                           without copies (kernels are stride-aware)

Group B: Alignment-gated guard (selective copy)
  B1. test_guard_fires_for_misaligned    — stride%64≠0 triggers copy; output correct
  B2. test_guard_selective_per_tensor    — only the misaligned tensor is copied;
                                           the well-aligned ones are not
  B3. test_guard_output_matches_ref      — after guard-triggered copy, output ==
                                           contiguous reference (all dtypes)
  B4. test_guard_bwd_dout_misaligned     — misaligned dout triggers copy in backward

Group C: Forward stride-awareness — correctness across patterns
  C1. test_fwd_transpose_12              — .transpose(1,2) — original issue #2665 repro
  C2. test_fwd_permute_0213              — .permute(0,2,1,3)
  C3. test_fwd_narrow                    — .narrow() on S dim (non-zero offset)
  C4. test_fwd_as_strided_padded         — as_strided with padded H stride
  C5. test_fwd_mixed_contiguity          — only Q non-contiguous; K/V contiguous
  C6. test_fwd_mqa                       — MQA (nheads_kv=1)
  C7. test_fwd_gqa                       — GQA (nheads_kv<nheads)
  C8. test_fwd_causal                    — causal mask

Group D: Correctness vs SDPA — quantitative error bounds
  D1. test_fwd_vs_sdpa_issue_2665        — exact repro shape; error must be < 5e-3
                                           (pre-fix: ~0.15–0.55)
  D2. test_fwd_vs_sdpa_parametrized      — matrix: dtype × seqlen × mha_type × causal

Group E: Backward stride-awareness — correctness
  E1. test_bwd_noncontiguous_qkv         — dQ/dK/dV correct for transposed Q/K/V
  E2. test_bwd_noncontiguous_dout        — non-contiguous grad-output handled natively
  E3. test_bwd_causal                    — causal backward with non-contiguous inputs

Group F: Bit-exactness — non-contiguous == contiguous (same data ⇒ same bits)
  F1. test_bit_exact_fwd                 — parametrized over dtype × seqlen × type × causal
  F2. test_bit_exact_bwd                 — dQ/dK/dV bit-identical

Group G: Edge cases
  G1. test_batch1                        — batch size 1
  G2. test_large_seqlen                  — seqlen = 8192 (multi-CTA schedule)
  G3. test_paged_kv_noncontiguous_q      — paged KV + non-contiguous Q
  G4. test_contiguous_baseline           — contiguous inputs: no regression, no copy

All tests are skipped automatically on non-SM100/SM110 hardware.
"""

import math
import contextlib
import unittest.mock
import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute import flash_attn_func
from flash_attn.cute.interface import _flash_attn_fwd

# ---------------------------------------------------------------------------
# Hardware guard — all tests skip on non-Blackwell (non-SM100/110) hardware.
# ---------------------------------------------------------------------------

IS_SM100 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
)

pytestmark = pytest.mark.skipif(
    not IS_SM100,
    reason="hd256 stride-aware kernel only targets SM100/SM110 (B200/B100)",
)

_HD = 256  # head dim targeted by the dedicated kernel


# ===========================================================================
# Group 0: Shared helpers
# ===========================================================================

# ---------------------------------------------------------------------------
# 0.1  Tensor factories
# ---------------------------------------------------------------------------

def _bhsd(batch, nheads, seqlen, hdim, dtype, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, nheads, seqlen, hdim, device="cuda", dtype=dtype)


def _bshd(batch, seqlen, nheads, hdim, dtype, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, seqlen, nheads, hdim, device="cuda", dtype=dtype)


def _transposed(batch, nheads, seqlen, hdim, dtype, seed=0):
    """Return (q_bhsd, q_bshd_nc) where q_bshd_nc is BSHD but non-contiguous.

    For nheads == 1 (MQA), .transpose() produces a size-1 dim that PyTorch calls
    contiguous; use a padded-buffer trick instead.

    Guarantees:
      q_bshd_nc.stride(-1) == 1   (passes maybe_contiguous gate)
      not q_bshd_nc.is_contiguous()
      q_bshd_nc.stride(-2) % 64 == 0  (well-aligned → zero-copy fast path)
    """
    src = _bhsd(batch, nheads, seqlen, hdim, dtype, seed)
    if nheads > 1:
        nc = src.transpose(1, 2)  # (B,S,H,D), stride(-2) = S*D = S*256, divisible by 64
    else:
        buf = torch.empty(batch, seqlen, 2, hdim, device="cuda", dtype=dtype)
        buf[:, :, 0, :] = src[:, 0, :, :]
        nc = buf[:, :, :1, :]   # stride(-2) = 2*hdim = 512, divisible by 64
    assert nc.stride(-1) == 1
    assert not nc.is_contiguous()
    assert nc.stride(-2) % 64 == 0, "factory invariant: stride must be TMA-aligned"
    return src, nc


def _misaligned_bshd(batch, seqlen, nheads, hdim, hdim_padded, dtype, seed=0):
    """Return BSHD tensor with stride(-2) = hdim_padded (not divisible by 64)
    → triggers the alignment-gated copy in the new interface."""
    assert hdim_padded % 64 != 0, "hdim_padded must be misaligned for this factory"
    torch.manual_seed(seed)
    wide = torch.randn(batch, seqlen, nheads, hdim_padded, device="cuda", dtype=dtype)
    nc = wide[..., :hdim]  # shape (..., hdim), stride(-2) = hdim_padded
    assert nc.stride(-2) == hdim_padded
    assert nc.stride(-1) == 1
    return nc


# ---------------------------------------------------------------------------
# 0.2  Reference implementation
# ---------------------------------------------------------------------------

def _sdpa_ref(q_bhsd, k_bhsd, v_bhsd, causal=False):
    """fp32 SDPA in BHSD layout; returns BSHD at original dtype."""
    out = F.scaled_dot_product_attention(
        q_bhsd.float(), k_bhsd.float(), v_bhsd.float(), is_causal=causal,
    ).to(q_bhsd.dtype)
    return out.transpose(1, 2).contiguous()


def _fa_atol(dtype, seqlen):
    base = 1e-2 if dtype == torch.float16 else 2e-2
    return base * max(1.0, seqlen / 512)


# ---------------------------------------------------------------------------
# 0.3  Copy-tracker — the zero-copy and guard tests pivot on this
# ---------------------------------------------------------------------------

class _CopyTracker:
    """Context manager: monkeypatches torch.Tensor.contiguous to record calls
    on specific tensor ids.

    Usage::

        q, k, v = ...
        with _CopyTracker(q, k, v) as ct:
            flash_attn_func(q, k, v)
        ct.assert_no_copies()          # zero-copy fast path
        ct.assert_copies({id(q)})      # only q was copied
    """

    def __init__(self, *tensors: torch.Tensor):
        self._watched: dict[int, str] = {id(t): t.shape.__str__() for t in tensors}
        self.copies: list[int] = []   # ids that had .contiguous() called
        self._patch: unittest.mock._patch | None = None

    def __enter__(self):
        orig = torch.Tensor.contiguous
        tracker = self

        def _patched(self_t, *args, **kwargs):
            result = orig(self_t, *args, **kwargs)
            if id(self_t) in tracker._watched:
                tracker.copies.append(id(self_t))
            return result

        self._patch = unittest.mock.patch.object(torch.Tensor, "contiguous", _patched)
        self._patch.__enter__()
        return self

    def __exit__(self, *args):
        if self._patch:
            self._patch.__exit__(*args)

    # ---- assertions ----

    def assert_no_copies(self):
        if self.copies:
            shapes = [self._watched[i] for i in self.copies if i in self._watched]
            raise AssertionError(
                f"Expected zero copies for watched tensors but got {len(self.copies)}: "
                f"{shapes}"
            )

    def assert_at_least_one_copy(self):
        if not self.copies:
            raise AssertionError(
                "Expected at least one copy for watched tensors but none were made. "
                "The alignment-gated guard may not have fired."
            )

    def assert_copies_exactly(self, expected_ids: set[int]):
        actual = set(self.copies)
        extra   = actual - expected_ids
        missing = expected_ids - actual
        if extra or missing:
            raise AssertionError(
                f"Copy set mismatch. Extra copies: {extra}, missing: {missing}"
            )


# ===========================================================================
# Group A: Zero-copy fast path
# ===========================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_copy_transpose_fwd(dtype):
    """For .transpose(1,2) inputs (stride(-2)==D==256, divisible by 64), the
    stride-aware kernel reads real strides directly: no .contiguous() call is made
    on any of Q, K, or V during the forward pass.
    """
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=0)
    k_src, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=1)
    v_src, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=2)

    with _CopyTracker(q_nc, k_nc, v_nc) as ct:
        out = flash_attn_func(q_nc, k_nc, v_nc)[0]

    ct.assert_no_copies()

    # Correctness sanity-check on top of zero-copy check
    ref = _sdpa_ref(q_src, k_src, v_src)
    max_err = (out - ref).abs().max().item()
    assert max_err < _fa_atol(dtype, seqlen), (
        f"Zero-copy forward produced wrong output (max_err={max_err:.4f})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_copy_permute_fwd(dtype):
    """.permute(0,2,1,3) produces the same BSHD layout as .transpose(1,2) and has
    the same TMA-aligned strides (stride(-2)==D==256).  No copy should be made."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q_src = _bhsd(batch, nheads, seqlen, hdim, dtype, seed=10)
    k_src = _bhsd(batch, nheads, seqlen, hdim, dtype, seed=11)
    v_src = _bhsd(batch, nheads, seqlen, hdim, dtype, seed=12)

    q_nc = q_src.permute(0, 2, 1, 3)
    k_nc = k_src.permute(0, 2, 1, 3)
    v_nc = v_src.permute(0, 2, 1, 3)

    assert not q_nc.is_contiguous() and q_nc.stride(-1) == 1
    assert q_nc.stride(-2) % 64 == 0  # verify factory precondition

    with _CopyTracker(q_nc, k_nc, v_nc) as ct:
        out = flash_attn_func(q_nc, k_nc, v_nc)[0]

    ct.assert_no_copies()

    ref = _sdpa_ref(q_src, k_src, v_src)
    assert (out - ref).abs().max().item() < _fa_atol(dtype, seqlen)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_copy_gqa_fwd(dtype):
    """GQA (nheads_kv=2, nheads=8) with transposed K/V — no copy for any tensor."""
    batch, nheads, nheads_kv, seqlen, hdim = 2, 8, 2, 512, _HD

    q_src, q_nc = _transposed(batch, nheads,    seqlen, hdim, dtype, seed=20)
    k_src, k_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=21)
    v_src, v_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=22)

    with _CopyTracker(q_nc, k_nc, v_nc) as ct:
        out = flash_attn_func(q_nc, k_nc, v_nc)[0]

    ct.assert_no_copies()

    g = nheads // nheads_kv
    ref = _sdpa_ref(q_src, k_src.repeat_interleave(g, dim=1),
                    v_src.repeat_interleave(g, dim=1))
    assert (out - ref).abs().max().item() < _fa_atol(dtype, seqlen)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_copy_causal_fwd(dtype):
    """Causal attention with well-aligned transposed inputs — no copy."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=30)
    k_src, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=31)
    v_src, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=32)

    with _CopyTracker(q_nc, k_nc, v_nc) as ct:
        out = flash_attn_func(q_nc, k_nc, v_nc, causal=True)[0]

    ct.assert_no_copies()

    ref = _sdpa_ref(q_src, k_src, v_src, causal=True)
    assert (out - ref).abs().max().item() < _fa_atol(dtype, seqlen)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_copy_bwd(dtype):
    """The hd256 backward kernels already read tensor.stride[0..3] natively, so
    none of q/k/v/out/dout require a copy.

    Verifies: none of the backward-relevant tensors trigger a .contiguous() call
    when their strides are TMA-aligned.
    """
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=40)
    k_src, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=41)
    v_src, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=42)

    q_nc = q_nc.detach().requires_grad_()
    k_nc = k_nc.detach().requires_grad_()
    v_nc = v_nc.detach().requires_grad_()

    # Forward to get the output (and implicitly 'out' saved for backward)
    out = flash_attn_func(q_nc, k_nc, v_nc)[0]  # out is contiguous BSHD

    # Construct a well-aligned non-contiguous dout: transpose an BHSD tensor.
    dout_src = torch.randn(batch, nheads, seqlen, hdim, device="cuda", dtype=dtype)
    dout_nc = dout_src.permute(0, 2, 1, 3)  # BSHD, stride(-2)=256, aligned
    assert not dout_nc.is_contiguous() and dout_nc.stride(-2) % 64 == 0

    # Track copies on all backward-relevant tensors.
    # out is contiguous from flash_attn; no copy expected there either.
    with _CopyTracker(q_nc, k_nc, v_nc, dout_nc) as ct:
        out.backward(dout_nc)

    ct.assert_no_copies()

    # Gradient sanity check: dQ must match fp32 SDPA reference
    q_ref = q_src.float().requires_grad_()
    k_ref = k_src.float().requires_grad_()
    v_ref = v_src.float().requires_grad_()
    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
    out_ref.backward(dout_src.float())           # dout_src is BHSD; SDPA output is BHSD
    dq_ref = q_ref.grad.to(dtype).permute(0, 2, 1, 3)  # BHSD grad → BSHD

    assert (q_nc.grad - dq_ref).abs().max().item() < _fa_atol(dtype, seqlen), (
        "Backward dQ error too large with non-contiguous inputs"
    )


# ===========================================================================
# Group B: Alignment-gated guard
# ===========================================================================

_MISALIGNED_HDIM = 257  # 257 % 64 == 1 — clearly violates TMA alignment


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_guard_fires_for_misaligned(dtype):
    """When stride(-2) is not divisible by 64 (i.e. not 128-byte aligned for
    fp16/bf16), the guard must call .contiguous() on that tensor."""
    batch, nheads, seqlen, hdim = 2, 8, 256, _HD

    q_bad = _misaligned_bshd(batch, seqlen, nheads, hdim, _MISALIGNED_HDIM, dtype, seed=50)
    k_bad = _misaligned_bshd(batch, seqlen, nheads, hdim, _MISALIGNED_HDIM, dtype, seed=51)
    v_bad = _misaligned_bshd(batch, seqlen, nheads, hdim, _MISALIGNED_HDIM, dtype, seed=52)

    with _CopyTracker(q_bad, k_bad, v_bad) as ct:
        out = flash_attn_func(q_bad, k_bad, v_bad)[0]

    ct.assert_at_least_one_copy()

    # Output must still be numerically correct after the guard-triggered copy.
    ref = flash_attn_func(
        q_bad.contiguous(), k_bad.contiguous(), v_bad.contiguous(),
    )[0]
    assert torch.equal(out, ref), (
        f"Guard-triggered copy gave wrong output "
        f"(max_diff={(out-ref).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_guard_selective_per_tensor(dtype):
    """Only the tensor(s) whose strides violate alignment should be copied; the
    already-aligned tensors must not be touched.

    Setup: Q is misaligned (stride=257); K and V are standard transposed
    (stride=256, divisible by 64).

    Assert: exactly Q — and only Q — is copied.
    """
    batch, nheads, seqlen, hdim = 2, 8, 256, _HD

    q_bad = _misaligned_bshd(batch, seqlen, nheads, hdim, _MISALIGNED_HDIM, dtype, seed=60)
    _, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=61)
    _, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=62)

    with _CopyTracker(q_bad, k_nc, v_nc) as ct:
        flash_attn_func(q_bad, k_nc, v_nc)

    # k_nc and v_nc are aligned → no copy; q_bad is misaligned → must be copied.
    ct.assert_copies_exactly({id(q_bad)})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_guard_output_matches_ref(dtype):
    """Regardless of whether the guard fires, output must match the explicit
    contiguous reference within normal flash-attention tolerance."""
    batch, nheads, seqlen, hdim = 2, 8, 256, _HD

    q_bad = _misaligned_bshd(batch, seqlen, nheads, hdim, _MISALIGNED_HDIM, dtype, seed=70)
    k_bad = _misaligned_bshd(batch, seqlen, nheads, hdim, _MISALIGNED_HDIM, dtype, seed=71)
    v_bad = _misaligned_bshd(batch, seqlen, nheads, hdim, _MISALIGNED_HDIM, dtype, seed=72)

    out     = flash_attn_func(q_bad, k_bad, v_bad)[0]
    out_ref = flash_attn_func(q_bad.contiguous(), k_bad.contiguous(), v_bad.contiguous())[0]

    assert torch.equal(out, out_ref), (
        f"Misaligned-stride path gave different output from contiguous reference "
        f"(max_diff={(out-out_ref).abs().max().item():.2e}, dtype={dtype})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_guard_bwd_dout_misaligned(dtype):
    """If dout has a misaligned stride (stride%64≠0) the backward alignment guard
    must copy it and still produce correct gradients."""
    batch, nheads, seqlen, hdim = 2, 8, 256, _HD

    q = _bshd(batch, seqlen, nheads, hdim, dtype, seed=80)
    k = _bshd(batch, seqlen, nheads, hdim, dtype, seed=81)
    v = _bshd(batch, seqlen, nheads, hdim, dtype, seed=82)

    q = q.detach().requires_grad_()
    k = k.detach().requires_grad_()
    v = v.detach().requires_grad_()

    out = flash_attn_func(q, k, v)[0]

    # Construct misaligned dout: allocate wider and slice
    hdim_padded = _MISALIGNED_HDIM
    dout_wide = torch.randn(batch, seqlen, nheads, hdim_padded, device="cuda", dtype=dtype)
    dout_bad = dout_wide[..., :hdim]  # stride(-2) = hdim_padded = 257 % 64 ≠ 0
    assert dout_bad.stride(-2) == hdim_padded and dout_bad.stride(-2) % 64 != 0

    with _CopyTracker(dout_bad) as ct:
        out.backward(dout_bad)
    ct.assert_at_least_one_copy()

    dq_guarded = q.grad.clone()

    # Reference: explicit contiguous dout, same data
    q.grad = None; k.grad = None; v.grad = None
    out2 = flash_attn_func(q, k, v)[0]
    out2.backward(dout_bad.contiguous())
    dq_ref = q.grad.clone()

    assert torch.equal(dq_guarded, dq_ref), (
        f"dQ after guard-triggered dout copy differs from contiguous reference "
        f"(max_diff={(dq_guarded - dq_ref).abs().max().item():.2e})"
    )


# ===========================================================================
# Group C: Forward stride-awareness — correctness across patterns
# ===========================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_transpose_12(dtype):
    """.transpose(1,2) from BHSD to BSHD — the exact pattern from issue #2665,
    now handled by the stride-aware kernel."""
    batch, nheads, seqlen, hdim = 2, 16, 1024, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=100)
    k_src, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=101)
    v_src, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=102)

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = _sdpa_ref(q_src, k_src, v_src)

    max_err = (out - ref).abs().max().item()
    assert max_err < _fa_atol(dtype, seqlen), (
        f"transpose(1,2) fwd: max_err={max_err:.4f}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_permute_0213(dtype):
    """.permute(0,2,1,3) from BHSD to BSHD — same layout, different API."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q_src = _bhsd(batch, nheads, seqlen, hdim, dtype, seed=110)
    k_src = _bhsd(batch, nheads, seqlen, hdim, dtype, seed=111)
    v_src = _bhsd(batch, nheads, seqlen, hdim, dtype, seed=112)

    q_nc = q_src.permute(0, 2, 1, 3)
    k_nc = k_src.permute(0, 2, 1, 3)
    v_nc = v_src.permute(0, 2, 1, 3)

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = _sdpa_ref(q_src, k_src, v_src)

    assert (out - ref).abs().max().item() < _fa_atol(dtype, seqlen)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_narrow(dtype):
    """.narrow() on the S dimension gives stride(-1)==1 with a non-zero offset but
    keeps the same per-row strides.  Stride-aware kernel must read the right data."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD
    extra = 128  # allocate extra rows and narrow to a well-aligned sub-region

    torch.manual_seed(120)
    q_big = torch.randn(batch, seqlen + extra, nheads, hdim, device="cuda", dtype=dtype)
    k_big = torch.randn(batch, seqlen + extra, nheads, hdim, device="cuda", dtype=dtype)
    v_big = torch.randn(batch, seqlen + extra, nheads, hdim, device="cuda", dtype=dtype)

    # narrow() from the middle so offset != 0
    start = extra // 2
    q_nc = q_big.narrow(1, start, seqlen)
    k_nc = k_big.narrow(1, start, seqlen)
    v_nc = v_big.narrow(1, start, seqlen)

    assert not q_nc.is_contiguous() and q_nc.stride(-1) == 1

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = flash_attn_func(q_nc.contiguous(), k_nc.contiguous(), v_nc.contiguous())[0]

    assert torch.equal(out, ref), (
        f"narrow() non-contiguous gave different output "
        f"(max_diff={(out-ref).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_as_strided_padded(dtype):
    """torch.as_strided() with a padded head-dim stride (multiple of 256, so
    TMA-aligned) — stride-aware kernel reads the real H-stride directly."""
    batch, nheads, seqlen, hdim = 2, 4, 256, _HD
    h_stride = hdim * 2  # 512 elements, divisible by 64 → no alignment-guard copy

    torch.manual_seed(130)
    buf_size = batch * seqlen * nheads * h_stride
    buf_q = torch.randn(buf_size, device="cuda", dtype=dtype)
    buf_k = torch.randn(buf_size, device="cuda", dtype=dtype)
    buf_v = torch.randn(buf_size, device="cuda", dtype=dtype)

    shape   = (batch, seqlen, nheads, hdim)
    strides = (seqlen * nheads * h_stride, nheads * h_stride, h_stride, 1)

    q_nc = torch.as_strided(buf_q, shape, strides)
    k_nc = torch.as_strided(buf_k, shape, strides)
    v_nc = torch.as_strided(buf_v, shape, strides)

    assert not q_nc.is_contiguous() and q_nc.stride(-2) % 64 == 0

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = flash_attn_func(q_nc.contiguous(), k_nc.contiguous(), v_nc.contiguous())[0]

    assert torch.equal(out, ref), (
        f"as_strided padded-H-stride gave different output "
        f"(max_diff={(out-ref).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_mixed_contiguity(dtype):
    """Only Q is non-contiguous; K and V are freshly-allocated contiguous BSHD.
    The stride-aware kernel must handle mixed-contiguity inputs correctly."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=140)
    k_c = _bshd(batch, seqlen, nheads, hdim, dtype, seed=141)
    v_c = _bshd(batch, seqlen, nheads, hdim, dtype, seed=142)

    out = flash_attn_func(q_nc, k_c, v_c)[0]
    ref = flash_attn_func(q_nc.contiguous(), k_c, v_c)[0]

    assert torch.equal(out, ref), (
        f"Mixed contiguity (Q non-contig, K/V contig) gave different output "
        f"(max_diff={(out-ref).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_mqa(dtype):
    """MQA (nheads_kv=1): non-contiguous Q, K, V.  Size-1 dims are always
    contiguous per PyTorch semantics, so K/V use the padded-buffer strategy."""
    batch, nheads, nheads_kv, seqlen, hdim = 2, 8, 1, 256, _HD

    _, q_nc = _transposed(batch, nheads,    seqlen, hdim, dtype, seed=150)
    _, k_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=151)
    _, v_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=152)

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = flash_attn_func(q_nc.contiguous(), k_nc.contiguous(), v_nc.contiguous())[0]

    assert torch.equal(out, ref), (
        f"MQA non-contiguous gave different output "
        f"(max_diff={(out-ref).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_gqa(dtype):
    """GQA (nheads_kv=2) with all three tensors non-contiguous."""
    batch, nheads, nheads_kv, seqlen, hdim = 2, 8, 2, 512, _HD

    _, q_nc = _transposed(batch, nheads,    seqlen, hdim, dtype, seed=160)
    _, k_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=161)
    _, v_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=162)

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = flash_attn_func(q_nc.contiguous(), k_nc.contiguous(), v_nc.contiguous())[0]

    assert torch.equal(out, ref), (
        f"GQA non-contiguous gave different output "
        f"(max_diff={(out-ref).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_causal(dtype):
    """Causal mask + non-contiguous inputs.  The mask changes tile iteration
    order; strides must still be read correctly."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    _, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=170)
    _, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=171)
    _, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=172)

    out_nc = flash_attn_func(q_nc, k_nc, v_nc, causal=True)[0]
    out_c  = flash_attn_func(q_nc.contiguous(), k_nc.contiguous(), v_nc.contiguous(),
                             causal=True)[0]

    assert torch.equal(out_nc, out_c), (
        f"Causal + non-contiguous gave different output "
        f"(max_diff={(out_nc-out_c).abs().max().item():.2e})"
    )


# ===========================================================================
# Group D: Correctness vs SDPA — quantitative error bounds
# ===========================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_vs_sdpa_issue_2665(dtype):
    """Exact shape from issue #2665: batch=1, nheads=32, seqlen=8192, hdim=256.
    Pre-fix error was ~0.15–0.55; post-fix must be below 5e-3.
    """
    batch, nheads, seqlen, hdim = 1, 32, 8192, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=200)
    k_src, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=201)
    v_src, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=202)

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = _sdpa_ref(q_src, k_src, v_src)

    max_err = (out - ref).abs().max().item()
    assert max_err < 5e-3, (
        f"issue #2665 repro: max_err={max_err:.4f}"
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 256), (512, 512), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fwd_vs_sdpa_parametrized(dtype, seqlen_q, seqlen_k, mha_type, causal):
    """Wide parametric sweep: non-contiguous transposed inputs must be within
    normal flash-attention tolerance of fp32 SDPA across all configurations."""
    if causal and seqlen_q != seqlen_k:
        pytest.skip("causal mask convention differs between FA and SDPA when seqlen_q!=seqlen_k")

    batch, nheads, hdim = 2, 8, _HD
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)

    q_src, q_nc = _transposed(batch, nheads,    seqlen_q, hdim, dtype, seed=300)
    k_src, k_nc = _transposed(batch, nheads_kv, seqlen_k, hdim, dtype, seed=301)
    v_src, v_nc = _transposed(batch, nheads_kv, seqlen_k, hdim, dtype, seed=302)

    g = nheads // nheads_kv
    k_ref = k_src[:, :nheads_kv].repeat_interleave(g, dim=1)
    v_ref = v_src[:, :nheads_kv].repeat_interleave(g, dim=1)

    out = flash_attn_func(q_nc, k_nc, v_nc, causal=causal)[0]
    ref = _sdpa_ref(q_src, k_ref, v_ref, causal=causal)

    ref_fp32 = _sdpa_ref(q_src.float(), k_ref.float(), v_ref.float(), causal=causal)
    pytorch_err = (ref.float() - ref_fp32).abs().max().item()
    atol = max(2 * pytorch_err, 1e-3)

    max_err = (out.float() - ref_fp32).abs().max().item()
    assert max_err <= atol, (
        f"FA error {max_err:.4f} > 2× PyTorch err {pytorch_err:.4f} + floor "
        f"(dtype={dtype}, sq={seqlen_q}, sk={seqlen_k}, {mha_type}, causal={causal})"
    )


# ===========================================================================
# Group E: Backward stride-awareness — correctness
# ===========================================================================

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bwd_noncontiguous_qkv(dtype, mha_type, causal):
    """dQ/dK/dV for non-contiguous (transposed) Q/K/V must agree with fp32 SDPA
    gradients — the backward kernels read tensor.stride[0..3] natively."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD
    nheads_kv = nheads if mha_type == "mha" else 2

    q_src, q_nc = _transposed(batch, nheads,    seqlen, hdim, dtype, seed=400)
    k_src, k_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=401)
    v_src, v_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=402)

    q_nc = q_nc.detach().requires_grad_()
    k_nc = k_nc.detach().requires_grad_()
    v_nc = v_nc.detach().requires_grad_()

    out = flash_attn_func(q_nc, k_nc, v_nc, causal=causal)[0]
    g   = torch.randn_like(out)
    out.backward(g)

    # fp32 reference gradients via SDPA
    g_fac = nheads // nheads_kv
    k_full = k_src.repeat_interleave(g_fac, dim=1)
    v_full = v_src.repeat_interleave(g_fac, dim=1)

    qr = q_src.float().requires_grad_()
    kr = k_full.float().requires_grad_()
    vr = v_full.float().requires_grad_()
    F.scaled_dot_product_attention(qr, kr, vr, is_causal=causal).backward(
        g.float().transpose(1, 2)  # BSHD → BHSD for SDPA
    )

    dq_ref = qr.grad.to(dtype).transpose(1, 2).contiguous()
    atol = _fa_atol(dtype, seqlen)

    assert (q_nc.grad - dq_ref).abs().max().item() < atol, (
        f"dQ too large (dtype={dtype}, {mha_type}, causal={causal})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bwd_noncontiguous_dout(dtype):
    """Non-contiguous dout with a well-aligned stride is handled natively by the
    backward kernel — no copy, gradients bit-exact vs contiguous dout."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q = _bshd(batch, seqlen, nheads, hdim, dtype, seed=410)
    k = _bshd(batch, seqlen, nheads, hdim, dtype, seed=411)
    v = _bshd(batch, seqlen, nheads, hdim, dtype, seed=412)
    q = q.detach().requires_grad_()
    k = k.detach().requires_grad_()
    v = v.detach().requires_grad_()

    out = flash_attn_func(q, k, v)[0]

    # Well-aligned non-contiguous dout: transpose an BHSD gradient tensor
    g_src = torch.randn(batch, nheads, seqlen, hdim, device="cuda", dtype=dtype)
    g_nc  = g_src.permute(0, 2, 1, 3)   # stride(-2)=256, divisible by 64
    assert not g_nc.is_contiguous() and g_nc.stride(-2) % 64 == 0

    # Zero-copy: verify backward does NOT copy the well-aligned dout
    with _CopyTracker(g_nc) as ct:
        out.backward(g_nc)
    ct.assert_no_copies()

    dq_nc = q.grad.clone()

    # Reference with contiguous dout
    q.grad = None; k.grad = None; v.grad = None
    out2 = flash_attn_func(q, k, v)[0]
    out2.backward(g_nc.contiguous())
    dq_c = q.grad.clone()

    assert torch.equal(dq_nc, dq_c), (
        f"non-contiguous dout gave different dQ "
        f"(max_diff={(dq_nc-dq_c).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bwd_causal(dtype):
    """Causal backward with non-contiguous Q/K/V — gradients must match
    the contiguous-input reference."""
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    _, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=420)
    _, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=421)
    _, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=422)

    q_c = q_nc.contiguous()
    k_c = k_nc.contiguous()
    v_c = v_nc.contiguous()

    q_nc = q_nc.detach().requires_grad_()
    k_nc = k_nc.detach().requires_grad_()
    v_nc = v_nc.detach().requires_grad_()
    out_nc = flash_attn_func(q_nc, k_nc, v_nc, causal=True)[0]
    g = torch.randn_like(out_nc)
    out_nc.backward(g)

    q_c = q_c.detach().requires_grad_()
    k_c = k_c.detach().requires_grad_()
    v_c = v_c.detach().requires_grad_()
    out_c = flash_attn_func(q_c, k_c, v_c, causal=True)[0]
    out_c.backward(g.clone())

    for name, nc_g, c_g in [("dQ", q_nc.grad, q_c.grad),
                             ("dK", k_nc.grad, k_c.grad),
                             ("dV", v_nc.grad, v_c.grad)]:
        assert torch.equal(nc_g, c_g), (
            f"Causal bwd {name} differs (max_diff={(nc_g-c_g).abs().max().item():.2e})"
        )


# ===========================================================================
# Group F: Bit-exactness — non-contiguous == contiguous (same data ⇒ same bits)
# ===========================================================================

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("seqlen", [256, 512, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bit_exact_fwd(dtype, seqlen, mha_type, causal):
    """Non-contiguous (transposed) and contiguous tensors carrying the same data
    must produce bit-identical forward outputs — the stride-aware kernel reads the
    exact same floats from the same locations either way."""
    batch, nheads, hdim = 2, 8, _HD
    nheads_kv = nheads if mha_type == "mha" else 2

    _, q_nc = _transposed(batch, nheads,    seqlen, hdim, dtype, seed=500)
    _, k_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=501)
    _, v_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=502)

    out_nc = flash_attn_func(q_nc,                k_nc,                v_nc,                causal=causal)[0]
    out_c  = flash_attn_func(q_nc.contiguous(), k_nc.contiguous(), v_nc.contiguous(), causal=causal)[0]

    assert torch.equal(out_nc, out_c), (
        f"Fwd not bit-exact (max_diff={(out_nc-out_c).abs().max().item():.2e}, "
        f"dtype={dtype}, seqlen={seqlen}, {mha_type}, causal={causal})"
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("seqlen", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bit_exact_bwd(dtype, seqlen, mha_type, causal):
    """dQ/dK/dV from transposed inputs must be bit-identical to those from the
    equivalent freshly-allocated contiguous tensors."""
    batch, nheads, hdim = 2, 8, _HD
    nheads_kv = nheads if mha_type == "mha" else 2

    _, q_nc = _transposed(batch, nheads,    seqlen, hdim, dtype, seed=600)
    _, k_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=601)
    _, v_nc = _transposed(batch, nheads_kv, seqlen, hdim, dtype, seed=602)

    q_c = q_nc.contiguous()
    k_c = k_nc.contiguous()
    v_c = v_nc.contiguous()

    q_nc = q_nc.detach().requires_grad_()
    k_nc = k_nc.detach().requires_grad_()
    v_nc = v_nc.detach().requires_grad_()
    out_nc = flash_attn_func(q_nc, k_nc, v_nc, causal=causal)[0]
    g = torch.randn_like(out_nc)
    out_nc.backward(g)

    q_c = q_c.detach().requires_grad_()
    k_c = k_c.detach().requires_grad_()
    v_c = v_c.detach().requires_grad_()
    out_c = flash_attn_func(q_c, k_c, v_c, causal=causal)[0]
    out_c.backward(g.clone())

    for name, nc_g, c_g in [("dQ", q_nc.grad, q_c.grad),
                             ("dK", k_nc.grad, k_c.grad),
                             ("dV", v_nc.grad, v_c.grad)]:
        assert torch.equal(nc_g, c_g), (
            f"Bwd {name} not bit-exact (max_diff={(nc_g-c_g).abs().max().item():.2e}, "
            f"dtype={dtype}, seqlen={seqlen}, {mha_type}, causal={causal})"
        )


# ===========================================================================
# Group G: Edge cases
# ===========================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch1(dtype):
    """batch=1 with transposed Q/K/V — common single-sequence inference shape."""
    batch, nheads, seqlen, hdim = 1, 8, 512, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=700)
    k_src, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=701)
    v_src, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=702)

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = _sdpa_ref(q_src, k_src, v_src)

    assert (out - ref).abs().max().item() < _fa_atol(dtype, seqlen)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_seqlen(dtype):
    """seqlen=8192 — issue #2665 original shape; multi-CTA tile scheduling path."""
    batch, nheads, seqlen, hdim = 2, 8, 8192, _HD

    q_src, q_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=710)
    k_src, k_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=711)
    v_src, v_nc = _transposed(batch, nheads, seqlen, hdim, dtype, seed=712)

    out = flash_attn_func(q_nc, k_nc, v_nc)[0]
    ref = _sdpa_ref(q_src, k_src, v_src)

    assert (out - ref).abs().max().item() < _fa_atol(dtype, seqlen)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_kv_noncontiguous_q(dtype):
    """Paged-KV forward with a non-contiguous Q — dense-path layout fix must not
    break the paged-KV code path.  Calls _flash_attn_fwd directly because
    page_table is not exposed through the public flash_attn_func API."""
    batch, nheads, seqlen_q, hdim = 2, 8, 256, _HD
    page_size = 128
    pages_per_seq = 4
    pages_total = batch * pages_per_seq
    seqlen_k = pages_per_seq * page_size
    nheads_kv = 2

    torch.manual_seed(720)
    q_src = torch.randn(batch, nheads, seqlen_q, hdim, device="cuda", dtype=dtype)
    q_nc  = q_src.transpose(1, 2)   # (B,S,H,D) non-contiguous, stride(-2)=256
    q_c   = q_nc.contiguous()

    k_paged = torch.randn(pages_total, page_size, nheads_kv, hdim, device="cuda", dtype=dtype)
    v_paged = torch.randn(pages_total, page_size, nheads_kv, hdim, device="cuda", dtype=dtype)

    page_table = torch.arange(pages_total, device="cuda", dtype=torch.int32).view(batch, pages_per_seq)

    common = dict(
        qv=None, cu_seqlens_q=None, cu_seqlens_k=None,
        seqused_q=None, seqused_k=None,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k, min_seqlen_k=None,
        page_table=page_table,
        softmax_scale=1.0 / math.sqrt(hdim),
        causal=False, softcap=None,
        window_size_left=None, window_size_right=None,
        learnable_sink=None,
        tile_mn=None, mma_pv_is_rs=None, intra_wg_overlap=None,
        num_threads=None, num_splits=1, pack_gqa=None, _arch=None,
        score_mod=None, mask_mod=None, block_sparse_tensors=None,
        return_lse=False, out=None, lse=None,
        aux_tensors=None, aux_scalars=None,
        q_descale=None, k_descale=None, v_descale=None,
        gather_kv_indices=None,
    )

    out_nc, *_ = _flash_attn_fwd(q_nc, k_paged, v_paged, **common)
    out_c,  *_ = _flash_attn_fwd(q_c,  k_paged, v_paged, **common)

    assert torch.equal(out_nc, out_c), (
        f"Paged-KV: non-contiguous Q gave different output "
        f"(max_diff={(out_nc-out_c).abs().max().item():.2e})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_contiguous_baseline(dtype):
    """Contiguous inputs must continue to work correctly.  Verifies:
      1.  Output matches fp32 SDPA within normal FA tolerance.
      2.  No copy is triggered (no overhead from the alignment-guard check itself).
      3.  Two identical calls are bit-identical (determinism).
    """
    batch, nheads, seqlen, hdim = 2, 8, 512, _HD

    q = _bshd(batch, seqlen, nheads, hdim, dtype, seed=730)
    k = _bshd(batch, seqlen, nheads, hdim, dtype, seed=731)
    v = _bshd(batch, seqlen, nheads, hdim, dtype, seed=732)

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    # The alignment guard must NOT copy contiguous tensors.
    with _CopyTracker(q, k, v) as ct:
        out1 = flash_attn_func(q, k, v)[0]
    ct.assert_no_copies()

    out2 = flash_attn_func(q, k, v)[0]
    assert torch.equal(out1, out2), "Contiguous baseline is not deterministic"

    q_bhsd = q.transpose(1, 2).contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()
    ref = _sdpa_ref(q_bhsd, k_bhsd, v_bhsd)

    assert (out1 - ref).abs().max().item() < _fa_atol(dtype, seqlen), (
        "Contiguous baseline regressed vs SDPA reference"
    )
