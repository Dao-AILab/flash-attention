"""Regression test for the SM120 dQ postprocess `stmatrix` bug.

`flash_attn.cute.flash_bwd_postprocess.FlashAttentionBackwardPostprocess` calls
`utils.get_smem_store_atom(self.arch, ...)` for the rmem->smem dQ store. On
SM120 `self.arch == 120 >= 90`, which selects the Hopper `stmatrix` atom — but
the underlying tiled MMA on SM120 is the SM80 `mma.sync.aligned.m16n8k16`,
whose output register layout does NOT match `stmatrix`'s expected layout.

The analogous forward bug was fixed in commit bc67a9c. This test guards the
backward analog: gradients must stay within bf16 tolerance of an fp32 SDPA
reference. With the bug present, dQ shows small but structured deviations
(only some elements change, only dQ — never dK/dV/out). Without the fix the
worst observed FA4-vs-fp32-SDPA / SDPA-bf16-vs-fp32-SDPA ratio is roughly the
same as with the fix on these shapes, but the dQ bits are NOT bitwise stable
across kernel revisions. The test below is therefore a conservative
tolerance-based regression: it catches gross scrambling (the kind the forward
bug produced before bc67a9c) and any regression that pushes us outside bf16
noise.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def _sm120_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = torch.cuda.get_device_capability(0)
    if cc != (12, 0):
        pytest.skip(f"SM120-only test (got sm_{cc[0]}{cc[1]})")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_bwd_dq_within_bf16_noise(dtype, D, causal):
    """FA4 backward gradients must stay within ~5x of an SDPA-of-same-dtype
    baseline (vs fp32 SDPA truth). The SM120-stmatrix-on-SM80-MMA bug would
    scramble bytes during the dQ rmem->smem transfer in the postprocess; with
    a sufficiently large blast radius this would push dQ well outside that
    band. Even when the data flow partially hides the scrambling (the
    pre-store s2r 1D load and the stmatrix store happen to share part of
    their layout), a regression that re-enables the broken path will show up
    as inflated max diffs.
    """
    # Single seqlen keeps the JIT-compile budget reasonable while still being
    # large enough to exercise the multi-block dQ postprocess path.
    S = 1024
    _sm120_only()

    from flash_attn.cute import flash_attn_func

    B, H = 2, 8
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)

    out = flash_attn_func(q, k, v, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    torch.manual_seed(1)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    # fp32 ground truth via SDPA-math
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    with sdpa_kernel([SDPBackend.MATH]):
        ref_out = F.scaled_dot_product_attention(
            qf.transpose(1, 2).contiguous(),
            kf.transpose(1, 2).contiguous(),
            vf.transpose(1, 2).contiguous(),
            is_causal=causal,
        ).transpose(1, 2).contiguous()
    ref_out.backward(grad_out.float())

    # Same-dtype SDPA-bf16/fp16 baseline (gives us the "noise floor")
    qb = q.detach().clone().requires_grad_(True)
    kb = k.detach().clone().requires_grad_(True)
    vb = v.detach().clone().requires_grad_(True)
    with sdpa_kernel([SDPBackend.MATH]):
        out_b = F.scaled_dot_product_attention(
            qb.transpose(1, 2).contiguous(),
            kb.transpose(1, 2).contiguous(),
            vb.transpose(1, 2).contiguous(),
            is_causal=causal,
        ).transpose(1, 2).contiguous()
    out_b.backward(grad_out)

    pairs = [
        ("dq", q.grad, qf.grad, qb.grad),
        ("dk", k.grad, kf.grad, kb.grad),
        ("dv", v.grad, vf.grad, vb.grad),
    ]
    PASS_RATIO = 5.0  # FA4 max-diff must be within 5x of SDPA-dtype max-diff
    failures = []
    for name, fa, ref32, refdtype in pairs:
        diff_fa = (fa.float() - ref32).abs()
        diff_baseline = (refdtype.float() - ref32).abs()
        fa_max = float(diff_fa.max())
        bl_max = float(diff_baseline.max())
        ratio = fa_max / max(bl_max, 1e-6)
        msg = (f"{name}: FA4 max={fa_max:.5f}  SDPA-{dtype} max={bl_max:.5f}  "
               f"ratio={ratio:.2f}x")
        if ratio > PASS_RATIO:
            failures.append(msg)

    if failures:
        pytest.fail("Gradient deviation exceeds bf16/fp16 noise band:\n  " + "\n  ".join(failures))


@pytest.mark.parametrize("D", [64, 128])
def test_sm120_postprocess_uses_universal_copy_for_dq_store(D):
    """White-box guard: ensure the postprocess does NOT pass an arch >= 90
    into `get_smem_store_atom` for the SM80-MMA store path on Blackwell.
    This is the actual bug source — even if the numerical impact on a given
    config is small, the wrong store atom is wrong.
    """
    import re

    src = (
        Path(__file__).resolve().parents[2]
        / "flash_attn"
        / "cute"
        / "flash_bwd_postprocess.py"
    ).read_text()
    # The fix introduces a `store_atom_arch` variable that picks 80 for
    # arch in [8, 12] before calling get_smem_store_atom. Concretely, the
    # bare `self.arch` must not reach get_smem_store_atom's `arch` parameter,
    # whether passed positionally (first arg) or by keyword (arch=self.arch).
    # Whitespace-agnostic so a reformat can't silently disarm the guard, and
    # kwarg-aware so a positional->keyword refactor can't either.
    positional = re.search(r"get_smem_store_atom\(\s*self\.arch\s*,", src)
    keyword = re.search(r"get_smem_store_atom\([^)]*\barch\s*=\s*self\.arch\b", src)
    assert positional is None and keyword is None, (
        "flash_bwd_postprocess passes self.arch (==120 on SM120) into "
        "get_smem_store_atom, which selects stmatrix despite the SM80 MMA "
        "output layout. See commit bc67a9c for the forward-side analog."
    )
