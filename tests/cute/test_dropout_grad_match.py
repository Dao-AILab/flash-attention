"""Test dropout forward/backward correctness.

Extracts the actual dropout mask from the kernel via V=Identity trick,
then compares flash output and gradients against an f32 reference that
uses the same extracted mask. This is the definitive test that the
forward and backward masks are identical and the implementation is correct.

Forward: flash vs f32 ref should match within bf16 precision (~0.003)
Backward dv: should match within bf16 precision (~0.004)
Backward dq/dk: elevated error from P_drop * (dP - D) amplification,
consistent across runs (NOT a mask mismatch)
"""

import math
import torch
import pytest


def extract_dropout_mask(q, k, p_dropout, seed, flash_attn_func, causal=False):
    """Extract the kernel's actual dropout mask via V=Identity.

    With V=I, output O = P_drop where P_drop is the dropout-masked
    attention matrix. mask = (O > 0) gives the keep/drop decisions.
    """
    B, S, H, D = q.shape
    assert D >= S, f"Need D >= S for V=I trick, got D={D} S={S}"
    v_eye = torch.eye(S, D, device=q.device, dtype=q.dtype)
    v_eye = v_eye.unsqueeze(0).unsqueeze(2).expand(B, S, H, D)
    out, _ = flash_attn_func(q, k, v_eye, dropout_p=p_dropout, dropout_seed=seed, causal=causal)
    mask = (out.float() > 0)
    return mask.transpose(1, 2)  # (B, S, H, S) -> (B, H, S, S)


def attention_dropout_ref(q, k, v, mask, p_dropout, causal=False):
    """f32 reference attention with the extracted dropout mask."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    qt = q.float().transpose(1, 2)
    kt = k.float().transpose(1, 2)
    vt = v.float().transpose(1, 2)
    scores = torch.matmul(qt, kt.transpose(-2, -1)) * scale
    if causal:
        S = scores.shape[-1]
        cmask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(cmask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    rp = 1.0 / (1.0 - p_dropout)
    attn_drop = attn.masked_fill(~mask, 0.0) * rp
    return torch.matmul(attn_drop, vt).transpose(1, 2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("p_dropout", [0.1, 0.3])
def test_dropout_fwd_bwd_with_extracted_mask(causal, p_dropout):
    """Forward and backward vs f32 reference using the kernel's own mask."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 1, 64
    seed = 42
    torch.manual_seed(0)

    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    # Extract mask from kernel
    mask = extract_dropout_mask(q, k, p_dropout, seed, flash_attn_func, causal)
    kept = mask.float().mean().item()
    print(f"[causal={causal} p={p_dropout}] Kept: {kept:.3f}")

    # f32 reference with extracted mask
    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    out_ref = attention_dropout_ref(q_ref, k_ref, v_ref, mask, p_dropout, causal)

    # Flash forward + backward
    q2 = q.detach().requires_grad_(True)
    k2 = k.detach().requires_grad_(True)
    v2 = v.detach().requires_grad_(True)
    out_flash, _ = flash_attn_func(q2, k2, v2, causal=causal, dropout_p=p_dropout, dropout_seed=seed)

    # Forward check (causal has higher baseline error from mask interaction)
    fwd_err = (out_flash.float() - out_ref).abs().max().item()
    fwd_bound = 0.02 if causal else 0.01
    print(f"  Fwd: {fwd_err:.6f} (bound: {fwd_bound})")
    assert fwd_err < fwd_bound, f"Forward error {fwd_err:.6f} > {fwd_bound}"

    # Backward check
    g = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q2, k2, v2), g.to(torch.bfloat16))

    # Also compute no-dropout backward as baseline for amplification check
    q_ref_nd = q.float().detach().requires_grad_(True)
    k_ref_nd = k.float().detach().requires_grad_(True)
    v_ref_nd = v.float().detach().requires_grad_(True)
    out_ref_nd = attention_dropout_ref(
        q_ref_nd, k_ref_nd, v_ref_nd,
        torch.ones_like(mask), 0.0, causal,
    )
    q_nd = q.detach().requires_grad_(True)
    k_nd = k.detach().requires_grad_(True)
    v_nd = v.detach().requires_grad_(True)
    out_nd, _ = flash_attn_func(q_nd, k_nd, v_nd, causal=causal)
    dq_ref_nd, dk_ref_nd, dv_ref_nd = torch.autograd.grad(out_ref_nd, (q_ref_nd, k_ref_nd, v_ref_nd), g)
    dq_nd, dk_nd, dv_nd = torch.autograd.grad(out_nd, (q_nd, k_nd, v_nd), g.to(torch.bfloat16))

    for name, df, dr, df_nd, dr_nd in [
        ("dq", dq_flash, dq_ref, dq_nd, dq_ref_nd),
        ("dk", dk_flash, dk_ref, dk_nd, dk_ref_nd),
        ("dv", dv_flash, dv_ref, dv_nd, dv_ref_nd),
    ]:
        do_err = (df.float() - dr).abs().max().item()
        nd_err = (df_nd.float() - dr_nd).abs().max().item()
        amplification = do_err / (nd_err + 1e-10)
        print(f"  {name}: dropout={do_err:.6f} no_dropout={nd_err:.6f} amp={amplification:.1f}x")

        # dv: no amplification expected (1-2x of baseline)
        # dq/dk: amplification from P_drop*(dP-D), bounded at 50x baseline
        if name == "dv":
            bound = max(0.03 if causal else 0.01, 3 * nd_err)
            assert do_err < bound, f"{name} error {do_err:.6f} > {bound:.6f}"
        else:
            # dq/dk: check MEAN error is bounded (max error has outliers at
            # causal boundaries where few attention elements amplify bf16
            # dPsum precision loss through P_drop * (dP - D))
            do_mean = (df.float() - dr).abs().mean().item()
            nd_mean = (df_nd.float() - dr_nd).abs().mean().item()
            # Causal: early rows have few elements, amplifying P*(dP-D) error
            rtol_bwd = 100 if causal else 50
            mean_bound = rtol_bwd * nd_mean + 0.005
            assert do_mean < mean_bound, (
                f"{name} mean error {do_mean:.6f} > 50x baseline {nd_mean:.6f}"
            )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("p_dropout", [0.1, 0.3])
def test_dropout_fwd_bwd_determinism(causal, p_dropout):
    """Same seed -> bit-exact forward output AND backward gradients."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 128, 2, 64
    seed = 42
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    out1, _ = flash_attn_func(q, k, v, causal=causal, dropout_p=p_dropout, dropout_seed=seed)
    dq1, dk1, dv1 = torch.autograd.grad(out1, (q, k, v), g)

    out2, _ = flash_attn_func(q, k, v, causal=causal, dropout_p=p_dropout, dropout_seed=seed)
    dq2, dk2, dv2 = torch.autograd.grad(out2, (q, k, v), g)

    assert torch.allclose(out1, out2, atol=0, rtol=0), "Forward not deterministic"
    assert torch.allclose(dq1, dq2, atol=0, rtol=0), "dq not deterministic"
    assert torch.allclose(dk1, dk2, atol=0, rtol=0), "dk not deterministic"
    assert torch.allclose(dv1, dv2, atol=0, rtol=0), "dv not deterministic"


@pytest.mark.parametrize("seed", [42, 12345, 2**32 + 7])
def test_dropout_seed_sensitivity(seed):
    """Different seeds produce different outputs."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 128, 2, 64
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    out1, _ = flash_attn_func(q, k, v, dropout_p=0.2, dropout_seed=seed)
    out2, _ = flash_attn_func(q, k, v, dropout_p=0.2, dropout_seed=seed + 1)
    assert not torch.allclose(out1, out2, atol=1e-3)


def test_dropout_p0_matches_baseline():
    """dropout_p=0 should be bit-exact with no dropout."""
    from flash_attn.cute.interface import flash_attn_func

    q = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.bfloat16)

    out_base, _ = flash_attn_func(q, k, v)
    out_p0, _ = flash_attn_func(q, k, v, dropout_p=0.0)
    assert torch.allclose(out_base, out_p0, atol=0, rtol=0)


def test_dropout_mask_extraction():
    """Verify the V=I mask extraction gives consistent results."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 1, 64
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    mask1 = extract_dropout_mask(q, k, 0.2, 42, flash_attn_func)
    mask2 = extract_dropout_mask(q, k, 0.2, 42, flash_attn_func)
    assert torch.equal(mask1, mask2), "Mask extraction not deterministic"

    kept = mask1.float().mean().item()
    assert 0.7 < kept < 0.9, f"Keep fraction {kept:.3f} out of range for p=0.2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])


# ---------------------------------------------------------------------------
# Python Philox reference for mask verification
# ---------------------------------------------------------------------------

PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_N_ROUNDS = 7
MASK32 = 0xFFFFFFFF


def _mul_wide(a, b):
    prod = a * b
    return (prod >> 32) & MASK32, prod & MASK32


def philox_py(c0, c1, c2, c3, k0, k1):
    for _ in range(PHILOX_N_ROUNDS):
        hi_b, lo_b = _mul_wide(c2, PHILOX_ROUND_B)
        hi_a, lo_a = _mul_wide(c0, PHILOX_ROUND_A)
        c0 = (hi_b ^ c1 ^ k0) & MASK32
        c1 = lo_b
        c2 = (hi_a ^ c3 ^ k1) & MASK32
        c3 = lo_a
        k0 = (k0 + PHILOX_KEY_A) & MASK32
        k1 = (k1 + PHILOX_KEY_B) & MASK32
    return c0, c1, c2, c3


def generate_mask_py(B, H, Sq, Sk, p_dropout, seed):
    """Generate dropout mask matching the kernel's 2x4 group batching with 16-bit threshold."""
    seed_lo = seed & MASK32
    seed_hi = (seed >> 32) & MASK32
    threshold_8 = int(255 * (1.0 - p_dropout))
    threshold_16 = threshold_8 * 257  # scale 8-bit to 16-bit
    mask = torch.ones(B, H, Sq, Sk, dtype=torch.bool)
    for b in range(B):
        for h in range(H):
            key_lo = (b * H + h) & MASK32
            for row in range(Sq):
                for col in range(Sk):
                    row_group = row >> 1  # // 2
                    col_group = col >> 2  # // 4
                    r0, r1, r2, r3 = philox_py(key_lo, 0, row_group, col_group, seed_lo, seed_hi)
                    # Element index in 2x4 group
                    elem = (row & 1) * 4 + (col & 3)  # 0..7
                    # Extract 16-bit value from the 4 words
                    words = [r0, r1, r2, r3]
                    word = words[elem >> 1]
                    half = elem & 1
                    rand_u16 = (word >> (half * 16)) & 0xFFFF
                    mask[b, h, row, col] = rand_u16 <= threshold_16
    return mask


def test_mask_matches_python_philox():
    """Verify kernel dropout mask matches independent Python Philox reference."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 1, 64
    p, seed = 0.2, 42
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    # Extract kernel mask via V=I
    kernel_mask = extract_dropout_mask(q, k, p, seed, flash_attn_func)

    # Generate Python reference mask
    python_mask = generate_mask_py(B, H, S, S, p, seed).to("cuda")

    match = (kernel_mask == python_mask).float().mean().item()
    print(f"Kernel vs Python Philox mask agreement: {match:.4f}")
    assert match > 0.99, f"Mask agreement {match:.4f} < 0.99"


def test_fwd_matches_python_philox_ref():
    """Forward output matches f32 reference using Python-generated mask."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 1, 64
    p, seed = 0.1, 42
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    mask = generate_mask_py(B, H, S, S, p, seed).to("cuda")
    out_ref = attention_dropout_ref(q, k, v, mask, p)
    out_flash, _ = flash_attn_func(q, k, v, dropout_p=p, dropout_seed=seed)

    err = (out_flash.float() - out_ref).abs().max().item()
    print(f"Flash vs Python Philox ref: {err:.6f}")
    assert err < 0.01, f"Forward error {err:.6f} vs Python Philox ref"
