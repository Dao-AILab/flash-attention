"""Test that dropout forward/backward matches reference with the same mask.

Uses the standard FA numerical error metric:
    |flash - ref_f32| <= rtol * |ref_bf16 - ref_f32| + atol

where ref_f32 uses f32 inputs and ref_bf16 uses bf16-quantized inputs
(both computed via attention_ref in f32 arithmetic). This isolates the
error from bf16 input quantization, which is the same error source as
the flash kernel.
"""

import torch
import pytest

from flash_attn.cute.testing import attention_ref

# ---------------------------------------------------------------------------
# Python Philox 4x32 (matching quack constants and our keying layout)
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


def philox_4x32_py(c0, c1, c2, c3, k0, k1):
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


def generate_dropout_mask(batch, nheads, seqlen_q, seqlen_k, p_dropout, seed):
    seed_lo = seed & MASK32
    seed_hi = (seed >> 32) & MASK32
    p_keep_uint8 = int(255 * (1.0 - p_dropout))
    mask = torch.ones(batch, nheads, seqlen_q, seqlen_k, dtype=torch.bool)
    for b in range(batch):
        for h in range(nheads):
            rng_key_lo = (b * nheads + h) & MASK32
            for row in range(seqlen_q):
                for col in range(seqlen_k):
                    r0, _, _, _ = philox_4x32_py(
                        rng_key_lo, 0, row, col, seed_lo, seed_hi
                    )
                    mask[b, h, row, col] = (r0 & 255) <= p_keep_uint8
    return mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("p_dropout", [0.1, 0.3])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dropout_fwd_bwd(causal, p_dropout, dtype):
    """Verify flash dropout error is bounded by bf16 quantization error.

    |flash - ref_f32| <= rtol * |ref_bf16 - ref_f32| + atol

    ref_f32: attention_ref with f32 inputs + dropout mask
    ref_bf16: attention_ref with bf16-quantized inputs + dropout mask
    flash: flash_attn_func with dropout
    """
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 2, 64
    seed = 42
    rtol = 2
    torch.manual_seed(0)

    q_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)

    # bf16-quantized inputs (same precision loss as flash kernel sees)
    q_bf16 = q_ref.detach().to(dtype).float().requires_grad_(True)
    k_bf16 = k_ref.detach().to(dtype).float().requires_grad_(True)
    v_bf16 = v_ref.detach().to(dtype).float().requires_grad_(True)

    q = q_ref.detach().to(dtype).requires_grad_(True)
    k = k_ref.detach().to(dtype).requires_grad_(True)
    v = v_ref.detach().to(dtype).requires_grad_(True)

    mask = generate_dropout_mask(B, H, S, S, p_dropout, seed).to("cuda")

    # ref_f32: f32 inputs
    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None,
        causal=causal, dropout_p=p_dropout, dropout_mask=mask,
    )
    # ref_bf16: bf16-quantized inputs (still computed in f32 arithmetic)
    out_pt, _ = attention_ref(
        q_bf16, k_bf16, v_bf16, None, None,
        causal=causal, dropout_p=p_dropout, dropout_mask=mask,
    )
    # flash kernel
    out_flash, _ = flash_attn_func(
        q, k, v, causal=causal, dropout_p=p_dropout, dropout_seed=seed,
    )

    # Forward check
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    flash_err = (out_flash.float() - out_ref).abs().max().item()
    pt_err = (out_pt - out_ref).abs().max().item()
    fwd_bound = rtol * pt_err + fwd_atol

    print(f"[causal={causal} p={p_dropout}] Fwd: flash={flash_err:.6f} "
          f"pt={pt_err:.6f} bound={fwd_bound:.6f}")
    assert flash_err <= fwd_bound, (
        f"Fwd error {flash_err:.6f} > {rtol}*{pt_err:.6f}+{fwd_atol:.6f}={fwd_bound:.6f}"
    )

    # Backward check
    g = torch.randn_like(out_ref)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), g.to(dtype))
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_bf16, k_bf16, v_bf16), g)

    for name, df, dr, dp in [
        ("dq", dq_flash, dq_ref, dq_pt),
        ("dk", dk_flash, dk_ref, dk_pt),
        ("dv", dv_flash, dv_ref, dv_pt),
    ]:
        bwd_atol = 2 * (dr + 0.3 - 0.3 - dr).abs().max().item()
        grad_err = (df.float() - dr).abs().max().item()
        grad_pt = (dp - dr).abs().max().item()
        grad_bound = rtol * grad_pt + bwd_atol
        print(f"  {name}: flash={grad_err:.6f} pt={grad_pt:.6f} bound={grad_bound:.6f}")
        assert grad_err <= grad_bound, (
            f"{name} error {grad_err:.6f} > {rtol}*{grad_pt:.6f}+{bwd_atol:.6f}={grad_bound:.6f}"
        )


@pytest.mark.parametrize("seed", [42, 12345, 2**32 + 7])
def test_dropout_mask_matches_reference(seed):
    """Verify forward output matches reference using standard error metric."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 2, 64
    p_dropout = 0.2
    dtype = torch.bfloat16
    rtol = 2
    torch.manual_seed(0)

    q_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    v_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    q_bf16 = q_ref.to(dtype).float()
    k_bf16 = k_ref.to(dtype).float()
    v_bf16 = v_ref.to(dtype).float()
    q = q_ref.to(dtype)
    k = k_ref.to(dtype)
    v = v_ref.to(dtype)

    mask = generate_dropout_mask(B, H, S, S, p_dropout, seed).to("cuda")

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None,
        dropout_p=p_dropout, dropout_mask=mask,
    )
    out_pt, _ = attention_ref(
        q_bf16, k_bf16, v_bf16, None, None,
        dropout_p=p_dropout, dropout_mask=mask,
    )
    out_flash, _ = flash_attn_func(
        q, k, v, dropout_p=p_dropout, dropout_seed=seed,
    )

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    flash_err = (out_flash.float() - out_ref).abs().max().item()
    pt_err = (out_pt - out_ref).abs().max().item()
    bound = rtol * pt_err + fwd_atol

    print(f"[seed={seed}] flash={flash_err:.6f} pt={pt_err:.6f} bound={bound:.6f}")
    assert flash_err <= bound, (
        f"Error {flash_err:.6f} > {rtol}*{pt_err:.6f}+{fwd_atol:.6f}={bound:.6f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
