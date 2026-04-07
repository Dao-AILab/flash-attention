"""Test dropout forward/backward correctness.

Extracts the actual dropout mask from the kernel via V=Identity trick,
then compares flash output and gradients against an f32 reference that
uses the same extracted mask. This is the definitive test that the
forward and backward masks are identical and the implementation is correct.

Forward: flash vs f32 ref should match within bf16 precision (~0.003)
Backward dq/dk/dv: all within ~1-3x of no-dropout baseline error.
The backward uses the correct gradient formula dS = P_drop * dP - P * D
(not the approximation P_drop * (dP - D) used in FA2 C++).
"""

import math
import torch
import pytest

from flash_attn.cute.testing import attention_ref


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
    """Forward and backward vs f32 reference using the repo's standard metric.

    Uses the same error metric as test_flash_attn.py:
      flash_err <= 2 * pt_err + atol
    where pt_err is the bf16 PyTorch baseline error vs f32 reference.
    """
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 1, 64
    seed = 42
    torch.manual_seed(0)

    q_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16).requires_grad_()
    k_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16).requires_grad_()
    v_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16).requires_grad_()

    # Extract mask from kernel
    mask = extract_dropout_mask(
        q_ref.detach(), k_ref.detach(), p_dropout, seed, flash_attn_func, causal
    )
    kept = mask.float().mean().item()
    print(f"[causal={causal} p={p_dropout}] Kept: {kept:.3f}")

    # f32 reference (gold standard) — uses attention_ref from the repo
    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, dropout_p=p_dropout, dropout_mask=mask,
        causal=causal, upcast=True,
    )
    # bf16 baseline — same metric as test_flash_attn.py
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, dropout_p=p_dropout, dropout_mask=mask,
        causal=causal, upcast=False, reorder_ops=True,
    )

    # Flash forward + backward
    q = q_ref.detach().requires_grad_(True)
    k = k_ref.detach().requires_grad_(True)
    v = v_ref.detach().requires_grad_(True)
    out_flash, _ = flash_attn_func(q, k, v, causal=causal, dropout_p=p_dropout, dropout_seed=seed)

    # Forward: repo standard metric
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2
    flash_err = (out_flash.float() - out_ref).abs().max().item()
    pt_err = (out_pt - out_ref).abs().max().item()
    print(f"  Fwd: flash={flash_err:.6f} pt={pt_err:.6f} ratio={flash_err / max(pt_err, 1e-10):.2f}x")
    assert flash_err <= rtol * pt_err + fwd_atol, (
        f"Forward error {flash_err:.6f} > {rtol} * {pt_err:.6f} + {fwd_atol:.6f}"
    )

    # Backward: repo standard metric for dq, dk, dv
    g = torch.randn_like(out_flash)
    dq, dk, dv = torch.autograd.grad(out_flash, (q, k, v), g)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)

    for name, d, d_ref, d_pt in [("dq", dq, dq_ref, dq_pt), ("dk", dk, dk_ref, dk_pt), ("dv", dv, dv_ref, dv_pt)]:
        d_atol = 2 * (d_ref + 0.3 - 0.3 - d_ref).abs().max().item()
        d_err = (d.float() - d_ref).abs().max().item()
        d_pt_err = (d_pt - d_ref).abs().max().item()
        d_ratio = d_err / max(d_pt_err, 1e-10)
        print(f"  {name}: flash={d_err:.6f} pt={d_pt_err:.6f} ratio={d_ratio:.2f}x")
        assert d_err <= rtol * d_pt_err + d_atol, (
            f"{name} error {d_err:.6f} > {rtol} * {d_pt_err:.6f} + {d_atol:.6f}"
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


def test_dropout_varlen():
    """Dropout works with variable-length sequences."""
    from flash_attn.cute.interface import flash_attn_varlen_func

    torch.manual_seed(0)
    # Two sequences of different lengths
    seqlens = [48, 64]
    total = sum(seqlens)
    H, D = 2, 64
    cu_seqlens = torch.tensor([0, seqlens[0], sum(seqlens)], dtype=torch.int32, device="cuda")
    q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    seed = 42
    p = 0.2
    out1, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max(seqlens), max(seqlens),
        dropout_p=p, dropout_seed=seed,
    )
    dq1, dk1, dv1 = torch.autograd.grad(out1, (q, k, v), torch.randn_like(out1))

    # Determinism
    out2, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max(seqlens), max(seqlens),
        dropout_p=p, dropout_seed=seed,
    )
    assert torch.equal(out1, out2), "Varlen dropout not deterministic"

    # p=0 baseline
    out_base, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max(seqlens), max(seqlens),
    )
    assert not torch.equal(out1, out_base), "Dropout had no effect on varlen"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])


# ---------------------------------------------------------------------------
# Python Philox reference with MMA-layout keying (m16n8k16)
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


def generate_mask_mma_layout(B, H, Sq, Sk, p_dropout, seed,
                             tile_m=64, tile_n=64, num_warps=4):
    """Generate dropout mask matching the kernel's MMA-layout Philox keying.

    Reproduces the exact m16n8k16 accumulator thread-to-element mapping used by
    apply_dropout_mask with logical_divide.

    CuTe m16n8k16 accumulator layout per thread (lane_id):
      t1 = lane_id // 4 (row base, 0..7)
      t0 = lane_id % 4  (col pair, 0..3)
      V-mode shape (2,2) with strides (2,1) gives flat reg → (v0, v1):
        reg 0 → (v0=0, v1=0): row = t1,     col = t0 * 2
        reg 1 → (v0=0, v1=1): row = t1,     col = t0 * 2 + 1
        reg 2 → (v0=1, v1=0): row = t1 + 8, col = t0 * 2
        reg 3 → (v0=1, v1=1): row = t1 + 8, col = t0 * 2 + 1

    Kernel keying:
      philox_offset = (batch * nheads + head) * 32 + lane_id
      block_row = m_block * (tile_m // 16) + warp_id + m * num_warps
      block_col = n_block * (tile_n // 32) + n_half
      u16_idx = j * 4 + reg  (j from logical_divide pair-half, reg 0..3)
    """
    seed_lo = seed & MASK32
    seed_hi = (seed >> 32) & MASK32
    threshold_8 = int(255 * (1.0 - p_dropout))
    threshold_16 = threshold_8 * 257

    mask = torch.ones(B, H, Sq, Sk, dtype=torch.bool)
    n_m_blocks = (Sq + tile_m - 1) // tile_m
    n_n_blocks = (Sk + tile_n - 1) // tile_n

    # CuTe m16n8k16 layout: row = t1 + (reg%2)*8, col = t0*2 + (reg//2)
    # where t0 = lane_id % 4, t1 = lane_id // 4

    for b in range(B):
        for h in range(H):
            for m_block in range(n_m_blocks):
                for n_block in range(n_n_blocks):
                    for warp_id in range(num_warps):
                        for lane_id in range(32):
                            t0 = lane_id % 4
                            t1 = lane_id // 4

                            philox_offset = ((b * H + h) * 32 + lane_id) & MASK32

                            # MMA_M = tile_m / (16 * num_warps), MMA_N = tile_n / 8
                            mma_m = tile_m // (16 * num_warps)
                            mma_n = tile_n // 8
                            # After logical_divide by 2: (2, mma_n // 2)
                            mma_n_half = mma_n // 2

                            for m in range(mma_m):
                                block_row = (m_block * (tile_m // 16)
                                             + warp_id + m * num_warps)
                                for n_half in range(mma_n_half):
                                    block_col = (n_block * (tile_n // 32)
                                                 + n_half)

                                    r0, r1, r2, r3 = philox_py(
                                        philox_offset, 0,
                                        block_row, block_col,
                                        seed_lo, seed_hi,
                                    )
                                    words = [r0, r1, r2, r3]

                                    for j in range(2):
                                        for reg in range(4):
                                            u16_idx = j * 4 + reg
                                            word = words[u16_idx >> 1]
                                            half = u16_idx & 1
                                            rand_u16 = (word >> (half * 16)) & 0xFFFF

                                            # CuTe m16n8k16 mapping (V-mode strides (2,1))
                                            # reg → (v0, v1): v0 = reg // 2, v1 = reg % 2
                                            n_idx = j + 2 * n_half
                                            row_in_mma = t1 + (reg // 2) * 8
                                            col_in_mma = t0 * 2 + (reg % 2)
                                            global_row = (m_block * tile_m
                                                          + warp_id * 16
                                                          + m * num_warps * 16
                                                          + row_in_mma)
                                            global_col = (n_block * tile_n
                                                          + n_idx * 8
                                                          + col_in_mma)

                                            if global_row < Sq and global_col < Sk:
                                                mask[b, h, global_row, global_col] = (
                                                    rand_u16 <= threshold_16
                                                )
    return mask


def test_mask_matches_python_philox():
    """Verify kernel dropout mask matches independent MMA-layout Python Philox."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 1, 64
    p, seed = 0.2, 42
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    kernel_mask = extract_dropout_mask(q, k, p, seed, flash_attn_func)
    python_mask = generate_mask_mma_layout(B, H, S, S, p, seed).to("cuda")

    match = (kernel_mask == python_mask).float().mean().item()
    print(f"Kernel vs MMA-layout Python Philox: {match:.4f}")
    assert match > 0.99, f"Mask agreement {match:.4f} < 0.99"


def test_fwd_matches_python_philox_ref():
    """Forward output matches f32 reference using MMA-layout Python mask."""
    from flash_attn.cute.interface import flash_attn_func

    B, S, H, D = 1, 64, 1, 64
    p, seed = 0.1, 42
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    mask = generate_mask_mma_layout(B, H, S, S, p, seed).to("cuda")
    out_ref = attention_dropout_ref(q, k, v, mask, p)
    out_flash, _ = flash_attn_func(q, k, v, dropout_p=p, dropout_seed=seed)

    err = (out_flash.float() - out_ref).abs().max().item()
    print(f"Flash vs MMA-layout Python Philox ref: {err:.6f}")
    assert err < 0.01, f"Forward error {err:.6f} vs MMA-layout Python Philox ref"
