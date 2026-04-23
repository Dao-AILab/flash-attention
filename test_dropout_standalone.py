"""Standalone dropout test - bypasses FA2 CUDA import."""

import torch

# Import directly from the cute subpackage, which is pip-installed as flash-attn-4
from flash_attn.cute.interface import flash_attn_func


def test_fwd_no_dropout():
    """Basic forward without dropout (baseline)."""
    q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    print("Forward (no dropout)...")
    out, lse = flash_attn_func(q, k, v)
    print(f"  Output shape: {out.shape}, mean: {out.float().mean():.4f}")
    return out


def test_fwd_dropout_zero():
    """Forward with dropout_p=0.0 should match no-dropout."""
    q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    out1, _ = flash_attn_func(q, k, v)
    out2, _ = flash_attn_func(q, k, v, dropout_p=0.0)
    match = torch.allclose(out1, out2, atol=0, rtol=0)
    print(f"dropout_p=0.0 matches no-dropout: {match}")
    assert match, "dropout_p=0.0 should produce identical results"


def test_fwd_dropout():
    """Forward with dropout_p>0 changes output."""
    q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    out_no, _ = flash_attn_func(q, k, v)
    out_do, _ = flash_attn_func(q, k, v, dropout_p=0.5, dropout_seed=12345)
    diff = (out_no.float() - out_do.float()).abs().max().item()
    print(f"dropout_p=0.5 max diff from no-dropout: {diff:.4f}")
    assert diff > 0.01, "Dropout should change output"


def test_fwd_deterministic():
    """Same seed = identical output."""
    q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    out1, _ = flash_attn_func(q, k, v, dropout_p=0.3, dropout_seed=98765)
    out2, _ = flash_attn_func(q, k, v, dropout_p=0.3, dropout_seed=98765)
    match = torch.allclose(out1, out2, atol=0, rtol=0)
    print(f"Same seed deterministic: {match}")
    assert match, "Same seed should produce identical output"


def test_bwd_dropout():
    """Forward+backward with dropout produces valid gradients."""
    q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = flash_attn_func(q, k, v, dropout_p=0.1, dropout_seed=42)
    loss = out.sum()
    loss.backward()
    q_ok = q.grad is not None and not torch.all(q.grad == 0)
    k_ok = k.grad is not None and not torch.all(k.grad == 0)
    v_ok = v.grad is not None and not torch.all(v.grad == 0)
    print(f"Backward grads: q={q_ok}, k={k_ok}, v={v_ok}")
    assert q_ok and k_ok and v_ok, "All gradients should be non-zero"


if __name__ == "__main__":
    tests = [
        test_fwd_no_dropout,
        test_fwd_dropout_zero,
        test_fwd_dropout,
        test_fwd_deterministic,
        test_bwd_dropout,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}\n")
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}\n")
            import traceback
            traceback.print_exc()
            break
    print("Done!")
