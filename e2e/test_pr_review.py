"""E2E tests targeting the 4 PR review comments on SM70/V100."""
import torch
import pytest


def get_cc():
    return torch.cuda.get_device_capability()


def is_volta():
    major, _ = get_cc()
    return major == 7


print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute capability: {get_cc()}")
print()


# ============================================================
# 1. splitkv kernel path (force via flash_attn_with_kvcache)
# ============================================================
def test_splitkv_kvcache():
    """PR comment #1: splitkv kernel should work on SM70."""
    from flash_attn import flash_attn_with_kvcache

    B, S_q, S_kv, H, D = 2, 1, 64, 4, 64
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B, S_q, H, D, device=device, dtype=dtype)
    k_cache = torch.randn(B, S_kv, H, D, device=device, dtype=dtype)
    v_cache = torch.randn(B, S_kv, H, D, device=device, dtype=dtype)

    out = flash_attn_with_kvcache(q, k_cache, v_cache)

    # Compare with manual attention
    scale = D ** -0.5
    scores = torch.einsum("bshd,bthd->bsht", q.float(), k_cache.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    ref = torch.einsum("bsht,bthd->bshd", attn, v_cache.float())

    err = (out.float() - ref).abs().max().item()
    print(f"  splitkv (kvcache): max_err={err:.6f}  {'OK' if err < 0.002 else 'FAIL'}")
    assert err < 0.002, f"splitkv error too large: {err}"


# ============================================================
# 2. varlen forward dropout guard
# ============================================================
def test_varlen_fwd_dropout_blocked():
    """PR comment #2: varlen forward with dropout should be blocked on Volta."""
    if not is_volta():
        print("  varlen dropout guard: SKIP (not Volta)")
        return

    from flash_attn import flash_attn_varlen_func

    B, S, H, D = 2, 64, 4, 64
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B * S, H, D, device=device, dtype=dtype)
    k = torch.randn(B * S, H, D, device=device, dtype=dtype)
    v = torch.randn(B * S, H, D, device=device, dtype=dtype)
    cu = torch.arange(0, (B + 1) * S, S, device=device, dtype=torch.int32)

    try:
        flash_attn_varlen_func(q, k, v, cu, cu, S, S, dropout_p=0.1)
        print("  varlen dropout guard: FAIL (no error raised)")
        assert False, "Should have raised error"
    except RuntimeError as e:
        if "dropout" in str(e).lower():
            print(f"  varlen dropout guard: OK (correctly blocked: {e})")
        else:
            print(f"  varlen dropout guard: FAIL (wrong error: {e})")
            raise


# ============================================================
# 3. varlen forward causal hdim>192 guard
# ============================================================
def test_varlen_fwd_causal_hdim256_blocked():
    """PR comment #3: varlen forward causal + hdim>192 should be blocked on Volta."""
    if not is_volta():
        print("  varlen causal hdim>192 guard: SKIP (not Volta)")
        return

    from flash_attn import flash_attn_varlen_func

    B, S, H, D = 1, 32, 2, 256
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B * S, H, D, device=device, dtype=dtype)
    k = torch.randn(B * S, H, D, device=device, dtype=dtype)
    v = torch.randn(B * S, H, D, device=device, dtype=dtype)
    cu = torch.arange(0, (B + 1) * S, S, device=device, dtype=torch.int32)

    try:
        flash_attn_varlen_func(q, k, v, cu, cu, S, S, causal=True)
        print("  varlen causal hdim>192 guard: FAIL (no error raised)")
        assert False, "Should have raised error"
    except RuntimeError as e:
        if "head_dim" in str(e).lower() or "192" in str(e):
            print(f"  varlen causal hdim>192 guard: OK (correctly blocked: {e})")
        else:
            print(f"  varlen causal hdim>192 guard: FAIL (wrong error: {e})")
            raise


# ============================================================
# 4. ALiBi backward blocked on Volta
# ============================================================
def test_alibi_bwd_blocked():
    """PR comment #4: ALiBi backward should be blocked on Volta."""
    if not is_volta():
        print("  ALiBi bwd guard: SKIP (not Volta)")
        return

    from flash_attn import flash_attn_func

    B, S, H, D = 2, 64, 4, 64
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
    alibi_slopes = torch.randn(H, device=device, dtype=torch.float32)

    # Forward with ALiBi should work
    out = flash_attn_func(q, k, v, alibi_slopes=alibi_slopes)
    print(f"  ALiBi forward: OK (output shape={out.shape})")

    # Backward with ALiBi should be blocked
    try:
        out.sum().backward()
        print("  ALiBi bwd guard: FAIL (no error raised)")
        assert False, "Should have raised error"
    except RuntimeError as e:
        if "alibi" in str(e).lower():
            print(f"  ALiBi bwd guard: OK (correctly blocked: {e})")
        else:
            print(f"  ALiBi bwd guard: FAIL (wrong error: {e})")
            raise


# ============================================================
# 5. varlen forward+backward (positive test - should work)
# ============================================================
def test_varlen_fwd_bwd():
    """Positive test: varlen forward+backward should work on SM70."""
    from flash_attn import flash_attn_varlen_func

    B, S, H, D = 2, 64, 4, 64
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B * S, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B * S, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B * S, H, D, device=device, dtype=dtype, requires_grad=True)
    cu = torch.arange(0, (B + 1) * S, S, device=device, dtype=torch.int32)

    out = flash_attn_varlen_func(q, k, v, cu, cu, S, S)
    loss = out.sum()
    loss.backward()

    has_grads = q.grad is not None and k.grad is not None and v.grad is not None
    print(f"  varlen fwd+bwd: {'OK' if has_grads else 'FAIL'} (grads computed={has_grads})")
    assert has_grads


# ============================================================
# 6. KV-cache forward causal hdim>192 guard
# ============================================================
def test_kvcache_fwd_causal_hdim256_blocked():
    """PR comment #6: KV-cache forward causal + hdim>192 should be blocked on Volta."""
    if not is_volta():
        print("  kvcache causal hdim>192 guard: SKIP (not Volta)")
        return

    from flash_attn import flash_attn_with_kvcache

    B, S_q, S_kv, H, D = 1, 4, 32, 2, 256
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(B, S_q, H, D, device=device, dtype=dtype)
    k_cache = torch.randn(B, S_kv, H, D, device=device, dtype=dtype)
    v_cache = torch.randn(B, S_kv, H, D, device=device, dtype=dtype)

    try:
        flash_attn_with_kvcache(q, k_cache, v_cache, causal=True)
        print("  kvcache causal hdim>192 guard: FAIL (no error raised)")
        assert False, "Should have raised error"
    except RuntimeError as e:
        if "head_dim" in str(e).lower() or "192" in str(e):
            print(f"  kvcache causal hdim>192 guard: OK (correctly blocked: {e})")
        else:
            print(f"  kvcache causal hdim>192 guard: FAIL (wrong error: {e})")
            raise


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("=== PR Review Comment Tests ===")
    print()

    print("[1] splitkv kernel (KV-cache inference)")
    test_splitkv_kvcache()
    print()

    print("[2] varlen forward dropout guard")
    test_varlen_fwd_dropout_blocked()
    print()

    print("[3] varlen forward causal hdim>192 guard")
    test_varlen_fwd_causal_hdim256_blocked()
    print()

    print("[4] ALiBi backward guard")
    test_alibi_bwd_blocked()
    print()

    print("[5] varlen forward+backward (positive test)")
    test_varlen_fwd_bwd()
    print()

    print("[6] KV-cache forward causal hdim>192 guard")
    test_kvcache_fwd_causal_hdim256_blocked()
    print()

    print("=== All PR review tests passed ===")
