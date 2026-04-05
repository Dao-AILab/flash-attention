"""End-to-end dropout correctness test.

Verifies:
1. Forward with dropout=0.0 matches forward without dropout
2. Forward with dropout>0 changes output
3. Same seed produces identical outputs (deterministic)
4. Different seeds produce different outputs
5. Forward+backward with dropout produces valid gradients
6. Dropout scaling: output mean is preserved in expectation
"""

import torch
import pytest


def get_tensors(batch, seqlen_q, seqlen_k, nheads, headdim, dtype=torch.bfloat16):
    q = torch.randn(batch, seqlen_q, nheads, headdim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen_k, nheads, headdim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen_k, nheads, headdim, device="cuda", dtype=dtype, requires_grad=True)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
def test_dropout_zero_matches_no_dropout(causal):
    """dropout_p=0.0 should produce identical results to no-dropout."""
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(42)
    q, k, v = get_tensors(2, 128, 128, 4, 64)

    out_no_drop, _ = flash_attn_func(q, k, v, causal=causal)
    out_drop_0, _ = flash_attn_func(q, k, v, causal=causal, dropout_p=0.0)

    torch.testing.assert_close(out_no_drop, out_drop_0, atol=0, rtol=0)


def test_dropout_changes_output():
    """dropout_p>0 should produce different output than dropout_p=0."""
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(42)
    q, k, v = get_tensors(2, 128, 128, 4, 64)

    out_no_drop, _ = flash_attn_func(q, k, v)
    out_with_drop, _ = flash_attn_func(q, k, v, dropout_p=0.5, dropout_seed=12345)

    # Should not be identical
    assert not torch.allclose(out_no_drop, out_with_drop, atol=1e-3), \
        "Dropout output should differ from no-dropout output"


def test_dropout_deterministic():
    """Same seed should produce identical outputs."""
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(42)
    q, k, v = get_tensors(2, 128, 128, 4, 64)

    seed = 98765
    out1, _ = flash_attn_func(q, k, v, dropout_p=0.3, dropout_seed=seed)
    out2, _ = flash_attn_func(q, k, v, dropout_p=0.3, dropout_seed=seed)

    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_dropout_different_seeds():
    """Different seeds should produce different outputs."""
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(42)
    q, k, v = get_tensors(2, 128, 128, 4, 64)

    out1, _ = flash_attn_func(q, k, v, dropout_p=0.3, dropout_seed=111)
    out2, _ = flash_attn_func(q, k, v, dropout_p=0.3, dropout_seed=222)

    assert not torch.allclose(out1, out2, atol=1e-3), \
        "Different seeds should produce different outputs"


@pytest.mark.parametrize("causal", [False, True])
def test_dropout_backward(causal):
    """Forward+backward with dropout should produce valid gradients."""
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(42)
    q, k, v = get_tensors(2, 128, 128, 4, 64)

    out, _ = flash_attn_func(q, k, v, causal=causal, dropout_p=0.1, dropout_seed=42)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None, "q.grad should not be None"
    assert k.grad is not None, "k.grad should not be None"
    assert v.grad is not None, "v.grad should not be None"
    assert not torch.all(q.grad == 0), "q.grad should not be all zeros"
    assert not torch.all(k.grad == 0), "k.grad should not be all zeros"
    assert not torch.all(v.grad == 0), "v.grad should not be all zeros"


def test_dropout_scaling_preserves_mean():
    """With dropout scaling (1/(1-p)), expected output mean should be similar."""
    from flash_attn.cute.interface import flash_attn_func

    torch.manual_seed(42)
    # Use larger tensors for better statistics
    q, k, v = get_tensors(4, 256, 256, 8, 64)

    out_no_drop, _ = flash_attn_func(q, k, v)

    # Average over many seeds to check expected value
    means = []
    for seed in range(10):
        out_drop, _ = flash_attn_func(q, k, v, dropout_p=0.1, dropout_seed=seed)
        means.append(out_drop.float().mean().item())

    avg_drop = sum(means) / len(means)
    no_drop_mean = out_no_drop.float().mean().item()

    # With proper scaling, means should be within ~10% of each other
    rel_diff = abs(avg_drop - no_drop_mean) / (abs(no_drop_mean) + 1e-6)
    assert rel_diff < 0.15, f"Mean mismatch: no_drop={no_drop_mean:.4f}, avg_drop={avg_drop:.4f}, rel_diff={rel_diff:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
