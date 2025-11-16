# Copyright (c) 2025, Tri Dao.
# Tests for SM120 (RTX 50) Flash Attention implementation.
# SM120 uses tcgen05 SuperMMA instructions but lacks Tensor Memory,
# so all intermediate tensors use shared memory instead.

import pytest
import torch

from flash_attn.cute.testing import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)
from flash_attn.cute.interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

# Check if we're running on SM120 (compute capability 12.x)
is_sm120 = torch.cuda.get_device_capability("cuda")[0] == 12

# Skip all tests if not on SM120
pytestmark = pytest.mark.skipif(
    not is_sm120, reason="SM120 tests require compute capability 12.x (RTX 50)"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 96, 128])  # SM120 supports these head dims
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 64),
        (64, 128),
        (128, 128),
        (128, 256),
        (256, 256),
        (256, 512),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        # Test edge cases
        (113, 203),
        (128, 217),
        (108, 256),
    ],
)
def test_flash_attn_sm120_output(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    local,
    softcap,
    has_learnable_sink,
    mha_type,
    dtype,
):
    """Test Flash Attention forward pass on SM120."""
    if (causal or local) and seqlen_k < seqlen_q:
        pytest.skip("Causal attention requires seqlen_k >= seqlen_q")
    
    device = "cuda"
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    batch_size = 9 if seqlen_k <= 2048 else 2
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    dtype_ref = dtype
    
    dv_vals = [128] if d == 192 else ([d] if d != 128 else [64, d])
    
    for dv in dv_vals:
        q_ref = torch.randn(
            batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref
        )
        if softcap > 0.0:
            q_ref = q_ref * softcap / 4
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        
        k_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        
        v_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        
        window_size = (
            (None, None) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        )
        
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None
        
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        
        out_ref, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap,
        )
        
        out_pt, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )
        
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3
        
        out, _ = flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            learnable_sink=learnable_sink,
            num_splits=1,
        )
        
        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        
        # Check that FlashAttention's numerical error is at most rtol times the numerical error
        # of a Pytorch implementation.
        assert (out - out_ref).abs().max().item() <= rtol * (
            out_pt - out_ref
        ).abs().max().item() + fwd_atol
        
        # Test backward pass
        if (
            not dv > 256
            and softcap == 0.0
            and not local
            and dv == d
            and learnable_sink is None
        ):
            g = torch.randn_like(out)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(
                out_ref, (q_ref, k_ref, v_ref), g
            )
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            
            print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
            print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
            print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
            print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
            
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
            assert (dq - dq_ref).abs().max().item() <= rtol * (
                dq_pt - dq_ref
            ).abs().max().item() + dq_atol
            
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item()
            assert (dk - dk_ref).abs().max().item() <= rtol * (
                dk_pt - dk_ref
            ).abs().max().item() + dk_atol
            
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item()
            assert (dv - dv_ref).abs().max().item() <= rtol * (
                dv_pt - dv_ref
            ).abs().max().item() + dv_atol


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("has_learnable_sink", [False, True])
@pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 128),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (113, 203),
        (128, 217),
        (108, 256),
    ],
)
def test_flash_attn_sm120_varlen_output(
    seqlen_q,
    seqlen_k,
    d,
    causal,
    local,
    softcap,
    has_learnable_sink,
    mha_type,
    dtype,
):
    """Test Flash Attention varlen forward pass on SM120."""
    if (causal or local):
        seqlen_k = seqlen_q
    
    device = "cuda"
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    
    batch_size = 49 if seqlen_q <= 1024 else 7
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    dtype_ref = dtype
    
    dv_vals = [128] if d == 192 else ([d] if d != 128 else [64, d])
    
    for dv in dv_vals:
        q_ref = torch.randn(
            batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref
        )
        if softcap > 0.0:
            q_ref = (q_ref * softcap / 4).detach().requires_grad_()
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        
        k_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        
        v_ref = (
            torch.randn(
                batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref
            )
            .to(dtype)
            .to(dtype_ref)
            .requires_grad_()
        )
        
        window_size = (
            (None, None) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        )
        
        if has_learnable_sink:
            learnable_sink = torch.randn(nheads, dtype=torch.bfloat16, device=device)
        else:
            learnable_sink = None
        
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, device, mode="random", zero_lengths=False
        )
        
        if causal or local:
            key_padding_mask = query_padding_mask
        
        (
            q_unpad,
            k_unpad,
            v_unpad,
            qv_unpad,  # Will be None since qv=None
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            qv,  # Will be None since qv=None
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(
            q,
            k,
            v,
            query_padding_mask,
            key_padding_mask,
            qv=None,
            kvpacked=False,
        )
        
        q_unpad, k_unpad, v_unpad = [
            x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)
        ]
        
        out_ref, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap,
        )
        
        out_pt, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
        )
        
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3
        
        out, _ = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            learnable_sink=learnable_sink,
            num_splits=1,
        )
        
        out = output_pad_fn(out)
        
        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        
        assert (out - out_ref).abs().max().item() <= rtol * (
            out_pt - out_ref
        ).abs().max().item() + fwd_atol
        
        # Test backward pass
        if (
            not dv > 256
            and softcap == 0.0
            and not local
            and dv == d
            and learnable_sink is None
        ):
            g = torch.randn_like(out)
            dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
                out, (q_unpad, k_unpad, v_unpad), g
            )
            
            dq = dq_pad_fn(dq_unpad)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(
                out_ref, (q_ref, k_ref, v_ref), g
            )
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            
            print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
            
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
            assert (dq - dq_ref).abs().max().item() <= rtol * (
                dq_pt - dq_ref
            ).abs().max().item() + dq_atol
            
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item()
            assert (dk - dk_ref).abs().max().item() <= rtol * (
                dk_pt - dk_ref
            ).abs().max().item() + dk_atol
            
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item()
            assert (dv - dv_ref).abs().max().item() <= rtol * (
                dv_pt - dv_ref
            ).abs().max().item() + dv_atol


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_sm120_basic(seqlen, d, causal, dtype):
    """Basic smoke test for SM120 Flash Attention."""
    device = "cuda"
    torch.random.manual_seed(0)
    
    batch_size = 4
    nheads = 8
    
    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    
    out, lse = flash_attn_func(q, k, v, causal=causal)
    
    assert out.shape == (batch_size, seqlen, nheads, d)
    assert lse.shape == (batch_size, nheads, seqlen)
    assert out.dtype == dtype
    assert lse.dtype == torch.float32
    
    # Test backward
    g = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
def test_flash_attn_sm120_mha_types(d, mha_type, dtype):
    """Test different MHA types (MHA, MQA, GQA) on SM120."""
    device = "cuda"
    torch.random.manual_seed(0)
    
    batch_size = 2
    seqlen = 256
    nheads = 8
    nheads_kv = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    
    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen, nheads_kv, d, device=device, dtype=dtype, requires_grad=True
    )
    
    out, _ = flash_attn_func(q, k, v, causal=True)
    
    assert out.shape == (batch_size, seqlen, nheads, d)
    
    # Test backward
    g = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape

