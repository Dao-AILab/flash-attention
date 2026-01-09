# Test torch.compile compatibility for Flash Attention 4 (FA4).

import pytest
import torch
from torch import nn
from torch.library import opcheck


from flash_attn.cute import flash_attn_func


class SimpleAttention(nn.Module):
    """Simple attention module using flash_attn_func"""

    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        out = flash_attn_func(q, k, v)
        out = out.reshape(B, N, self.embed_size)
        out = self.out_proj(out)
        return out


def test_opcheck():
    # Define sample inputs for flash_attn_fwd
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)

    sample_args_fwd = (q, k, v)

    opcheck(torch.ops.flash_attn_cute.flash_attn_fwd, sample_args_fwd)

    # Prepare inputs for flash_attn_bwd (simulate a forward pass to produce 'out' and 'lse')
    out, lse = torch.ops.flash_attn_cute.flash_attn_fwd(*sample_args_fwd)
    dout = torch.randn_like(out)

    sample_args_bwd = (q, k, v, out, dout, lse)

    opcheck(torch.ops.flash_attn_cute.flash_attn_bwd, sample_args_bwd)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_minimal_compile(backend):
    """Compile top-level interface"""
    # Create simple inputs
    batch_size = 2
    seqlen = 128
    num_heads = 4
    head_dim = 64

    q = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )

    # Run without compilation (baseline)
    out_eager = flash_attn_func(q, k, v)

    # Compile and run
    compiled_fn = torch.compile(
        lambda q, k, v: flash_attn_func(q, k, v),
        backend=backend,
        fullgraph=True,
    )
    out_compiled = compiled_fn(q, k, v)

    # Verify outputs match
    assert torch.allclose(out_eager, out_compiled, rtol=1e-3, atol=1e-3), (
        f"Outputs differ between eager and compiled with {backend} backend"
    )


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_integration_compile(backend):
    """Integration test: full attention module"""

    # Create model and input
    B = 4
    N = 256
    embed_dim = 512
    num_heads = 8
    model = SimpleAttention(embed_dim, num_heads).cuda().bfloat16()
    input_tensor = torch.randn(B, N, embed_dim, device="cuda", dtype=torch.bfloat16)

    # Run without compilation
    output_eager = model(input_tensor)

    # Compile and run
    compiled_model = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
    )
    output_compiled = compiled_model(input_tensor)

    # Verify outputs match
    assert torch.allclose(output_eager, output_compiled, rtol=1e-3, atol=1e-3), (
        f"Outputs differ between eager and compiled with {backend} backend"
    )


def test_integration_backward():
    """Test backward pass through compiled model and compare gradients."""

    # Create model and input
    batch_size = 4
    seqlen = 256
    embed_dim = 512
    num_heads = 8

    # Create two separate models for eager and compiled runs
    model_eager = SimpleAttention(embed_dim, num_heads).cuda().bfloat16()
    model_compiled = SimpleAttention(embed_dim, num_heads).cuda().bfloat16()
    model_compiled.load_state_dict(model_eager.state_dict())

    input_tensor_eager = torch.randn(
        batch_size,
        seqlen,
        embed_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    input_tensor_compiled = input_tensor_eager.clone().detach().requires_grad_(True)

    # Eager run
    output_eager = model_eager(input_tensor_eager)
    loss_eager = output_eager.sum()
    loss_eager.backward()

    # Compiled run
    compiled_model = torch.compile(
        model_compiled,
        fullgraph=True,
    )
    output_compiled = compiled_model(input_tensor_compiled)
    loss_compiled = output_compiled.sum()
    loss_compiled.backward()

    # Check that gradients were computed
    assert model_compiled.qkv_proj.weight.grad is not None, (
        "qkv_proj.weight.grad should be computed"
    )
    assert model_compiled.out_proj.weight.grad is not None, (
        "out_proj.weight.grad should be computed"
    )

    # Check that gradients match
    assert torch.allclose(
        model_eager.qkv_proj.weight.grad, model_compiled.qkv_proj.weight.grad
    ), "qkv_proj.weight.grad does not match between eager and compiled"
    assert torch.allclose(
        model_eager.out_proj.weight.grad, model_compiled.out_proj.weight.grad
    ), "out_proj.weight.grad does not match between eager and compiled"


@pytest.mark.parametrize(
    "test_case",
    [
        ("causal", {"causal": True}),
        ("window", {"window_size": (64, 64)}),
        ("gqa", {"pack_gqa": True}),
        ("deterministic", {"deterministic": True}),
        ("softcap", {"softcap": 15.0}),
    ],
)
def test_compile_with_parameters(test_case):
    """Test compilation with different flash_attn_func parameters."""
    test_name, kwargs = test_case

    batch_size = 2
    seqlen = 512
    num_heads = 8
    head_dim = 64

    q = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )

    # Eager execution
    out_eager = flash_attn_func(q, k, v, **kwargs)

    # Compiled execution
    compiled_fn = torch.compile(
        lambda q, k, v: flash_attn_func(q, k, v, **kwargs),
        fullgraph=True,
    )
    out_compiled = compiled_fn(q, k, v)

    # Verify outputs match
    assert torch.allclose(out_eager, out_compiled, rtol=1e-3, atol=1e-3), (
        f"{test_name}: outputs differ between eager and compiled"
    )


def test_export():
    """Test torch.export functionality."""

    class SimpleAttention(nn.Module):
        def __init__(self, embed_size, num_heads):
            super().__init__()
            self.embed_size = embed_size
            self.num_heads = num_heads
            self.head_dim = embed_size // num_heads
            self.qkv_proj = nn.Linear(embed_size, 3 * embed_size)
            self.out_proj = nn.Linear(embed_size, embed_size)

        def forward(self, x):
            N, seq_length, _ = x.shape
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(N, seq_length, self.num_heads, self.head_dim)
            k = k.view(N, seq_length, self.num_heads, self.head_dim)
            v = v.view(N, seq_length, self.num_heads, self.head_dim)
            out = flash_attn_func(q, k, v)
            out = out.reshape(N, seq_length, self.embed_size)
            out = self.out_proj(out)
            return out

    model = SimpleAttention(512, 8).cuda().bfloat16()
    input_tensor = torch.randn(2, 128, 512, device="cuda", dtype=torch.bfloat16)

    # Get baseline output
    expected = model(input_tensor)

    # Export the model
    exported = torch.export.export(model, (input_tensor,))

    # Run exported model
    output = exported.module()(input_tensor)

    # Verify outputs match
    assert torch.allclose(expected, output, rtol=1e-3, atol=1e-3), (
        "Exported model output differs from expected"
    )
