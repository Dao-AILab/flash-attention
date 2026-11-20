"""Regression coverage for the ordinary dense-inference PT2 boundary.

These tests need a CUDA device supported by FA4. They intentionally do not cover
training, variable-length attention, or the richer attention interfaces.
"""

import importlib

import pytest
import torch


@pytest.fixture(scope="module")
def interface():
    if not torch.cuda.is_available():
        pytest.skip("FA4 export regression tests require a supported CUDA GPU")
    return importlib.import_module("flash_attn.cute.interface")


def make_inputs(batch, dtype):
    generator = torch.Generator(device="cuda").manual_seed(2026)
    # Last dimension contiguous; sequence/head transpose matches provider inputs.
    return tuple(
        torch.randn(batch, 12, 128, 64, device="cuda", dtype=dtype, generator=generator)
        .transpose(1, 2)
        for _ in range(3)
    )


def reference_attention(q, k, v):
    scores = torch.einsum("bqhd,bkhd->bhqk", q.double(), k.double()) * q.shape[-1] ** -0.5
    return (
        torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.double()),
        scores.logsumexp(-1),
    )


def assert_boundary(exported):
    targets = [str(node.target) for node in exported.graph.nodes if node.op == "call_function"]
    assert any("flash_attn_cute.flash_attn_fwd_dense_inference" in target for target in targets)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("return_lse", [False, True])
@torch.no_grad()
def test_dense_inference_numerics_and_opcheck(interface, dtype, batch, return_lse):
    inputs = make_inputs(batch, dtype)
    reference, reference_lse = reference_attention(*inputs)
    tolerance = 0.01 if dtype == torch.float16 else 0.06
    output, lse = interface.flash_attn_func(*inputs, return_lse=return_lse)
    torch.testing.assert_close(output.double(), reference, atol=tolerance, rtol=tolerance)
    assert output.shape == inputs[0].shape
    assert output.dtype == dtype
    assert output.is_contiguous()
    if return_lse:
        torch.testing.assert_close(lse.double(), reference_lse, atol=tolerance, rtol=tolerance)
    else:
        assert lse is None
    result = torch.library.opcheck(
        torch.ops.flash_attn_cute.flash_attn_fwd_dense_inference.default,
        (*inputs, None, False, 0.0, None, None, 1, None, return_lse),
    )
    assert all(value == "SUCCESS" for value in result.values())


class Attention(torch.nn.Module):
    def forward(self, q, k, v):
        from flash_attn.cute.interface import flash_attn_func

        return flash_attn_func(q, k, v)[0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("strict", [False, True])
@torch.no_grad()
def test_dense_inference_export_roundtrip(interface, dtype, batch, strict, tmp_path):
    inputs = make_inputs(batch, dtype)
    expected, _ = reference_attention(*inputs)
    tolerance = 0.01 if dtype == torch.float16 else 0.06
    exported = torch.export.export(Attention().eval(), inputs, strict=strict)
    assert_boundary(exported)
    destination = tmp_path / "dense-attention.pt2"
    torch.export.save(exported, destination)
    loaded = torch.export.load(destination)
    assert_boundary(loaded)
    torch.testing.assert_close(
        loaded.module()(*inputs).double(), expected, atol=tolerance, rtol=tolerance
    )


@torch.no_grad()
def test_dense_inference_aoti_package_roundtrip(interface, tmp_path):
    import torch._inductor

    inputs = make_inputs(1, torch.float16)
    expected, _ = reference_attention(*inputs)
    exported = torch.export.export(Attention().eval(), inputs, strict=True)
    assert_boundary(exported)
    package = tmp_path / "dense-attention-aoti.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
    loaded = torch._inductor.aoti_load_package(str(package))
    torch.testing.assert_close(loaded(*inputs).double(), expected, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("option", ["scales", "config", "aux_tensors", "aux_scalars"])
def test_extended_inputs_keep_existing_route(interface, monkeypatch, option):
    sentinel = object()
    calls = []

    def fallback(*args):
        calls.append(args)
        return sentinel

    def unexpected_boundary(*args, **kwargs):
        pytest.fail("Extended inputs must retain the existing autograd-function route")

    monkeypatch.setattr(interface.FlashAttnFunc, "apply", fallback)
    monkeypatch.setattr(interface, "_flash_attn_fwd_dense_inference", unexpected_boundary)
    inputs = make_inputs(1, torch.float16)
    value = object()
    assert interface.flash_attn_func(*inputs, **{option: value}) is sentinel
    assert any(argument is value for argument in calls[0])


@pytest.mark.parametrize("grad_input", [0, 1, 2])
def test_training_keeps_existing_route(interface, monkeypatch, grad_input):
    sentinel = object()
    monkeypatch.setattr(interface.FlashAttnFunc, "apply", lambda *args: sentinel)

    def unexpected_boundary(*args, **kwargs):
        pytest.fail("Inputs requiring gradients must retain their existing route")

    monkeypatch.setattr(interface, "_flash_attn_fwd_dense_inference", unexpected_boundary)
    inputs = make_inputs(1, torch.float16)
    inputs[grad_input].requires_grad_(True)
    assert interface.flash_attn_func(*inputs) is sentinel
