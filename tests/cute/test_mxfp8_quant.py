"""Host-side MXFP8 quantization and interleaved scale layout."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def blockscaled_utils():
    # This pure PyTorch module must be testable without importing the CUDA-only
    # flash_attn.cute package initializer on CPU development machines.
    source = (
        Path(__file__).resolve().parents[2] / "flash_attn/cute/blockscaled_utils.py"
    )
    spec = importlib.util.spec_from_file_location("blockscaled_utils", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v_quantizer_key_groups_and_interleaved_layout(blockscaled_utils):
    # Both physical MN and K span more than one chunk; neither tail is aligned.
    generator = torch.Generator().manual_seed(0)
    v = torch.randn(2, 129, 2, 129, generator=generator, dtype=torch.bfloat16)
    v *= torch.exp2(torch.arange(129) % 9 - 4)
    v[:, 32:64] *= 16
    v[1] *= 0.25
    v[:, :, 1] *= 2
    v_fp8, sfv, scales = blockscaled_utils.quantize_v_to_fp8_with_sf(v)
    assert v_fp8.shape == v.shape
    assert v_fp8.dtype == torch.float8_e4m3fn
    assert v_fp8.is_contiguous()
    assert sfv.shape == (2, 2, 2, 2, 32, 4, 4)
    assert sfv.dtype == torch.float8_e8m0fnu
    assert sfv.is_contiguous()
    assert scales.shape == (2, 5, 2, 129)
    actual = blockscaled_utils.dequantize_v_fp8_with_sf(v_fp8, scales)
    sf_bytes = sfv.view(torch.uint8)
    for b in range(2):
        for h in range(2):
            for d in range(129):
                for g in range(5):
                    values = v[b, 32 * g : 32 * (g + 1), h, d].double()
                    # Scalar FP64 ceil rule is independent of the vectorized quantizer.
                    exponent = math.ceil(math.log2(values.abs().max().item() / 448.0))
                    expected_scale = 2.0**exponent
                    assert scales[b, g, h, d].item() == expected_scale
                    assert (
                        sf_bytes[b, h, d // 128, g // 4, d % 32, d % 128 // 32, g % 4]
                        == exponent + 127
                    )
                    expected = (values / expected_scale).to(
                        torch.float8_e4m3fn
                    ).float() * expected_scale
                    torch.testing.assert_close(
                        actual[b, 32 * g : 32 * (g + 1), h, d], expected, rtol=0, atol=0
                    )
    assert torch.isfinite(actual).all()


def test_v_quantizer_ceil_rule_zero_and_tiny_groups(blockscaled_utils):
    # 449 must select scale 2, not saturate under the floor-rule scale 1.
    v = torch.zeros(1, 65, 1, 4)
    v[:, :32, :, 1] = 448
    v[:, :32, :, 2] = 449
    v[:, :32, :, 3] = -(2.0**-100)
    v[:, 32:64] = 1
    v[:, 64:] = 1024
    v_fp8, sfv, scales = blockscaled_utils.quantize_v_to_fp8_with_sf(v)
    torch.testing.assert_close(
        scales[0, 0, 0], torch.tensor([1, 1, 2, 2.0**-108]), rtol=0, atol=0
    )
    actual = blockscaled_utils.dequantize_v_fp8_with_sf(v_fp8, scales)
    assert actual[0, 0, 0, 0] == 0
    assert actual[0, 0, 0, 1] == 448
    assert actual[0, 0, 0, 2] == 448
    assert actual[0, 0, 0, 3] == -(2.0**-100)
    torch.testing.assert_close(actual[:, 32:], v[:, 32:], rtol=0, atol=0)
    assert torch.isfinite(sfv.float()).all()


@pytest.mark.parametrize("operand", ["qk", "v"])
def test_ceil_rule_near_power_of_two_scale(blockscaled_utils, operand):
    boundary = torch.tensor(448.0 * 2.0**-100)
    above = torch.nextafter(boundary, torch.tensor(float("inf")))
    if operand == "qk":  # Groups along hdim.
        x = torch.zeros(1, 1, 1, 64)
        x[..., 0], x[..., 32] = boundary, above
        _, _, scales = blockscaled_utils.quantize_to_fp8_with_sf(x)
    else:  # Groups along keys.
        x = torch.stack((boundary, above)).reshape(1, 1, 1, 2)
        _, _, scales = blockscaled_utils.quantize_v_to_fp8_with_sf(x)
    # FP32 log2 rounds both ratios to -100, but the second needs the next scale.
    torch.testing.assert_close(
        scales.flatten(), torch.tensor([2.0**-100, 2.0**-99]), rtol=0, atol=0
    )


def test_v_quantizer_restarts_groups_per_batch(blockscaled_utils):
    v = torch.ones(2, 33, 1, 64, dtype=torch.bfloat16)
    v[1] *= 1024
    whole = blockscaled_utils.quantize_v_to_fp8_with_sf(v)
    for b in range(2):
        separate = blockscaled_utils.quantize_v_to_fp8_with_sf(v[b : b + 1])
        for actual, expected in zip(whole, separate):
            # Byte comparisons also work for float8 dtypes lacking comparison kernels.
            torch.testing.assert_close(
                actual[b : b + 1].view(torch.uint8),
                expected.view(torch.uint8),
                rtol=0,
                atol=0,
            )


def test_qk_interleave_matches_chunk_layout(blockscaled_utils):
    # Q/K scales: MN = seqlen (unaligned, two chunks), K = ceil(hdim / 32) (unaligned).
    generator = torch.Generator().manual_seed(0)
    sf = torch.randint(0, 255, (2, 130, 3, 5), generator=generator, dtype=torch.uint8)
    chunks = blockscaled_utils.interleave_sf(sf.view(torch.float8_e8m0fnu))
    assert chunks.shape == (2, 3, 2, 2, 32, 4, 4) and chunks.is_contiguous()
    chunk_bytes = chunks.view(torch.uint8)
    for s in (0, 31, 32, 127, 128, 129):
        for j in range(5):
            assert torch.equal(
                chunk_bytes[:, :, s // 128, j // 4, s % 32, s % 128 // 32, j % 4],
                sf[:, s, :, j],
            )
    # Padding rows/columns are zero bytes, not stale memory.
    assert chunk_bytes[:, :, 1, :, 2:, :, :].eq(0).all()
    assert chunk_bytes[:, :, :, 1, :, :, 1:].eq(0).all()
