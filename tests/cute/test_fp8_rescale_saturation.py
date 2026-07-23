# Regression test for the fp8 lazy-rescale saturation bug: with a rescale
# threshold above log2(448) - max_offset(8) = 0.807, P values overflow e4m3's
# finite range (448) at the P convert while row_sum accumulates the
# unsaturated fp32 value, underweighting dominant keys on peaked softmax rows.
# The trigger is the logit distribution (std >~ 0.8), not shape -- so this
# test sweeps logit std and compares against an fp32 reference computed from
# the dequantized inputs (isolating kernel numerics from quantization noise).
import math

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd

B, H, S, D = 1, 4, 4096, 128
SCALE = 1.0 / math.sqrt(D)
E4M3 = torch.float8_e4m3fn


def _sm100_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _ref_fp32(qf, kf, vf):
    logits = torch.einsum("bshd,bthd->bhst", qf, kf) * SCALE
    p = torch.softmax(logits.float(), dim=-1)
    return torch.einsum("bhst,bthd->bshd", p, vf.float())


def _quant_per_tensor(x):
    amax = x.abs().amax().clamp(min=1e-8)
    scale = 448.0 / amax
    xq = (x * scale).clamp(-448, 448).to(E4M3)
    descale = torch.full((B, H), 1.0 / scale, device=x.device, dtype=torch.float32)
    return xq, descale


def _pearson(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


@pytest.mark.skipif(not _sm100_available(), reason="requires SM100/SM103")
@pytest.mark.parametrize("logit_std", [0.1, 0.5, 0.8, 1.0, 1.5, 2.5, 5.0])
def test_fp8_no_rescale_saturation(logit_std):
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    # calibrate q so the attention logits have the target std
    sample = torch.einsum("bshd,bthd->bhst", q[:, :256].float(), k[:, :256].float()) * SCALE
    q = q * (logit_std / sample.std().clamp(min=1e-6)).to(torch.bfloat16)

    q8, qs = _quant_per_tensor(q.float())
    k8, ks = _quant_per_tensor(k.float())
    v8, vs = _quant_per_tensor(v.float())
    # dequantized-exact reference: quantization error cancels on both sides
    ref = _ref_fp32(q8.float() * qs[0, 0], k8.float() * ks[0, 0], v8.float() * vs[0, 0])

    out = _flash_attn_fwd(
        q8, k8, v8, causal=False, q_descale=qs, k_descale=ks, v_descale=vs
    )[0]
    p = _pearson(out, ref)
    # stock threshold 4.0 fails std >= 0.8 cells (pearson 0.92-0.99);
    # any threshold <= 0.807 passes all cells at fp8-noise residuals.
    assert p >= 0.999, f"logit_std={logit_std}: pearson {p:.6f} < 0.999"
