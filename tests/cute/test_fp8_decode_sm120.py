"""fp8 (e4m3) KV-cache decode correctness on consumer Blackwell (sm_120).

The SM120 GEMV decode kernel (``flash_fwd_decode_sm120.FlashAttentionDecodeSm120``)
is the *only* sm_120 path that can consume an fp8 K/V cache: fp8 prefill is a
no-go and the standard SM120 forward asserts ``q.dtype == k.dtype == v.dtype``.
``interface.py`` therefore auto-routes a bf16/fp16 Q + fp8 (e4m3/e5m2) K/V +
``k_descale``/``v_descale`` + ``seqlen_q == 1`` call to this kernel *regardless*
of the ``FLASH_ATTENTION_SM120_DECODE_KERNEL`` env flag (the flag stays a manual
override for the bf16 decode kernel).

These tests quantize K/V per-(batch, kv-head) to e4m3, dequantize to fp32, run an
fp32 SDPA reference (bottom-right causal), and check the kernel output against
that fp8-quantized reference.  Tolerance is rel-err < 1e-2 (fp8 quant noise; the
observed baseline is ~1.7e-3).

The reference applies the same per-(batch, kv-head) fp8 quantization to K/V and
runs an fp32 SDPA on the dequantized tensors.

Runs both with and without the env flag (parametrized): the auto-enable path must
pass without the flag, and the flag must remain a harmless no-op for fp8 K/V.

Skips when not on sm_120 (compute capability 12.x) or when CUDA is unavailable.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd, _fp8_decode_dsl_supported


FP8 = torch.float8_e4m3fn
E4M3_MAX = 448.0


def _sm120_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cc = torch.cuda.get_device_capability(0)
    if cc != (12, 0):
        pytest.skip(f"Test targets sm_120, current device is sm_{cc[0]}{cc[1]}")
    if not _fp8_decode_dsl_supported():
        from importlib.metadata import version as _pkg_version
        try:
            _v = _pkg_version("nvidia-cutlass-dsl")
        except Exception:
            _v = "<unknown>"
        pytest.skip(
            f"sm120 fp8 KV-cache decode is unsupported on nvidia-cutlass-dsl {_v} "
            "(4.5.x >= 4.5.2 DSL codegen regression: nvgpu.cvt_fpext rejects scalar "
            "f8E4M3FN). Install 4.5.1 to exercise the fp8 decode path."
        )


def _quantize_kv_e4m3(x: torch.Tensor):
    """x: (b, s, h, d).  Per-(b, h) amax -> per-tensor scale.

    Returns (q_fp8, descale) where descale is the (b, h) dequant multiplier.
    """
    amax = x.abs().amax(dim=(1, 3)).clamp(min=1e-8)  # (b, h)
    scale = E4M3_MAX / amax
    descale = (amax / E4M3_MAX).to(torch.float32)  # (b, h)
    xq = (x.float() * scale[:, None, :, None]).clamp(-E4M3_MAX, E4M3_MAX).to(FP8)
    return xq, descale


def _sdpa_ref(q, k, v, scale, causal):
    """q: (b, sq, hq, d); k/v: (b, sk, hkv, d).  GQA via repeat; bottom-right causal."""
    b, sq, hq, d = q.shape
    _, sk, hkv, _ = k.shape
    r = hq // hkv
    k = k.repeat_interleave(r, dim=2)
    v = v.repeat_interleave(r, dim=2)
    qt = q.transpose(1, 2)  # (b, hq, sq, d)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    scores = torch.einsum("bhqd,bhkd->bhqk", qt, kt) * scale
    if causal:
        # bottom-right: query i attends keys 0..(sk - sq + i)
        qi = torch.arange(sq, device=q.device)[:, None]
        ki = torch.arange(sk, device=q.device)[None, :]
        allowed = ki <= (sk - sq + qi)
        scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    p = scores.softmax(dim=-1)
    o = torch.einsum("bhqk,bhkd->bhqd", p, vt)
    return o.transpose(1, 2)  # (b, sq, hq, d)


def _ref_fp8(q, kq, vq, kdesc, vdesc, scale, causal):
    """Dequantize fp8 -> fp32, then fp32 SDPA (the fp8-quantized reference)."""
    kf = kq.float() * kdesc[:, None, :, None]
    vf = vq.float() * vdesc[:, None, :, None]
    return _sdpa_ref(q.float(), kf, vf, scale, causal)


def _relerr(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


REL_TOL = 1e-2  # fp8 quant noise; observed baseline ~1.7e-3


# (hq, hkv): GQA R in {2, 4, 8} -> R=hq/hkv = 2 (32/16), 4 (32/8), 8 (32/4)
@pytest.mark.parametrize("hq,hkv", [(32, 16), (32, 8), (32, 4)])
@pytest.mark.parametrize("sk", [4096, 16384])
@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("env_flag", [False, True], ids=["noenv", "env"])
def test_fp8_decode_correctness(hq, hkv, sk, batch, env_flag, monkeypatch):
    _sm120_only()
    head_dim = 128
    seqlen_q = 1
    causal = True

    if env_flag:
        monkeypatch.setenv("FLASH_ATTENTION_SM120_DECODE_KERNEL", "1")
    else:
        monkeypatch.delenv("FLASH_ATTENTION_SM120_DECODE_KERNEL", raising=False)

    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(batch, seqlen_q, hq, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(batch, sk, hkv, head_dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(batch, sk, hkv, head_dim, device=dev, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(head_dim)

    kq, kdesc = _quantize_kv_e4m3(k)
    vq, vdesc = _quantize_kv_e4m3(v)

    out, _ = _flash_attn_fwd(
        q,
        kq,
        vq,
        softmax_scale=scale,
        causal=causal,
        k_descale=kdesc,
        v_descale=vdesc,
        pack_gqa=False,
    )

    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (batch, seqlen_q, hq, head_dim)

    ref = _ref_fp8(q, kq, vq, kdesc, vdesc, scale, causal)
    err = _relerr(out, ref)
    assert err < REL_TOL, (
        f"fp8 decode rel-err {err:.4e} >= {REL_TOL:.0e} "
        f"(b{batch} sk{sk} h{hq}/{hkv} d{head_dim} env={env_flag})"
    )


def test_fp8_decode_auto_enables_without_env_flag(monkeypatch):
    """fp8 K/V decode must route to the decode kernel even with the env flag unset.

    Without auto-enable a user passing an fp8 K/V cache would hit the
    ``q.dtype == k.dtype == v.dtype`` assert in ``_flash_attn_fwd`` and could not
    use their cache at all.  This is a focused guard on that usability contract.
    """
    _sm120_only()
    monkeypatch.delenv("FLASH_ATTENTION_SM120_DECODE_KERNEL", raising=False)
    assert "FLASH_ATTENTION_SM120_DECODE_KERNEL" not in os.environ

    head_dim, sk, hq, hkv = 128, 4096, 32, 4
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(16, 1, hq, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(16, sk, hkv, head_dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(16, sk, hkv, head_dim, device=dev, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(head_dim)
    kq, kdesc = _quantize_kv_e4m3(k)
    vq, vdesc = _quantize_kv_e4m3(v)

    # Would raise AssertionError on the standard path; succeeds only via decode.
    out, _ = _flash_attn_fwd(
        q, kq, vq, softmax_scale=scale, causal=True,
        k_descale=kdesc, v_descale=vdesc, pack_gqa=False,
    )
    err = _relerr(out, _ref_fp8(q, kq, vq, kdesc, vdesc, scale, True))
    assert err < REL_TOL, f"rel-err {err:.4e}"
