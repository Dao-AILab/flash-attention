"""Mirrored BF16 backward of the MXFP8 forwards (flash_attn/cute/mxfp8_mirror.py; NOTE [MXFP8
mirrored BF16 backward] in flash_attn/cute/flash_bwd_sm100.py)."""

import math

import pytest
import torch

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute.blockscaled_utils import (  # noqa: E402
    dequantize_fp8_with_sf,
    dequantize_v_fp8_with_sf,
    interleave_sf,
    quantize_to_fp8_with_sf,
    quantize_v_to_fp8_with_sf,
)
from flash_attn.cute.config import Mxfp8Mirror, select_bwd_config  # noqa: E402
from flash_attn.cute.interface import (  # noqa: E402
    MxMirrorBwd,
    _flash_attn_bwd,
    _flash_attn_fwd,
    _get_bwd_inputs,
)
from flash_attn.cute.mxfp8_mirror import mxfp8_mirror_backward, quantize_for_mirror  # noqa: E402
from flash_attn.cute.mxfp8_quant import quantize_keys, quantize_rows  # noqa: E402

from mxfp8_mirror_reference import (  # noqa: E402
    autograd_backward,
    autograd_per_head,
    emulate_backward,
    oracle_per_head,
    per_head,
    rel_l2,
)

# "qk": QK-only MXFP8 (BF16 V); otherwise native PV with that BlockScales.pv_normalizer
MODES = ["qk", "rounded_p", "fp32"]
# (batch, seqlen_q, seqlen_k, heads_q, heads_kv): aligned; GQA with partial tiles; causal offset
# 256 (2CTA keeps the pair's band); offset 128 (hd128 falls back to 1CTA); seqlen_q > seqlen_k
# (causal rows without a visible key: m = -inf, L = 0); seqlen_q <= 128 (one forward query tile
# of 128 rows)
SHAPES = [(1, 512, 512, 2, 2), (2, 300, 300, 4, 2), (1, 384, 640, 2, 1), (1, 384, 512, 2, 2),
          (1, 512, 384, 2, 2), (1, 96, 352, 2, 2)]


@pytest.fixture(autouse=True)
def requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("requires SM100 / SM110")


def make_inputs(b, sq, sk, hq, hk, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, sq, hq, d, generator=g, device="cuda")
    k = torch.randn(b, sk, hk, d, generator=g, device="cuda")
    v = torch.randn(b, sk, hk, d, generator=g, device="cuda")
    # nonuniform MX scale groups: per-token magnitudes and a few outlier channels
    q = q * torch.exp(0.5 * torch.randn(b, sq, hq, 1, generator=g, device="cuda"))
    k[..., :2] *= 4
    v = v * torch.exp2((torch.arange(sk, device="cuda") // 32 % 4 - 2).float())[None, :, None, None]
    dout = torch.randn(b, sq, hq, d, generator=g, device="cuda")
    return [t.bfloat16() for t in (q, k, v, dout)]


def mxfp8_forward(q, k, v, causal, softmax_scale, mode):
    """The MXFP8 forward with stats_out on BF16 inputs: (O, the mirror's operands, stats)."""
    inputs = quantize_for_mirror(q, k, v, causal, None if mode == "qk" else mode)
    stats = torch.empty(q.shape[0], q.shape[2], q.shape[1], 2, dtype=torch.float32, device="cuda")
    scales = inputs.scales._replace(stats_out=stats)
    out = _flash_attn_fwd(
        inputs.q8, inputs.k8, inputs.v, softmax_scale=softmax_scale, causal=causal, scales=scales
    )[0]
    return out, inputs.operands, stats


def run(mode, causal, shape, d, deterministic=False, seed=0):
    b, sq, sk, hq, hk = shape
    q, k, v, dout = make_inputs(b, sq, sk, hq, hk, d, seed)
    scale = 1 / math.sqrt(d)
    out, operands, stats = mxfp8_forward(q, k, v, causal, scale, mode)
    grads = mxfp8_mirror_backward(dout, out, operands, stats, causal, scale, deterministic)
    v_band = operands.v_band if operands.v_band is not None else operands.v_hat
    refs = (operands.q_hat, operands.k_hat, operands.v_hat, v_band)
    kw = dict(softmax_scale=scale, causal=causal, native_pv=mode != "qk")
    return dict(q=q, k=k, v=v, dout=dout, out=out, stats=stats, grads=grads, refs=refs, kw=kw)


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("seqlen", [300, 512])
def test_quantize_rows_matches_eager(seqlen, d):
    """The fused quantizer writes the eager quantizer's payload and interleave_sf scales byte for
    byte (including the padding of partial 128-row tiles and of hdim 64's half-filled scale
    chunk), and the exact BF16 dequantization."""
    x = make_inputs(2, seqlen, seqlen, 3, 3, d)[0]
    x8, sf, x_hat = quantize_rows(x)
    ref8, ref_sf, ref_scale = quantize_to_fp8_with_sf(x)
    assert torch.equal(x8.view(torch.uint8), ref8.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8), interleave_sf(ref_sf).view(torch.uint8).reshape(-1))
    assert torch.equal(x_hat, dequantize_fp8_with_sf(ref8, ref_scale).bfloat16())


@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("seqlen", [300, 512])
def test_quantize_keys_matches_eager(seqlen, d):
    """Native PV's V along keys: the eager quantizer's payload and interleave_sf scales byte for
    byte (a partial last 32-key group and partial scale tiles included), and its dequantization."""
    v = make_inputs(2, seqlen, seqlen, 3, 3, d)[2]
    v8, sf, v_hat = quantize_keys(v)
    ref8, ref_sf, ref_scale = quantize_v_to_fp8_with_sf(v)
    assert torch.equal(v8.view(torch.uint8), ref8.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8), ref_sf.view(torch.uint8).reshape(-1))
    assert torch.equal(v_hat, dequantize_v_fp8_with_sf(ref8, ref_scale).bfloat16())


def pv_normalizer(mode):
    return "fp32" if mode == "qk" else mode  # QK-only normalizes by the FP32 sum


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_mirror_matches_emulator(mode, causal, d, shape):
    """The kernel matches its kernel-faithful emulator to accumulation-order noise (both rounded
    to the BF16 outputs), and the mirrored oracle to the recipe's inherent BF16-class error.

    The S accumulation order differs (tensor core vs FP64), so a P~ on an E4M3 rounding knife edge
    can round the other way: one flip moves a P-hat by 1/16, so native PV's dV gets a looser bound.
    Both bounds are empirical, at least 2x above the largest error measured on B200.
    """
    if d == 256 and mode != "qk":
        pytest.skip("hdim 256 native PV runs the MXFP8 backward (test_mxfp8_hd256_bwd.py)")
    r = run(mode, causal, shape, d)
    emu = per_head(emulate_backward, r["refs"], r["out"], r["dout"], r["stats"], **r["kw"])
    orc = per_head(oracle_per_head, r["refs"], r["out"], r["dout"], r["stats"], **r["kw"],
                   pv_normalizer=pv_normalizer(mode))
    errors = {}
    for name, got, e, o in zip("qkv", r["grads"], emu, orc):
        assert torch.isfinite(got).all(), f"d{name} is not finite"
        errors[name] = (rel_l2(got, e.to(torch.bfloat16)), rel_l2(got, o))
    for name, (vs_emu, vs_oracle) in errors.items():
        # 5e-4: one BF16 dS flip weighs more with few queries (seqlen_q = 96 measured 2.5e-4)
        bound = 6e-4 if name == "v" and mode != "qk" else 5e-4
        assert vs_emu <= bound, f"d{name}: kernel vs emulator {vs_emu:.2e} ({errors})"
        assert vs_oracle <= 3e-2, f"d{name}: kernel vs oracle {vs_oracle:.2e} ({errors})"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_recipe_is_the_straight_through_derivative(mode, causal):
    """The recipe's closed form equals torch.autograd of an FP64 quantized forward at the final max
    with straight-through roundings (native PV: V-hat = q(V), band elements on V), given that
    forward's O and L; the kernel tracks it to BF16-class error."""
    r = run(mode, causal, (1, 384, 384, 2, 1), 64)
    qh, kh, vh, v = r["refs"]
    for h in range(qh.shape[2]):
        heads = (qh[0, :, h], kh[0, :, 0], vh[0, :, 0], v[0, :, 0])
        m = r["stats"][0, h, :, 0]
        auto, o, L = autograd_backward(*heads, r["dout"][0, :, h], m, **r["kw"],
                                       pv_normalizer=pv_normalizer(mode))
        recipe = emulate_backward(*heads, o, r["dout"][0, :, h], m, L, **r["kw"], faithful=False)
        for name, x, y in zip("qkv", recipe, auto):
            assert rel_l2(x, y) <= 1e-6, f"d{name}: recipe vs autograd {rel_l2(x, y):.2e}"
    auto = per_head(autograd_per_head, r["refs"], r["out"], r["dout"], r["stats"], **r["kw"],
                    pv_normalizer=pv_normalizer(mode))
    for name, got, y in zip("qkv", r["grads"], auto):
        assert rel_l2(got, y) <= 1.5e-2, f"d{name}: kernel vs autograd {rel_l2(got, y):.2e}"


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("mode", ["qk", "rounded_p"])
def test_causal_no_future_leakage(mode, d):
    """dQ rows before t0 are bitwise unchanged when keys / values / dO at positions >= t0 change
    (deterministic, so the dQ adds keep their order)."""
    shape, t0 = (1, 640, 640, 2, 2), 300
    a = run(mode, True, shape, d, deterministic=True)
    q, k, v, dout = (a[n].clone() for n in ("q", "k", "v", "dout"))
    k[:, t0:] = torch.randn_like(k[:, t0:]) * 3
    v[:, t0:] = torch.randn_like(v[:, t0:]) * 30
    dout[:, t0:] = torch.randn_like(dout[:, t0:])
    scale = 1 / math.sqrt(d)
    out, operands, stats = mxfp8_forward(q, k, v, True, scale, mode)
    dq, _, _ = mxfp8_mirror_backward(dout, out, operands, stats, True, scale, True)
    assert torch.equal(dq[:, :t0].view(torch.int16), a["grads"][0][:, :t0].view(torch.int16))
    assert not torch.equal(dq[:, t0:], a["grads"][0][:, t0:])


@pytest.mark.parametrize("mode", ["qk", "rounded_p"])
def test_deterministic(mode):
    a = run(mode, True, (2, 300, 300, 4, 2), 128, deterministic=True)
    b = run(mode, True, (2, 300, 300, 4, 2), 128, deterministic=True)
    for x, y in zip(a["grads"], b["grads"]):
        assert torch.equal(x.view(torch.int16), y.view(torch.int16))


def test_mirror_rejects_mismatched_stats_and_scope():
    r = run("qk", False, (1, 256, 256, 2, 2), 64)
    qh, kh, v, _ = r["refs"]
    args = (qh, kh, v, r["out"], r["dout"], None)
    with pytest.raises(ValueError, match="v_bf16 is required exactly"):
        _flash_attn_bwd(*args, causal=True, mx_mirror=MxMirrorBwd(r["stats"], Mxfp8Mirror.PV))
    short_stats = MxMirrorBwd(r["stats"][:, :1].contiguous(), Mxfp8Mirror.QK)
    with pytest.raises(ValueError, match="stats must have shape"):
        _flash_attn_bwd(*args, mx_mirror=short_stats)
    with pytest.raises(ValueError, match="does not support local"):
        _flash_attn_bwd(*args, window_size_left=64, window_size_right=0,
                        mx_mirror=MxMirrorBwd(r["stats"], Mxfp8Mirror.QK))


def test_hd128_band_split_across_pair_uses_1cta():
    """A causal offset of 128 (mod 256) puts the two KV tiles of a 2CTA pair on different sides
    of a band edge: the selection falls back to the 1CTA kernel; other offsets keep 2CTA."""
    inputs = dict(arch=100, device_index=None, block_sparse_tensors=None, V_in_regs=False,
                  causal=True, local=False, deterministic=False, dtype="torch.bfloat16",
                  head_dim=128, head_dim_v=128, num_heads=2, num_heads_kv=2, batch_size=1,
                  has_cu_seqlens=False, mxfp8_mirror=Mxfp8Mirror.PV)
    for sq, sk, two_cta in ((512, 512, True), (384, 640, True), (384, 512, False)):
        cfg = select_bwd_config(_get_bwd_inputs(
            **inputs, total_q=sq, total_k=sk, max_seqlen_q=sq, max_seqlen_k=sk))
        assert cfg.use_2cta_instrs == two_cta, (sq, sk)
    qk = select_bwd_config(_get_bwd_inputs(
        **{**inputs, "mxfp8_mirror": Mxfp8Mirror.QK}, total_q=384, total_k=512,
        max_seqlen_q=384, max_seqlen_k=512))
    assert qk.use_2cta_instrs
