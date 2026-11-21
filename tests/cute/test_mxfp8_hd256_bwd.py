"""The MXFP8 hdim-256 backward on the general SM100 kernel (NOTE [MXFP8 hd256 backward] in
flash_attn/cute/flash_bwd_sm100.py) and its fused prep (flash_attn/cute/mxfp8_hd256_prep.py):
gradients against the kernel-faithful emulator and the FP64 mirrored oracle, the causal band,
determinism, causal leakage, prep identity, and operand validation."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute import blockscaled_utils as bsu  # noqa: E402
from flash_attn.cute import mxfp8_hd256_prep as prep  # noqa: E402
from flash_attn.cute.interface import (  # noqa: E402
    BlockScales,
    MxHd256BwdOperands,
    _flash_attn_bwd,
    _flash_attn_fwd,
)

from mxfp8_mirror_reference import (  # noqa: E402
    emulate_mx_backward,
    oracle_per_head,
    per_head,
    rel_l2,
)

DIM = 256
P_OFFSET = 8  # the native PV forward's P exponent offset (BlockScales.stats_out)
NORMALIZERS = ["rounded_p", "fp32"]


@pytest.fixture(autouse=True)
def requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("requires SM100 / SM110")


def inputs(b, s, h, seed=0):
    """BF16 Q / K / V / dO with different magnitudes across the 32-channel groups and across tokens,
    so every scale layout sees nonuniform scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    groups = torch.tensor(
        [0.5, 2.0, 1.0, 0.25, 2.0, 0.125, 1.0, 0.5], device="cuda"
    ).repeat_interleave(32)
    out = []
    for i in range(4):
        x = torch.randn(b, s, h, DIM, generator=g, device="cuda")
        x *= groups if i % 2 == 0 else groups.flip(0)
        if i >= 2:  # V and dO: per-token magnitudes for the along-token scales
            x *= torch.exp2(torch.randint(-2, 3, (1, s, 1, 1), generator=g, device="cuda").float())
        out.append(x.bfloat16())
    return out


# ------------------------------------------------ eager reference of the backward operands
def chunk_offset(r, g, kc_stride, mn_chunk_stride=0):
    """Byte of (row r, group g) in BlockScaledBasicChunk order."""
    return (
        16 * (r % 32)
        + 4 * ((r % 128) // 32)
        + mn_chunk_stride * (r // 128)
        + g % 4
        + kc_stride * (g // 4)
    )


def pair_tiles(sf_raw, rows):
    """(b, s, h, 8) UE8M0 scales along head_dim -> flat backward pair tiles (NOTE [MXFP8 hd256
    backward]): rows=128 the queries' lane-halves tile, rows=64 the keys' duplicated tile."""
    dev = sf_raw.device
    if rows == 128:
        n = torch.arange(128, device=dev)[:, None].expand(128, 8)
        g = torch.arange(8, device=dev)[None, :].expand(128, 8)
        dst, src = 16 * (n % 32) + 4 * ((n % 64) // 32) + 512 * (n // 64) + g % 4 + 8 * (g // 4), n
    else:
        r = torch.arange(128, device=dev)[:, None].expand(128, 8)
        g = torch.arange(8, device=dev)[None, :].expand(128, 8)
        dst, src = chunk_offset(r, g, kc_stride=512), r % 64
    b, s, h, _ = sf_raw.shape
    x = sf_raw.view(torch.uint8).permute(0, 2, 1, 3)
    x = F.pad(x, (0, 0, 0, math.ceil(s / rows) * rows - s)).reshape(b, h, -1, rows, 8)
    out = torch.empty(*x.shape[:3], 1024, dtype=torch.uint8, device=dev)
    out[..., dst.flatten()] = x[..., src, g].reshape(*x.shape[:3], -1)
    return out.reshape(-1).view(torch.float8_e8m0fnu)


def eager_forward(q, k, v, causal, pv_normalizer):
    """The MXFP8 forward on eagerly quantized operands: (out, lse, stats, dequantized (qh, kh, vh),
    payloads)."""
    q8, sfq, sq = bsu.quantize_to_fp8_with_sf(q)
    k8, sfk, sk = bsu.quantize_to_fp8_with_sf(k)
    v8, sfv_il, sv = bsu.quantize_v_to_fp8_with_sf(v)
    b, s, h, _ = q.shape
    stats = torch.empty(b, h, s, 2, device="cuda")
    scales = BlockScales(
        bsu.interleave_sf(sfq),
        bsu.interleave_sf(sfk),
        sfv_il,
        v if causal else None,
        pv_normalizer=pv_normalizer,
        stats_out=stats,
    )
    out, lse = _flash_attn_fwd(q8, k8, v8, causal=causal, return_lse=True, scales=scales)[:2]
    deq = (
        bsu.dequantize_fp8_with_sf(q8, sq),
        bsu.dequantize_fp8_with_sf(k8, sk),
        bsu.dequantize_v_fp8_with_sf(v8, sv),
    )
    return out, lse, stats, deq, (q8, sfq, k8, sfk, sfv_il, v8)


def eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, dout, stats, causal,
                   softmax_scale=DIM**-0.5):
    """Eager MxHd256BwdOperands (NOTE [MXFP8 hd256 backward])."""
    ops = dict(q=q8, sfq=pair_tiles(sfq, 128), k=k8, sfk=pair_tiles(sfk, 64))
    v8, sfv, sv = bsu.quantize_to_fp8_with_sf(vh)
    if causal:  # the band residual R = MX(V - Vhat_d) as keys [s, 2s)
        r8, sfr, _ = bsu.quantize_to_fp8_with_sf(v.float() - bsu.dequantize_fp8_with_sf(v8, sv))
        v8, sfv = torch.cat([v8, r8], dim=1), torch.cat([sfv, sfr], dim=1)
    ops.update(v=v8, sfv=pair_tiles(sfv, 64))
    do2 = dout.float() * (256.0 / stats[..., 1]).transpose(1, 2).unsqueeze(-1)
    do8, sfdo, sdo = bsu.quantize_to_fp8_with_sf(do2)
    ops.update(do=do8, sfdo=pair_tiles(sfdo, 128))
    ops["do_band"] = (do2 * (1.0 / 256.0)).bfloat16() if causal else None  # dO / L
    ops["qt"], ops["sfqt"], _ = bsu.quantize_v_to_fp8_with_sf(qh)
    ops["dot"], ops["sfdot"], _ = bsu.quantize_v_to_fp8_with_sf(do2)
    ops["sfqt"], ops["sfdot"] = ops["sfqt"].reshape(-1), ops["sfdot"].reshape(-1)
    ops["k_hat"] = kh.bfloat16()
    ops["dpsum"] = (
        (bsu.dequantize_fp8_with_sf(do8, sdo) * out.float()).sum(-1).transpose(1, 2).contiguous()
    )
    # the kernels' fp32(fp32(softmax_scale) * fp32(log2(e)))
    c = torch.tensor(softmax_scale) * torch.tensor(math.log2(math.e))
    ops["lse_log2"] = (stats[..., 0] * c.item() - P_OFFSET).contiguous()
    return MxHd256BwdOperands(**ops)


def mx_backward(ops, out, dout, lse, causal, deterministic=False):
    del lse  # the operands carry the LSE slot
    return _flash_attn_bwd(
        out,
        ops.k_hat,
        out,
        out,
        dout,
        None,
        causal=causal,
        deterministic=deterministic,
        mx_hd256=ops,
    )[:3]


def fused_backward(q, k, v, dout, causal, pv_normalizer, deterministic=False):
    """The fused prep's forward and backward: (grads, (out, lse, stats, Q-hat, K-hat, V-hat))."""
    fwd = prep.quantize_forward(q, k, v, causal)
    b, s, h, _ = q.shape
    stats = torch.empty(b, h, s, 2, device="cuda")
    scales = BlockScales(
        fwd.sfq, fwd.sfk, fwd.sfv, v if causal else None, pv_normalizer=pv_normalizer,
        stats_out=stats,
    )
    out, lse = _flash_attn_fwd(fwd.q8, fwd.k8, fwd.v8, causal=causal, return_lse=True,
                               scales=scales)[:2]
    ops = prep.backward_operands(fwd.bwd, out, dout, stats, DIM**-0.5)
    grads = mx_backward(ops, out, dout, lse, causal, deterministic)
    return grads, (out, lse, stats)


def eager_backward(q, k, v, dout, causal, pv_normalizer, deterministic=False):
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(
        q, k, v, causal, pv_normalizer
    )
    ops = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, dout, stats, causal)
    grads = mx_backward(ops, out, dout, lse, causal, deterministic)
    return grads, (out, lse, stats, qh, kh, vh)


# (1, 544, 2): partial 128-query tiles and scale chunks (non-causal); (2, 2048, 3): batch and
# head strides
@pytest.mark.parametrize(
    "shape,causal",
    [
        ((1, 1024, 2), False),
        ((1, 1024, 2), True),
        ((1, 544, 2), False),
        ((2, 2048, 3), False),
        ((2, 2048, 3), True),
    ],
)
@pytest.mark.parametrize("pv_normalizer", NORMALIZERS)
def test_backward_matches_emulator_and_oracle(shape, causal, pv_normalizer):
    """The kernel matches its kernel-faithful emulator (emulate_mx_backward) to accumulation-order
    noise, and the FP64 mirrored oracle (the derivative of the online forward) to the emulator's
    own distance from it. An E4M3 knife edge of P or dS'' can round the other way under another
    S accumulation order, so the emulator bound is empirical: 2x above the largest error
    measured on B200."""
    q, k, v, do = inputs(*shape, seed=2)
    grads, (out, _, stats, qh, kh, vh) = eager_backward(q, k, v, do, causal, pv_normalizer)
    refs = (qh, kh, vh, v)
    kw = dict(softmax_scale=DIM**-0.5, causal=causal, native_pv=True)
    emu = per_head(emulate_mx_backward, refs, out, do, stats, **kw)
    orc = per_head(oracle_per_head, refs, out, do, stats, **kw, pv_normalizer=pv_normalizer)
    errors = {
        name: (rel_l2(g, e.to(torch.bfloat16)), rel_l2(g, o), rel_l2(e, o))
        for name, g, e, o in zip("qkv", grads, emu, orc)
    }
    for name, g in zip("qkv", grads):
        assert torch.isfinite(g.float()).all(), f"d{name} is not finite"
    for name, (vs_emu, vs_oracle, emu_vs_oracle) in errors.items():
        assert vs_emu <= 1e-3, f"d{name}: kernel vs emulator {vs_emu:.2e} ({errors})"
        # 2e-3: the BF16 output rounding the FP64 oracle does not have
        assert vs_oracle <= 1.1 * emu_vs_oracle + 2e-3, f"d{name}: vs oracle ({errors})"


@pytest.mark.parametrize("pv_normalizer", NORMALIZERS)
def test_causal_band_dv_reads_bf16_p(pv_normalizer):
    # seqlen 256: every causal step is a band step (NOTE [MXFP8 hd256 backward]), so dV is BF16
    # P~ against BF16 dO / L; an E4M3 P would put dV ~0.03 away from the BF16-P emulator
    q, k, v, do = inputs(1, 256, 2, seed=7)
    grads, (out, _, stats, qh, kh, vh) = eager_backward(q, k, v, do, True, pv_normalizer)
    kw = dict(softmax_scale=DIM**-0.5, causal=True, native_pv=True)
    dv_emu = per_head(emulate_mx_backward, (qh, kh, vh, v), out, do, stats, **kw)[2]
    assert rel_l2(grads[2], dv_emu.to(torch.bfloat16)) < 5e-3


@pytest.mark.parametrize("causal", [False, True])
def test_backward_is_deterministic(causal):
    # several waves of 2CTA tiles. deterministic=True orders the fp32 dQ adds; dK / dV are stored
    # directly (MHA), so they repeat bitwise in either mode
    q, k, v, do = inputs(2, 4096, 16, seed=1)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(
        q, k, v, causal, "rounded_p"
    )
    ops = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, causal)
    first = mx_backward(ops, out, do, lse, causal, deterministic=True)
    for _ in range(8):
        again = mx_backward(ops, out, do, lse, causal, deterministic=True)
        assert all(torch.equal(a, b) for a, b in zip(first, again))
    nondet = mx_backward(ops, out, do, lse, causal)
    assert torch.equal(nondet[1], first[1]) and torch.equal(nondet[2], first[2])
    assert rel_l2(nondet[0], first[0]) < 1e-3


@pytest.mark.parametrize("pv_normalizer", NORMALIZERS)
def test_causal_dq_does_not_see_future_keys(pv_normalizer):
    q, k, v, do = inputs(1, 2048, 2, seed=4)
    # deterministic=True: the fp32 dQ adds in a fixed order, so the +0 adds of keys >= t0 are
    # bitwise inert
    t0 = 1101
    (dq, _, _), (out, lse, *_) = eager_backward(q, k, v, do, True, pv_normalizer, True)
    k2, v2, do2 = k.clone(), v.clone(), do.clone()
    k2[:, t0:] *= 16
    v2[:, t0:] *= 16
    do2[:, t0:] = torch.randn_like(do2[:, t0:])
    (dq2, _, _), (out2, lse2, *_) = eager_backward(q, k2, v2, do2, True, pv_normalizer, True)
    assert torch.equal(out2[:, :t0], out[:, :t0]) and torch.equal(lse2[..., :t0], lse[..., :t0])
    assert torch.equal(dq2[:, :t0], dq[:, :t0])


@pytest.mark.parametrize(
    "shape,causal", [((1, 1024, 2), False), ((1, 1024, 2), True), ((1, 544, 2), False)]
)
@pytest.mark.parametrize("softmax_scale", [DIM**-0.5, 0.07])
def test_fused_prep_matches_eager(shape, causal, softmax_scale):
    q, k, v, do = inputs(*shape, seed=5)
    fwd = prep.quantize_forward(q, k, v, causal)
    out_e, lse_e, stats, (qh, kh, vh), (q8, sfq, k8, sfk, sfv_il, v8) = eager_forward(
        q, k, v, causal, "rounded_p"
    )
    for a, b in ((fwd.q8, q8), (fwd.k8, k8), (fwd.v8, v8), (fwd.sfv, sfv_il),
                 (fwd.sfq, bsu.interleave_sf(sfq)), (fwd.sfk, bsu.interleave_sf(sfk))):
        assert torch.equal(a.view(torch.uint8).reshape(-1), b.view(torch.uint8).reshape(-1))
    fused = prep.backward_operands(fwd.bwd, out_e, do, stats, softmax_scale)
    eager = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out_e, do, stats, causal,
                           softmax_scale)
    for name in MxHd256BwdOperands._fields:
        a, b = getattr(fused, name), getattr(eager, name)
        assert (a is None) == (b is None), name
        if a is None:
            continue
        if name == "dpsum":
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)), (
                name
            )
    if softmax_scale != DIM**-0.5:
        return  # the forward above ran at the default scale
    # the fused operands give the eager gradients up to D'''s FP32 summation order
    grads, _ = fused_backward(q, k, v, do, causal, "rounded_p")
    eager_grads = mx_backward(eager, out_e, do, lse_e, causal)
    for g, e in zip(grads, eager_grads):
        assert rel_l2(g, e) < 1e-3


@pytest.mark.parametrize("causal", [False, True])
def test_band_dout_required_exactly_when_causal(causal):
    q, k, v, do = inputs(1, 256, 2, seed=8)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(
        q, k, v, causal, "rounded_p"
    )
    ops = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, causal)
    ops = ops._replace(do_band=None if causal else do)
    with pytest.raises(ValueError, match="do_band"):
        mx_backward(ops, out, do, lse, causal)


def test_operands_are_validated():
    q, k, v, do = inputs(1, 512, 2, seed=8)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(
        q, k, v, False, "rounded_p"
    )
    ops = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, False)
    with pytest.raises(ValueError, match="sfdot"):
        mx_backward(ops._replace(sfdot=ops.sfdot[:-16]), out, do, lse, False)
    with pytest.raises(ValueError, match="dlse"):
        _flash_attn_bwd(out, ops.k_hat, out, out, do, lse, dlse=torch.zeros_like(lse), mx_hd256=ops)
    with pytest.raises(ValueError, match="MHA only"):
        half = (out[:, :, :1],) * 2
        _flash_attn_bwd(out, *half, out, do, lse, mx_hd256=ops)
