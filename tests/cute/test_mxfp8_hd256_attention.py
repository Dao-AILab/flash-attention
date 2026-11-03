"""MXFP8 hdim-256 backward (NOTE [MXFP8 hd256 backward]) and its autograd entry point (NOTE [MXFP8 hd256
attention]): gradients against the FP64 mirrored reference, determinism, causal leakage, fused prep identity;
the MX dS mode (NOTE [MXFP8 hd256 dS for dQ]) against exact mode and an emulation of its dQ."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute import blockscaled_utils as bsu  # noqa: E402
from flash_attn.cute import mxfp8_hd256_attention as attn  # noqa: E402
from flash_attn.cute import mxfp8_hd256_prep as prep  # noqa: E402
from flash_attn.cute.config import mxds_slice_bytes  # noqa: E402
from flash_attn.cute.interface import (  # noqa: E402
    BlockScales,
    MxHd256BwdOperands,
    _flash_attn_bwd,
    _flash_attn_fwd,
)

DIM = 256
MAX_OFFSET = {"exact": 8.0, "fast": 4.0}


@pytest.fixture(autouse=True)
def requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU")


def inputs(b, s, h, seed=0):
    """BF16 Q / K / V / dO with different magnitudes across the 32-channel groups and across tokens, so every
    scale layout sees nonuniform scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    groups = torch.tensor([0.5, 2.0, 1.0, 0.25, 2.0, 0.125, 1.0, 0.5], device="cuda").repeat_interleave(32)
    out = []
    for i in range(4):
        x = torch.randn(b, s, h, DIM, generator=g, device="cuda")
        x *= groups if i % 2 == 0 else groups.flip(0)
        if i >= 2:  # V and dO: per-token magnitudes for the along-token scales
            x *= torch.exp2(torch.randint(-2, 3, (1, s, 1, 1), generator=g, device="cuda").float())
        out.append(x.bfloat16())
    return out


# ---------------------------------------------------------------- eager reference of the backward operands
def chunk_offset(r, g, kc_stride, mn_chunk_stride=0):
    """Byte of (row r, group g) in BlockScaledBasicChunk order."""
    return 16 * (r % 32) + 4 * ((r % 128) // 32) + mn_chunk_stride * (r // 128) + g % 4 + kc_stride * (g // 4)


def pair_tiles(sf_raw, rows):
    """(b, s, h, 8) UE8M0 scales along head_dim -> flat dK / dV kernel pair tiles (NOTE [MXFP8 hd256 backward]):
    rows=128 the queries' lane-halves tile, rows=64 the keys' duplicated tile."""
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


def eager_forward(q, k, v, causal, mode):
    """The MXFP8 forward on eagerly quantized operands: (out, lse, stats, dequantized (qh, kh, vh), payloads)."""
    q8, sfq, sq = bsu.quantize_to_fp8_with_sf(q)
    k8, sfk, sk = bsu.quantize_to_fp8_with_sf(k)
    v8, sfv_il, sv = bsu.quantize_v_to_fp8_with_sf(v)
    b, s, h, _ = q.shape
    stats = torch.empty(b, h, s, 2, device="cuda")
    scales = BlockScales(
        bsu.interleave_sf(sfq), bsu.interleave_sf(sfk), sfv_il, v if causal else None, pv_softmax=mode,
        stats_out=stats,
    )
    out, lse = _flash_attn_fwd(q8, k8, v8, causal=causal, return_lse=True, scales=scales)[:2]
    deq = (bsu.dequantize_fp8_with_sf(q8, sq), bsu.dequantize_fp8_with_sf(k8, sk),
           bsu.dequantize_v_fp8_with_sf(v8, sv))
    return out, lse, stats, deq, (q8, sfq, k8, sfk, sfv_il, v8)


def eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, dout, stats, mode, causal):
    """Eager MxHd256BwdOperands and BF16 K-hat (NOTE [MXFP8 hd256 backward])."""

    def along_d(name, x8, sf_raw, rows):
        return {name: x8, "sf" + name: bsu.interleave_sf(sf_raw).reshape(-1),
                "sf" + name + "_dkdv": pair_tiles(sf_raw, rows)}

    ops = {**along_d("q", q8, sfq, 128), **along_d("k", k8, sfk, 64)}
    v8, sfv, sv = bsu.quantize_to_fp8_with_sf(vh)
    if causal:  # the band residual R = MX(V - Vhat_d) as keys [s, 2s)
        r8, sfr, _ = bsu.quantize_to_fp8_with_sf(v.float() - bsu.dequantize_fp8_with_sf(v8, sv))
        v8, sfv = torch.cat([v8, r8], dim=1), torch.cat([sfv, sfr], dim=1)
    ops.update(along_d("v", v8, sfv, 64))
    do2 = dout.float() * (256.0 / stats[..., 1]).transpose(1, 2).unsqueeze(-1)
    do8, sfdo, sdo = bsu.quantize_to_fp8_with_sf(do2)
    ops.update(along_d("do", do8, sfdo, 128))
    ops["do_band"] = (do2 * (1.0 / 256.0)).bfloat16() if causal else None  # dO / L
    ops["qt"], ops["sfqt"], _ = bsu.quantize_v_to_fp8_with_sf(qh)
    ops["dot"], ops["sfdot"], _ = bsu.quantize_v_to_fp8_with_sf(do2)
    ops["sfqt"], ops["sfdot"] = ops["sfqt"].reshape(-1), ops["sfdot"].reshape(-1)
    ops["dpsum"] = (bsu.dequantize_fp8_with_sf(do8, sdo) * out.float()).sum(-1).transpose(1, 2).contiguous()
    c = float(torch.tensor(DIM**-0.5 * math.log2(math.e), dtype=torch.float32))
    row_max = stats[..., 0] * c
    ops["lse_log2"] = (row_max - MAX_OFFSET[mode] + 8.0).contiguous()
    ops["lse_log2_dkdv"] = (row_max - MAX_OFFSET[mode]).contiguous()
    return MxHd256BwdOperands(**ops), kh.bfloat16()


def eager_backward(q, k, v, dout, causal, mode):
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, causal, mode)
    ops, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, dout, stats, mode, causal)
    grads = _flash_attn_bwd(k_hat, k_hat, k_hat, out, dout, lse, causal=causal, mx_hd256=ops)[:3]
    return grads, (out, lse, stats, qh, kh, vh)


def mirrored_reference(qh, kh, vh, v, out, dout, stats, mode, causal):
    """FP64 dQ / dK / dV of the NOTE's formulas, from the forward's dequantized operands and stats_out."""
    d64 = torch.float64
    scale = DIM**-0.5
    c = float(torch.tensor(scale * math.log2(math.e), dtype=torch.float32))
    qh, kh, vh, v, out, dout = (t.to(d64).transpose(1, 2) for t in (qh, kh, vh, v, out, dout))
    m = (stats[..., 0] * c).to(d64)[..., None]  # M, rounded to FP32 like the kernels
    L = stats[..., 1].to(d64)[..., None]
    s = qh @ kh.transpose(-1, -2)
    p = torch.exp2(s * c - (m - MAX_OFFSET[mode]))
    i, j = torch.arange(s.shape[-2], device="cuda")[:, None], torch.arange(s.shape[-1], device="cuda")[None, :]
    if causal:
        p = p.masked_fill(j > i, 0.0)
    p_hat = p.float().to(torch.float8_e4m3fn).to(d64)
    dp = dout @ vh.transpose(-1, -2)
    if causal:  # the forward's BF16 band: the original V, and P in BF16
        band = (j >= 256 * (i // 256)) & (j <= i)
        dp = torch.where(band, dout @ v.transpose(-1, -2), dp)
        do_band = (dout.float() * (256.0 / L.float()) * (1.0 / 256.0)).bfloat16().to(d64)
        p_band = p.float().bfloat16().to(d64)
        dv = (torch.where(band, 0.0, p_hat) / L).transpose(-1, -2) @ dout
        dv += torch.where(band, p_band, 0.0).transpose(-1, -2) @ do_band
    else:
        dv = (p_hat / L).transpose(-1, -2) @ dout
    ds = p / L * (dp - (dout * out).sum(-1, keepdim=True))
    grads = scale * ds @ kh, scale * ds.transpose(-1, -2) @ qh, dv
    return [g.transpose(1, 2) for g in grads]


def mxds_dq_emulation(qh, kh, vh, v, out, dout, stats, mode, causal):
    """dQ of the MX dS mode in FP64 from the kernels' operands: dS x 2^8 = P~ (dP'' - D'') with
    dO'' = 2^8 dO / L and V-hat (+ the band residual R) in MXFP8 along head_dim, in E4M3 with one
    ceil-rule scale per 32 x 32 block (NOTE [MXFP8 hd256 dS for dQ]), times K-hat requantized along keys;
    BF16."""
    d64 = torch.float64
    c = float(torch.tensor(DIM**-0.5 * math.log2(math.e), dtype=torch.float32))

    def mx_d(x):  # MXFP8 along head_dim, dequantized
        x8, _, sx = bsu.quantize_to_fp8_with_sf(x)
        return bsu.dequantize_fp8_with_sf(x8, sx)

    kt8, _, skt = bsu.quantize_v_to_fp8_with_sf(kh)  # K-hat along keys
    do2 = mx_d(dout.float() * (256.0 / stats[..., 1]).transpose(1, 2).unsqueeze(-1))
    vd = mx_d(vh)
    qh, kh, vd, do2, out = (t.to(d64).transpose(1, 2) for t in (qh, kh, vd, do2, out))
    m = (stats[..., 0] * c).to(d64)[..., None]
    p = torch.exp2(qh @ kh.transpose(-1, -2) * c - (m - MAX_OFFSET[mode]))
    dp = do2 @ vd.transpose(-1, -2)
    i, j = torch.arange(p.shape[-2], device="cuda")[:, None], torch.arange(p.shape[-1], device="cuda")[None, :]
    if causal:
        p = p.masked_fill(j > i, 0.0)
        r = mx_d(v.float() - vd.transpose(1, 2).float()).to(d64).transpose(1, 2)
        band = (j >= 256 * (i // 256)) & (j <= i)
        dp = torch.where(band, dp + do2 @ r.transpose(-1, -2), dp)
    ds = p * (dp - (do2 * out).sum(-1, keepdim=True))  # dS x 2^8
    b, h, sq, sk = ds.shape
    blocks = ds.float().reshape(b, h, sq // 32, 32, sk // 32, 32)
    amax = blocks.abs().amax(dim=(3, 5), keepdim=True)
    mantissa, exponent = torch.frexp(amax)
    exponent = torch.where(amax == 0, 0, (exponent - 9 + (mantissa > 0.875)).clamp(-127, 127))
    scale = torch.exp2(exponent.float())
    ds_hat = ((blocks / scale).to(torch.float8_e4m3fn).float() * scale / 256).reshape(b, h, sq, sk)
    kt = bsu.dequantize_v_fp8_with_sf(kt8, skt).transpose(1, 2)
    return (DIM**-0.5 * ds_hat.double() @ kt.double()).transpose(1, 2).bfloat16()


def with_mxds(ops, workspace_bytes=None):
    """The MX dS mode's operands from exact-mode MxHd256BwdOperands."""
    return ops._replace(sfdo=None, mxds=prep.kt_operands(ops.k, ops.sfk, DIM**-0.5, workspace_bytes))


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# (1, 544, 2): partial 128-query tiles and scale chunks (non-causal); (2, 2048, 3): batch and head strides
@pytest.mark.parametrize("shape,causal", [
    ((1, 1024, 2), False), ((1, 1024, 2), True), ((1, 544, 2), False), ((2, 2048, 3), False), ((2, 2048, 3), True),
])
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_backward_matches_mirrored_reference(shape, causal, mode):
    q, k, v, do = inputs(*shape, seed=2)
    grads, (out, _, stats, qh, kh, vh) = eager_backward(q, k, v, do, causal, mode)
    for g, r in zip(grads, mirrored_reference(qh, kh, vh, v, out, do, stats, mode, causal)):
        assert torch.isfinite(g.float()).all()
        # dO'' and dS quantization (MXFP8) and the BF16 outputs: ~0.03 on these inputs
        assert rel(g, r) < 0.06


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_causal_band_dv_reads_bf16_p(mode):
    # seqlen 256: every causal dK / dV step is a band step (NOTE [MXFP8 hd256 band dV]), so dV is
    # BF16 P~ against BF16 dO / L and is exact up to that rounding; e4m3 P gives ~0.03 here
    q, k, v, do = inputs(1, 256, 2, seed=7)
    grads, (out, _, stats, qh, kh, vh) = eager_backward(q, k, v, do, True, mode)
    dv_ref = mirrored_reference(qh, kh, vh, v, out, do, stats, mode, True)[2]
    assert rel(grads[2], dv_ref) < 5e-3


@pytest.mark.parametrize("causal", [False, True])
def test_backward_is_deterministic(causal):
    # several waves of 2CTA tiles, so the PDL overlap and the dQ kernel's second compute warp group (causal, or
    # sm_107) run
    q, k, v, do = inputs(2, 4096, 16, seed=1)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, causal, "exact")
    ops, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, "exact", causal)

    def backward():
        return _flash_attn_bwd(k_hat, k_hat, k_hat, out, do, lse, causal=causal, mx_hd256=ops)[:3]

    first = backward()
    for _ in range(8):
        assert all(torch.equal(a, b) for a, b in zip(first, backward()))


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_causal_dq_does_not_see_future_keys(mode):
    q, k, v, do = inputs(1, 2048, 2, seed=4)
    t0 = 1101
    (dq, _, _), (out, lse, _, _, _, _) = eager_backward(q, k, v, do, True, mode)
    k2, v2 = k.clone(), v.clone()
    k2[:, t0:] *= 16
    v2[:, t0:] *= 16
    (dq2, _, _), (out2, lse2, _, _, _, _) = eager_backward(q, k2, v2, do, True, mode)
    assert torch.equal(out2[:, :t0], out[:, :t0]) and torch.equal(lse2[..., :t0], lse[..., :t0])
    assert torch.equal(dq2[:, :t0], dq[:, :t0])


@pytest.mark.parametrize("shape,causal", [((1, 1024, 2), False), ((1, 1024, 2), True), ((1, 544, 2), False)])
def test_fused_prep_matches_eager(shape, causal):
    q, k, v, do = inputs(*shape, seed=5)
    out, lse, saved = attn.mxfp8_hd256_forward(q, k, v, causal=causal)
    out_e, lse_e, stats, (qh, kh, vh), (q8, sfq, k8, sfk, sfv_il, v8) = eager_forward(q, k, v, causal, "exact")
    assert torch.equal(out, out_e) and torch.equal(lse, lse_e) and torch.equal(saved.stats, stats)
    fwd = prep.quantize_forward(q, k, v, causal)
    for a, b in ((fwd.v8, v8), (fwd.sfv, sfv_il)):
        assert torch.equal(a.view(torch.uint8).reshape(-1), b.view(torch.uint8).reshape(-1))
    fused = prep.backward_operands(fwd.bwd, out, do, stats, saved.softmax_scale, "exact")
    eager, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, "exact", causal)
    assert torch.equal(fwd.k_hat, k_hat)
    for name in MxHd256BwdOperands._fields:
        a, b = getattr(fused, name), getattr(eager, name)
        assert (a is None) == (b is None), name
        if a is None:
            continue
        if name == "dpsum":
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)), name


@pytest.mark.parametrize("causal", [False, True])
def test_band_dout_required_exactly_when_causal(causal):
    q, k, v, do = inputs(1, 256, 2, seed=8)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, causal, "exact")
    ops, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, "exact", causal)
    ops = ops._replace(do_band=None if causal else do)
    with pytest.raises(ValueError, match="do_band"):
        _flash_attn_bwd(k_hat, k_hat, k_hat, out, do, lse, causal=causal, mx_hd256=ops)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_autograd_gradients(causal, mode):
    q, k, v, do = inputs(1, 1024, 2, seed=6)
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    attn.mxfp8_hd256_attn_func(*leaves, causal=causal, pv_softmax=mode).backward(do)
    _, _, saved = attn.mxfp8_hd256_forward(q, k, v, causal=causal, pv_softmax=mode)
    halves = attn.mxfp8_hd256_backward(saved, do)
    eager, _ = eager_backward(q, k, v, do, causal, mode)
    ref = [t.float().transpose(1, 2).requires_grad_() for t in (q, k, v)]
    F.scaled_dot_product_attention(*ref, is_causal=causal).backward(do.float().transpose(1, 2))
    for t, g, e, r in zip(leaves, halves, eager, ref):
        assert torch.equal(t.grad, g)
        # the fused prep's operands are byte-identical to the eager ones except D'' (FP32 sum order)
        assert rel(g, e) < 1e-2
        assert rel(g, r.grad.transpose(1, 2)) < 0.2  # MXFP8 gradients vs FP32 attention: ~0.1


# ---------------------------------------------------------------- MX dS mode (bwd_mode="mxds")
@pytest.mark.parametrize("shape,causal", [
    ((1, 1024, 2), False), ((1, 1024, 2), True), ((2, 2048, 3), False), ((2, 2048, 3), True),
])
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_mxds_matches_exact_dkdv_and_emulated_dq(shape, causal, mode):
    q, k, v, do = inputs(*shape, seed=2)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, causal, mode)
    ops, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, mode, causal)
    exact = _flash_attn_bwd(k_hat, k_hat, k_hat, out, do, lse, causal=causal, mx_hd256=ops)[:3]
    dq, dk, dv = _flash_attn_bwd(k_hat, k_hat, k_hat, out, do, lse, causal=causal, mx_hd256=with_mxds(ops))[:3]
    # dK / dV come from the same dK / dV kernel; causal includes the band dV
    assert torch.equal(dk, exact[1]) and torch.equal(dv, exact[2])
    ref = mirrored_reference(qh, kh, vh, v, out, do, stats, mode, causal)
    assert torch.isfinite(dq.float()).all()
    # E4M3 rounding of dS computed in FP32 (kernel) vs FP64 (emulation)
    assert rel(dq, mxds_dq_emulation(qh, kh, vh, v, out, do, stats, mode, causal)) < 2e-3
    # the second dS quantization: ~0.038 vs exact mode's ~0.027 on these inputs
    assert rel(dq, ref[0]) < 0.06


def test_mxds_k_requantization_matches_eager():
    q, k, v, do = inputs(2, 512, 3, seed=9)
    _, _, _, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, False, "exact")
    kt = prep.kt_operands(k8, bsu.interleave_sf(sfk).reshape(-1), DIM**-0.5)
    kt8, sfkt, _ = bsu.quantize_v_to_fp8_with_sf(kh)
    assert kt.alpha == 1.0  # 256^-0.5 = 2^-4 rides in the scales
    assert torch.equal(kt.kt.view(torch.uint8), kt8.view(torch.uint8))
    assert torch.equal(kt.sfkt.view(torch.uint8), sfkt.reshape(-1).view(torch.uint8) - 4)


@pytest.mark.parametrize("causal", [False, True])
def test_mxds_is_deterministic_and_chunk_invariant(causal):
    # 32 (batch, head) slices: the default budget runs one chunk spanning both batches, a budget of 4 (1)
    # slices runs 8 (32) chunks within a batch
    b, s, h = 2, 4096, 16
    q, k, v, do = inputs(b, s, h, seed=1)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, causal, "exact")
    ops, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, "exact", causal)

    def backward(workspace_bytes=None):
        mx = with_mxds(ops, workspace_bytes)
        return _flash_attn_bwd(k_hat, k_hat, k_hat, out, do, lse, causal=causal, mx_hd256=mx)[:3]

    first = backward()
    for _ in range(4):
        assert all(torch.equal(a, b) for a, b in zip(first, backward()))
    for slices_per_chunk in (4, 1):
        chunked = backward(slices_per_chunk * mxds_slice_bytes(s, s))
        assert all(torch.equal(a, b) for a, b in zip(first, chunked))


@pytest.mark.parametrize("t0", [1088, 1101])
def test_mxds_causal_dq_rows_before_the_key_group_do_not_see_future_keys(t0):
    # K-hat's scales along keys are shared by 32-key groups: keys >= t0 may change the dQ rows in
    # [32 floor(t0 / 32), t0) (accuracy, NOTE [MXFP8 hd256 dS for dQ]); rows below are bitwise unaffected
    q, k, v, do = inputs(1, 2048, 2, seed=4)
    k2, v2 = k.clone(), v.clone()
    k2[:, t0:] *= 16
    v2[:, t0:] *= 16
    dqs = []
    for kk, vv in ((k, v), (k2, v2)):
        leaves = [t.clone().requires_grad_() for t in (q, kk, vv)]
        attn.mxfp8_hd256_attn_func(*leaves, causal=True, bwd_mode="mxds").backward(do)
        dqs.append(leaves[0].grad)
    t_safe = 32 * (t0 // 32)
    assert torch.equal(dqs[0][:, :t_safe], dqs[1][:, :t_safe])


@pytest.mark.parametrize("causal", [False, True])
def test_mxds_autograd(causal):
    q, k, v, do = inputs(1, 1024, 2, seed=6)
    grads = {}
    for bwd_mode in ("exact", "mxds"):
        leaves = [t.clone().requires_grad_() for t in (q, k, v)]
        attn.mxfp8_hd256_attn_func(*leaves, causal=causal, bwd_mode=bwd_mode).backward(do)
        grads[bwd_mode] = [t.grad for t in leaves]
    _, _, saved = attn.mxfp8_hd256_forward(q, k, v, causal=causal, bwd_mode="mxds")
    halves = attn.mxfp8_hd256_backward(saved, do)
    assert all(torch.equal(a, b) for a, b in zip(grads["mxds"], halves))
    assert torch.equal(grads["mxds"][1], grads["exact"][1]) and torch.equal(grads["mxds"][2], grads["exact"][2])
    assert rel(grads["mxds"][0], grads["exact"][0]) < 0.06
    with pytest.raises(NotImplementedError, match="256"):
        attn.mxfp8_hd256_attn_func(*(t[:, :544] for t in (q, k, v)), bwd_mode="mxds")


@pytest.mark.parametrize("causal", [False, True])
def test_mxds_non_power_of_two_softmax_scale(causal):
    # softmax_scale 0.1 = 1.6 * 2^-4: 2^-4 rides in the K-hat scales, 1.6 is the GEMM epilogue's multiplier
    q, k, v, do = inputs(1, 1024, 2, seed=10)
    grads = {}
    for bwd_mode in ("exact", "mxds"):
        leaves = [t.clone().requires_grad_() for t in (q, k, v)]
        attn.mxfp8_hd256_attn_func(*leaves, causal=causal, softmax_scale=0.1, bwd_mode=bwd_mode).backward(do)
        grads[bwd_mode] = [t.grad for t in leaves]
    assert torch.equal(grads["mxds"][1], grads["exact"][1]) and torch.equal(grads["mxds"][2], grads["exact"][2])
    # the two dQ errors vs the oracle (~0.027, ~0.038) are mostly independent: ~0.03 apart
    assert rel(grads["mxds"][0], grads["exact"][0]) < 0.05


def test_mxds_operands_are_validated():
    q, k, v, do = inputs(1, 544, 2, seed=11)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, False, "exact")
    ops, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, "exact", False)
    with pytest.raises(ValueError, match="256"):
        _flash_attn_bwd(k_hat, k_hat, k_hat, out, do, lse, mx_hd256=with_mxds(ops))
    q, k, v, do = inputs(1, 512, 2, seed=11)
    out, lse, stats, (qh, kh, vh), (q8, sfq, k8, sfk, _, _) = eager_forward(q, k, v, False, "exact")
    ops, k_hat = eager_operands(q8, sfq, k8, sfk, qh, kh, vh, v, out, do, stats, "exact", False)
    mx = with_mxds(ops)
    bad_cases = (
        (dict(sfkt=mx.mxds.sfkt[:-16]), "mxds.sfkt"),
        (dict(workspace_bytes=0), "workspace_bytes"),
        (dict(workspace_bytes=mxds_slice_bytes(512, 512) - 1), "needs at least"),
    )
    for bad, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            _flash_attn_bwd(k_hat, k_hat, k_hat, out, do, lse, mx_hd256=mx._replace(mxds=mx.mxds._replace(**bad)))
