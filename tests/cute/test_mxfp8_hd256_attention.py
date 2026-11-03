"""MXFP8 hdim-256 backward (NOTE [MXFP8 hd256 backward]): gradients against the FP64 mirrored reference,
determinism, causal leakage."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.utils.blockscaled_layout")
from flash_attn.cute import blockscaled_utils as bsu  # noqa: E402
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
    if causal:  # the forward's BF16 band reads the original V
        band = (j >= 256 * (i // 256)) & (j <= i)
        dp = torch.where(band, dout @ v.transpose(-1, -2), dp)
    ds = p / L * (dp - (dout * out).sum(-1, keepdim=True))
    grads = scale * ds @ kh, scale * ds.transpose(-1, -2) @ qh, (p_hat / L).transpose(-1, -2) @ dout
    return [g.transpose(1, 2) for g in grads]


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
