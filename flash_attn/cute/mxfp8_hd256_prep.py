"""Fused Triton operand prep for the MXFP8 hdim-256 backward (NOTE [MXFP8 hd256 backward] in
flash_bwd_sm100.py).

NOTE [MXFP8 hd256 prep]
One pass per tensor instead of chains of eager FP32 torch ops. Each kernel writes its scales
straight into every layout a consumer reads: the forward's interleave_sf buffers and the
backward's 1024 B pair tiles (NOTE [MXFP8 hd256 backward] in flash_bwd_sm100.py).
  forward (`quantize_forward`), one pass per input:
    Q -> E4M3 + UE8M0 per 32 along head_dim, and Q_T = Q-hat requantized along queries (dK's B
         operand);
    K -> E4M3 + UE8M0 per 32 along head_dim, and K-hat in BF16 (dQ = dS K);
    V -> E4M3 + UE8M0 per 32 keys (the forward's PV), and V-hat requantized along head_dim (dP's
         A operand); causal adds the band residual R = MX(V - rq(V-hat)) as keys [s, 2s) of the
         same buffers.
  backward (`backward_operands`), one pass over dO: dO'' = 2^8 dO / L along head_dim and along
    queries, D'' = rowsum(dequant(dO''_d) * O), and the LSE slot M - off; causal adds dO / L in
    BF16 (the band's dV operand).
For finite inputs, payloads and scales are byte-identical to blockscaled_utils' eager quantizers
(same ceil scale rule, exact power-of-two scaling, round-to-nearest-even E4M3, no FMA contraction
where the eager path rounds twice); D'' is an FP32 sum in a different order.
"""

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flash_attn.cute.config import Mxfp8Mirror
from flash_attn.cute.interface import MxHd256BwdOperands
from flash_attn.cute.mxfp8_quant import exp2_int, interleaved_sf_offset, mx_exponent, to_e4m3


@triton.jit
def _quant_d(x, ROWS: tl.constexpr, COLS: tl.constexpr = 256):
    """x (ROWS, COLS) FP32 -> (E4M3 (ROWS, COLS), exponents (ROWS, COLS / 32)) with groups of 32
    along columns."""
    g = tl.reshape(x, (ROWS, COLS // 32, 32))
    e = mx_exponent(tl.max(tl.abs(g), axis=2))
    q = to_e4m3(g, e[:, :, None])
    return tl.reshape(q, (ROWS, COLS)), e


@triton.jit
def _deq_d(x8, e, ROWS: tl.constexpr, COLS: tl.constexpr = 256):
    g = tl.reshape(x8.to(tl.float32), (ROWS, COLS // 32, 32)) * exp2_int(e)[:, :, None]
    return tl.reshape(g, (ROWS, COLS))


@triton.jit
def _store_sf_d(
    e,
    bh,
    t,
    n_tok,
    il_ptr,
    dk_ptr,
    DK: tl.constexpr,
    G0=0,
    NG: tl.constexpr = 8,
    IL: tl.constexpr = True,
):
    """Store along-d exponents e (ROWS, NG) of tokens t (ROWS,) (batch-head bh, n_tok tokens) and
    groups [G0, G0 + NG) into the interleave_sf buffer (if IL) and the backward's pair tiles (DK 1:
    the queries' lane-halves tile of 128, 2: the keys' duplicated tile of 64)."""
    g = G0 + tl.arange(0, NG)[None, :]
    byte = (e + 127).to(tl.uint8)
    tt = t[:, None]
    if IL:
        tl.store(il_ptr + interleaved_sf_offset(bh, tt, g, tl.cdiv(n_tok, 128), 2), byte)
    if DK == 1:
        n = tt % 128
        dst = 16 * (n % 32) + 4 * ((n % 64) // 32) + 512 * (n // 64) + g % 4 + 8 * (g // 4)
        tl.store(dk_ptr + (bh * tl.cdiv(n_tok, 128) + tt // 128) * 1024 + dst, byte)
    if DK == 2:
        base = (bh * tl.cdiv(n_tok, 64) + tt // 64) * 1024
        for hi in tl.static_range(2):
            r = tt % 64 + 64 * hi
            tl.store(
                dk_ptr + base + 16 * (r % 32) + 4 * ((r % 128) // 32) + g % 4 + 512 * (g // 4), byte
            )


@triton.jit
def _store_sf_t(e, bh, gk, n_tok, il_ptr, C0=0, NC: tl.constexpr = 256):
    """Store along-token exponents e (NC,) of token group gk and channels [C0, C0 + NC) into the
    interleave_sf buffer with MN = channels, K = token groups."""
    c = C0 + tl.arange(0, NC)
    byte = (e + 127).to(tl.uint8)
    tl.store(il_ptr + interleaved_sf_offset(bh, c, gk, 2, tl.cdiv(tl.cdiv(n_tok, 32), 4)), byte)


@triton.jit
def _q_kernel(x_ptr, x8_ptr, il_ptr, dk_ptr, qt8_ptr, qt_il_ptr, S, H):
    """Q tile (b, 32 queries, h): E4M3 + UE8M0 along head_dim (interleaved, and the backward's
    lane-halves tiles) and, from the same Q-hat, Q_T = rq(Q-hat) along queries (interleaved MN =
    channels)."""
    pid_g, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // H, pid_bh % H
    t = pid_g * 32 + tl.arange(0, 32)
    row = (b * S + t) * H + h
    offs = row[:, None] * 256 + tl.arange(0, 256)[None, :]
    q, e = _quant_d(tl.load(x_ptr + offs).to(tl.float32), 32)
    tl.store(x8_ptr + offs, q)
    _store_sf_d(e, pid_bh, t, S, il_ptr, dk_ptr, 1)
    qh = _deq_d(q, e, 32)
    et = mx_exponent(tl.max(tl.abs(qh), axis=0))
    tl.store(qt8_ptr + offs, to_e4m3(qh, et[None, :]))
    _store_sf_t(et, pid_bh, pid_g, S, qt_il_ptr)


@triton.jit
def _k_kernel(x_ptr, x8_ptr, il_ptr, dk_ptr, kb_ptr, S, H, KB: tl.constexpr):
    """K tile: E4M3 + UE8M0 along head_dim (interleaved, duplicated pair tiles) and, if KB, K-hat in
    BF16."""
    pid_g, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // H, pid_bh % H
    t = pid_g * 32 + tl.arange(0, 32)
    row = (b * S + t) * H + h
    offs = row[:, None] * 256 + tl.arange(0, 256)[None, :]
    q, e = _quant_d(tl.load(x_ptr + offs).to(tl.float32), 32)
    tl.store(x8_ptr + offs, q)
    _store_sf_d(e, pid_bh, t, S, il_ptr, dk_ptr, 2)
    if KB:
        tl.store(kb_ptr + offs, _deq_d(q, e, 32).to(tl.bfloat16))


@triton.jit
def _v_kernel(x_ptr, x8_ptr, il_ptr, vd8_ptr, vd_dk_ptr, S, H, RES: tl.constexpr, CH: tl.constexpr):
    """V tile (b, 32 keys, h, CH channels from CH * program_id(2)): E4M3 + UE8M0 per (32 keys,
    channel) for the forward (interleaved MN = channels), then V-hat requantized along head_dim into
    keys [0, S) of the dP operand and, causal (RES), the residual R = MX(V - rq(V-hat)) into keys
    [S, 2S) (duplicated pair-tile scales). Every scale group lies inside one program's channels (CH
    a multiple of 32)."""
    pid_g, pid_bh, pid_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b, h = pid_bh // H, pid_bh % H
    t = pid_g * 32 + tl.arange(0, 32)
    c = pid_c * CH + tl.arange(0, CH)
    offs = ((b * S + t) * H + h)[:, None] * 256 + c[None, :]
    x = tl.load(x_ptr + offs).to(tl.float32)
    ek = mx_exponent(tl.max(tl.abs(x), axis=0))
    v8 = to_e4m3(x, ek[None, :])
    tl.store(x8_ptr + offs, v8)
    _store_sf_t(ek, pid_bh, pid_g, S, il_ptr, pid_c * CH, CH)
    q, e = _quant_d(v8.to(tl.float32) * exp2_int(ek)[None, :], 32, CH)
    s_out = S * (2 if RES else 1)
    orow = (b * s_out + t) * H + h
    g0 = pid_c * (CH // 32)
    tl.store(vd8_ptr + orow[:, None] * 256 + c[None, :], q)
    _store_sf_d(e, pid_bh, t, s_out, vd_dk_ptr, vd_dk_ptr, 2, g0, CH // 32, IL=False)
    if RES:
        rq, re = _quant_d(x - _deq_d(q, e, 32, CH), 32, CH)
        rrow = (b * s_out + S + t) * H + h
        tl.store(vd8_ptr + rrow[:, None] * 256 + c[None, :], rq)
        _store_sf_d(re, pid_bh, S + t, s_out, vd_dk_ptr, vd_dk_ptr, 2, g0, CH // 32, IL=False)


@triton.jit
def _dout_kernel(
    do_ptr,
    o_ptr,
    stats_ptr,
    do8_ptr,
    sfdo_ptr,
    dot8_ptr,
    sfdot_il_ptr,
    dpsum_ptr,
    lse_ptr,
    dob_ptr,
    S,
    H,
    c_log2,
    off,
    BAND: tl.constexpr,
):
    """One (b, 32-query group, h) tile: dO'' = (256 / L) dO along head_dim (pair-tile scales) and
    along queries (interleaved MN = channels) in E4M3 + UE8M0, D'' = rowsum(dequant(dO''_d) O), and
    the LSE slot M - off with M = m c; per-row outputs (B, H, S). BAND (causal): also dO'' / 256 =
    dO / L in BF16."""
    pid_g, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // H, pid_bh % H
    t = pid_g * 32 + tl.arange(0, 32)
    row = (b * S + t) * H + h
    offs = row[:, None] * 256 + tl.arange(0, 256)[None, :]
    srow = pid_bh * S + t
    m = tl.load(stats_ptr + srow * 2)
    lq = tl.load(stats_ptr + srow * 2 + 1)
    mult = tl.math.div_rn(tl.full([32], 256.0, tl.float32), lq)
    x = tl.load(do_ptr + offs).to(tl.float32) * mult[:, None]
    if BAND:
        tl.store(dob_ptr + offs, (x * (1.0 / 256.0)).to(tl.bfloat16))
    q, e = _quant_d(x, 32)
    tl.store(do8_ptr + offs, q)
    _store_sf_d(e, pid_bh, t, S, sfdo_ptr, sfdo_ptr, 1, IL=False)
    et = mx_exponent(tl.max(tl.abs(x), axis=0))
    tl.store(dot8_ptr + offs, to_e4m3(x, et[None, :]))
    _store_sf_t(et, pid_bh, pid_g, S, sfdot_il_ptr)
    o = tl.load(o_ptr + offs).to(tl.float32)
    tl.store(dpsum_ptr + srow, tl.sum(_deq_d(q, e, 32) * o, axis=1))
    # mul.rn: no FMA contraction; M = m c is rounded before the subtraction (BlockScales.stats_out)
    mm = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [m, tl.full([32], c_log2, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(lse_ptr + srow, mm - off)


# channels per V emit program: 256 spills the tile at 2 warps
_V_CHANNELS = 128


def _v_warps(causal: bool, dev: torch.device) -> int:
    """V emit warps, from sweeps at B4 S16k: causal 2 / non-causal 1 on sm_100 and sm_103, the
    reverse on sm_107."""
    if torch.cuda.get_device_capability(dev) == (10, 7):
        return 1 if causal else 2
    return 2 if causal else 1


def _bytes(n, dev, s, mult):
    """Scale-layout buffer; zero padding (as the eager layouts) only when the tokens leave partial
    tiles."""
    return (torch.zeros if s % mult else torch.empty)(n, dtype=torch.uint8, device=dev)


def _e8(x):
    return x.view(torch.float8_e8m0fnu)


def _il_d_bytes(b, h, s):
    return b * h * triton.cdiv(s, 128) * 2 * 512


def _il_t_bytes(b, h, s):
    return b * h * 2 * triton.cdiv(triton.cdiv(s, 32), 4) * 512


class ForwardOnlyOperands(NamedTuple):
    """The MxHd256BwdOperands fields that depend only on the forward's inputs."""

    q: torch.Tensor
    sfq: torch.Tensor
    k: torch.Tensor
    sfk: torch.Tensor
    v: torch.Tensor
    sfv: torch.Tensor
    qt: torch.Tensor
    sfqt: torch.Tensor
    k_hat: torch.Tensor


class ForwardOperands(NamedTuple):
    """quantize_forward's outputs: the forward's E4M3 payloads with their BlockScales buffers and
    the backward's forward-only operands."""

    q8: torch.Tensor
    k8: torch.Tensor
    v8: torch.Tensor
    sfq: torch.Tensor
    sfk: torch.Tensor
    sfv: torch.Tensor
    bwd: ForwardOnlyOperands


def quantize_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> ForwardOperands:
    """Contiguous BF16 (B, S, H, 256) Q / K / V, S a multiple of 32 -> ForwardOperands, one pass per
    input."""
    b, s, h, d = q.shape
    assert d == 256 and s % 32 == 0 and all(t.is_contiguous() for t in (q, k, v))
    dev = q.device
    n_il_d, n_il_t = _il_d_bytes(b, h, s), _il_t_bytes(b, h, s)
    grid = (s // 32, b * h)
    q8, qt8 = (torch.empty_like(q, dtype=torch.float8_e4m3fn) for _ in range(2))
    sfq, sfqt = _bytes(n_il_d, dev, s, 128), _bytes(n_il_t, dev, s, 128)
    sfq_dk = _bytes(b * h * triton.cdiv(s, 128) * 1024, dev, s, 128)
    _q_kernel[grid](q, q8, sfq, sfq_dk, qt8, sfqt, s, h, num_warps=4)
    k8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    kb = torch.empty_like(k)
    sfk = _bytes(n_il_d, dev, s, 128)
    sfk_dk = _bytes(b * h * triton.cdiv(s, 64) * 1024, dev, s, 64)
    _k_kernel[grid](k, k8, sfk, sfk_dk, kb, s, h, KB=True, num_warps=4)
    s_v = 2 * s if causal else s
    v8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    sfv = _bytes(n_il_t, dev, s, 128)
    vd8 = torch.empty(b, s_v, h, 256, dtype=torch.float8_e4m3fn, device=dev)
    vd_dk = _bytes(b * h * triton.cdiv(s_v, 64) * 1024, dev, s_v, 64)
    _v_kernel[(*grid, 256 // _V_CHANNELS)](
        v, v8, sfv, vd8, vd_dk, s, h, RES=causal, CH=_V_CHANNELS, num_warps=_v_warps(causal, dev)
    )
    bwd = ForwardOnlyOperands(
        q=q8,
        sfq=_e8(sfq_dk),
        k=k8,
        sfk=_e8(sfk_dk),
        v=vd8,
        sfv=_e8(vd_dk),
        qt=qt8,
        sfqt=_e8(sfqt),
        k_hat=kb,
    )
    return ForwardOperands(q8, k8, v8, _e8(sfq), _e8(sfk), _e8(sfv), bwd)


def backward_operands(
    fwd: ForwardOnlyOperands,
    out: torch.Tensor,
    dout: torch.Tensor,
    stats: torch.Tensor,
    softmax_scale: float,
) -> MxHd256BwdOperands:
    """The backward's operands: the forward-only ones `fwd` (ForwardOperands.bwd) plus one pass over
    dO with the forward's contiguous O and stats_out (B, H, S, 2)."""
    b, s, h, _ = out.shape
    assert dout.shape == out.shape and stats.shape == (b, h, s, 2)
    assert all(t.is_contiguous() for t in (out, dout, stats))
    dev = out.device
    do8, dot8 = (torch.empty_like(dout, dtype=torch.float8_e4m3fn) for _ in range(2))
    sfdo = _bytes(b * h * triton.cdiv(s, 128) * 1024, dev, s, 128)
    sfdot = _bytes(_il_t_bytes(b, h, s), dev, s, 128)
    dpsum, lse = (torch.empty(b, h, s, dtype=torch.float32, device=dev) for _ in range(2))
    band = fwd.v.shape[1] == 2 * s  # causal: the V copy carries the band residual
    dob = torch.empty_like(dout) if band else None
    # c = fp32(fp32(softmax_scale) * fp32(log2(e))), the kernels' softmax_scale_log2, as
    # BlockScales.stats_out defines M = m c
    f32 = torch.float32
    c = (torch.tensor(softmax_scale, dtype=f32) * torch.tensor(math.log2(math.e), dtype=f32)).item()
    _dout_kernel[(s // 32, b * h)](
        dout,
        out,
        stats,
        do8,
        sfdo,
        dot8,
        sfdot,
        dpsum,
        lse,
        dob,
        s,
        h,
        c,
        float(Mxfp8Mirror.PV.p_offset),
        BAND=band,
        num_warps=4,
    )
    return MxHd256BwdOperands(
        **fwd._asdict(),
        do=do8,
        sfdo=_e8(sfdo),
        dot=dot8,
        sfdot=_e8(sfdot),
        lse_log2=lse,
        dpsum=dpsum,
        do_band=dob,
    )
