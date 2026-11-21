"""References for the mirrored BF16 backward of the hd64 / hd128 MXFP8 forwards.

NOTE [MXFP8 mirrored BF16 backward] in flash_attn/cute/flash_bwd_sm100.py. Every function works on
one (batch, head) at a time in FP64 / FP32 on (seqlen, dim) tensors; `per_head` maps them over
(batch, seqlen, heads, dim) inputs with GQA K / V expansion.

- `emulate_backward`: the kernel's recipe from the saved (m, L): P~ recomputed at the final max,
  one FFMA rounding into exp2; `faithful=True` adds the kernel's roundings (dO' and dS in BF16,
  FP32 S / dP / D), `faithful=False` keeps everything in FP64 (the recipe itself).
  `round_dout=False` ablates only the dO' = bf16(dO / L) rounding.
- `online_forward` + `oracle_backward`: the FP64 derivative of the function the forward computed,
  including its online max (blocks visited under an earlier running max were rounded there).
- `autograd_backward`: torch.autograd through an FP64 forward at the final max with
  straight-through roundings (checks the recipe's formula).
- `emulate_mx_backward`: the hdim-256 block-scaled backward's recipe (NOTE [MXFP8 hd256
  backward]): the same derivative with MXFP8 dP / dV / dK operands, BF16 dS for dQ, and the
  causal band's dP residual and BF16 dV.
"""

import math

import torch

LOG2E = math.log2(math.e)
F32, F64 = torch.float32, torch.float64
BAND_ROWS, BLOCK_N = 256, 128


def bf16(x):
    return x.to(torch.bfloat16).to(x.dtype)


def e4m3(x):
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def mx(x: torch.Tensor, dim: int) -> torch.Tensor:
    """x rounded to MXFP8 and dequantized: E4M3 with one power-of-two scale per 32 elements along
    dim, 2^ceil(log2(amax / 448)) (the forward's and the kernels' scale rule)."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    groups = torch.nn.functional.pad(x, (0, -n % 32)).unflatten(-1, (-1, 32)).double()
    amax = groups.abs().amax(-1, keepdim=True)
    mantissa, exponent = torch.frexp(amax)
    exponent = torch.where(amax == 0, 0, (exponent - 9 + (mantissa > 0.875)).clamp(-127, 127))
    scale = torch.exp2(exponent.double())
    q = (groups / scale).clamp(-448, 448).float().to(torch.float8_e4m3fn).double() * scale
    return q.flatten(-2)[..., :n].to(x.dtype).movedim(-1, dim)


def scale_log2(softmax_scale: float) -> torch.Tensor:
    """The kernels' FP32 softmax_scale * log2(e)."""
    return torch.tensor(softmax_scale, dtype=F32) * torch.tensor(LOG2E, dtype=F32)


def masks(sq: int, sk: int, causal: bool, native_pv: bool, device):
    """(visible, band): band = elements the causal native PV forward runs with BF16 P and V."""
    rows = torch.arange(sq, device=device)[:, None]
    cols = torch.arange(sk, device=device)[None, :]
    visible = cols <= rows + sk - sq if causal else torch.ones(sq, sk, dtype=torch.bool, device=device)
    if not (causal and native_pv):
        return visible, torch.zeros_like(visible)
    tile_start = rows // BAND_ROWS * BAND_ROWS
    band = (cols // BLOCK_N + 1) * BLOCK_N > tile_start + sk - sq
    return visible, visible & band


def slot(m: torch.Tensor, c: torch.Tensor, offset: int) -> torch.Tensor:
    """The preprocess's LSE slot fp32(m * c) - offset (0 for rows without keys)."""
    s = (m * c.to(m.device)) - offset
    return torch.where(torch.isneginf(m), torch.zeros_like(s), s)


def emulate_backward(qh, kh, vh, v, out, dout, m, L, *, softmax_scale, causal, native_pv,
                     faithful=True, round_dout=True):
    """(dq, dk, dv) of the recipe for one head: (seqlen, dim) tensors, m / L the forward's stats."""
    sq, sk = qh.shape[0], kh.shape[0]
    c = scale_log2(softmax_scale).to(qh.device)
    offset = 8 if native_pv else 0
    visible, band = masks(sq, sk, causal, native_pv, qh.device)
    s = qh.double() @ kh.double().T
    if faithful:  # FP32 S, one FFMA rounding into exp2
        x = (s.float().double() * c.double() - slot(m.float(), c, offset).double()[:, None]).float()
    else:  # FP64 at the final max (as autograd_backward)
        final = torch.where(torch.isneginf(m), torch.zeros_like(m), m.float() * c).double()
        x = s * c.double() + offset - final[:, None]
    p = torch.exp2(x).masked_fill(~visible, 0.0)
    p_hat = torch.where(band, bf16(p), e4m3(p)) if native_pv else bf16(p)
    inv_L = torch.where(L > 0, 1.0 / L.float(), torch.zeros_like(L.float()))
    do = dout.float() * inv_L[:, None]
    if faithful and round_dout:
        do = bf16(do)
    if not faithful:
        do = dout.double() / L.double()[:, None]
        do = torch.where(L[:, None] > 0, do, torch.zeros_like(do))
    d = (do.double() * out.double()).sum(-1)
    dp_hat = do.double() @ vh.double().T
    dp = torch.where(band, do.double() @ v.double().T, dp_hat) if native_pv else dp_hat
    if faithful:
        d, dp = d.float(), dp.float()
    ds = p * (dp - d[:, None])
    if faithful:
        ds = bf16(ds)
    dv = p_hat.double().T @ do.double()
    dk = softmax_scale * ds.double().T @ qh.double()
    dq = softmax_scale * ds.double() @ kh.double()
    return dq, dk, dv


def emulate_mx_backward(qh, kh, vh, v, out, dout, m, L, *, softmax_scale, causal, native_pv=True):
    """(dq, dk, dv) of the hdim-256 block-scaled backward for one head, in FP32 with its roundings:
    dO'' = 2^8 dO / L in MXFP8 along head_dim (dP, D'') and along queries (dV), V-hat requantized
    along head_dim with the causal band's residual R = MX(V - V-hat_d), dS'' rounded to MXFP8
    along queries for dK and to BF16 for dQ, the band's dV on BF16 P~ and dO / L."""
    assert native_pv, "the block-scaled backward mirrors native PV"
    sq, sk = qh.shape[0], kh.shape[0]
    c = scale_log2(softmax_scale).to(qh.device)
    visible, band = masks(sq, sk, causal, True, qh.device)
    s = (qh.double() @ kh.double().T).float()
    x = (s.double() * c.double() - slot(m.float(), c, 8).double()[:, None]).float()
    p = torch.exp2(x).masked_fill(~visible, 0.0)
    do2 = dout.float() * (torch.tensor(256.0, device=qh.device) / L.float())[:, None]
    do_d, do_q = mx(do2, 1), mx(do2, 0)
    v_d = mx(vh.float(), 1)
    dp = do_d.double() @ v_d.double().T
    if causal:
        v_band = v_d.double() + mx(v.float() - v_d, 1).double()
        dp = torch.where(band, do_d.double() @ v_band.T, dp)
    d = (do_d.double() * out.double()).sum(-1).float()
    ds = (p * (dp.float() - d[:, None])).float()
    dv = e4m3(p).masked_fill(band, 0.0).double().T @ do_q.double() / 256
    if causal:
        dv += bf16(p).masked_fill(~band, 0.0).double().T @ bf16(do2 / 256).double()
    dk = softmax_scale / 256 * mx(ds, 0).double().T @ mx(qh.float(), 0).double()
    dq = softmax_scale / 256 * bf16(ds).double() @ kh.double()
    return dq, dk, dv


def online_forward(qh, kh, vh, v, softmax_scale, *, causal, native_pv, pv_normalizer):
    """FP64 view of the online forward (reverse 128-key blocks; native PV updates the max every
    block, offset 8; QK-only keeps a stale max up to 8 log2 units, offset 0): the rounded P referred
    to the final max (ph_eff), P~ at the final max (pt_eff), L, O, and the final scaled max."""
    sq, sk = qh.shape[0], kh.shape[0]
    c = scale_log2(softmax_scale).to(qh.device)
    offset, threshold = (8.0, 0.0) if native_pv else (0.0, 8.0)
    visible, band = masks(sq, sk, causal, native_pv, qh.device)
    s = (qh.double() @ kh.double().T).float().masked_fill(~visible, -math.inf)
    mc = torch.full((sq,), -math.inf, dtype=F32, device=qh.device)
    ph_all = torch.zeros(sq, sk, dtype=F32, device=qh.device)
    mc_blk = torch.zeros(sq, sk, dtype=F32, device=qh.device)
    for start in reversed(range(0, sk, BLOCK_N)):
        cols = slice(start, start + BLOCK_N)
        cand = s[:, cols].amax(1) * c
        new = torch.maximum(mc, cand)
        keep = (mc > -math.inf) & ((mc - new) >= -threshold)
        new = torch.where(keep, mc, new)
        safe = torch.where(new > -math.inf, new, torch.zeros_like(new))
        x = (s[:, cols].double() * c.double() + (offset - safe).double()[:, None]).float()
        pt = torch.exp2(x).masked_fill(~visible[:, cols], 0.0)
        if native_pv:
            ph = torch.where(band[:, cols], bf16(pt), e4m3(pt))
        else:
            ph = bf16(pt)
        ph_all[:, cols] = ph
        mc_blk[:, cols] = safe[:, None]
        mc = new
    final = torch.where(mc > -math.inf, mc, torch.zeros_like(mc)).double()
    ph_eff = ph_all.double() * torch.exp2(mc_blk.double() - final[:, None])
    pt_eff = torch.exp2(s.double() * c.double() + offset - final[:, None]).masked_fill(~visible, 0.0)
    L = ph_eff.sum(1) if pv_normalizer == "rounded_p" else pt_eff.sum(1)
    v_sel = vh if native_pv else v
    o = (ph_eff * ~band) @ v_sel.double() + (ph_eff * band) @ v.double()
    o = o / torch.where(L > 0, L, torch.ones_like(L))[:, None]  # rows without keys: O = 0
    return dict(ph_eff=ph_eff, pt_eff=pt_eff, L=L, o=o, band=band, visible=visible, mc=mc)


def oracle_backward(fw, qh, kh, vh, v, dout, softmax_scale, *, native_pv):
    """FP64 derivative of the online forward `fw` (straight-through roundings, constant max)."""
    do, L = dout.double(), torch.where(fw["L"] > 0, fw["L"], torch.ones_like(fw["L"]))[:, None]
    v_sel = vh if native_pv else v
    dp = torch.where(fw["band"], do @ v.double().T, do @ v_sel.double().T)
    d = (do * fw["o"]).sum(1, keepdim=True)
    ds = fw["pt_eff"] / L * (dp - d)
    return (softmax_scale * ds @ kh.double(), softmax_scale * ds.T @ qh.double(),
            (fw["ph_eff"] / L).T @ do)


def autograd_backward(qh, kh, vh, v, dout, m, softmax_scale, *, causal, native_pv, pv_normalizer):
    """torch.autograd of an FP64 forward at the final max (from stats m) with straight-through P
    roundings; band elements read V, the others V-hat (native PV). Returns ((dq, dk, dv), O, L)
    with dV = dO-hat + dV (straight-through V-hat = q(V)) and the forward's FP64 O and L."""
    sq, sk = qh.shape[0], kh.shape[0]
    c = scale_log2(softmax_scale).to(qh.device).double()
    offset = 8.0 if native_pv else 0.0
    visible, band = masks(sq, sk, causal, native_pv, qh.device)
    leaves = [t.detach().double().requires_grad_() for t in (qh, kh, vh, v)]
    q_, k_, vh_, v_ = leaves
    final = torch.where(torch.isneginf(m), torch.zeros_like(m), m.float() * c.float()).double()
    s = (q_ @ k_.T).masked_fill(~visible, -math.inf)
    pt = torch.exp2(s * c + offset - final[:, None])
    rounded = torch.where(band, bf16(pt), e4m3(pt)) if native_pv else bf16(pt)
    ph = pt + (rounded - pt).detach()
    L = ph.sum(1) if pv_normalizer == "rounded_p" else pt.sum(1)
    if native_pv:
        o = ((ph * ~band) @ vh_ + (ph * band) @ v_) / L[:, None]
    else:
        o = (ph @ v_) / L[:, None]
    gq, gk, gvh, gv = torch.autograd.grad(o, leaves, dout.double(), allow_unused=True)
    dv = gvh + gv if native_pv else gv
    return (gq, gk, dv), o.detach(), L.detach()


def per_head(fn, operands, out, dout, stats, **kw):
    """Apply a per-head reference over (batch, seqlen, heads, dim) tensors; K / V heads are shared
    by qhead_per_kvhead query heads (GQA) and their gradients summed. Returns FP64 (dq, dk, dv)."""
    qh, kh, vh, v = operands
    b, _, hq, _ = qh.shape
    hk = kh.shape[2]
    rep = hq // hk
    dq = torch.zeros(qh.shape, dtype=F64, device=qh.device)
    dk = torch.zeros(kh.shape, dtype=F64, device=qh.device)
    dv = torch.zeros(vh.shape, dtype=F64, device=qh.device)
    for bi in range(b):
        for h in range(hq):
            g = h // rep
            gq, gk, gv = fn(
                qh[bi, :, h], kh[bi, :, g], vh[bi, :, g], v[bi, :, g], out[bi, :, h],
                dout[bi, :, h], stats[bi, h, :, 0], stats[bi, h, :, 1], **kw,
            )
            dq[bi, :, h] = gq
            dk[bi, :, g] += gk
            dv[bi, :, g] += gv
    return dq, dk, dv


def oracle_per_head(qh, kh, vh, v, out, dout, m, L, *, softmax_scale, causal, native_pv,
                    pv_normalizer):
    fw = online_forward(qh, kh, vh, v, softmax_scale, causal=causal, native_pv=native_pv,
                        pv_normalizer=pv_normalizer)
    return oracle_backward(fw, qh, kh, vh, v, dout, softmax_scale, native_pv=native_pv)


def autograd_per_head(qh, kh, vh, v, out, dout, m, L, *, softmax_scale, causal, native_pv,
                      pv_normalizer):
    return autograd_backward(qh, kh, vh, v, dout, m, softmax_scale, causal=causal,
                             native_pv=native_pv, pv_normalizer=pv_normalizer)[0]


def ideal_backward(q, k, v, dout, softmax_scale, causal):
    """FP64 gradient of exact attention on the BF16 inputs (batch, seqlen, heads, dim)."""
    rep = q.shape[2] // k.shape[2]
    leaves = [t.detach().double().requires_grad_() for t in (q, k, v)]
    q_, k_, v_ = leaves
    kk, vv = (t.repeat_interleave(rep, dim=2) for t in (k_, v_))
    s = torch.einsum("bqhd,bkhd->bhqk", q_, kk) * softmax_scale
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        visible, _ = masks(sq, sk, True, False, q.device)
        s = s.masked_fill(~visible, -math.inf)
    o = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1).nan_to_num(), vv)
    return torch.autograd.grad(o, leaves, dout.double())


def rel_l2(x, ref) -> float:
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm()).item()
