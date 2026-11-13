"""Tiled attention oracle for quant-dequant of attention probabilities.

Quantized P is relative to a running row max, so the result depends on the order
and grouping in which a row sees its keys. This oracle replays that schedule:
KV tiles of ``tile_n`` keys visited last to first, the KV range of each query
tile, SplitKV splits, and the masks. It works on one KV head and the query heads
that share it.

Two paths:

* ``path="math"``: FP64 arithmetic; only the codec runs in FP32. With an identity
  codec this is plain attention, for any schedule.
* ``path="impl"``: keeps the kernel's conversions: FP32 scores, the separately
  rounded bias, FP32 probabilities, the V-dtype (BF16) operand, FP32 accumulation
  and a BF16 output.

It can also compute, per row, the allowance for codec decisions that the kernel's
arithmetic could flip (see docs/p-quant-contract.md, section 7).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from flash_attn.cute.p_quant import (
    NVFP4_BLOCK,
    NVFP4_GLOBAL_SCALE,
    NVFP4_SCALE_FACTOR,
    PQuantFormat,
    resolve_p_quant,
)
from p_quant_oracle import ORACLE_CODECS, e2m1_round, e4m3_round, qdq

LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)
_U32 = 2.0**-24
# Relative error bound of the polynomial exp2 emulation; a conservative bound for
# every element (hardware exp2 is far tighter).
EXP2_REL_ERR = 9e-5


@dataclass(frozen=True)
class Schedule:
    """Inputs that decide which KV tiles a row visits, mirroring ``BlockInfo``.

    tile_m is the effective query tile in rows: ``q_stage * tile_m * cta_group_size``.
    pack_factor is ``qhead_per_kvhead`` under pack-GQA (rows are ``q_pos * g + h``), else 1.
    window_left / window_right of None mean unbounded on that side; ``causal`` forces the
    right bound to the diagonal.
    num_splits > 1 or n_blocks_per_split enables SplitKV.
    ex2_emu_* say which elements of a tile use the polynomial exp2 (see ``emulated_exp2_mask``).
    """

    tile_n: int
    tile_m: int
    pack_factor: int = 1
    causal: bool = False
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    num_splits: int = 1
    n_blocks_per_split: Optional[int] = None
    # exp2 emulation knobs of the forward config (0 = hardware exp2 everywhere).
    ex2_emu_freq: int = 0
    ex2_emu_res: int = 4
    ex2_emu_start_frg: int = 0

    @property
    def is_local(self) -> bool:
        return not self.causal and (
            self.window_left is not None or self.window_right is not None
        )

    @property
    def is_split_kv(self) -> bool:
        return self.num_splits > 1 or self.n_blocks_per_split is not None


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def n_block_range(
    sch: Schedule, seqlen_q: int, seqlen_k: int, m_block: int, split_idx: int = 0
) -> Tuple[int, int]:
    """KV tile range [min, max) of one query tile and split (``BlockInfo.get_n_block_min_max``)."""
    g = sch.pack_factor
    n_block_max = _ceil_div(seqlen_k, sch.tile_n)
    if sch.causal or (sch.is_local and sch.window_right is not None):
        m_idx_max = (m_block + 1) * sch.tile_m
        if g > 1:
            m_idx_max = _ceil_div(m_idx_max, g)
        n_idx = m_idx_max + seqlen_k - seqlen_q
        n_idx_right = n_idx if sch.causal else n_idx + sch.window_right
        n_block_max = min(n_block_max, _ceil_div(n_idx_right, sch.tile_n))
    n_block_min = 0
    if sch.is_local and sch.window_left is not None:
        m_idx_min = m_block * sch.tile_m
        if g > 1:
            m_idx_min = m_idx_min // g
        n_idx_left = m_idx_min + seqlen_k - seqlen_q - sch.window_left
        n_block_min = max(n_idx_left // sch.tile_n, 0)
    if sch.is_split_kv:
        if sch.n_blocks_per_split is not None:
            per_split = sch.n_blocks_per_split
        elif n_block_max <= n_block_min:
            per_split = 0
        else:
            per_split = (
                n_block_max - n_block_min + sch.num_splits - 1
            ) // sch.num_splits
        n_block_min = n_block_min + split_idx * per_split
        n_block_max = min(n_block_min + per_split, n_block_max)
    return n_block_min, n_block_max


def num_splits_for(sch: Schedule, seqlen_q: int, seqlen_k: int, m_block: int) -> int:
    """Number of splits a query tile is divided into."""
    if not sch.is_split_kv:
        return 1
    if sch.n_blocks_per_split is None:
        return sch.num_splits
    base = Schedule(
        sch.tile_n,
        sch.tile_m,
        sch.pack_factor,
        sch.causal,
        sch.window_left,
        sch.window_right,
    )
    lo, hi = n_block_range(base, seqlen_q, seqlen_k, m_block)
    return max(1, _ceil_div(max(hi - lo, 0), sch.n_blocks_per_split))


def row_tiles(sch: Schedule, seqlen_q: int, seqlen_k: int, q_pos: int, head: int = 0):
    """Per split, the KV tiles (descending) the row of (q_pos, head) visits."""
    row = q_pos * sch.pack_factor + head if sch.pack_factor > 1 else q_pos
    m_block = row // sch.tile_m
    out: List[List[int]] = []
    for split in range(num_splits_for(sch, seqlen_q, seqlen_k, m_block)):
        lo, hi = n_block_range(sch, seqlen_q, seqlen_k, m_block, split)
        out.append(list(range(hi - 1, lo - 1, -1)))
    return out


def key_visible(
    sch: Schedule, seqlen_q: int, seqlen_k: int, q_pos: int
) -> torch.Tensor:
    """Boolean mask over keys for one query position (sequence end, causal, window)."""
    j = torch.arange(seqlen_k)
    c = q_pos + seqlen_k - seqlen_q
    vis = torch.ones(seqlen_k, dtype=torch.bool)
    if sch.causal:
        vis &= j <= c
    elif sch.window_right is not None:
        vis &= j <= c + sch.window_right
    if not sch.causal and sch.window_left is not None:
        vis &= j >= c - sch.window_left
    return vis


# Cubic used by the kernel's exp2 emulation (flash_attn/cute/utils.py, POLY_EX2[3]).
_POLY_EX2 = (
    1.0,
    0.695146143436431884765625,
    0.227564394474029541015625,
    0.077119089663028717041015625,
)


def emulated_exp2_mask(sch: Schedule, width: int) -> torch.Tensor:
    """Which columns of a tile use the polynomial exp2 (``SoftmaxSm100.apply_exp2_convert``)."""
    i = torch.arange(width)
    if sch.ex2_emu_freq == 0:
        return torch.zeros(width, dtype=torch.bool)
    frg, pair = i // 32, (i % 32) & ~1
    frg_cnt = sch.tile_n // 32
    hardware = (
        (pair % sch.ex2_emu_freq < sch.ex2_emu_freq - sch.ex2_emu_res)
        | (frg >= frg_cnt - 1)
        | (frg < sch.ex2_emu_start_frg)
    )
    return ~hardware


def exp2_emulated(z: torch.Tensor) -> torch.Tensor:
    """The kernel's polynomial exp2 on FP32 inputs (held in FP64), operation for operation:
    clamp at -127, floor, cubic in the fractional part with one rounding per fused step,
    then the integer part added to the exponent."""
    zc = torch.clamp(z, min=-127.0)
    fl = torch.floor(zc)
    frac = _fl32(zc - fl)
    acc = torch.full_like(frac, _POLY_EX2[3])
    for c in _POLY_EX2[2::-1]:
        acc = _fl32(acc * frac + c)
    out = torch.ldexp(acc, fl.to(torch.int64))
    return torch.where(zc <= -127.0, torch.zeros_like(out), out)


_FLT_MIN = (
    2.0**-126
)  # smallest normal FP32; bounds flush-to-zero and emulated-exp2 underflow


def _gamma(n: int, v: float) -> float:
    """Standard accumulation factor n v / (1 - n v) for n rounded operations of unit error v."""
    return n * v / (1 - n * v)


def _fl32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(torch.float64)


def _ulp32(x: torch.Tensor) -> torch.Tensor:
    x32 = x.abs().to(torch.float32)
    return (torch.nextafter(x32, torch.full_like(x32, float("inf"))) - x32).to(
        torch.float64
    )


def _pad_to_block(x: torch.Tensor) -> torch.Tensor:
    pad = (-x.shape[-1]) % NVFP4_BLOCK
    return torch.nn.functional.pad(x, (0, pad)) if pad else x


def _codec(x: torch.Tensor, fmt: Optional[PQuantFormat], rounding: str) -> torch.Tensor:
    """Apply the codec to FP64 codec inputs through FP32, keys on the last axis."""
    if fmt is None or fmt.codec == "identity":
        return x
    n = x.shape[-1]
    return qdq(_pad_to_block(x.to(torch.float32)), fmt, rounding)[..., :n].to(
        torch.float64
    )


_MAX_SCALE_CODES = 127  # there are 127 non-negative E4M3 values


def _e4m3_next_up(x: torch.Tensor) -> torch.Tensor:
    """The next representable E4M3 value above a non-negative E4M3 value (FP32 in, FP32 out)."""
    _, e = torch.frexp(x)
    binade = (e - 1).clamp(min=-6, max=8)  # subnormals share the smallest normal binade
    step = torch.ldexp(torch.ones_like(x), binade - 3)
    return torch.clamp(x + step, max=448.0)


def _reachable_error(
    x: torch.Tensor, delta: torch.Tensor, y: torch.Tensor, fmt: PQuantFormat, to_operand
) -> torch.Tensor:
    """Largest |operand - y| reachable when each codec input may move by +-delta.

    Block-aware for NVFP4: the block scale is decided by all inputs of the block, and
    every reachable scale code is propagated to every key of the block.
    """
    lo = (x - delta).clamp(min=0.0).to(torch.float32)
    hi = (x + delta).to(torch.float32)
    lo = torch.where(x == 0, torch.zeros_like(lo), lo)  # masked keys are exactly zero
    hi = torch.where(x == 0, torch.zeros_like(hi), hi)
    if fmt.block == 1:
        # Elementwise codecs are monotone, so the two interval ends bound the operands.
        codec = ORACLE_CODECS[fmt.codec]
        cands = [codec(lo, "rne"), codec(hi, "rne")]
    else:
        n = x.shape[-1]
        lo_b, hi_b = _pad_to_block(lo), _pad_to_block(hi)
        factor = torch.tensor(NVFP4_SCALE_FACTOR, dtype=torch.float32)
        glob = torch.tensor(NVFP4_GLOBAL_SCALE, dtype=torch.float32)

        def block_max(v):
            return v.reshape(*v.shape[:-1], -1, NVFP4_BLOCK).amax(-1, keepdim=True)

        sig_lo = e4m3_round(block_max(lo_b) * factor)
        sig_hi = e4m3_round(block_max(hi_b) * factor)
        # Every E4M3 scale code from sig_lo to sig_hi is reachable (the block max can sit
        # anywhere in its interval). For each one, the code decision is monotone in the
        # input, so the two interval ends bound the reachable operands.
        cands = []
        sigma = sig_lo
        for _ in range(_MAX_SCALE_CODES):
            eff = sigma / glob
            safe = torch.where(sigma == 0, torch.ones_like(eff), eff)
            for v in (lo_b, hi_b):
                blocks = v.reshape(*v.shape[:-1], -1, NVFP4_BLOCK)
                out = torch.where(
                    sigma == 0,
                    torch.zeros_like(blocks),
                    e2m1_round(blocks / safe) * safe,
                )
                cands.append(out.reshape(v.shape)[..., :n])
            if (sigma >= sig_hi).all():
                break
            sigma = torch.minimum(_e4m3_next_up(sigma), sig_hi)
        else:
            raise AssertionError("input interval spans too many E4M3 scale codes")
    err = torch.zeros_like(x)
    for c in cands:
        err = torch.maximum(err, (to_operand(c.to(torch.float64)) - y).abs())
    return err


@dataclass
class OracleResult:
    out: torch.Tensor  # [Sq, G, Dv]
    lse: torch.Tensor  # [Sq, G], natural log; -inf for empty rows
    prob_allowance: Optional[torch.Tensor] = None  # [Sq, G]
    out_allowance: Optional[torch.Tensor] = None  # [Sq, G, Dv]
    eligible_keys: Optional[torch.Tensor] = None  # [Sq, G, Sk] bool


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    sch: Schedule,
    p_quant=None,
    path: str = "math",
    rounding: str = "rne",
    quantized_row_sum: bool = False,
    with_allowance: bool = False,
    block_aware: bool = True,
    exp2_rel_err: float = EXP2_REL_ERR,
    forget_running_max: bool = False,
) -> OracleResult:
    """Attention for one KV head under a visit schedule.

    q: [Sq, G, D] (the G query heads that share this KV head), k: [Sk, D], v: [Sk, Dv].
    ``quantized_row_sum``, ``block_aware=False`` and ``forget_running_max`` (the uncertainty
    of the running max taken from the current tile only) are deliberately wrong variants
    that tests use to show the checks detect them.
    """
    assert path in ("math", "impl")
    fmt = resolve_p_quant(p_quant)
    impl = path == "impl"
    seqlen_q, n_heads, _ = q.shape
    seqlen_k, dv = v.shape
    q64, k64, v64 = q.double(), k.double(), v.double()
    offset = float(fmt.max_offset) if fmt is not None else 0.0
    alpha = softmax_scale * LOG2_E  # scores to log2 units
    a = float(torch.tensor(alpha, dtype=torch.float32)) if impl else alpha

    def to_operand(x):  # the existing conversion to the V dtype
        return x.to(torch.bfloat16).to(torch.float64) if impl else x

    out = torch.zeros(seqlen_q, n_heads, dv, dtype=torch.float64)
    lse = torch.full((seqlen_q, n_heads), float("-inf"), dtype=torch.float64)
    prob_allow = torch.zeros(seqlen_q, n_heads, dtype=torch.float64)
    out_allow = torch.zeros(seqlen_q, n_heads, dv, dtype=torch.float64)
    eligible = torch.zeros(seqlen_q, n_heads, seqlen_k, dtype=torch.bool)

    for q_pos in range(seqlen_q):
        vis = key_visible(sch, seqlen_q, seqlen_k, q_pos)
        for h in range(n_heads):
            d_all = k64 @ q64[q_pos, h]  # raw dot products, [Sk]
            if impl:
                d_all = _fl32(d_all)
            abs_dot = k64.abs() @ q64[q_pos, h].abs()
            # Tensor-core accumulation envelope of each raw score (contract 7.1):
            # gamma_h(2u) * sum|q k| plus an underflow term for flushed products.
            e_d_all = _gamma(q.shape[-1], 2 * _U32) * abs_dot + q.shape[-1] * _FLT_MIN
            partial = []  # per split: (out_s, lse_s, [(keys, a_i * r_i / D_s)])
            for tiles in row_tiles(sch, seqlen_q, seqlen_k, q_pos, h):
                m = float("-inf")
                m_lo = m_hi = float(
                    "-inf"
                )  # bounds of the running max the kernel may hold
                den = 0.0
                num = torch.zeros(dv, dtype=torch.float64)
                seen = []  # (keys, a_i, row max when the tile was processed)
                for n_block in tiles:
                    lo, hi = (
                        n_block * sch.tile_n,
                        min((n_block + 1) * sch.tile_n, seqlen_k),
                    )
                    keys = torch.arange(lo, hi)
                    tvis = vis[lo:hi]
                    d = torch.where(
                        tvis, d_all[lo:hi], torch.full_like(d_all[lo:hi], float("-inf"))
                    )
                    tile_max = float(d.max()) if d.numel() else float("-inf")
                    m_new = max(m, tile_max)
                    if tvis.any():
                        m_lo = max(m_lo, float((d - e_d_all[lo:hi])[tvis].max()))
                        m_hi = max(m_hi, float((d + e_d_all[lo:hi])[tvis].max()))
                    if m_new == float("-inf"):
                        continue  # nothing visible yet
                    acc_scale = 0.0 if m == float("-inf") else 2.0 ** ((m - m_new) * a)
                    if impl:
                        acc_scale = float(torch.tensor(acc_scale, dtype=torch.float32))
                        bias = _fl32(
                            torch.tensor(offset) - _fl32(torch.tensor(m_new * a))
                        )
                        z = _fl32(d * a + bias)  # fused multiply-add: one rounding
                        x = _fl32(torch.exp2(z))
                        emu = emulated_exp2_mask(sch, hi - lo)
                        if emu.any():
                            x = torch.where(emu, exp2_emulated(z), x)
                    else:
                        x = torch.exp2((d - m_new) * a + offset)
                    x = torch.where(tvis, x, torch.zeros_like(x))
                    y = to_operand(_codec(x, fmt, rounding))
                    den_terms = y if quantized_row_sum else x
                    den = den * acc_scale + float(den_terms.sum())
                    num = num * acc_scale + y @ v64[lo:hi]
                    if impl:
                        den = float(torch.tensor(den, dtype=torch.float32))
                        num = _fl32(num)
                    if with_allowance and fmt is not None and fmt.codec != "identity":
                        # Path bound on the codec input (contract 7.1), then the margin.
                        e_d = e_d_all[lo:hi]
                        if forget_running_max:
                            e_m = float(e_d[tvis].max()) if tvis.any() else 0.0
                        else:
                            # The running max may come from an earlier tile: its uncertainty
                            # is the interval of the max over every key seen in this split.
                            e_m = max(m_hi - m_new, m_new - m_lo)
                        half_ulps = 0.5 * (
                            _ulp32(torch.tensor(m_new * a))
                            + _ulp32(torch.tensor(offset - m_new * a))
                            + _ulp32((d.clamp(min=-1e30) - m_new) * a + offset)
                        )
                        e_z = (
                            abs(a) * (e_d + e_m)
                            + (d.clamp(min=-1e30) - m_new).abs() * abs(a) * _U32
                        )
                        e_z = torch.where(tvis, e_z + half_ulps, torch.zeros_like(e_z))
                        # The impl path reproduces the polynomial exp2; the math path uses the
                        # exact exp2, so emulated columns carry the polynomial's error bound.
                        emu_err = (
                            torch.zeros_like(x)
                            if impl
                            else emulated_exp2_mask(sch, hi - lo).double()
                            * exp2_rel_err
                        )
                        eps = (
                            x * (torch.exp2(e_z) - 1)
                            + x * emu_err
                            + 2.5 * _ulp32(x)
                            + _FLT_MIN
                        )
                        delta = 2 * eps + 4 * _ulp32(x)
                        if block_aware:
                            a_i = _reachable_error(x, delta, y, fmt, to_operand)
                        else:
                            # Wrong on purpose: each key only looks at its own code decision,
                            # keeping the nominal block scale.
                            a_i = torch.zeros_like(x)
                            for xb in ((x - delta).clamp(min=0.0), x + delta):
                                xb = torch.where(x == 0, x, xb)
                                yb = to_operand(
                                    _codec_fixed_scale(xb, x, fmt, rounding)
                                )
                                a_i = torch.maximum(a_i, (yb - y).abs())
                        seen.append((keys, a_i, m_new))
                    m = m_new
                if den == 0.0:
                    partial.append(
                        (torch.zeros(dv, dtype=torch.float64), float("-inf"), [])
                    )
                    continue
                lse_s = (m * a + math.log2(den) - offset) * LN2
                allow = [
                    (keys, a_i * 2.0 ** ((m_t - m) * a) / den)
                    for keys, a_i, m_t in seen
                ]
                partial.append((num / den, lse_s, allow))
            lses = torch.tensor([p[1] for p in partial], dtype=torch.float64)
            if torch.isinf(lses).all():
                continue  # empty row: O = 0, LSE = -inf
            total = torch.logsumexp(lses, dim=0)
            weights = torch.exp(lses - total)
            lse[q_pos, h] = total
            row_out = sum(w * p[0] for w, p in zip(weights, partial))
            out[q_pos, h] = to_operand(row_out) if impl else row_out
            for w, p in zip(weights, partial):
                for keys, contrib in p[2]:
                    contrib = contrib * w
                    prob_allow[q_pos, h] += float(contrib.sum())
                    out_allow[q_pos, h] += contrib @ v64[keys].abs()
                    eligible[q_pos, h, keys] |= contrib > 0
    res = OracleResult(out, lse)
    if with_allowance:
        res.prob_allowance, res.out_allowance, res.eligible_keys = (
            prob_allow,
            out_allow,
            eligible,
        )
    return res


def _codec_fixed_scale(xb, x_nominal, fmt, rounding):
    """Codec of ``xb`` with the NVFP4 block scale taken from ``x_nominal`` (not from ``xb``)."""
    if fmt.codec != "nvfp4":
        return _codec(xb, fmt, rounding)
    n = xb.shape[-1]
    nom = _pad_to_block(x_nominal.to(torch.float32)).reshape(-1, NVFP4_BLOCK)
    blocks = _pad_to_block(xb.to(torch.float32)).reshape(-1, NVFP4_BLOCK)
    sigma = e4m3_round(
        nom.amax(-1, keepdim=True)
        * torch.tensor(NVFP4_SCALE_FACTOR, dtype=torch.float32)
    )
    eff = sigma / torch.tensor(NVFP4_GLOBAL_SCALE, dtype=torch.float32)
    safe = torch.where(sigma == 0, torch.ones_like(eff), eff)
    out = torch.where(
        sigma == 0, torch.zeros_like(blocks), e2m1_round(blocks / safe, rounding) * safe
    )
    return out.reshape(-1)[:n].to(torch.float64)


def merge_partials(outs: List[torch.Tensor], lses: List[torch.Tensor]):
    """Merge per-chunk outputs by their LSE, as SplitKV and context parallel do."""
    stacked = torch.stack(lses)  # [n, ...]
    total = torch.logsumexp(stacked, dim=0)
    weights = torch.exp(stacked - total)
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
    out = sum(w.unsqueeze(-1) * o for w, o in zip(weights, outs))
    return out, total


def textbook_attention(q, k, v, softmax_scale: float, sch: Schedule, p_quant=None):
    """Quantize the fully normalized row at once (no tiling): what a schedule-blind
    reference would compute. Used only to show it is not a valid kernel reference."""
    fmt = resolve_p_quant(p_quant)
    whole = Schedule(
        tile_n=1 << 30,
        tile_m=1 << 30,
        causal=sch.causal,
        window_left=sch.window_left,
        window_right=sch.window_right,
    )
    return tiled_attention(q, k, v, softmax_scale, whole, fmt)
