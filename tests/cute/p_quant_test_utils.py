"""Helpers for the P quant-dequant kernel tests: resolved-config capture, the oracle schedule
of a kernel call, whole-batch oracle evaluation, and the FP64 surrogate-gradient reference."""

import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch

from flash_attn.cute import interface
from flash_attn.cute.config import needs_split_metadata
from p_quant_attention_oracle import Schedule, key_visible, tiled_attention


def device_arch() -> int:
    """Architecture the kernels are built for (honours FLASH_ATTENTION_ARCH)."""
    return interface._get_device_arch()


# Identity of the oracle the comparisons were made with (its two source files).
ORACLE_IDENTITY = hashlib.sha256(
    b"".join(
        (Path(__file__).parent / name).read_bytes()
        for name in ("p_quant_attention_oracle.py", "p_quant_oracle.py")
    )
).hexdigest()


def record_evidence(kind: str, **fields) -> None:
    """Append one line to $P_QUANT_EVIDENCE_DIR/evidence.jsonl, when that directory is set
    (the hardware batch sets it): what each test resolved and compared against."""
    directory = os.environ.get("P_QUANT_EVIDENCE_DIR")
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    test = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
    with open(os.path.join(directory, "evidence.jsonl"), "a") as f:
        f.write(json.dumps(dict(kind=kind, test=test, **fields), default=repr) + "\n")


def _split_counts_for_record(counts):
    if counts is None:
        return None  # one static split count for the whole call
    try:
        return [int(c) for c in counts.detach().cpu().tolist()]
    except Exception as exc:  # a FakeTensor in the compile pass has no values
        return f"unavailable ({type(exc).__name__})"


class CapturedConfigs(list):
    """(FwdConfig, inputs) of each forward call, plus the per-sequence split counts the
    forward handed to its scheduler when it prepared them (None for static splits)."""

    def __init__(self):
        super().__init__()
        self.split_counts = []


@contextmanager
def capture_fwd_config():
    """Record what every forward call resolves, so the oracle replays the schedule the kernel
    really used instead of re-deriving it."""
    captured = CapturedConfigs()
    original_config = interface._get_fwd_config
    original_combine = interface._flash_attn_fwd_combine

    def recording_config(**kwargs):
        result = original_config(**kwargs)
        captured.append(result)
        captured.split_counts.append(None)
        return result

    def recording_combine(*args, **kwargs):
        # Every SplitKV forward ends in the combine, which gets the per-sequence counts the
        # prepare-scheduler kernel filled (None for one static count). The tile-cumsum helper
        # also receives them, but only for batches large enough to use its search hint.
        counts = kwargs.get("num_splits_dynamic_ptr")
        if counts is not None and captured.split_counts:
            captured.split_counts[-1] = counts
        return original_combine(*args, **kwargs)

    # The combine reads its compile cache through the module global, i.e. this wrapper.
    recording_combine.compile_cache = original_combine.compile_cache
    interface._get_fwd_config = recording_config
    interface._flash_attn_fwd_combine = recording_combine
    try:
        yield captured
    finally:
        interface._get_fwd_config = original_config
        interface._flash_attn_fwd_combine = original_combine
        for index, (config, inputs) in enumerate(captured):
            record_evidence(
                "forward_call",
                index=index,
                config=repr(config),
                inputs=repr(inputs),
                per_sequence_split_counts=_split_counts_for_record(
                    captured.split_counts[index]
                ),
            )


def per_sequence_split_counts(captured: CapturedConfigs, index: int = 0):
    """Split count of each sequence for call ``index``, read back from the scheduler metadata
    the kernel used; None when the call used one static count."""
    counts = captured.split_counts[index]
    return None if counts is None else [int(c) for c in counts.detach().cpu().tolist()]


def schedule_of(
    config,
    inputs,
    *,
    qhead_per_kvhead: int,
    causal: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    seqlen_k_per_split: Optional[int] = None,
    num_splits: Optional[int] = None,
) -> Schedule:
    """Oracle schedule of a forward call from its resolved config (contract section 5).

    When the forward prepares per-sequence split counts, ``num_splits`` must be the count of
    the sequence the schedule is for (see ``per_sequence_split_counts``); config.num_splits
    is then only their upper bound.
    """
    assert not config.use_swapab
    if needs_split_metadata(config, inputs):
        if num_splits is None:
            raise ValueError(
                "this call uses per-sequence split counts: pass the sequence's count"
            )
        assert 1 <= num_splits <= config.num_splits, (num_splits, config.num_splits)
    else:
        assert num_splits is None or num_splits == config.num_splits
        num_splits = config.num_splits
    causal, _local, wl, wr = interface._resolve_causal_local_window(
        causal, window_size_left, window_size_right, None
    )
    schedule = Schedule(
        tile_n=config.tile_n,
        # A scheduler work tile spans q_stage query tiles per CTA and cta_group_size CTAs.
        tile_m=config.q_stage * config.tile_m * (2 if config.use_2cta_instrs else 1),
        pack_factor=qhead_per_kvhead if config.pack_gqa else 1,
        causal=causal,
        window_left=None if causal else wl,
        window_right=None if causal else wr,
        num_splits=num_splits,
        n_blocks_per_split=(
            None if seqlen_k_per_split is None else seqlen_k_per_split // config.tile_n
        ),
        ex2_emu_freq=config.ex2_emu_freq,
        ex2_emu_start_frg=config.ex2_emu_start_frg,
    )
    record_evidence("schedule", schedule=repr(schedule), oracle=ORACLE_IDENTITY)
    return schedule


def oracle_batch(
    q, k, v, softmax_scale, sch: Schedule, p_quant=None, seqlens=None, **kwargs
):
    """Run the tiled oracle over a dense batch.

    q: [B, Sq, H, D], k / v: [B, Sk, Hkv, D] (any float dtype; values are taken as given).
    seqlens: optional per-batch (seqlen_q, seqlen_k) of the valid prefix (seqused / varlen).
    Returns out [B, Sq, H, Dv], lse [B, H, Sq], and (with with_allowance) prob_allowance
    [B, H, Sq] and out_allowance [B, Sq, H, Dv]. Rows and keys past a sequence's length stay 0
    / -inf.
    """
    q, k, v = (
        q.detach().cpu().double(),
        k.detach().cpu().double(),
        v.detach().cpu().double(),
    )
    b, sq, h, _ = q.shape
    hk = k.shape[2]
    g = h // hk
    out = torch.zeros(b, sq, h, v.shape[-1], dtype=torch.float64)
    lse = torch.full((b, h, sq), float("-inf"), dtype=torch.float64)
    prob_allow = torch.zeros(b, h, sq, dtype=torch.float64)
    out_allow = torch.zeros_like(out)
    for bi in range(b):
        lq, lk = (sq, k.shape[1]) if seqlens is None else seqlens[bi]
        if lq == 0:
            continue
        for a in range(hk):
            res = tiled_attention(
                q[bi, :lq, a * g : (a + 1) * g],
                k[bi, :lk, a],
                v[bi, :lk, a],
                softmax_scale,
                sch,
                p_quant,
                **kwargs,
            )
            out[bi, :lq, a * g : (a + 1) * g] = res.out
            lse[bi, a * g : (a + 1) * g, :lq] = res.lse.transpose(0, 1)
            if res.prob_allowance is not None:
                prob_allow[bi, a * g : (a + 1) * g, :lq] = res.prob_allowance.transpose(
                    0, 1
                )
                out_allow[bi, :lq, a * g : (a + 1) * g] = res.out_allowance
    return out, lse, prob_allow, out_allow


def surrogate_grads(
    q,
    k,
    v,
    out_q,
    lse,
    dout,
    softmax_scale,
    sch: Schedule,
    dlse=None,
    operand_dtype=None,
    seqlens=None,
):
    """Surrogate gradient of the contract (section 4) for a dense batch.

    Inputs are promoted to FP64; ``out_q`` and ``lse`` are the forward's saved output and
    natural-log LSE. ``operand_dtype`` (e.g. torch.bfloat16) rounds P before dV, dS before
    dQ / dK, and the returned gradients, as the backward kernel does; None keeps FP64.
    Returns dq, dk, dv.
    """
    q, k, v, out_q, lse, dout = (
        t.detach().cpu().double() for t in (q, k, v, out_q, lse, dout)
    )
    dlse = None if dlse is None else dlse.detach().cpu().double()
    b, sq, h, _ = q.shape
    sk, hk = k.shape[1], k.shape[2]
    g = h // hk
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    def rnd(x):
        return x if operand_dtype is None else x.to(operand_dtype).double()

    for bi in range(b):
        lq, lk = (sq, sk) if seqlens is None else seqlens[bi]
        vis = (
            torch.stack([key_visible(sch, lq, lk, i) for i in range(lq)])
            if lq
            else None
        )
        for hi in range(h):
            if lq == 0:
                continue
            a = hi // g
            s = softmax_scale * q[bi, :lq, hi] @ k[bi, :lk, a].T  # [lq, lk]
            row_lse = lse[bi, hi, :lq]
            empty = torch.isinf(row_lse)
            p = torch.exp(
                s - torch.where(empty, torch.zeros_like(row_lse), row_lse)[:, None]
            )
            p = torch.where(vis & ~empty[:, None], p, torch.zeros_like(p))
            g_out = dout[bi, :lq, hi]
            d = (g_out * out_q[bi, :lq, hi]).sum(-1)
            if dlse is not None:
                d = d - dlse[bi, hi, :lq]
            dp = g_out @ v[bi, :lk, a].T
            ds = rnd(p * (dp - d[:, None]))
            dq[bi, :lq, hi] = softmax_scale * ds @ k[bi, :lk, a]
            dk[bi, :lk, a] += softmax_scale * ds.T @ q[bi, :lq, hi]
            dv[bi, :lk, a] += rnd(p).T @ g_out
    return rnd(dq), rnd(dk), rnd(dv)


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16)


def default_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def rounding_noise(impl_control: torch.Tensor, math_control: torch.Tensor) -> float:
    """Size of pure rounding effects: implementation path against mathematical path, both with
    the identity control (no codec decisions to flip)."""
    return (impl_control.double() - math_control.double()).abs().max().item()


def assert_matches_oracle(
    name, actual, expected, noise: float, allowance=None, rtol=2.0
):
    """The repository's tolerance rule, per element, plus an optional per-element allowance.

    |actual - expected| <= rtol * noise + atol (+ allowance). ``expected`` is the
    implementation-path oracle, ``noise`` comes from ``rounding_noise``, and ``allowance`` is
    the oracle's output allowance for codec decisions the kernel arithmetic may flip.
    """
    actual, expected = actual.detach().cpu().double(), expected.double()
    atol = 2 * (expected.float() + 0.3 - 0.3 - expected.float()).abs().max().item()
    bound = rtol * noise + atol
    limit = bound if allowance is None else bound + allowance
    diff = (actual - expected).abs()
    excess = diff - limit
    assert (excess <= 0).all(), (
        f"{name}: worst excess {excess.max().item():.3e} over bound {bound:.3e} "
        f"(max diff {diff.max().item():.3e})"
    )


def assert_identity_lse_gate(
    name, lse, lse_control, lse_impl, lse_math, valid: Optional[torch.Tensor] = None
):
    """The LSE gate of the contract (section 6) for one kernel call, with the rule of the dense
    forward test: LSE is bit-identical to the identity control's, -inf exactly on the oracle's
    empty rows, and within ``2 * max|lse_impl - lse_math| + 1e-4`` of the oracle elsewhere.

    All LSEs are in the dense layout [B, H, Sq]; ``lse_impl`` / ``lse_math`` are the identity
    control's oracle LSEs on the implementation and mathematical paths. ``valid`` selects the
    rows that exist (inside the used query length); every row when None.
    """
    got, control = lse.detach().cpu(), lse_control.detach().cpu()
    assert got.shape == control.shape == lse_impl.shape, (
        got.shape,
        control.shape,
        lse_impl.shape,
    )
    if valid is None:
        valid = torch.ones_like(got, dtype=torch.bool)
    assert valid.any(), f"{name}: no row to gate"
    assert not torch.isnan(got[valid]).any(), f"{name}: NaN in LSE"
    assert torch.equal(got[valid], control[valid]), (
        f"{name}: LSE differs from the identity control"
    )
    empty = torch.isinf(lse_impl)
    assert torch.equal(torch.isinf(got)[valid], empty[valid]), (
        f"{name}: empty rows differ from the oracle's"
    )
    assert (got[empty & valid] < 0).all()
    rows = valid & ~empty
    if rows.any():
        lse_tol = 2 * (lse_impl - lse_math)[rows].abs().max().item() + 1e-4
        worst = (got.double() - lse_impl)[rows].abs().max().item()
        assert worst <= lse_tol, f"{name}: LSE off by {worst:.3e} > {lse_tol:.3e}"


# Budgets of docs/p-quant-contract.md, section 8, for random inputs.
MAX_ROW_ALLOWANCE = 2.0**-3
WIDE_ROW_ALLOWANCE = 2.0**-5
MAX_WIDE_ROW_SHARE = 0.05
LOOSE_ROW_ALLOWANCE = 2.0**-8
MAX_LOOSE_ROW_SHARE = 0.30


def check_allowance_budgets(
    prob_allowance: torch.Tensor, valid: Optional[torch.Tensor] = None
):
    """Random-input budgets: no row's check is vacuous and most rows are checked at BF16
    precision. ``valid`` selects the rows that exist (non-empty, inside the sequence)."""
    rows = prob_allowance if valid is None else prob_allowance[valid]
    if rows.numel() == 0:
        return
    assert rows.max() <= MAX_ROW_ALLOWANCE, f"row allowance {rows.max().item():.3e}"
    wide = (rows > WIDE_ROW_ALLOWANCE).double().mean().item()
    assert wide <= MAX_WIDE_ROW_SHARE, f"{wide:.1%} of rows have a wide allowance"
    loose = (rows > LOOSE_ROW_ALLOWANCE).double().mean().item()
    assert loose <= MAX_LOOSE_ROW_SHARE, f"{loose:.1%} of rows have a loose allowance"


def settle_single_row(
    q, k, v, softmax_scale, sch: Schedule, p_quant, inactive_score=-200.0
):
    """Constructed single-query inputs must need no allowance: deactivate (score far below
    the max, along the first feature) every key whose codec decision the kernel arithmetic
    could flip, until none is left. q: [1, 1, 1, D] with q[..., 0] = 1; k: [1, Sk, 1, D].
    Returns the adjusted k (a copy, on k's device)."""
    k = k.clone()
    for _ in range(16):
        res = tiled_attention(
            q[0].double().cpu(),
            k[0, :, 0].double().cpu(),
            v[0, :, 0].double().cpu(),
            softmax_scale,
            sch,
            p_quant,
            path="impl",
            with_allowance=True,
        )
        flagged = res.eligible_keys[0, 0]
        if not flagged.any():
            return k
        k[0, flagged.to(k.device), 0, 0] = inactive_score
    raise AssertionError("could not move the inputs away from every decision boundary")
